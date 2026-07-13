//! Worker lifecycle supervision (SPEC §2.2, Phase 2 subset): spawn the
//! Python worker per model, poll `Health` over the frozen proto, and
//! restart with exponential backoff on crash — at most
//! [`MAX_RESTART_ATTEMPTS`] times, then the model requires a manual reset
//! (gateway restart, until the admin API lands in Phase 9).
//!
//! In-flight requests are not torn down here: a dying worker breaks its
//! `Submit` streams, and the HTTP layer maps those transport errors to
//! structured 502s.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{HealthRequest, InfoRequest, StatsRequest, WorkerState};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep, timeout};

use crate::config::KilnConfig;
use crate::metrics::Metrics;
use crate::registry::{ModelEntry, Registry, RegistryError, WorkerStatus};

/// Health poll cadence while Ready (SPEC §5: default 1s).
const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Per-RPC deadline for a single health call.
const HEALTH_RPC_TIMEOUT: Duration = Duration::from_secs(2);
/// A worker silent for longer than this is treated as crashed (SPEC §5: 3s).
const HEALTH_MISSED_DEADLINE: Duration = Duration::from_secs(3);
/// Poll cadence while waiting for the model to load.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(250);
/// Give up on a load that never reaches READY (tiny models load in seconds;
/// large ones in minutes — this is a hang guard, not a performance bar).
const READY_DEADLINE: Duration = Duration::from_secs(600);
/// Automatic restarts per crash loop before requiring manual reset.
const MAX_RESTART_ATTEMPTS: u32 = 3;
/// A worker Ready for at least this long resets the crash-loop counter.
const STABLE_RESET: Duration = Duration::from_secs(60);

fn backoff(attempt: u32) -> Duration {
    // 500ms, 1s, 2s, ... capped at 10s.
    let exp = attempt.saturating_sub(1).min(5);
    Duration::from_millis(500 << exp).min(Duration::from_secs(10))
}

pub struct Supervisor {
    tasks: Vec<JoinHandle<()>>,
    shutdown: watch::Sender<bool>,
}

impl Supervisor {
    /// Builds the registry from config and spawns one supervision task per
    /// model. Workers begin loading immediately.
    pub fn start(
        config: &KilnConfig,
        metrics: Arc<Metrics>,
    ) -> Result<(Arc<Registry>, Self), RegistryError> {
        let (registry, senders) = Registry::from_config(config)?;
        let registry = Arc::new(registry);
        let (shutdown, _) = watch::channel(false);

        let mut tasks = Vec::new();
        for (entry, status_tx) in registry.iter().zip(senders) {
            let argv = match entry.worker_kind {
                crate::config::WorkerKind::Rust => {
                    let mut argv = config.server.rust_worker_argv.clone();
                    // SPEC §10 [defaults]: SSD tier flags for the rust
                    // worker (it derives `<cache_dir>/<fingerprint>/blocks`
                    // itself). The python worker has no cold tier.
                    if config.defaults.ssd_tier {
                        argv.push("--ssd-dir".to_owned());
                        argv.push(
                            crate::registry::expand_tilde(&config.server.cache_dir)
                                .display()
                                .to_string(),
                        );
                        argv.push("--ssd-max-gb".to_owned());
                        argv.push(config.defaults.ssd_cache_max_gb.to_string());
                    }
                    // SPEC §7.4: opt-in paged-attention kernel flag.
                    if config.defaults.paged_attention_kernel {
                        argv.push("--paged-attention-kernel".to_owned());
                    }
                    argv
                }
                _ => config.server.python_worker_argv.clone(),
            };
            let ctx = SuperviseCtx {
                entry: Arc::clone(entry),
                argv,
                status_tx,
                metrics: Arc::clone(&metrics),
                shutdown: shutdown.subscribe(),
            };
            tasks.push(tokio::spawn(supervise(ctx)));
        }
        Ok((registry, Self { tasks, shutdown }))
    }

    /// Signals every supervision task to kill its worker and waits for them.
    pub async fn shutdown(self) {
        self.shutdown.send_replace(true);
        for task in self.tasks {
            // A panicked supervision task is already logged by tokio; there
            // is nothing further to unwind during shutdown.
            let _ = task.await;
        }
    }
}

struct SuperviseCtx {
    entry: Arc<ModelEntry>,
    argv: Vec<String>,
    status_tx: watch::Sender<WorkerStatus>,
    metrics: Arc<Metrics>,
    shutdown: watch::Receiver<bool>,
}

enum RunExit {
    /// Worker died or went silent; `ready_for` is how long it served READY.
    Crashed {
        ready_for: Option<Duration>,
    },
    Shutdown,
}

async fn supervise(mut ctx: SuperviseCtx) {
    let model = ctx.entry.id.clone();
    let mut attempts: u32 = 0;
    loop {
        ctx.status_tx.send_replace(WorkerStatus::Starting);
        match run_once(&mut ctx).await {
            RunExit::Shutdown => {
                ctx.status_tx.send_replace(WorkerStatus::Stopped);
                ctx.metrics.worker_up.with_label_values(&[&model]).set(0);
                return;
            }
            RunExit::Crashed { ready_for } => {
                ctx.metrics.worker_up.with_label_values(&[&model]).set(0);
                ctx.metrics
                    .worker_restarts_total
                    .with_label_values(&[&model])
                    .inc();
                if ready_for.is_some_and(|d| d >= STABLE_RESET) {
                    attempts = 0;
                }
                attempts += 1;
                if attempts > MAX_RESTART_ATTEMPTS {
                    tracing::error!(model = %model, attempts,
                        "worker exceeded restart budget; marking failed (manual reset required)");
                    ctx.status_tx.send_replace(WorkerStatus::Failed);
                    return;
                }
                let delay = backoff(attempts);
                tracing::warn!(model = %model, attempt = attempts, delay_ms = delay.as_millis() as u64,
                    "worker crashed; restarting after backoff");
                ctx.status_tx
                    .send_replace(WorkerStatus::Restarting { attempt: attempts });
                tokio::select! {
                    _ = sleep(delay) => {}
                    _ = wait_shutdown(&mut ctx.shutdown) => {
                        ctx.status_tx.send_replace(WorkerStatus::Stopped);
                        return;
                    }
                }
            }
        }
    }
}

/// One worker lifetime: spawn → wait READY → monitor until crash/shutdown.
async fn run_once(ctx: &mut SuperviseCtx) -> RunExit {
    let entry = &ctx.entry;
    let mut child = match spawn_worker(ctx) {
        Ok(child) => child,
        Err(err) => {
            tracing::error!(model = %entry.id, error = %err, argv = ?ctx.argv,
                "failed to spawn worker process");
            return RunExit::Crashed { ready_for: None };
        }
    };
    forward_output(child.stdout.take(), &entry.id, "stdout");
    forward_output(child.stderr.take(), &entry.id, "stderr");
    // Saved up front: after child.wait() reaps, child.id() is None, but the
    // process group (wrapper + python) may still need sweeping.
    let pgid = child.id();
    tracing::info!(model = %entry.id, pid = pgid,
        socket = %entry.socket_path.display(), "worker spawned");

    let mut client = WorkerClient::new(entry.channel.clone());

    // -- wait for READY --------------------------------------------------
    let load_deadline = Instant::now() + READY_DEADLINE;
    let mut poll = interval(READY_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            status = child.wait() => {
                tracing::error!(model = %entry.id, status = ?status.ok(),
                    "worker exited while loading");
                kill_group(pgid, &entry.id).await;
                return RunExit::Crashed { ready_for: None };
            }
            _ = wait_shutdown(&mut ctx.shutdown) => {
                kill_and_reap(&mut child, pgid, &entry.id).await;
                return RunExit::Shutdown;
            }
            _ = poll.tick() => {
                if Instant::now() > load_deadline {
                    tracing::error!(model = %entry.id, "worker never reached READY; recycling");
                    kill_and_reap(&mut child, pgid, &entry.id).await;
                    return RunExit::Crashed { ready_for: None };
                }
                // A failed call means the socket is not up yet (or timed
                // out) — keep waiting until the load deadline; child exit
                // is caught above.
                if let Ok(Ok(resp)) = timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await {
                    match resp.into_inner().state() {
                        WorkerState::Ready => break,
                        WorkerState::Unhealthy => {
                            tracing::error!(model = %entry.id,
                                "worker reported UNHEALTHY during load (model load failed?); recycling");
                            kill_and_reap(&mut child, pgid, &entry.id).await;
                            return RunExit::Crashed { ready_for: None };
                        }
                        _ => {} // Loading — keep waiting.
                    }
                }
            }
        }
    }

    // -- READY -----------------------------------------------------------
    let ready_at = Instant::now();
    refresh_info(ctx, &mut client).await;
    ctx.status_tx.send_replace(WorkerStatus::Ready);
    ctx.metrics.worker_up.with_label_values(&[&entry.id]).set(1);
    tracing::info!(model = %entry.id, load_ms = ready_at.elapsed().as_millis() as u64,
        "worker ready");

    // -- monitor ----------------------------------------------------------
    let mut poll = interval(HEALTH_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last_ok = Instant::now();
    // SPEC §5/§2.3: Stats is polled alongside Health and re-exported with
    // a `model` label. A worker without it (the python worker today)
    // answers UNIMPLEMENTED once and is not asked again this lifetime.
    let mut stats_supported = true;
    loop {
        tokio::select! {
            status = child.wait() => {
                tracing::error!(model = %entry.id, status = ?status.ok(), "worker process exited");
                kill_group(pgid, &entry.id).await;
                return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
            }
            _ = wait_shutdown(&mut ctx.shutdown) => {
                kill_and_reap(&mut child, pgid, &entry.id).await;
                return RunExit::Shutdown;
            }
            _ = poll.tick() => {
                match timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await {
                    Ok(Ok(resp)) => {
                        let status = resp.into_inner();
                        if status.state() == WorkerState::Unhealthy {
                            tracing::error!(model = %entry.id, detail = %status.detail,
                                "worker self-reported UNHEALTHY; recycling");
                            kill_and_reap(&mut child, pgid, &entry.id).await;
                            return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
                        }
                        last_ok = Instant::now();
                        if stats_supported {
                            match timeout(HEALTH_RPC_TIMEOUT, client.stats(StatsRequest {})).await {
                                Ok(Ok(resp)) => {
                                    ctx.metrics.worker_stats.record(&entry.id, &resp.into_inner());
                                }
                                Ok(Err(status)) if status.code() == tonic::Code::Unimplemented => {
                                    tracing::debug!(model = %entry.id,
                                        "worker does not implement Stats; skipping re-export");
                                    stats_supported = false;
                                }
                                // Transient failures: Health owns crash
                                // detection; stats just misses a sample.
                                _ => {}
                            }
                        }
                    }
                    _ => {
                        if last_ok.elapsed() > HEALTH_MISSED_DEADLINE {
                            tracing::error!(model = %entry.id,
                                silent_ms = last_ok.elapsed().as_millis() as u64,
                                "worker missed health deadline; recycling");
                            kill_and_reap(&mut child, pgid, &entry.id).await;
                            return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
                        }
                    }
                }
            }
        }
    }
}

fn spawn_worker(ctx: &SuperviseCtx) -> std::io::Result<Child> {
    let entry = &ctx.entry;
    let mut cmd = Command::new(&ctx.argv[0]);
    cmd.args(&ctx.argv[1..])
        .arg("--model")
        .arg(&entry.model_path)
        .arg("--socket")
        .arg(&entry.socket_path)
        .arg("--model-id")
        .arg(&entry.id)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own process group (pgid = child pid): the configured argv may be a
        // wrapper (the default `uv run` is), so kills must target the whole
        // group or the actual python worker survives as an orphan.
        .process_group(0)
        // Safety net: never leave an orphaned worker if the gateway dies.
        .kill_on_drop(true);
    cmd.spawn()
}

/// Re-logs each worker output line under the gateway's structured logging so
/// worker crashes are diagnosable from gateway logs alone.
fn forward_output(
    stream: Option<impl AsyncRead + Unpin + Send + 'static>,
    model: &str,
    source: &'static str,
) {
    let Some(stream) = stream else { return };
    let model = model.to_string();
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::info!(target: "kiln::worker", model = %model, source, "{line}");
        }
    });
}

/// Fetches `GetInfo`, caches it on the entry, and verifies the gateway's
/// template matches the worker's (SPEC §5 `chat_template_hash`).
async fn refresh_info(ctx: &SuperviseCtx, client: &mut WorkerClient<tonic::transport::Channel>) {
    let entry = &ctx.entry;
    match timeout(HEALTH_RPC_TIMEOUT, client.get_info(InfoRequest {})).await {
        Ok(Ok(resp)) => {
            let info = resp.into_inner();
            if let Some(template) = &entry.template
                && !info.chat_template_hash.is_empty()
                && info.chat_template_hash != template.source_hash()
            {
                tracing::warn!(model = %entry.id,
                    gateway_hash = %template.source_hash(),
                    worker_hash = %info.chat_template_hash,
                    "chat template mismatch between gateway and worker");
            }
            *entry.info.write().await = Some(info);
        }
        other => {
            tracing::warn!(model = %entry.id, result = ?other.err(),
                "GetInfo failed after READY; usage limits may be unavailable");
        }
    }
}

async fn kill_and_reap(child: &mut Child, pgid: Option<u32>, model: &str) {
    // Phase 2 shutdown is SIGKILL; graceful Drain+SIGTERM arrives with
    // eviction in Phase 9 (SPEC §2.2). The worker holds no durable state yet.
    kill_group(pgid, model).await;
    if let Err(err) = child.start_kill()
        && err.kind() != std::io::ErrorKind::InvalidInput
    {
        tracing::warn!(model = %model, error = %err, "failed to kill worker");
    }
    let _ = child.wait().await;
}

/// SIGKILLs the worker's whole process group (pgid == spawned pid, via
/// `process_group(0)`). The configured argv may be a wrapper — the default
/// `uv run` is — so killing only the direct child would orphan the
/// model-loaded python process underneath. `/bin/kill` is shelled out to
/// because signaling a pgid needs `libc::kill`, and unsafe code is confined
/// to kiln-mlx (CLAUDE.md).
async fn kill_group(pgid: Option<u32>, model: &str) {
    let Some(pgid) = pgid else { return };
    match Command::new("/bin/kill")
        .args(["-9", "--", &format!("-{pgid}")])
        .status()
        .await
    {
        Ok(status) if status.success() => {}
        // Non-zero usually means the group is already fully dead — the
        // normal case after a clean worker crash.
        Ok(_) => tracing::debug!(model = %model, pgid, "process group already gone"),
        Err(err) => tracing::warn!(model = %model, pgid, error = %err,
            "failed to run /bin/kill for process group"),
    }
}

async fn wait_shutdown(rx: &mut watch::Receiver<bool>) {
    // wait_for only errors when the sender is dropped; treat that as
    // shutdown too so supervision tasks never outlive the gateway.
    let _ = rx.wait_for(|v| *v).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff(1), Duration::from_millis(500));
        assert_eq!(backoff(2), Duration::from_secs(1));
        assert_eq!(backoff(3), Duration::from_secs(2));
        assert_eq!(backoff(30), Duration::from_secs(10));
    }
}
