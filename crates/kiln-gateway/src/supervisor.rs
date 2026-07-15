//! Worker lifecycle supervision (SPEC §2.2): spawn one worker per model,
//! poll `Health` over the frozen proto, and drive the Phase 9 lifecycle —
//! machine-budget admission with LRU eviction (Drain → SIGTERM →
//! SIGKILL-after-grace), TTL idle auto-unload, on-demand reload, and
//! crash restarts with exponential backoff (at most
//! [`MAX_RESTART_ATTEMPTS`] per loop, then the model is `Failed` and
//! requires a manual reset — gateway restart, until the admin API lands
//! in Phase 10).
//!
//! Each model gets one supervision task, driven by two inputs: the
//! [`Lifecycle`] command channel (`Load` / `Unload`) and the worker
//! process itself. Loads are serialized machine-wide by the lifecycle's
//! load permit; the initial loads are additionally sequenced in config
//! order by a bootstrap task so startup memory pressure resolves
//! deterministically.
//!
//! In-flight requests are not torn down here: a dying worker breaks its
//! `Submit` streams, and the HTTP layer maps those transport errors to
//! structured 502s. A graceful drain precedes the signals on the
//! deliberate-unload path only; gateway shutdown remains SIGKILL (the
//! worker holds no durable state that needs it).

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    DrainMode, DrainRequest, HealthRequest, HealthStatus, InfoRequest, StatsRequest, WorkerState,
};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep, timeout};

use crate::config::KilnConfig;
use crate::lifecycle::{self, Command as LifecycleCommand, Lifecycle};
use crate::metrics::Metrics;
use crate::registry::{ModelEntry, Registry, RegistryError, UnloadReason, WorkerStatus};

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
/// Graceful-drain bound during a deliberate unload (SPEC §2.2): in-flight
/// requests get this long to finish before SIGTERM.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);
/// SIGTERM → SIGKILL escalation grace (SPEC §2.2).
const TERM_GRACE: Duration = Duration::from_secs(5);
/// Poll cadence while waiting for a graceful drain to empty the worker.
const DRAIN_POLL: Duration = Duration::from_millis(250);
/// Conservative overhead added to the on-disk-weights load projection
/// (Phase 9 part 3 ruling: reserve high, reconcile down at the first
/// measured heartbeat). Idle footprints measured 17-33 MB over raw
/// weight bytes across the pinned fleet; 64 MiB covers that with margin
/// so admissions racing a load window cannot consume unprojected bytes.
const LOAD_OVERHEAD_MARGIN_BYTES: u64 = 64 * 1024 * 1024;

fn backoff(attempt: u32) -> Duration {
    // 500ms, 1s, 2s, ... capped at 10s.
    let exp = attempt.saturating_sub(1).min(5);
    Duration::from_millis(500 << exp).min(Duration::from_secs(10))
}

#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error("memory budget: {0}")]
    Budget(String),
}

pub struct Supervisor {
    tasks: Vec<JoinHandle<()>>,
    shutdown: watch::Sender<bool>,
}

impl Supervisor {
    /// Builds the registry and lifecycle from config and spawns one
    /// supervision task per model plus the bootstrap task that sequences
    /// the initial loads in config order.
    pub fn start(
        config: &KilnConfig,
        metrics: Arc<Metrics>,
    ) -> Result<(Arc<Registry>, Arc<Lifecycle>, Self), StartError> {
        let (registry, senders) = Registry::from_config(config)?;
        let registry = Arc::new(registry);
        let (lifecycle, receivers) =
            Lifecycle::new(config, &registry, Arc::clone(&metrics)).map_err(StartError::Budget)?;
        let lifecycle = Arc::new(lifecycle);
        tracing::info!(
            budget_bytes = lifecycle.budget_bytes(),
            total_unified_bytes = lifecycle.total_bytes().unwrap_or(0),
            fraction = config.memory.budget_fraction,
            explicit_budget = config.memory.budget_bytes.is_some(),
            "machine memory budget (SPEC 2.3)"
        );
        let (shutdown, _) = watch::channel(false);

        let mut tasks = Vec::new();
        for ((entry, status_tx), cmd_rx) in registry.iter().zip(senders).zip(receivers) {
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
                    // SPEC §6.5/§10 [model.speculative]: draft path was
                    // resolved (and gated to rust workers) at registry
                    // build; the worker enforces the ADR 0005 attach
                    // gates and goes UNHEALTHY on an incompatible pair.
                    if let (Some(draft), Some(spec)) =
                        (&entry.draft_path, &entry.config.speculative)
                    {
                        argv.push("--draft-model".to_owned());
                        argv.push(draft.display().to_string());
                        argv.push("--draft-gamma".to_owned());
                        argv.push(spec.gamma.to_string());
                    }
                    argv
                }
                _ => config.server.python_worker_argv.clone(),
            };
            // Load-time projection (SPEC §2.3): weight bytes on disk for
            // target + draft, plus a conservative runtime-overhead margin
            // (Phase 9 part 3 ruling: projections reserve on the HIGH
            // side and heartbeats release the difference). Measured idle
            // footprints run 17-33 MB over raw weight bytes across the
            // pinned fleet (tokenizer, runtime, small buffers); without
            // the margin, a request admission racing this load window
            // could consume that sliver and transiently overshoot. The
            // first post-READY heartbeat replaces the whole projection
            // with the measured footprint before the load permit is
            // released.
            let weights_bytes = lifecycle::weights_bytes_on_disk(&entry.model_path)
                + entry
                    .draft_path
                    .as_deref()
                    .map(lifecycle::weights_bytes_on_disk)
                    .unwrap_or(0);
            let projected_bytes = match weights_bytes {
                0 => 0,
                bytes => bytes + LOAD_OVERHEAD_MARGIN_BYTES,
            };
            if projected_bytes == 0 {
                tracing::warn!(model = %entry.id, path = %entry.model_path.display(),
                    "no *.safetensors found; load projection is 0 bytes (budget still \
                     enforced from measured heartbeats)");
            }
            let ctx = SuperviseCtx {
                entry: Arc::clone(entry),
                argv,
                status_tx,
                metrics: Arc::clone(&metrics),
                shutdown: shutdown.subscribe(),
                cmd_rx,
                lifecycle: Arc::clone(&lifecycle),
                projected_bytes,
            };
            tasks.push(tokio::spawn(supervise(ctx)));
        }

        // Initial loads, sequenced in config order: the LRU clock starts at
        // READY time, so startup eviction order stays deterministic instead
        // of racing on the load permit.
        {
            let registry = Arc::clone(&registry);
            let lifecycle = Arc::clone(&lifecycle);
            let mut shutdown = shutdown.subscribe();
            tasks.push(tokio::spawn(async move {
                for entry in registry.iter() {
                    lifecycle.boot_load(&entry.id);
                    let mut status = entry.status.clone();
                    loop {
                        let settled = matches!(
                            *status.borrow_and_update(),
                            WorkerStatus::Ready
                                | WorkerStatus::Unloaded { .. }
                                | WorkerStatus::Failed
                                | WorkerStatus::Stopped
                        );
                        if settled {
                            break;
                        }
                        tokio::select! {
                            changed = status.changed() => {
                                if changed.is_err() {
                                    break;
                                }
                            }
                            _ = wait_shutdown(&mut shutdown) => return,
                        }
                    }
                }
                tracing::info!("initial model loads settled");
            }));
        }

        Ok((registry, lifecycle, Self { tasks, shutdown }))
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
    cmd_rx: mpsc::UnboundedReceiver<LifecycleCommand>,
    lifecycle: Arc<Lifecycle>,
    /// Load-time budget projection: weight bytes on disk (target + draft).
    projected_bytes: u64,
}

enum RunExit {
    /// Worker died or went silent; `ready_for` is how long it served READY.
    Crashed {
        ready_for: Option<Duration>,
    },
    /// Load rejected up front: over budget with no evictable model.
    BudgetRejected,
    /// Deliberate unload (eviction / idle TTL) completed; memory released.
    Unloaded {
        reason: UnloadReason,
    },
    Shutdown,
}

async fn supervise(mut ctx: SuperviseCtx) {
    let model = ctx.entry.id.clone();
    loop {
        // Idle until asked to load. The initial status (Starting, from
        // registry build) keeps /readyz unavailable until the bootstrap
        // task's first Load resolves.
        match wait_for_load(&mut ctx).await {
            Wait::Shutdown => {
                ctx.status_tx.send_replace(WorkerStatus::Stopped);
                return;
            }
            Wait::Load => {}
        }
        // One load → run → restart-on-crash cycle; ends when the model
        // unloads (back to idle), fails, or the gateway shuts down.
        let mut attempts: u32 = 0;
        loop {
            ctx.status_tx.send_replace(WorkerStatus::Starting);
            match run_once(&mut ctx).await {
                RunExit::Shutdown => {
                    ctx.status_tx.send_replace(WorkerStatus::Stopped);
                    return;
                }
                RunExit::BudgetRejected => {
                    ctx.status_tx.send_replace(WorkerStatus::Unloaded {
                        reason: UnloadReason::OverBudget,
                    });
                    break;
                }
                RunExit::Unloaded { reason } => {
                    ctx.status_tx
                        .send_replace(WorkerStatus::Unloaded { reason });
                    break;
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
                        park_failed(&mut ctx).await;
                        return;
                    }
                    let delay = backoff(attempts);
                    tracing::warn!(model = %model, attempt = attempts, delay_ms = delay.as_millis() as u64,
                        "worker crashed; restarting after backoff");
                    ctx.status_tx
                        .send_replace(WorkerStatus::Restarting { attempt: attempts });
                    match backoff_wait(&mut ctx, delay).await {
                        Backoff::Elapsed => {}
                        Backoff::Unloaded(reason) => {
                            ctx.status_tx
                                .send_replace(WorkerStatus::Unloaded { reason });
                            break;
                        }
                        Backoff::Shutdown => {
                            ctx.status_tx.send_replace(WorkerStatus::Stopped);
                            return;
                        }
                    }
                }
            }
        }
    }
}

enum Wait {
    Load,
    Shutdown,
}

/// Parks an unloaded model: acks unloads trivially (nothing is running)
/// and wakes on the next Load.
async fn wait_for_load(ctx: &mut SuperviseCtx) -> Wait {
    loop {
        tokio::select! {
            _ = wait_shutdown(&mut ctx.shutdown) => return Wait::Shutdown,
            cmd = ctx.cmd_rx.recv() => match cmd {
                Some(LifecycleCommand::Load) => return Wait::Load,
                Some(LifecycleCommand::Unload { done, .. }) => {
                    let _ = done.send(());
                }
                None => return Wait::Shutdown,
            },
        }
    }
}

/// Terminal FAILED state (SPEC §2.2 manual reset): refuses loads, acks
/// unloads (nothing is running), and only shutdown ends it. The status
/// stays `Failed` so operators see why the model is dark.
async fn park_failed(ctx: &mut SuperviseCtx) {
    loop {
        tokio::select! {
            _ = wait_shutdown(&mut ctx.shutdown) => return,
            cmd = ctx.cmd_rx.recv() => match cmd {
                Some(LifecycleCommand::Load) => {
                    tracing::warn!(model = %ctx.entry.id,
                        "load requested for a FAILED model; manual reset required (restart the gateway)");
                }
                Some(LifecycleCommand::Unload { done, .. }) => {
                    let _ = done.send(());
                }
                None => return,
            },
        }
    }
}

enum Backoff {
    Elapsed,
    Unloaded(UnloadReason),
    Shutdown,
}

/// Crash-restart backoff that stays responsive: an Unload during the wait
/// cancels the pending restart (nothing is running — the worker just
/// died), which is also how an eviction races cleanly with a crash.
async fn backoff_wait(ctx: &mut SuperviseCtx, delay: Duration) -> Backoff {
    let deadline = sleep(delay);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => return Backoff::Elapsed,
            _ = wait_shutdown(&mut ctx.shutdown) => return Backoff::Shutdown,
            cmd = ctx.cmd_rx.recv() => match cmd {
                Some(LifecycleCommand::Load) => {} // restart already pending
                Some(LifecycleCommand::Unload { reason, done }) => {
                    let _ = done.send(());
                    return Backoff::Unloaded(reason);
                }
                None => return Backoff::Shutdown,
            },
        }
    }
}

/// One worker lifetime: budget acquisition (with LRU eviction) → spawn →
/// wait READY → monitor until crash, unload, or shutdown.
async fn run_once(ctx: &mut SuperviseCtx) -> RunExit {
    let entry = Arc::clone(&ctx.entry);

    // -- machine budget (SPEC §2.3), one load at a time --------------------
    let lifecycle = Arc::clone(&ctx.lifecycle);
    let permit = tokio::select! {
        permit = lifecycle.load_permit().lock() => permit,
        _ = wait_shutdown(&mut ctx.shutdown) => return RunExit::Shutdown,
    };
    loop {
        // Charged, not just used: request-admission reservations awaiting
        // heartbeat confirmation are real obligations this load must not
        // double-spend (Phase 9 part 3 reservation ledger).
        let used = ctx.lifecycle.charged_bytes();
        let budget = ctx.lifecycle.budget_bytes();
        if used.saturating_add(ctx.projected_bytes) <= budget {
            break;
        }
        let Some(victim) = ctx.lifecycle.pick_victim(&entry.id) else {
            tracing::error!(model = %entry.id,
                projected_bytes = ctx.projected_bytes, used_bytes = used, budget_bytes = budget,
                "load rejected: machine budget exceeded and no evictable model \
                 (candidates must be loaded, unpinned, and outside their TTL lease)");
            return RunExit::BudgetRejected;
        };
        tracing::warn!(model = %entry.id, victim = %victim,
            projected_bytes = ctx.projected_bytes, used_bytes = used, budget_bytes = budget,
            "machine budget exceeded; evicting LRU model");
        if !ctx.lifecycle.evict(&victim).await {
            tracing::error!(model = %entry.id, victim = %victim,
                "eviction did not complete; rejecting load");
            return RunExit::BudgetRejected;
        }
        if *ctx.shutdown.borrow() {
            return RunExit::Shutdown;
        }
    }
    // Reserve the projection; the first measured heartbeat replaces it
    // below, before the load permit is released.
    ctx.lifecycle.record_usage(&entry.id, ctx.projected_bytes);

    // -- spawn --------------------------------------------------------------
    let mut child = match spawn_worker(ctx) {
        Ok(child) => child,
        Err(err) => {
            tracing::error!(model = %entry.id, error = %err, argv = ?ctx.argv,
                "failed to spawn worker process");
            release(ctx).await;
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

    // -- wait for READY ----------------------------------------------------
    let load_deadline = Instant::now() + READY_DEADLINE;
    let mut poll = interval(READY_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            status = child.wait() => {
                tracing::error!(model = %entry.id, status = ?status.ok(),
                    "worker exited while loading");
                kill_group(pgid, &entry.id).await;
                release(ctx).await;
                return RunExit::Crashed { ready_for: None };
            }
            _ = wait_shutdown(&mut ctx.shutdown) => {
                kill_and_reap(&mut child, pgid, &entry.id).await;
                release(ctx).await;
                return RunExit::Shutdown;
            }
            _ = poll.tick() => {
                if Instant::now() > load_deadline {
                    tracing::error!(model = %entry.id, "worker never reached READY; recycling");
                    kill_and_reap(&mut child, pgid, &entry.id).await;
                    release(ctx).await;
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
                            release(ctx).await;
                            return RunExit::Crashed { ready_for: None };
                        }
                        _ => {} // Loading — keep waiting.
                    }
                }
            }
        }
    }

    // -- READY ---------------------------------------------------------------
    let ready_at = Instant::now();
    refresh_info(ctx, &mut client).await;
    // Swap the reservation for a measured footprint before releasing the
    // load permit, so the next load in line budgets against real bytes.
    if let Ok(Ok(resp)) = timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await {
        record_memory(ctx, &resp.into_inner());
    }
    // The LRU/TTL clock starts at READY.
    ctx.lifecycle.touch(&entry.id);
    ctx.status_tx.send_replace(WorkerStatus::Ready);
    ctx.metrics.worker_up.with_label_values(&[&entry.id]).set(1);
    tracing::info!(model = %entry.id, load_ms = ready_at.elapsed().as_millis() as u64,
        used_bytes = ctx.lifecycle.used_bytes(), budget_bytes = ctx.lifecycle.budget_bytes(),
        "worker ready");
    drop(permit);

    // -- monitor -------------------------------------------------------------
    let ttl = match entry.config.ttl_seconds {
        0 => None,
        secs => Some(Duration::from_secs(secs)),
    };
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
                release(ctx).await;
                return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
            }
            _ = wait_shutdown(&mut ctx.shutdown) => {
                kill_and_reap(&mut child, pgid, &entry.id).await;
                release(ctx).await;
                return RunExit::Shutdown;
            }
            cmd = ctx.cmd_rx.recv() => match cmd {
                // Load while loaded is a no-op; a closed channel means the
                // gateway is going down and the shutdown arm will fire.
                Some(LifecycleCommand::Load) | None => {}
                Some(LifecycleCommand::Unload { reason, done }) => {
                    let exit = unload(ctx, &mut child, pgid, &mut client, reason).await;
                    let _ = done.send(());
                    return exit;
                }
            },
            _ = poll.tick() => {
                match timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await {
                    Ok(Ok(resp)) => {
                        let status = resp.into_inner();
                        if status.state() == WorkerState::Unhealthy {
                            tracing::error!(model = %entry.id, detail = %status.detail,
                                "worker self-reported UNHEALTHY; recycling");
                            kill_and_reap(&mut child, pgid, &entry.id).await;
                            release(ctx).await;
                            return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
                        }
                        last_ok = Instant::now();
                        record_memory(ctx, &status);
                        if status.requests_running + status.requests_waiting > 0 {
                            // In-flight work counts as use: it holds the LRU
                            // position and the TTL idle clock alike.
                            ctx.lifecycle.touch(&entry.id);
                        } else if let Some(ttl) = ttl
                            && ctx.lifecycle.idle(&entry.id) >= ttl
                        {
                            tracing::info!(model = %entry.id, ttl_s = ttl.as_secs(),
                                idle_ms = ctx.lifecycle.idle(&entry.id).as_millis() as u64,
                                "idle past ttl_seconds; auto-unloading");
                            return unload(ctx, &mut child, pgid, &mut client, UnloadReason::IdleTtl).await;
                        }
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
                            release(ctx).await;
                            return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
                        }
                    }
                }
            }
        }
    }
}

/// Deliberate unload, per the SPEC §2.2 eviction contract: `Drain`
/// (graceful, bounded) → SIGTERM → SIGKILL after grace. Deterministic
/// memory reclamation is the process exit (SPEC §1.1), so the group is
/// always swept at the end.
async fn unload(
    ctx: &SuperviseCtx,
    child: &mut Child,
    pgid: Option<u32>,
    client: &mut WorkerClient<tonic::transport::Channel>,
    reason: UnloadReason,
) -> RunExit {
    let entry = &ctx.entry;
    ctx.status_tx.send_replace(WorkerStatus::Draining);
    tracing::info!(model = %entry.id, reason = reason.label(), "unloading worker");

    // 1. Graceful drain, best-effort (the python worker answers
    //    UNIMPLEMENTED) and bounded by DRAIN_DEADLINE.
    let deadline = Instant::now() + DRAIN_DEADLINE;
    let mut remaining = 0u32;
    match timeout(
        HEALTH_RPC_TIMEOUT,
        client.drain(DrainRequest {
            mode: DrainMode::Graceful as i32,
            deadline_ms: DRAIN_DEADLINE.as_millis() as u64,
        }),
    )
    .await
    {
        Ok(Ok(ack)) => remaining = ack.into_inner().requests_remaining,
        _ => tracing::debug!(model = %entry.id,
            "Drain RPC unavailable; escalating straight to SIGTERM"),
    }
    while remaining > 0 && Instant::now() < deadline && !*ctx.shutdown.borrow() {
        if child.try_wait().ok().flatten().is_some() {
            break; // died on its own; signals below are no-ops
        }
        sleep(DRAIN_POLL).await;
        match timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await {
            Ok(Ok(resp)) => {
                let health = resp.into_inner();
                remaining = health.requests_running + health.requests_waiting;
            }
            _ => break, // dead socket: the signals finish the job
        }
    }

    // 2. SIGTERM the process group; 3. SIGKILL after grace.
    signal_group(pgid, "-TERM", &entry.id).await;
    if timeout(TERM_GRACE, child.wait()).await.is_err() {
        tracing::warn!(model = %entry.id, grace_ms = TERM_GRACE.as_millis() as u64,
            "worker survived SIGTERM grace; sending SIGKILL");
    }
    // Sweep the whole group regardless: the direct child may be a wrapper
    // whose descendants outlive it (the `uv run` python case).
    kill_group(pgid, &entry.id).await;
    let _ = child.wait().await;

    release(ctx).await;
    ctx.metrics
        .worker_unloads_total
        .with_label_values(&[&entry.id, reason.label()])
        .inc();
    tracing::info!(model = %entry.id, reason = reason.label(),
        used_bytes = ctx.lifecycle.used_bytes(), "worker unloaded; memory released");
    RunExit::Unloaded { reason }
}

/// Records one heartbeat's memory numbers: the budget ledger, the pool
/// materialization gauge feeding per-request admission, and the per-model
/// gauges (SPEC §2.3).
fn record_memory(ctx: &SuperviseCtx, health: &HealthStatus) {
    let Some(report) = &health.memory else {
        return;
    };
    let footprint = lifecycle::footprint_bytes(report);
    ctx.lifecycle.record_usage(&ctx.entry.id, footprint);
    ctx.lifecycle
        .record_pool_materialized(&ctx.entry.id, report.kv_pool_allocated_bytes);
    ctx.metrics
        .worker_memory
        .record(&ctx.entry.id, report, footprint);
}

/// Releases everything a dead worker was charged for: budget ledger,
/// memory gauges, up-gauge, and the cached `GetInfo`.
async fn release(ctx: &SuperviseCtx) {
    ctx.lifecycle.clear_usage(&ctx.entry.id);
    ctx.metrics.worker_memory.clear(&ctx.entry.id);
    ctx.metrics
        .worker_up
        .with_label_values(&[&ctx.entry.id])
        .set(0);
    *ctx.entry.info.write().await = None;
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
            // Full-pool cost for per-request admission (SPEC §2.3/§6.4);
            // 0 (no gating) for workers that report no pool geometry.
            // Prefer the whole-worker commitment (target + draft pools,
            // Phase 9 part 3): the target-only product under-projects a
            // draft-carrying worker by an entire draft pool.
            let commitment = match info.kv_pool_commitment_bytes {
                0 => info.kv_bytes_per_block.saturating_mul(info.kv_pool_blocks),
                bytes => bytes,
            };
            ctx.lifecycle.set_pool_commitment(&entry.id, commitment);
            *entry.info.write().await = Some(info);
        }
        other => {
            tracing::warn!(model = %entry.id, result = ?other.err(),
                "GetInfo failed after READY; usage limits may be unavailable");
        }
    }
}

async fn kill_and_reap(child: &mut Child, pgid: Option<u32>, model: &str) {
    // Immediate SIGKILL: crash recycling and gateway shutdown (the worker
    // holds no durable state). The graceful Drain → SIGTERM → SIGKILL
    // ladder lives in `unload` and runs only for deliberate unloads.
    kill_group(pgid, model).await;
    if let Err(err) = child.start_kill()
        && err.kind() != std::io::ErrorKind::InvalidInput
    {
        tracing::warn!(model = %model, error = %err, "failed to kill worker");
    }
    let _ = child.wait().await;
}

/// SIGKILLs the worker's whole process group (pgid == spawned pid, via
/// `process_group(0)`).
async fn kill_group(pgid: Option<u32>, model: &str) {
    signal_group(pgid, "-9", model).await;
}

/// Signals the worker's whole process group. The configured argv may be a
/// wrapper — the default `uv run` is — so signaling only the direct child
/// would orphan the model-loaded python process underneath. `/bin/kill` is
/// shelled out to because signaling a pgid needs `libc::kill`, and unsafe
/// code is confined to kiln-mlx (CLAUDE.md).
async fn signal_group(pgid: Option<u32>, signal: &str, model: &str) {
    let Some(pgid) = pgid else { return };
    match Command::new("/bin/kill")
        .args([signal, "--", &format!("-{pgid}")])
        .status()
        .await
    {
        Ok(status) if status.success() => {}
        // Non-zero usually means the group is already fully dead — the
        // normal case after a clean worker crash.
        Ok(_) => tracing::debug!(model = %model, pgid, signal, "process group already gone"),
        Err(err) => tracing::warn!(model = %model, pgid, signal, error = %err,
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
