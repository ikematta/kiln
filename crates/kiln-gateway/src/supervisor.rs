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
    WorkerStats,
};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep, timeout};

use crate::config::{KilnConfig, WorkerKind};
use crate::lifecycle::{self, Command as LifecycleCommand, Lifecycle};
use crate::metrics::Metrics;
use crate::registry::{ModelEntry, Registry, RegistryError, UnloadReason, WorkerStatus};

/// Health poll cadence while Ready (SPEC §5: default 1s).
const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Per-RPC deadline for a single health call.
const HEALTH_RPC_TIMEOUT: Duration = Duration::from_secs(2);
/// A worker silent for longer than this is treated as crashed (SPEC §5: 3s).
const HEALTH_MISSED_DEADLINE: Duration = Duration::from_secs(3);
/// Floor between two "Stats sample dropped" warnings for one worker. Drops
/// in between are still counted (`kiln_stats_samples_dropped_total`) and
/// summed into the next warning, so a wedged worker cannot flood the log.
const STATS_DROP_WARN_INTERVAL: Duration = Duration::from_secs(30);
/// Max age of the cached system-memory snapshot the request-admission path
/// prices against; health polls refresh it in the background past this.
const SYSTEM_PROBE_MAX_AGE: Duration = Duration::from_secs(2);
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
pub(crate) const LOAD_OVERHEAD_MARGIN_BYTES: u64 = 64 * 1024 * 1024;

/// Crash-restart backoff curve; also reused by the MCP client's reconnect
/// loop (crate::mcp) so external-server retries pace like worker restarts.
pub(crate) fn backoff(attempt: u32) -> Duration {
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
    spawner: ModelSpawner,
}

/// Everything needed to spawn one model's supervision task; shared by
/// [`Supervisor::start`] for the boot-time fleet and (via
/// [`Supervisor::spawner`]) by the runtime add-model path, so a model
/// registered through `POST /admin/models` is supervised by exactly the
/// same code as a configured one.
#[derive(Clone)]
pub struct ModelSpawner {
    config: Arc<KilnConfig>,
    metrics: Arc<Metrics>,
    lifecycle: Arc<Lifecycle>,
    shutdown: watch::Sender<bool>,
    /// Shared with [`Supervisor::shutdown`], which drains and awaits every
    /// handle — including tasks spawned after boot.
    tasks: Arc<std::sync::Mutex<Vec<JoinHandle<()>>>>,
}

impl ModelSpawner {
    /// Spawner for handler unit tests that build registry/lifecycle by
    /// hand instead of through [`Supervisor::start`]. Tasks it spawns are
    /// dropped with the test runtime.
    #[cfg(test)]
    pub(crate) fn test_stub(
        config: Arc<KilnConfig>,
        metrics: Arc<Metrics>,
        lifecycle: Arc<Lifecycle>,
    ) -> Self {
        Self {
            config,
            metrics,
            lifecycle,
            shutdown: watch::channel(false).0,
            tasks: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Spawns the supervision task for one registry entry and registers
    /// its handle for gateway shutdown.
    pub(crate) fn spawn(
        &self,
        entry: Arc<ModelEntry>,
        status_tx: watch::Sender<WorkerStatus>,
        cmd_rx: mpsc::UnboundedReceiver<LifecycleCommand>,
    ) {
        let projected_bytes = load_projection(&entry);
        if projected_bytes == 0 {
            tracing::warn!(model = %entry.id, path = %entry.model_path.display(),
                "no *.safetensors found; load projection is 0 bytes (budget still \
                 enforced from measured heartbeats)");
        }
        let ctx = SuperviseCtx {
            argv: worker_argv(&self.config, &entry),
            entry,
            status_tx,
            metrics: Arc::clone(&self.metrics),
            shutdown: self.shutdown.subscribe(),
            cmd_rx,
            lifecycle: Arc::clone(&self.lifecycle),
            projected_bytes,
        };
        let handle = tokio::spawn(supervise(ctx));
        self.tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(handle);
    }
}

/// Builds the worker argv for one entry: worker binary prefix from config
/// plus the SPEC §10 flags (SSD tier, paged-attention kernel, speculative
/// draft — rust workers only; the python worker has none of them).
fn worker_argv(config: &KilnConfig, entry: &ModelEntry) -> Vec<String> {
    match entry.worker_kind {
        crate::config::WorkerKind::Rust => {
            let mut argv = config.server.rust_worker_argv.clone();
            // SPEC §10 [defaults]: SSD tier flags for the rust worker (it
            // derives `<cache_dir>/<fingerprint>/blocks` itself).
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
            // SPEC §6.5/§10 [model.speculative]: draft path was resolved
            // (and gated to rust workers) at registry build; the worker
            // enforces the ADR 0005 attach gates and goes UNHEALTHY on an
            // incompatible pair.
            if let (Some(draft), Some(spec)) = (&entry.draft_path, &entry.config.speculative) {
                argv.push("--draft-model".to_owned());
                argv.push(draft.display().to_string());
                argv.push("--draft-gamma".to_owned());
                argv.push(spec.gamma.to_string());
            }
            argv
        }
        _ => config.server.python_worker_argv.clone(),
    }
}

/// Load-time projection (SPEC §2.3): weight bytes on disk for target +
/// draft, plus a conservative runtime-overhead margin (Phase 9 part 3
/// ruling: projections reserve on the HIGH side and heartbeats release
/// the difference). Measured idle footprints run 17-33 MB over raw
/// weight bytes across the pinned fleet (tokenizer, runtime, small
/// buffers); without the margin, a request admission racing this load
/// window could consume that sliver and transiently overshoot. The first
/// post-READY heartbeat replaces the whole projection with the measured
/// footprint before the load permit is released. `run_once` additionally
/// prices in the pool commitment remembered from the model's last READY
/// (see `Lifecycle::known_pool_commitment_bytes`) — this function is the
/// weights-only base.
pub(crate) fn load_projection(entry: &ModelEntry) -> u64 {
    let weights_bytes = lifecycle::weights_bytes_on_disk(&entry.model_path)
        + entry
            .draft_path
            .as_deref()
            .map(lifecycle::weights_bytes_on_disk)
            .unwrap_or(0);
    match weights_bytes {
        0 => 0,
        bytes => bytes + LOAD_OVERHEAD_MARGIN_BYTES,
    }
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
        let system = lifecycle.system_memory();
        tracing::info!(
            budget_bytes = lifecycle.budget_bytes(),
            total_unified_bytes = lifecycle.total_bytes().unwrap_or(0),
            fraction = config.memory.budget_fraction,
            explicit_budget = config.memory.budget_bytes.is_some(),
            min_available_bytes = lifecycle.min_available_bytes(),
            system_available_bytes = system.map(|m| m.available_bytes).unwrap_or(0),
            swap_used_bytes = system.map(|m| m.swap_used_bytes).unwrap_or(0),
            pressure_level = system.map(|m| m.pressure_level).unwrap_or(0),
            "machine memory budget (SPEC 2.3) + live system snapshot"
        );
        let (shutdown, _) = watch::channel(false);
        let spawner = ModelSpawner {
            config: Arc::new(config.clone()),
            metrics,
            lifecycle: Arc::clone(&lifecycle),
            shutdown,
            tasks: Arc::new(std::sync::Mutex::new(Vec::new())),
        };

        for ((entry, status_tx), cmd_rx) in
            registry.entries().into_iter().zip(senders).zip(receivers)
        {
            spawner.spawn(entry, status_tx, cmd_rx);
        }

        // Initial loads, sequenced in config order: the LRU clock starts at
        // READY time, so startup eviction order stays deterministic instead
        // of racing on the load permit. Runtime-added models are not boot
        // models: the snapshot below is taken before any add can land.
        {
            let entries = registry.entries();
            let lifecycle = Arc::clone(&lifecycle);
            let mut shutdown = spawner.shutdown.subscribe();
            let handle = tokio::spawn(async move {
                for entry in entries {
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
            });
            spawner
                .tasks
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(handle);
        }

        Ok((registry, lifecycle, Self { spawner }))
    }

    /// A cloneable handle for spawning supervision tasks after boot
    /// (`POST /admin/models`); shutdown waits for those too.
    pub fn spawner(&self) -> ModelSpawner {
        self.spawner.clone()
    }

    /// Signals every supervision task to kill its worker and waits for them.
    pub async fn shutdown(self) {
        self.spawner.shutdown.send_replace(true);
        let tasks =
            std::mem::take(&mut *self.spawner.tasks.lock().unwrap_or_else(|e| e.into_inner()));
        for task in tasks {
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
    /// Load rejected up front by the system-memory gate: the machine
    /// cannot grant the projected bytes without swapping, even though the
    /// configured budget has room (2026-07-21 field finding).
    SystemRejected,
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
                RunExit::SystemRejected => {
                    ctx.status_tx.send_replace(WorkerStatus::Unloaded {
                        reason: UnloadReason::SystemMemory,
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
    // Price the load at weights PLUS the pool commitment remembered from
    // this model's last READY (0 on a first-ever load). A demand-driven
    // reload is immediately followed by the triggering request's pool
    // growth, and weights-only pricing can admit the worker into headroom
    // that growth does not fit — stranding it READY-with-cold-pool behind
    // the per-request denial path, which has no eviction lever (the
    // 2026-07-23 soak burst-starvation root cause).
    let pool_hint = ctx.lifecycle.known_pool_commitment_bytes(&entry.id);
    let projected_bytes = ctx.projected_bytes.saturating_add(pool_hint);
    loop {
        // Charged, not just used: request-admission reservations awaiting
        // heartbeat confirmation are real obligations this load must not
        // double-spend (Phase 9 part 3 reservation ledger).
        let used = ctx.lifecycle.charged_bytes();
        let budget = ctx.lifecycle.budget_bytes();
        if used.saturating_add(projected_bytes) <= budget {
            break;
        }
        let Some(victim) = ctx.lifecycle.pick_victim(&entry.id) else {
            tracing::error!(model = %entry.id,
                projected_bytes, pool_hint_bytes = pool_hint,
                used_bytes = used, budget_bytes = budget,
                "load rejected: machine budget exceeded and no evictable model \
                 (candidates must be loaded, unpinned, and outside their TTL lease)");
            ctx.metrics
                .load_rejects_total
                .with_label_values(&[&entry.id, lifecycle::AdmitConstraint::Budget.label()])
                .inc();
            return RunExit::BudgetRejected;
        };
        tracing::warn!(model = %entry.id, victim = %victim,
            projected_bytes, pool_hint_bytes = pool_hint,
            used_bytes = used, budget_bytes = budget,
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
    // System-memory gate (2026-07-21 field finding): the budget above is
    // cut from INSTALLED memory and cannot see what other processes hold —
    // it admitted an 11.5 GB model on a 16 GB machine that was already
    // 4.4 GB into swap. Price the load against a FRESH probe of what the
    // OS can actually grant, after the eviction loop so freed bytes count.
    // No eviction on a system refusal: the budget check above already
    // handles Kiln-caused contention, so a shortfall here is external
    // (other processes) — evicting our own fleet cannot fix it and a
    // laggy kernel pressure signal would spiral through every victim.
    {
        let lifecycle = Arc::clone(&ctx.lifecycle);
        let projected = projected_bytes;
        // A panicked probe task must not take down supervision: fail open
        // (tokio logs the panic) exactly like a failed probe.
        let verdict = tokio::task::spawn_blocking(move || lifecycle.admit_load_system(projected))
            .await
            .unwrap_or(Ok(()));
        if let Err(denial) = verdict {
            tracing::error!(model = %entry.id,
                projected_bytes = denial.needed_bytes,
                system_available_bytes = denial.available_bytes,
                min_available_bytes = denial.min_available_bytes,
                swap_used_bytes = denial.swap_used_bytes,
                pressure_level = denial.pressure_level,
                constraint = denial.constraint.label(),
                budget_bytes = ctx.lifecycle.budget_bytes(),
                charged_bytes = ctx.lifecycle.charged_bytes(),
                "load rejected: the machine cannot grant these bytes without \
                 swapping (fits the configured budget, but real availability \
                 or kernel pressure says otherwise)");
            ctx.metrics
                .load_rejects_total
                .with_label_values(&[&entry.id, denial.constraint.label()])
                .inc();
            return RunExit::SystemRejected;
        }
    }
    // Reserve the projection; the first measured heartbeat replaces it
    // below, before the load permit is released.
    ctx.lifecycle.record_usage(&entry.id, projected_bytes);

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
    // The load priced the pool (reload path): convert that room into an
    // admission reservation BEFORE the measured-footprint swap below can
    // release it to racing request admissions. Seeded before the swap,
    // the interim double-count (projection + reservation) only refuses
    // racers for a moment — the conservative direction.
    if pool_hint > 0 {
        ctx.lifecycle.reserve_pool_room(&entry.id);
    }
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
    let mut stats = StatsReexport::new(&entry.id, entry.worker_kind, &ctx.metrics, last_ok);
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
                match bounded(timeout(HEALTH_RPC_TIMEOUT, client.health(HealthRequest {})).await) {
                    Ok(status) => {
                        if status.state() == WorkerState::Unhealthy {
                            tracing::error!(model = %entry.id, detail = %status.detail,
                                "worker self-reported UNHEALTHY; recycling");
                            kill_and_reap(&mut child, pgid, &entry.id).await;
                            release(ctx).await;
                            return RunExit::Crashed { ready_for: Some(ready_at.elapsed()) };
                        }
                        last_ok = Instant::now();
                        record_memory(ctx, &status);
                        // Keep the request path's system snapshot warm
                        // (background, rate-limited, one probe in flight).
                        ctx.lifecycle
                            .refresh_system_memory_soon(SYSTEM_PROBE_MAX_AGE);
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
                        if stats.polling() {
                            let reply = bounded(
                                timeout(HEALTH_RPC_TIMEOUT, client.stats(StatsRequest {})).await,
                            );
                            stats.observe(reply, &ctx.metrics, Instant::now());
                        }
                    }
                    Err(failure) => {
                        // No Health, no Stats: the re-export misses this
                        // tick's sample too. Crash detection stays Health's.
                        stats.health_failed(&failure, &ctx.metrics, Instant::now());
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

/// Why a deadline-bounded unary worker RPC produced no reply.
#[derive(Debug, thiserror::Error)]
enum RpcFailure {
    #[error("no reply within {}ms", HEALTH_RPC_TIMEOUT.as_millis())]
    TimedOut,
    #[error("{:?}: {}", .0.code(), .0.message())]
    Status(tonic::Status),
}

/// Folds the [`HEALTH_RPC_TIMEOUT`] deadline into the RPC's own result.
fn bounded<T>(
    result: Result<Result<tonic::Response<T>, tonic::Status>, Elapsed>,
) -> Result<T, RpcFailure> {
    match result {
        Ok(Ok(resp)) => Ok(resp.into_inner()),
        Ok(Err(status)) => Err(RpcFailure::Status(status)),
        Err(_) => Err(RpcFailure::TimedOut),
    }
}

/// Why a health-poll tick left the `Stats` re-export without a fresh
/// sample; labels `kiln_stats_samples_dropped_total{reason}`.
#[derive(Debug, Clone, Copy)]
enum StatsDrop {
    /// Health failed, so Stats was not asked this tick.
    HealthFailed,
    /// The Stats call exceeded [`HEALTH_RPC_TIMEOUT`].
    Timeout,
    /// The Stats call failed with a status other than UNIMPLEMENTED.
    Error,
    /// The worker does not serve Stats; polling stops for its lifetime.
    Unimplemented,
}

impl StatsDrop {
    const ALL: [Self; 4] = [
        Self::HealthFailed,
        Self::Timeout,
        Self::Error,
        Self::Unimplemented,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::HealthFailed => "health_failed",
            Self::Timeout => "timeout",
            Self::Error => "error",
            Self::Unimplemented => "unimplemented",
        }
    }
}

/// One worker process's `Stats` re-export (SPEC §5/§2.3): the polling
/// latch, plus the bookkeeping that makes a missed sample visible. A
/// missed sample leaves `/metrics` serving the previous snapshot, which
/// looks exactly like a healthy one, so every miss is counted by reason
/// and logged at warn, rate-limited per worker to one line per
/// [`STATS_DROP_WARN_INTERVAL`]. A fresh sample after a warned gap logs
/// once more, so the stale window is bracketed in the log.
///
/// Built at READY by each `run_once`, so it lives exactly as long as one
/// worker process. The UNIMPLEMENTED latch is therefore per process:
/// whether a worker serves Stats is fixed by its binary, so asking the
/// same process again cannot change the answer, and every restart (a
/// new process, possibly a new binary) probes afresh.
struct StatsReexport {
    model: String,
    kind: WorkerKind,
    /// False once this worker answered UNIMPLEMENTED.
    polling: bool,
    /// When `/metrics` last got a fresh sample (READY, before the first).
    fresh_at: Instant,
    /// Consecutive ticks without a fresh sample: the current gap.
    gap: u64,
    /// The current gap has been reported at warn.
    gap_warned: bool,
    last_warn: Option<Instant>,
    /// Drops counted since the last warn line but not logged on their own.
    unlogged: u64,
}

impl StatsReexport {
    fn new(model: &str, kind: WorkerKind, metrics: &Metrics, now: Instant) -> Self {
        // Every reason exists at 0 from READY, so a scrape tells "no
        // drops" (present, 0) apart from "not tracked" (absent).
        for reason in StatsDrop::ALL {
            metrics
                .stats_samples_dropped_total
                .with_label_values(&[model, reason.label()]);
        }
        Self {
            model: model.to_string(),
            kind,
            polling: true,
            fresh_at: now,
            gap: 0,
            gap_warned: false,
            last_warn: None,
            unlogged: 0,
        }
    }

    /// Whether Stats should be asked this tick.
    fn polling(&self) -> bool {
        self.polling
    }

    /// Applies one Stats reply (Health succeeded this tick).
    fn observe(&mut self, reply: Result<WorkerStats, RpcFailure>, metrics: &Metrics, now: Instant) {
        if !self.polling {
            return;
        }
        match reply {
            Ok(stats) => self.fresh(&stats, metrics, now),
            Err(RpcFailure::Status(status)) if status.code() == tonic::Code::Unimplemented => {
                self.disable(&status, metrics);
            }
            Err(failure @ RpcFailure::Status(_)) => {
                self.miss(StatsDrop::Error, &failure, metrics, now);
            }
            Err(failure @ RpcFailure::TimedOut) => {
                self.miss(StatsDrop::Timeout, &failure, metrics, now);
            }
        }
    }

    /// Health failed this tick, so Stats was not asked: a missed sample
    /// like any other, unless this worker has no Stats to miss.
    fn health_failed(&mut self, failure: &RpcFailure, metrics: &Metrics, now: Instant) {
        if self.polling {
            self.miss(StatsDrop::HealthFailed, failure, metrics, now);
        }
    }

    fn fresh(&mut self, stats: &WorkerStats, metrics: &Metrics, now: Instant) {
        metrics.worker_stats.record(&self.model, stats);
        if self.gap_warned {
            tracing::info!(model = %self.model, missed = self.gap,
                stale_ms = now.duration_since(self.fresh_at).as_millis() as u64,
                "worker Stats sampling recovered; /metrics is current again");
        }
        self.fresh_at = now;
        self.gap = 0;
        self.gap_warned = false;
    }

    fn miss(&mut self, reason: StatsDrop, failure: &RpcFailure, metrics: &Metrics, now: Instant) {
        metrics
            .stats_samples_dropped_total
            .with_label_values(&[&self.model, reason.label()])
            .inc();
        self.gap += 1;
        if self
            .last_warn
            .is_some_and(|at| now.duration_since(at) < STATS_DROP_WARN_INTERVAL)
        {
            self.unlogged += 1;
            return;
        }
        tracing::warn!(model = %self.model, reason = %reason.label(), error = %failure,
            consecutive = self.gap, suppressed = self.unlogged,
            stale_ms = now.duration_since(self.fresh_at).as_millis() as u64,
            "worker Stats sample dropped; /metrics keeps serving the last good sample \
             for this model");
        self.last_warn = Some(now);
        self.unlogged = 0;
        self.gap_warned = true;
    }

    fn disable(&mut self, status: &tonic::Status, metrics: &Metrics) {
        self.polling = false;
        metrics
            .stats_samples_dropped_total
            .with_label_values(&[&self.model, StatsDrop::Unimplemented.label()])
            .inc();
        match self.kind {
            WorkerKind::Python => tracing::info!(model = %self.model, worker = "python",
                detail = %status.message(),
                "worker does not serve Stats (expected for the python worker); its \
                 Stats-mirror series on /metrics stay unrefreshed until it restarts"),
            // The registry resolves `auto` before any spawn, so this is a
            // rust worker — which always serves Stats. UNIMPLEMENTED from
            // one means a stale or mismatched worker binary.
            WorkerKind::Rust | WorkerKind::Auto => tracing::warn!(model = %self.model,
                worker = self.kind.as_config_str(), detail = %status.message(),
                "rust worker answered Stats with UNIMPLEMENTED (stale or mismatched \
                 worker binary?); its Stats-mirror series on /metrics stay unrefreshed \
                 until it restarts"),
        }
    }
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

    /// Runs `f` under a thread-scoped fmt subscriber and returns the log
    /// lines it emitted (timestamps off, so lines are stable).
    fn logs(f: impl FnOnce()) -> Vec<String> {
        let sink = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let writer = Arc::clone(&sink);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || SinkWriter(Arc::clone(&writer)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .without_time()
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = sink.lock().expect("log sink").clone();
        String::from_utf8(bytes)
            .expect("utf-8 logs")
            .lines()
            .map(str::to_string)
            .collect()
    }

    struct SinkWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for SinkWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log sink").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn dropped(metrics: &Metrics, reason: &str) -> u64 {
        metrics
            .stats_samples_dropped_total
            .with_label_values(&["m", reason])
            .get()
    }

    fn unimplemented() -> Result<WorkerStats, RpcFailure> {
        Err(RpcFailure::Status(tonic::Status::unimplemented(
            "Method not found!",
        )))
    }

    #[test]
    fn stats_drop_series_exist_at_zero_from_ready() {
        let metrics = Metrics::new().expect("metrics build");
        let _stats = StatsReexport::new("m", WorkerKind::Rust, &metrics, Instant::now());
        let text = metrics.encode().expect("encode");
        for reason in ["health_failed", "timeout", "error", "unimplemented"] {
            let needle =
                format!("kiln_stats_samples_dropped_total{{model=\"m\",reason=\"{reason}\"}} 0");
            assert!(text.contains(&needle), "missing {needle} in:\n{text}");
        }
    }

    #[test]
    fn stats_drops_are_counted_by_reason_and_keep_the_last_sample() {
        let metrics = Metrics::new().expect("metrics build");
        let t0 = Instant::now();
        let at = |secs| t0 + Duration::from_secs(secs);
        let mut stats = StatsReexport::new("m", WorkerKind::Rust, &metrics, t0);
        let sample = WorkerStats {
            kv_blocks_allocated: 8,
            kv_blocks_free: 504,
            ..WorkerStats::default()
        };
        let lines = logs(|| {
            stats.observe(Ok(sample), &metrics, at(1));
            stats.observe(Err(RpcFailure::TimedOut), &metrics, at(2));
            let gone = tonic::Status::unavailable("socket gone");
            stats.observe(Err(RpcFailure::Status(gone)), &metrics, at(3));
            stats.health_failed(&RpcFailure::TimedOut, &metrics, at(4));
        });
        assert_eq!(dropped(&metrics, "timeout"), 1);
        assert_eq!(dropped(&metrics, "error"), 1);
        assert_eq!(dropped(&metrics, "health_failed"), 1);
        assert_eq!(dropped(&metrics, "unimplemented"), 0);
        assert!(stats.polling(), "transient failures never stop polling");
        // The scrape still carries the last good sample; the counter above
        // is what tells it apart from a fresh one.
        let text = metrics.encode().expect("encode");
        for needle in [
            "kiln_worker_kv_blocks_allocated{model=\"m\"} 8",
            "kiln_worker_kv_blocks_free{model=\"m\"} 504",
        ] {
            assert!(text.contains(needle), "missing {needle} in:\n{text}");
        }
        // One warn line for the gap, naming the model, the reason, and
        // how stale the re-export already was; the next two drops land
        // inside the rate-limit window and are only counted.
        assert_eq!(lines.len(), 1, "{lines:#?}");
        let warn = &lines[0];
        for needle in [
            "WARN",
            "worker Stats sample dropped",
            "model=m",
            "reason=timeout",
            "error=no reply within 2000ms",
            "consecutive=1",
            "stale_ms=1000",
        ] {
            assert!(warn.contains(needle), "missing {needle} in: {warn}");
        }
    }

    #[test]
    fn stats_drop_warnings_are_rate_limited_and_bracket_the_gap() {
        let metrics = Metrics::new().expect("metrics build");
        let t0 = Instant::now();
        let window = STATS_DROP_WARN_INTERVAL.as_secs();
        let at = |secs| t0 + Duration::from_secs(secs);
        let mut stats = StatsReexport::new("m", WorkerKind::Rust, &metrics, t0);

        // A gap of window + 2 dropped ticks: a warn at its first tick, one
        // more once the window has passed (carrying the count it held
        // back), and an info when sampling resumes.
        let lines = logs(|| {
            for tick in 1..=window + 2 {
                stats.observe(Err(RpcFailure::TimedOut), &metrics, at(tick));
            }
            stats.observe(Ok(WorkerStats::default()), &metrics, at(window + 3));
        });
        assert_eq!(dropped(&metrics, "timeout"), window + 2);
        assert_eq!(lines.len(), 3, "{lines:#?}");
        assert!(lines[0].contains("WARN") && lines[0].contains("suppressed=0"));
        assert!(
            lines[1].contains("WARN")
                && lines[1].contains(&format!("suppressed={}", window - 1))
                && lines[1].contains(&format!("consecutive={}", window + 1)),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("INFO")
                && lines[2].contains("worker Stats sampling recovered")
                && lines[2].contains("model=m")
                && lines[2].contains(&format!("missed={}", window + 2))
                // No sample since READY (t0), so stale from t0.
                && lines[2].contains(&format!("stale_ms={}", (window + 3) * 1000)),
            "{}",
            lines[2]
        );

        // Steady sampling is silent.
        let quiet = logs(|| stats.observe(Ok(WorkerStats::default()), &metrics, at(window + 4)));
        assert!(quiet.is_empty(), "{quiet:#?}");

        // A short gap inside the window of the last warn is counted but
        // not logged — and neither is its recovery, so the log never
        // reports the end of a gap it did not report the start of.
        let quiet = logs(|| {
            stats.health_failed(&RpcFailure::TimedOut, &metrics, at(window + 5));
            stats.observe(Ok(WorkerStats::default()), &metrics, at(window + 6));
        });
        assert!(quiet.is_empty(), "{quiet:#?}");
        assert_eq!(dropped(&metrics, "health_failed"), 1);

        // The next warn carries forward every drop logged only by the
        // counter since the last warn: the first gap's final tick and the
        // short gap's one.
        let lines = logs(|| {
            stats.observe(Err(RpcFailure::TimedOut), &metrics, at(2 * window + 1));
        });
        assert_eq!(lines.len(), 1, "{lines:#?}");
        assert!(
            lines[0].contains("suppressed=2") && lines[0].contains("consecutive=1"),
            "{}",
            lines[0]
        );
    }

    #[test]
    fn stats_unimplemented_latches_once_and_logs_by_worker_kind() {
        for (kind, level, other) in [
            (WorkerKind::Python, "INFO", "expected for the python worker"),
            (
                WorkerKind::Rust,
                "WARN",
                "stale or mismatched worker binary",
            ),
        ] {
            let metrics = Metrics::new().expect("metrics build");
            let t0 = Instant::now();
            let at = |secs| t0 + Duration::from_secs(secs);
            let mut stats = StatsReexport::new("m", kind, &metrics, t0);
            let lines = logs(|| {
                stats.observe(unimplemented(), &metrics, at(1));
                // Latched: no further Stats calls, so nothing after this
                // is a dropped sample, and nothing more is logged.
                stats.observe(unimplemented(), &metrics, at(2));
                stats.health_failed(&RpcFailure::TimedOut, &metrics, at(3));
                stats.observe(Err(RpcFailure::TimedOut), &metrics, at(4));
            });
            assert!(!stats.polling(), "{kind:?}: UNIMPLEMENTED stops polling");
            assert_eq!(dropped(&metrics, "unimplemented"), 1, "{kind:?}");
            assert_eq!(dropped(&metrics, "health_failed"), 0, "{kind:?}");
            assert_eq!(dropped(&metrics, "timeout"), 0, "{kind:?}");
            assert_eq!(lines.len(), 1, "{kind:?}: logged once: {lines:#?}");
            for needle in [level, "model=m", "Method not found!", other] {
                assert!(
                    lines[0].contains(needle),
                    "missing {needle} in: {}",
                    lines[0]
                );
            }
        }
        // The latch belongs to one worker process: the next one probes.
        let metrics = Metrics::new().expect("metrics build");
        let next = StatsReexport::new("m", WorkerKind::Python, &metrics, Instant::now());
        assert!(next.polling());
    }

    #[tokio::test]
    async fn bounded_folds_the_deadline_into_the_rpc_result() {
        let elapsed = timeout(Duration::ZERO, std::future::pending::<()>())
            .await
            .expect_err("a pending future cannot beat a zero deadline");
        assert!(matches!(
            bounded::<WorkerStats>(Err(elapsed)),
            Err(RpcFailure::TimedOut)
        ));
        assert!(matches!(
            bounded::<WorkerStats>(Ok(Err(tonic::Status::unimplemented("")))),
            Err(RpcFailure::Status(status)) if status.code() == tonic::Code::Unimplemented
        ));
        let reply = bounded(Ok(Ok(tonic::Response::new(WorkerStats {
            requests_total: 3,
            ..WorkerStats::default()
        }))));
        assert_eq!(reply.expect("reply").requests_total, 3);
        // The warn line's `error=` field: code and message, no metadata.
        assert_eq!(
            RpcFailure::Status(tonic::Status::unavailable("socket gone")).to_string(),
            "Unavailable: socket gone"
        );
        assert_eq!(RpcFailure::TimedOut.to_string(), "no reply within 2000ms");
    }
}
