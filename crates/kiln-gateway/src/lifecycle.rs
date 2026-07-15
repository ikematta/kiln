//! Machine-level memory governance (SPEC §2.3) and the model lifecycle
//! handle (SPEC §2.2, Phase 9 part 1).
//!
//! One [`Lifecycle`] per gateway holds the machine memory budget
//! (`total_unified_memory × memory.budget_fraction`, or the explicit
//! `memory.budget_bytes` override) and a per-model slot: the supervision
//! task's command channel, the bytes currently charged against the budget
//! (a load-time reservation until the first heartbeat, then the measured
//! footprint from `Health`'s `MemoryReport`), and the LRU clock the
//! request path advances via [`Lifecycle::touch`].
//!
//! Eviction policy (SPEC §2.2): when a load would exceed the budget, the
//! least-recently-used model that is loaded, not pinned, and not inside
//! its TTL keep-alive lease is evicted first; with no such candidate the
//! load is rejected. `ttl_seconds` is a keep-alive lease: within
//! `ttl_seconds` of last use a model is protected from LRU eviction, and
//! once idle past it the supervisor auto-unloads it anyway.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use kiln_proto::v1::MemoryReport;
use tokio::sync::{mpsc, oneshot, watch};

use crate::config::KilnConfig;
use crate::metrics::Metrics;
use crate::registry::{Registry, UnloadReason, WorkerStatus};

/// Upper bound on one eviction round-trip: graceful drain deadline plus
/// SIGTERM grace plus scheduling slack (supervisor constants).
const EVICT_ACK_TIMEOUT: Duration = Duration::from_secs(45);

/// Supervision-task commands, sent through the per-model slot.
pub(crate) enum Command {
    /// (Re)load the model. No-op while loaded; ignored while `Failed`
    /// (manual reset required, SPEC §2.2).
    Load,
    /// Unload the running worker (Drain → SIGTERM → SIGKILL-after-grace).
    /// `done` fires once the process is gone and its bytes are released.
    Unload {
        reason: UnloadReason,
        done: oneshot::Sender<()>,
    },
}

pub(crate) struct Slot {
    cmd_tx: mpsc::UnboundedSender<Command>,
    status: watch::Receiver<WorkerStatus>,
    pinned: bool,
    ttl: Option<Duration>,
    /// Bytes currently charged against the machine budget; 0 = not loaded.
    usage_bytes: AtomicU64,
    /// Millis since [`Lifecycle::epoch`] of last request activity;
    /// initialized at READY, advanced by [`Lifecycle::touch`] and by the
    /// supervisor while the worker reports in-flight work.
    last_used_ms: AtomicU64,
    /// Full KV-pool cost once materialized (whole worker: target + draft
    /// pools, `WorkerInfo.kv_pool_commitment_bytes`, set at READY): what
    /// serving traffic on this worker will grow it to. 0 for non-paged
    /// workers (python) — no projection is possible, so requests are not
    /// gated.
    pool_commitment_bytes: AtomicU64,
    /// Pool bytes actually materialized so far (heartbeat
    /// `kv_pool_allocated_bytes`); the commitment minus this is the growth
    /// a request can still trigger.
    pool_materialized_bytes: AtomicU64,
    /// Pool growth ADMITTED but not yet confirmed by a heartbeat (Phase 9
    /// part 3 reservation ledger): charged against the budget from the
    /// admission decision onward, so concurrent admissions price against
    /// reserved-but-unconfirmed bytes instead of heartbeat-lagged
    /// footprints (the run-29436961038 TOCTOU). Reconciled downward as
    /// heartbeats report materialization; cleared with the worker. A
    /// reservation whose request never materializes the pool (cancel,
    /// error) lingers conservatively — it only ever under-reports
    /// headroom, and the next served request or unload reconciles it.
    pool_reserved_bytes: AtomicU64,
}

/// Why [`Lifecycle::admit_request`] refused a request: admitting it could
/// grow the worker by `needed_bytes` but only `headroom_bytes` of the
/// machine budget remain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryDenial {
    pub needed_bytes: u64,
    pub headroom_bytes: u64,
}

pub struct Lifecycle {
    budget_bytes: u64,
    total_bytes: Option<u64>,
    slots: HashMap<String, Slot>,
    /// One model (re)load at a time, machine-wide: budget acquisition,
    /// eviction, spawn, and wait-for-READY all happen under this permit,
    /// so concurrent loads cannot double-spend the same headroom (and one
    /// GPU only loads one weight set at a time anyway).
    load_permit: tokio::sync::Mutex<()>,
    /// Serializes admission check-and-reserve (and heartbeat
    /// reconciliation) so two admissions cannot both read the same
    /// headroom before either's reservation lands. Held only for
    /// non-blocking arithmetic — never across await.
    admission_lock: std::sync::Mutex<()>,
    metrics: Arc<Metrics>,
    epoch: Instant,
}

impl Lifecycle {
    /// Builds the budget and one slot per registry entry. The returned
    /// receivers are handed to the supervision tasks (same order as
    /// `registry.iter()`).
    pub(crate) fn new(
        config: &KilnConfig,
        registry: &Registry,
        metrics: Arc<Metrics>,
    ) -> Result<(Self, Vec<mpsc::UnboundedReceiver<Command>>), String> {
        let (budget_bytes, total_bytes) = machine_budget(config)?;
        metrics
            .memory_budget_bytes
            .set(i64::try_from(budget_bytes).unwrap_or(i64::MAX));

        let mut slots = HashMap::new();
        let mut receivers = Vec::new();
        for entry in registry.iter() {
            let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
            slots.insert(
                entry.id.clone(),
                Slot {
                    cmd_tx,
                    status: entry.status.clone(),
                    pinned: entry.config.pinned,
                    ttl: match entry.config.ttl_seconds {
                        0 => None,
                        secs => Some(Duration::from_secs(secs)),
                    },
                    usage_bytes: AtomicU64::new(0),
                    last_used_ms: AtomicU64::new(0),
                    pool_commitment_bytes: AtomicU64::new(0),
                    pool_materialized_bytes: AtomicU64::new(0),
                    pool_reserved_bytes: AtomicU64::new(0),
                },
            );
            receivers.push(cmd_rx);
        }
        Ok((
            Self {
                budget_bytes,
                total_bytes,
                slots,
                load_permit: tokio::sync::Mutex::new(()),
                admission_lock: std::sync::Mutex::new(()),
                metrics,
                epoch: Instant::now(),
            },
            receivers,
        ))
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn total_bytes(&self) -> Option<u64> {
        self.total_bytes
    }

    /// Sum of bytes currently charged against the budget from measured
    /// footprints (and load-time reservations recorded as usage).
    pub fn used_bytes(&self) -> u64 {
        self.slots
            .values()
            .map(|slot| slot.usage_bytes.load(Ordering::Acquire))
            .sum()
    }

    /// Admission reservations not yet confirmed by heartbeats.
    pub fn reserved_bytes(&self) -> u64 {
        self.slots
            .values()
            .map(|slot| slot.pool_reserved_bytes.load(Ordering::Acquire))
            .sum()
    }

    /// Everything the budget currently owes: measured/load-reserved usage
    /// PLUS admitted-but-unconfirmed pool growth. This — not
    /// [`Self::used_bytes`] — is what every admission decision (request
    /// growth and load alike) must price against, so decisions racing on
    /// heartbeat-stale state cannot jointly overshoot (Phase 9 part 3
    /// ruling, run 29436961038).
    pub fn charged_bytes(&self) -> u64 {
        self.used_bytes().saturating_add(self.reserved_bytes())
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// Marks the model as just-used (request routed to it, or the worker
    /// reported in-flight work): resets both the LRU position and the TTL
    /// idle clock.
    pub fn touch(&self, model_id: &str) {
        if let Some(slot) = self.slots.get(model_id) {
            slot.last_used_ms.store(self.now_ms(), Ordering::Release);
        }
    }

    /// Idle time since last use (for the supervisor's TTL sweep).
    pub(crate) fn idle(&self, model_id: &str) -> Duration {
        let Some(slot) = self.slots.get(model_id) else {
            return Duration::ZERO;
        };
        let last = slot.last_used_ms.load(Ordering::Acquire);
        Duration::from_millis(self.now_ms().saturating_sub(last))
    }

    /// Asks the supervision task to (re)load an unloaded model — the
    /// on-demand path behind TTL unloads and evictions. Cheap and
    /// non-blocking; a no-op unless the model is currently `Unloaded`.
    pub fn request_load(&self, model_id: &str) {
        if let Some(slot) = self.slots.get(model_id)
            && matches!(*slot.status.borrow(), WorkerStatus::Unloaded { .. })
        {
            let _ = slot.cmd_tx.send(Command::Load);
        }
    }

    /// Unconditional load for the startup bootstrap, which fires while the
    /// status is still the registry-initial `Starting` (so `request_load`'s
    /// `Unloaded` gate would drop it).
    pub(crate) fn boot_load(&self, model_id: &str) {
        if let Some(slot) = self.slots.get(model_id) {
            let _ = slot.cmd_tx.send(Command::Load);
        }
    }

    /// Charges `bytes` for the model (reservation at load, measured
    /// footprint on every heartbeat).
    pub(crate) fn record_usage(&self, model_id: &str, bytes: u64) {
        if let Some(slot) = self.slots.get(model_id) {
            slot.usage_bytes.store(bytes, Ordering::Release);
            self.export_gauges();
        }
    }

    /// Releases the model's bytes (worker exited: unload or crash),
    /// including its pool projection state and any pending reservation.
    pub(crate) fn clear_usage(&self, model_id: &str) {
        self.record_usage(model_id, 0);
        if let Some(slot) = self.slots.get(model_id) {
            slot.pool_commitment_bytes.store(0, Ordering::Release);
            slot.pool_materialized_bytes.store(0, Ordering::Release);
            slot.pool_reserved_bytes.store(0, Ordering::Release);
            self.export_gauges();
        }
    }

    /// Records the worker's full-pool cost (READY-time `WorkerInfo`
    /// geometry) for [`Lifecycle::admit_request`] projections.
    pub(crate) fn set_pool_commitment(&self, model_id: &str, bytes: u64) {
        if let Some(slot) = self.slots.get(model_id) {
            slot.pool_commitment_bytes.store(bytes, Ordering::Release);
        }
    }

    /// Records the pool bytes a heartbeat reports as materialized, and
    /// reconciles the reservation ledger against the now-confirmed
    /// reality (Phase 9 part 3 ruling):
    /// - the reservation shrinks to the growth still outstanding
    ///   (commitment − materialized) — confirmed bytes are in the
    ///   footprint now, so holding them reserved would double-charge;
    ///   over-reservations release the difference the same way;
    /// - materialization that NO reservation covered is alertable
    ///   (tracing::warn + kiln_admission_uncovered_bytes_total): memory
    ///   grew without being priced by an admission, which the ledger is
    ///   supposed to make impossible.
    pub(crate) fn record_pool_materialized(&self, model_id: &str, bytes: u64) {
        let Some(slot) = self.slots.get(model_id) else {
            return;
        };
        let _guard = self
            .admission_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let commitment = slot.pool_commitment_bytes.load(Ordering::Acquire);
        let previous = slot.pool_materialized_bytes.load(Ordering::Acquire);
        let reserved = slot.pool_reserved_bytes.load(Ordering::Acquire);
        // Growth inside the commitment since the last heartbeat (heartbeat
        // totals can exceed the commitment only if the projection is
        // incomplete — that excess is exactly what must alert).
        let delta = bytes.saturating_sub(previous);
        if delta > reserved && commitment > 0 {
            let uncovered = delta - reserved;
            tracing::warn!(model = %model_id, uncovered_bytes = uncovered,
                materialized_bytes = bytes, reserved_bytes = reserved,
                commitment_bytes = commitment,
                "pool grew beyond its admission reservation — unpriced memory");
            self.metrics
                .admission_uncovered_bytes_total
                .with_label_values(&[model_id])
                .inc_by(uncovered);
        }
        slot.pool_materialized_bytes.store(bytes, Ordering::Release);
        let outstanding = commitment.saturating_sub(bytes);
        slot.pool_reserved_bytes
            .store(reserved.min(outstanding), Ordering::Release);
        drop(_guard);
        self.export_gauges();
    }

    /// Per-request admission check (SPEC §2.3 second level / §8.2
    /// "admission check", Phase 9 part 2): serving a request on this
    /// worker may materialize the rest of its lazily-allocated KV pool —
    /// real machine bytes the load-time budget check never saw. Runs
    /// against LIVE numbers on every request: projected growth (full-pool
    /// commitment minus what heartbeats say is already materialized) must
    /// fit the machine headroom (budget minus the sum of measured
    /// footprints), so usage drift since load — pools, caches — is what
    /// the check prices in. Workers without pool geometry (python, or
    /// GetInfo not yet cached) are not gated; a fully-materialized pool
    /// projects zero growth and always passes — its bytes are already in
    /// the footprint sum.
    /// Reservation semantics (Phase 9 part 3 ruling, Option A): a passing
    /// admission RESERVES the projected growth against the budget under
    /// [`Self::admission_lock`], immediately visible to every concurrent
    /// decision through [`Self::charged_bytes`] — two admissions racing
    /// on heartbeat-stale footprints can no longer jointly overshoot.
    /// Growth already covered by an outstanding reservation (a second
    /// request on the same still-cold pool) passes without re-reserving:
    /// the pool materializes once. Heartbeats reconcile reservations in
    /// [`Self::record_pool_materialized`].
    pub fn admit_request(&self, model_id: &str) -> Result<(), MemoryDenial> {
        let Some(slot) = self.slots.get(model_id) else {
            return Ok(());
        };
        let _guard = self
            .admission_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let commitment = slot.pool_commitment_bytes.load(Ordering::Acquire);
        // Saturating: a heartbeat total exceeding the commitment means an
        // incomplete projection (alerted in record_pool_materialized) —
        // never a negative growth.
        let growth = commitment
            .saturating_sub(slot.pool_materialized_bytes.load(Ordering::Acquire))
            .saturating_sub(slot.pool_reserved_bytes.load(Ordering::Acquire));
        if growth == 0 {
            return Ok(());
        }
        let headroom = self.budget_bytes.saturating_sub(self.charged_bytes());
        if growth <= headroom {
            slot.pool_reserved_bytes.fetch_add(growth, Ordering::AcqRel);
            drop(_guard);
            self.export_gauges();
            return Ok(());
        }
        Err(MemoryDenial {
            needed_bytes: growth,
            headroom_bytes: headroom,
        })
    }

    fn export_gauges(&self) {
        self.metrics
            .memory_used_bytes
            .set(i64::try_from(self.used_bytes()).unwrap_or(i64::MAX));
        self.metrics
            .memory_reserved_bytes
            .set(i64::try_from(self.reserved_bytes()).unwrap_or(i64::MAX));
    }

    pub(crate) fn load_permit(&self) -> &tokio::sync::Mutex<()> {
        &self.load_permit
    }

    /// LRU eviction candidate, excluding the model asking for room:
    /// loaded (READY, bytes charged), not pinned, not inside its TTL
    /// keep-alive lease. `None` = the load must be rejected.
    pub(crate) fn pick_victim(&self, exclude: &str) -> Option<String> {
        let now = self.now_ms();
        self.slots
            .iter()
            .filter(|(id, slot)| {
                id.as_str() != exclude
                    && !slot.pinned
                    && slot.usage_bytes.load(Ordering::Acquire) > 0
                    && *slot.status.borrow() == WorkerStatus::Ready
                    && slot.ttl.is_none_or(|ttl| {
                        let idle = now.saturating_sub(slot.last_used_ms.load(Ordering::Acquire));
                        Duration::from_millis(idle) >= ttl
                    })
            })
            .min_by_key(|(_, slot)| slot.last_used_ms.load(Ordering::Acquire))
            .map(|(id, _)| id.clone())
    }

    /// Evicts `victim` and waits for its memory to be released. False =
    /// the eviction could not be confirmed (task gone or ack timed out);
    /// the caller must not assume the bytes came back.
    pub(crate) async fn evict(&self, victim: &str) -> bool {
        let Some(slot) = self.slots.get(victim) else {
            return false;
        };
        let (done_tx, done_rx) = oneshot::channel();
        if slot
            .cmd_tx
            .send(Command::Unload {
                reason: UnloadReason::Evicted,
                done: done_tx,
            })
            .is_err()
        {
            return false;
        }
        matches!(
            tokio::time::timeout(EVICT_ACK_TIMEOUT, done_rx).await,
            Ok(Ok(()))
        )
    }
}

/// Resolves the machine budget: the explicit `memory.budget_bytes`
/// override, else `total_unified_memory × memory.budget_fraction`
/// (SPEC §2.3). Errors only when no budget can be established.
fn machine_budget(config: &KilnConfig) -> Result<(u64, Option<u64>), String> {
    let total = total_unified_memory();
    if let Some(bytes) = config.memory.budget_bytes {
        return Ok((bytes, total));
    }
    match total {
        Some(total) => {
            let budget = (total as f64 * config.memory.budget_fraction) as u64;
            Ok((budget, Some(total)))
        }
        None => Err(
            "could not determine total unified memory for the machine budget; \
             set memory.budget_bytes explicitly"
                .to_string(),
        ),
    }
}

/// Total unified memory of this machine. Shelled out (`sysctl` /
/// `/proc/meminfo`) because reading it natively needs libc calls and
/// unsafe code is confined to kiln-mlx (CLAUDE.md), mirroring the
/// supervisor's `/bin/kill` precedent.
pub fn total_unified_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux is a compile-check target only (SPEC §1.2), but the fallback
        // keeps any incidental run honest.
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = text
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
}

/// Machine-memory footprint of one worker from its heartbeat report:
/// every MLX byte counted once. `mlx_active` already contains the weight
/// arrays and the preallocated KV pools, so weights + pool serve as a
/// floor under it (a worker whose runtime cannot report active memory
/// still gets charged for what it demonstrably holds), and the MLX buffer
/// cache — freed but retained — is real machine memory on top.
pub fn footprint_bytes(report: &MemoryReport) -> u64 {
    report
        .mlx_active_bytes
        .max(report.weights_bytes + report.kv_pool_allocated_bytes)
        + report.mlx_cache_bytes
}

/// Load-time projection for a model directory: the weight bytes on disk
/// (`*.safetensors`; quantized MLX arrays stay packed, so resident weight
/// bytes ≈ file bytes). An understatement of the full footprint — the KV
/// pool and cache land on top — which the first post-READY heartbeat
/// replaces with the measured number before the load permit is released.
pub fn weights_bytes_on_disk(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            if !name.to_string_lossy().ends_with(".safetensors") {
                return None;
            }
            Some(entry.metadata().ok()?.len())
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_unified_memory_is_detected() {
        let total = total_unified_memory().expect("detectable on dev/CI machines");
        assert!(total > 1 << 30, "implausibly small: {total}");
    }

    #[test]
    fn footprint_counts_every_mlx_byte_once() {
        // Active covers weights + pool: footprint = active + cache.
        let report = MemoryReport {
            weights_bytes: 700,
            kv_pool_allocated_bytes: 500,
            mlx_active_bytes: 1300,
            mlx_cache_bytes: 100,
            ..MemoryReport::default()
        };
        assert_eq!(footprint_bytes(&report), 1400);
        // Active unavailable (reported 0): weights + pool are the floor.
        let report = MemoryReport {
            weights_bytes: 700,
            kv_pool_allocated_bytes: 500,
            mlx_active_bytes: 0,
            mlx_cache_bytes: 100,
            ..MemoryReport::default()
        };
        assert_eq!(footprint_bytes(&report), 1300);
    }

    #[test]
    fn weights_on_disk_sums_only_safetensors() {
        let dir = std::env::temp_dir().join(format!(
            "kiln-lifecycle-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 128]).expect("write");
        std::fs::write(dir.join("model-2.safetensors"), vec![0u8; 64]).expect("write");
        std::fs::write(dir.join("config.json"), b"{}").expect("write");
        assert_eq!(weights_bytes_on_disk(&dir), 192);
        assert_eq!(weights_bytes_on_disk(&dir.join("missing")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Hand-built Lifecycle for victim-selection tests; the returned
    /// watch senders keep slot statuses alive/settable.
    fn lifecycle_with(
        slots: Vec<(&str, bool, Option<Duration>, u64, u64)>,
    ) -> (Lifecycle, Vec<watch::Sender<WorkerStatus>>) {
        let metrics = Arc::new(Metrics::new().expect("metrics"));
        let mut map = HashMap::new();
        let mut senders = Vec::new();
        for (id, pinned, ttl, usage, last_used_ms) in slots {
            let (status_tx, status_rx) = watch::channel(WorkerStatus::Ready);
            let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
            // The receiver end is dropped; these tests never send commands.
            map.insert(
                id.to_string(),
                Slot {
                    cmd_tx,
                    status: status_rx,
                    pinned,
                    ttl,
                    usage_bytes: AtomicU64::new(usage),
                    last_used_ms: AtomicU64::new(last_used_ms),
                    pool_commitment_bytes: AtomicU64::new(0),
                    pool_materialized_bytes: AtomicU64::new(0),
                    pool_reserved_bytes: AtomicU64::new(0),
                },
            );
            senders.push(status_tx);
        }
        (
            Lifecycle {
                budget_bytes: 1000,
                total_bytes: None,
                slots: map,
                load_permit: tokio::sync::Mutex::new(()),
                admission_lock: std::sync::Mutex::new(()),
                metrics,
                epoch: Instant::now() - Duration::from_secs(3600),
            },
            senders,
        )
    }

    #[test]
    fn victim_is_lru_among_unpinned_unleased_loaded_models() {
        let hour_ms = 3_600_000u64;
        let (lifecycle, _senders) = lifecycle_with(vec![
            // Pinned: never a victim, even as the machine-wide LRU.
            ("pinned", true, None, 400, 0),
            // Unloaded (no bytes charged): nothing to evict.
            ("unloaded", false, None, 0, 10),
            // Inside its TTL keep-alive lease (used just now, 10-min ttl).
            (
                "leased",
                false,
                Some(Duration::from_secs(600)),
                400,
                hour_ms,
            ),
            // TTL lease expired (used at t=0, 1s ttl): fair game.
            ("expired", false, Some(Duration::from_secs(1)), 400, 100),
            // Plain unpinned models; "old" is least recently used.
            ("old", false, None, 400, 200),
            ("new", false, None, 400, hour_ms),
        ]);
        // "expired" (100ms) beats "old" (200ms) on the LRU clock.
        assert_eq!(lifecycle.pick_victim("loader").as_deref(), Some("expired"));
        // The asking model never evicts itself.
        assert_eq!(lifecycle.pick_victim("expired").as_deref(), Some("old"));

        let (lifecycle, _senders) = lifecycle_with(vec![
            ("pinned", true, None, 400, 0),
            (
                "leased",
                false,
                Some(Duration::from_secs(600)),
                400,
                hour_ms,
            ),
        ]);
        // Only pinned/leased candidates left: the load must be rejected.
        assert_eq!(lifecycle.pick_victim("loader"), None);
    }

    #[test]
    fn victim_must_be_ready() {
        let (lifecycle, senders) = lifecycle_with(vec![("draining", false, None, 400, 0)]);
        assert_eq!(lifecycle.pick_victim("loader").as_deref(), Some("draining"));
        senders[0].send_replace(WorkerStatus::Draining);
        assert_eq!(lifecycle.pick_victim("loader"), None);
    }

    #[test]
    fn admit_request_gates_on_projected_pool_growth() {
        // One loaded worker charged 700 of a 1000-byte budget.
        let (lifecycle, _senders) = lifecycle_with(vec![("m", false, None, 700, 0)]);

        // No pool geometry recorded (python worker / GetInfo pending):
        // never gated.
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
        // Unknown model: the ready_entry 404 owns that path, not this gate.
        assert_eq!(lifecycle.admit_request("ghost"), Ok(()));

        // Pool commitment 500, nothing materialized: growth 500 > headroom
        // 300 — rejected with the numbers.
        lifecycle.set_pool_commitment("m", 500);
        assert_eq!(
            lifecycle.admit_request("m"),
            Err(MemoryDenial {
                needed_bytes: 500,
                headroom_bytes: 300,
            })
        );

        // Growth within headroom passes and is RESERVED (part 3): 250
        // already materialized leaves 250 to grow against 300 free.
        lifecycle.record_pool_materialized("m", 250);
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
        assert_eq!(lifecycle.reserved_bytes(), 250);

        // Fully materialized: the reservation reconciles away and zero
        // growth always passes — the bytes are in the footprint sum
        // already, even if the machine is over budget.
        lifecycle.record_pool_materialized("m", 500);
        assert_eq!(lifecycle.reserved_bytes(), 0);
        lifecycle.record_usage("m", 1200);
        assert_eq!(lifecycle.admit_request("m"), Ok(()));

        // A heartbeat total EXCEEDING the whole-worker commitment means
        // the projection was incomplete: growth still saturates to zero
        // (the admit passes; the bytes are in the footprint), but the
        // overflow is alertable as uncovered growth. Counter total: the
        // 250 this test materialized with no admission covering it
        // (step above) + this 400 overflow — every unpriced byte counts.
        lifecycle.record_pool_materialized("m", 900);
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
        assert_eq!(
            lifecycle
                .metrics
                .admission_uncovered_bytes_total
                .with_label_values(&["m"])
                .get(),
            650
        );

        // Worker gone: projection state clears with the usage.
        lifecycle.record_pool_materialized("m", 0);
        lifecycle.clear_usage("m");
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
    }

    #[test]
    fn admit_request_prices_in_other_models_drift() {
        // Two workers under a 1000-byte budget: "hot" drifted to 600 bytes
        // (pool materialized under traffic), "cold" idles at 250 with a
        // 400-byte pool nothing has touched. The load-time check passed
        // both; the per-request check must refuse cold's first request.
        let (lifecycle, _senders) = lifecycle_with(vec![
            ("hot", false, None, 600, 0),
            ("cold", false, None, 250, 0),
        ]);
        lifecycle.set_pool_commitment("cold", 400);
        assert_eq!(
            lifecycle.admit_request("cold"),
            Err(MemoryDenial {
                needed_bytes: 400,
                headroom_bytes: 150,
            })
        );
        // The hot model unloads: headroom recovers, the same request passes.
        lifecycle.clear_usage("hot");
        assert_eq!(lifecycle.admit_request("cold"), Ok(()));
    }

    #[test]
    fn concurrent_admissions_cannot_jointly_overshoot() {
        // The run-29436961038 TOCTOU as arithmetic: two cold models, usage
        // 250+250 of a 1000 budget, commitments 400 each. Pre-reservation,
        // both admissions read headroom 500 against heartbeat-stale
        // footprints and both passed — joint overshoot 300. Now the first
        // admission's reservation is immediately visible to the second.
        let (lifecycle, _senders) =
            lifecycle_with(vec![("a", false, None, 250, 0), ("b", false, None, 250, 0)]);
        lifecycle.set_pool_commitment("a", 400);
        lifecycle.set_pool_commitment("b", 400);
        assert_eq!(lifecycle.admit_request("a"), Ok(()));
        assert_eq!(lifecycle.reserved_bytes(), 400);
        assert_eq!(lifecycle.charged_bytes(), 900);
        assert_eq!(
            lifecycle.admit_request("b"),
            Err(MemoryDenial {
                needed_bytes: 400,
                headroom_bytes: 100,
            })
        );
        // A refusal reserves nothing.
        assert_eq!(lifecycle.reserved_bytes(), 400);
        // A second request on the same still-cold pool rides the
        // outstanding reservation: admitted, no double charge.
        assert_eq!(lifecycle.admit_request("a"), Ok(()));
        assert_eq!(lifecycle.reserved_bytes(), 400);
    }

    #[test]
    fn heartbeats_reconcile_reservations() {
        let (lifecycle, _senders) = lifecycle_with(vec![("m", false, None, 300, 0)]);
        lifecycle.set_pool_commitment("m", 400);
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
        assert_eq!(lifecycle.reserved_bytes(), 400);
        assert_eq!(lifecycle.charged_bytes(), 700);

        // Partial materialization: confirmed bytes now live in the
        // measured footprint; the reservation shrinks to the outstanding
        // growth — no double counting in either direction.
        lifecycle.record_pool_materialized("m", 150);
        lifecycle.record_usage("m", 450);
        assert_eq!(lifecycle.reserved_bytes(), 250);
        assert_eq!(lifecycle.charged_bytes(), 700);

        // Full materialization releases the reservation entirely.
        lifecycle.record_pool_materialized("m", 400);
        lifecycle.record_usage("m", 700);
        assert_eq!(lifecycle.reserved_bytes(), 0);
        assert_eq!(lifecycle.charged_bytes(), 700);

        // Every byte was priced before it materialized: nothing alerted.
        assert_eq!(
            lifecycle
                .metrics
                .admission_uncovered_bytes_total
                .with_label_values(&["m"])
                .get(),
            0
        );

        // Worker exit clears usage, projection, and reservation alike.
        lifecycle.clear_usage("m");
        assert_eq!(lifecycle.charged_bytes(), 0);
    }

    #[test]
    fn uncovered_growth_is_alertable() {
        let (lifecycle, _senders) = lifecycle_with(vec![("m", false, None, 300, 0)]);
        lifecycle.set_pool_commitment("m", 400);

        // Pool growth with NO covering reservation (an admission bypass,
        // or an under-projection): alertable, byte-accurate.
        lifecycle.record_pool_materialized("m", 100);
        assert_eq!(
            lifecycle
                .metrics
                .admission_uncovered_bytes_total
                .with_label_values(&["m"])
                .get(),
            100
        );

        // Growth covered by a reservation is silent.
        assert_eq!(lifecycle.admit_request("m"), Ok(()));
        assert_eq!(lifecycle.reserved_bytes(), 300);
        lifecycle.record_pool_materialized("m", 400);
        assert_eq!(
            lifecycle
                .metrics
                .admission_uncovered_bytes_total
                .with_label_values(&["m"])
                .get(),
            100
        );
        assert_eq!(lifecycle.reserved_bytes(), 0);
    }

    #[test]
    fn explicit_budget_bytes_overrides_the_fraction() {
        let mut config = KilnConfig::default();
        config.memory.budget_bytes = Some(1234);
        let (budget, _) = machine_budget(&config).expect("explicit budget");
        assert_eq!(budget, 1234);

        config.memory.budget_bytes = None;
        config.memory.budget_fraction = 0.5;
        let (budget, total) = machine_budget(&config).expect("fraction budget");
        let total = total.expect("total detected");
        assert_eq!(budget, (total as f64 * 0.5) as u64);
    }
}
