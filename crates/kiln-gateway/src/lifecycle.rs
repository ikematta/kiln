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

    /// Sum of bytes currently charged against the budget.
    pub fn used_bytes(&self) -> u64 {
        self.slots
            .values()
            .map(|slot| slot.usage_bytes.load(Ordering::Acquire))
            .sum()
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
            self.export_used();
        }
    }

    /// Releases the model's bytes (worker exited: unload or crash).
    pub(crate) fn clear_usage(&self, model_id: &str) {
        self.record_usage(model_id, 0);
    }

    fn export_used(&self) {
        self.metrics
            .memory_used_bytes
            .set(i64::try_from(self.used_bytes()).unwrap_or(i64::MAX));
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
