//! Worker-side engine wiring (Phase 4): one dedicated OS thread owns every
//! MLX value — the model, the `Stream`, and kiln-engine's continuous
//! batching loop (SPEC §6.2), which is `!Send` by construction. gRPC
//! handler tasks only enqueue submissions and read event channels.
//!
//! Requests stream concurrently: the loop drains the submission channel
//! between engine iterations, so new requests join the running batch at
//! the next step. Cancellation (and Drain, which cancels through the same
//! flags) is flag-based, checked between steps — well inside the ≤2-step
//! budget `Cancel` promises. Memory pressure is the engine's job now
//! (SPEC §6.1 preemption); the worker only surfaces the counters.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, ErrorCause, FinishKind, PenaltyOptions, SamplingOptions,
    SeqEvent, SsdParams,
};
use kiln_mlx::Stream;
use kiln_proto::v1::{
    FinishReason, Finished, MemoryReport, PrefixCacheHit, Priority, SamplingParams, StoppingParams,
    Timings, TokenChunk, TokenEvent, WorkerErrorCode, WorkerState, token_event,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::modelinfo::StaticInfo;

/// Proto: `repetition_window == 0` means "worker default (64)".
const DEFAULT_REPETITION_WINDOW: usize = 64;

/// SPEC §10 `[defaults]` ssd_cache_max_gb default.
const DEFAULT_SSD_MAX_BYTES: u64 = 64 << 30;

/// Prefix-cache/SSD settings from the CLI (SPEC §10 config flags).
#[derive(Debug, Clone)]
pub struct CacheOptions {
    pub prefix_cache: bool,
    /// Root cache directory (`$KILN_CACHE_DIR`); the worker stores slabs
    /// under `<root>/<model_fingerprint>/blocks/` (SPEC §6.4).
    pub ssd_dir: Option<PathBuf>,
    pub ssd_max_bytes: u64,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            prefix_cache: true,
            ssd_dir: None,
            ssd_max_bytes: DEFAULT_SSD_MAX_BYTES,
        }
    }
}

/// One queued generation request.
pub struct Submission {
    pub request_id: String,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub stopping: StoppingParams,
    pub priority: Priority,
    pub enqueued_at: Instant,
    pub handle: Arc<RequestHandle>,
    pub events: UnboundedSender<TokenEvent>,
}

/// Cancel/finish flags shared between the gRPC side and the engine thread.
/// `cancelled` is an `Arc` so it can be handed to the engine loop as the
/// request's cancel flag directly.
#[derive(Debug, Default)]
pub struct RequestHandle {
    pub cancelled: Arc<AtomicBool>,
    pub finished: AtomicBool,
}

/// Drain posture (proto `Drain`), monotonic: `NONE → GRACEFUL → IMMEDIATE`
/// (`fetch_max`; there is no un-drain — the gateway SIGTERMs next).
pub const DRAIN_NONE: u8 = 0;
pub const DRAIN_GRACEFUL: u8 = 1;
pub const DRAIN_IMMEDIATE: u8 = 2;

/// State shared between the engine thread and the gRPC services.
pub struct Shared {
    pub model_id: String,
    pub info: StaticInfo,
    state: std::sync::Mutex<(WorkerState, String)>,
    pub registry: std::sync::Mutex<HashMap<String, Arc<RequestHandle>>>,
    /// One of the `DRAIN_*` levels. Kept separate from `state` so a Drain
    /// received while the model is still loading survives the engine
    /// thread's `Loading → Ready` transition.
    pub drain: AtomicU8,
    /// Submissions accepted by gRPC but not yet drained into the engine.
    pub waiting: AtomicU32,
    /// Requests in the engine's WAITING queue (incl. preempted).
    pub engine_waiting: AtomicU32,
    pub running: AtomicU32,
    pub requests_total: AtomicU64,
    pub requests_failed: AtomicU64,
    pub requests_cancelled: AtomicU64,
    pub requests_preempted: AtomicU64,
    pub tokens_prefilled_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub kv_pool_allocated_bytes: AtomicU64,
    pub kv_pool_used_bytes: AtomicU64,
    pub kv_blocks_allocated: AtomicU64,
    pub kv_blocks_free: AtomicU64,
    pub engine_steps_total: AtomicU64,
    pub prefix_tokens_reused_total: AtomicU64,
    pub ssd_blocks_stored: AtomicU64,
    pub ssd_cache_bytes: AtomicU64,
    pub ssd_reads_total: AtomicU64,
    pub ssd_writes_total: AtomicU64,
    pub ssd_fingerprint_rejects_total: AtomicU64,
    /// Capability enum values advertised in `GetInfo` (set at startup
    /// from the cache flags; the engine thread may clear SSD_TIER if the
    /// store fails to open).
    pub capabilities: std::sync::Mutex<Vec<i32>>,
    started_at: Instant,
}

impl Shared {
    pub fn new(model_id: String, info: StaticInfo, cache: &CacheOptions) -> Self {
        use kiln_proto::v1::Capability;
        let mut capabilities = Vec::new();
        if cache.prefix_cache {
            capabilities.push(Capability::PrefixCache as i32);
            if cache.ssd_dir.is_some() {
                capabilities.push(Capability::SsdTier as i32);
            }
        }
        Self {
            model_id,
            info,
            state: std::sync::Mutex::new((WorkerState::Loading, String::new())),
            registry: std::sync::Mutex::new(HashMap::new()),
            drain: AtomicU8::new(DRAIN_NONE),
            waiting: AtomicU32::new(0),
            engine_waiting: AtomicU32::new(0),
            running: AtomicU32::new(0),
            requests_total: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_cancelled: AtomicU64::new(0),
            requests_preempted: AtomicU64::new(0),
            tokens_prefilled_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            kv_pool_allocated_bytes: AtomicU64::new(0),
            kv_pool_used_bytes: AtomicU64::new(0),
            kv_blocks_allocated: AtomicU64::new(0),
            kv_blocks_free: AtomicU64::new(0),
            engine_steps_total: AtomicU64::new(0),
            prefix_tokens_reused_total: AtomicU64::new(0),
            ssd_blocks_stored: AtomicU64::new(0),
            ssd_cache_bytes: AtomicU64::new(0),
            ssd_reads_total: AtomicU64::new(0),
            ssd_writes_total: AtomicU64::new(0),
            ssd_fingerprint_rejects_total: AtomicU64::new(0),
            capabilities: std::sync::Mutex::new(capabilities),
            started_at: Instant::now(),
        }
    }

    /// Advertised capability values (proto enum ints).
    pub fn capabilities(&self) -> Vec<i32> {
        match self.capabilities.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Drops SSD_TIER from the advertised capabilities (store failed to
    /// open; the engine degraded per the SPEC §6.4 silent-skip policy).
    fn clear_ssd_capability(&self) {
        use kiln_proto::v1::Capability;
        let mut guard = match self.capabilities.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.retain(|&capability| capability != Capability::SsdTier as i32);
    }

    pub fn state(&self) -> (WorkerState, String) {
        match self.state.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn set_state(&self, state: WorkerState, detail: impl Into<String>) {
        match self.state.lock() {
            Ok(mut guard) => *guard = (state, detail.into()),
            Err(poisoned) => *poisoned.into_inner() = (state, detail.into()),
        }
    }

    pub fn uptime_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    pub fn memory_report(&self) -> MemoryReport {
        // The mlx memory getters are allocator stat reads — safe off the
        // engine thread (no Array/Stream involved).
        MemoryReport {
            weights_bytes: self.info.weights_bytes,
            kv_pool_allocated_bytes: self.kv_pool_allocated_bytes.load(Ordering::Acquire),
            kv_pool_used_bytes: self.kv_pool_used_bytes.load(Ordering::Acquire),
            mlx_active_bytes: kiln_mlx::memory::active_memory().unwrap_or(0) as u64,
            mlx_cache_bytes: kiln_mlx::memory::cache_memory().unwrap_or(0) as u64,
            mlx_peak_bytes: kiln_mlx::memory::peak_memory().unwrap_or(0) as u64,
            process_rss_bytes: kiln_mlx::os::process_rss_bytes(),
            ssd_cache_bytes: self.ssd_cache_bytes.load(Ordering::Acquire),
        }
    }

    /// Marks a request finished and forgets it.
    pub(crate) fn retire(&self, request_id: &str) {
        let handle = match self.registry.lock() {
            Ok(mut guard) => guard.remove(request_id),
            Err(poisoned) => poisoned.into_inner().remove(request_id),
        };
        if let Some(handle) = handle {
            handle.finished.store(true, Ordering::Release);
        }
    }

    /// Flags every live request cancelled (Drain IMMEDIATE / deadline
    /// escalation). The engine honors the flags within its ≤2-step
    /// `Cancel` budget.
    pub(crate) fn cancel_all(&self) {
        let registry = match self.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for handle in registry.values() {
            handle.cancelled.store(true, Ordering::Release);
        }
    }

    /// Requests accepted but not yet finished (the registry only holds
    /// live entries — `retire` removes finished ones).
    pub(crate) fn live_requests(&self) -> u32 {
        let registry = match self.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        registry.len() as u32
    }
}

/// Runs on the dedicated engine thread. Owns the model + MLX stream +
/// batching loop.
pub fn engine_main(
    model_dir: PathBuf,
    shared: Arc<Shared>,
    rx: Receiver<Submission>,
    cache: CacheOptions,
) {
    kiln_mlx::init(); // swap out mlx-c's exit()-ing error handler first
    let stream = Stream::gpu();

    let load_started = Instant::now();
    let (model, eos_ids) = match kiln_models::AnyModel::load(&model_dir, &stream) {
        Ok(model) => {
            let eos: HashSet<u32> = model.eos_token_ids().into_iter().collect();
            (model, eos)
        }
        Err(err) => {
            // Report UNHEALTHY via Health and keep the process alive; the
            // gateway supervisor recycles it (SPEC §2.2). Exiting here would
            // race the supervisor's first health poll.
            tracing::error!(error = %err, "model load failed");
            shared.set_state(WorkerState::Unhealthy, format!("model load failed: {err}"));
            return;
        }
    };
    let dims = model.kv_dims();
    // Pool sizing: EngineConfig defaults (512 blocks x 32 tokens) until
    // memory-budget admission lands (SPEC §6.4 / §2.3, Phase 4 part 3+).
    // Prefix cache/SSD per the CLI flags (SPEC §6.3/§6.4): slabs live
    // under `<ssd_dir>/<model_fingerprint>/blocks/`.
    let config = EngineConfig {
        prefix_cache: cache.prefix_cache,
        ssd: cache
            .ssd_dir
            .as_ref()
            .filter(|_| cache.prefix_cache)
            .map(|root| SsdParams {
                dir: root.join(&shared.info.weights_fingerprint).join("blocks"),
                max_bytes: cache.ssd_max_bytes,
                fingerprint: shared.info.weights_fingerprint.clone(),
            }),
        ..EngineConfig::default()
    };
    let mut engine = match Engine::new(model, dims, config, stream) {
        Ok(engine) => engine,
        Err(err) => {
            tracing::error!(error = %err, "engine construction failed");
            shared.set_state(WorkerState::Unhealthy, format!("engine failed: {err}"));
            return;
        }
    };
    if let Some(reason) = engine.ssd_error() {
        // SPEC §6.4: the tier degrades silently for requests; say it once.
        tracing::warn!(model = %shared.model_id, reason, "ssd tier disabled");
        shared.clear_ssd_capability();
    }
    tracing::info!(
        model = %shared.model_id,
        load_ms = load_started.elapsed().as_millis() as u64,
        "model ready"
    );
    shared.set_state(WorkerState::Ready, "");

    'serve: loop {
        // Block for work when idle; otherwise drain whatever queued while
        // the last step ran, so new requests join the next iteration. An
        // idle engine with pending SSD flushes keeps ticking (bounded
        // waits) so persistence never depends on new traffic.
        if engine.is_idle() {
            if engine.has_pending_cache_io() {
                match rx.recv_timeout(std::time::Duration::from_millis(2)) {
                    Ok(submission) => submit(&mut engine, &shared, &eos_ids, submission),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        engine.flush_prefix_cache();
                        break 'serve;
                    }
                }
            } else {
                match rx.recv() {
                    Ok(submission) => submit(&mut engine, &shared, &eos_ids, submission),
                    Err(_) => break 'serve, // gRPC side is gone
                }
            }
        }
        loop {
            match rx.try_recv() {
                Ok(submission) => submit(&mut engine, &shared, &eos_ids, submission),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if engine.is_idle() && !engine.has_pending_cache_io() {
                        break 'serve;
                    }
                    break;
                }
            }
        }
        if let Err(err) = engine.step() {
            // Affected requests were already failed with in-band errors and
            // the engine reset itself; keep serving.
            tracing::error!(error = %err, "engine step failed");
        }
        publish_stats(&engine, &shared);
    }
}

/// Copies the engine's gauges/counters into `Shared` for Health and Stats
/// (the gRPC side never touches the engine directly).
fn publish_stats(engine: &Engine<kiln_models::AnyModel>, shared: &Shared) {
    shared
        .running
        .store(engine.num_running() as u32, Ordering::Release);
    shared
        .engine_waiting
        .store(engine.num_waiting() as u32, Ordering::Release);
    shared
        .requests_preempted
        .store(engine.preemptions(), Ordering::Release);
    shared
        .kv_pool_allocated_bytes
        .store(engine.kv_allocated_bytes(), Ordering::Release);
    shared
        .kv_pool_used_bytes
        .store(engine.kv_used_bytes(), Ordering::Release);
    let (blocks_used, blocks_free) = engine.kv_blocks();
    shared
        .kv_blocks_allocated
        .store(blocks_used, Ordering::Release);
    shared.kv_blocks_free.store(blocks_free, Ordering::Release);
    shared
        .engine_steps_total
        .store(engine.steps(), Ordering::Release);
    let cache = engine.cache_stats();
    shared
        .prefix_tokens_reused_total
        .store(cache.tokens_reused_total, Ordering::Release);
    shared
        .ssd_blocks_stored
        .store(cache.ssd_blocks_stored, Ordering::Release);
    shared
        .ssd_cache_bytes
        .store(cache.ssd_bytes_stored, Ordering::Release);
    shared
        .ssd_reads_total
        .store(cache.ssd_reads_total, Ordering::Release);
    shared
        .ssd_writes_total
        .store(cache.ssd_writes_total, Ordering::Release);
    shared
        .ssd_fingerprint_rejects_total
        .store(cache.ssd_fingerprint_rejects_total, Ordering::Release);
}

/// Maps a proto submission onto an engine request whose event sink speaks
/// `TokenEvent`.
fn submit(
    engine: &mut Engine<kiln_models::AnyModel>,
    shared: &Arc<Shared>,
    eos_ids: &HashSet<u32>,
    submission: Submission,
) {
    let Submission {
        request_id,
        prompt_ids,
        sampling,
        stopping,
        priority,
        enqueued_at,
        handle,
        events,
    } = submission;
    shared.waiting.fetch_sub(1, Ordering::AcqRel);
    shared.requests_total.fetch_add(1, Ordering::Relaxed);

    let seed_used = if sampling.seed != 0 {
        sampling.seed
    } else {
        random_seed()
    };
    let queued_ms = enqueued_at.elapsed().as_millis() as u64;
    let prompt_tokens = prompt_ids.len() as u32;

    let sampling_options = SamplingOptions {
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        seed: seed_used,
    };
    let penalties = PenaltyOptions {
        // Proto: both 0.0 and 1.0 mean "disabled" for the multiplicative one.
        repetition_penalty: if sampling.repetition_penalty == 0.0 {
            1.0
        } else {
            sampling.repetition_penalty
        },
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
    };
    let penalty_window = if sampling.repetition_window == 0 {
        DEFAULT_REPETITION_WINDOW
    } else {
        sampling.repetition_window as usize
    };

    let mut stop_tokens: HashSet<u32> = stopping.stop_token_ids.iter().copied().collect();
    if !stopping.ignore_eos {
        stop_tokens.extend(eos_ids);
    }

    let shared = Arc::clone(shared);
    let on_event = Box::new(move |event: SeqEvent| -> bool {
        match event {
            SeqEvent::Token(token) => events
                .send(TokenEvent {
                    event: Some(token_event::Event::Tokens(TokenChunk {
                        token_ids: vec![token],
                        text: String::new(), // gateway owns detokenization
                        chosen_logprobs: Vec::new(),
                        top_logprobs: Vec::new(),
                    })),
                })
                .is_ok(),
            SeqEvent::PrefixHit { tokens, from_ssd } => events
                .send(TokenEvent {
                    event: Some(token_event::Event::Cache(PrefixCacheHit {
                        tokens_reused: tokens,
                        from_ssd,
                    })),
                })
                .is_ok(),
            SeqEvent::Finished(summary) => {
                let mut finished = Finished {
                    prompt_tokens,
                    completion_tokens: summary.completion_tokens,
                    cached_prompt_tokens: summary.cached_prompt_tokens,
                    seed_used,
                    ..Finished::default()
                };
                match summary.reason {
                    FinishKind::Stop => {
                        finished.set_finish_reason(FinishReason::Stop);
                        if let Some(token) = summary.matched_stop_token {
                            finished.matched_stop = format!("token_id:{token}");
                        }
                    }
                    FinishKind::Length => finished.set_finish_reason(FinishReason::Length),
                    FinishKind::Cancelled => {
                        finished.set_finish_reason(FinishReason::Cancelled);
                        shared.requests_cancelled.fetch_add(1, Ordering::Relaxed);
                    }
                    FinishKind::Error => {
                        // Malformed input or an engine fault must never kill
                        // the worker (CLAUDE.md): errors flow in-band. Detail
                        // stays free of prompt content (shape/op messages).
                        let detail = summary.error.clone().unwrap_or_default();
                        tracing::error!(request_id = %request_id, error = %detail,
                            "generation failed");
                        finished.set_finish_reason(FinishReason::Error);
                        finished.set_error_code(match summary.error_cause {
                            // The request can never fit the KV pool: proto's
                            // admission-refusal code (SPEC §6.4).
                            Some(ErrorCause::Capacity) => WorkerErrorCode::WorkerErrorOomRejected,
                            _ => WorkerErrorCode::WorkerErrorInternal,
                        });
                        finished.error_detail = detail;
                        shared.requests_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                if summary.reason != FinishKind::Error {
                    shared
                        .tokens_prefilled_total
                        .fetch_add(u64::from(prompt_tokens), Ordering::Relaxed);
                    shared
                        .tokens_generated_total
                        .fetch_add(u64::from(summary.completion_tokens), Ordering::Relaxed);
                }
                let mut timings = Timings {
                    queued_ms,
                    prefill_ms: (summary.prefill_seconds * 1000.0) as u64,
                    decode_ms: (summary.decode_seconds * 1000.0) as u64,
                    ..Timings::default()
                };
                if summary.prefill_seconds > 0.0 {
                    timings.prefill_tokens_per_sec =
                        (f64::from(prompt_tokens) / summary.prefill_seconds) as f32;
                }
                if summary.decode_seconds > 0.0 && summary.completion_tokens > 1 {
                    timings.decode_tokens_per_sec =
                        (f64::from(summary.completion_tokens - 1) / summary.decode_seconds) as f32;
                }
                finished.timings = Some(timings);
                shared.retire(&request_id);
                let _ = events.send(TokenEvent {
                    event: Some(token_event::Event::Finished(finished)),
                });
                true
            }
        }
    });

    engine.submit(EngineRequest {
        prompt: prompt_ids,
        max_tokens: stopping.max_tokens as usize,
        sampling: sampling_options,
        penalties,
        penalty_window,
        stop_tokens,
        // Proto: BATCH is preempted first; UNSPECIFIED means INTERACTIVE.
        priority: match priority {
            Priority::Batch => kiln_engine::Priority::Batch,
            _ => kiln_engine::Priority::Interactive,
        },
        cancel: Arc::clone(&handle.cancelled),
        on_event,
    });
}

/// Non-zero random seed when the client leaves seed unset (echoed back in
/// `Finished.seed_used`).
fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let seed = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    if seed == 0 { 1 } else { seed }
}

/// Creates the engine thread; returns the submission sender.
pub fn spawn(
    model_dir: PathBuf,
    shared: Arc<Shared>,
    cache: CacheOptions,
) -> std::io::Result<Sender<Submission>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("kiln-engine".to_owned())
        .spawn(move || engine_main(model_dir, shared, rx, cache))?;
    Ok(tx)
}
