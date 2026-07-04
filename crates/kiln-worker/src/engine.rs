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
    SeqEvent,
};
use kiln_mlx::Stream;
use kiln_proto::v1::{
    FinishReason, Finished, MemoryReport, Priority, SamplingParams, StoppingParams, Timings,
    TokenChunk, TokenEvent, WorkerErrorCode, WorkerState, token_event,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::modelinfo::StaticInfo;

/// Proto: `repetition_window == 0` means "worker default (64)".
const DEFAULT_REPETITION_WINDOW: usize = 64;

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
    started_at: Instant,
}

impl Shared {
    pub fn new(model_id: String, info: StaticInfo) -> Self {
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
            started_at: Instant::now(),
        }
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
            ssd_cache_bytes: 0,
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
pub fn engine_main(model_dir: PathBuf, shared: Arc<Shared>, rx: Receiver<Submission>) {
    kiln_mlx::init(); // swap out mlx-c's exit()-ing error handler first
    let stream = Stream::gpu();

    let load_started = Instant::now();
    let (model, eos_ids) = match kiln_models::LlamaModel::load(&model_dir, &stream) {
        Ok(model) => {
            let eos: HashSet<u32> = model.config().eos_token_ids().into_iter().collect();
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
    let mut engine = match Engine::new(model, dims, EngineConfig::default(), stream) {
        Ok(engine) => engine,
        Err(err) => {
            tracing::error!(error = %err, "engine construction failed");
            shared.set_state(WorkerState::Unhealthy, format!("engine failed: {err}"));
            return;
        }
    };
    tracing::info!(
        model = %shared.model_id,
        load_ms = load_started.elapsed().as_millis() as u64,
        "model ready"
    );
    shared.set_state(WorkerState::Ready, "");

    'serve: loop {
        // Block for work when idle; otherwise drain whatever queued while
        // the last step ran, so new requests join the next iteration.
        if engine.is_idle() {
            match rx.recv() {
                Ok(submission) => submit(&mut engine, &shared, &eos_ids, submission),
                Err(_) => break 'serve, // gRPC side is gone
            }
        }
        loop {
            match rx.try_recv() {
                Ok(submission) => submit(&mut engine, &shared, &eos_ids, submission),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if engine.is_idle() {
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
    }
}

/// Maps a proto submission onto an engine request whose event sink speaks
/// `TokenEvent`.
fn submit(
    engine: &mut Engine<kiln_models::LlamaModel>,
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
            SeqEvent::Finished(summary) => {
                let mut finished = Finished {
                    prompt_tokens,
                    completion_tokens: summary.completion_tokens,
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
pub fn spawn(model_dir: PathBuf, shared: Arc<Shared>) -> std::io::Result<Sender<Submission>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("kiln-engine".to_owned())
        .spawn(move || engine_main(model_dir, shared, rx))?;
    Ok(tx)
}
