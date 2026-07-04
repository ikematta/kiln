//! Single-request engine (Phase 3): one dedicated OS thread owns every MLX
//! value — the model, the `Stream`, and all intermediate arrays (they are
//! `!Send` by construction). gRPC handler tasks only enqueue submissions and
//! read event channels; the continuous-batching engine replaces this loop in
//! Phase 4 (SPEC §6.2).
//!
//! Cancellation: `generate_with`'s per-token callback checks the cancel flag
//! between engine steps; on the pipelined path one extra step is already
//! scheduled — within the ≤2-step budget `Cancel` promises (SPEC §5).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use kiln_engine::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
use kiln_mlx::{Array, MlxError, Stream};
use kiln_proto::v1::{
    FinishReason, Finished, MemoryReport, SamplingParams, StoppingParams, Timings, TokenChunk,
    TokenEvent, WorkerErrorCode, WorkerState, token_event,
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
    pub enqueued_at: Instant,
    pub handle: Arc<RequestHandle>,
    pub events: UnboundedSender<TokenEvent>,
}

/// Cancel/finish flags shared between the gRPC side and the engine thread.
#[derive(Debug, Default)]
pub struct RequestHandle {
    pub cancelled: AtomicBool,
    pub finished: AtomicBool,
}

/// State shared between the engine thread and the gRPC services.
pub struct Shared {
    pub model_id: String,
    pub info: StaticInfo,
    state: std::sync::Mutex<(WorkerState, String)>,
    pub registry: std::sync::Mutex<HashMap<String, Arc<RequestHandle>>>,
    pub waiting: AtomicU32,
    pub running: AtomicU32,
    pub requests_total: AtomicU64,
    pub requests_failed: AtomicU64,
    pub requests_cancelled: AtomicU64,
    pub tokens_prefilled_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    started_at: Instant,
}

impl Shared {
    pub fn new(model_id: String, info: StaticInfo) -> Self {
        Self {
            model_id,
            info,
            state: std::sync::Mutex::new((WorkerState::Loading, String::new())),
            registry: std::sync::Mutex::new(HashMap::new()),
            waiting: AtomicU32::new(0),
            running: AtomicU32::new(0),
            requests_total: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_cancelled: AtomicU64::new(0),
            tokens_prefilled_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
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
            kv_pool_allocated_bytes: 0, // no paged pool until Phase 4
            kv_pool_used_bytes: 0,
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
}

/// Runs on the dedicated engine thread. Owns the model + MLX stream.
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
    tracing::info!(
        model = %shared.model_id,
        load_ms = load_started.elapsed().as_millis() as u64,
        "model ready"
    );
    shared.set_state(WorkerState::Ready, "");

    while let Ok(submission) = rx.recv() {
        shared.waiting.fetch_sub(1, Ordering::AcqRel);
        shared.running.store(1, Ordering::Release);
        shared.requests_total.fetch_add(1, Ordering::Relaxed);
        run_request(&model, &eos_ids, &shared, submission, &stream);
        shared.running.store(0, Ordering::Release);
    }
}

fn run_request(
    model: &kiln_models::LlamaModel,
    eos_ids: &HashSet<u32>,
    shared: &Shared,
    submission: Submission,
    stream: &Stream,
) {
    let Submission {
        request_id,
        prompt_ids,
        sampling,
        stopping,
        enqueued_at,
        handle,
        events,
    } = submission;
    let started_at = Instant::now();
    let prompt_tokens = prompt_ids.len() as u32;

    let seed_used = if sampling.seed != 0 {
        sampling.seed
    } else {
        random_seed()
    };
    let mut finished = Finished {
        prompt_tokens,
        seed_used,
        timings: Some(Timings {
            queued_ms: started_at.duration_since(enqueued_at).as_millis() as u64,
            ..Timings::default()
        }),
        ..Finished::default()
    };

    if handle.cancelled.load(Ordering::Acquire) {
        finished.set_finish_reason(FinishReason::Cancelled);
        shared.requests_cancelled.fetch_add(1, Ordering::Relaxed);
        shared.retire(&request_id);
        let _ = events.send(TokenEvent {
            event: Some(token_event::Event::Finished(finished)),
        });
        return;
    }

    let mut sampler = Sampler::new(SamplingOptions {
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        seed: seed_used,
    });
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
    let window = if sampling.repetition_window == 0 {
        DEFAULT_REPETITION_WINDOW
    } else {
        sampling.repetition_window as usize
    };
    let processor = (!penalties.is_disabled()).then_some(
        move |history: &[u32], logits: &Array, s: &Stream| -> Result<Array, MlxError> {
            let recent = &history[history.len().saturating_sub(window)..];
            apply_penalties(logits, recent, penalties, s)
        },
    );

    let mut stop_ids: HashSet<u32> = stopping.stop_token_ids.iter().copied().collect();
    if !stopping.ignore_eos {
        stop_ids.extend(eos_ids);
    }

    let mut finish_reason = FinishReason::Length;
    let mut matched_stop = String::new();
    let result = kiln_models::generate_with(
        model,
        &prompt_ids,
        stopping.max_tokens as usize,
        processor,
        |logprobs, s| sampler.sample(logprobs, s),
        |token| {
            if handle.cancelled.load(Ordering::Acquire) {
                finish_reason = FinishReason::Cancelled;
                return false;
            }
            if stop_ids.contains(&token) {
                // Counted in usage but NOT chunked: the gateway detokenizes
                // chunks verbatim, and stop text is excluded by contract.
                finish_reason = FinishReason::Stop;
                matched_stop = format!("token_id:{token}");
                return false;
            }
            // A dead receiver means the client is gone; stop generating.
            let sent = events.send(TokenEvent {
                event: Some(token_event::Event::Tokens(TokenChunk {
                    token_ids: vec![token],
                    text: String::new(), // gateway owns detokenization
                    chosen_logprobs: Vec::new(),
                    top_logprobs: Vec::new(),
                })),
            });
            if sent.is_err() {
                finish_reason = FinishReason::Cancelled;
                return false;
            }
            true
        },
        stream,
    );

    match result {
        Ok(output) => {
            let completion = output.tokens.len() as u32;
            shared
                .tokens_prefilled_total
                .fetch_add(u64::from(prompt_tokens), Ordering::Relaxed);
            shared
                .tokens_generated_total
                .fetch_add(u64::from(completion), Ordering::Relaxed);
            if finish_reason == FinishReason::Cancelled {
                shared.requests_cancelled.fetch_add(1, Ordering::Relaxed);
            }
            finished.set_finish_reason(finish_reason);
            finished.completion_tokens = completion;
            finished.matched_stop = matched_stop;
            if let Some(timings) = finished.timings.as_mut() {
                timings.prefill_ms = (output.prefill_seconds * 1000.0) as u64;
                timings.decode_ms = (output.decode_seconds * 1000.0) as u64;
                if output.prefill_seconds > 0.0 {
                    timings.prefill_tokens_per_sec =
                        (f64::from(prompt_tokens) / output.prefill_seconds) as f32;
                }
                if output.decode_seconds > 0.0 && completion > 1 {
                    timings.decode_tokens_per_sec =
                        (f64::from(completion - 1) / output.decode_seconds) as f32;
                }
            }
        }
        Err(err) => {
            // Malformed input must never kill the worker (CLAUDE.md): the
            // installed error handler already turned the MLX fault into an
            // Err. Full detail goes to worker logs; error_detail stays free
            // of prompt content (shape/op messages only).
            tracing::error!(request_id = %request_id, error = %err, "generation failed");
            shared.requests_failed.fetch_add(1, Ordering::Relaxed);
            finished.set_finish_reason(FinishReason::Error);
            finished.set_error_code(WorkerErrorCode::WorkerErrorInternal);
            finished.error_detail = err.to_string();
        }
    }

    shared.retire(&request_id);
    let _ = events.send(TokenEvent {
        event: Some(token_event::Event::Finished(finished)),
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
