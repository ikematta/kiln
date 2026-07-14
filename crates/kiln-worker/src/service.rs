//! The tonic `Worker` service (frozen `worker.proto` semantics, SPEC §5).
//!
//! Validation failures never abort the RPC or crash the worker: malformed
//! input yields a single `Finished{finish_reason=ERROR}` event (matching the
//! Python worker). `Drain` is flag-based: GRACEFUL stops admitting and lets
//! in-flight requests finish (an optional deadline escalates to
//! cancellation); IMMEDIATE additionally cancels everything live through
//! the same flags `Cancel` uses, so the engine loop needs no drain-specific
//! path. `Stats` arrives with Phase 4 part 4; `Tokenize` is UNIMPLEMENTED
//! by design — the gateway owns tokenization for Rust workers
//! (kiln-tokenize BOS contract).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use kiln_proto::v1::worker_server::Worker;
use kiln_proto::v1::{
    CancelAck, CancelRequest, DrainAck, DrainMode, DrainRequest, FinishReason, Finished,
    HealthRequest, HealthStatus, InfoRequest, RequestAdmitted, StatsRequest, SubmitRequest,
    TokenEvent, TokenizeRequest, TokenizeResponse, WorkerErrorCode, WorkerInfo, WorkerState,
    WorkerStats, submit_request, token_event,
};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tonic::{Request, Response, Status};

use crate::engine::{
    DRAIN_GRACEFUL, DRAIN_IMMEDIATE, DRAIN_NONE, RequestHandle, Shared, Submission,
};

pub struct WorkerService {
    shared: Arc<Shared>,
    submissions: Sender<Submission>,
}

impl WorkerService {
    pub fn new(shared: Arc<Shared>, submissions: Sender<Submission>) -> Self {
        Self {
            shared,
            submissions,
        }
    }
}

/// Adapts the engine's plain event channel to the `Result` stream tonic
/// wants. Dropping it (client disconnect) drops the receiver, which the
/// engine notices on its next send and treats as cancellation.
pub struct EventStream {
    rx: UnboundedReceiver<TokenEvent>,
}

impl tonic::codegen::tokio_stream::Stream for EventStream {
    type Item = Result<TokenEvent, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx).map(|event| event.map(Ok))
    }
}

/// A stream carrying exactly one `Finished{ERROR}` event (proto: malformed
/// input is an in-band error, not an RPC failure).
fn error_stream(code: WorkerErrorCode, detail: impl Into<String>) -> EventStream {
    let (tx, rx) = unbounded_channel();
    let mut finished = Finished {
        error_detail: detail.into(),
        ..Finished::default()
    };
    finished.set_finish_reason(FinishReason::Error);
    finished.set_error_code(code);
    let _ = tx.send(TokenEvent {
        event: Some(token_event::Event::Finished(finished)),
    });
    EventStream { rx }
}

#[tonic::async_trait]
impl Worker for WorkerService {
    async fn get_info(&self, _: Request<InfoRequest>) -> Result<Response<WorkerInfo>, Status> {
        let info = &self.shared.info;
        Ok(Response::new(WorkerInfo {
            model_id: self.shared.model_id.clone(),
            model_path: info.model_path.clone(),
            architecture: info.architecture.clone(),
            weights_fingerprint: info.weights_fingerprint.clone(),
            max_context_len: info.max_context_len,
            vocab_size: info.vocab_size,
            dtype: info.dtype.clone(),
            // PREFIX_CACHE/SSD_TIER per the startup flags (Phase 5);
            // GRAMMAR once the engine thread builds the llguidance
            // environment (Phase 7); SPECULATIVE once a compatible draft
            // is attached (Phase 8). No LOGPROBS yet; notably NOT
            // TOKENIZER_OWNED — the gateway tokenizes for this worker.
            capabilities: self.shared.capabilities(),
            worker_kind: "rust".to_owned(),
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            kv_block_size: kiln_engine::DEFAULT_BLOCK_SIZE as u32,
            chat_template_hash: info.chat_template_hash.clone(),
            // Diagnostics only (ADR 0002 B'): the engine enforces
            // determinism itself by sub-batching greedy/seeded rows.
            max_deterministic_decode_width: self
                .shared
                .deterministic_decode_width
                .load(Ordering::Acquire),
        }))
    }

    async fn health(&self, _: Request<HealthRequest>) -> Result<Response<HealthStatus>, Status> {
        let (mut state, detail) = self.shared.state();
        // Draining is a posture over Ready, not a loading/fault state.
        if state == WorkerState::Ready && self.shared.drain.load(Ordering::Acquire) != DRAIN_NONE {
            state = WorkerState::Draining;
        }
        let waiting = self.shared.waiting.load(Ordering::Acquire)
            + self.shared.engine_waiting.load(Ordering::Acquire);
        Ok(Response::new(HealthStatus {
            state: state as i32,
            memory: Some(self.shared.memory_report()),
            requests_waiting: waiting,
            requests_running: self.shared.running.load(Ordering::Acquire),
            uptime_ms: self.shared.uptime_ms(),
            detail,
        }))
    }

    type SubmitStream = EventStream;

    async fn submit(
        &self,
        request: Request<SubmitRequest>,
    ) -> Result<Response<Self::SubmitStream>, Status> {
        let mut request = request.into_inner();
        // Unknown enum values decode as UNSPECIFIED (treated INTERACTIVE);
        // read before fields are moved out of the message below.
        let priority = request.priority();
        // Draining rejects new work in-band (proto WORKER_ERROR_DRAINING),
        // so the gateway gets a structured, retriable-elsewhere error.
        if self.shared.drain.load(Ordering::Acquire) != DRAIN_NONE {
            return Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorDraining,
                "worker is draining; not admitting new requests",
            )));
        }
        let (state, _) = self.shared.state();
        if state != WorkerState::Ready {
            return Err(Status::unavailable(format!(
                "worker not ready (state={state:?})"
            )));
        }

        let invalid = |detail: &str| {
            Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorInvalidRequest,
                detail,
            )))
        };
        if request.request_id.is_empty() {
            return invalid("missing request_id");
        }
        if request.echo_prompt {
            return invalid("echo_prompt is not supported by the rust worker");
        }
        let stopping = request.stopping.unwrap_or_default();
        if stopping.max_tokens == 0 {
            return invalid("max_tokens must be >= 1");
        }
        if !stopping.stop_strings.is_empty() {
            // Stop strings need detokenized text, which lives gateway-side
            // for rust workers. Rejecting (vs silently ignoring) keeps a
            // misconfigured caller loud.
            return invalid("stop_strings are matched by the gateway for rust workers");
        }
        let prompt_ids = match request.input {
            Some(submit_request::Input::TokenIds(ids)) => ids.ids,
            Some(submit_request::Input::RawText(_)) => {
                return invalid(
                    "rust worker does not own a tokenizer (no CAPABILITY_TOKENIZER_OWNED); \
                     send token_ids",
                );
            }
            None => return invalid("missing input (token_ids)"),
        };
        if prompt_ids.is_empty() {
            return invalid("empty prompt");
        }
        let max_ctx = self.shared.info.max_context_len;
        if max_ctx > 0
            && prompt_ids.len() as u64 + u64::from(stopping.max_tokens) > u64::from(max_ctx)
        {
            return Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorCtxOverflow,
                format!(
                    "prompt ({}) + max_tokens ({}) exceeds context ({max_ctx})",
                    prompt_ids.len(),
                    stopping.max_tokens
                ),
            )));
        }
        // Architecture parity envelope (Phase 6 Task 2, see modelinfo.rs):
        // prompts past this bound have no reference-shaped prefill.
        let max_prompt = self.shared.info.max_prompt_len;
        if max_prompt > 0 && prompt_ids.len() as u64 > u64::from(max_prompt) {
            return Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorCtxOverflow,
                format!(
                    "prompt ({}) exceeds this architecture's prefill bound ({max_prompt})",
                    prompt_ids.len(),
                ),
            )));
        }

        // Structured output (SPEC §12 Phase 7): compile the GrammarSpec
        // here on the handler task, so compile faults reject in-band
        // (proto GRAMMAR_UNSUPPORTED / GRAMMAR_COMPILE) and the engine
        // thread only ever sees ready-to-mask grammars.
        let grammar = match request.grammar.take() {
            None => None,
            Some(spec) => {
                let Some(env) = self.shared.grammar_env() else {
                    return Ok(Response::new(error_stream(
                        WorkerErrorCode::WorkerErrorGrammarUnsupported,
                        "this worker cannot serve grammars (no llguidance environment; \
                         CAPABILITY_GRAMMAR was not advertised)",
                    )));
                };
                use kiln_proto::v1::grammar_spec::Grammar as Spec;
                let compiled = match spec.grammar {
                    Some(Spec::JsonSchema(schema)) => env.compile_json_schema(&schema),
                    Some(Spec::Regex(regex)) => env.compile_regex(&regex),
                    Some(Spec::Lark(_)) => {
                        return Ok(Response::new(error_stream(
                            WorkerErrorCode::WorkerErrorGrammarUnsupported,
                            "lark grammars are not supported (json_schema and regex only)",
                        )));
                    }
                    None => return invalid("GrammarSpec without a grammar variant"),
                };
                match compiled {
                    Ok(grammar) => Some(grammar),
                    Err(err) => {
                        return Ok(Response::new(error_stream(
                            WorkerErrorCode::WorkerErrorGrammarCompile,
                            err.to_string(),
                        )));
                    }
                }
            }
        };

        // Register (duplicate ids rejected) and enqueue.
        let handle = Arc::new(RequestHandle::default());
        {
            let mut registry = match self.shared.registry.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if registry.contains_key(&request.request_id) {
                return invalid(&format!(
                    "request_id already active: {}",
                    request.request_id
                ));
            }
            registry.insert(request.request_id.clone(), Arc::clone(&handle));
        }
        // Close the race with a concurrent Drain: either its cancel sweep
        // saw our registry entry, or this re-check sees the drain flag.
        if self.shared.drain.load(Ordering::Acquire) != DRAIN_NONE {
            self.shared.retire(&request.request_id);
            return Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorDraining,
                "worker is draining; not admitting new requests",
            )));
        }

        let (events_tx, events_rx) = unbounded_channel();
        let queue_position = self.shared.waiting.load(Ordering::Acquire)
            + self.shared.running.load(Ordering::Acquire);
        let admitted = TokenEvent {
            event: Some(token_event::Event::Admitted(RequestAdmitted {
                queue_position,
                prompt_tokens: prompt_ids.len() as u32,
                prefill_chunks_estimated: prompt_ids
                    .len()
                    .div_ceil(kiln_engine::DEFAULT_PREFILL_CHUNK)
                    as u32,
            })),
        };
        // Sent before the sender is handed to the engine, so Admitted is
        // always the first event on the stream.
        let _ = events_tx.send(admitted);

        self.shared.waiting.fetch_add(1, Ordering::AcqRel);
        let submission = Submission {
            request_id: request.request_id.clone(),
            prompt_ids,
            sampling: request.sampling.unwrap_or_default(),
            stopping,
            grammar,
            priority,
            enqueued_at: Instant::now(),
            handle,
            events: events_tx,
        };
        if self.submissions.send(submission).is_err() {
            // Engine thread is gone (load failed / fatal fault).
            self.shared.waiting.fetch_sub(1, Ordering::AcqRel);
            self.shared.retire(&request.request_id);
            return Err(Status::unavailable("engine thread is not running"));
        }

        Ok(Response::new(EventStream { rx: events_rx }))
    }

    async fn cancel(&self, request: Request<CancelRequest>) -> Result<Response<CancelAck>, Status> {
        let request_id = request.into_inner().request_id;
        let registry = match self.shared.registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let found = match registry.get(&request_id) {
            Some(handle) if !handle.finished.load(Ordering::Acquire) => {
                handle.cancelled.store(true, Ordering::Release);
                true
            }
            _ => false,
        };
        Ok(Response::new(CancelAck { found }))
    }

    async fn drain(&self, request: Request<DrainRequest>) -> Result<Response<DrainAck>, Status> {
        let request = request.into_inner();
        // Proto: UNSPECIFIED is treated as GRACEFUL. The posture only
        // escalates (fetch_max) — re-draining GRACEFUL after IMMEDIATE
        // must not resurrect admission.
        let (level, mode) = match request.mode() {
            DrainMode::Immediate => (DRAIN_IMMEDIATE, DrainMode::Immediate),
            _ => (DRAIN_GRACEFUL, DrainMode::Graceful),
        };
        self.shared.drain.fetch_max(level, Ordering::AcqRel);
        tracing::info!(mode = ?mode, deadline_ms = request.deadline_ms, "draining");
        if mode == DrainMode::Immediate {
            // Cancel in-flight work through the same flags Cancel uses;
            // the engine honors them within its ≤2-step budget.
            self.shared.cancel_all();
        } else if request.deadline_ms > 0 {
            // GRACEFUL with a deadline: give in-flight requests that long
            // to finish, then escalate to cancellation (the gateway
            // SIGTERMs after the drain deadline either way — cancelling
            // lets streams end with a clean CANCELLED instead of a 502).
            let shared = Arc::clone(&self.shared);
            let deadline = Duration::from_millis(request.deadline_ms);
            tokio::spawn(async move {
                tokio::time::sleep(deadline).await;
                if shared.live_requests() > 0 {
                    tracing::info!("graceful drain deadline passed; cancelling in-flight");
                    shared.cancel_all();
                }
            });
        }
        Ok(Response::new(DrainAck {
            requests_remaining: self.shared.live_requests(),
        }))
    }

    async fn stats(&self, _: Request<StatsRequest>) -> Result<Response<WorkerStats>, Status> {
        let shared = &self.shared;
        let load = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Acquire);
        Ok(Response::new(WorkerStats {
            requests_total: load(&shared.requests_total),
            requests_failed: load(&shared.requests_failed),
            requests_cancelled: load(&shared.requests_cancelled),
            requests_preempted: load(&shared.requests_preempted),
            tokens_prefilled_total: load(&shared.tokens_prefilled_total),
            tokens_generated_total: load(&shared.tokens_generated_total),
            prefix_tokens_reused_total: load(&shared.prefix_tokens_reused_total),
            kv_blocks_allocated: load(&shared.kv_blocks_allocated),
            kv_blocks_free: load(&shared.kv_blocks_free),
            ssd_blocks_stored: load(&shared.ssd_blocks_stored),
            ssd_reads_total: load(&shared.ssd_reads_total),
            ssd_writes_total: load(&shared.ssd_writes_total),
            ssd_fingerprint_rejects_total: load(&shared.ssd_fingerprint_rejects_total),
            spec_tokens_proposed_total: load(&shared.spec_tokens_proposed_total),
            spec_tokens_accepted_total: load(&shared.spec_tokens_accepted_total),
            engine_steps_total: load(&shared.engine_steps_total),
            // The step-overhead percentiles need an in-engine reservoir
            // (criterion covers the gate today) — zero until they exist.
            engine_step_overhead_us_p50: 0.0,
            engine_step_overhead_us_p99: 0.0,
        }))
    }

    async fn tokenize(
        &self,
        _: Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        Err(Status::unimplemented(
            "the gateway owns tokenization for rust workers (no CAPABILITY_TOKENIZER_OWNED)",
        ))
    }
}
