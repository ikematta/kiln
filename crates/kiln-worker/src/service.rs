//! The tonic `Worker` service (frozen `worker.proto` semantics, SPEC §5).
//!
//! Validation failures never abort the RPC or crash the worker: malformed
//! input yields a single `Finished{finish_reason=ERROR}` event (matching the
//! Python worker). `Drain`/`Stats` arrive with their phases; `Tokenize` is
//! UNIMPLEMENTED by design — the gateway owns tokenization for Rust workers
//! (kiln-tokenize BOS contract).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::task::{Context, Poll};
use std::time::Instant;

use kiln_proto::v1::worker_server::Worker;
use kiln_proto::v1::{
    CancelAck, CancelRequest, DrainAck, DrainRequest, FinishReason, Finished, HealthRequest,
    HealthStatus, InfoRequest, RequestAdmitted, StatsRequest, SubmitRequest, TokenEvent,
    TokenizeRequest, TokenizeResponse, WorkerErrorCode, WorkerInfo, WorkerState, WorkerStats,
    submit_request, token_event,
};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tonic::{Request, Response, Status};

use crate::engine::{RequestHandle, Shared, Submission};

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
            // No LOGPROBS/GRAMMAR/PREFIX_CACHE yet; notably NOT
            // TOKENIZER_OWNED — the gateway tokenizes for this worker.
            capabilities: Vec::new(),
            worker_kind: "rust".to_owned(),
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            kv_block_size: 0, // contiguous cache until Phase 4
            chat_template_hash: info.chat_template_hash.clone(),
        }))
    }

    async fn health(&self, _: Request<HealthRequest>) -> Result<Response<HealthStatus>, Status> {
        let (state, detail) = self.shared.state();
        Ok(Response::new(HealthStatus {
            state: state as i32,
            memory: Some(self.shared.memory_report()),
            requests_waiting: self.shared.waiting.load(Ordering::Acquire),
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
        let request = request.into_inner();
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
        if request.grammar.is_some() {
            return Ok(Response::new(error_stream(
                WorkerErrorCode::WorkerErrorGrammarUnsupported,
                "rust worker does not support grammars until Phase 7",
            )));
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

        let (events_tx, events_rx) = unbounded_channel();
        let queue_position = self.shared.waiting.load(Ordering::Acquire)
            + self.shared.running.load(Ordering::Acquire);
        let admitted = TokenEvent {
            event: Some(token_event::Event::Admitted(RequestAdmitted {
                queue_position,
                prompt_tokens: prompt_ids.len() as u32,
                prefill_chunks_estimated: prompt_ids
                    .len()
                    .div_ceil(kiln_models::generate::PREFILL_CHUNK)
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

    async fn drain(&self, _: Request<DrainRequest>) -> Result<Response<DrainAck>, Status> {
        // Eviction (Drain + SIGTERM) arrives in Phase 9; the Python worker
        // inherits UNIMPLEMENTED here too.
        Err(Status::unimplemented("Drain arrives in Phase 9"))
    }

    async fn stats(&self, _: Request<StatsRequest>) -> Result<Response<WorkerStats>, Status> {
        Err(Status::unimplemented(
            "Stats arrives with the Phase 4 engine",
        ))
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
