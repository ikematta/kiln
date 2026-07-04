//! `POST /v1/chat/completions` (SPEC §8.2): validate → render chat template →
//! tokenize → `Submit` → translate the worker's `TokenEvent` stream into an
//! OpenAI response (JSON or SSE).
//!
//! Tokenization/detokenization ownership depends on the worker kind:
//! - **Python worker**: owns its tokenizer. The gateway tokenizes via the
//!   `Tokenize` RPC (`add_special_tokens=false`; the template supplies BOS)
//!   and passes `TokenChunk.text` through; stop strings are matched in the
//!   worker.
//! - **Rust worker**: the gateway tokenizes locally with kiln-tokenize (same
//!   BOS contract), receives bare token ids, detokenizes them incrementally
//!   ([`StreamingDecoder`]), and matches stop strings itself — on a match it
//!   cancels the worker request and reports `finish_reason: "stop"`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    CancelRequest, FinishReason, Finished, Priority, StoppingParams, SubmitRequest, TokenChunk,
    TokenEvent, TokenIds, TokenizeRequest, submit_request, token_event,
};
use kiln_tokenize::{StopStringMatcher, StreamingDecoder};
use tonic::Streaming;
use tonic::transport::Channel;

use crate::app::{AppState, RequestId};
use crate::error::ApiError;
use crate::openai::{
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, Choice,
    ChunkChoice, Delta, Usage, ValidatedChat,
};
use crate::registry::{ModelEntry, WorkerStatus};

/// Request body cap; chat requests are text, not uploads.
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .unwrap_or_else(|| RequestId(uuid::Uuid::now_v7().to_string()));

    // Manual body handling so malformed JSON still yields an OpenAI-shaped
    // error instead of axum's plain-text rejection.
    let bytes = match axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return ApiError::invalid_request(format!("failed to read request body: {err}"))
                .into_response();
        }
    };
    let parsed: ChatCompletionRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid request JSON: {err}"))
                .into_response();
        }
    };

    let model = parsed.model.clone();
    match handle(Arc::clone(&state), parsed, request_id).await {
        Ok(response) => response,
        Err(err) => {
            state
                .metrics
                .chat_completions_total
                .with_label_values(&[&model, err.outcome()])
                .inc();
            tracing::info!(target: "kiln::chat", model = %model, code = err.code,
                status = err.status.as_u16(), "chat completion failed: {}", err.message);
            err.into_response()
        }
    }
}

async fn handle(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
    request_id: RequestId,
) -> Result<Response, ApiError> {
    let entry = state
        .registry
        .get(&request.model)
        .cloned()
        .ok_or_else(|| ApiError::model_not_found(&request.model))?;

    match entry.status() {
        WorkerStatus::Ready => {}
        WorkerStatus::Starting | WorkerStatus::Stopped => {
            return Err(ApiError::model_loading(&entry.id));
        }
        WorkerStatus::Restarting { .. } => {
            return Err(ApiError::worker_crashed(format!(
                "the worker for model '{}' crashed and is restarting; retry shortly",
                entry.id
            )));
        }
        WorkerStatus::Failed => return Err(ApiError::worker_failed(&entry.id)),
    }

    let validated = request.validate()?;
    let prompt = render_prompt(&entry, &validated)?;

    let mut client = WorkerClient::new(entry.channel.clone());
    // BOS contract (kiln-tokenize crate docs): the rendered template already
    // contains BOS, so both paths encode WITHOUT special tokens — locally
    // for Rust workers, via the Tokenize RPC for tokenizer-owning workers.
    let token_ids = match &entry.tokenizer {
        Some(tokenizer) => tokenizer
            .encode(&prompt, false)
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?,
        None => {
            client
                .tokenize(TokenizeRequest {
                    text: prompt,
                    add_special_tokens: false,
                })
                .await
                .map_err(|status| ApiError::from_worker_status(&status))?
                .into_inner()
                .token_ids
        }
    };
    if token_ids.is_empty() {
        return Err(ApiError::invalid_request("rendered prompt is empty"));
    }
    let prompt_tokens = token_ids.len() as u32;

    let max_context_len = entry
        .info
        .read()
        .await
        .as_ref()
        .map(|info| info.max_context_len)
        .unwrap_or(0);
    let max_tokens = validated.effective_max_tokens(prompt_tokens, max_context_len)?;

    // Stop strings: matched in the worker when it detokenizes (python), in
    // the gateway when it does (rust — the worker rejects stop_strings).
    let pipeline = TextPipeline::for_entry(&entry, &validated.stop_strings);
    let worker_stop_strings = match pipeline {
        TextPipeline::Passthrough => validated.stop_strings.clone(),
        TextPipeline::Decode { .. } => Vec::new(),
    };

    let submit = SubmitRequest {
        request_id: request_id.0.clone(),
        input: Some(submit_request::Input::TokenIds(TokenIds { ids: token_ids })),
        sampling: Some(validated.sampling),
        stopping: Some(StoppingParams {
            max_tokens,
            stop_token_ids: Vec::new(),
            stop_strings: worker_stop_strings,
            ignore_eos: false,
        }),
        grammar: None,
        priority: Priority::Interactive as i32,
        prefix_hint: 0,
        echo_prompt: false,
    };
    let events = client
        .submit(submit)
        .await
        .map_err(|status| ApiError::from_worker_status(&status))?
        .into_inner();

    let ctx = CompletionCtx {
        state,
        model: entry.id.clone(),
        completion_id: format!("chatcmpl-{}", request_id.0.replace('-', "")),
        created: unix_now(),
        request_id: request_id.0.clone(),
        channel: entry.channel.clone(),
    };
    if validated.stream {
        Ok(stream_response(
            ctx,
            events,
            pipeline,
            validated.include_usage,
        ))
    } else {
        collect_response(ctx, events, pipeline)
            .await
            .map(IntoResponse::into_response)
    }
}

/// Turns worker `TokenChunk`s into client-visible text.
enum TextPipeline {
    /// Tokenizer-owning worker: chunks carry final text (stops already
    /// applied worker-side).
    Passthrough,
    /// Rust worker: chunks carry bare ids; the gateway detokenizes and
    /// matches stop strings.
    Decode {
        decoder: StreamingDecoder,
        matcher: StopStringMatcher,
        matched: bool,
    },
}

impl TextPipeline {
    fn for_entry(entry: &ModelEntry, stop_strings: &[String]) -> Self {
        match &entry.tokenizer {
            Some(tokenizer) => TextPipeline::Decode {
                decoder: StreamingDecoder::new(Arc::clone(tokenizer)),
                matcher: StopStringMatcher::new(stop_strings),
                matched: false,
            },
            None => TextPipeline::Passthrough,
        }
    }

    /// Text safe to emit for this chunk. Flips [`Self::stop_matched`] when a
    /// stop string fires; everything after the match is dropped.
    fn push(&mut self, chunk: TokenChunk) -> Result<String, ApiError> {
        match self {
            TextPipeline::Passthrough => Ok(chunk.text),
            TextPipeline::Decode {
                decoder,
                matcher,
                matched,
            } => {
                if *matched {
                    return Ok(String::new());
                }
                let text = decoder
                    .push(&chunk.token_ids)
                    .map_err(|err| ApiError::internal(format!("detokenization failed: {err}")))?;
                let (released, hit) = matcher.push(&text);
                *matched = hit.is_some();
                Ok(released)
            }
        }
    }

    /// Final text at natural end-of-stream: decoder tail + held stop-string
    /// prefix (a tokenizer-owning worker already flushed on its side).
    fn finish(&mut self) -> Result<String, ApiError> {
        match self {
            TextPipeline::Passthrough => Ok(String::new()),
            TextPipeline::Decode {
                decoder,
                matcher,
                matched,
            } => {
                if *matched {
                    return Ok(String::new());
                }
                let tail = decoder
                    .finalize()
                    .map_err(|err| ApiError::internal(format!("detokenization failed: {err}")))?;
                let (mut released, hit) = matcher.push(&tail);
                *matched = hit.is_some();
                if hit.is_none() {
                    released.push_str(&matcher.flush());
                }
                Ok(released)
            }
        }
    }

    fn stop_matched(&self) -> bool {
        matches!(self, TextPipeline::Decode { matched: true, .. })
    }
}

fn render_prompt(entry: &ModelEntry, validated: &ValidatedChat) -> Result<String, ApiError> {
    let template = entry.template.as_ref().ok_or_else(|| {
        ApiError::invalid_request(format!(
            "model '{}' has no chat template; chat completions are unavailable for it",
            entry.id
        ))
    })?;
    template
        .render(&validated.messages, true)
        .map_err(|err| ApiError::invalid_request(format!("chat template rejected messages: {err}")))
}

struct CompletionCtx {
    state: Arc<AppState>,
    model: String,
    completion_id: String,
    created: u64,
    request_id: String,
    channel: Channel,
}

impl CompletionCtx {
    /// Best-effort worker-side cancellation after a gateway-side stop-string
    /// match; generation past the match is wasted work, not a correctness
    /// problem, so failures only log.
    async fn cancel_worker(&self) {
        let mut client = WorkerClient::new(self.channel.clone());
        if let Err(status) = client
            .cancel(CancelRequest {
                request_id: self.request_id.clone(),
            })
            .await
        {
            tracing::debug!(target: "kiln::chat", model = %self.model,
                request_id = %self.request_id, code = ?status.code(),
                "cancel after stop-string match failed");
        }
    }
}

impl CompletionCtx {
    fn record_ok(&self, finished: &Finished) {
        let metrics = &self.state.metrics;
        metrics
            .chat_completions_total
            .with_label_values(&[&self.model, "ok"])
            .inc();
        metrics
            .prompt_tokens_total
            .with_label_values(&[&self.model])
            .inc_by(u64::from(finished.prompt_tokens));
        metrics
            .completion_tokens_total
            .with_label_values(&[&self.model])
            .inc_by(u64::from(finished.completion_tokens));
    }

    fn record_err(&self, err: &ApiError) {
        self.state
            .metrics
            .chat_completions_total
            .with_label_values(&[&self.model, err.outcome()])
            .inc();
    }
}

/// Terminal outcome of a worker stream, normalized.
enum StreamEnd {
    Done {
        finished: Finished,
        finish_reason: &'static str,
    },
    Failed(ApiError),
}

fn classify_finished(finished: Finished, stop_matched: bool) -> StreamEnd {
    // A gateway-side stop-string match ends the request as a normal "stop"
    // regardless of how the worker's stream terminates afterwards (usually
    // CANCELLED, from our own Cancel; racily STOP/LENGTH). The client's
    // completion was already correct when the match fired.
    if stop_matched {
        return StreamEnd::Done {
            finish_reason: "stop",
            finished,
        };
    }
    match finished.finish_reason() {
        FinishReason::Stop => StreamEnd::Done {
            finish_reason: "stop",
            finished,
        },
        FinishReason::Length => StreamEnd::Done {
            finish_reason: "length",
            finished,
        },
        FinishReason::Error => StreamEnd::Failed(ApiError::from_worker_finished(&finished)),
        // Otherwise the gateway only cancels when the client is gone; nobody
        // reads this response, but keep the accounting honest.
        FinishReason::Cancelled => {
            StreamEnd::Failed(ApiError::worker_crashed("request was cancelled"))
        }
        FinishReason::PreemptedFatal | FinishReason::Unspecified => StreamEnd::Failed(
            ApiError::worker_crashed("request could not be completed (preempted); retry"),
        ),
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

async fn collect_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
) -> Result<axum::Json<ChatCompletion>, ApiError> {
    let mut content = String::new();
    let end = loop {
        match events.message().await {
            Ok(Some(event)) => match event.event {
                Some(token_event::Event::Tokens(chunk)) => {
                    let was_matched = pipeline.stop_matched();
                    content.push_str(&pipeline.push(chunk)?);
                    if !was_matched && pipeline.stop_matched() {
                        // Keep draining until Finished so usage stays real.
                        ctx.cancel_worker().await;
                    }
                }
                Some(token_event::Event::Finished(finished)) => {
                    content.push_str(&pipeline.finish()?);
                    break classify_finished(finished, pipeline.stop_matched());
                }
                // Admitted / PrefixCacheHit are observability-only here.
                _ => {}
            },
            Ok(None) => {
                break StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                ));
            }
            Err(status) => break StreamEnd::Failed(ApiError::from_worker_status(&status)),
        }
    };

    match end {
        StreamEnd::Failed(err) => Err(err),
        StreamEnd::Done {
            finished,
            finish_reason,
        } => {
            ctx.record_ok(&finished);
            Ok(axum::Json(ChatCompletion {
                id: ctx.completion_id.clone(),
                object: "chat.completion",
                created: ctx.created,
                model: ctx.model.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content,
                    },
                    logprobs: None,
                    finish_reason,
                }],
                usage: usage_of(&finished),
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming (SSE)
// ---------------------------------------------------------------------------

fn stream_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
    include_usage: bool,
) -> Response {
    // With include_usage, data chunks carry `"usage": null` and a final
    // chunk carries the usage object (OpenAI semantics).
    let usage_null: Option<Option<Usage>> = if include_usage { Some(None) } else { None };

    let stream = async_stream::stream! {
        let chunk = |choices: Vec<ChunkChoice>, usage: Option<Option<Usage>>| ChatCompletionChunk {
            id: ctx.completion_id.clone(),
            object: "chat.completion.chunk",
            created: ctx.created,
            model: ctx.model.clone(),
            choices,
            usage,
        };

        // Role preamble chunk.
        yield Ok::<SseEvent, Infallible>(sse_json(&chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: Some(String::new()) },
                logprobs: None,
                finish_reason: None,
            }],
            usage_null,
        )));

        loop {
            let end = match events.message().await {
                Ok(Some(event)) => match event.event {
                    Some(token_event::Event::Tokens(tc)) => {
                        let was_matched = pipeline.stop_matched();
                        let text = match pipeline.push(tc) {
                            Ok(text) => text,
                            Err(err) => {
                                // Detok failure is a gateway bug; surface it
                                // as the terminal error event.
                                ctx.record_err(&err);
                                yield Ok(sse_json(&err.body()));
                                return;
                            }
                        };
                        if !was_matched && pipeline.stop_matched() {
                            ctx.cancel_worker().await;
                        }
                        if !text.is_empty() {
                            yield Ok(sse_json(&chunk(
                                vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta { role: None, content: Some(text) },
                                    logprobs: None,
                                    finish_reason: None,
                                }],
                                usage_null,
                            )));
                        }
                        continue;
                    }
                    Some(token_event::Event::Finished(finished)) => {
                        match pipeline.finish() {
                            Ok(tail) if !tail.is_empty() => {
                                yield Ok(sse_json(&chunk(
                                    vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta { role: None, content: Some(tail) },
                                        logprobs: None,
                                        finish_reason: None,
                                    }],
                                    usage_null,
                                )));
                            }
                            Ok(_) => {}
                            Err(err) => {
                                ctx.record_err(&err);
                                yield Ok(sse_json(&err.body()));
                                return;
                            }
                        }
                        classify_finished(finished, pipeline.stop_matched())
                    }
                    _ => continue,
                },
                Ok(None) => StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                )),
                Err(status) => StreamEnd::Failed(ApiError::from_worker_status(&status)),
            };

            match end {
                StreamEnd::Done { finished, finish_reason } => {
                    yield Ok(sse_json(&chunk(
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            logprobs: None,
                            finish_reason: Some(finish_reason),
                        }],
                        usage_null,
                    )));
                    if include_usage {
                        yield Ok(sse_json(&chunk(Vec::new(), Some(Some(usage_of(&finished))))));
                    }
                    yield Ok(SseEvent::default().data("[DONE]"));
                    ctx.record_ok(&finished);
                }
                StreamEnd::Failed(err) => {
                    // Headers already went out as 200; surface the failure as
                    // a terminal SSE error event (and no [DONE]).
                    tracing::warn!(target: "kiln::chat", model = %ctx.model, code = err.code,
                        "streaming completion failed mid-stream: {}", err.message);
                    ctx.record_err(&err);
                    yield Ok(sse_json(&err.body()));
                }
            }
            return;
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn sse_json<T: serde::Serialize>(value: &T) -> SseEvent {
    match serde_json::to_string(value) {
        Ok(json) => SseEvent::default().data(json),
        Err(err) => {
            // Unreachable with our own Serialize types; degrade loudly
            // rather than panicking in the streaming path.
            tracing::error!(error = %err, "failed to serialize SSE payload");
            SseEvent::default().data("{}")
        }
    }
}

fn usage_of(finished: &Finished) -> Usage {
    Usage {
        prompt_tokens: finished.prompt_tokens,
        completion_tokens: finished.completion_tokens,
        total_tokens: finished.prompt_tokens + finished.completion_tokens,
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
