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
//!
//! # Finish-reason precedence (gateway stop match vs worker terminal event)
//!
//! A gateway-side stop-string match ALWAYS wins: the response reports
//! `finish_reason: "stop"` no matter how the worker's stream terminates
//! afterwards — CANCELLED (the usual case, from our own Cancel), STOP (an
//! EOS raced ahead of the Cancel), LENGTH (the worker hit max_tokens before
//! the Cancel landed), or even ERROR. This is deliberate, not a race bug:
//! the client-visible completion is defined by the gateway's text pipeline,
//! which truncated at the match, so by OpenAI semantics the completion
//! ended at a stop sequence. The worker's terminal reason describes its own
//! post-match continuation, which the client never saw — and a
//! tokenizer-owning worker (python) stops at exactly the match point and
//! reports STOP for the identical request. A worker LENGTH being reported
//! as "stop" is therefore the *correct* account of the response the client
//! received, not a silent misreport.
//!
//! Usage follows the same boundary: on a gateway-side match,
//! `completion_tokens` is the number of tokens consumed up to and including
//! the token whose text completed the stop string
//! ([`TextPipeline::apply_usage`]) — never the worker's total, which
//! includes cancel-overshoot tokens. This is the same count the
//! tokenizer-owning worker reports, so usage is identical across workers
//! (asserted by the cross-worker parity e2e test).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    CancelRequest, FinishReason, Finished, StoppingParams, SubmitRequest, TokenChunk, TokenEvent,
    TokenIds, TokenizeRequest, submit_request, token_event,
};
use kiln_tokenize::{StopStringMatcher, StreamingDecoder, ToolCallParser, ToolEvent};
use tonic::Streaming;
use tonic::transport::Channel;

use crate::app::{AppState, RequestId};
use crate::error::ApiError;
use crate::openai::{
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, Choice,
    ChunkChoice, Delta, DeltaFunction, DeltaToolCall, ResponseFunction, ResponseToolCall, Usage,
    ValidatedChat,
};
use crate::registry::{ModelEntry, WorkerStatus};
use crate::timeout::{Deadlines, TimeoutScope, WorkerEvent, next_event};

/// Request body cap (SPEC §8.3 request size limits); completion requests
/// are text, not uploads, and every downstream cost (JSON parse, template
/// render, tokenization) is linear in this bound. Token-shaped ceilings are
/// enforced separately: the model-context check in `effective_max_tokens`
/// and the per-key tpm reservation. Shared with `/v1/completions` and
/// `/v1/messages`.
pub(crate) const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

/// Reads the request body under [`MAX_BODY_BYTES`], distinguishing the
/// over-limit case (413, SPEC §8.3) from a malformed or interrupted body
/// (400). Callers render their surface's error envelope.
pub(crate) async fn read_body(
    request: axum::extract::Request,
) -> Result<axum::body::Bytes, ApiError> {
    axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(|err| {
            let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&err);
            while let Some(cause) = source {
                if cause.is::<http_body_util::LengthLimitError>() {
                    return ApiError::request_too_large(MAX_BODY_BYTES);
                }
                source = cause.source();
            }
            ApiError::invalid_request(format!("failed to read request body: {err}"))
        })
}

/// Registry lookup + worker-status gate shared by every completion-shaped
/// endpoint: only a Ready worker receives requests; every other state maps
/// to the matching client-visible error. Routing a request counts as use
/// (LRU position + TTL idle clock), and a request for an Unloaded model is
/// the on-demand reload trigger (SPEC §2.2): the load starts in the
/// background while the client gets the retriable `model_loading` error.
pub(crate) fn ready_entry(state: &AppState, model: &str) -> Result<Arc<ModelEntry>, ApiError> {
    let entry = state
        .registry
        .get(model)
        .ok_or_else(|| ApiError::model_not_found(model))?;
    match entry.status() {
        WorkerStatus::Ready => {
            state.lifecycle.touch(&entry.id);
            Ok(entry)
        }
        WorkerStatus::Starting | WorkerStatus::Stopped => Err(ApiError::model_loading(&entry.id)),
        WorkerStatus::Unloaded { .. } => {
            state.lifecycle.request_load(&entry.id);
            Err(ApiError::model_loading(&entry.id))
        }
        WorkerStatus::Draining => Err(ApiError::model_unloading(&entry.id)),
        WorkerStatus::Restarting { .. } => Err(ApiError::worker_crashed(format!(
            "the worker for model '{}' crashed and is restarting; retry shortly",
            entry.id
        ))),
        WorkerStatus::Failed => Err(ApiError::worker_failed(&entry.id)),
    }
}

/// Per-request memory admission (SPEC §2.3/§8.2, Phase 9 part 2), shared by
/// every completion-shaped endpoint: refuses the request when serving it
/// could materialize KV-pool bytes the machine no longer has room for —
/// under the configured budget OR under real system availability/pressure
/// (the tightest bound wins; `denial.constraint` names it). Runs against
/// live heartbeat numbers, so drift since load (pools, caches) is priced
/// in — this is the per-request check, distinct from the load-time
/// `load()` gate.
pub(crate) fn admit_memory(state: &AppState, entry: &ModelEntry) -> Result<(), ApiError> {
    state.lifecycle.admit_request(&entry.id).map_err(|denial| {
        state
            .metrics
            .admission_rejects_total
            .with_label_values(&[&entry.id])
            .inc();
        tracing::warn!(target: "kiln::admission", model = %entry.id,
            needed_bytes = denial.needed_bytes, headroom_bytes = denial.headroom_bytes,
            constraint = denial.constraint.label(),
            "request rejected: projected pool growth exceeds machine headroom");
        ApiError::insufficient_memory(&entry.id, denial)
    })
}

/// Prompt-text → token ids under the entry's tokenization ownership: locally
/// for Rust-worker models, via the worker's `Tokenize` RPC otherwise.
/// `add_special_tokens` follows the BOS contract: false when the text came
/// through a chat template (the template supplies BOS), true for raw
/// completions prompts.
pub(crate) async fn encode_prompt(
    entry: &ModelEntry,
    client: &mut WorkerClient<Channel>,
    prompt: String,
    add_special_tokens: bool,
) -> Result<Vec<u32>, ApiError> {
    match &entry.tokenizer {
        Some(tokenizer) => tokenizer
            .encode(&prompt, add_special_tokens)
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}"))),
        None => Ok(client
            .tokenize(TokenizeRequest {
                text: prompt,
                add_special_tokens,
            })
            .await
            .map_err(|status| ApiError::from_worker_status(&status))?
            .into_inner()
            .token_ids),
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    // Timeout budgets anchor here — arrival, not prefill start — so queue
    // wait under admission pressure counts (crate::timeout module docs).
    let arrival = tokio::time::Instant::now();
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .unwrap_or_else(|| RequestId(uuid::Uuid::now_v7().to_string()));
    let rate = request
        .extensions()
        .get::<crate::ratelimit::RateLimitHandle>()
        .cloned();

    // Manual body handling so malformed JSON still yields an OpenAI-shaped
    // error instead of axum's plain-text rejection.
    let bytes = match read_body(request).await {
        Ok(bytes) => bytes,
        Err(err) => return err.into_response(),
    };
    let parsed: ChatCompletionRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid request JSON: {err}"))
                .into_response();
        }
    };

    let model = parsed.model.clone();
    match handle(Arc::clone(&state), parsed, request_id, rate, arrival).await {
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
    rate: Option<crate::ratelimit::RateLimitHandle>,
    arrival: tokio::time::Instant,
) -> Result<Response, ApiError> {
    let deadlines = Deadlines::start(&state.timeouts, arrival);
    let entry = ready_entry(&state, &request.model)?;
    admit_memory(&state, &entry)?;

    let mut validated = request.validate()?;

    // MCP tools in scope (SPEC §8.4, crate::mcp module docs): merged into
    // the request unless it opted out of tools, a structured-output
    // grammar constrains the decode, or the model has no tool-call format
    // (it cannot emit calls — requests without client tools keep working
    // rather than turning into 400s). A non-empty set switches the request
    // onto the multi-round execution path; empty leaves the historical
    // single-round path untouched.
    let format = entry
        .template
        .as_ref()
        .and_then(|template| template.tool_call_format());
    let mut mcp_tools =
        if validated.tools_disabled || validated.grammar.is_some() || format.is_none() {
            crate::mcp::McpToolSet::empty()
        } else {
            state.mcp.snapshot()
        };
    if !mcp_tools.is_empty() {
        mcp_tools.merge_into(&mut validated.tools, &entry.id);
    }
    if let (false, Some(format)) = (mcp_tools.is_empty(), format) {
        let mut rounds = McpRounds {
            state: Arc::clone(&state),
            entry: Arc::clone(&entry),
            messages: std::mem::take(&mut validated.messages),
            tools: std::mem::take(&mut validated.tools),
            format,
            sampling: validated.sampling,
            client_max_tokens: validated.max_tokens,
            stop_strings: validated.stop_strings.clone(),
            priority: validated.priority as i32,
            rate,
            base_request_id: request_id.0.clone(),
            thinking_disabled: false,
            mcp: mcp_tools,
            round: 0,
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
        };
        let mut ctx = CompletionCtx {
            state: Arc::clone(&state),
            model: entry.id.clone(),
            completion_id: format!("chatcmpl-{}", request_id.0.replace('-', "")),
            created: unix_now(),
            request_id: request_id.0.clone(),
            channel: entry.channel.clone(),
            requests_total: state.metrics.chat_completions_total.clone(),
            tpm: None,
        };
        // Round 1 submits here so its failures surface as proper HTTP
        // errors, exactly like the single-round path; only rounds 2+ can
        // fail mid-stream.
        let first = rounds.submit_round(&mut ctx).await?;
        return if validated.stream {
            Ok(stream_mcp_response(
                ctx,
                rounds,
                first,
                validated.include_usage,
                deadlines,
            ))
        } else {
            collect_mcp_response(ctx, rounds, first, deadlines)
                .await
                .map(IntoResponse::into_response)
        };
    }

    let prompt = render_prompt(&entry, &validated)?;
    // Tool-call parsing (SPEC §8.2): the family format comes from the
    // model's own chat template; a model without a known format cannot
    // honor `tools`, so reject up front rather than stream raw markers.
    let tool_parser = if validated.tools.is_empty() {
        None
    } else {
        let format = format.ok_or_else(|| {
            ApiError::invalid_request(format!(
                "model '{}' has no known tool-call format; 'tools' is not supported for it",
                entry.id
            ))
        })?;
        Some(ToolCallParser::new(format, &validated.tools))
    };

    let mut client = WorkerClient::new(entry.channel.clone());
    // BOS contract (kiln-tokenize crate docs): the rendered template already
    // contains BOS, so encode WITHOUT special tokens.
    let token_ids = encode_prompt(&entry, &mut client, prompt, false).await?;
    if token_ids.is_empty() {
        return Err(ApiError::invalid_request("rendered prompt is empty"));
    }
    let prompt_tokens = token_ids.len() as u32;

    let (max_context_len, worker_grammar_capable) = {
        let info = entry.info.read().await;
        (
            info.as_ref().map(|info| info.max_context_len).unwrap_or(0),
            info.as_ref().is_some_and(|info| {
                info.capabilities
                    .contains(&(kiln_proto::v1::Capability::Grammar as i32))
            }),
        )
    };
    let max_tokens = validated.effective_max_tokens(prompt_tokens, max_context_len)?;

    // SPEC §5 capability gating: structured output requires the worker to
    // advertise CAPABILITY_GRAMMAR (the python worker lacks it in v1) —
    // 400 here instead of a worker-side in-band rejection.
    let grammar = validated.grammar.take();
    if grammar.is_some() && !worker_grammar_capable {
        return Err(ApiError::invalid_request(
            "this model's worker does not support structured output (response_format); \
             serve the model with the rust worker to enable it",
        ));
    }

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
        grammar,
        priority: validated.priority as i32,
        prefix_hint: 0,
        echo_prompt: false,
    };
    // tpm reservation (SPEC §8.3): the worst case (prompt + max_tokens) is
    // held from here until the request settles; record_ok refunds the
    // unused remainder (crate::ratelimit module docs). Taken last, right
    // before Submit, so validation failures never forfeit budget.
    let tpm = crate::ratelimit::reserve_completion_tokens(
        &state,
        rate.as_ref(),
        prompt_tokens,
        max_tokens,
    )?;
    let events = match client.submit(submit).await {
        Ok(response) => response.into_inner(),
        Err(status) => {
            // The request never reached the engine: release the hold.
            if let Some(tpm) = &tpm {
                tpm.settle(0);
            }
            return Err(ApiError::from_worker_status(&status));
        }
    };

    let requests_total = state.metrics.chat_completions_total.clone();
    let ctx = CompletionCtx {
        state,
        model: entry.id.clone(),
        completion_id: format!("chatcmpl-{}", request_id.0.replace('-', "")),
        created: unix_now(),
        request_id: request_id.0.clone(),
        channel: entry.channel.clone(),
        requests_total,
        tpm,
    };
    if validated.stream {
        Ok(stream_response(
            ctx,
            events,
            pipeline,
            tool_parser,
            validated.include_usage,
            deadlines,
        ))
    } else {
        collect_response(ctx, events, pipeline, tool_parser, deadlines)
            .await
            .map(IntoResponse::into_response)
    }
}

/// Turns worker `TokenChunk`s into client-visible text. Shared with
/// `/v1/completions`, whose text pipeline is identical (module docs above
/// on finish-reason precedence and usage apply to both endpoints).
pub(crate) enum TextPipeline {
    /// Tokenizer-owning worker: chunks carry final text (stops already
    /// applied worker-side).
    Passthrough,
    /// Rust worker: chunks carry bare ids; the gateway detokenizes and
    /// matches stop strings.
    Decode {
        decoder: StreamingDecoder,
        matcher: StopStringMatcher,
        /// Tokens consumed from the worker so far.
        tokens_seen: u32,
        /// Frozen at the chunk whose text completed a stop string — the
        /// client-visible completion length (see the module docs on usage).
        tokens_at_match: Option<u32>,
        /// The stop string that fired (the Anthropic API reports it as
        /// `stop_sequence`; a tokenizer-owning worker reports its own via
        /// `Finished.matched_stop`).
        stop_hit: Option<String>,
    },
}

impl TextPipeline {
    pub(crate) fn for_entry(entry: &ModelEntry, stop_strings: &[String]) -> Self {
        match &entry.tokenizer {
            Some(tokenizer) => TextPipeline::Decode {
                decoder: StreamingDecoder::new(Arc::clone(tokenizer)),
                matcher: StopStringMatcher::new(stop_strings),
                tokens_seen: 0,
                tokens_at_match: None,
                stop_hit: None,
            },
            None => TextPipeline::Passthrough,
        }
    }

    /// Text safe to emit for this chunk. Flips [`Self::stop_matched`] when a
    /// stop string fires; everything after the match is dropped.
    pub(crate) fn push(&mut self, chunk: TokenChunk) -> Result<String, ApiError> {
        match self {
            TextPipeline::Passthrough => Ok(chunk.text),
            TextPipeline::Decode {
                decoder,
                matcher,
                tokens_seen,
                tokens_at_match,
                stop_hit,
            } => {
                if tokens_at_match.is_some() {
                    // Post-match chunks are the worker's cancel overshoot;
                    // they are not part of the client-visible completion.
                    return Ok(String::new());
                }
                *tokens_seen += chunk.token_ids.len() as u32;
                let text = decoder
                    .push(&chunk.token_ids)
                    .map_err(|err| ApiError::internal(format!("detokenization failed: {err}")))?;
                let (released, hit) = matcher.push(&text);
                if hit.is_some() {
                    *tokens_at_match = Some(*tokens_seen);
                    *stop_hit = hit;
                }
                Ok(released)
            }
        }
    }

    /// Final text at natural end-of-stream: decoder tail + held stop-string
    /// prefix (a tokenizer-owning worker already flushed on its side).
    pub(crate) fn finish(&mut self) -> Result<String, ApiError> {
        match self {
            TextPipeline::Passthrough => Ok(String::new()),
            TextPipeline::Decode {
                decoder,
                matcher,
                tokens_seen,
                tokens_at_match,
                stop_hit,
            } => {
                if tokens_at_match.is_some() {
                    return Ok(String::new());
                }
                let tail = decoder
                    .finalize()
                    .map_err(|err| ApiError::internal(format!("detokenization failed: {err}")))?;
                let (mut released, hit) = matcher.push(&tail);
                if hit.is_some() {
                    *tokens_at_match = Some(*tokens_seen);
                    *stop_hit = hit;
                } else {
                    released.push_str(&matcher.flush());
                }
                Ok(released)
            }
        }
    }

    pub(crate) fn stop_matched(&self) -> bool {
        matches!(
            self,
            TextPipeline::Decode {
                tokens_at_match: Some(_),
                ..
            }
        )
    }

    /// The stop string a gateway-side match fired on, when [`Self::stop_matched`].
    /// `None` on the passthrough pipeline — there the worker matched, and
    /// reports the string via `Finished.matched_stop`.
    pub(crate) fn matched_stop(&self) -> Option<&str> {
        match self {
            TextPipeline::Decode { stop_hit, .. } => stop_hit.as_deref(),
            TextPipeline::Passthrough => None,
        }
    }

    /// On a gateway-side match, overrides the worker-reported completion
    /// count with the client-visible one (tokens up to and including the
    /// match) — the worker's total includes cancel overshoot the client
    /// never saw, and the tokenizer-owning worker stops (and counts) at
    /// exactly this point, so usage agrees across workers.
    pub(crate) fn apply_usage(&self, finished: &mut Finished) {
        if let TextPipeline::Decode {
            tokens_at_match: Some(count),
            ..
        } = self
        {
            finished.completion_tokens = *count;
        }
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
        .render_with_tools(&validated.messages, true, &validated.tools)
        .map_err(|err| ApiError::invalid_request(format!("chat template rejected messages: {err}")))
}

// ---------------------------------------------------------------------------
// Tool-call assembly (shared by the streaming and non-streaming paths)
// ---------------------------------------------------------------------------

/// Routes the text pipeline's output through the tool-call parser when the
/// request has tools, or passes it through as plain content otherwise.
/// Shared with `/v1/messages` (crate::messages), whose adapter consumes the
/// same [`ToolEvent`] stream under Anthropic framing.
pub(crate) struct ToolRoute {
    parser: Option<ToolCallParser>,
    /// Calls whose arguments finished cleanly — drives the
    /// `finish_reason: "tool_calls"` upgrade.
    calls_completed: usize,
}

impl ToolRoute {
    pub(crate) fn new(parser: Option<ToolCallParser>) -> Self {
        Self {
            parser,
            calls_completed: 0,
        }
    }

    pub(crate) fn push(&mut self, text: String) -> Vec<ToolEvent> {
        match &mut self.parser {
            None if text.is_empty() => Vec::new(),
            None => vec![ToolEvent::Content(text)],
            Some(parser) => {
                let events = parser.push(&text);
                self.count(&events);
                events
            }
        }
    }

    /// End-of-stream: the pipeline's tail plus the parser's flush.
    pub(crate) fn finish(&mut self, tail: String) -> Vec<ToolEvent> {
        match &mut self.parser {
            None if tail.is_empty() => Vec::new(),
            None => vec![ToolEvent::Content(tail)],
            Some(parser) => {
                let mut events = parser.push(&tail);
                events.extend(parser.finish());
                self.count(&events);
                events
            }
        }
    }

    fn count(&mut self, events: &[ToolEvent]) {
        self.calls_completed += events
            .iter()
            .filter(|e| matches!(e, ToolEvent::CallEnd { .. }))
            .count();
    }

    /// Calls whose arguments finished cleanly (Anthropic's `tool_use`
    /// stop-reason upgrade keys on this, like OpenAI's `tool_calls`).
    pub(crate) fn calls_completed(&self) -> usize {
        self.calls_completed
    }

    /// OpenAI reports a completion that ended by emitting tool calls as
    /// `finish_reason: "tool_calls"`; a truncated one stays `"length"`.
    fn adjust_reason(&self, reason: &'static str) -> &'static str {
        if self.calls_completed > 0 && reason == "stop" {
            "tool_calls"
        } else {
            reason
        }
    }
}

fn new_call_id() -> String {
    // v7 like the request ids (the uuid feature already enabled); clients
    // only require uniqueness.
    format!("call_{}", uuid::Uuid::now_v7().simple())
}

/// One tool event → one SSE delta (`CallEnd` has no OpenAI representation).
fn tool_event_delta(event: ToolEvent) -> Option<Delta> {
    match event {
        ToolEvent::Content(text) => Some(Delta {
            content: Some(text),
            ..Delta::default()
        }),
        ToolEvent::CallStart { index, name } => Some(Delta {
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: Some(new_call_id()),
                call_type: Some("function"),
                function: DeltaFunction {
                    name: Some(name),
                    arguments: String::new(),
                },
            }]),
            ..Delta::default()
        }),
        ToolEvent::CallArgs { index, delta } => Some(Delta {
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: None,
                call_type: None,
                function: DeltaFunction {
                    name: None,
                    arguments: delta,
                },
            }]),
            ..Delta::default()
        }),
        ToolEvent::CallEnd { .. } => None,
    }
}

/// Per-request context shared by the chat and text-completions handlers:
/// response identity, metrics wiring (`requests_total` is the endpoint's own
/// counter), and the worker channel for post-match cancellation.
pub(crate) struct CompletionCtx {
    pub(crate) state: Arc<AppState>,
    pub(crate) model: String,
    pub(crate) completion_id: String,
    pub(crate) created: u64,
    pub(crate) request_id: String,
    pub(crate) channel: Channel,
    /// `kiln_chat_completions_total` or `kiln_completions_total`.
    pub(crate) requests_total: prometheus::IntCounterVec,
    /// The key's tpm hold (SPEC §8.3), settled to actual usage in
    /// [`Self::record_ok`]. Errors and client disconnects drop it
    /// unsettled, forfeiting the full reservation until the bucket
    /// refills — deliberate (crate::ratelimit module docs).
    pub(crate) tpm: Option<crate::ratelimit::TpmReservation>,
}

impl CompletionCtx {
    /// Best-effort worker-side cancellation through the Cancel RPC (the
    /// ≤2-step stop bound the workers promise), used after a gateway-side
    /// stop-string match and by timeout aborts. Returns the worker's
    /// `CancelAck.found`; RPC failures only log — generation past the
    /// abort point is wasted work, not a correctness problem.
    pub(crate) async fn cancel_worker(&self) -> bool {
        let mut client = WorkerClient::new(self.channel.clone());
        match client
            .cancel(CancelRequest {
                request_id: self.request_id.clone(),
            })
            .await
        {
            Ok(ack) => ack.into_inner().found,
            Err(status) => {
                tracing::debug!(target: "kiln::chat", model = %self.model,
                    request_id = %self.request_id, code = ?status.code(),
                    "worker cancel RPC failed");
                false
            }
        }
    }

    /// Aborts a timed-out request (SPEC §8.3): cancels through the worker's
    /// Cancel path — the same mechanism as a user-initiated stop, no
    /// separate teardown — bumps `kiln_timeout_total`, and returns the
    /// client-facing 504. The tpm reservation is deliberately left
    /// unsettled: it forfeits exactly like a client disconnect
    /// (crate::ratelimit module docs — refunds for abandoned work would
    /// let a client burn GPU without depleting its budget), and the
    /// forfeit self-heals within the refill minute.
    pub(crate) async fn abort_for_timeout(&self, scope: TimeoutScope) -> ApiError {
        let cancel_found = self.cancel_worker().await;
        self.state
            .metrics
            .timeouts_total
            .with_label_values(&[&self.model, scope.label()])
            .inc();
        let timeouts = &self.state.timeouts;
        let (budget, err) = match scope {
            TimeoutScope::Ttft => {
                let secs = timeouts.ttft.map(|d| d.as_secs()).unwrap_or(0);
                (secs, ApiError::ttft_timeout(&self.model, secs))
            }
            TimeoutScope::Total => {
                let secs = timeouts.total.map(|d| d.as_secs()).unwrap_or(0);
                (secs, ApiError::total_timeout(&self.model, secs))
            }
        };
        tracing::info!(target: "kiln::timeout", model = %self.model,
            request_id = %self.request_id, scope = scope.label(),
            budget_secs = budget, cancel_found,
            "request aborted by timeout; cancelled via worker Cancel");
        err
    }
}

impl CompletionCtx {
    pub(crate) fn record_ok(&self, finished: &Finished) {
        // Reconcile the tpm hold to client-visible usage (`finished` has
        // already been through apply_usage at every call site).
        if let Some(tpm) = &self.tpm {
            tpm.settle(u64::from(finished.prompt_tokens) + u64::from(finished.completion_tokens));
        }
        let metrics = &self.state.metrics;
        self.requests_total
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

    pub(crate) fn record_err(&self, err: &ApiError) {
        self.requests_total
            .with_label_values(&[&self.model, err.outcome()])
            .inc();
    }
}

/// Terminal outcome of a worker stream, normalized.
pub(crate) enum StreamEnd {
    Done {
        finished: Finished,
        finish_reason: &'static str,
    },
    Failed(ApiError),
}

pub(crate) fn classify_finished(finished: Finished, stop_matched: bool) -> StreamEnd {
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
// MCP execution loop (SPEC §8.4, crate::mcp module docs)
// ---------------------------------------------------------------------------

/// Drives the MCP execution loop's generation rounds: each round renders
/// the accumulated conversation, tokenizes, reserves tpm, and submits —
/// the same pipeline the single-round path runs once. Shared with
/// `/v1/messages` (crate::messages), whose loop differs only in framing.
pub(crate) struct McpRounds {
    pub(crate) state: Arc<AppState>,
    pub(crate) entry: Arc<ModelEntry>,
    /// The conversation, growing by an assistant turn + tool results per
    /// executed round.
    pub(crate) messages: Vec<kiln_tokenize::ChatMessage>,
    /// Merged tool definitions (client + MCP), rendered every round.
    pub(crate) tools: Vec<serde_json::Value>,
    pub(crate) format: kiln_tokenize::ToolCallFormat,
    pub(crate) sampling: kiln_proto::v1::SamplingParams,
    /// The client's explicit max_tokens; None = fill the remaining context
    /// per round (chat surface only — messages requires it). Bounds each
    /// round's generation; response usage sums the rounds.
    pub(crate) client_max_tokens: Option<u32>,
    pub(crate) stop_strings: Vec<String>,
    pub(crate) priority: i32,
    pub(crate) rate: Option<crate::ratelimit::RateLimitHandle>,
    pub(crate) base_request_id: String,
    /// Messages-surface render flag (`enable_thinking=false`).
    pub(crate) thinking_disabled: bool,
    pub(crate) mcp: crate::mcp::McpToolSet,
    pub(crate) round: usize,
    /// Client-visible usage accumulated across rounds.
    pub(crate) prompt_tokens_total: u64,
    pub(crate) completion_tokens_total: u64,
}

/// One round's live worker stream plus its fresh text/tool pipelines.
pub(crate) struct RoundStream {
    pub(crate) events: Streaming<TokenEvent>,
    pub(crate) pipeline: TextPipeline,
    pub(crate) parser: ToolCallParser,
    pub(crate) prompt_tokens: u32,
}

impl McpRounds {
    /// Renders, tokenizes, reserves, submits the next round; `ctx` is
    /// updated to the round's worker request id and tpm hold so cancel and
    /// settlement always target the live round.
    pub(crate) async fn submit_round(
        &mut self,
        ctx: &mut CompletionCtx,
    ) -> Result<RoundStream, ApiError> {
        self.round += 1;
        // Per-round memory admission: a later round's prompt has grown and
        // headroom may have moved since arrival — same live gate as the
        // single-round path.
        admit_memory(&self.state, &self.entry)?;
        let template = self.entry.template.as_ref().ok_or_else(|| {
            // Guaranteed by construction: the loop path requires a
            // tool-call format, which only a template provides.
            ApiError::internal("MCP loop on a model without a chat template")
        })?;
        let extra: &[(&str, serde_json::Value)] = if self.thinking_disabled {
            &[("enable_thinking", serde_json::Value::Bool(false))]
        } else {
            &[]
        };
        let prompt = template
            .render_full(&self.messages, true, &self.tools, extra)
            .map_err(|err| {
                ApiError::invalid_request(format!("chat template rejected messages: {err}"))
            })?;
        let mut client = WorkerClient::new(self.entry.channel.clone());
        let token_ids = encode_prompt(&self.entry, &mut client, prompt, false).await?;
        if token_ids.is_empty() {
            return Err(ApiError::invalid_request("rendered prompt is empty"));
        }
        let prompt_tokens = token_ids.len() as u32;

        let max_context_len = {
            let info = self.entry.info.read().await;
            info.as_ref().map(|info| info.max_context_len).unwrap_or(0)
        };
        // Same policy as ValidatedChat::effective_max_tokens, recomputed
        // per round because the prompt grows with every executed round.
        if max_context_len > 0 && prompt_tokens >= max_context_len {
            return Err(ApiError::context_length_exceeded(format!(
                "prompt is {prompt_tokens} tokens but the model's context is {max_context_len}"
            )));
        }
        let max_tokens = match self.client_max_tokens {
            Some(requested) => requested,
            None if max_context_len > 0 => max_context_len - prompt_tokens,
            None => crate::openai::FALLBACK_MAX_TOKENS,
        };

        let pipeline = TextPipeline::for_entry(&self.entry, &self.stop_strings);
        let worker_stop_strings = match pipeline {
            TextPipeline::Passthrough => self.stop_strings.clone(),
            TextPipeline::Decode { .. } => Vec::new(),
        };
        // Round ids stay tied to the request id for log correlation; the
        // suffix keeps each worker submission unique.
        let round_request_id = if self.round == 1 {
            self.base_request_id.clone()
        } else {
            format!("{}-r{}", self.base_request_id, self.round)
        };
        let submit = SubmitRequest {
            request_id: round_request_id.clone(),
            input: Some(submit_request::Input::TokenIds(TokenIds { ids: token_ids })),
            sampling: Some(self.sampling),
            stopping: Some(StoppingParams {
                max_tokens,
                stop_token_ids: Vec::new(),
                stop_strings: worker_stop_strings,
                ignore_eos: false,
            }),
            grammar: None, // grammar requests never enter the loop path
            priority: self.priority,
            prefix_hint: 0,
            echo_prompt: false,
        };
        // tpm reserves per round (crate::mcp module docs): every round is
        // real work, priced like the single-round path prices its one.
        let tpm = crate::ratelimit::reserve_completion_tokens(
            &self.state,
            self.rate.as_ref(),
            prompt_tokens,
            max_tokens,
        )?;
        let events = match client.submit(submit).await {
            Ok(response) => response.into_inner(),
            Err(status) => {
                if let Some(tpm) = &tpm {
                    tpm.settle(0);
                }
                return Err(ApiError::from_worker_status(&status));
            }
        };
        ctx.request_id = round_request_id;
        ctx.tpm = tpm;
        Ok(RoundStream {
            events,
            pipeline,
            parser: ToolCallParser::new(self.format, &self.tools),
            prompt_tokens,
        })
    }

    /// Whether a finished round's calls should be executed gateway-side:
    /// natural stop, at least one call, every call complete, every name
    /// MCP-sourced, and the round cap not yet reached (crate::mcp module
    /// docs — anything else returns the turn to the client).
    pub(crate) fn should_execute(
        &self,
        finish_reason: &str,
        stop_matched: bool,
        calls: &[(String, String)],
        calls_completed: usize,
    ) -> bool {
        finish_reason == "stop"
            && !stop_matched
            && !calls.is_empty()
            && calls_completed == calls.len()
            && calls.iter().all(|(name, _)| self.mcp.contains(name))
            && self.round < crate::mcp::MAX_ROUNDS
    }

    /// Executes the round's calls and appends the assistant turn plus one
    /// `tool` message per result — the Phase 7 round-trip message flow,
    /// driven gateway-side. `assistant_content` is the round's client-shaped
    /// content (chat: verbatim including any think text; messages: text
    /// segments only — each surface feeds back exactly what a faithful
    /// client of that surface would). Err = the total budget expired
    /// mid-call; abort the request.
    pub(crate) async fn execute_round(
        &mut self,
        assistant_content: String,
        calls: Vec<(String, String)>,
        deadlines: &Deadlines,
    ) -> Result<(), TimeoutScope> {
        let tool_calls = calls
            .iter()
            .map(|(name, arguments)| kiln_tokenize::MessageToolCall {
                call_type: "function".to_owned(),
                function: kiln_tokenize::MessageToolFunction {
                    name: name.clone(),
                    // Completed calls carry parseable JSON in practice; a
                    // degenerate emission falls back to {} — mirroring the
                    // wire validation's treatment of empty arguments.
                    arguments: serde_json::from_str(arguments)
                        .unwrap_or_else(|_| serde_json::json!({})),
                },
            })
            .collect();
        self.messages.push(kiln_tokenize::ChatMessage {
            role: "assistant".to_owned(),
            content: assistant_content,
            tool_calls: Some(tool_calls),
        });
        for (name, arguments) in &calls {
            let result = self
                .mcp
                .call(&self.state.metrics, name, arguments, deadlines.total())
                .await;
            match result {
                crate::mcp::ToolCallResult::Text(text) => {
                    self.messages
                        .push(kiln_tokenize::ChatMessage::text("tool", text));
                }
                crate::mcp::ToolCallResult::TotalTimeout => return Err(TimeoutScope::Total),
            }
        }
        Ok(())
    }
}

impl CompletionCtx {
    /// Settles an intermediate MCP-loop round: reconcile its tpm hold to
    /// actuals and count its tokens. `requests_total` is deliberately not
    /// touched — the request counts once, at its terminal round.
    pub(crate) fn settle_round(&mut self, finished: &Finished) {
        if let Some(tpm) = self.tpm.take() {
            tpm.settle(u64::from(finished.prompt_tokens) + u64::from(finished.completion_tokens));
        }
        let metrics = &self.state.metrics;
        metrics
            .prompt_tokens_total
            .with_label_values(&[&self.model])
            .inc_by(u64::from(finished.prompt_tokens));
        metrics
            .completion_tokens_total
            .with_label_values(&[&self.model])
            .inc_by(u64::from(finished.completion_tokens));
    }
}

/// Non-streaming chat under the MCP loop: rounds accumulate content (the
/// client-visible completion concatenates every round, matching what a
/// streaming client would have watched arrive); executed rounds' calls
/// vanish into the conversation, a terminal round's calls surface as
/// normal `tool_calls`.
async fn collect_mcp_response(
    mut ctx: CompletionCtx,
    mut rounds: McpRounds,
    first: RoundStream,
    mut deadlines: Deadlines,
) -> Result<axum::Json<ChatCompletion>, ApiError> {
    let mut content_all = String::new();
    let mut next_round = Some(first);
    loop {
        let RoundStream {
            mut events,
            mut pipeline,
            parser,
            ..
        } = match next_round.take() {
            Some(round) => round,
            None => rounds.submit_round(&mut ctx).await?,
        };
        let mut route = ToolRoute::new(Some(parser));
        let mut content = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        let apply = |events: Vec<ToolEvent>, content: &mut String, calls: &mut Vec<_>| {
            for event in events {
                match event {
                    ToolEvent::Content(text) => content.push_str(&text),
                    ToolEvent::CallStart { name, .. } => calls.push((name, String::new())),
                    ToolEvent::CallArgs { index, delta } => calls[index].1.push_str(&delta),
                    ToolEvent::CallEnd { .. } => {}
                }
            }
        };
        let end = loop {
            match next_event(&mut events, &mut deadlines).await {
                WorkerEvent::Event(event) => match event.event {
                    Some(token_event::Event::Tokens(chunk)) => {
                        let was_matched = pipeline.stop_matched();
                        let text = pipeline.push(chunk)?;
                        apply(route.push(text), &mut content, &mut calls);
                        if !was_matched && pipeline.stop_matched() {
                            ctx.cancel_worker().await;
                        }
                    }
                    Some(token_event::Event::Finished(mut finished)) => {
                        let tail = pipeline.finish()?;
                        apply(route.finish(tail), &mut content, &mut calls);
                        pipeline.apply_usage(&mut finished);
                        break classify_finished(finished, pipeline.stop_matched());
                    }
                    _ => {}
                },
                WorkerEvent::Closed => {
                    break StreamEnd::Failed(ApiError::worker_crashed(
                        "the worker stream ended without a result (worker crashed mid-request)",
                    ));
                }
                WorkerEvent::Rpc(status) => {
                    break StreamEnd::Failed(ApiError::from_worker_status(&status));
                }
                WorkerEvent::TimedOut(scope) => {
                    break StreamEnd::Failed(ctx.abort_for_timeout(scope).await);
                }
            }
        };

        match end {
            StreamEnd::Failed(err) => return Err(err),
            StreamEnd::Done {
                finished,
                finish_reason,
            } => {
                rounds.prompt_tokens_total += u64::from(finished.prompt_tokens);
                rounds.completion_tokens_total += u64::from(finished.completion_tokens);
                if rounds.should_execute(
                    finish_reason,
                    pipeline.stop_matched(),
                    &calls,
                    route.calls_completed(),
                ) {
                    ctx.settle_round(&finished);
                    content_all.push_str(&content);
                    if let Err(scope) = rounds.execute_round(content, calls, &deadlines).await {
                        return Err(ctx.abort_for_timeout(scope).await);
                    }
                    continue;
                }

                content_all.push_str(&content);
                ctx.record_ok(&finished);
                let tool_calls = (!calls.is_empty()).then(|| {
                    calls
                        .into_iter()
                        .map(|(name, arguments)| ResponseToolCall {
                            id: new_call_id(),
                            call_type: "function",
                            function: ResponseFunction { name, arguments },
                        })
                        .collect::<Vec<_>>()
                });
                let content = match &tool_calls {
                    Some(_) if content_all.is_empty() => None,
                    _ => Some(content_all),
                };
                return Ok(axum::Json(ChatCompletion {
                    id: ctx.completion_id.clone(),
                    object: "chat.completion",
                    created: ctx.created,
                    model: ctx.model.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: AssistantMessage {
                            role: "assistant",
                            content,
                            tool_calls,
                        },
                        logprobs: None,
                        finish_reason: route.adjust_reason(finish_reason),
                    }],
                    usage: rounds_usage(&rounds),
                }));
            }
        }
    }
}

/// Summed usage across the loop's rounds — the client pays for (and sees)
/// every round, like the reference APIs' server-side tools.
pub(crate) fn rounds_usage(rounds: &McpRounds) -> Usage {
    let prompt = u32::try_from(rounds.prompt_tokens_total).unwrap_or(u32::MAX);
    let completion = u32::try_from(rounds.completion_tokens_total).unwrap_or(u32::MAX);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt.saturating_add(completion),
    }
}

/// Streaming chat under the MCP loop: one SSE stream spans every round.
/// Content deltas stream live; tool-call deltas are withheld until the
/// round settles — an executed round's calls are internal plumbing the
/// client never sees, a terminal round's flush as ordinary `tool_calls`
/// deltas (OpenAI's accumulation contract doesn't constrain granularity).
fn stream_mcp_response(
    mut ctx: CompletionCtx,
    mut rounds: McpRounds,
    first: RoundStream,
    include_usage: bool,
    mut deadlines: Deadlines,
) -> Response {
    let usage_null: Option<Option<Usage>> = if include_usage { Some(None) } else { None };

    // Captured by value: the chunk closures must not borrow `ctx`, which
    // the round loop needs mutably.
    let completion_id = ctx.completion_id.clone();
    let created = ctx.created;
    let model = ctx.model.clone();
    let stream = async_stream::stream! {
        let chunk = move |choices: Vec<ChunkChoice>, usage: Option<Option<Usage>>| ChatCompletionChunk {
            id: completion_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices,
            usage,
        };
        let delta_chunk = |delta: Delta, usage: Option<Option<Usage>>| chunk(
            vec![ChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason: None,
            }],
            usage,
        );

        // Role preamble chunk, once for the whole loop.
        yield Ok::<SseEvent, Infallible>(sse_json(&delta_chunk(
            Delta { role: Some("assistant"), content: Some(String::new()), tool_calls: None },
            usage_null,
        )));

        let mut next_round = Some(first);
        'rounds: loop {
            let RoundStream { mut events, mut pipeline, parser, .. } = match next_round.take() {
                Some(round) => round,
                None => match rounds.submit_round(&mut ctx).await {
                    Ok(round) => round,
                    Err(err) => {
                        ctx.record_err(&err);
                        yield Ok(sse_json(&err.body()));
                        return;
                    }
                },
            };
            let mut route = ToolRoute::new(Some(parser));
            let mut content = String::new();
            let mut calls: Vec<(String, String)> = Vec::new();

            let end = loop {
                match next_event(&mut events, &mut deadlines).await {
                    WorkerEvent::Event(event) => match event.event {
                        Some(token_event::Event::Tokens(tc)) => {
                            let was_matched = pipeline.stop_matched();
                            let text = match pipeline.push(tc) {
                                Ok(text) => text,
                                Err(err) => {
                                    ctx.record_err(&err);
                                    yield Ok(sse_json(&err.body()));
                                    return;
                                }
                            };
                            if !was_matched && pipeline.stop_matched() {
                                ctx.cancel_worker().await;
                            }
                            for event in route.push(text) {
                                match event {
                                    ToolEvent::Content(text) => {
                                        content.push_str(&text);
                                        yield Ok(sse_json(&delta_chunk(Delta {
                                            content: Some(text),
                                            ..Delta::default()
                                        }, usage_null)));
                                    }
                                    ToolEvent::CallStart { name, .. } =>
                                        calls.push((name, String::new())),
                                    ToolEvent::CallArgs { index, delta } =>
                                        calls[index].1.push_str(&delta),
                                    ToolEvent::CallEnd { .. } => {}
                                }
                            }
                            continue;
                        }
                        Some(token_event::Event::Finished(mut finished)) => {
                            match pipeline.finish() {
                                Ok(tail) => {
                                    for event in route.finish(tail) {
                                        match event {
                                            ToolEvent::Content(text) => {
                                                content.push_str(&text);
                                                yield Ok(sse_json(&delta_chunk(Delta {
                                                    content: Some(text),
                                                    ..Delta::default()
                                                }, usage_null)));
                                            }
                                            ToolEvent::CallStart { name, .. } =>
                                                calls.push((name, String::new())),
                                            ToolEvent::CallArgs { index, delta } =>
                                                calls[index].1.push_str(&delta),
                                            ToolEvent::CallEnd { .. } => {}
                                        }
                                    }
                                }
                                Err(err) => {
                                    ctx.record_err(&err);
                                    yield Ok(sse_json(&err.body()));
                                    return;
                                }
                            }
                            pipeline.apply_usage(&mut finished);
                            break classify_finished(finished, pipeline.stop_matched());
                        }
                        _ => continue,
                    },
                    WorkerEvent::Closed => break StreamEnd::Failed(ApiError::worker_crashed(
                        "the worker stream ended without a result (worker crashed mid-request)",
                    )),
                    WorkerEvent::Rpc(status) =>
                        break StreamEnd::Failed(ApiError::from_worker_status(&status)),
                    WorkerEvent::TimedOut(scope) =>
                        break StreamEnd::Failed(ctx.abort_for_timeout(scope).await),
                }
            };

            match end {
                StreamEnd::Failed(err) => {
                    tracing::warn!(target: "kiln::chat", model = %ctx.model, code = err.code,
                        "streaming completion failed mid-stream: {}", err.message);
                    ctx.record_err(&err);
                    yield Ok(sse_json(&err.body()));
                    return;
                }
                StreamEnd::Done { finished, finish_reason } => {
                    rounds.prompt_tokens_total += u64::from(finished.prompt_tokens);
                    rounds.completion_tokens_total += u64::from(finished.completion_tokens);
                    if rounds.should_execute(
                        finish_reason,
                        pipeline.stop_matched(),
                        &calls,
                        route.calls_completed(),
                    ) {
                        ctx.settle_round(&finished);
                        if let Err(scope) = rounds.execute_round(content, calls, &deadlines).await {
                            let err = ctx.abort_for_timeout(scope).await;
                            ctx.record_err(&err);
                            yield Ok(sse_json(&err.body()));
                            return;
                        }
                        continue 'rounds;
                    }

                    // Terminal round: flush withheld calls as deltas, then
                    // finish exactly like the single-round stream.
                    for (index, (name, arguments)) in calls.iter().enumerate() {
                        yield Ok(sse_json(&delta_chunk(Delta {
                            tool_calls: Some(vec![DeltaToolCall {
                                index,
                                id: Some(new_call_id()),
                                call_type: Some("function"),
                                function: DeltaFunction {
                                    name: Some(name.clone()),
                                    arguments: String::new(),
                                },
                            }]),
                            ..Delta::default()
                        }, usage_null)));
                        if !arguments.is_empty() {
                            yield Ok(sse_json(&delta_chunk(Delta {
                                tool_calls: Some(vec![DeltaToolCall {
                                    index,
                                    id: None,
                                    call_type: None,
                                    function: DeltaFunction {
                                        name: None,
                                        arguments: arguments.clone(),
                                    },
                                }]),
                                ..Delta::default()
                            }, usage_null)));
                        }
                    }
                    yield Ok(sse_json(&chunk(
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            logprobs: None,
                            finish_reason: Some(route.adjust_reason(finish_reason)),
                        }],
                        usage_null,
                    )));
                    if include_usage {
                        yield Ok(sse_json(&chunk(Vec::new(), Some(Some(rounds_usage(&rounds))))));
                    }
                    yield Ok(SseEvent::default().data("[DONE]"));
                    ctx.record_ok(&finished);
                    return;
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

async fn collect_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
    tool_parser: Option<ToolCallParser>,
    mut deadlines: Deadlines,
) -> Result<axum::Json<ChatCompletion>, ApiError> {
    let mut route = ToolRoute::new(tool_parser);
    let mut content = String::new();
    // (name, arguments) per call, in emission order.
    let mut calls: Vec<(String, String)> = Vec::new();
    let apply = |events: Vec<ToolEvent>, content: &mut String, calls: &mut Vec<_>| {
        for event in events {
            match event {
                ToolEvent::Content(text) => content.push_str(&text),
                ToolEvent::CallStart { name, .. } => calls.push((name, String::new())),
                ToolEvent::CallArgs { index, delta } => calls[index].1.push_str(&delta),
                ToolEvent::CallEnd { .. } => {}
            }
        }
    };
    let end = loop {
        match next_event(&mut events, &mut deadlines).await {
            WorkerEvent::Event(event) => match event.event {
                Some(token_event::Event::Tokens(chunk)) => {
                    let was_matched = pipeline.stop_matched();
                    let text = pipeline.push(chunk)?;
                    apply(route.push(text), &mut content, &mut calls);
                    if !was_matched && pipeline.stop_matched() {
                        // Drain until Finished afterwards: prompt_tokens and
                        // timings come from it (completion_tokens is
                        // overridden by apply_usage — module docs).
                        ctx.cancel_worker().await;
                    }
                }
                Some(token_event::Event::Finished(mut finished)) => {
                    let tail = pipeline.finish()?;
                    apply(route.finish(tail), &mut content, &mut calls);
                    pipeline.apply_usage(&mut finished);
                    break classify_finished(finished, pipeline.stop_matched());
                }
                // Admitted / PrefixCacheHit are observability-only here.
                _ => {}
            },
            WorkerEvent::Closed => {
                break StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                ));
            }
            WorkerEvent::Rpc(status) => {
                break StreamEnd::Failed(ApiError::from_worker_status(&status));
            }
            // Partial output on the non-streaming path is discarded: the
            // client gets the 504, never a truncated completion.
            WorkerEvent::TimedOut(scope) => {
                break StreamEnd::Failed(ctx.abort_for_timeout(scope).await);
            }
        }
    };

    match end {
        StreamEnd::Failed(err) => Err(err),
        StreamEnd::Done {
            finished,
            finish_reason,
        } => {
            ctx.record_ok(&finished);
            let tool_calls = (!calls.is_empty()).then(|| {
                calls
                    .into_iter()
                    .map(|(name, arguments)| ResponseToolCall {
                        id: new_call_id(),
                        call_type: "function",
                        function: ResponseFunction { name, arguments },
                    })
                    .collect::<Vec<_>>()
            });
            // OpenAI shape: `content` is null on a tool-calls-only turn.
            let content = match &tool_calls {
                Some(_) if content.is_empty() => None,
                _ => Some(content),
            };
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
                        tool_calls,
                    },
                    logprobs: None,
                    finish_reason: route.adjust_reason(finish_reason),
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
    tool_parser: Option<ToolCallParser>,
    include_usage: bool,
    mut deadlines: Deadlines,
) -> Response {
    // With include_usage, data chunks carry `"usage": null` and a final
    // chunk carries the usage object (OpenAI semantics).
    let usage_null: Option<Option<Usage>> = if include_usage { Some(None) } else { None };
    let mut route = ToolRoute::new(tool_parser);

    let stream = async_stream::stream! {
        let chunk = |choices: Vec<ChunkChoice>, usage: Option<Option<Usage>>| ChatCompletionChunk {
            id: ctx.completion_id.clone(),
            object: "chat.completion.chunk",
            created: ctx.created,
            model: ctx.model.clone(),
            choices,
            usage,
        };
        let delta_chunk = |delta: Delta, usage: Option<Option<Usage>>| chunk(
            vec![ChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason: None,
            }],
            usage,
        );

        // Role preamble chunk.
        yield Ok::<SseEvent, Infallible>(sse_json(&delta_chunk(
            Delta { role: Some("assistant"), content: Some(String::new()), tool_calls: None },
            usage_null,
        )));

        loop {
            let end = match next_event(&mut events, &mut deadlines).await {
                WorkerEvent::Event(event) => match event.event {
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
                        for event in route.push(text) {
                            if let Some(delta) = tool_event_delta(event) {
                                yield Ok(sse_json(&delta_chunk(delta, usage_null)));
                            }
                        }
                        continue;
                    }
                    Some(token_event::Event::Finished(mut finished)) => {
                        match pipeline.finish() {
                            Ok(tail) => {
                                for event in route.finish(tail) {
                                    if let Some(delta) = tool_event_delta(event) {
                                        yield Ok(sse_json(&delta_chunk(delta, usage_null)));
                                    }
                                }
                            }
                            Err(err) => {
                                ctx.record_err(&err);
                                yield Ok(sse_json(&err.body()));
                                return;
                            }
                        }
                        pipeline.apply_usage(&mut finished);
                        classify_finished(finished, pipeline.stop_matched())
                    }
                    _ => continue,
                },
                WorkerEvent::Closed => StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                )),
                WorkerEvent::Rpc(status) => StreamEnd::Failed(ApiError::from_worker_status(&status)),
                // Deltas already sent stand as delivered; the stream ends
                // with the terminal timeout error event instead of [DONE].
                WorkerEvent::TimedOut(scope) => StreamEnd::Failed(ctx.abort_for_timeout(scope).await),
            };

            match end {
                StreamEnd::Done { finished, finish_reason } => {
                    yield Ok(sse_json(&chunk(
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            logprobs: None,
                            finish_reason: Some(route.adjust_reason(finish_reason)),
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

pub(crate) fn sse_json<T: serde::Serialize>(value: &T) -> SseEvent {
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

pub(crate) fn usage_of(finished: &Finished) -> Usage {
    Usage {
        prompt_tokens: finished.prompt_tokens,
        completion_tokens: finished.completion_tokens,
        total_tokens: finished.prompt_tokens + finished.completion_tokens,
    }
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
