//! `POST /v1/messages` (SPEC §8.1): the Anthropic Messages adapter over the
//! same pipeline as `/v1/chat/completions` — validate → render chat template
//! → tokenize → `Submit` → translate the worker's `TokenEvent` stream into an
//! Anthropic response (JSON or named-event SSE).
//!
//! Everything under the wire format is shared with the chat endpoint
//! (crate::chat): worker readiness gating, prompt encoding, the
//! [`TextPipeline`] (incremental detok + stop strings, including the
//! finish-reason precedence and usage semantics documented there), the
//! [`ToolRoute`] tool-call parsing, and terminal-event classification. What
//! this module owns is the *framing*: Anthropic content blocks instead of
//! OpenAI choices, and two adapter-only features —
//!
//! - **Thinking blocks**: models trained to reason in `<think>` tags
//!   (detected from the chat template) have those regions extracted by the
//!   streaming [`ThinkParser`] and surfaced as `thinking` content blocks,
//!   separate from `text`. On the OpenAI endpoint the same region is plain
//!   content — that difference is deliberate (SPEC §8.1 puts thinking
//!   passthrough on the Anthropic surface only).
//! - **`stop_sequence` attribution**: Anthropic reports *which* stop
//!   sequence fired. A gateway-side match knows it directly
//!   ([`TextPipeline::matched_stop`]); a tokenizer-owning worker reports it
//!   via `Finished.matched_stop` — which also carries the EOS token text on
//!   a natural stop, so the value counts only if it is one of the request's
//!   `stop_sequences`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    Finished, StoppingParams, SubmitRequest, TokenEvent, TokenIds, submit_request, token_event,
};
use kiln_tokenize::{ThinkEvent, ThinkParser, ToolCallParser, ToolEvent};
use tonic::Streaming;

use crate::anthropic::{
    BlockDelta, ContentBlock, ContentBlockDeltaEvent, ContentBlockStartEvent,
    ContentBlockStopEvent, MessageDelta, MessageDeltaEvent, MessageDeltaUsage, MessageStartEvent,
    MessageStopEvent, MessagesRequest, MessagesResponse, Usage, ValidatedMessages,
};
use crate::app::{AppState, RequestId};
use crate::chat::{
    CompletionCtx, MAX_BODY_BYTES, StreamEnd, TextPipeline, ToolRoute, admit_memory,
    classify_finished, encode_prompt, ready_entry,
};
use crate::error::ApiError;
use crate::registry::ModelEntry;

pub async fn messages(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .unwrap_or_else(|| RequestId(uuid::Uuid::now_v7().to_string()));
    let rate = request
        .extensions()
        .get::<crate::ratelimit::RateLimitHandle>()
        .cloned();

    let bytes = match axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return ApiError::invalid_request(format!("failed to read request body: {err}"))
                .into_anthropic_response();
        }
    };
    let parsed: MessagesRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid request JSON: {err}"))
                .into_anthropic_response();
        }
    };

    let model = parsed.model.clone();
    match handle(Arc::clone(&state), parsed, request_id, rate).await {
        Ok(response) => response,
        Err(err) => {
            state
                .metrics
                .messages_total
                .with_label_values(&[&model, err.outcome()])
                .inc();
            tracing::info!(target: "kiln::messages", model = %model, code = err.code,
                status = err.status.as_u16(), "messages request failed: {}", err.message);
            err.into_anthropic_response()
        }
    }
}

async fn handle(
    state: Arc<AppState>,
    request: MessagesRequest,
    request_id: RequestId,
    rate: Option<crate::ratelimit::RateLimitHandle>,
) -> Result<Response, ApiError> {
    let entry = ready_entry(&state, &request.model)?;
    admit_memory(&state, &entry)?;
    let validated = request.validate()?;

    let tool_parser = if validated.tools.is_empty() {
        None
    } else {
        let format = entry
            .template
            .as_ref()
            .and_then(|template| template.tool_call_format())
            .ok_or_else(|| {
                ApiError::invalid_request(format!(
                    "model '{}' has no known tool-call format; 'tools' is not supported for it",
                    entry.id
                ))
            })?;
        Some(ToolCallParser::new(format, &validated.tools))
    };
    // Thinking-block extraction only for models that emit the tags; a
    // non-thinking model's output never routes through the parser, so user
    // text mentioning `<think>` cannot be misclassified.
    let think = entry
        .template
        .as_ref()
        .is_some_and(|template| template.emits_think_tags())
        .then(ThinkParser::new);

    let prompt = render_prompt(&entry, &validated)?;
    let mut client = WorkerClient::new(entry.channel.clone());
    // BOS contract (kiln-tokenize crate docs): the rendered template already
    // contains BOS, so encode WITHOUT special tokens.
    let token_ids = encode_prompt(&entry, &mut client, prompt, false).await?;
    if token_ids.is_empty() {
        return Err(ApiError::invalid_request("rendered prompt is empty"));
    }
    let prompt_tokens = token_ids.len() as u32;

    let max_context_len = {
        let info = entry.info.read().await;
        info.as_ref().map(|info| info.max_context_len).unwrap_or(0)
    };
    let max_tokens = validated.effective_max_tokens(prompt_tokens, max_context_len)?;

    // Stop sequences: matched in the worker when it detokenizes (python),
    // in the gateway when it does (rust) — same split as chat.
    let pipeline = TextPipeline::for_entry(&entry, &validated.stop_sequences);
    let worker_stop_strings = match pipeline {
        TextPipeline::Passthrough => validated.stop_sequences.clone(),
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
        priority: validated.priority as i32,
        prefix_hint: 0,
        echo_prompt: false,
    };
    // tpm reservation (SPEC §8.3): worst case held until settle, unused
    // remainder refunded by record_ok — same flow as chat. The wrapper
    // converts a denial into the Anthropic rate_limit_error envelope.
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

    let requests_total = state.metrics.messages_total.clone();
    let ctx = CompletionCtx {
        state,
        model: entry.id.clone(),
        completion_id: format!("msg_{}", request_id.0.replace('-', "")),
        created: crate::chat::unix_now(),
        request_id: request_id.0.clone(),
        channel: entry.channel.clone(),
        requests_total,
        tpm,
    };
    let route = SegmentRoute {
        tools: ToolRoute::new(tool_parser),
        think,
    };
    if validated.stream {
        Ok(stream_response(
            ctx,
            events,
            pipeline,
            route,
            prompt_tokens,
            validated.stop_sequences,
        ))
    } else {
        collect_response(ctx, events, pipeline, route, validated.stop_sequences)
            .await
            .map(IntoResponse::into_response)
    }
}

fn render_prompt(entry: &ModelEntry, validated: &ValidatedMessages) -> Result<String, ApiError> {
    let template = entry.template.as_ref().ok_or_else(|| {
        ApiError::invalid_request(format!(
            "model '{}' has no chat template; messages are unavailable for it",
            entry.id
        ))
    })?;
    // `thinking: {"type": "disabled"}` renders the non-thinking prompt on
    // thinking-trained templates (Qwen3's `enable_thinking`); templates
    // without the variable ignore it.
    let extra: &[(&str, serde_json::Value)] = if validated.thinking_disabled {
        &[("enable_thinking", serde_json::Value::Bool(false))]
    } else {
        &[]
    };
    template
        .render_full(&validated.messages, true, &validated.tools, extra)
        .map_err(|err| ApiError::invalid_request(format!("chat template rejected messages: {err}")))
}

// ---------------------------------------------------------------------------
// Segment routing: pipeline text → tool-call parsing → think extraction
// ---------------------------------------------------------------------------

/// One increment of the response under Anthropic framing.
enum Segment {
    Thinking(String),
    Text(String),
    ToolStart { name: String },
    ToolArgs { delta: String },
    ToolEnd,
}

/// Fans pipeline text through the tool-call parser, then classifies the
/// content runs into thinking vs text via the model's think tags.
struct SegmentRoute {
    tools: ToolRoute,
    think: Option<ThinkParser>,
}

impl SegmentRoute {
    fn push(&mut self, text: String) -> Vec<Segment> {
        let events = self.tools.push(text);
        self.map(events, false)
    }

    fn finish(&mut self, tail: String) -> Vec<Segment> {
        let events = self.tools.finish(tail);
        self.map(events, true)
    }

    fn calls_completed(&self) -> usize {
        self.tools.calls_completed()
    }

    fn map(&mut self, events: Vec<ToolEvent>, at_end: bool) -> Vec<Segment> {
        let mut out = Vec::new();
        for event in events {
            match event {
                ToolEvent::Content(text) => match &mut self.think {
                    None => out.push(Segment::Text(text)),
                    Some(parser) => {
                        for piece in parser.push(&text) {
                            out.push(match piece {
                                ThinkEvent::Thinking(t) => Segment::Thinking(t),
                                ThinkEvent::Text(t) => Segment::Text(t),
                            });
                        }
                    }
                },
                ToolEvent::CallStart { name, .. } => {
                    // The think parser may be holding a partial tag or a
                    // whitespace run; flush it so held text lands before
                    // the tool block, preserving order.
                    self.flush_think(&mut out);
                    out.push(Segment::ToolStart { name });
                }
                ToolEvent::CallArgs { delta, .. } => out.push(Segment::ToolArgs { delta }),
                ToolEvent::CallEnd { .. } => out.push(Segment::ToolEnd),
            }
        }
        if at_end {
            self.flush_think(&mut out);
        }
        out
    }

    fn flush_think(&mut self, out: &mut Vec<Segment>) {
        if let Some(parser) = &mut self.think {
            for piece in parser.finish() {
                out.push(match piece {
                    ThinkEvent::Thinking(t) => Segment::Thinking(t),
                    ThinkEvent::Text(t) => Segment::Text(t),
                });
            }
        }
    }
}

/// Anthropic `stop_reason` from the normalized terminal state. Completed
/// tool calls upgrade a natural stop to `tool_use` (the OpenAI adapter's
/// `tool_calls` rule); a matched stop counts as `stop_sequence` only when it
/// is one of the *request's* sequences — `Finished.matched_stop` also
/// carries EOS token text on a natural worker stop (module docs).
fn anthropic_stop_reason(
    finish_reason: &'static str,
    calls_completed: usize,
    matched_stop: Option<String>,
    stop_sequences: &[String],
) -> (&'static str, Option<String>) {
    if finish_reason == "length" {
        return ("max_tokens", None);
    }
    if calls_completed > 0 {
        return ("tool_use", None);
    }
    match matched_stop {
        Some(matched) if stop_sequences.contains(&matched) => ("stop_sequence", Some(matched)),
        _ => ("end_turn", None),
    }
}

/// The matched stop string: the gateway matcher's hit (rust path), else the
/// worker-reported one (python path; empty = none).
fn matched_stop_of(pipeline: &TextPipeline, finished: &Finished) -> Option<String> {
    pipeline
        .matched_stop()
        .map(str::to_owned)
        .or_else(|| (!finished.matched_stop.is_empty()).then(|| finished.matched_stop.clone()))
}

fn new_tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::now_v7().simple())
}

fn usage_of(finished: &Finished) -> Usage {
    Usage {
        input_tokens: finished.prompt_tokens,
        output_tokens: finished.completion_tokens,
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

/// Accumulating counterpart of the SSE block state machine.
#[derive(Default)]
struct BlockAssembler {
    blocks: Vec<Block>,
}

enum Block {
    Thinking(String),
    Text(String),
    ToolUse { name: String, arguments: String },
}

impl BlockAssembler {
    fn push(&mut self, segment: Segment) {
        match segment {
            Segment::Thinking(text) => match self.blocks.last_mut() {
                Some(Block::Thinking(prev)) => prev.push_str(&text),
                _ => self.blocks.push(Block::Thinking(text)),
            },
            Segment::Text(text) => match self.blocks.last_mut() {
                Some(Block::Text(prev)) => prev.push_str(&text),
                _ => self.blocks.push(Block::Text(text)),
            },
            Segment::ToolStart { name } => self.blocks.push(Block::ToolUse {
                name,
                arguments: String::new(),
            }),
            Segment::ToolArgs { delta } => {
                if let Some(Block::ToolUse { arguments, .. }) = self.blocks.last_mut() {
                    arguments.push_str(&delta);
                }
            }
            Segment::ToolEnd => {}
        }
    }

    fn into_content(self, model: &str) -> Vec<ContentBlock> {
        let mut content = Vec::new();
        for block in self.blocks {
            match block {
                Block::Thinking(thinking) if !thinking.is_empty() => {
                    content.push(ContentBlock::Thinking {
                        thinking,
                        signature: "",
                    });
                }
                Block::Text(text) if !text.is_empty() => {
                    content.push(ContentBlock::Text { text });
                }
                Block::ToolUse { name, arguments } => {
                    // `input` is an object on this API; arguments that never
                    // became valid JSON (length truncation) cannot be
                    // represented — drop the block, the `max_tokens` stop
                    // reason tells the story. (Streaming shows the partial
                    // bytes instead; same divergence as the reference API.)
                    let input = if arguments.is_empty() {
                        Ok(serde_json::json!({}))
                    } else {
                        serde_json::from_str(&arguments)
                    };
                    match input {
                        Ok(input) => content.push(ContentBlock::ToolUse {
                            id: new_tool_use_id(),
                            name,
                            input,
                        }),
                        Err(err) => {
                            tracing::debug!(target: "kiln::messages", model = %model,
                                tool = %name, error = %err,
                                "dropping tool_use block with non-JSON arguments");
                        }
                    }
                }
                Block::Thinking(_) | Block::Text(_) => {}
            }
        }
        content
    }
}

async fn collect_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
    mut route: SegmentRoute,
    stop_sequences: Vec<String>,
) -> Result<axum::Json<MessagesResponse>, ApiError> {
    let mut assembler = BlockAssembler::default();
    let (end, matched) = loop {
        match events.message().await {
            Ok(Some(event)) => match event.event {
                Some(token_event::Event::Tokens(chunk)) => {
                    let was_matched = pipeline.stop_matched();
                    let text = pipeline.push(chunk)?;
                    for segment in route.push(text) {
                        assembler.push(segment);
                    }
                    if !was_matched && pipeline.stop_matched() {
                        ctx.cancel_worker().await;
                    }
                }
                Some(token_event::Event::Finished(mut finished)) => {
                    let tail = pipeline.finish()?;
                    for segment in route.finish(tail) {
                        assembler.push(segment);
                    }
                    pipeline.apply_usage(&mut finished);
                    let matched = matched_stop_of(&pipeline, &finished);
                    break (
                        classify_finished(finished, pipeline.stop_matched()),
                        matched,
                    );
                }
                // Admitted / PrefixCacheHit are observability-only here.
                _ => {}
            },
            Ok(None) => {
                break (
                    StreamEnd::Failed(ApiError::worker_crashed(
                        "the worker stream ended without a result (worker crashed mid-request)",
                    )),
                    None,
                );
            }
            Err(status) => {
                break (
                    StreamEnd::Failed(ApiError::from_worker_status(&status)),
                    None,
                );
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
            let (stop_reason, stop_sequence) = anthropic_stop_reason(
                finish_reason,
                route.calls_completed(),
                matched,
                &stop_sequences,
            );
            Ok(axum::Json(MessagesResponse {
                id: ctx.completion_id.clone(),
                response_type: "message",
                role: "assistant",
                model: ctx.model.clone(),
                content: assembler.into_content(&ctx.model),
                stop_reason: Some(stop_reason),
                stop_sequence,
                usage: usage_of(&finished),
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming (named-event SSE)
// ---------------------------------------------------------------------------

/// Anthropic SSE frames carry the event name both as the SSE `event:` field
/// and as the payload's `type` — set both.
fn sse_event<T: serde::Serialize>(name: &'static str, payload: &T) -> SseEvent {
    crate::chat::sse_json(payload).event(name)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenKind {
    Thinking,
    Text,
    Tool,
}

/// The content-block state machine: opens/closes indexed blocks as the
/// segment kind changes and frames deltas accordingly.
struct BlockStreamer {
    next_index: usize,
    open: Option<OpenKind>,
}

impl BlockStreamer {
    fn new() -> Self {
        Self {
            next_index: 0,
            open: None,
        }
    }

    fn index(&self) -> usize {
        self.next_index - 1
    }

    fn on_segment(&mut self, segment: Segment, out: &mut Vec<SseEvent>) {
        match segment {
            Segment::Thinking(text) => {
                if text.is_empty() {
                    return;
                }
                self.ensure_open(
                    OpenKind::Thinking,
                    ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: "",
                    },
                    out,
                );
                self.delta(BlockDelta::Thinking { thinking: text }, out);
            }
            Segment::Text(text) => {
                if text.is_empty() {
                    return;
                }
                self.ensure_open(
                    OpenKind::Text,
                    ContentBlock::Text {
                        text: String::new(),
                    },
                    out,
                );
                self.delta(BlockDelta::Text { text }, out);
            }
            Segment::ToolStart { name } => {
                self.close(out);
                self.start(
                    OpenKind::Tool,
                    ContentBlock::ToolUse {
                        id: new_tool_use_id(),
                        name,
                        input: serde_json::json!({}),
                    },
                    out,
                );
            }
            Segment::ToolArgs { delta } => {
                if self.open == Some(OpenKind::Tool) && !delta.is_empty() {
                    self.delta(
                        BlockDelta::InputJson {
                            partial_json: delta,
                        },
                        out,
                    );
                }
            }
            Segment::ToolEnd => self.close(out),
        }
    }

    fn ensure_open(&mut self, kind: OpenKind, start: ContentBlock, out: &mut Vec<SseEvent>) {
        if self.open != Some(kind) {
            self.close(out);
            self.start(kind, start, out);
        }
    }

    fn start(&mut self, kind: OpenKind, content_block: ContentBlock, out: &mut Vec<SseEvent>) {
        out.push(sse_event(
            "content_block_start",
            &ContentBlockStartEvent {
                event_type: "content_block_start",
                index: self.next_index,
                content_block,
            },
        ));
        self.next_index += 1;
        self.open = Some(kind);
    }

    fn delta(&self, delta: BlockDelta, out: &mut Vec<SseEvent>) {
        out.push(sse_event(
            "content_block_delta",
            &ContentBlockDeltaEvent {
                event_type: "content_block_delta",
                index: self.index(),
                delta,
            },
        ));
    }

    fn close(&mut self, out: &mut Vec<SseEvent>) {
        if self.open.take().is_some() {
            out.push(sse_event(
                "content_block_stop",
                &ContentBlockStopEvent {
                    event_type: "content_block_stop",
                    index: self.index(),
                },
            ));
        }
    }
}

fn stream_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
    mut route: SegmentRoute,
    prompt_tokens: u32,
    stop_sequences: Vec<String>,
) -> Response {
    let stream = async_stream::stream! {
        // message_start carries the message skeleton; deltas fill it in.
        let skeleton = MessagesResponse {
            id: ctx.completion_id.clone(),
            response_type: "message",
            role: "assistant",
            model: ctx.model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: Usage { input_tokens: prompt_tokens, output_tokens: 0 },
        };
        yield Ok::<SseEvent, Infallible>(sse_event("message_start", &MessageStartEvent {
            event_type: "message_start",
            message: &skeleton,
        }));

        let mut streamer = BlockStreamer::new();
        loop {
            let (end, matched) = match events.message().await {
                Ok(Some(event)) => match event.event {
                    Some(token_event::Event::Tokens(chunk)) => {
                        let was_matched = pipeline.stop_matched();
                        let text = match pipeline.push(chunk) {
                            Ok(text) => text,
                            Err(err) => {
                                // Detok failure is a gateway bug; surface it
                                // as the terminal error event.
                                ctx.record_err(&err);
                                yield Ok(sse_event("error", &err.anthropic_body()));
                                return;
                            }
                        };
                        if !was_matched && pipeline.stop_matched() {
                            ctx.cancel_worker().await;
                        }
                        let mut frames = Vec::new();
                        for segment in route.push(text) {
                            streamer.on_segment(segment, &mut frames);
                        }
                        for frame in frames {
                            yield Ok(frame);
                        }
                        continue;
                    }
                    Some(token_event::Event::Finished(mut finished)) => {
                        match pipeline.finish() {
                            Ok(tail) => {
                                let mut frames = Vec::new();
                                for segment in route.finish(tail) {
                                    streamer.on_segment(segment, &mut frames);
                                }
                                streamer.close(&mut frames);
                                for frame in frames {
                                    yield Ok(frame);
                                }
                            }
                            Err(err) => {
                                ctx.record_err(&err);
                                yield Ok(sse_event("error", &err.anthropic_body()));
                                return;
                            }
                        }
                        pipeline.apply_usage(&mut finished);
                        let matched = matched_stop_of(&pipeline, &finished);
                        (classify_finished(finished, pipeline.stop_matched()), matched)
                    }
                    _ => continue,
                },
                Ok(None) => (StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                )), None),
                Err(status) => (StreamEnd::Failed(ApiError::from_worker_status(&status)), None),
            };

            match end {
                StreamEnd::Done { finished, finish_reason } => {
                    let (stop_reason, stop_sequence) = anthropic_stop_reason(
                        finish_reason,
                        route.calls_completed(),
                        matched,
                        &stop_sequences,
                    );
                    yield Ok(sse_event("message_delta", &MessageDeltaEvent {
                        event_type: "message_delta",
                        delta: MessageDelta { stop_reason, stop_sequence },
                        usage: MessageDeltaUsage {
                            output_tokens: finished.completion_tokens,
                        },
                    }));
                    yield Ok(sse_event("message_stop", &MessageStopEvent {
                        event_type: "message_stop",
                    }));
                    ctx.record_ok(&finished);
                }
                StreamEnd::Failed(err) => {
                    // Headers already went out as 200; surface the failure
                    // as Anthropic's terminal `error` event.
                    tracing::warn!(target: "kiln::messages", model = %ctx.model, code = err.code,
                        "streaming messages request failed mid-stream: {}", err.message);
                    ctx.record_err(&err);
                    yield Ok(sse_event("error", &err.anthropic_body()));
                }
            }
            return;
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
