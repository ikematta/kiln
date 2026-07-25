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
    CompletionCtx, McpRounds, RoundStream, StreamEnd, TextPipeline, ToolRoute, admit_memory,
    classify_finished, encode_prompt, read_body, ready_entry,
};
use crate::error::ApiError;
use crate::registry::ModelEntry;
use crate::timeout::{Deadlines, WorkerEvent, next_event};

pub async fn messages(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    // Timeout budgets anchor at arrival (crate::timeout module docs).
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

    let bytes = match read_body(request).await {
        Ok(bytes) => bytes,
        Err(err) => return err.into_anthropic_response(),
    };
    let parsed: MessagesRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid request JSON: {err}"))
                .into_anthropic_response();
        }
    };

    let model = parsed.model.clone();
    match handle(Arc::clone(&state), parsed, request_id, rate, arrival).await {
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
    arrival: tokio::time::Instant,
) -> Result<Response, ApiError> {
    let deadlines = Deadlines::start(&state.timeouts, arrival);
    let entry = ready_entry(&state, &request.model)?;
    admit_memory(&state, &entry)?;
    let mut validated = request.validate()?;

    // MCP tools in scope (SPEC §8.4): same gates and merge as the chat
    // endpoint — crate::chat::handle. A non-empty set switches onto the
    // multi-round execution path below.
    let format = entry
        .template
        .as_ref()
        .and_then(|template| template.tool_call_format());
    let emits_think = entry
        .template
        .as_ref()
        .is_some_and(|template| template.emits_think_tags());
    let mut mcp_tools = if validated.tools_disabled || format.is_none() {
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
            client_max_tokens: Some(validated.max_tokens),
            stop_strings: validated.stop_sequences.clone(),
            priority: validated.priority as i32,
            rate,
            base_request_id: request_id.0.clone(),
            thinking_disabled: validated.thinking_disabled,
            mcp: mcp_tools,
            round: 0,
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
        };
        let mut ctx = CompletionCtx {
            state: Arc::clone(&state),
            model: entry.id.clone(),
            completion_id: format!("msg_{}", request_id.0.replace('-', "")),
            created: crate::chat::unix_now(),
            request_id: request_id.0.clone(),
            channel: entry.channel.clone(),
            requests_total: state.metrics.messages_total.clone(),
            tpm: None,
        };
        // Round 1 submits here so its failures surface as proper HTTP
        // errors, exactly like the single-round path.
        let first = rounds.submit_round(&mut ctx).await?;
        let stop_sequences = validated.stop_sequences.clone();
        return if validated.stream {
            Ok(stream_mcp_messages(
                ctx,
                rounds,
                first,
                emits_think,
                stop_sequences,
                deadlines,
            ))
        } else {
            collect_mcp_messages(ctx, rounds, first, emits_think, stop_sequences, deadlines)
                .await
                .map(IntoResponse::into_response)
        };
    }

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
    // Thinking-block extraction only for models that emit the tags; a
    // non-thinking model's output never routes through the parser, so user
    // text mentioning `<think>` cannot be misclassified.
    let think = emits_think.then(ThinkParser::new);

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
            deadlines,
        ))
    } else {
        collect_response(
            ctx,
            events,
            pipeline,
            route,
            validated.stop_sequences,
            deadlines,
        )
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
// MCP execution loop (SPEC §8.4) — Anthropic framing over crate::chat's
// round machinery. Thinking/text blocks accumulate across rounds; an
// executed round's tool_use blocks are internal plumbing the client never
// sees, a terminal round's surface as ordinary blocks.
// ---------------------------------------------------------------------------

/// Routes one round's segments: thinking/text flow through (to the
/// assembler or streamer) and into the assistant-history text; tool
/// segments are withheld until the round settles. History gets text
/// segments only — replayed thinking is dropped exactly as the request
/// validator drops client-replayed thinking blocks (crate::anthropic
/// module docs), so the loop feeds back what a faithful client would.
fn split_segments(
    segments: Vec<Segment>,
    text_content: &mut String,
    calls: &mut Vec<(String, String)>,
    tool_segments: &mut Vec<Segment>,
    mut live: impl FnMut(Segment),
) {
    for segment in segments {
        match segment {
            Segment::Thinking(_) => live(segment),
            Segment::Text(text) => {
                text_content.push_str(&text);
                live(Segment::Text(text));
            }
            Segment::ToolStart { name } => {
                calls.push((name.clone(), String::new()));
                tool_segments.push(Segment::ToolStart { name });
            }
            Segment::ToolArgs { delta } => {
                if let Some((_, arguments)) = calls.last_mut() {
                    arguments.push_str(&delta);
                }
                tool_segments.push(Segment::ToolArgs { delta });
            }
            Segment::ToolEnd => tool_segments.push(Segment::ToolEnd),
        }
    }
}

async fn collect_mcp_messages(
    mut ctx: CompletionCtx,
    mut rounds: McpRounds,
    first: RoundStream,
    emits_think: bool,
    stop_sequences: Vec<String>,
    mut deadlines: Deadlines,
) -> Result<axum::Json<MessagesResponse>, ApiError> {
    let mut assembler = BlockAssembler::default();
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
        let mut route = SegmentRoute {
            tools: ToolRoute::new(Some(parser)),
            think: emits_think.then(ThinkParser::new),
        };
        let mut text_content = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        let mut tool_segments: Vec<Segment> = Vec::new();

        let (end, matched) = loop {
            match next_event(&mut events, &mut deadlines).await {
                WorkerEvent::Event(event) => match event.event {
                    Some(token_event::Event::Tokens(chunk)) => {
                        let was_matched = pipeline.stop_matched();
                        let text = pipeline.push(chunk)?;
                        split_segments(
                            route.push(text),
                            &mut text_content,
                            &mut calls,
                            &mut tool_segments,
                            |segment| assembler.push(segment),
                        );
                        if !was_matched && pipeline.stop_matched() {
                            ctx.cancel_worker().await;
                        }
                    }
                    Some(token_event::Event::Finished(mut finished)) => {
                        let tail = pipeline.finish()?;
                        split_segments(
                            route.finish(tail),
                            &mut text_content,
                            &mut calls,
                            &mut tool_segments,
                            |segment| assembler.push(segment),
                        );
                        pipeline.apply_usage(&mut finished);
                        let matched = matched_stop_of(&pipeline, &finished);
                        break (
                            classify_finished(finished, pipeline.stop_matched()),
                            matched,
                        );
                    }
                    _ => {}
                },
                WorkerEvent::Closed => {
                    break (
                        StreamEnd::Failed(ApiError::worker_crashed(
                            "the worker stream ended without a result (worker crashed mid-request)",
                        )),
                        None,
                    );
                }
                WorkerEvent::Rpc(status) => {
                    break (
                        StreamEnd::Failed(ApiError::from_worker_status(&status)),
                        None,
                    );
                }
                WorkerEvent::TimedOut(scope) => {
                    break (StreamEnd::Failed(ctx.abort_for_timeout(scope).await), None);
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
                    if let Err(scope) = rounds.execute_round(text_content, calls, &deadlines).await
                    {
                        return Err(ctx.abort_for_timeout(scope).await);
                    }
                    continue;
                }

                for segment in tool_segments {
                    assembler.push(segment);
                }
                ctx.record_ok(&finished);
                let (stop_reason, stop_sequence) = anthropic_stop_reason(
                    finish_reason,
                    route.calls_completed(),
                    matched,
                    &stop_sequences,
                );
                let prompt = u32::try_from(rounds.prompt_tokens_total).unwrap_or(u32::MAX);
                let completion = u32::try_from(rounds.completion_tokens_total).unwrap_or(u32::MAX);
                return Ok(axum::Json(MessagesResponse {
                    id: ctx.completion_id.clone(),
                    response_type: "message",
                    role: "assistant",
                    model: ctx.model.clone(),
                    content: assembler.into_content(&ctx.model),
                    stop_reason: Some(stop_reason),
                    stop_sequence,
                    usage: Usage {
                        input_tokens: prompt,
                        output_tokens: completion,
                    },
                }));
            }
        }
    }
}

/// Streaming messages under the MCP loop: one named-event SSE stream spans
/// every round. `message_start` carries the first round's prompt tokens
/// (later rounds don't exist yet when it goes out); the terminal
/// `message_delta` usage reports the summed output tokens.
fn stream_mcp_messages(
    mut ctx: CompletionCtx,
    mut rounds: McpRounds,
    first: RoundStream,
    emits_think: bool,
    stop_sequences: Vec<String>,
    mut deadlines: Deadlines,
) -> Response {
    let stream = async_stream::stream! {
        let skeleton = MessagesResponse {
            id: ctx.completion_id.clone(),
            response_type: "message",
            role: "assistant",
            model: ctx.model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: Usage { input_tokens: first.prompt_tokens, output_tokens: 0 },
        };
        yield Ok::<SseEvent, Infallible>(sse_event("message_start", &MessageStartEvent {
            event_type: "message_start",
            message: &skeleton,
        }));

        let mut streamer = BlockStreamer::new();
        let mut next_round = Some(first);
        'rounds: loop {
            let RoundStream { mut events, mut pipeline, parser, .. } = match next_round.take() {
                Some(round) => round,
                None => match rounds.submit_round(&mut ctx).await {
                    Ok(round) => round,
                    Err(err) => {
                        ctx.record_err(&err);
                        yield Ok(sse_event("error", &err.anthropic_body()));
                        return;
                    }
                },
            };
            let mut route = SegmentRoute {
                tools: ToolRoute::new(Some(parser)),
                think: emits_think.then(ThinkParser::new),
            };
            let mut text_content = String::new();
            let mut calls: Vec<(String, String)> = Vec::new();
            let mut tool_segments: Vec<Segment> = Vec::new();

            let (end, matched) = loop {
                match next_event(&mut events, &mut deadlines).await {
                    WorkerEvent::Event(event) => match event.event {
                        Some(token_event::Event::Tokens(chunk)) => {
                            let was_matched = pipeline.stop_matched();
                            let text = match pipeline.push(chunk) {
                                Ok(text) => text,
                                Err(err) => {
                                    ctx.record_err(&err);
                                    yield Ok(sse_event("error", &err.anthropic_body()));
                                    return;
                                }
                            };
                            if !was_matched && pipeline.stop_matched() {
                                ctx.cancel_worker().await;
                            }
                            let mut frames = Vec::new();
                            split_segments(
                                route.push(text),
                                &mut text_content,
                                &mut calls,
                                &mut tool_segments,
                                |segment| streamer.on_segment(segment, &mut frames),
                            );
                            for frame in frames {
                                yield Ok(frame);
                            }
                            continue;
                        }
                        Some(token_event::Event::Finished(mut finished)) => {
                            match pipeline.finish() {
                                Ok(tail) => {
                                    let mut frames = Vec::new();
                                    split_segments(
                                        route.finish(tail),
                                        &mut text_content,
                                        &mut calls,
                                        &mut tool_segments,
                                        |segment| streamer.on_segment(segment, &mut frames),
                                    );
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
                            break (classify_finished(finished, pipeline.stop_matched()), matched);
                        }
                        _ => continue,
                    },
                    WorkerEvent::Closed => break (StreamEnd::Failed(ApiError::worker_crashed(
                        "the worker stream ended without a result (worker crashed mid-request)",
                    )), None),
                    WorkerEvent::Rpc(status) =>
                        break (StreamEnd::Failed(ApiError::from_worker_status(&status)), None),
                    WorkerEvent::TimedOut(scope) =>
                        break (StreamEnd::Failed(ctx.abort_for_timeout(scope).await), None),
                }
            };

            match end {
                StreamEnd::Failed(err) => {
                    tracing::warn!(target: "kiln::messages", model = %ctx.model, code = err.code,
                        "streaming messages request failed mid-stream: {}", err.message);
                    ctx.record_err(&err);
                    yield Ok(sse_event("error", &err.anthropic_body()));
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
                        if let Err(scope) = rounds.execute_round(text_content, calls, &deadlines).await {
                            let err = ctx.abort_for_timeout(scope).await;
                            ctx.record_err(&err);
                            yield Ok(sse_event("error", &err.anthropic_body()));
                            return;
                        }
                        continue 'rounds;
                    }

                    // Terminal round: flush withheld tool blocks, then
                    // finish exactly like the single-round stream.
                    let mut frames = Vec::new();
                    for segment in tool_segments {
                        streamer.on_segment(segment, &mut frames);
                    }
                    streamer.close(&mut frames);
                    for frame in frames {
                        yield Ok(frame);
                    }
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
                            output_tokens: u32::try_from(rounds.completion_tokens_total)
                                .unwrap_or(u32::MAX),
                        },
                    }));
                    yield Ok(sse_event("message_stop", &MessageStopEvent {
                        event_type: "message_stop",
                    }));
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
    mut deadlines: Deadlines,
) -> Result<axum::Json<MessagesResponse>, ApiError> {
    let mut assembler = BlockAssembler::default();
    let (end, matched) = loop {
        match next_event(&mut events, &mut deadlines).await {
            WorkerEvent::Event(event) => match event.event {
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
            WorkerEvent::Closed => {
                break (
                    StreamEnd::Failed(ApiError::worker_crashed(
                        "the worker stream ended without a result (worker crashed mid-request)",
                    )),
                    None,
                );
            }
            WorkerEvent::Rpc(status) => {
                break (
                    StreamEnd::Failed(ApiError::from_worker_status(&status)),
                    None,
                );
            }
            // Partial output on the non-streaming path is discarded.
            WorkerEvent::TimedOut(scope) => {
                break (StreamEnd::Failed(ctx.abort_for_timeout(scope).await), None);
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
    mut deadlines: Deadlines,
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
            let (end, matched) = match next_event(&mut events, &mut deadlines).await {
                WorkerEvent::Event(event) => match event.event {
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
                WorkerEvent::Closed => (StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                )), None),
                WorkerEvent::Rpc(status) => (StreamEnd::Failed(ApiError::from_worker_status(&status)), None),
                // Frames already sent stand; the stream ends with the
                // terminal `error` event instead of message_stop.
                WorkerEvent::TimedOut(scope) => (StreamEnd::Failed(ctx.abort_for_timeout(scope).await), None),
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
