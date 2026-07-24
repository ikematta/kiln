//! `POST /v1/completions` (SPEC §8.1) — the OpenAI legacy text-completions
//! endpoint: validate → tokenize the raw prompt (no chat template; special
//! tokens ON, the raw-prompt BOS contract) → `Submit` → translate the
//! worker's `TokenEvent` stream into a `text_completion` response (JSON or
//! SSE).
//!
//! Everything downstream of validation — worker-status gating, tokenization
//! ownership, the text pipeline, stop-string precedence, and usage
//! accounting — is the chat machinery ([`crate::chat`]); its module docs on
//! finish-reason precedence apply here verbatim. Only the request/response
//! wire shapes differ.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    StoppingParams, SubmitRequest, TokenEvent, TokenIds, submit_request, token_event,
};
use tonic::Streaming;

use crate::app::{AppState, RequestId};
use crate::chat::{
    CompletionCtx, MAX_BODY_BYTES, StreamEnd, TextPipeline, admit_memory, classify_finished,
    encode_prompt, ready_entry, sse_json, unix_now, usage_of,
};
use crate::error::ApiError;
use crate::openai::{CompletionChoice, CompletionRequest, TextCompletion, Usage};

pub async fn completions(
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

    // Manual body handling so malformed JSON still yields an OpenAI-shaped
    // error instead of axum's plain-text rejection (as in chat).
    let bytes = match axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return ApiError::invalid_request(format!("failed to read request body: {err}"))
                .into_response();
        }
    };
    let parsed: CompletionRequest = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid request JSON: {err}"))
                .into_response();
        }
    };

    let model = parsed.model.clone();
    match handle(Arc::clone(&state), parsed, request_id, rate).await {
        Ok(response) => response,
        Err(err) => {
            state
                .metrics
                .completions_total
                .with_label_values(&[&model, err.outcome()])
                .inc();
            tracing::info!(target: "kiln::completions", model = %model, code = err.code,
                status = err.status.as_u16(), "completion failed: {}", err.message);
            err.into_response()
        }
    }
}

async fn handle(
    state: Arc<AppState>,
    request: CompletionRequest,
    request_id: RequestId,
    rate: Option<crate::ratelimit::RateLimitHandle>,
) -> Result<Response, ApiError> {
    let entry = ready_entry(&state, &request.model)?;
    admit_memory(&state, &entry)?;
    let validated = request.validate()?;

    let mut client = WorkerClient::new(entry.channel.clone());
    // Raw-prompt BOS contract: no template supplies BOS here, so encode WITH
    // special tokens — exactly what mlx-lm does with a raw generate prompt.
    let token_ids = encode_prompt(&entry, &mut client, validated.prompt.clone(), true).await?;
    if token_ids.is_empty() {
        return Err(ApiError::invalid_request("prompt tokenized to no tokens"));
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
        priority: validated.priority as i32,
        prefix_hint: 0,
        echo_prompt: false,
    };
    // tpm reservation (SPEC §8.3): worst case held until settle, unused
    // remainder refunded by record_ok — same flow as chat.
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

    let requests_total = state.metrics.completions_total.clone();
    let ctx = CompletionCtx {
        state,
        model: entry.id.clone(),
        completion_id: format!("cmpl-{}", request_id.0.replace('-', "")),
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
            validated.include_usage,
        ))
    } else {
        collect_response(ctx, events, pipeline)
            .await
            .map(IntoResponse::into_response)
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

async fn collect_response(
    ctx: CompletionCtx,
    mut events: Streaming<TokenEvent>,
    mut pipeline: TextPipeline,
) -> Result<axum::Json<TextCompletion>, ApiError> {
    let mut text = String::new();
    let end = loop {
        match events.message().await {
            Ok(Some(event)) => match event.event {
                Some(token_event::Event::Tokens(chunk)) => {
                    let was_matched = pipeline.stop_matched();
                    text.push_str(&pipeline.push(chunk)?);
                    if !was_matched && pipeline.stop_matched() {
                        ctx.cancel_worker().await;
                    }
                }
                Some(token_event::Event::Finished(mut finished)) => {
                    text.push_str(&pipeline.finish()?);
                    pipeline.apply_usage(&mut finished);
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
            Ok(axum::Json(TextCompletion {
                id: ctx.completion_id.clone(),
                object: "text_completion",
                created: ctx.created,
                model: ctx.model.clone(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text,
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                }],
                usage: Some(Some(usage_of(&finished))),
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
    // chunk carries the usage object (OpenAI semantics, as in chat).
    let usage_null: Option<Option<Usage>> = if include_usage { Some(None) } else { None };

    let stream = async_stream::stream! {
        let chunk = |choices: Vec<CompletionChoice>, usage: Option<Option<Usage>>| TextCompletion {
            id: ctx.completion_id.clone(),
            object: "text_completion",
            created: ctx.created,
            model: ctx.model.clone(),
            choices,
            usage,
        };
        let text_choice = |text: String, finish_reason: Option<&'static str>| CompletionChoice {
            index: 0,
            text,
            logprobs: None,
            finish_reason,
        };

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
                                yield Ok::<SseEvent, Infallible>(sse_json(&err.body()));
                                return;
                            }
                        };
                        if !was_matched && pipeline.stop_matched() {
                            ctx.cancel_worker().await;
                        }
                        if !text.is_empty() {
                            yield Ok(sse_json(&chunk(vec![text_choice(text, None)], usage_null)));
                        }
                        continue;
                    }
                    Some(token_event::Event::Finished(mut finished)) => {
                        match pipeline.finish() {
                            Ok(tail) if !tail.is_empty() => {
                                yield Ok(sse_json(&chunk(vec![text_choice(tail, None)], usage_null)));
                            }
                            Ok(_) => {}
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
                Ok(None) => StreamEnd::Failed(ApiError::worker_crashed(
                    "the worker stream ended without a result (worker crashed mid-request)",
                )),
                Err(status) => StreamEnd::Failed(ApiError::from_worker_status(&status)),
            };

            match end {
                StreamEnd::Done { finished, finish_reason } => {
                    yield Ok(sse_json(&chunk(
                        vec![text_choice(String::new(), Some(finish_reason))],
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
                    tracing::warn!(target: "kiln::completions", model = %ctx.model, code = err.code,
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
