//! OpenAI wire types (SPEC §8.1) — chat completions and legacy text
//! completions — and validation into the worker-protocol request shape.
//! Unknown top-level fields are ignored, like the reference implementation;
//! unsupported *features* are rejected with clear 400s rather than silently
//! dropped.

use kiln_proto::v1::SamplingParams;
use kiln_tokenize::ChatMessage;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;

/// Generated tokens cap when the client sends no max_tokens and the worker
/// has not reported a context length.
const FALLBACK_MAX_TOKENS: u32 = 1024;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<RequestMessage>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stop: Option<StopSpec>,
    pub seed: Option<i64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: bool,
    pub top_logprobs: Option<u32>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub response_format: Option<ResponseFormat>,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
}

/// `content` is a plain string or an array of typed parts; only `text`
/// parts are accepted in Phase 2 (no vision until Phase 11).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

/// The request after validation: template-ready messages plus the proto
/// sampling/stopping parameters.
#[derive(Debug)]
pub struct ValidatedChat {
    pub messages: Vec<ChatMessage>,
    pub sampling: SamplingParams,
    /// None = derive from context length at submit time.
    pub max_tokens: Option<u32>,
    pub stop_strings: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

impl ChatCompletionRequest {
    pub fn validate(&self) -> Result<ValidatedChat, ApiError> {
        let invalid = ApiError::invalid_request;

        if self.messages.is_empty() {
            return Err(invalid("'messages' must be a non-empty array"));
        }
        if let Some(n) = self.n
            && n != 1
        {
            return Err(invalid("'n' > 1 is not supported"));
        }
        if self.logprobs || self.top_logprobs.is_some() {
            return Err(invalid(
                "'logprobs' is not supported by this model's worker",
            ));
        }
        if self.tools.is_some() || self.tool_choice.is_some() {
            return Err(invalid("tool calling is not supported yet (Kiln Phase 7)"));
        }
        if let Some(format) = &self.response_format
            && format.format_type != "text"
        {
            return Err(invalid(
                "'response_format' other than 'text' is not supported yet (Kiln Phase 7)",
            ));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(invalid("'stream_options' requires 'stream': true"));
        }

        let (sampling, stop_strings) = validate_sampling(
            self.temperature,
            self.top_p,
            self.presence_penalty,
            self.frequency_penalty,
            self.seed,
            self.stop.as_ref(),
        )?;

        let max_tokens = match self.max_completion_tokens.or(self.max_tokens) {
            Some(0) => return Err(invalid("'max_completion_tokens' must be >= 1")),
            other => other,
        };

        let mut messages = Vec::with_capacity(self.messages.len());
        for message in &self.messages {
            let role = match message.role.as_str() {
                // OpenAI treats `developer` as the successor of `system`.
                "developer" => "system".to_string(),
                "system" | "user" | "assistant" => message.role.clone(),
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "unsupported message role '{other}' (tool messages arrive with Kiln Phase 7)"
                    )));
                }
            };
            let content = match &message.content {
                None => String::new(),
                Some(MessageContent::Text(text)) => text.clone(),
                Some(MessageContent::Parts(parts)) => {
                    let mut combined = String::new();
                    for part in parts {
                        if part.part_type != "text" {
                            return Err(ApiError::invalid_request(format!(
                                "unsupported content part type '{}'",
                                part.part_type
                            )));
                        }
                        combined.push_str(part.text.as_deref().unwrap_or(""));
                    }
                    combined
                }
            };
            messages.push(ChatMessage { role, content });
        }

        Ok(ValidatedChat {
            messages,
            sampling,
            max_tokens,
            stop_strings,
            stream: self.stream,
            include_usage: self
                .stream_options
                .as_ref()
                .is_some_and(|o| o.include_usage),
        })
    }
}

/// Sampling/stop validation shared by chat and text completions — the two
/// endpoints accept identical ranges, so one place owns them.
fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<i64>,
    stop: Option<&StopSpec>,
) -> Result<(SamplingParams, Vec<String>), ApiError> {
    let invalid = ApiError::invalid_request;

    let temperature = temperature.unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(invalid("'temperature' must be in [0, 2]"));
    }
    let top_p = top_p.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&top_p) {
        return Err(invalid("'top_p' must be in (0, 1]"));
    }
    let presence_penalty = presence_penalty.unwrap_or(0.0);
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    if !(-2.0..=2.0).contains(&presence_penalty) || !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(invalid(
            "'presence_penalty' and 'frequency_penalty' must be in [-2, 2]",
        ));
    }
    let seed = match seed {
        None => 0, // proto: 0 = worker chooses
        Some(s) if s >= 0 => s as u64,
        Some(_) => return Err(invalid("'seed' must be non-negative")),
    };

    let stop_strings = match stop {
        None => Vec::new(),
        Some(StopSpec::One(s)) => vec![s.clone()],
        Some(StopSpec::Many(list)) => {
            if list.len() > 4 {
                return Err(invalid("'stop' supports at most 4 sequences"));
            }
            list.clone()
        }
    };

    Ok((
        SamplingParams {
            temperature,
            top_p,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 0.0,
            frequency_penalty,
            presence_penalty,
            repetition_window: 0,
            seed,
            logprobs_top_n: 0,
        },
        stop_strings,
    ))
}

impl ValidatedChat {
    /// Effective max_tokens: explicit client value, else fill the remaining
    /// context, else a conservative fallback when the context is unknown.
    pub fn effective_max_tokens(
        &self,
        prompt_tokens: u32,
        max_context_len: u32,
    ) -> Result<u32, ApiError> {
        if max_context_len > 0 && prompt_tokens >= max_context_len {
            return Err(ApiError::context_length_exceeded(format!(
                "prompt is {prompt_tokens} tokens but the model's context is {max_context_len}"
            )));
        }
        Ok(match self.max_tokens {
            Some(requested) => requested,
            None if max_context_len > 0 => max_context_len - prompt_tokens,
            None => FALLBACK_MAX_TOKENS,
        })
    }
}

// ---------------------------------------------------------------------------
// Legacy text completions (`POST /v1/completions`, SPEC §8.1)
// ---------------------------------------------------------------------------

/// OpenAI's documented default when the client sends no `max_tokens` on the
/// legacy completions endpoint (unlike chat, which fills the context).
const DEFAULT_COMPLETION_MAX_TOKENS: u32 = 16;

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Option<PromptSpec>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<StopSpec>,
    pub seed: Option<i64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub n: Option<u32>,
    pub best_of: Option<u32>,
    /// Integer on this endpoint (top-N logprobs), not the chat bool.
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub echo: bool,
    pub suffix: Option<String>,
    pub user: Option<String>,
}

/// The `prompt` forms OpenAI accepts. Token-id forms are parsed so they can
/// be *rejected by name* in [`CompletionRequest::validate`] instead of
/// surfacing as an opaque serde untagged-enum error.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptSpec {
    Text(String),
    Many(Vec<String>),
    Tokens(Vec<i64>),
    ManyTokens(Vec<Vec<i64>>),
}

/// The completions request after validation: the raw prompt (no chat
/// template) plus the proto sampling/stopping parameters.
#[derive(Debug)]
pub struct ValidatedCompletion {
    pub prompt: String,
    pub sampling: SamplingParams,
    /// None = OpenAI's legacy default of 16, applied at submit time.
    pub max_tokens: Option<u32>,
    pub stop_strings: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

impl CompletionRequest {
    pub fn validate(&self) -> Result<ValidatedCompletion, ApiError> {
        let invalid = ApiError::invalid_request;

        let prompt = match &self.prompt {
            None => return Err(invalid("'prompt' is required")),
            Some(PromptSpec::Text(text)) => text.clone(),
            Some(PromptSpec::Many(prompts)) => match prompts.as_slice() {
                [single] => single.clone(),
                _ => return Err(invalid("'prompt' arrays must contain exactly one string")),
            },
            Some(PromptSpec::Tokens(_)) | Some(PromptSpec::ManyTokens(_)) => {
                return Err(invalid("token-id prompts are not supported; send a string"));
            }
        };
        if prompt.is_empty() {
            return Err(invalid("'prompt' must be non-empty"));
        }
        if let Some(n) = self.n
            && n != 1
        {
            return Err(invalid("'n' > 1 is not supported"));
        }
        if let Some(best_of) = self.best_of
            && best_of != 1
        {
            return Err(invalid("'best_of' > 1 is not supported"));
        }
        if self.logprobs.is_some() {
            return Err(invalid(
                "'logprobs' is not supported by this model's worker",
            ));
        }
        if self.echo {
            return Err(invalid("'echo' is not supported"));
        }
        if self.suffix.is_some() {
            return Err(invalid("'suffix' is not supported"));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(invalid("'stream_options' requires 'stream': true"));
        }
        let max_tokens = match self.max_tokens {
            Some(0) => return Err(invalid("'max_tokens' must be >= 1")),
            other => other,
        };

        let (sampling, stop_strings) = validate_sampling(
            self.temperature,
            self.top_p,
            self.presence_penalty,
            self.frequency_penalty,
            self.seed,
            self.stop.as_ref(),
        )?;

        Ok(ValidatedCompletion {
            prompt,
            sampling,
            max_tokens,
            stop_strings,
            stream: self.stream,
            include_usage: self
                .stream_options
                .as_ref()
                .is_some_and(|o| o.include_usage),
        })
    }
}

impl ValidatedCompletion {
    /// Effective max_tokens: explicit client value, else OpenAI's legacy
    /// default of 16. Prompt overflow is the same clean 400 as chat.
    pub fn effective_max_tokens(
        &self,
        prompt_tokens: u32,
        max_context_len: u32,
    ) -> Result<u32, ApiError> {
        if max_context_len > 0 && prompt_tokens >= max_context_len {
            return Err(ApiError::context_length_exceeded(format!(
                "prompt is {prompt_tokens} tokens but the model's context is {max_context_len}"
            )));
        }
        Ok(self.max_tokens.unwrap_or(DEFAULT_COMPLETION_MAX_TOKENS))
    }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub logprobs: Option<()>,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub logprobs: Option<()>,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Omitted unless `stream_options.include_usage`: then `null` on data
    /// chunks and an object on the final usage chunk (OpenAI semantics).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}

/// Choice for both the non-streaming completion and its SSE chunks (OpenAI
/// uses the same shape; chunks carry `finish_reason: null` until the end).
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<()>,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct TextCompletion {
    pub id: String,
    pub object: &'static str, // "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    /// Omitted on data chunks of a stream without `include_usage` (chunk
    /// reuse); always present on the non-streaming response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str, // "model"
    pub created: u64,
    pub owned_by: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("request parses")
    }

    #[test]
    fn minimal_request_validates() {
        let req = request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let v = req.validate().expect("valid");
        assert_eq!(v.messages.len(), 1);
        assert_eq!(v.sampling.temperature, 1.0);
        assert_eq!(v.sampling.seed, 0);
        assert!(!v.stream);
        assert_eq!(v.max_tokens, None);
    }

    #[test]
    fn developer_role_maps_to_system_and_parts_concatenate() {
        let req = request(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ]},
            ],
        }));
        let v = req.validate().expect("valid");
        assert_eq!(v.messages[0].role, "system");
        assert_eq!(v.messages[1].content, "ab");
    }

    #[test]
    fn rejects_unsupported_features() {
        for (body, needle) in [
            (
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2}),
                "'n' > 1",
            ),
            (
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "logprobs": true}),
                "logprobs",
            ),
            (
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": []}),
                "tool calling",
            ),
            (
                serde_json::json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}),
                "unsupported message role",
            ),
            (
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 9.0}),
                "'temperature'",
            ),
            (
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema"}}),
                "response_format",
            ),
            (
                serde_json::json!({"model": "m", "messages": []}),
                "non-empty",
            ),
        ] {
            let err = request(body).validate().expect_err("must reject");
            assert!(err.message.contains(needle), "{} !~ {needle}", err.message);
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn effective_max_tokens_paths() {
        let v = request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
        }))
        .validate()
        .expect("valid");
        // Remaining context when known.
        assert_eq!(v.effective_max_tokens(100, 1000).expect("fits"), 900);
        // Fallback when unknown.
        assert_eq!(
            v.effective_max_tokens(100, 0).expect("fits"),
            FALLBACK_MAX_TOKENS
        );
        // Overflow is a clean 400.
        let err = v.effective_max_tokens(1000, 1000).expect_err("overflow");
        assert_eq!(err.code, "context_length_exceeded");
    }

    fn completion_request(body: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(body).expect("request parses")
    }

    #[test]
    fn minimal_completion_request_validates() {
        let req = completion_request(serde_json::json!({"model": "m", "prompt": "Once"}));
        let v = req.validate().expect("valid");
        assert_eq!(v.prompt, "Once");
        assert_eq!(v.max_tokens, None);
        assert_eq!(v.sampling.temperature, 1.0);
        assert!(!v.stream);
        // OpenAI legacy default applies at submit time.
        assert_eq!(v.effective_max_tokens(10, 1000).expect("fits"), 16);
        assert_eq!(v.effective_max_tokens(10, 0).expect("fits"), 16);
    }

    #[test]
    fn completion_prompt_forms() {
        let v = completion_request(serde_json::json!({"model": "m", "prompt": ["one"]}))
            .validate()
            .expect("single-element array is valid");
        assert_eq!(v.prompt, "one");

        for (body, needle) in [
            (serde_json::json!({"model": "m"}), "'prompt' is required"),
            (serde_json::json!({"model": "m", "prompt": ""}), "non-empty"),
            (
                serde_json::json!({"model": "m", "prompt": ["a", "b"]}),
                "exactly one string",
            ),
            (
                serde_json::json!({"model": "m", "prompt": [1, 2, 3]}),
                "token-id prompts",
            ),
            (
                serde_json::json!({"model": "m", "prompt": [[1, 2], [3]]}),
                "token-id prompts",
            ),
        ] {
            let err = completion_request(body)
                .validate()
                .expect_err("must reject");
            assert!(err.message.contains(needle), "{} !~ {needle}", err.message);
        }
    }

    #[test]
    fn completion_rejects_unsupported_features() {
        for (extra, needle) in [
            (serde_json::json!({"n": 2}), "'n' > 1"),
            (serde_json::json!({"best_of": 3}), "'best_of' > 1"),
            (serde_json::json!({"logprobs": 5}), "logprobs"),
            (serde_json::json!({"echo": true}), "'echo'"),
            (serde_json::json!({"suffix": "end"}), "'suffix'"),
            (serde_json::json!({"max_tokens": 0}), "'max_tokens'"),
            (
                serde_json::json!({"stream_options": {"include_usage": true}}),
                "'stream_options' requires",
            ),
            (serde_json::json!({"temperature": 9.0}), "'temperature'"),
        ] {
            let mut body = serde_json::json!({"model": "m", "prompt": "x"});
            for (k, v) in extra.as_object().expect("object") {
                body[k] = v.clone();
            }
            let err = completion_request(body.clone())
                .validate()
                .expect_err("must reject");
            assert!(
                err.message.contains(needle),
                "{}: {} !~ {needle}",
                body,
                err.message
            );
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn completion_context_overflow_is_a_clean_400() {
        let v = completion_request(serde_json::json!({"model": "m", "prompt": "x"}))
            .validate()
            .expect("valid");
        let err = v.effective_max_tokens(1000, 1000).expect_err("overflow");
        assert_eq!(err.code, "context_length_exceeded");
    }

    #[test]
    fn text_completion_usage_serialization() {
        // Non-streaming: usage always an object.
        let full = TextCompletion {
            id: "cmpl-x".into(),
            object: "text_completion",
            created: 1,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "hi".into(),
                logprobs: None,
                finish_reason: Some("stop"),
            }],
            usage: Some(Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            })),
        };
        let json = serde_json::to_value(&full).expect("serializes");
        assert_eq!(json["usage"]["total_tokens"], 3);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        // Stream data chunk without include_usage: field omitted entirely.
        let chunk = TextCompletion {
            usage: None,
            ..full
        };
        let json = serde_json::to_value(&chunk).expect("serializes");
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn stream_usage_chunk_serialization() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-x".into(),
            object: "chat.completion.chunk",
            created: 1,
            model: "m".into(),
            choices: vec![],
            usage: Some(None),
        };
        let json = serde_json::to_value(&chunk).expect("serializes");
        assert!(json["usage"].is_null());
        let chunk = ChatCompletionChunk {
            usage: None,
            ..chunk
        };
        let json = serde_json::to_value(&chunk).expect("serializes");
        assert!(json.get("usage").is_none());
    }
}
