//! Anthropic Messages API wire types (SPEC §8.1) and validation into the
//! same internal request shape the OpenAI adapter uses — `/v1/messages` is
//! a second adapter over the shared chat pipeline (templating, incremental
//! detok, stop strings, tool-call parsing), not a parallel serving path.
//! Unknown top-level fields are ignored; unsupported *features* are
//! rejected with clear 400s rather than silently dropped.

use kiln_proto::v1::SamplingParams;
use kiln_tokenize::{ChatMessage, MessageToolCall, MessageToolFunction};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    /// Required by the Anthropic API (no server-side default).
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<RequestMessage>,
    /// Top-level system prompt: a string or `[{type: "text", text}]` blocks.
    pub system: Option<SystemSpec>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    pub tools: Option<Vec<RequestTool>>,
    pub tool_choice: Option<serde_json::Value>,
    /// `{type: enabled|adaptive|disabled, ...}` — "disabled" renders the
    /// template with `enable_thinking=false` (thinking-trained templates
    /// honor it; others ignore the variable). Budgets are not enforceable
    /// on open models and are accepted-and-ignored.
    pub thinking: Option<ThinkingSpec>,
    /// Accepted and ignored (abuse-tracing hint in the reference API).
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<RequestContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum RequestContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    /// Assistant history: a prior tool call being replayed.
    #[serde(rename = "tool_use")]
    ToolUse {
        #[allow(dead_code)]
        id: Option<String>,
        name: String,
        input: serde_json::Value,
    },
    /// User turn: the result of a prior tool call. `tool_use_id` is
    /// accepted but unused — the supported templates key tool results on
    /// position, not id (same contract as the OpenAI adapter).
    #[serde(rename = "tool_result")]
    ToolResult {
        #[allow(dead_code)]
        tool_use_id: Option<String>,
        content: Option<ToolResultContent>,
        /// Accepted; the model sees the error text itself.
        #[serde(default)]
        #[allow(dead_code)]
        is_error: bool,
    },
    /// Assistant history: replayed thinking. Dropped — the supported
    /// thinking-trained templates strip reasoning from prior turns
    /// themselves, so the model never expects it back.
    #[serde(rename = "thinking")]
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[allow(dead_code)]
        data: Option<String>,
    },
    #[serde(rename = "image")]
    Image {},
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<RequestContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemSpec {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

/// Anthropic tool definition: flat `{name, description, input_schema}`
/// (server-tool types like `web_search_*` carry a versioned `type` and are
/// rejected — Kiln only executes client tools).
#[derive(Debug, Deserialize)]
pub struct RequestTool {
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingSpec {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[allow(dead_code)]
    pub budget_tokens: Option<u64>,
}

/// The request after validation: template-ready messages plus the proto
/// sampling parameters — the same internal shape `ValidatedChat` produces,
/// so the submit path is shared.
#[derive(Debug)]
pub struct ValidatedMessages {
    pub messages: Vec<ChatMessage>,
    pub sampling: SamplingParams,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
    /// Tool definitions converted to the OpenAI shape the chat templates
    /// and tool-call parsers consume.
    pub tools: Vec<serde_json::Value>,
    pub stream: bool,
    /// `thinking: {"type": "disabled"}` — render with `enable_thinking=false`.
    pub thinking_disabled: bool,
}

impl MessagesRequest {
    pub fn validate(&self) -> Result<ValidatedMessages, ApiError> {
        let invalid = ApiError::invalid_request;

        let max_tokens = match self.max_tokens {
            None => return Err(invalid("'max_tokens' is required")),
            Some(0) => return Err(invalid("'max_tokens' must be >= 1")),
            Some(n) => n,
        };
        if self.messages.is_empty() {
            return Err(invalid("'messages' must be a non-empty array"));
        }

        let temperature = self.temperature.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&temperature) {
            return Err(invalid("'temperature' must be in [0, 1]"));
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&top_p) {
            return Err(invalid("'top_p' must be in (0, 1]"));
        }
        let top_k = match self.top_k {
            None => 0,
            Some(k) if (0..=i64::from(u32::MAX)).contains(&k) => k as u32,
            Some(_) => return Err(invalid("'top_k' must be a non-negative integer")),
        };

        let thinking_disabled = match &self.thinking {
            None => false,
            Some(spec) => match spec.thinking_type.as_str() {
                // Enabled variants are the models' native behavior;
                // budgets are unenforceable and ignored.
                "enabled" | "adaptive" => false,
                "disabled" => true,
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "unsupported 'thinking.type' '{other}' \
                         (expected 'enabled', 'adaptive', or 'disabled')"
                    )));
                }
            },
        };

        let mut tools = match &self.tools {
            None => Vec::new(),
            Some(tools) => tools
                .iter()
                .map(convert_tool)
                .collect::<Result<Vec<_>, _>>()?,
        };
        match &self.tool_choice {
            None => {}
            Some(serde_json::Value::Object(choice)) => {
                if choice
                    .get("disable_parallel_tool_use")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    return Err(invalid(
                        "'tool_choice.disable_parallel_tool_use' is not supported \
                         (the gateway cannot bound how many calls a model emits)",
                    ));
                }
                match choice.get("type").and_then(|t| t.as_str()) {
                    Some("auto") => {}
                    Some("none") => tools.clear(),
                    Some(other @ ("any" | "tool")) => {
                        return Err(ApiError::invalid_request(format!(
                            "'tool_choice' type '{other}' is not supported \
                             (forced tool calls need grammar-coupled decoding)"
                        )));
                    }
                    other => {
                        return Err(ApiError::invalid_request(format!(
                            "unsupported 'tool_choice' type '{}'",
                            other.unwrap_or_default()
                        )));
                    }
                }
            }
            Some(_) => return Err(invalid("'tool_choice' must be an object")),
        }

        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            messages.push(ChatMessage::text("system", system_text(system)?));
        }
        for message in &self.messages {
            match message.role.as_str() {
                "user" => convert_user_message(message, &mut messages)?,
                "assistant" => messages.push(convert_assistant_message(message)?),
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "unsupported message role '{other}' (expected 'user' or 'assistant')"
                    )));
                }
            }
        }

        Ok(ValidatedMessages {
            messages,
            sampling: SamplingParams {
                temperature,
                top_p,
                top_k,
                min_p: 0.0,
                repetition_penalty: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_window: 0,
                seed: 0,
                logprobs_top_n: 0,
            },
            max_tokens,
            stop_sequences: self.stop_sequences.clone().unwrap_or_default(),
            tools,
            stream: self.stream,
            thinking_disabled,
        })
    }
}

impl ValidatedMessages {
    /// Prompt-overflow gate; the (required) client `max_tokens` passes
    /// through — the worker truncates at the context edge like every other
    /// endpoint.
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
        Ok(self.max_tokens)
    }
}

fn system_text(system: &SystemSpec) -> Result<String, ApiError> {
    match system {
        SystemSpec::Text(text) => Ok(text.clone()),
        SystemSpec::Blocks(blocks) => {
            let mut combined = String::new();
            for block in blocks {
                if block.block_type != "text" {
                    return Err(ApiError::invalid_request(format!(
                        "unsupported 'system' block type '{}' (only 'text')",
                        block.block_type
                    )));
                }
                combined.push_str(block.text.as_deref().unwrap_or(""));
            }
            Ok(combined)
        }
    }
}

/// Anthropic `{name, description, input_schema}` → the OpenAI function
/// shape the chat templates render and the parsers take hints from.
fn convert_tool(tool: &RequestTool) -> Result<serde_json::Value, ApiError> {
    if let Some(tool_type) = &tool.tool_type
        && tool_type != "custom"
    {
        return Err(ApiError::invalid_request(format!(
            "unsupported tool type '{tool_type}' (only client tools; \
             Anthropic server tools are not available)"
        )));
    }
    let name = tool
        .name
        .as_deref()
        .filter(|n| !n.is_empty())
        .ok_or_else(|| ApiError::invalid_request("each tool requires a non-empty 'name'"))?;
    let mut function = serde_json::Map::new();
    function.insert("name".into(), name.into());
    if let Some(description) = &tool.description {
        function.insert("description".into(), description.clone().into());
    }
    function.insert(
        "parameters".into(),
        tool.input_schema
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
    );
    Ok(serde_json::json!({"type": "function", "function": function}))
}

/// A user turn: text runs become a user message; `tool_result` blocks
/// become `tool` messages, order preserved (results conventionally lead).
fn convert_user_message(
    message: &RequestMessage,
    out: &mut Vec<ChatMessage>,
) -> Result<(), ApiError> {
    let blocks = match &message.content {
        MessageContent::Text(text) => {
            out.push(ChatMessage::text("user", text.clone()));
            return Ok(());
        }
        MessageContent::Blocks(blocks) => blocks,
    };
    let mut pending = String::new();
    for block in blocks {
        match block {
            RequestContentBlock::Text { text } => pending.push_str(text),
            RequestContentBlock::ToolResult { content, .. } => {
                if !pending.is_empty() {
                    out.push(ChatMessage::text("user", std::mem::take(&mut pending)));
                }
                out.push(ChatMessage::text("tool", tool_result_text(content)?));
            }
            RequestContentBlock::Image {} => {
                return Err(ApiError::invalid_request(
                    "image content is not supported (no vision models in v1)",
                ));
            }
            RequestContentBlock::ToolUse { .. } => {
                return Err(ApiError::invalid_request(
                    "'tool_use' blocks are only valid on assistant messages",
                ));
            }
            RequestContentBlock::Thinking { .. } | RequestContentBlock::RedactedThinking { .. } => {
                return Err(ApiError::invalid_request(
                    "thinking blocks are only valid on assistant messages",
                ));
            }
        }
    }
    if !pending.is_empty() || out.last().is_none_or(|m| m.role != "tool") {
        out.push(ChatMessage::text("user", pending));
    }
    Ok(())
}

fn convert_assistant_message(message: &RequestMessage) -> Result<ChatMessage, ApiError> {
    let blocks = match &message.content {
        MessageContent::Text(text) => return Ok(ChatMessage::text("assistant", text.clone())),
        MessageContent::Blocks(blocks) => blocks,
    };
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            RequestContentBlock::Text { text } => content.push_str(text),
            RequestContentBlock::ToolUse { name, input, .. } => {
                tool_calls.push(MessageToolCall {
                    call_type: "function".to_owned(),
                    function: MessageToolFunction {
                        name: name.clone(),
                        arguments: input.clone(),
                    },
                });
            }
            // Replayed reasoning: dropped — the templates strip prior-turn
            // thinking themselves (module docs).
            RequestContentBlock::Thinking { .. } | RequestContentBlock::RedactedThinking { .. } => {
            }
            RequestContentBlock::ToolResult { .. } => {
                return Err(ApiError::invalid_request(
                    "'tool_result' blocks are only valid on user messages",
                ));
            }
            RequestContentBlock::Image {} => {
                return Err(ApiError::invalid_request(
                    "image content is not supported (no vision models in v1)",
                ));
            }
        }
    }
    Ok(ChatMessage {
        role: "assistant".to_owned(),
        content,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    })
}

fn tool_result_text(content: &Option<ToolResultContent>) -> Result<String, ApiError> {
    match content {
        None => Ok(String::new()),
        Some(ToolResultContent::Text(text)) => Ok(text.clone()),
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut combined = String::new();
            for block in blocks {
                match block {
                    RequestContentBlock::Text { text } => combined.push_str(text),
                    _ => {
                        return Err(ApiError::invalid_request(
                            "'tool_result.content' blocks must be 'text'",
                        ));
                    }
                }
            }
            Ok(combined)
        }
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A response content block; also the `content_block` payload of a
/// streaming `content_block_start` (with empty text / `{}` input).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        /// Anthropic's replay-integrity signature; Kiln serves open models,
        /// so there is nothing to sign — always empty.
        signature: &'static str,
    },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str, // "message"
    pub role: &'static str, // "assistant"
    pub model: String,
    pub content: Vec<ContentBlock>,
    /// `null` only in the streaming `message_start` skeleton.
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

// ---------------------------------------------------------------------------
// Streaming events (`data` payloads; the SSE `event:` name matches `type`)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct MessageStartEvent<'a> {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "message_start"
    pub message: &'a MessagesResponse,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "content_block_start"
    pub index: usize,
    pub content_block: ContentBlock,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "content_block_delta"
    pub index: usize,
    pub delta: BlockDelta,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum BlockDelta {
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "content_block_stop"
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "message_delta"
    pub delta: MessageDelta,
    pub usage: MessageDeltaUsage,
}

#[derive(Debug, Serialize)]
pub struct MessageDelta {
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str, // "message_stop"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(body).expect("request parses")
    }

    #[test]
    fn minimal_request_validates() {
        let v = request(serde_json::json!({
            "model": "m",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .validate()
        .expect("valid");
        assert_eq!(v.max_tokens, 32);
        assert_eq!(v.messages.len(), 1);
        assert_eq!(v.messages[0].role, "user");
        assert_eq!(v.sampling.temperature, 1.0);
        assert_eq!(v.sampling.top_k, 0);
        assert!(!v.stream);
        assert!(!v.thinking_disabled);
    }

    #[test]
    fn system_prompt_prepends_a_system_message() {
        // String form and block form are equivalent.
        for system in [
            serde_json::json!("be brief"),
            serde_json::json!([{"type": "text", "text": "be brief"}]),
        ] {
            let v = request(serde_json::json!({
                "model": "m",
                "max_tokens": 8,
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .validate()
            .expect("valid");
            assert_eq!(v.messages[0].role, "system");
            assert_eq!(v.messages[0].content, "be brief");
            assert_eq!(v.messages[1].role, "user");
        }
    }

    #[test]
    fn tool_history_round_trips_into_template_shape() {
        let v = request(serde_json::json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me check", "signature": ""},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"},
                ]},
            ],
            "tools": [{"name": "get_weather", "description": "Weather.",
                       "input_schema": {"type": "object"}}],
        }))
        .validate()
        .expect("valid");

        // Anthropic tool shape converted to the OpenAI shape templates render.
        assert_eq!(
            v.tools[0],
            serde_json::json!({"type": "function", "function": {
                "name": "get_weather", "description": "Weather.",
                "parameters": {"type": "object"},
            }})
        );
        let assistant = &v.messages[1];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content, "Checking."); // thinking dropped
        let calls = assistant.tool_calls.as_ref().expect("kept");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({"city": "Paris"})
        );
        // tool_result became a tool-role message, no empty user message.
        assert_eq!(v.messages[2].role, "tool");
        assert_eq!(v.messages[2].content, "21C");
        assert_eq!(v.messages.len(), 3);
    }

    #[test]
    fn tool_choice_none_drops_tools_and_thinking_disabled_maps() {
        let v = request(serde_json::json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "f"}],
            "tool_choice": {"type": "none"},
            "thinking": {"type": "disabled"},
        }))
        .validate()
        .expect("valid");
        assert!(v.tools.is_empty());
        assert!(v.thinking_disabled);
    }

    #[test]
    fn rejects_unsupported_features() {
        for (body, needle) in [
            (
                serde_json::json!({"model": "m",
                    "messages": [{"role": "user", "content": "x"}]}),
                "'max_tokens' is required",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8, "messages": []}),
                "non-empty",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "system", "content": "x"}]}),
                "unsupported message role",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8, "temperature": 1.5,
                    "messages": [{"role": "user", "content": "x"}]}),
                "'temperature'",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8, "top_k": -1,
                    "messages": [{"role": "user", "content": "x"}]}),
                "'top_k'",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [{"name": "f"}], "tool_choice": {"type": "any"}}),
                "tool_choice",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [{"name": "f"}],
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}}),
                "disable_parallel_tool_use",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [{"type": "web_search_20260209", "name": "web_search"}]}),
                "unsupported tool type",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {}}]}]}),
                "image content",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": [
                        {"type": "tool_use", "id": "t", "name": "f", "input": {}}]}]}),
                "only valid on assistant",
            ),
            (
                serde_json::json!({"model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": {"type": "extended"}}),
                "thinking.type",
            ),
        ] {
            let err = request(body.clone()).validate().expect_err("must reject");
            assert!(
                err.message.contains(needle),
                "{body}: {} !~ {needle}",
                err.message
            );
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn response_and_event_wire_shapes() {
        // The `anthropic` SDK validates these shapes strictly (pydantic);
        // pin the field spellings.
        let response = MessagesResponse {
            id: "msg_1".into(),
            response_type: "message",
            role: "assistant",
            model: "m".into(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hm".into(),
                    signature: "",
                },
                ContentBlock::Text { text: "hi".into() },
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "f".into(),
                    input: serde_json::json!({"a": 1}),
                },
            ],
            stop_reason: Some("tool_use"),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 5,
            },
        };
        let json = serde_json::to_value(&response).expect("serializes");
        assert_eq!(json["type"], "message");
        assert_eq!(json["content"][0]["type"], "thinking");
        assert_eq!(json["content"][0]["thinking"], "hm");
        assert_eq!(json["content"][0]["signature"], "");
        assert_eq!(json["content"][1]["text"], "hi");
        assert_eq!(json["content"][2]["input"]["a"], 1);
        assert_eq!(json["stop_reason"], "tool_use");
        assert!(json["stop_sequence"].is_null());
        assert_eq!(json["usage"]["input_tokens"], 3);

        let delta = ContentBlockDeltaEvent {
            event_type: "content_block_delta",
            index: 1,
            delta: BlockDelta::InputJson {
                partial_json: "{\"a\":".into(),
            },
        };
        let json = serde_json::to_value(&delta).expect("serializes");
        assert_eq!(json["delta"]["type"], "input_json_delta");
        assert_eq!(json["delta"]["partial_json"], "{\"a\":");
    }
}
