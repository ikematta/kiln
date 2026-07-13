//! Hugging Face chat-template rendering via minijinja (SPEC §8.2).
//!
//! The gateway renders the model's own template (from `chat_template.jinja`
//! or `tokenizer_config.json`) into the exact prompt string, then hands it to
//! the worker for tokenization. Source selection order and the sha256 source
//! hash intentionally mirror the Python worker's `modelinfo.py`, so the hash
//! here is comparable with `WorkerInfo.chat_template_hash`.

use std::fmt::Write as _;
use std::path::Path;

use minijinja::{Environment, ErrorKind, UndefinedBehavior, Value};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("no chat template in {0} (chat_template.jinja or tokenizer_config.json)")]
    Missing(String),
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid tokenizer_config.json: {0}")]
    Config(String),
    #[error("chat template error: {0}")]
    Template(#[from] Box<minijinja::Error>),
}

impl From<minijinja::Error> for TemplateError {
    fn from(err: minijinja::Error) -> Self {
        Self::Template(Box::new(err))
    }
}

/// One conversation turn, in the shape HF templates expect.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Assistant tool-call history, for templates that re-serialize prior
    /// calls (Llama/Qwen check `'tool_calls' in message` / truthiness, so
    /// the key must be absent — not null — when there are none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MessageToolCall>>,
}

impl ChatMessage {
    /// A plain text turn (no tool calls).
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
        }
    }
}

/// One prior tool call in an assistant message, template-ready: `arguments`
/// is the parsed object (templates apply `| tojson` themselves), not the
/// OpenAI wire's JSON string.
#[derive(Debug, Clone, Serialize)]
pub struct MessageToolCall {
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: MessageToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A compiled chat template plus the special tokens it interpolates.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
    source_hash: String,
    bos_token: String,
    eos_token: String,
    tool_call_format: Option<crate::toolcall::ToolCallFormat>,
    emits_think_tags: bool,
}

const TEMPLATE_NAME: &str = "chat";

impl ChatTemplate {
    /// Loads the template from a local model directory: `chat_template.jinja`
    /// if present, else the `chat_template` key of `tokenizer_config.json`
    /// (string, or HF's list-of-named-templates form — `default` preferred).
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, TemplateError> {
        let dir = dir.as_ref();
        let read = |path: &Path| -> Result<Option<String>, TemplateError> {
            if !path.is_file() {
                return Ok(None);
            }
            std::fs::read_to_string(path)
                .map(Some)
                .map_err(|source| TemplateError::Io {
                    path: path.display().to_string(),
                    source,
                })
        };

        let tokenizer_config: Option<serde_json::Value> =
            match read(&dir.join("tokenizer_config.json"))? {
                Some(text) => Some(
                    serde_json::from_str(&text)
                        .map_err(|e| TemplateError::Config(e.to_string()))?,
                ),
                None => None,
            };

        let source = match read(&dir.join("chat_template.jinja"))? {
            Some(text) => text,
            None => tokenizer_config
                .as_ref()
                .and_then(|cfg| template_from_config(cfg.get("chat_template")))
                .ok_or_else(|| TemplateError::Missing(dir.display().to_string()))?,
        };

        let token = |key: &str| -> String {
            tokenizer_config
                .as_ref()
                .and_then(|cfg| special_token(cfg.get(key)))
                .unwrap_or_default()
        };
        Self::new(source, token("bos_token"), token("eos_token"))
    }

    /// Compiles a template from its source text and special tokens.
    pub fn new(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, TemplateError> {
        let source_hash = hex_sha256(source.as_bytes());
        let tool_call_format = crate::toolcall::ToolCallFormat::detect(&source);
        let emits_think_tags = crate::think::template_emits_think_tags(&source);
        let mut env = Environment::new();
        // HF templates probe optional variables (`tools`, `date_string`, ...)
        // with `is defined`; strict undefined would reject them.
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        // Python string/dict methods (`.strip()`, `.items()`, ...) used
        // throughout HF-ecosystem templates.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        // Override minijinja's builtin tojson: templates interpolate tool
        // definitions into the PROMPT with it, and the model-visible bytes
        // must match the reference (transformers) rendering — Python
        // json.dumps formatting, not serde_json's compact form.
        env.add_filter("tojson", tojson_python_style);
        env.add_template_owned(TEMPLATE_NAME, source)?;
        Ok(Self {
            env,
            source_hash,
            bos_token,
            eos_token,
            tool_call_format,
            emits_think_tags,
        })
    }

    /// The tool-call format this model family emits, detected from the
    /// template source (SPEC §8.2 "selected by model metadata"); `None`
    /// means the model has no known tool-call convention and tool requests
    /// must be rejected.
    pub fn tool_call_format(&self) -> Option<crate::toolcall::ToolCallFormat> {
        self.tool_call_format
    }

    /// Whether this model wraps its reasoning in `<think>` tags (detected
    /// from the template source, like [`Self::tool_call_format`]); drives
    /// thinking-block extraction on the Anthropic API surface (SPEC §8.1).
    pub fn emits_think_tags(&self) -> bool {
        self.emits_think_tags
    }

    /// Renders the conversation into the model's prompt string.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TemplateError> {
        self.render_with(messages, add_generation_prompt, &[])
    }

    /// [`Self::render`] with the request's tool definitions exposed to the
    /// template as the standard `tools` variable (HF convention: the full
    /// OpenAI-shape objects). An empty slice renders exactly like
    /// [`Self::render`] — the variable stays undefined, which tool-aware
    /// templates probe for.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: &[serde_json::Value],
    ) -> Result<String, TemplateError> {
        self.render_full(messages, add_generation_prompt, tools, &[])
    }

    /// [`Self::render_with_tools`] plus extra template variables layered on
    /// top of the standard context — e.g. `enable_thinking=false` to make a
    /// thinking-trained template (Qwen3) render its non-thinking prompt for
    /// the Anthropic `thinking: {"type": "disabled"}` request option.
    /// Templates that don't reference a variable ignore it (lenient
    /// undefined), so passing extras is safe across model families. Extras
    /// are plain JSON so callers don't need a minijinja dependency.
    pub fn render_full(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: &[serde_json::Value],
        extra: &[(&str, serde_json::Value)],
    ) -> Result<String, TemplateError> {
        let mut ctx: Vec<(&str, Value)> = Vec::with_capacity(extra.len() + 1);
        if !tools.is_empty() {
            ctx.push(("tools", Value::from_serialize(tools)));
        }
        for (key, value) in extra {
            ctx.push((key, Value::from_serialize(value)));
        }
        self.render_with(messages, add_generation_prompt, &ctx)
    }

    /// [`Self::render`] with extra template variables layered on top of the
    /// standard context — e.g. pinning `date_string` so Llama 3.x templates
    /// don't interpolate today's date via `strftime_now` (the golden
    /// harness needs render-stable prompts).
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        extra: &[(&str, Value)],
    ) -> Result<String, TemplateError> {
        let template = self.env.get_template(TEMPLATE_NAME)?;
        let mut ctx: std::collections::BTreeMap<&str, Value> = std::collections::BTreeMap::new();
        ctx.insert("messages", Value::from_serialize(messages));
        ctx.insert("add_generation_prompt", Value::from(add_generation_prompt));
        ctx.insert("bos_token", Value::from(self.bos_token.as_str()));
        ctx.insert("eos_token", Value::from(self.eos_token.as_str()));
        for (key, value) in extra {
            ctx.insert(key, value.clone());
        }
        let rendered = template.render(Value::from_serialize(&ctx))?;
        Ok(rendered)
    }

    /// sha256 hex of the template source; comparable with the Python worker's
    /// `WorkerInfo.chat_template_hash` (empty there means "unknown").
    pub fn source_hash(&self) -> &str {
        &self.source_hash
    }

    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }
}

/// `chat_template` config value: a template string, or HF's list form
/// `[{"name": ..., "template": ...}]` where we prefer the `default` entry.
fn template_from_config(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(entries) => {
            let get = |name: &str| {
                entries.iter().find_map(|e| {
                    (e.get("name")?.as_str()? == name)
                        .then(|| e.get("template")?.as_str().map(str::to_owned))?
                })
            };
            get("default").or_else(|| {
                entries
                    .first()?
                    .get("template")?
                    .as_str()
                    .map(str::to_owned)
            })
        }
        _ => None,
    }
}

/// Special tokens serialize either as a bare string or as an AddedToken
/// object `{"content": "...", ...}`; absent/null means "no such token".
fn special_token(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        // Writing to a String cannot fail; ignore the fmt::Result.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// `| tojson` with Python `json.dumps` formatting, matching the filter
/// transformers installs for chat templates: `ensure_ascii=False`,
/// insertion-order keys, separators `", "`/`": "` (or `","` between items
/// when `indent` is given, each item then on its own indented line).
/// serde_json cannot produce these exact bytes, hence the custom writer.
fn tojson_python_style(
    value: &minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<String, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(value).map_err(|err| {
        minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("value is not JSON-serializable: {err}"),
        )
    })?;
    let mut out = String::new();
    write_python_json(&json, indent, 0, &mut out);
    Ok(out)
}

fn write_python_json(
    value: &serde_json::Value,
    indent: Option<usize>,
    level: usize,
    out: &mut String,
) {
    let newline_pad = |out: &mut String, level: usize| {
        if let Some(width) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
    };
    match value {
        serde_json::Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline_pad(out, level + 1);
                // serde_json string escaping matches json.dumps for the
                // ensure_ascii=False case (both escape only control chars,
                // quotes, and backslashes).
                let _ = write!(out, "{}", serde_json::Value::from(key.as_str()));
                out.push_str(": ");
                write_python_json(item, indent, level + 1, out);
            }
            newline_pad(out, level);
            out.push('}');
        }
        serde_json::Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline_pad(out, level + 1);
                write_python_json(item, indent, level + 1, out);
            }
            newline_pad(out, level);
            out.push(']');
        }
        // Empty containers, scalars, strings: serde_json and json.dumps
        // already agree ("{}", "[]", integers, true/false/null, string
        // escaping). Exotic floats (1e30: "1e30" vs Python's "1e+30") can
        // diverge; tool schemas with such values are out of scope.
        other => {
            let _ = write!(out, "{other}");
        }
    }
}

/// HF templates call `raise_exception("...")` for unsupported inputs; the
/// message must surface as a render error, not a crash.
fn raise_exception(message: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)` — local current time, Python strftime syntax
/// (Llama 3.x templates use it for `date_string`).
fn strftime_now(format: String) -> Result<String, minijinja::Error> {
    let mut out = String::new();
    write!(out, "{}", chrono::Local::now().format(&format)).map_err(|_| {
        minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("invalid strftime format: {format:?}"),
        )
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_string_from_config() {
        let value = serde_json::json!("hello {{ messages }}");
        assert_eq!(
            template_from_config(Some(&value)).as_deref(),
            Some("hello {{ messages }}")
        );
    }

    #[test]
    fn template_list_prefers_default() {
        let value = serde_json::json!([
            {"name": "tool_use", "template": "T"},
            {"name": "default", "template": "D"},
        ]);
        assert_eq!(template_from_config(Some(&value)).as_deref(), Some("D"));
    }

    #[test]
    fn template_list_falls_back_to_first() {
        let value = serde_json::json!([{"name": "rag", "template": "R"}]);
        assert_eq!(template_from_config(Some(&value)).as_deref(), Some("R"));
    }

    #[test]
    fn special_token_forms() {
        let bare = serde_json::json!("<s>");
        let added = serde_json::json!({"content": "<s>", "lstrip": false});
        let null = serde_json::json!(null);
        assert_eq!(special_token(Some(&bare)).as_deref(), Some("<s>"));
        assert_eq!(special_token(Some(&added)).as_deref(), Some("<s>"));
        assert_eq!(special_token(Some(&null)), None);
        assert_eq!(special_token(None), None);
    }

    #[test]
    fn raise_exception_surfaces_as_render_error() {
        let template = ChatTemplate::new(
            "{{ raise_exception('bad role') }}".into(),
            String::new(),
            String::new(),
        )
        .expect("compiles");
        let err = template.render(&[], false).expect_err("must fail");
        assert!(err.to_string().contains("bad role"), "got: {err}");
    }

    #[test]
    fn strftime_now_formats_current_date() {
        let template = ChatTemplate::new(
            "{{ strftime_now('%Y-%m-%d') }}".into(),
            String::new(),
            String::new(),
        )
        .expect("compiles");
        let out = template.render(&[], false).expect("renders");
        assert_eq!(out, chrono::Local::now().format("%Y-%m-%d").to_string());
    }

    #[test]
    fn source_hash_is_sha256_of_source() {
        let template =
            ChatTemplate::new("abc".into(), String::new(), String::new()).expect("compiles");
        // sha256("abc")
        assert_eq!(
            template.source_hash(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
