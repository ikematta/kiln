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
}

/// A compiled chat template plus the special tokens it interpolates.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
    source_hash: String,
    bos_token: String,
    eos_token: String,
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
        let mut env = Environment::new();
        // HF templates probe optional variables (`tools`, `date_string`, ...)
        // with `is defined`; strict undefined would reject them.
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        // Python string/dict methods (`.strip()`, `.items()`, ...) used
        // throughout HF-ecosystem templates.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(TEMPLATE_NAME, source)?;
        Ok(Self {
            env,
            source_hash,
            bos_token,
            eos_token,
        })
    }

    /// Renders the conversation into the model's prompt string.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TemplateError> {
        self.render_with(messages, add_generation_prompt, &[])
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
