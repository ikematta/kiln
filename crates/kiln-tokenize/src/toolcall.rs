//! Streaming tool-call parsers (SPEC §8.2): turn the incremental decoder's
//! text stream into OpenAI-shape `tool_calls` events instead of raw text.
//!
//! Three model-family formats, selected from model metadata (the chat
//! template source — the template both teaches the model the format and
//! serializes prior calls in it, so it is the authoritative marker):
//!
//! - **Hermes** ([`ToolCallFormat::Hermes`], Qwen 2.5/3 and NousResearch
//!   lineage): `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`
//!   blocks, multiple blocks for parallel calls, free text (e.g. Qwen3
//!   `<think>` blocks) allowed around them.
//! - **Llama 3.x** ([`ToolCallFormat::Llama`]): the whole completion is the
//!   call — optional `<|python_tag|>`, then `{"name": ..., "parameters":
//!   {...}}`, `;`-separated for multiple calls. Real Llama-3.2 output also
//!   mirrors the prompt's OpenAI tool shape back
//!   (`{"type": "function", "function": {...}}`); that wrapper unwraps.
//!   Anything that doesn't start like a call is plain content.
//! - **Qwen XML-ish** ([`ToolCallFormat::QwenXml`], Qwen3-Coder):
//!   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...`
//!   Parameter values are raw text; they are coerced back to JSON using the
//!   request's tool schemas ([`ToolCallParser::new`] takes the tools).
//!
//! # Streaming contract
//!
//! [`ToolCallParser::push`] accepts arbitrary text segments (the
//! [`StreamingDecoder`](crate::StreamingDecoder)'s output for Rust workers,
//! `TokenChunk.text` passthrough for tokenizer-owning workers) and emits
//! [`ToolEvent`]s. The invariant the tests enforce: for any chunking of the
//! same text, the event stream *reassembles identically* — same
//! `Content` concatenation, same calls in the same order, each call's
//! `CallArgs` deltas concatenating to the same arguments string. Delta
//! granularity may differ; totals may not.
//!
//! Text handling outside calls: a text run (stream start → first call,
//! between calls, after the last call) that is whitespace-only is dropped;
//! a run with substance passes through verbatim. This keeps the inter-call
//! `\n` separators models emit out of `content` without eating real text.
//!
//! Malformed input never panics and never goes silent mid-call: a block
//! that stops looking like a call *before* its name is known is replayed
//! as plain content; after the name is known (a `CallStart` is already
//! out) the call is closed with what streamed and the junk tail is
//! dropped. A call truncated by the token limit flushes its partial
//! arguments and closes — mirroring OpenAI's own behavior under
//! `finish_reason: "length"`.

use std::collections::HashMap;

/// Which family format a model emits, detected from its chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// `<tool_call>{json}</tool_call>` (Qwen 2.5/3, Hermes lineage).
    Hermes,
    /// Optional `<|python_tag|>` + bare JSON call(s) (Llama 3.x).
    Llama,
    /// `<tool_call><function=..><parameter=..>` XML-ish (Qwen3-Coder).
    QwenXml,
}

impl ToolCallFormat {
    /// Detects the format from the chat template source. Precedence
    /// matters: Qwen3-Coder templates contain both `<function=` and
    /// `<tool_call>`; plain Qwen/Hermes only the latter; Llama templates
    /// carry the ipython environment marker instead.
    pub fn detect(template_source: &str) -> Option<Self> {
        if template_source.contains("<function=") {
            Some(Self::QwenXml)
        } else if template_source.contains("<tool_call>") {
            Some(Self::Hermes)
        } else if template_source.contains("<|python_tag|>")
            || template_source.contains("Environment: ipython")
        {
            Some(Self::Llama)
        } else {
            None
        }
    }
}

/// One parsed increment of the model's output stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    /// Plain assistant text (safe to emit as a content delta).
    Content(String),
    /// A tool call's name is known; `index` is 0-based per response.
    CallStart { index: usize, name: String },
    /// A fragment of the call's `arguments` JSON string. Fragments for one
    /// call concatenate to the full arguments text.
    CallArgs { index: usize, delta: String },
    /// The call's arguments are complete.
    CallEnd { index: usize },
}

/// Streaming parser for one response.
pub struct ToolCallParser {
    inner: Inner,
}

enum Inner {
    Hermes(HermesParser),
    Llama(LlamaParser),
    QwenXml(QwenXmlParser),
}

impl ToolCallParser {
    /// `tools` are the request's OpenAI-shape tool definitions; only the
    /// Qwen XML format needs them (declared parameter types drive value
    /// coercion), the others ignore them.
    pub fn new(format: ToolCallFormat, tools: &[serde_json::Value]) -> Self {
        let inner = match format {
            ToolCallFormat::Hermes => Inner::Hermes(HermesParser::new()),
            ToolCallFormat::Llama => Inner::Llama(LlamaParser::new()),
            ToolCallFormat::QwenXml => Inner::QwenXml(QwenXmlParser::new(param_type_hints(tools))),
        };
        Self { inner }
    }

    /// Feeds the next text segment; returns the events it completes.
    pub fn push(&mut self, text: &str) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        match &mut self.inner {
            Inner::Hermes(p) => p.push(text, &mut out),
            Inner::Llama(p) => p.push(text, &mut out),
            Inner::QwenXml(p) => p.push(text, &mut out),
        }
        out
    }

    /// Ends the stream: flushes a truncated call (partial arguments, then
    /// `CallEnd`) or replays an undecided block as content.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        match &mut self.inner {
            Inner::Hermes(p) => p.finish(&mut out),
            Inner::Llama(p) => p.finish(&mut out),
            Inner::QwenXml(p) => p.finish(&mut out),
        }
        out
    }
}

/// `tool name → parameter name → declared JSON-schema type` from the
/// request's tool definitions (for XML parameter coercion).
fn param_type_hints(tools: &[serde_json::Value]) -> HashMap<String, HashMap<String, String>> {
    let mut hints = HashMap::new();
    for tool in tools {
        let Some(function) = tool.get("function") else {
            continue;
        };
        let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let mut params = HashMap::new();
        if let Some(props) = function
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (key, prop) in props {
                if let Some(ty) = prop.get("type").and_then(|t| t.as_str()) {
                    params.insert(key.clone(), ty.to_owned());
                }
            }
        }
        hints.insert(name.to_owned(), params);
    }
    hints
}

// ---------------------------------------------------------------------------
// Text-run gate: whitespace-only runs outside calls are dropped
// ---------------------------------------------------------------------------

/// Gates one text run (stream start → call, call → call, call → EOS):
/// leading whitespace is held until the run shows substance (then flushed
/// with it) or the run ends at a call/EOS (then dropped). Once a run has
/// substance, everything passes verbatim.
#[derive(Default)]
struct RunGate {
    held_ws: String,
    seen_substance: bool,
}

impl RunGate {
    fn feed(&mut self, text: &str, out: &mut Vec<ToolEvent>) {
        if text.is_empty() {
            return;
        }
        if self.seen_substance {
            out.push(ToolEvent::Content(text.to_owned()));
        } else if text.chars().all(char::is_whitespace) {
            self.held_ws.push_str(text);
        } else {
            self.seen_substance = true;
            let mut released = std::mem::take(&mut self.held_ws);
            released.push_str(text);
            out.push(ToolEvent::Content(released));
        }
    }

    /// The run ended at a call boundary or EOS: held whitespace is dropped.
    fn end_run(&mut self) {
        self.held_ws.clear();
        self.seen_substance = false;
    }
}

// ---------------------------------------------------------------------------
// Marker scanning with prefix holdback
// ---------------------------------------------------------------------------

/// Longest strict suffix of `buf` that is a prefix of `marker` (text that
/// may become the marker on the next push and must be held).
fn marker_holdback(buf: &str, marker: &str) -> usize {
    let max = buf.len().min(marker.len() - 1);
    for k in (1..=max).rev() {
        if !buf.is_char_boundary(buf.len() - k) {
            continue;
        }
        if marker.starts_with(&buf[buf.len() - k..]) {
            return k;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Incremental JSON payload scanning (Hermes + Llama)
// ---------------------------------------------------------------------------

/// What is known about a `{"name": ..., "arguments"/"parameters": ...}`
/// payload so far. Byte offsets index the scanned string and stay valid
/// across pushes (the buffer is append-only).
#[derive(Debug, Default, Clone, PartialEq)]
struct PayloadInfo {
    /// Complete, unescaped `name` value.
    name: Option<String>,
    /// Arguments value span: start, and end (exclusive) once complete.
    args: Option<(usize, Option<usize>)>,
}

#[derive(Debug, PartialEq)]
enum PayloadScan {
    /// Everything so far is consistent with a call payload; needs more text.
    Partial(PayloadInfo),
    /// The payload object closed at byte `end` (exclusive).
    Complete(PayloadInfo, usize),
    /// Structurally not a call payload (progress up to the failure is
    /// reported so already-emitted events stay consistent).
    Failed(PayloadInfo),
}

/// Scans a candidate call payload: a JSON object whose interesting keys are
/// `name` and `arguments`/`parameters`, with one optional level of the
/// OpenAI `{"type": "function", "function": {...}}` wrapper. The scan is
/// re-run over the whole buffered payload on every push (payloads are
/// small); it never allocates except for the unescaped name.
fn scan_payload(payload: &str) -> PayloadScan {
    let s = payload.as_bytes();
    let mut info = PayloadInfo::default();
    let mut i = skip_ws(s, 0);
    if i >= s.len() {
        return PayloadScan::Partial(info);
    }
    if s[i] != b'{' {
        return PayloadScan::Failed(info);
    }
    i += 1;
    let mut in_wrapper = false;
    loop {
        i = skip_ws(s, i);
        if i >= s.len() {
            return PayloadScan::Partial(info);
        }
        // Key position: a closing brace ends the current object level.
        if s[i] == b'}' {
            if in_wrapper {
                in_wrapper = false;
                i = skip_ws(s, i + 1);
                if i >= s.len() {
                    return PayloadScan::Partial(info);
                }
                match s[i] {
                    b',' => {
                        i += 1;
                        continue;
                    }
                    b'}' => return PayloadScan::Complete(info, i + 1),
                    _ => return PayloadScan::Failed(info),
                }
            }
            return PayloadScan::Complete(info, i + 1);
        }
        if s[i] != b'"' {
            return PayloadScan::Failed(info);
        }
        let Some(key_end) = scan_json_string(s, i) else {
            return PayloadScan::Partial(info);
        };
        let key = &payload[i + 1..key_end - 1];
        i = skip_ws(s, key_end);
        if i >= s.len() {
            return PayloadScan::Partial(info);
        }
        if s[i] != b':' {
            return PayloadScan::Failed(info);
        }
        i = skip_ws(s, i + 1);
        if i >= s.len() {
            return PayloadScan::Partial(info);
        }

        if key == "function" && !in_wrapper && s[i] == b'{' {
            // Descend into the wrapper; its keys are scanned like the
            // outer object's.
            in_wrapper = true;
            i += 1;
            continue;
        }
        match key {
            "name" => {
                if s[i] != b'"' {
                    return PayloadScan::Failed(info);
                }
                let Some(end) = scan_json_string(s, i) else {
                    return PayloadScan::Partial(info);
                };
                match serde_json::from_str::<String>(&payload[i..end]) {
                    Ok(name) => info.name = Some(name),
                    Err(_) => return PayloadScan::Failed(info),
                }
                i = end;
            }
            "arguments" | "parameters" => {
                let start = i;
                match scan_json_value(s, i) {
                    Some(end) => {
                        info.args = Some((start, Some(end)));
                        i = end;
                    }
                    None => {
                        info.args = Some((start, None));
                        return PayloadScan::Partial(info);
                    }
                }
            }
            _ => match scan_json_value(s, i) {
                Some(end) => i = end,
                None => return PayloadScan::Partial(info),
            },
        }

        i = skip_ws(s, i);
        if i >= s.len() {
            return PayloadScan::Partial(info);
        }
        match s[i] {
            b',' => i += 1,
            b'}' => {} // handled at the top of the loop
            _ => return PayloadScan::Failed(info),
        }
    }
}

fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

/// End (exclusive) of the JSON string starting at the `"` at `start`;
/// `None` while unterminated.
fn scan_json_string(s: &[u8], start: usize) -> Option<usize> {
    let mut escape = false;
    for (offset, &b) in s[start + 1..].iter().enumerate() {
        if escape {
            escape = false;
        } else if b == b'\\' {
            escape = true;
        } else if b == b'"' {
            return Some(start + 1 + offset + 1);
        }
    }
    None
}

/// End (exclusive) of the JSON value starting at `start`; `None` while it
/// needs more bytes. Structure-only: brace/bracket balance and string
/// boundaries, no validity check (the value text streams verbatim).
fn scan_json_value(s: &[u8], start: usize) -> Option<usize> {
    match s[start] {
        b'"' => scan_json_string(s, start),
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escape = false;
            for (offset, &b) in s[start..].iter().enumerate() {
                if in_string {
                    if escape {
                        escape = false;
                    } else if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_string = false;
                    }
                } else {
                    match b {
                        b'"' => in_string = true,
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(start + offset + 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        }
        // Bare scalar (number, true/false/null): complete once a JSON
        // delimiter follows.
        _ => s[start..]
            .iter()
            .position(|b| matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n'))
            .map(|offset| start + offset),
    }
}

// ---------------------------------------------------------------------------
// Per-call progress shared by the two JSON-payload formats
// ---------------------------------------------------------------------------

/// Tracks what has been emitted for the call currently being parsed.
#[derive(Default)]
struct CallProgress {
    name_emitted: bool,
    /// Bytes of the arguments value already emitted (relative to its start).
    args_emitted: usize,
    /// Whether any arguments text was emitted at all (a call may have none).
    had_args: bool,
}

impl CallProgress {
    /// Emits everything `info` newly proves: the `CallStart` once the name
    /// is known, then argument bytes up to the known-safe extent
    /// (`buf.len()` while the value is still open — an open value runs to
    /// the end of the buffer).
    fn emit(&mut self, info: &PayloadInfo, buf: &str, index: usize, out: &mut Vec<ToolEvent>) {
        if !self.name_emitted
            && let Some(name) = &info.name
        {
            out.push(ToolEvent::CallStart {
                index,
                name: name.clone(),
            });
            self.name_emitted = true;
        }
        if !self.name_emitted {
            // Arguments before the name are held (all-at-once flush later);
            // starting the call without a name is not representable.
            return;
        }
        if let Some((start, end)) = info.args {
            let extent = end.unwrap_or(buf.len());
            let from = start + self.args_emitted;
            if extent > from {
                out.push(ToolEvent::CallArgs {
                    index,
                    delta: buf[from..extent].to_owned(),
                });
                self.args_emitted = extent - start;
                self.had_args = true;
            }
        }
    }

    /// Closes the call: a call that never streamed arguments gets the
    /// canonical empty object so clients always receive valid JSON.
    fn end(&mut self, index: usize, out: &mut Vec<ToolEvent>) {
        if !self.had_args {
            out.push(ToolEvent::CallArgs {
                index,
                delta: "{}".to_owned(),
            });
        }
        out.push(ToolEvent::CallEnd { index });
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Hermes: <tool_call>{json}</tool_call>
// ---------------------------------------------------------------------------

const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";

struct HermesParser {
    state: HermesState,
    buf: String,
    gate: RunGate,
    next_index: usize,
    progress: CallProgress,
}

#[derive(PartialEq)]
enum HermesState {
    /// Scanning content for the next `<tool_call>`.
    Outside,
    /// Buffering the JSON payload after `<tool_call>`.
    Payload,
    /// Payload object closed; consuming whitespace + `</tool_call>`.
    Closing,
}

impl HermesParser {
    fn new() -> Self {
        Self {
            state: HermesState::Outside,
            buf: String::new(),
            gate: RunGate::default(),
            next_index: 0,
            progress: CallProgress::default(),
        }
    }

    fn push(&mut self, text: &str, out: &mut Vec<ToolEvent>) {
        self.buf.push_str(text);
        loop {
            match self.state {
                HermesState::Outside => {
                    if let Some(idx) = self.buf.find(HERMES_OPEN) {
                        let pre: String = self.buf.drain(..idx + HERMES_OPEN.len()).collect();
                        self.gate.feed(&pre[..idx], out);
                        self.gate.end_run();
                        self.state = HermesState::Payload;
                        continue;
                    }
                    let hold = marker_holdback(&self.buf, HERMES_OPEN);
                    let release: String = self.buf.drain(..self.buf.len() - hold).collect();
                    self.gate.feed(&release, out);
                    return;
                }
                HermesState::Payload => match scan_payload(&self.buf) {
                    PayloadScan::Partial(info) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        return;
                    }
                    PayloadScan::Complete(info, end) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        if self.progress.name_emitted {
                            self.progress.end(self.next_index, out);
                            self.next_index += 1;
                            self.buf.drain(..end);
                            self.state = HermesState::Closing;
                        } else {
                            // A JSON object without a name is not a call:
                            // replay the whole block as content.
                            self.degrade_to_content(out);
                        }
                        continue;
                    }
                    PayloadScan::Failed(info) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        if self.progress.name_emitted {
                            // The CallStart is already out; close the call
                            // and drop the junk tail.
                            self.progress.end(self.next_index, out);
                            self.next_index += 1;
                            self.buf.clear();
                            self.state = HermesState::Outside;
                        } else {
                            self.degrade_to_content(out);
                        }
                        continue;
                    }
                },
                HermesState::Closing => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return;
                    }
                    if let Some(rest) = trimmed.strip_prefix(HERMES_CLOSE) {
                        self.buf = rest.to_owned();
                        self.state = HermesState::Outside;
                        continue;
                    }
                    if HERMES_CLOSE.starts_with(trimmed) {
                        return; // may complete on the next push
                    }
                    // No closing tag; whatever follows is a new content run
                    // (the call itself already ended cleanly).
                    self.state = HermesState::Outside;
                }
            }
        }
    }

    /// Replays an aborted block (opener + buffered payload) as content and
    /// resumes content scanning with what remains.
    fn degrade_to_content(&mut self, out: &mut Vec<ToolEvent>) {
        let replay = format!("{HERMES_OPEN}{}", self.buf);
        self.buf.clear();
        self.state = HermesState::Outside;
        // Feed through the gate directly (not the marker scan: this text
        // was already inspected and rejected).
        self.gate.feed(&replay, out);
    }

    fn finish(&mut self, out: &mut Vec<ToolEvent>) {
        match self.state {
            HermesState::Outside => {
                let tail = std::mem::take(&mut self.buf);
                self.gate.feed(&tail, out);
            }
            HermesState::Payload => {
                if let PayloadScan::Partial(info) = scan_payload(&self.buf) {
                    self.progress.emit(&info, &self.buf, self.next_index, out);
                }
                if self.progress.name_emitted {
                    // Truncated call (token limit): close with what streamed.
                    self.progress.end(self.next_index, out);
                    self.next_index += 1;
                } else {
                    self.degrade_to_content(out);
                }
            }
            HermesState::Closing => {}
        }
        self.gate.end_run();
    }
}

// ---------------------------------------------------------------------------
// Llama 3.x: optional <|python_tag|> + bare JSON call(s)
// ---------------------------------------------------------------------------

const PYTHON_TAG: &str = "<|python_tag|>";

struct LlamaParser {
    state: LlamaState,
    buf: String,
    gate: RunGate,
    next_index: usize,
    progress: CallProgress,
    tag_consumed: bool,
}

#[derive(PartialEq)]
enum LlamaState {
    /// Deciding whether the completion is a call or plain text.
    Start,
    /// Buffering a call payload.
    Payload,
    /// After a payload: `;` starts another call; anything else is content.
    AfterPayload,
    /// The completion is plain text; everything passes through.
    Text,
}

impl LlamaParser {
    fn new() -> Self {
        Self {
            state: LlamaState::Start,
            buf: String::new(),
            gate: RunGate::default(),
            next_index: 0,
            progress: CallProgress::default(),
            tag_consumed: false,
        }
    }

    fn push(&mut self, text: &str, out: &mut Vec<ToolEvent>) {
        self.buf.push_str(text);
        loop {
            match self.state {
                LlamaState::Start => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return; // hold leading whitespace until decidable
                    }
                    if let Some(rest) = trimmed.strip_prefix(PYTHON_TAG) {
                        self.buf = rest.to_owned();
                        self.tag_consumed = true;
                        self.state = LlamaState::Payload;
                        continue;
                    }
                    if PYTHON_TAG.starts_with(trimmed) {
                        return; // may become the tag on the next push
                    }
                    if trimmed.starts_with('{') {
                        self.buf = trimmed.to_owned();
                        self.state = LlamaState::Payload;
                        continue;
                    }
                    // Not a call: plain text from here on (leading
                    // whitespace included — it is part of the content run).
                    let all = std::mem::take(&mut self.buf);
                    self.gate.feed(&all, out);
                    self.state = LlamaState::Text;
                    return;
                }
                LlamaState::Payload => match scan_payload(&self.buf) {
                    PayloadScan::Partial(info) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        return;
                    }
                    PayloadScan::Complete(info, end) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        if self.progress.name_emitted {
                            self.progress.end(self.next_index, out);
                            self.next_index += 1;
                            self.buf.drain(..end);
                            self.state = LlamaState::AfterPayload;
                        } else {
                            self.degrade_to_text(out);
                        }
                        continue;
                    }
                    PayloadScan::Failed(info) => {
                        self.progress.emit(&info, &self.buf, self.next_index, out);
                        if self.progress.name_emitted {
                            self.progress.end(self.next_index, out);
                            self.next_index += 1;
                            self.buf.clear();
                            self.state = LlamaState::Text;
                        } else {
                            self.degrade_to_text(out);
                        }
                        continue;
                    }
                },
                LlamaState::AfterPayload => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return;
                    }
                    if let Some(rest) = trimmed.strip_prefix(';') {
                        self.buf = rest.to_owned();
                        self.state = LlamaState::Payload;
                        continue;
                    }
                    // Trailing content after the call(s): a new run.
                    self.buf = trimmed.to_owned();
                    self.state = LlamaState::Text;
                }
                LlamaState::Text => {
                    let all = std::mem::take(&mut self.buf);
                    self.gate.feed(&all, out);
                    return;
                }
            }
        }
    }

    /// The buffered block is not a call after all: replay it (with the
    /// python tag, if one was consumed) as plain content, permanently.
    fn degrade_to_text(&mut self, out: &mut Vec<ToolEvent>) {
        let mut replay = String::new();
        if self.tag_consumed {
            replay.push_str(PYTHON_TAG);
        }
        replay.push_str(&self.buf);
        self.buf.clear();
        self.state = LlamaState::Text;
        self.gate.feed(&replay, out);
    }

    fn finish(&mut self, out: &mut Vec<ToolEvent>) {
        match self.state {
            LlamaState::Start | LlamaState::Text => {
                let tail = std::mem::take(&mut self.buf);
                self.gate.feed(&tail, out);
            }
            LlamaState::Payload => {
                if let PayloadScan::Partial(info) = scan_payload(&self.buf) {
                    self.progress.emit(&info, &self.buf, self.next_index, out);
                }
                if self.progress.name_emitted {
                    self.progress.end(self.next_index, out);
                    self.next_index += 1;
                } else {
                    self.degrade_to_text(out);
                }
            }
            LlamaState::AfterPayload => {}
        }
        self.gate.end_run();
    }
}

// ---------------------------------------------------------------------------
// Qwen XML-ish (Qwen3-Coder)
// ---------------------------------------------------------------------------

const XML_FUNCTION_OPEN: &str = "<function=";
const XML_FUNCTION_CLOSE: &str = "</function>";
const XML_PARAMETER_OPEN: &str = "<parameter=";
const XML_PARAMETER_CLOSE: &str = "</parameter>";

struct QwenXmlParser {
    state: XmlState,
    buf: String,
    gate: RunGate,
    next_index: usize,
    hints: HashMap<String, HashMap<String, String>>,
    /// Name of the function currently being parsed (type-hint lookup).
    current_fn: String,
    /// Parameters emitted for the current call (comma placement).
    params_emitted: usize,
}

#[derive(PartialEq)]
enum XmlState {
    /// Scanning content for `<tool_call>`.
    Outside,
    /// After `<tool_call>`: expecting `<function=NAME>`.
    PreFunction,
    /// Inside `<function=...>`: expecting `<parameter=` or `</function>`.
    InFunction,
    /// Inside `<parameter=KEY>`: buffering the value to `</parameter>`.
    /// `ate_nl` records whether the single formatting newline after the
    /// `>` has been consumed (it may arrive in a later push).
    ParamValue { key: String, ate_nl: bool },
    /// After `</function>`: consuming whitespace + `</tool_call>`.
    Closing,
}

impl QwenXmlParser {
    fn new(hints: HashMap<String, HashMap<String, String>>) -> Self {
        Self {
            state: XmlState::Outside,
            buf: String::new(),
            gate: RunGate::default(),
            next_index: 0,
            hints,
            current_fn: String::new(),
            params_emitted: 0,
        }
    }

    fn push(&mut self, text: &str, out: &mut Vec<ToolEvent>) {
        self.buf.push_str(text);
        loop {
            match &self.state {
                XmlState::Outside => {
                    if let Some(idx) = self.buf.find(HERMES_OPEN) {
                        let pre: String = self.buf.drain(..idx + HERMES_OPEN.len()).collect();
                        self.gate.feed(&pre[..idx], out);
                        self.gate.end_run();
                        self.state = XmlState::PreFunction;
                        continue;
                    }
                    let hold = marker_holdback(&self.buf, HERMES_OPEN);
                    let release: String = self.buf.drain(..self.buf.len() - hold).collect();
                    self.gate.feed(&release, out);
                    return;
                }
                XmlState::PreFunction => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return;
                    }
                    if let Some(rest) = trimmed.strip_prefix(XML_FUNCTION_OPEN) {
                        let Some(gt) = rest.find('>') else {
                            // Name still streaming; keep everything buffered
                            // (re-trimmed and re-matched on the next push).
                            return;
                        };
                        let name = rest[..gt].to_owned();
                        self.buf = rest[gt + 1..].to_owned();
                        out.push(ToolEvent::CallStart {
                            index: self.next_index,
                            name: name.clone(),
                        });
                        out.push(ToolEvent::CallArgs {
                            index: self.next_index,
                            delta: "{".to_owned(),
                        });
                        self.current_fn = name;
                        self.params_emitted = 0;
                        self.state = XmlState::InFunction;
                        continue;
                    }
                    if XML_FUNCTION_OPEN.starts_with(trimmed) {
                        return;
                    }
                    // Not the XML call shape: replay the block as content.
                    let replay = format!("{HERMES_OPEN}{}", self.buf);
                    self.buf.clear();
                    self.state = XmlState::Outside;
                    self.gate.feed(&replay, out);
                    continue;
                }
                XmlState::InFunction => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return;
                    }
                    if let Some(rest) = trimmed.strip_prefix(XML_PARAMETER_OPEN) {
                        let Some(gt) = rest.find('>') else {
                            return;
                        };
                        let key = rest[..gt].to_owned();
                        self.buf = rest[gt + 1..].to_owned();
                        self.state = XmlState::ParamValue { key, ate_nl: false };
                        continue;
                    }
                    if let Some(rest) = trimmed.strip_prefix(XML_FUNCTION_CLOSE) {
                        self.buf = rest.to_owned();
                        out.push(ToolEvent::CallArgs {
                            index: self.next_index,
                            delta: "}".to_owned(),
                        });
                        out.push(ToolEvent::CallEnd {
                            index: self.next_index,
                        });
                        self.next_index += 1;
                        self.state = XmlState::Closing;
                        continue;
                    }
                    if XML_PARAMETER_OPEN.starts_with(trimmed)
                        || XML_FUNCTION_CLOSE.starts_with(trimmed)
                    {
                        return;
                    }
                    // Junk inside the function block: the CallStart is out,
                    // so close the call and drop the tail.
                    self.abort_call(out);
                    continue;
                }
                XmlState::ParamValue { key, ate_nl } => {
                    if !ate_nl {
                        if self.buf.is_empty() {
                            return;
                        }
                        // The template writes the value on its own line:
                        // consume exactly one formatting newline (a sloppy
                        // same-line value simply has none to consume).
                        if self.buf.starts_with('\n') {
                            self.buf.drain(..1);
                        }
                        let key = key.clone();
                        self.state = XmlState::ParamValue { key, ate_nl: true };
                        continue;
                    }
                    let Some(idx) = self.buf.find(XML_PARAMETER_CLOSE) else {
                        return; // value still streaming (buffered whole)
                    };
                    let key = key.clone();
                    let mut value = self.buf[..idx].to_owned();
                    if value.ends_with('\n') {
                        value.pop();
                    }
                    self.buf.drain(..idx + XML_PARAMETER_CLOSE.len());
                    self.emit_param(&key, &value, out);
                    self.state = XmlState::InFunction;
                    continue;
                }
                XmlState::Closing => {
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return;
                    }
                    if let Some(rest) = trimmed.strip_prefix(HERMES_CLOSE) {
                        self.buf = rest.to_owned();
                        self.state = XmlState::Outside;
                        continue;
                    }
                    if HERMES_CLOSE.starts_with(trimmed) {
                        return;
                    }
                    self.state = XmlState::Outside; // call already ended cleanly
                }
            }
        }
    }

    /// Encodes one completed parameter into the arguments object under
    /// construction and emits it as a single delta.
    fn emit_param(&mut self, key: &str, value: &str, out: &mut Vec<ToolEvent>) {
        let declared = self
            .hints
            .get(&self.current_fn)
            .and_then(|params| params.get(key))
            .map(String::as_str);
        let mut delta = String::new();
        if self.params_emitted > 0 {
            delta.push(',');
        }
        delta.push_str(&serde_json::Value::from(key).to_string());
        delta.push(':');
        delta.push_str(&encode_xml_param(value, declared));
        self.params_emitted += 1;
        out.push(ToolEvent::CallArgs {
            index: self.next_index,
            delta,
        });
    }

    /// Closes a call whose block went malformed after `CallStart`.
    fn abort_call(&mut self, out: &mut Vec<ToolEvent>) {
        out.push(ToolEvent::CallArgs {
            index: self.next_index,
            delta: "}".to_owned(),
        });
        out.push(ToolEvent::CallEnd {
            index: self.next_index,
        });
        self.next_index += 1;
        self.buf.clear();
        self.state = XmlState::Outside;
    }

    fn finish(&mut self, out: &mut Vec<ToolEvent>) {
        match std::mem::replace(&mut self.state, XmlState::Outside) {
            XmlState::Outside => {
                let tail = std::mem::take(&mut self.buf);
                self.gate.feed(&tail, out);
            }
            XmlState::PreFunction => {
                // Never got a name: replay the block as content.
                let replay = format!("{HERMES_OPEN}{}", self.buf);
                self.buf.clear();
                self.gate.feed(&replay, out);
            }
            XmlState::ParamValue { key, ate_nl } => {
                // Truncated mid-value (token limit): keep the partial value.
                let mut value = std::mem::take(&mut self.buf);
                if !ate_nl && value.starts_with('\n') {
                    value.drain(..1);
                }
                if value.ends_with('\n') {
                    value.pop();
                }
                self.emit_param(&key, &value, out);
                self.abort_call(out);
            }
            XmlState::InFunction => self.abort_call(out),
            XmlState::Closing => {}
        }
        self.gate.end_run();
    }
}

/// JSON-encodes an XML parameter value using its declared schema type
/// (mirrors the reference Qwen3-Coder parser): `string` stays verbatim
/// (quoted); `boolean` accepts the template's Python-style `True`/`False`;
/// `integer`/`number` parse numerically; anything else is used verbatim
/// when it is valid JSON. Unparseable values fall back to a JSON string —
/// the client always receives valid arguments JSON.
fn encode_xml_param(raw: &str, declared: Option<&str>) -> String {
    let quoted = || serde_json::Value::from(raw).to_string();
    match declared {
        Some("string") => quoted(),
        Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => "true".to_owned(),
            "false" => "false".to_owned(),
            _ => quoted(),
        },
        Some("integer") => match raw.trim().parse::<i64>() {
            Ok(value) => value.to_string(),
            Err(_) => quoted(),
        },
        Some("number") => match raw.trim().parse::<f64>() {
            // Match the fixture generator: integral values without an
            // explicit decimal point serialize as integers.
            Ok(value) if value.fract() == 0.0 && !raw.contains('.') && value.abs() < 1e15 => {
                (value as i64).to_string()
            }
            Ok(value) => serde_json::Number::from_f64(value).map_or_else(quoted, |n| n.to_string()),
            Err(_) => quoted(),
        },
        _ => {
            if serde_json::from_str::<serde_json::Value>(raw).is_ok() {
                raw.to_owned()
            } else {
                quoted()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `text` through a fresh parser in one push and reassembles.
    fn parse_once(format: ToolCallFormat, tools: &[serde_json::Value], text: &str) -> Reassembled {
        let mut parser = ToolCallParser::new(format, tools);
        let mut events = parser.push(text);
        events.extend(parser.finish());
        reassemble(&events)
    }

    #[derive(Debug, PartialEq, Default)]
    struct Reassembled {
        content: String,
        calls: Vec<(String, String)>, // (name, arguments)
    }

    fn reassemble(events: &[ToolEvent]) -> Reassembled {
        let mut result = Reassembled::default();
        for event in events {
            match event {
                ToolEvent::Content(text) => result.content.push_str(text),
                ToolEvent::CallStart { index, name } => {
                    assert_eq!(*index, result.calls.len(), "indices are sequential");
                    result.calls.push((name.clone(), String::new()));
                }
                ToolEvent::CallArgs { index, delta } => {
                    result.calls[*index].1.push_str(delta);
                }
                ToolEvent::CallEnd { index } => {
                    assert_eq!(*index, result.calls.len() - 1);
                }
            }
        }
        result
    }

    #[test]
    fn detect_format_from_template_markers() {
        assert_eq!(
            ToolCallFormat::detect("... <tools> ... <tool_call>\n{...}"),
            Some(ToolCallFormat::Hermes)
        );
        assert_eq!(
            ToolCallFormat::detect("... <tool_call>\n<function=example> ..."),
            Some(ToolCallFormat::QwenXml)
        );
        assert_eq!(
            ToolCallFormat::detect("Environment: ipython\n ..."),
            Some(ToolCallFormat::Llama)
        );
        assert_eq!(ToolCallFormat::detect("{{ messages }}"), None);
    }

    #[test]
    fn hermes_single_call() {
        let got = parse_once(
            ToolCallFormat::Hermes,
            &[],
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>",
        );
        assert_eq!(got.content, "");
        assert_eq!(got.calls, vec![("f".into(), "{\"a\": 1}".into())]);
    }

    #[test]
    fn hermes_text_around_calls() {
        let got = parse_once(
            ToolCallFormat::Hermes,
            &[],
            "lead-in\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\ntrailing text",
        );
        // Both runs have substance, so both pass verbatim (including the
        // newline that opens the trailing run).
        assert_eq!(got.content, "lead-in\n\ntrailing text");
        assert_eq!(got.calls, vec![("f".into(), "{}".into())]);
    }

    #[test]
    fn hermes_arguments_before_name_are_held_then_flushed() {
        let text = "<tool_call>{\"arguments\": {\"a\": 1}, \"name\": \"f\"}</tool_call>";
        let got = parse_once(ToolCallFormat::Hermes, &[], text);
        assert_eq!(got.calls, vec![("f".into(), "{\"a\": 1}".into())]);
        // And chunked: no args may escape before the CallStart.
        let mut parser = ToolCallParser::new(ToolCallFormat::Hermes, &[]);
        let mut events = Vec::new();
        for c in text.chars() {
            events.extend(parser.push(&c.to_string()));
        }
        events.extend(parser.finish());
        let first_call_event = events
            .iter()
            .position(|e| matches!(e, ToolEvent::CallStart { .. }))
            .expect("has a call");
        assert!(
            events[..first_call_event]
                .iter()
                .all(|e| !matches!(e, ToolEvent::CallArgs { .. })),
        );
        assert_eq!(reassemble(&events), got);
    }

    #[test]
    fn hermes_missing_name_replays_as_content() {
        let text = "<tool_call>{\"arguments\": {\"a\": 1}}</tool_call>";
        let got = parse_once(ToolCallFormat::Hermes, &[], text);
        assert_eq!(got.calls, vec![]);
        assert!(got.content.starts_with("<tool_call>"), "{}", got.content);
    }

    #[test]
    fn hermes_truncated_call_closes_with_partial_args() {
        let mut parser = ToolCallParser::new(ToolCallFormat::Hermes, &[]);
        let mut events = parser.push("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": ");
        events.extend(parser.finish());
        let got = reassemble(&events);
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].0, "f");
        assert_eq!(got.calls[0].1, "{\"a\": ");
        assert!(events.contains(&ToolEvent::CallEnd { index: 0 }));
    }

    #[test]
    fn hermes_call_without_arguments_key_gets_empty_object() {
        let got = parse_once(
            ToolCallFormat::Hermes,
            &[],
            "<tool_call>{\"name\": \"ping\"}</tool_call>",
        );
        assert_eq!(got.calls, vec![("ping".into(), "{}".into())]);
    }

    #[test]
    fn llama_plain_text_passes_through() {
        let got = parse_once(
            ToolCallFormat::Llama,
            &[],
            "A kiln is an oven for firing clay.",
        );
        assert_eq!(got.content, "A kiln is an oven for firing clay.");
        assert!(got.calls.is_empty());
    }

    #[test]
    fn llama_semicolon_separated_calls() {
        let got = parse_once(
            ToolCallFormat::Llama,
            &[],
            "{\"name\": \"a\", \"parameters\": {\"x\": 1}}; {\"name\": \"b\", \"parameters\": {\"y\": 2}}",
        );
        assert_eq!(
            got.calls,
            vec![
                ("a".into(), "{\"x\": 1}".into()),
                ("b".into(), "{\"y\": 2}".into()),
            ]
        );
        assert_eq!(got.content, "");
    }

    #[test]
    fn llama_invalid_json_degrades_to_content() {
        let text = "{not json at all";
        let got = parse_once(ToolCallFormat::Llama, &[], text);
        assert_eq!(got.content, text);
        assert!(got.calls.is_empty());
    }

    #[test]
    fn llama_python_tag_with_garbage_replays_tag_in_content() {
        let text = "<|python_tag|>print('hi')";
        let got = parse_once(ToolCallFormat::Llama, &[], text);
        assert_eq!(got.content, text);
        assert!(got.calls.is_empty());
    }

    #[test]
    fn qwen_xml_multiline_and_typed_params() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_tests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "verbose": {"type": "boolean"},
                        "retries": {"type": "integer"},
                    },
                },
            },
        })];
        let got = parse_once(
            ToolCallFormat::QwenXml,
            &tools,
            "<tool_call>\n<function=run_tests>\n<parameter=path>\na\nb\n</parameter>\n\
             <parameter=verbose>\nTrue\n</parameter>\n<parameter=retries>\n3\n</parameter>\n\
             </function>\n</tool_call>",
        );
        assert_eq!(
            got.calls,
            vec![(
                "run_tests".into(),
                "{\"path\":\"a\\nb\",\"verbose\":true,\"retries\":3}".into()
            )]
        );
    }

    #[test]
    fn qwen_xml_no_params_yields_empty_object() {
        let got = parse_once(
            ToolCallFormat::QwenXml,
            &[],
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
        );
        assert_eq!(got.calls, vec![("ping".into(), "{}".into())]);
    }

    #[test]
    fn qwen_xml_unknown_param_type_uses_json_detection() {
        let got = parse_once(
            ToolCallFormat::QwenXml,
            &[],
            "<tool_call>\n<function=f>\n<parameter=obj>\n{\"a\": 1}\n</parameter>\n\
             <parameter=word>\nhello\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            got.calls,
            vec![("f".into(), "{\"obj\":{\"a\": 1},\"word\":\"hello\"}".into())]
        );
    }

    #[test]
    fn whitespace_only_run_between_calls_is_dropped() {
        let got = parse_once(
            ToolCallFormat::Hermes,
            &[],
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n\
             <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>\n",
        );
        assert_eq!(got.content, "");
        assert_eq!(got.calls.len(), 2);
    }

    #[test]
    fn substantive_run_between_calls_is_kept_verbatim() {
        let got = parse_once(
            ToolCallFormat::Hermes,
            &[],
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\nand also\n\
             <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(got.content, "\nand also\n");
        assert_eq!(got.calls.len(), 2);
    }

    /// The Phase 4/5-style invariant, on synthetic edge inputs: every
    /// chunking reassembles identically to the single push. (The fixture
    /// suite in tests/ runs the same check over real model output.)
    #[test]
    fn chunk_split_invariance_on_edge_inputs() {
        let cases: Vec<(ToolCallFormat, &str)> = vec![
            (
                ToolCallFormat::Hermes,
                "x<tool_call>{\"name\": \"f\", \"arguments\": {\"s\": \"a\\\"b\"}}</tool_call>y",
            ),
            (ToolCallFormat::Hermes, "almost <tool_cal but not"),
            (
                ToolCallFormat::Llama,
                "<|python_tag|>{\"name\": \"f\", \"parameters\": {}}",
            ),
            (ToolCallFormat::Llama, "<|python_not_the_tag|> text"),
            (
                ToolCallFormat::QwenXml,
                "<tool_call>\n<function=f>\n<parameter=k>\nv1\nv2\n</parameter>\n</function>\n</tool_call>",
            ),
        ];
        for (format, text) in cases {
            let whole = parse_once(format, &[], text);
            for chunk_len in [1usize, 2, 3, 5, 7] {
                let mut parser = ToolCallParser::new(format, &[]);
                let mut events = Vec::new();
                let chars: Vec<char> = text.chars().collect();
                for chunk in chars.chunks(chunk_len) {
                    events.extend(parser.push(&chunk.iter().collect::<String>()));
                }
                events.extend(parser.finish());
                assert_eq!(
                    reassemble(&events),
                    whole,
                    "chunk_len={chunk_len} diverged for {text:?}"
                );
            }
        }
    }
}
