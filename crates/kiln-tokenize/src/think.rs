//! Streaming `<think>` block extraction (SPEC §8.1): models trained to
//! reason before answering (Qwen3, R1 distills) emit their reasoning
//! wrapped in `<think>…</think>` tags. The Anthropic `/v1/messages`
//! adapter surfaces those regions as proper `thinking` content blocks;
//! this parser splits the detokenized text stream into thinking and text
//! runs, tag markers removed.
//!
//! Whether a model emits the tags is detected from its chat template
//! ([`crate::ChatTemplate::emits_think_tags`]) — thinking-trained
//! templates reference `</think>` (history stripping, `enable_thinking`
//! handling). Models without the marker never get a parser, so user text
//! that happens to contain `<think>` is never misclassified.
//!
//! # Semantics
//!
//! - Output is identical however the input is chunked (the same
//!   chunk-split invariance bar as the tool-call parsers); partial tags
//!   spanning pushes are held back until they resolve.
//! - Tag markers are removed. Whitespace adjacent to a tag boundary is
//!   trimmed — the run before `<think>`, the run after it, the run before
//!   `</think>`, and the run after it — so block contents start and end
//!   on substance. Interior whitespace is untouched, and a response with
//!   no think tags streams through byte-for-byte (whitespace runs may be
//!   held until the next push, released verbatim).
//! - A `<think>` left unclosed at end-of-stream (length truncation)
//!   yields a truncated thinking run; an unresolved partial tag replays
//!   as literal content of the current mode.

/// One classified segment of the output stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkEvent {
    /// Reasoning text from inside a `<think>` block (tags excluded).
    Thinking(String),
    /// Ordinary response text.
    Text(String),
}

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Text,
    Think,
}

/// Streaming splitter for one response.
#[derive(Debug)]
pub struct ThinkParser {
    mode: Mode,
    /// Unemitted tail: at most a whitespace run plus a partial tag prefix.
    buffer: String,
    /// Drop leading whitespace of the next emission (set right after a
    /// tag so blocks don't start with the model's cosmetic newlines).
    skip_leading: bool,
}

impl Default for ThinkParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkParser {
    pub fn new() -> Self {
        Self {
            mode: Mode::Text,
            buffer: String::new(),
            skip_leading: false,
        }
    }

    /// Feeds the next text segment; returns the events it completes.
    pub fn push(&mut self, text: &str) -> Vec<ThinkEvent> {
        self.buffer.push_str(text);
        let mut out = Vec::new();
        loop {
            let tag = match self.mode {
                Mode::Text => OPEN_TAG,
                Mode::Think => CLOSE_TAG,
            };
            if let Some(idx) = self.buffer.find(tag) {
                let head = self.buffer[..idx].trim_end().to_owned();
                self.emit(head, &mut out);
                self.buffer.drain(..idx + tag.len());
                self.mode = match self.mode {
                    Mode::Text => Mode::Think,
                    Mode::Think => Mode::Text,
                };
                self.skip_leading = true;
                continue;
            }
            // No full tag: hold back the longest suffix that could still
            // become one — a partial tag prefix, plus the whitespace run
            // before it (trimmed if the tag completes on a later push).
            let partial = longest_suffix_prefixing(&self.buffer, tag);
            let kept = self.buffer.len() - partial;
            let ws = self.buffer[..kept].len() - self.buffer[..kept].trim_end().len();
            let safe: String = self.buffer.drain(..kept - ws).collect();
            self.emit(safe, &mut out);
            return out;
        }
    }

    /// End-of-stream: an unresolved hold is literal content. In text mode
    /// it flushes verbatim (parity with the un-parsed stream); a truncated
    /// thinking block drops its trailing whitespace (block boundary).
    pub fn finish(&mut self) -> Vec<ThinkEvent> {
        let tail = std::mem::take(&mut self.buffer);
        let tail = match self.mode {
            Mode::Text => tail,
            Mode::Think => tail.trim_end().to_owned(),
        };
        let mut out = Vec::new();
        self.emit(tail, &mut out);
        out
    }

    fn emit(&mut self, mut text: String, out: &mut Vec<ThinkEvent>) {
        if self.skip_leading {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                return;
            }
            text = trimmed.to_owned();
            self.skip_leading = false;
        }
        if text.is_empty() {
            return;
        }
        match (self.mode, out.last_mut()) {
            (Mode::Text, Some(ThinkEvent::Text(prev))) => prev.push_str(&text),
            (Mode::Think, Some(ThinkEvent::Thinking(prev))) => prev.push_str(&text),
            (Mode::Text, _) => out.push(ThinkEvent::Text(text)),
            (Mode::Think, _) => out.push(ThinkEvent::Thinking(text)),
        }
    }
}

/// Length of the longest strict suffix of `text` that is a proper prefix
/// of `tag` (0 when none) — the bytes that could still grow into the tag.
fn longest_suffix_prefixing(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len() - 1);
    for k in (1..=max).rev() {
        if !text.is_char_boundary(text.len() - k) {
            continue;
        }
        if tag
            .as_bytes()
            .starts_with(&text.as_bytes()[text.len() - k..])
        {
            return k;
        }
    }
    0
}

/// Whether a chat template belongs to a thinking-trained model: every such
/// template references the closing tag (history stripping, non-thinking
/// injection), and non-thinking templates never do.
pub fn template_emits_think_tags(template_source: &str) -> bool {
    template_source.contains(CLOSE_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the parser over `input` split into `size`-char chunks and
    /// merges adjacent same-kind events for comparison.
    fn replay(input: &str, size: usize) -> Vec<ThinkEvent> {
        let mut parser = ThinkParser::new();
        let mut events = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        for chunk in chars.chunks(size) {
            events.extend(parser.push(&chunk.iter().collect::<String>()));
        }
        events.extend(parser.finish());
        merge(events)
    }

    fn merge(events: Vec<ThinkEvent>) -> Vec<ThinkEvent> {
        let mut out: Vec<ThinkEvent> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (Some(ThinkEvent::Text(prev)), ThinkEvent::Text(t)) => prev.push_str(&t),
                (Some(ThinkEvent::Thinking(prev)), ThinkEvent::Thinking(t)) => prev.push_str(&t),
                (_, event) => out.push(event),
            }
        }
        out
    }

    fn text(s: &str) -> ThinkEvent {
        ThinkEvent::Text(s.to_owned())
    }

    fn think(s: &str) -> ThinkEvent {
        ThinkEvent::Thinking(s.to_owned())
    }

    /// Every case must produce identical output at every chunk size —
    /// the same invariance bar as the tool-call parser suite.
    fn assert_all_splits(input: &str, expected: &[ThinkEvent]) {
        for size in [1, 2, 3, 5, 7, 11, input.len().max(1)] {
            assert_eq!(replay(input, size), expected, "chunk size {size}");
        }
    }

    #[test]
    fn qwen3_shape_splits_into_thinking_then_text() {
        assert_all_splits(
            "<think>\nThe user wants a greeting.\n</think>\n\nHello!",
            &[think("The user wants a greeting."), text("Hello!")],
        );
    }

    #[test]
    fn no_tags_passes_through_verbatim() {
        assert_all_splits(
            "plain answer, no tags\n",
            &[text("plain answer, no tags\n")],
        );
    }

    #[test]
    fn text_before_and_between_blocks_is_preserved() {
        assert_all_splits(
            "intro <think>a</think> middle <think>b</think> outro",
            &[
                text("intro"),
                think("a"),
                text("middle"),
                think("b"),
                text("outro"),
            ],
        );
    }

    #[test]
    fn unclosed_think_truncates_as_thinking() {
        assert_all_splits(
            "<think>\nreasoning that got cut of",
            &[think("reasoning that got cut of")],
        );
    }

    #[test]
    fn partial_open_tag_at_eos_is_literal_text() {
        assert_all_splits("answer ends with <thi", &[text("answer ends with <thi")]);
    }

    #[test]
    fn partial_close_tag_inside_think_is_literal_thinking() {
        assert_all_splits("<think>uses </thi", &[think("uses </thi")]);
    }

    #[test]
    fn false_prefix_disambiguates_and_releases() {
        assert_all_splits(
            "a <thinker> is not a tag",
            &[text("a <thinker> is not a tag")],
        );
    }

    #[test]
    fn empty_think_block_yields_nothing() {
        // Qwen3's template injects `<think>\n\n</think>` for non-thinking
        // mode; if the model echoes an empty block, no thinking event.
        assert_all_splits("<think>\n\n</think>\n\nanswer", &[text("answer")]);
    }

    #[test]
    fn tags_inside_think_content_are_content() {
        assert_all_splits(
            "<think>nested <think> stays</think>done",
            &[think("nested <think> stays"), text("done")],
        );
    }

    #[test]
    fn multibyte_boundaries_are_respected() {
        assert_all_splits(
            "<think>窯は熱い</think>窯 kiln",
            &[think("窯は熱い"), text("窯 kiln")],
        );
    }

    #[test]
    fn interior_whitespace_survives_boundary_trim() {
        assert_all_splits(
            "<think>line one\n\nline two</think>para one\n\npara two\n",
            &[
                think("line one\n\nline two"),
                text("para one\n\npara two\n"),
            ],
        );
    }

    #[test]
    fn template_detection() {
        assert!(template_emits_think_tags(
            "{%- if '</think>' in content %}{{ content.split('</think>')[-1] }}{%- endif %}"
        ));
        assert!(!template_emits_think_tags(
            "{{ bos_token }}{% for message in messages %}...{% endfor %}"
        ));
    }
}
