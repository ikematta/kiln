//! Incremental stop-string matching over streamed detokenized text.
//!
//! Rust port of the Python worker's `stops.py`, with the same contract —
//! for Rust workers the match runs in the GATEWAY (which owns
//! detokenization), not the worker. Text is pushed in arbitrary segments;
//! the matcher never releases text that is, or could become, part of a stop
//! string, and matched stop text is dropped entirely (SPEC §5: "matched
//! text excluded").

/// Stateful matcher for one request's stream.
#[derive(Debug)]
pub struct StopStringMatcher {
    stops: Vec<String>,
    /// Length of the longest stop string (bytes); bounds the holdback.
    stops_max: usize,
    buffer: String,
}

impl StopStringMatcher {
    pub fn new(stop_strings: &[String]) -> Self {
        let stops: Vec<String> = stop_strings
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let stops_max = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            stops_max,
            buffer: String::new(),
        }
    }

    /// Feeds a new text segment. Returns `(released, matched)`: `released`
    /// is text now safe to emit; `matched` is the stop string that fired, if
    /// any. After a match the matcher drops everything from the match onward.
    pub fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_owned(), None);
        }
        self.buffer.push_str(text);

        let mut earliest: Option<(usize, &str)> = None;
        for stop in &self.stops {
            if let Some(idx) = self.buffer.find(stop.as_str())
                && earliest.is_none_or(|(e, _)| idx < e)
            {
                earliest = Some((idx, stop));
            }
        }
        if let Some((idx, stop)) = earliest {
            let matched = stop.to_owned();
            let released = self.buffer[..idx].to_owned();
            self.buffer.clear();
            return (released, Some(matched));
        }

        // Hold back the longest buffer suffix that is a prefix of any stop
        // string (it may complete on the next push).
        let max_hold = self.buffer.len().min(self.stops_max.saturating_sub(1));
        let mut hold_len = 0;
        for k in (1..=max_hold).rev() {
            if !self.buffer.is_char_boundary(self.buffer.len() - k) {
                continue;
            }
            let suffix = &self.buffer[self.buffer.len() - k..];
            if self.stops.iter().any(|s| s.starts_with(suffix)) {
                hold_len = k;
                break;
            }
        }
        let released: String = self.buffer.drain(..self.buffer.len() - hold_len).collect();
        (released, None)
    }

    /// Releases any held text (call once the stream finished unmatched).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopStringMatcher {
        StopStringMatcher::new(&stops.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn no_stops_passes_through() {
        let mut m = matcher(&[]);
        assert_eq!(m.push("hello"), ("hello".into(), None));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn match_within_one_segment_excludes_stop_text() {
        let mut m = matcher(&["STOP"]);
        let (released, matched) = m.push("before STOP after");
        assert_eq!(released, "before ");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn match_split_across_segments() {
        let mut m = matcher(&["STOP"]);
        let (released, matched) = m.push("text ST");
        assert_eq!(released, "text ");
        assert_eq!(matched, None);
        let (released, matched) = m.push("OP tail");
        assert_eq!(released, "");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn false_prefix_is_released_on_disambiguation() {
        let mut m = matcher(&["STOP"]);
        let (released, matched) = m.push("a ST");
        assert_eq!((released.as_str(), matched), ("a ", None));
        let (released, matched) = m.push("ART");
        assert_eq!((released.as_str(), matched), ("START", None));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn earliest_of_multiple_stops_wins() {
        let mut m = matcher(&["zzz", "bb"]);
        let (released, matched) = m.push("a bb then zzz");
        assert_eq!(released, "a ");
        assert_eq!(matched.as_deref(), Some("bb"));
    }

    #[test]
    fn flush_releases_held_prefix() {
        let mut m = matcher(&["\n\n"]);
        let (released, matched) = m.push("line\n");
        assert_eq!((released.as_str(), matched), ("line", None));
        assert_eq!(m.flush(), "\n");
    }

    #[test]
    fn multibyte_boundaries_are_respected() {
        let mut m = matcher(&["窯止"]);
        let (released, matched) = m.push("hot 窯");
        assert_eq!((released.as_str(), matched), ("hot ", None));
        let (released, matched) = m.push("止 rest");
        assert_eq!(released, "");
        assert_eq!(matched.as_deref(), Some("窯止"));
    }
}
