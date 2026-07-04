//! Streaming-detokenizer correctness against full decode (CLAUDE.md: the
//! incremental decoder is fuzz-tested against full decode; never emit
//! partial code points).
//!
//! Uses the pinned Llama-3.2 tokenizer (env-gated on `KILN_TEST_MODELS`):
//! byte-level BPE with multi-token UTF-8 for emoji/CJK — the hard cases.

use std::path::PathBuf;
use std::sync::Arc;

use kiln_tokenize::{StreamingDecoder, Tokenizer};

fn tokenizer() -> Option<Arc<Tokenizer>> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join("llama-3.2-1b-4bit");
    dir.join("tokenizer.json")
        .is_file()
        .then(|| Arc::new(Tokenizer::from_model_dir(&dir).expect("loads")))
}

/// Feeds `ids` in chunks of `sizes` (cycled); returns streamed text +
/// finalize tail.
fn stream_in_chunks(tokenizer: &Arc<Tokenizer>, ids: &[u32], sizes: &[usize]) -> String {
    let mut decoder = StreamingDecoder::new(Arc::clone(tokenizer));
    let mut out = String::new();
    let mut cursor = 0;
    let mut size_idx = 0;
    while cursor < ids.len() {
        let n = sizes[size_idx % sizes.len()].max(1).min(ids.len() - cursor);
        size_idx += 1;
        out.push_str(&decoder.push(&ids[cursor..cursor + n]).expect("push"));
        cursor += n;
    }
    out.push_str(&decoder.finalize().expect("finalize"));
    out
}

#[test]
fn streaming_matches_full_decode() {
    let Some(tokenizer) = tokenizer() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or model missing");
        return;
    };

    let corpora = [
        "The kiln is very hot today.",
        "the kiln 窯 is hot \u{1f525}\u{1f525} — múltiplos idiomas, знаки, 日本語のテキスト",
        "emoji storm \u{1f9e8}\u{1f525}\u{1f30b}\u{2764}\u{fe0f}\u{200d}\u{1f525} zwj sequences",
        "code: fn main() { println!(\"\\n\\t\"); } // stop\nnewlines\n\n\n",
        "ゼロ幅接合子と結合文字: e\u{301} n\u{303} ❤️‍🔥 👨‍👩‍👧‍👦",
    ];
    for text in corpora {
        let ids = tokenizer.encode(text, false).expect("encodes");
        let full = tokenizer.decode(&ids, false).expect("decodes");
        // Token-at-a-time is the hardest schedule: every multi-token
        // codepoint is split.
        assert_eq!(
            stream_in_chunks(&tokenizer, &ids, &[1]),
            full,
            "one-by-one: {text:?}"
        );
        for sizes in [&[2_usize, 1, 3][..], &[5][..], &[1, 7, 2][..]] {
            assert_eq!(
                stream_in_chunks(&tokenizer, &ids, sizes),
                full,
                "{sizes:?}: {text:?}"
            );
        }
    }
}

#[test]
fn streaming_matches_full_decode_fuzz() {
    let Some(tokenizer) = tokenizer() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or model missing");
        return;
    };
    let vocab = tokenizer.vocab_size() as u32;

    // Deterministic xorshift so failures reproduce; prints the seed on
    // failure via the assert message.
    let mut state = 0x9e3779b97f4a7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for case in 0..300 {
        let len = (next() % 48 + 1) as usize;
        let ids: Vec<u32> = (0..len)
            .map(|_| (next() % u64::from(vocab)) as u32)
            .collect();
        let full = tokenizer.decode(&ids, false).expect("decodes");

        let mut decoder = StreamingDecoder::new(Arc::clone(&tokenizer));
        let mut streamed = String::new();
        let mut cursor = 0;
        while cursor < ids.len() {
            let n = (next() % 4 + 1) as usize;
            let n = n.min(ids.len() - cursor);
            let piece = decoder.push(&ids[cursor..cursor + n]).expect("push");
            // Never emit an incomplete code point mid-stream.
            assert!(
                !piece.ends_with('\u{FFFD}') || full.ends_with('\u{FFFD}'),
                "case {case}: partial code point emitted mid-stream"
            );
            streamed.push_str(&piece);
            cursor += n;
        }
        streamed.push_str(&decoder.finalize().expect("finalize"));
        assert_eq!(streamed, full, "case {case}: ids {ids:?}");
    }
}
