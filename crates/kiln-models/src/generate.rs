//! Single-request generation loop — the Phase 3 v0 decode path (the
//! continuous-batching engine replaces this in Phase 4, SPEC §6.2).
//!
//! Control flow mirrors `mlx_lm.generate.generate_step` so both timing
//! characteristics and greedy outputs match the reference:
//! - chunked prefill over `prompt[..n-1]` (logits of prefill positions are
//!   never evaluated — the lm_head matmul for them is dead graph), evaluating
//!   only the KV state per chunk;
//! - the last prompt token runs through the first sampled step;
//! - decode pipelining via `async_eval`: step `i+1`'s graph is built and
//!   scheduled before step `i`'s token is read back;
//! - `clear_cache()` after each prefill chunk and every 256 decode steps.
//!
//! No EOS/stop handling here by design: fixtures compare exactly
//! `max_tokens` greedy tokens (see `scripts/gen-golden.py`); stop conditions
//! are worker-level concerns layered on top later.

use std::time::Instant;

use kiln_mlx::{Array, MlxError, Stream, async_eval, eval, memory, ops};

use crate::kv_cache::KvCache;
use crate::llama::{LlamaModel, ModelError};

/// mlx-lm's `prefill_step_size` (SPEC §6.2 default `prefill_chunk` is 2048
/// too).
pub const PREFILL_CHUNK: usize = 2048;

/// Result of a [`generate`] run.
#[derive(Debug)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    /// Wall time from submit to the first token being readable.
    pub prefill_seconds: f64,
    /// Wall time for the remaining `tokens.len() - 1` decode steps.
    pub decode_seconds: f64,
}

impl GenerateOutput {
    pub fn decode_tokens_per_sec(&self) -> f64 {
        let n = self.tokens.len().saturating_sub(1);
        if self.decode_seconds > 0.0 {
            n as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
}

/// Generates exactly `max_tokens` tokens from `prompt` with `sample`
/// (logprobs `[1, vocab]` -> sampled token `[1]` u32).
pub fn generate(
    model: &LlamaModel,
    prompt: &[u32],
    max_tokens: usize,
    mut sample: impl FnMut(&Array, &Stream) -> Result<Array, MlxError>,
    s: &Stream,
) -> Result<GenerateOutput, ModelError> {
    if prompt.is_empty() {
        return Err(ModelError::Mismatch("empty prompt".to_owned()));
    }
    if max_tokens == 0 {
        return Err(ModelError::Mismatch("max_tokens must be >= 1".to_owned()));
    }

    let mut caches = model.make_cache();
    let start = Instant::now();

    // Chunked prefill over everything but the last prompt token.
    let total = prompt.len();
    let mut processed = 0;
    while total - processed > 1 {
        let n = (total - processed - 1).min(PREFILL_CHUNK);
        let chunk = Array::from_u32_slice(&prompt[processed..processed + n], &[1, n as i32])?;
        drop(model.forward(&chunk, &mut caches, s)?);
        let state: Vec<&Array> = caches.iter().flat_map(KvCache::state).collect();
        eval(&state)?;
        processed += n;
        memory::clear_cache()?;
    }

    // One sampled step: forward -> last-position logits -> logprobs -> token.
    let mut step = |tokens: &Array, caches: &mut Vec<KvCache>| -> Result<Array, ModelError> {
        let logits = model.forward(tokens, caches, s)?;
        let (l, vocab) = (logits.dim(1), logits.dim(2));
        let last = ops::slice(&logits, &[0, l - 1, 0], &[1, l, vocab], s)?;
        let last = ops::reshape(&last, &[1, vocab], s)?;
        // mlx-lm: logprobs = logits - logsumexp(logits, keepdims=True)
        let logprobs = ops::subtract(&last, &ops::logsumexp(&last, true, s)?, s)?;
        Ok(sample(&logprobs, s)?)
    };

    let tail = Array::from_u32_slice(&prompt[total - 1..], &[1, 1])?;
    let mut y = step(&tail, &mut caches)?;
    async_eval(&[&y])?;

    let mut tokens = Vec::with_capacity(max_tokens);
    let mut prefill_seconds = 0.0;
    let mut decode_start = Instant::now();
    for i in 0..max_tokens {
        // Build + schedule the next step before blocking on this token.
        let next = if i + 1 < max_tokens {
            let y_in = ops::reshape(&y, &[1, 1], s)?;
            let next = step(&y_in, &mut caches)?;
            async_eval(&[&next])?;
            Some(next)
        } else {
            None
        };
        let token = y.item_u32()?;
        if i == 0 {
            prefill_seconds = start.elapsed().as_secs_f64();
            decode_start = Instant::now();
        }
        tokens.push(token);
        if i.is_multiple_of(256) {
            memory::clear_cache()?;
        }
        if let Some(next) = next {
            y = next;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    Ok(GenerateOutput {
        tokens,
        prefill_seconds,
        decode_seconds,
    })
}
