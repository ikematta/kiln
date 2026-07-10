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

/// This path and the batched engine must share one canonical prefill
/// schedule (Phase 5 Option B, PROGRESS 2026-07-04): the batching suite
/// pins them bit-identical, and the prefix cache resumes on the
/// schedule's fine boundaries. See `kiln_engine::canonical_prefill_len`.
pub const PREFILL_FINE_CHUNK: usize = kiln_engine::DEFAULT_PREFILL_FINE_CHUNK;

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
    sample: impl FnMut(&Array, &Stream) -> Result<Array, MlxError>,
    s: &Stream,
) -> Result<GenerateOutput, ModelError> {
    generate_with(
        model,
        prompt,
        max_tokens,
        None::<fn(&[u32], &Array, &Stream) -> Result<Array, MlxError>>,
        sample,
        |_| true,
        s,
    )
}

/// [`generate`] with the worker-facing hooks:
///
/// - `process_logits(history, logits)` runs on the last-position logits
///   BEFORE logprob normalization (mlx-lm's logits-processor slot — where
///   penalties live). `history` follows mlx-lm's semantics: the last prompt
///   token plus everything generated so far, including the token being fed
///   this step. Because Kiln's penalties compute their token window
///   host-side, a `Some` processor forfeits the one-step async_eval
///   pipeline: each token is read back before the next step's graph is
///   built. `None` keeps the fully pipelined greedy/sampling path.
/// - `on_token(token)` fires per generated token (the worker streams a
///   `TokenChunk` and checks stop conditions here); returning `false` stops
///   generation. On the pipelined path one extra step is already scheduled
///   when it returns `false` — the ≤2-step cancel bound the proto promises.
pub fn generate_with<P, S, T>(
    model: &LlamaModel,
    prompt: &[u32],
    max_tokens: usize,
    mut process_logits: Option<P>,
    mut sample: S,
    mut on_token: T,
    s: &Stream,
) -> Result<GenerateOutput, ModelError>
where
    P: FnMut(&[u32], &Array, &Stream) -> Result<Array, MlxError>,
    S: FnMut(&Array, &Stream) -> Result<Array, MlxError>,
    T: FnMut(u32) -> bool,
{
    if prompt.is_empty() {
        return Err(ModelError::Mismatch("empty prompt".to_owned()));
    }
    if max_tokens == 0 {
        return Err(ModelError::Mismatch("max_tokens must be >= 1".to_owned()));
    }

    let mut caches = model.make_cache();
    let start = Instant::now();

    // Chunked prefill over everything but the last prompt token, on the
    // engine's canonical schedule INCLUDING its ADR 0002 padding rule:
    // sub-32-row ragged pieces off the super-chunk grid run with pad rows
    // (copies of the piece's last token; excluded from the caches by
    // `forward_padded`) so both paths stay bit-identical to each other and
    // to the mlx-lm reference at every prompt length — for QUANTIZED
    // checkpoints. Dense (unquantized) trunks are not bit-reproducible
    // under this fine-grid schedule (the engine runs them monolithic; see
    // AnyModel::monolithic_prefill_required); this path always uses the
    // fine grid and is only parity-meaningful for quantized models.
    let total = prompt.len();
    let mut processed = 0;
    while total - processed > 1 {
        let n = kiln_engine::canonical_prefill_len(
            processed,
            total - 1,
            PREFILL_CHUNK,
            PREFILL_FINE_CHUNK,
        );
        let pad = if n < kiln_engine::PREFILL_PAD_MIN_ROWS && processed % PREFILL_CHUNK != 0 {
            (kiln_engine::PREFILL_PAD_MIN_ROWS - n).min(processed)
        } else {
            0
        };
        let mut ids = prompt[processed..processed + n].to_vec();
        ids.extend(std::iter::repeat_n(ids[n - 1], pad));
        let chunk = Array::from_u32_slice(&ids, &[1, (n + pad) as i32])?;
        drop(model.forward_padded(&chunk, pad as i32, &mut caches, s)?);
        let state: Vec<&Array> = caches.iter().flat_map(KvCache::state).collect();
        eval(&state)?;
        processed += n;
        memory::clear_cache()?;
    }

    // One sampled step: forward -> last-position logits -> [processor] ->
    // logprobs -> token.
    let step = |tokens: &Array,
                caches: &mut Vec<KvCache>,
                history: &[u32],
                process_logits: &mut Option<P>,
                sample: &mut S|
     -> Result<Array, ModelError> {
        let logits = model.forward(tokens, caches, s)?;
        let (l, vocab) = (logits.dim(1), logits.dim(2));
        let last = ops::slice(&logits, &[0, l - 1, 0], &[1, l, vocab], s)?;
        let mut last = ops::reshape(&last, &[1, vocab], s)?;
        if let Some(process) = process_logits.as_mut() {
            last = process(history, &last, s)?;
        }
        // mlx-lm: logprobs = logits - logsumexp(logits, keepdims=True)
        let logprobs = ops::subtract(&last, &ops::logsumexp(&last, true, s)?, s)?;
        Ok(sample(&logprobs, s)?)
    };

    // mlx-lm processor history: the sampled steps' inputs — last prompt
    // token, then each generated token as it is fed back.
    let mut history: Vec<u32> = vec![prompt[total - 1]];
    let tail = Array::from_u32_slice(&prompt[total - 1..], &[1, 1])?;
    let mut y = step(
        &tail,
        &mut caches,
        &history,
        &mut process_logits,
        &mut sample,
    )?;
    async_eval(&[&y])?;

    let mut tokens = Vec::with_capacity(max_tokens);
    let mut prefill_seconds = 0.0;
    let mut decode_start = Instant::now();

    if process_logits.is_none() {
        // Pipelined: step i+1's graph is built and scheduled before token i
        // is read back.
        for i in 0..max_tokens {
            let next = if i + 1 < max_tokens {
                let y_in = ops::reshape(&y, &[1, 1], s)?;
                let next = step(
                    &y_in,
                    &mut caches,
                    &history,
                    &mut process_logits,
                    &mut sample,
                )?;
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
            if !on_token(token) {
                break;
            }
            match next {
                Some(next) => y = next,
                None => break,
            }
        }
    } else {
        // Sequential: the host-side penalty window needs token i's value
        // before step i+1's graph exists.
        for i in 0..max_tokens {
            let token = y.item_u32()?;
            if i == 0 {
                prefill_seconds = start.elapsed().as_secs_f64();
                decode_start = Instant::now();
            }
            tokens.push(token);
            history.push(token);
            if i.is_multiple_of(256) {
                memory::clear_cache()?;
            }
            if !on_token(token) || i + 1 == max_tokens {
                break;
            }
            let y_in = ops::reshape(&y, &[1, 1], s)?;
            y = step(
                &y_in,
                &mut caches,
                &history,
                &mut process_logits,
                &mut sample,
            )?;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    Ok(GenerateOutput {
        tokens,
        prefill_seconds,
        decode_seconds,
    })
}
