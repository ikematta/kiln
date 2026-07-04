//! Token sampling on-GPU with MLX ops (SPEC §6.6).
//!
//! Greedy (`temperature == 0`) is `argmax` over the logprobs — the
//! golden-parity path. The stochastic chain applies top-p → min-p → top-k
//! filters then a categorical draw, mirroring `mlx_lm.sample_utils` filter
//! semantics; randomness comes from a per-request key derived from the
//! request seed (never MLX's global RNG state), so a given (seed, logprobs
//! stream) is reproducible per worker version.
//!
//! Repetition/presence/frequency penalties ([`apply_penalties`]) operate on
//! *logits before logprob normalization* over the request's recent-token
//! window, exactly like mlx-lm's logits processors; the engine loop wires
//! them ahead of the sampler (Phase 4's per-request pipeline, SPEC §6.2).

use kiln_mlx::{Array, Dtype, MlxError, Stream, ops, random};

/// Sampling knobs, mirroring `SamplingParams` in `worker.proto` (disabled
/// values follow the proto conventions: `top_p >= 1.0`, `top_k == 0`,
/// `min_p == 0.0` are no-ops).
#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
        }
    }
}

/// Stateful sampler: holds the PRNG key chain for one request.
#[derive(Debug)]
pub struct Sampler {
    options: SamplingOptions,
    key: Option<Array>,
}

impl Sampler {
    /// Greedy argmax sampler (temperature 0).
    pub fn greedy() -> Self {
        Self::new(SamplingOptions::default())
    }

    pub fn new(options: SamplingOptions) -> Self {
        Self { options, key: None }
    }

    /// Samples one token id (`[1]` u32) from logprobs `[1, vocab]`.
    pub fn sample(&mut self, logprobs: &Array, s: &Stream) -> Result<Array, MlxError> {
        let opts = self.options;
        if opts.temperature == 0.0 {
            return ops::argmax(logprobs, -1, false, s);
        }

        let mut x = logprobs.clone();
        if opts.top_p > 0.0 && opts.top_p < 1.0 {
            x = apply_top_p(&x, opts.top_p, s)?;
        }
        if opts.min_p != 0.0 {
            x = apply_min_p(&x, opts.min_p, s)?;
        }
        let vocab = x.dim(-1);
        if opts.top_k > 0 && (opts.top_k as i32) < vocab {
            x = apply_top_k(&x, opts.top_k as i32, s)?;
        }

        // categorical_sampling(logprobs, temp): draw over logprobs / temp.
        let scaled = ops::multiply(&x, &Array::from_f32(1.0 / opts.temperature), s)?;
        let key = self.next_key(s)?;
        random::categorical(&scaled, &key, s)
    }

    /// Splits the request key chain and returns this draw's subkey.
    fn next_key(&mut self, s: &Stream) -> Result<Array, MlxError> {
        let key = match self.key.take() {
            Some(key) => key,
            None => random::key(self.options.seed)?,
        };
        let (next, sub) = random::split(&key, s)?;
        self.key = Some(next);
        Ok(sub)
    }
}

/// `-inf` in the dtype of `like` (keeps filters from promoting f16 logprobs).
fn neg_inf_like(like: &Array, s: &Stream) -> Result<Array, MlxError> {
    let dtype = like.dtype().unwrap_or(Dtype::Float32);
    ops::astype(&Array::from_f32(f32::NEG_INFINITY), dtype, s)
}

/// Nucleus filter: keep the smallest set of tokens whose cumulative
/// probability exceeds `top_p`; the rest go to `-inf`.
fn apply_top_p(logprobs: &Array, top_p: f32, s: &Stream) -> Result<Array, MlxError> {
    let probs = ops::exp(logprobs, s)?;
    let sorted_indices = ops::argsort(logprobs, -1, s)?; // ascending
    let sorted_probs = ops::take_along_axis(&probs, &sorted_indices, -1, s)?;
    let cumulative = ops::cumsum(&sorted_probs, -1, false, true, s)?;

    // Rearrange cumulative probs back to vocabulary order.
    let vocab = sorted_indices.dim(-1);
    let positions = ops::arange(0.0, f64::from(vocab), 1.0, Dtype::Uint32, s)?;
    let positions = ops::reshape(&positions, &[1, vocab], s)?;
    let zeros = ops::zeros(&[1, vocab], Dtype::Uint32, s)?;
    let inverse = ops::put_along_axis(&zeros, &sorted_indices, &positions, -1, s)?;
    let cumulative = ops::take_along_axis(&cumulative, &inverse, -1, s)?;

    let keep = ops::greater(&cumulative, &Array::from_f32(1.0 - top_p), s)?;
    ops::where_cond(&keep, logprobs, &neg_inf_like(logprobs, s)?, s)
}

/// Min-p filter: drop tokens whose probability is below `min_p` times the
/// top token's probability.
fn apply_min_p(logprobs: &Array, min_p: f32, s: &Stream) -> Result<Array, MlxError> {
    let top = ops::max(logprobs, -1, true, s)?;
    let threshold = ops::add(&top, &Array::from_f32(min_p.ln()), s)?;
    let remove = ops::less(logprobs, &threshold, s)?;
    ops::where_cond(&remove, &neg_inf_like(logprobs, s)?, logprobs, s)
}

/// Top-k filter: everything outside the k most probable tokens goes to
/// `-inf`.
fn apply_top_k(logprobs: &Array, k: i32, s: &Stream) -> Result<Array, MlxError> {
    let vocab = logprobs.dim(-1);
    let neg = ops::negative(logprobs, s)?;
    let by_rank = ops::argpartition(&neg, k - 1, -1, s)?;
    let mask_idx = ops::slice(&by_rank, &[0, k], &[1, vocab], s)?;
    ops::put_along_axis(logprobs, &mask_idx, &neg_inf_like(logprobs, s)?, -1, s)
}

/// Penalty knobs, mirroring `SamplingParams` (1.0 / 0.0 / 0.0 = disabled).
#[derive(Debug, Clone, Copy)]
pub struct PenaltyOptions {
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl PenaltyOptions {
    pub fn is_disabled(&self) -> bool {
        self.repetition_penalty == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
    }
}

/// Applies repetition/presence/frequency penalties to raw logits
/// (`[1, vocab]`) for the tokens in `recent` (the request's window,
/// SPEC §6.6), gathered/scattered on-GPU like mlx-lm's processors:
/// - repetition: `l < 0 ? l * p : l / p` (CTRL-style, sign-aware),
/// - presence: subtract `p` once per distinct recent token,
/// - frequency: subtract `p * count(token in recent)`.
pub fn apply_penalties(
    logits: &Array,
    recent: &[u32],
    options: PenaltyOptions,
    s: &Stream,
) -> Result<Array, MlxError> {
    if recent.is_empty() || options.is_disabled() {
        return Ok(logits.clone());
    }
    // Distinct tokens + occurrence counts, computed host-side (the window is
    // tiny); scatter with distinct indices keeps put_along_axis well-defined.
    let mut distinct: Vec<u32> = Vec::new();
    let mut counts: Vec<f32> = Vec::new();
    for &token in recent {
        match distinct.iter().position(|&t| t == token) {
            Some(i) => counts[i] += 1.0,
            None => {
                distinct.push(token);
                counts.push(1.0);
            }
        }
    }
    let n = distinct.len() as i32;
    let indices = Array::from_u32_slice(&distinct, &[1, n])?;

    let mut selected = ops::take_along_axis(logits, &indices, -1, s)?;
    if options.repetition_penalty != 1.0 {
        let p = Array::from_f32(options.repetition_penalty);
        let scaled_down = ops::divide(&selected, &p, s)?;
        let scaled_up = ops::multiply(&selected, &p, s)?;
        let negative = ops::less(&selected, &Array::from_f32(0.0), s)?;
        selected = ops::where_cond(&negative, &scaled_up, &scaled_down, s)?;
    }
    if options.presence_penalty != 0.0 {
        selected = ops::subtract(&selected, &Array::from_f32(options.presence_penalty), s)?;
    }
    if options.frequency_penalty != 0.0 {
        let counts = Array::from_f32_slice(&counts, &[1, n])?;
        let penalty = ops::multiply(&counts, &Array::from_f32(options.frequency_penalty), s)?;
        selected = ops::subtract(&selected, &penalty, s)?;
    }
    let dtype = logits.dtype().unwrap_or(Dtype::Float32);
    let selected = ops::astype(&selected, dtype, s)?;
    ops::put_along_axis(logits, &indices, &selected, -1, s)
}
