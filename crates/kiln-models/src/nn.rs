//! Shared model building blocks (the analogues of `mlx.nn` layers +
//! `mlx_lm.models.rope_utils` that every architecture module composes).
//!
//! Everything here is ported op-for-op from the pinned reference — same MLX
//! kernels in the same order, so greedy decoding stays bit-identical to the
//! golden fixtures (SPEC §11.2). Do not "improve" numerics here without
//! re-running the golden harness.

use kiln_engine::{PagedKv, SeqStep, StepBatch, StepInput};
use kiln_mlx::fast::{self, SdpaMask};
use kiln_mlx::{Array, Dtype, MlxError, Stream, ops};

use crate::config::{ConfigError, Quantization, RopeScaling};
use crate::kv_cache::KvCache;
use crate::weights::{WeightStore, WeightsError};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error("weights/config mismatch: {0}")]
    Mismatch(String),
}

/// Linear projection: affine-quantized (mlx-lm `QuantizedLinear`) or dense.
/// A module is quantized iff its `.scales` tensor exists in the checkpoint —
/// that is how mlx-lm's `class_predicate` decides too, so mixed-precision
/// checkpoints resolve per layer.
#[derive(Debug)]
pub(crate) enum Linear {
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    },
    Dense {
        weight: Array,
        bias: Option<Array>,
    },
}

impl Linear {
    pub(crate) fn load(
        store: &mut WeightStore,
        prefix: &str,
        quantization: Option<Quantization>,
    ) -> Result<Self, ModelError> {
        let bias = store.take_optional(&format!("{prefix}.bias"));
        if store.contains(&format!("{prefix}.scales")) {
            let q = quantization.ok_or_else(|| {
                ModelError::Mismatch(format!(
                    "{prefix} has quantized tensors but config.json has no quantization block"
                ))
            })?;
            Ok(Linear::Quantized {
                weight: store.take(&format!("{prefix}.weight"))?,
                scales: store.take(&format!("{prefix}.scales"))?,
                biases: store.take(&format!("{prefix}.biases"))?,
                bias,
                group_size: q.group_size,
                bits: q.bits,
            })
        } else {
            Ok(Linear::Dense {
                weight: store.take(&format!("{prefix}.weight"))?,
                bias,
            })
        }
    }

    pub(crate) fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
        let (y, bias) = match self {
            Linear::Quantized {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            } => (
                ops::quantized_matmul(x, weight, scales, biases, true, *group_size, *bits, s)?,
                bias,
            ),
            Linear::Dense { weight, bias } => {
                // nn.Linear: x @ weight.T
                let wt = ops::transpose(weight, &[1, 0], s)?;
                (ops::matmul(x, &wt, s)?, bias)
            }
        };
        match bias {
            Some(b) => ops::add(&y, b, s),
            None => Ok(y),
        }
    }

    /// `(in_features, out_features)` — the shape key for the calibration
    /// probe's dedup.
    fn io_features(&self) -> (i32, i32) {
        match self {
            Linear::Quantized {
                scales, group_size, ..
            } => (scales.dim(1) * group_size, scales.dim(0)),
            Linear::Dense { weight, .. } => (weight.dim(1), weight.dim(0)),
        }
    }
}

/// Token embedding, quantized (mlx-lm `QuantizedEmbedding`) or dense; also
/// serves as the tied `lm_head` via [`Self::as_linear`].
#[derive(Debug)]
pub(crate) enum Embedding {
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
    Dense {
        weight: Array,
    },
}

impl Embedding {
    pub(crate) fn load(
        store: &mut WeightStore,
        prefix: &str,
        quantization: Option<Quantization>,
    ) -> Result<Self, ModelError> {
        if store.contains(&format!("{prefix}.scales")) {
            let q = quantization.ok_or_else(|| {
                ModelError::Mismatch(format!(
                    "{prefix} has quantized tensors but config.json has no quantization block"
                ))
            })?;
            Ok(Embedding::Quantized {
                weight: store.take(&format!("{prefix}.weight"))?,
                scales: store.take(&format!("{prefix}.scales"))?,
                biases: store.take(&format!("{prefix}.biases"))?,
                group_size: q.group_size,
                bits: q.bits,
            })
        } else {
            Ok(Embedding::Dense {
                weight: store.take(&format!("{prefix}.weight"))?,
            })
        }
    }

    /// `ids [B, L] -> [B, L, hidden]`.
    pub(crate) fn lookup(&self, ids: &Array, s: &Stream) -> Result<Array, MlxError> {
        match self {
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // QuantizedEmbedding: gather packed rows, dequantize them.
                let w = ops::take(weight, ids, 0, s)?;
                let sc = ops::take(scales, ids, 0, s)?;
                let bi = ops::take(biases, ids, 0, s)?;
                ops::dequantize(&w, &sc, &bi, *group_size, *bits, s)
            }
            Embedding::Dense { weight } => ops::take(weight, ids, 0, s),
        }
    }

    /// `h [B, L, hidden] -> logits [B, L, vocab]` (tied word embeddings).
    pub(crate) fn as_linear(&self, h: &Array, s: &Stream) -> Result<Array, MlxError> {
        match self {
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => ops::quantized_matmul(h, weight, scales, biases, true, *group_size, *bits, s),
            Embedding::Dense { weight } => {
                let wt = ops::transpose(weight, &[1, 0], s)?;
                ops::matmul(h, &wt, s)
            }
        }
    }
}

/// A Python-float scalar as MLX's weak-promotion rules see it: the value is
/// held in float32 and cast to the tensor operand's dtype at the op, so the
/// op itself runs in the tensor dtype (mirroring `0.5 * x` on an f16 `x` in
/// the reference — a plain f32 Kiln scalar would promote the op to f32
/// instead and change bits).
fn weak_scalar(v: f64, like: &Array, s: &Stream) -> Result<Array, MlxError> {
    let dtype = like.dtype().ok_or_else(|| MlxError {
        message: "weak scalar against an array with an unsupported dtype".to_owned(),
    })?;
    ops::astype(&Array::from_f32(v as f32), dtype, s)
}

/// Gated-MLP activation — which `mlx.nn` function the reference architecture
/// applies to the gate projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activation {
    /// llama/qwen: `silu(gate) * up`.
    Silu,
    /// gemma2/gemma3: `gelu_approx(gate) * up`.
    GeluApprox,
}

/// `mlx.nn.gelu_approx` (tanh approximation), op-for-op in the reference's
/// evaluation order: `(0.5 * x) * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
/// x**3)))`, all scalars weak (op dtype = x dtype).
fn gelu_approx(x: &Array, s: &Stream) -> Result<Array, MlxError> {
    let cube = ops::power(x, &weak_scalar(3.0, x, s)?, s)?;
    let inner = ops::add(
        x,
        &ops::multiply(&weak_scalar(0.044715, x, s)?, &cube, s)?,
        s,
    )?;
    let scaled = ops::multiply(
        &weak_scalar((2.0 / std::f64::consts::PI).sqrt(), x, s)?,
        &inner,
        s,
    )?;
    let gate = ops::add(&weak_scalar(1.0, x, s)?, &ops::tanh(&scaled, s)?, s)?;
    ops::multiply(&ops::multiply(&weak_scalar(0.5, x, s)?, x, s)?, &gate, s)
}

/// How an architecture parameterizes RMSNorm weights. Gemma stores `w` and
/// applies `rms_norm(x, 1 + w)`; the `1 + w` is folded in once at load (the
/// same add the reference issues per call — identical bits, fewer ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormStyle {
    Plain,
    OnePlus,
}

impl NormStyle {
    fn apply(self, w: Array, s: &Stream) -> Result<Array, ModelError> {
        match self {
            NormStyle::Plain => Ok(w),
            NormStyle::OnePlus => {
                let effective = ops::add(&weak_scalar(1.0, &w, s)?, &w, s)?;
                effective.eval()?;
                Ok(effective)
            }
        }
    }
}

/// The gated MLP shared by llama/qwen2/qwen3 (`down(silu(gate(x)) * up(x))`)
/// and gemma2/gemma3 (`down(gelu_approx(gate(x)) * up(x))`).
#[derive(Debug)]
pub(crate) struct Mlp {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    pub(crate) fn load(
        store: &mut WeightStore,
        prefix: &str,
        quantization: Option<Quantization>,
        activation: Activation,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            gate_proj: Linear::load(store, &format!("{prefix}.gate_proj"), quantization)?,
            up_proj: Linear::load(store, &format!("{prefix}.up_proj"), quantization)?,
            down_proj: Linear::load(store, &format!("{prefix}.down_proj"), quantization)?,
            activation,
        })
    }

    pub(crate) fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
        let gate = self.gate_proj.forward(x, s)?;
        let up = self.up_proj.forward(x, s)?;
        let activated = match self.activation {
            // swiglu(gate, up) = silu(gate) * up ; silu(x) = x * sigmoid(x)
            Activation::Silu => ops::multiply(&gate, &ops::sigmoid(&gate, s)?, s)?,
            Activation::GeluApprox => gelu_approx(&gate, s)?,
        };
        self.down_proj
            .forward(&ops::multiply(&activated, &up, s)?, s)
    }
}

/// Finds the smallest M in `2..=32` at which `f`'s row 0 stops being
/// bit-identical to its M=1 result (33 when it never diverges) — one
/// shape's contribution to [`CausalLm::calibrate_deterministic_width`].
/// `base` is a `[1, 1, hidden]` activation row, tiled to `k` features.
fn probe_row_stability(
    k: i32,
    base: &Array,
    s: &Stream,
    f: impl Fn(&Array) -> Result<Array, MlxError>,
) -> Result<usize, ModelError> {
    let reps = ((k + base.dim(2) - 1) / base.dim(2)) as usize;
    let cats: Vec<&Array> = std::iter::repeat_n(base, reps).collect();
    let wide = ops::concatenate(&cats, 2, s)?;
    let x1 = ops::contiguous(&ops::slice(&wide, &[0, 0, 0], &[1, 1, k], s)?, s)?;
    x1.eval()?;
    let row0 = |y: &Array| -> Result<Vec<u8>, MlxError> {
        let n = y.dim(2);
        let row = ops::contiguous(&ops::slice(y, &[0, 0, 0], &[1, 1, n], s)?, s)?;
        row.eval()?;
        row.data_raw_bytes()
    };
    let reference = row0(&f(&x1)?)?;
    for m in 2..=32 {
        let rows: Vec<&Array> = std::iter::repeat_n(&x1, m).collect();
        let xm = ops::contiguous(&ops::concatenate(&rows, 1, s)?, s)?;
        if row0(&f(&xm)?)? != reference {
            return Ok(m);
        }
    }
    Ok(33)
}

/// Prepends `pad` zero rows along `axis` (ADR 0002 kernel-class padding);
/// the exact input when `pad == 0` — no graph nodes added.
fn pad_front(a: &Array, pad: i32, axis: usize, s: &Stream) -> Result<Array, MlxError> {
    if pad == 0 {
        return Ok(a.clone());
    }
    let mut shape = a.shape();
    shape[axis] = pad;
    let dtype = a.dtype().ok_or_else(|| MlxError {
        message: "pad rows requested for an array with an unsupported dtype".to_owned(),
    })?;
    let zeros = ops::zeros(&shape, dtype, s)?;
    ops::concatenate(&[&zeros, a], axis as i32, s)
}

/// Appends `pad` zero rows along `axis`; the exact input when `pad == 0`.
fn pad_back(a: &Array, pad: i32, axis: usize, s: &Stream) -> Result<Array, MlxError> {
    if pad == 0 {
        return Ok(a.clone());
    }
    let mut shape = a.shape();
    shape[axis] = pad;
    let dtype = a.dtype().ok_or_else(|| MlxError {
        message: "pad rows requested for an array with an unsupported dtype".to_owned(),
    })?;
    let zeros = ops::zeros(&shape, dtype, s)?;
    ops::concatenate(&[a, &zeros], axis as i32, s)
}

/// Per-head RMSNorm on Q/K after the `[B, L, H, D]` reshape and before the
/// transpose (mlx-lm `qwen3.Attention`: `q_norm(q.reshape(B, L, H, -1))`).
#[derive(Debug)]
pub(crate) struct QkNorm {
    pub(crate) q_weight: Array,
    pub(crate) k_weight: Array,
    pub(crate) eps: f32,
}

/// Geometry + variant switches an architecture module passes to
/// [`Attention::load`]; everything else about the attention is shared.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionShape {
    pub(crate) n_heads: i32,
    pub(crate) n_kv_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) traditional_rope: bool,
    /// `Some(eps)` loads `q_norm`/`k_norm` weights (qwen3/gemma3 qk-norm;
    /// the weight parameterization follows the trunk's [`NormStyle`]).
    pub(crate) qk_norm_eps: Option<f32>,
    /// SDPA scale, already computed with the architecture's own f64
    /// formula (gemma `query_pre_attn_scalar`); `None` = `head_dim**-0.5`.
    pub(crate) scale_override: Option<f64>,
    /// gemma2 attention logit softcapping — selects the reference's manual
    /// attention (`tanh(scores/cap)*cap`, no `mx.fast` SDPA).
    pub(crate) attn_logit_softcapping: Option<f32>,
}

/// GQA attention shared by llama/qwen2/qwen3 — identical module math in
/// mlx-lm; the only variation points are the shape, the RoPE variant, bias
/// tensors (resolved from the checkpoint by [`Linear::load`]), and qk-norm.
#[derive(Debug)]
pub(crate) struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// `scale` as an ndim-0 f32 array — the paged-attention kernel's
    /// `scale_val` input, built once so the per-step hot path allocates
    /// nothing for it.
    scale_arr: Array,
    traditional_rope: bool,
    attn_softcap: Option<f32>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    qk_norm: Option<QkNorm>,
    rope: Rope,
}

impl Attention {
    pub(crate) fn load(
        store: &mut WeightStore,
        prefix: &str,
        quantization: Option<Quantization>,
        shape: &AttentionShape,
        rope: Rope,
        norm_style: NormStyle,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        let qk_norm = match shape.qk_norm_eps {
            Some(eps) => Some(QkNorm {
                q_weight: norm_style.apply(store.take(&format!("{prefix}.q_norm.weight"))?, s)?,
                k_weight: norm_style.apply(store.take(&format!("{prefix}.k_norm.weight"))?, s)?,
                eps,
            }),
            None => None,
        };
        // Python computes the scale in double precision and the FFI
        // narrows it; same here. Gemma passes its own
        // `query_pre_attn_scalar` formula through `scale_override`.
        let scale = shape
            .scale_override
            .unwrap_or_else(|| f64::from(shape.head_dim).powf(-0.5)) as f32;
        Ok(Self {
            n_heads: shape.n_heads,
            n_kv_heads: shape.n_kv_heads,
            head_dim: shape.head_dim,
            scale,
            scale_arr: Array::from_f32(scale),
            traditional_rope: shape.traditional_rope,
            attn_softcap: shape.attn_logit_softcapping,
            q_proj: Linear::load(store, &format!("{prefix}.q_proj"), quantization)?,
            k_proj: Linear::load(store, &format!("{prefix}.k_proj"), quantization)?,
            v_proj: Linear::load(store, &format!("{prefix}.v_proj"), quantization)?,
            o_proj: Linear::load(store, &format!("{prefix}.o_proj"), quantization)?,
            qk_norm,
            rope,
        })
    }

    /// The reference's manual attention for softcapped architectures
    /// (`mlx_lm.models.gemma2.Attention.__call__` after the cache update):
    /// scale into Q, GQA group reshape, `Q @ K^T`, `tanh(scores/cap)*cap`,
    /// boolean causal mask via `where`, precise softmax, `probs @ V`.
    /// `q` is `[B, H, L, D]`, `k`/`v` are `[B, KV, S, D]`; `q_offset` is the
    /// absolute position of `q`'s first row (0 in every reference-shaped
    /// prefill; the offset-aware mask generalizes the same predicate).
    #[allow(clippy::too_many_arguments)]
    fn manual_softcap_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        cap: f32,
        mask: SdpaMask,
        q_offset: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let (b, l, d) = (q.dim(0), q.dim(2), q.dim(3));
        let (kv_heads, s_len) = (k.dim(1), k.dim(2));
        let repeats = self.n_heads / self.n_kv_heads;

        let qs = ops::multiply(q, &weak_scalar(f64::from(self.scale), q, s)?, s)?;
        let (qs, kx, vx) = if repeats > 1 {
            (
                ops::reshape(&qs, &[b, kv_heads, repeats, l, d], s)?,
                ops::reshape(k, &[b, kv_heads, 1, s_len, d], s)?,
                ops::reshape(v, &[b, kv_heads, 1, s_len, d], s)?,
            )
        } else {
            (qs, k.clone(), v.clone())
        };
        let ndim_axes: &[i32] = if repeats > 1 {
            &[0, 1, 2, 4, 3]
        } else {
            &[0, 1, 3, 2]
        };
        let scores = ops::matmul(&qs, &ops::transpose(&kx, ndim_axes, s)?, s)?;
        let scores = ops::multiply(
            &ops::tanh(
                &ops::divide(&scores, &weak_scalar(f64::from(cap), &scores, s)?, s)?,
                s,
            )?,
            &weak_scalar(f64::from(cap), &scores, s)?,
            s,
        )?;
        let scores = match mask {
            SdpaMask::None => scores,
            SdpaMask::Causal => {
                // Boolean `linds >= rinds` causal mask (reference
                // `create_causal_mask`), realized as `linds + 1 > rinds`
                // over integer indices; masked lanes get
                // `finfo(dtype).min` via `where`, exactly as the
                // reference's bool-mask branch does.
                let linds1 = ops::reshape(
                    &ops::arange(
                        f64::from(q_offset) + 1.0,
                        f64::from(q_offset + l) + 1.0,
                        1.0,
                        Dtype::Int32,
                        s,
                    )?,
                    &[l, 1],
                    s,
                )?;
                let rinds = ops::reshape(
                    &ops::arange(0.0, f64::from(s_len), 1.0, Dtype::Int32, s)?,
                    &[1, s_len],
                    s,
                )?;
                let keep = ops::greater(&linds1, &rinds, s)?;
                let dtype = scores.dtype().ok_or_else(|| MlxError {
                    message: "softcap scores with an unsupported dtype".to_owned(),
                })?;
                // mx.finfo(dtype).min per dtype: bf16's is NOT f32::MIN
                // (that would round to -inf in bf16).
                let neg_min = match dtype {
                    Dtype::Float16 => -65504.0,
                    Dtype::Bfloat16 => -(2.0 - f64::powi(2.0, -7)) * f64::powi(2.0, 127),
                    _ => f64::from(f32::MIN),
                };
                ops::where_cond(&keep, &scores, &weak_scalar(neg_min, &scores, s)?, s)?
            }
        };
        let probs = ops::softmax(&scores, -1, true, s)?;
        let out = ops::matmul(&probs, &vx, s)?;
        if repeats > 1 {
            ops::reshape(&out, &[b, self.n_heads, l, d], s)
        } else {
            Ok(out)
        }
    }

    /// Contiguous-cache forward. `pad` rows (ADR 0002 kernel-class padding,
    /// appended to `x`'s token axis by the caller) go through the trunk
    /// matmuls but are excluded from the cache, and the query block is
    /// front-padded to keep SDPA in the tiled-kernel class; the pad lanes
    /// of the output are deterministic zero filler. `pad == 0` builds
    /// exactly the pre-ADR-0002 graph.
    fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        mask: SdpaMask,
        pad: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let (b, l) = (x.dim(0), x.dim(1) - pad);

        let queries = self.q_proj.forward(x, s)?;
        let keys = self.k_proj.forward(x, s)?;
        let values = self.v_proj.forward(x, s)?;
        let (queries, keys, values) = if pad > 0 {
            (
                ops::slice(
                    &queries,
                    &[0, 0, 0],
                    &[b, l, self.n_heads * self.head_dim],
                    s,
                )?,
                ops::slice(
                    &keys,
                    &[0, 0, 0],
                    &[b, l, self.n_kv_heads * self.head_dim],
                    s,
                )?,
                ops::slice(
                    &values,
                    &[0, 0, 0],
                    &[b, l, self.n_kv_heads * self.head_dim],
                    s,
                )?,
            )
        } else {
            (queries, keys, values)
        };

        // [B, L, H*D] -> [B, L, H, D] -> (qk-norm) -> [B, H, L, D]
        let queries = ops::reshape(&queries, &[b, l, self.n_heads, self.head_dim], s)?;
        let queries = self.norm_q(&queries, s)?;
        let queries = ops::transpose(&queries, &[0, 2, 1, 3], s)?;
        let keys = ops::reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim], s)?;
        let keys = self.norm_k(&keys, s)?;
        let keys = ops::transpose(&keys, &[0, 2, 1, 3], s)?;
        let values = ops::reshape(&values, &[b, l, self.n_kv_heads, self.head_dim], s)?;
        let values = ops::transpose(&values, &[0, 2, 1, 3], s)?;

        let offset = cache.offset();
        let queries = self
            .rope
            .apply(&queries, self.head_dim, self.traditional_rope, offset, s)?;
        let keys = self
            .rope
            .apply(&keys, self.head_dim, self.traditional_rope, offset, s)?;

        let (keys, values) = cache.update_and_fetch(&keys, &values, s)?;

        let out = if let Some(cap) = self.attn_softcap {
            // Softcapped architectures run reference-shaped (monolithic)
            // prefill pieces only, so kernel-class pads never arise here.
            if pad > 0 {
                return Err(MlxError {
                    message: "kernel-class pad rows on a softcapped-attention forward".to_owned(),
                });
            }
            self.manual_softcap_attention(&queries, &keys, &values, cap, mask, offset, s)?
        } else {
            // Pad queries sit in FRONT of the real rows: SDPA's causal mask
            // is bottom-right aligned, so the real rows keep their exact
            // attention spans and the pad rows (garbage in, discarded out)
            // attend shorter prefixes.
            let queries = pad_front(&queries, pad, 2, s)?;
            let out =
                fast::scaled_dot_product_attention(&queries, &keys, &values, self.scale, mask, s)?;
            if pad > 0 {
                ops::slice(
                    &out,
                    &[0, 0, pad, 0],
                    &[b, self.n_heads, pad + l, self.head_dim],
                    s,
                )?
            } else {
                out
            }
        };
        let out = ops::transpose(&out, &[0, 2, 1, 3], s)?;
        let out = ops::reshape(&out, &[b, l, self.n_heads * self.head_dim], s)?;
        // Refill the pad lanes (zeros — deterministic, discarded) so the
        // o_proj and everything downstream keep the padded row count.
        let out = pad_back(&out, pad, 1, s)?;
        self.o_proj.forward(&out, s)
    }

    fn norm_q(&self, q: &Array, s: &Stream) -> Result<Array, MlxError> {
        match &self.qk_norm {
            Some(n) => fast::rms_norm(q, &n.q_weight, n.eps, s),
            None => Ok(q.clone()),
        }
    }

    fn norm_k(&self, k: &Array, s: &Stream) -> Result<Array, MlxError> {
        match &self.qk_norm {
            Some(n) => fast::rms_norm(k, &n.k_weight, n.eps, s),
            None => Ok(k.clone()),
        }
    }

    /// Paged-attention step (SPEC §7.4 v0): per sequence, RoPE at its own
    /// offset, write this step's K/V into the pools, then gather the
    /// sequence's blocks into a contiguous view for fused SDPA. All writes
    /// chain onto the pools before any gather references them, so MLX
    /// donates the pool buffers instead of copying.
    ///
    /// For a single sequence the op sequence (shapes, masks, kernel input
    /// layouts) is identical to [`Self::forward`] over a contiguous cache —
    /// that equivalence is what keeps golden parity under paging.
    ///
    /// `pad` (> 0 only on single-sequence ragged prefill pieces — ADR 0002)
    /// counts kernel-class pad rows the caller appended to `x`'s token
    /// axis: they ride the trunk matmuls, front-pad the SDPA query block,
    /// and are refilled as zero lanes for the o_proj; their K/V is never
    /// written and their outputs are never selected.
    fn forward_step(
        &self,
        x: &Array,
        seqs: &[SeqStep],
        kv: &mut PagedKv,
        layer: usize,
        pad: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let queries = self.q_proj.forward(x, s)?;
        let keys = self.k_proj.forward(x, s)?;
        let values = self.v_proj.forward(x, s)?;

        // Phase 1: per-sequence (qk-norm +) RoPE + pool writes. Segments
        // address REAL rows only, so the pad rows at the tail of `x` are
        // naturally excluded from every K/V write.
        let single = seqs.len() == 1 && pad == 0;
        let mut roped: Vec<Array> = Vec::with_capacity(seqs.len());
        let mut start = 0;
        for seq in seqs {
            let l = seq.len;
            // qk-norm slots between the reshape and the transpose, exactly
            // as in [`Self::forward`]; rms_norm reduces over the last axis
            // only, so norming the sliced segment matches norming the full
            // batch row-for-row.
            let segment =
                |a: &Array, heads: i32, norm: Option<&Array>| -> Result<Array, MlxError> {
                    let seg = if single {
                        a.clone()
                    } else {
                        ops::slice(a, &[0, start, 0], &[1, start + l, heads * self.head_dim], s)?
                    };
                    let seg = ops::reshape(&seg, &[1, l, heads, self.head_dim], s)?;
                    let seg = match (norm, &self.qk_norm) {
                        (Some(weight), Some(n)) => fast::rms_norm(&seg, weight, n.eps, s)?,
                        _ => seg,
                    };
                    ops::transpose(&seg, &[0, 2, 1, 3], s)
                };
            let q = segment(
                &queries,
                self.n_heads,
                self.qk_norm.as_ref().map(|n| &n.q_weight),
            )?;
            let k = segment(
                &keys,
                self.n_kv_heads,
                self.qk_norm.as_ref().map(|n| &n.k_weight),
            )?;
            let v = segment(&values, self.n_kv_heads, None)?;
            let q = self
                .rope
                .apply(&q, self.head_dim, self.traditional_rope, seq.offset, s)?;
            let k = self
                .rope
                .apply(&k, self.head_dim, self.traditional_rope, seq.offset, s)?;
            kv.write(layer, &seq.writes, &k, &v, s)?;
            roped.push(q);
            start += l;
        }

        // Phase 2: per-sequence attention over the full history. Decode-
        // shaped segments take the block-table-aware kernel when the SPEC
        // §7.4 flag prepared inputs for them (same values, same reduction
        // order, no gather copy — see kiln-engine's paged_attn module docs
        // for the parity argument); everything else (prefill pieces,
        // padded pieces, softcapped architectures) gathers + fused SDPA
        // exactly as before. Pad queries sit in FRONT of the real rows
        // (SDPA's causal mask is bottom-right aligned, so real rows keep
        // their exact spans); the pad outputs are sliced away, then
        // refilled as zero lanes so the o_proj keeps the padded row count.
        let mut outs: Vec<Array> = Vec::with_capacity(seqs.len());
        for (seq, q) in seqs.iter().zip(&roped) {
            if pad == 0
                && self.attn_softcap.is_none()
                && let Some(paged) = &seq.paged_attn
            {
                debug_assert_eq!(seq.len, 1, "kernel inputs on a multi-row segment");
                let o = kv.paged_sdpa(layer, q, paged, &self.scale_arr, s)?;
                let o = ops::transpose(&o, &[0, 2, 1, 3], s)?;
                outs.push(ops::reshape(
                    &o,
                    &[1, seq.len, self.n_heads * self.head_dim],
                    s,
                )?);
                continue;
            }
            let (k, v) = kv.gather(layer, &seq.blocks, seq.offset + seq.len, s)?;
            let mask = if seq.len > 1 || pad > 0 {
                SdpaMask::Causal
            } else {
                SdpaMask::None
            };
            let o = if let Some(cap) = self.attn_softcap {
                // Softcapped architectures run reference-shaped
                // (monolithic) prefill pieces only — pads never arise.
                if pad > 0 {
                    return Err(MlxError {
                        message: "kernel-class pad rows on a softcapped-attention step".to_owned(),
                    });
                }
                self.manual_softcap_attention(q, &k, &v, cap, mask, seq.offset, s)?
            } else {
                let q = pad_front(q, pad, 2, s)?;
                let o = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask, s)?;
                if pad > 0 {
                    ops::slice(
                        &o,
                        &[0, 0, pad, 0],
                        &[1, self.n_heads, pad + seq.len, self.head_dim],
                        s,
                    )?
                } else {
                    o
                }
            };
            let o = ops::transpose(&o, &[0, 2, 1, 3], s)?;
            outs.push(ops::reshape(
                &o,
                &[1, seq.len, self.n_heads * self.head_dim],
                s,
            )?);
        }
        let out = match outs.as_slice() {
            [only] => only.clone(),
            many => {
                let refs: Vec<&Array> = many.iter().collect();
                ops::concatenate(&refs, 1, s)?
            }
        };
        let out = pad_back(&out, pad, 1, s)?;
        self.o_proj.forward(&out, s)
    }
}

/// Residual connection. With `clip_f16` (gemma3 `clip_residual`) an f16
/// residual is computed in f32 and clipped to the f16 range before
/// narrowing; any other dtype (and the non-clipping architectures) is a
/// plain add — exactly the reference's branch.
fn residual_add(x: &Array, y: &Array, clip_f16: bool, s: &Stream) -> Result<Array, MlxError> {
    if !clip_f16 || x.dtype() != Some(Dtype::Float16) {
        return ops::add(x, y, s);
    }
    let sum = ops::add(
        &ops::astype(x, Dtype::Float32, s)?,
        &ops::astype(y, Dtype::Float32, s)?,
        s,
    )?;
    // mx.finfo(mx.float16).max
    let bound = 65504.0f32;
    let clipped = ops::clip(&sum, &Array::from_f32(-bound), &Array::from_f32(bound), s)?;
    ops::astype(&clipped, Dtype::Float16, s)
}

/// Transformer block. Two reference shapes share it:
/// - llama/qwen pre-norm: `h = x + attn(norm(x)); out = h + mlp(norm(h))`;
/// - gemma "sandwich" (pre/post-feedforward norms present, post-attention
///   norm applied to the attention OUTPUT):
///   `h = x (+) norm(attn(norm(x))); out = h (+) norm(mlp(norm(h)))`
///   where `(+)` is [`residual_add`] (clipped for gemma3 f16).
#[derive(Debug)]
pub(crate) struct Block {
    input_layernorm: Array,
    post_attention_layernorm: Array,
    /// `Some` selects the gemma sandwich shape.
    pre_feedforward_layernorm: Option<Array>,
    post_feedforward_layernorm: Option<Array>,
    clip_residual_f16: bool,
    self_attn: Attention,
    mlp: Mlp,
    rms_eps: f32,
}

impl Block {
    /// Loads one decoder layer at `prefix` (`model.layers.N`).
    #[allow(clippy::too_many_arguments)] // load-time plumbing, one call site
    pub(crate) fn load(
        store: &mut WeightStore,
        prefix: &str,
        quantization: Option<Quantization>,
        shape: &AttentionShape,
        rope: Rope,
        rms_eps: f32,
        opts: &TrunkOptions,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        let norm = |w: Array| opts.norm_style.apply(w, s);
        let (pre_ffw, post_ffw) = if opts.sandwich_norms {
            (
                Some(norm(store.take(&format!(
                    "{prefix}.pre_feedforward_layernorm.weight"
                ))?)?),
                Some(norm(store.take(&format!(
                    "{prefix}.post_feedforward_layernorm.weight"
                ))?)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            input_layernorm: norm(store.take(&format!("{prefix}.input_layernorm.weight"))?)?,
            post_attention_layernorm: norm(
                store.take(&format!("{prefix}.post_attention_layernorm.weight"))?,
            )?,
            pre_feedforward_layernorm: pre_ffw,
            post_feedforward_layernorm: post_ffw,
            clip_residual_f16: opts.clip_residual_f16,
            self_attn: Attention::load(
                store,
                &format!("{prefix}.self_attn"),
                quantization,
                shape,
                rope,
                opts.norm_style,
                s,
            )?,
            mlp: Mlp::load(
                store,
                &format!("{prefix}.mlp"),
                quantization,
                opts.activation,
            )?,
            rms_eps,
        })
    }

    /// Everything after the attention sublayer — shared by both decode
    /// paths so the two block shapes exist exactly once.
    fn finish(&self, x: &Array, attn: &Array, s: &Stream) -> Result<Array, MlxError> {
        match (
            &self.pre_feedforward_layernorm,
            &self.post_feedforward_layernorm,
        ) {
            (Some(pre_ffw), Some(post_ffw)) => {
                let attn = fast::rms_norm(attn, &self.post_attention_layernorm, self.rms_eps, s)?;
                let h = residual_add(x, &attn, self.clip_residual_f16, s)?;
                let mlp = self
                    .mlp
                    .forward(&fast::rms_norm(&h, pre_ffw, self.rms_eps, s)?, s)?;
                let mlp = fast::rms_norm(&mlp, post_ffw, self.rms_eps, s)?;
                residual_add(&h, &mlp, self.clip_residual_f16, s)
            }
            _ => {
                let h = ops::add(x, attn, s)?;
                let normed = fast::rms_norm(&h, &self.post_attention_layernorm, self.rms_eps, s)?;
                let mlp = self.mlp.forward(&normed, s)?;
                ops::add(&h, &mlp, s)
            }
        }
    }

    fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        mask: SdpaMask,
        pad: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let normed = fast::rms_norm(x, &self.input_layernorm, self.rms_eps, s)?;
        let attn = self.self_attn.forward(&normed, cache, mask, pad, s)?;
        self.finish(x, &attn, s)
    }

    fn forward_step(
        &self,
        x: &Array,
        seqs: &[SeqStep],
        kv: &mut PagedKv,
        layer: usize,
        pad: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let normed = fast::rms_norm(x, &self.input_layernorm, self.rms_eps, s)?;
        let attn = self
            .self_attn
            .forward_step(&normed, seqs, kv, layer, pad, s)?;
        self.finish(x, &attn, s)
    }
}

/// Architecture switches for the shared trunk beyond [`AttentionShape`] —
/// one parity-proven `CausalLm` serves llama/qwen (defaults) and gemma
/// (sandwich blocks, `1+w` norms, gelu, embed scaling, softcapping).
#[derive(Debug)]
pub(crate) struct TrunkOptions {
    pub(crate) norm_style: NormStyle,
    pub(crate) activation: Activation,
    /// Load pre/post-feedforward norms and use the gemma sandwich block.
    pub(crate) sandwich_norms: bool,
    /// gemma3 `clip_residual` (f32 residual adds clipped to f16 range).
    pub(crate) clip_residual_f16: bool,
    /// Multiply embeddings by this constant after lookup (gemma's
    /// `sqrt(hidden_size)`, pre-built with the architecture's own rounding:
    /// a bf16 constant for gemma3, an f32 weak scalar for gemma2); it is
    /// `astype`d to the hidden dtype in-graph, as the reference does.
    pub(crate) embed_scale: Option<Array>,
    /// gemma2 final logit softcapping (`tanh(logits/cap)*cap`).
    pub(crate) final_logit_softcapping: Option<f32>,
}

impl Default for TrunkOptions {
    fn default() -> Self {
        Self {
            norm_style: NormStyle::Plain,
            activation: Activation::Silu,
            sandwich_norms: false,
            clip_residual_f16: false,
            embed_scale: None,
            final_logit_softcapping: None,
        }
    }
}

/// The shared causal-LM trunk: token embedding, `Block` stack, final
/// RMSNorm, (possibly tied) lm_head. Owns both decode paths so every
/// architecture built from it shares one parity-proven implementation.
#[derive(Debug)]
pub(crate) struct CausalLm {
    embed_tokens: Embedding,
    embed_scale: Option<Array>,
    blocks: Vec<Block>,
    norm: Array,
    /// `None` when `tie_word_embeddings` (logits via `embed_tokens.as_linear`).
    lm_head: Option<Linear>,
    final_logit_softcapping: Option<f32>,
    rms_eps: f32,
}

impl CausalLm {
    /// Consumes `store`, loading embedding + `num_layers` blocks + head. The
    /// checkpoint must be fully consumed: unexpected leftovers (beyond
    /// mlx-lm's sanitize-dropped `rotary_emb.inv_freq` tables and the stray
    /// `lm_head` on tied models) fail the load. `mk_rope` receives the layer
    /// index (gemma3 alternates local/global rope; other archs ignore it).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        mut store: WeightStore,
        quantization: Option<Quantization>,
        num_layers: usize,
        shape: &AttentionShape,
        rms_eps: f32,
        tie_word_embeddings: bool,
        opts: TrunkOptions,
        mk_rope: impl Fn(usize) -> Result<Rope, ModelError>,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        let embed_tokens = Embedding::load(&mut store, "model.embed_tokens", quantization)?;
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(Block::load(
                &mut store,
                &format!("model.layers.{i}"),
                quantization,
                shape,
                mk_rope(i)?,
                rms_eps,
                &opts,
                s,
            )?);
        }
        let norm = opts.norm_style.apply(store.take("model.norm.weight")?, s)?;
        let lm_head = if tie_word_embeddings {
            // mlx-lm sanitize drops a stray lm_head.weight on tied models.
            let _ = store.take_optional("lm_head.weight");
            let _ = store.take_optional("lm_head.scales");
            let _ = store.take_optional("lm_head.biases");
            None
        } else {
            Some(Linear::load(&mut store, "lm_head", quantization)?)
        };

        // mlx-lm sanitize: precomputed rotary tables are ignored.
        let leftovers: Vec<String> = store
            .remaining()
            .into_iter()
            .filter(|n| !n.contains("self_attn.rotary_emb.inv_freq"))
            .map(str::to_owned)
            .collect();
        if !leftovers.is_empty() {
            return Err(ModelError::Mismatch(format!(
                "unconsumed tensors in checkpoint: {leftovers:?}"
            )));
        }

        Ok(Self {
            embed_tokens,
            embed_scale: opts.embed_scale,
            blocks,
            norm,
            lm_head,
            final_logit_softcapping: opts.final_logit_softcapping,
            rms_eps,
        })
    }

    /// Embedding lookup + the gemma `sqrt(hidden)` scaling (reference:
    /// `h *= scale.astype(h.dtype)`); a plain lookup elsewhere.
    fn embed(&self, tokens: &Array, s: &Stream) -> Result<Array, MlxError> {
        let h = self.embed_tokens.lookup(tokens, s)?;
        match &self.embed_scale {
            None => Ok(h),
            Some(scale) => {
                let dtype = h.dtype().ok_or_else(|| MlxError {
                    message: "embedding output with an unsupported dtype".to_owned(),
                })?;
                ops::multiply(&h, &ops::astype(scale, dtype, s)?, s)
            }
        }
    }

    /// gemma2 final logit softcapping (`out = tanh(out/cap) * cap`); the
    /// identity elsewhere.
    fn softcap_logits(&self, logits: Array, s: &Stream) -> Result<Array, MlxError> {
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(cap) => {
                let capped = ops::tanh(
                    &ops::divide(&logits, &weak_scalar(f64::from(cap), &logits, s)?, s)?,
                    s,
                )?;
                ops::multiply(&capped, &weak_scalar(f64::from(cap), &logits, s)?, s)
            }
        }
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// ADR 0002 B' startup calibration: the widest per-forward row count
    /// at which every projection in this model still produces rows
    /// bit-identical to M=1 on THIS device.
    ///
    /// For each distinct linear shape (layer-0 attention + MLP
    /// projections and the (tied) lm_head, deduped by `(K, N)`), row 0 of
    /// the projection is computed at M = 1 and at rising M = 2..=32 over
    /// a realistic activation row (the token-0 embedding tiled to K); the
    /// first M whose row-0 bytes differ marks that shape's kernel-class
    /// dispatch threshold. The result is `min(thresholds) - 1`, capped at
    /// 32 when nothing diverges. Measuring the property directly (rather
    /// than replicating MLX's `get_qmv_batch_limit` device table) keeps
    /// the bound correct on GPUs the table does not describe, and probing
    /// through the same `Linear::forward`/`as_linear` code covers
    /// quantized and dense weights alike.
    pub(crate) fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        let block = self.blocks.first().ok_or_else(|| {
            ModelError::Mismatch("no layers to calibrate a deterministic width from".to_owned())
        })?;
        // Realistic activation row: the embedding of token 0.
        let ids = Array::from_u32_slice(&[0], &[1, 1])?;
        let base = self.embed_tokens.lookup(&ids, s)?;
        base.eval()?;

        let attn = &block.self_attn;
        let mut linears: Vec<&Linear> = vec![
            &attn.q_proj,
            &attn.k_proj,
            &attn.v_proj,
            &attn.o_proj,
            &block.mlp.gate_proj,
            &block.mlp.up_proj,
            &block.mlp.down_proj,
        ];
        if let Some(head) = &self.lm_head {
            linears.push(head);
        }
        let mut seen: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
        let mut min_threshold: usize = 33;
        for linear in linears {
            if seen.insert(linear.io_features()) {
                let threshold = probe_row_stability(linear.io_features().0, &base, s, |x| {
                    linear.forward(x, s)
                })?;
                min_threshold = min_threshold.min(threshold);
            }
        }
        if self.lm_head.is_none() {
            // Tied head: QuantizedEmbedding.as_linear is its own dispatch
            // site (same kernels, embedding-owned tensors).
            let hidden = base.dim(2);
            let threshold =
                probe_row_stability(hidden, &base, s, |x| self.embed_tokens.as_linear(x, s))?;
            min_threshold = min_threshold.min(threshold);
        }
        Ok(min_threshold.saturating_sub(1).clamp(1, 32))
    }

    /// Forward pass: `tokens [B, L]` (u32) -> logits `[B, L, vocab]`.
    /// Causal mask when `L > 1`, none for single-token decode — exactly
    /// mlx-lm's `create_attention_mask` for a cache without `make_mask`.
    ///
    /// The last `pad` positions of `tokens` are ADR 0002 kernel-class pad
    /// rows: they ride the trunk but never enter the caches, and their
    /// logit rows are zero-lane filler the caller must ignore.
    pub(crate) fn forward(
        &self,
        tokens: &Array,
        pad: i32,
        caches: &mut [KvCache],
        s: &Stream,
    ) -> Result<Array, ModelError> {
        if caches.len() != self.blocks.len() {
            return Err(ModelError::Mismatch(format!(
                "{} caches for {} layers",
                caches.len(),
                self.blocks.len()
            )));
        }
        if pad < 0 || pad >= tokens.dim(1) {
            return Err(ModelError::Mismatch(format!(
                "{pad} pad rows in a {}-token forward",
                tokens.dim(1)
            )));
        }
        let mask = if tokens.dim(1) > 1 {
            SdpaMask::Causal
        } else {
            SdpaMask::None
        };
        let mut h = self.embed(tokens, s)?;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            h = block.forward(&h, cache, mask, pad, s)?;
        }
        let h = fast::rms_norm(&h, &self.norm, self.rms_eps, s)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h, s)?,
            None => self.embed_tokens.as_linear(&h, s)?,
        };
        Ok(self.softcap_logits(logits, s)?)
    }

    /// The batched/paged forward pass (SPEC §6.2 step 2 / §7.2). The lm_head
    /// runs only over sampled positions — for prefill chunks it is skipped
    /// entirely (chunk logits would be dead graph).
    ///
    /// `batch.pad_rows` (ADR 0002, single-sequence ragged prefill pieces
    /// only): the trunk input is extended with that many pad rows (copies
    /// of the last real id — any deterministic in-vocab id works, the rows
    /// are pure kernel-class filler). They are never written to KV and
    /// never sampled; see `Attention::forward_step`.
    pub(crate) fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        let n = batch.num_tokens();
        let total: i32 = batch.seqs.iter().map(|seq| seq.len).sum();
        if n == 0 || total != n as i32 {
            return Err(MlxError {
                message: format!("step batch of {n} token(s) for {total} sequence position(s)"),
            });
        }
        if let Some(seq) = batch
            .seqs
            .iter()
            .find(|seq| seq.sample_rows < 0 || seq.sample_rows > seq.len)
        {
            return Err(MlxError {
                message: format!(
                    "segment samples {} of its {} position(s)",
                    seq.sample_rows, seq.len
                ),
            });
        }
        let pad = batch.pad_rows;
        if pad < 0 || (pad > 0 && batch.seqs.len() != 1) {
            return Err(MlxError {
                message: format!(
                    "{pad} pad rows on a {}-sequence step (pads apply to lone ragged \
                     prefill pieces only)",
                    batch.seqs.len()
                ),
            });
        }
        let tokens = match &batch.input {
            StepInput::Ids(ids) if pad > 0 => {
                let last = *ids.last().ok_or_else(|| MlxError {
                    message: "padded step with an empty id vector".to_owned(),
                })?;
                let mut padded = ids.clone();
                padded.extend(std::iter::repeat_n(last, pad as usize));
                Array::from_u32_slice(&padded, &[1, n as i32 + pad])?
            }
            StepInput::Ids(ids) => Array::from_u32_slice(ids, &[1, n as i32])?,
            // Pipelined decode: the previous step's sampled tokens, still
            // unevaluated — identical u32 values reach the embedding
            // lookup either way.
            StepInput::Lazy(tokens) if pad == 0 => tokens.clone(),
            StepInput::Lazy(_) => {
                return Err(MlxError {
                    message: "pad rows on a lazy (pipelined decode) step".to_owned(),
                });
            }
        };
        let mut h = self.embed(&tokens, s)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            h = block.forward_step(&h, &batch.seqs, kv, layer, pad, s)?;
        }

        // Sampled positions: the trailing `sample_rows` positions of each
        // sampling sequence — 1 for plain decode, the whole segment for a
        // speculative verify (SPEC §6.5). Always among the real rows: pads
        // sit behind them and are never selectable.
        let mut sampled: Vec<u32> = Vec::new();
        let mut pos: u32 = 0;
        for seq in &batch.seqs {
            pos += seq.len as u32;
            sampled.extend(pos - seq.sample_rows as u32..pos);
        }
        if sampled.is_empty() {
            return Ok(None);
        }
        let h = if sampled.len() == n && pad == 0 {
            h
        } else {
            let ids = Array::from_u32_slice(&sampled, &[sampled.len() as i32])?;
            ops::take(&h, &ids, 1, s)?
        };
        let h = fast::rms_norm(&h, &self.norm, self.rms_eps, s)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h, s)?,
            None => self.embed_tokens.as_linear(&h, s)?,
        };
        Ok(Some(self.softcap_logits(logits, s)?))
    }
}

/// RoPE variant resolved from `rope_scaling` (mlx-lm `initialize_rope`).
#[derive(Debug)]
pub(crate) enum Rope {
    /// `nn.RoPE`: frequencies derived from `base` inside the kernel
    /// (also covers `linear` scaling via `scale = 1/factor`).
    Plain { base: f32, scale: f32 },
    /// Precomputed per-pair frequencies (`Llama3RoPE` / `YarnRoPE`). Yarn
    /// additionally pre-scales the input by `mscale` when it is not 1.
    Freqs { freqs: Array, mscale: Option<f32> },
}

impl Rope {
    /// Builds the RoPE for one attention layer. `head_dim`/`base` come from
    /// the architecture's config; frequency tables are computed with the
    /// same MLX float32 graph as the Python reference so they match
    /// bit-for-bit on the device the reference ran on. (Across devices the
    /// graph's `power` op differs at ulp level — Metal transcendentals are
    /// not correctly rounded; see the yarn test's tolerance note.)
    pub(crate) fn new(
        scaling: &RopeScaling,
        head_dim: usize,
        base: f32,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        match *scaling {
            RopeScaling::Default => Ok(Rope::Plain { base, scale: 1.0 }),
            RopeScaling::Linear { factor } => Ok(Rope::Plain {
                base,
                scale: 1.0 / factor,
            }),
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings: old_context_len,
            } => {
                // Llama3RoPE.__init__, computed with the same MLX ops in the
                // same float32 graph so the frequencies match bit-for-bit.
                let dims = head_dim as f64;
                let half = ops::arange(0.0, dims, 2.0, Dtype::Float32, s)?;
                let exponents = ops::divide(&half, &Array::from_f32(dims as f32), s)?;
                let freqs = ops::power(&Array::from_f32(base), &exponents, s)?;
                let wavelens = ops::multiply(
                    &Array::from_f32((2.0 * std::f64::consts::PI) as f32),
                    &freqs,
                    s,
                )?;

                let low_freq_wavelen = Array::from_f32(old_context_len / low_freq_factor);
                let high_freq_wavelen = Array::from_f32(old_context_len / high_freq_factor);

                let scaled = ops::multiply(&freqs, &Array::from_f32(factor), s)?;
                let is_low = ops::greater(&wavelens, &low_freq_wavelen, s)?;
                let freqs = ops::where_cond(&is_low, &scaled, &freqs, s)?;

                let is_medium = ops::logical_and(
                    &ops::greater(&wavelens, &high_freq_wavelen, s)?,
                    &ops::less(&wavelens, &low_freq_wavelen, s)?,
                    s,
                )?;
                let smooth = ops::divide(
                    &ops::subtract(
                        &ops::divide(&Array::from_f32(old_context_len), &wavelens, s)?,
                        &Array::from_f32(low_freq_factor),
                        s,
                    )?,
                    &Array::from_f32(high_freq_factor - low_freq_factor),
                    s,
                )?;
                // freqs / ((1 - smooth) / factor + smooth)
                let denom = ops::add(
                    &ops::divide(
                        &ops::subtract(&Array::from_f32(1.0), &smooth, s)?,
                        &Array::from_f32(factor),
                        s,
                    )?,
                    &smooth,
                    s,
                )?;
                let smooth_freqs = ops::divide(&freqs, &denom, s)?;
                let freqs = ops::where_cond(&is_medium, &smooth_freqs, &freqs, s)?;
                freqs.eval()?;
                Ok(Rope::Freqs {
                    freqs,
                    mscale: None,
                })
            }
            RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
            } => {
                // YarnRoPE.__init__. The correction range and mscale are
                // host-side double-precision scalar math in the reference
                // (Python floats) — mirrored in f64 here; the frequency
                // table is the same MLX float32 graph.
                let dims = head_dim as f64;
                let base_f64 = f64::from(base);
                let correction_dim = |num_rotations: f64| -> f64 {
                    dims * (original_max_position_embeddings
                        / (num_rotations * 2.0 * std::f64::consts::PI))
                        .ln()
                        / (2.0 * base_f64.ln())
                };
                let low = correction_dim(beta_fast).floor().max(0.0);
                let high = correction_dim(beta_slow).ceil().min(dims - 1.0);
                let get_mscale = |scale: f64, m: f64| -> f64 {
                    if scale <= 1.0 {
                        1.0
                    } else {
                        0.1 * m * scale.ln() + 1.0
                    }
                };
                let m = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim);

                let half = ops::arange(0.0, dims, 2.0, Dtype::Float32, s)?;
                let exponents = ops::divide(&half, &Array::from_f32(dims as f32), s)?;
                let freq_extra = ops::power(&Array::from_f32(base), &exponents, s)?;
                let freq_inter = ops::multiply(&Array::from_f32(factor as f32), &freq_extra, s)?;

                // yarn_linear_ramp_mask(low, high, dims/2), then 1 - ramp.
                let high = if low == high { high + 0.001 } else { high };
                let positions = ops::arange(0.0, (head_dim / 2) as f64, 1.0, Dtype::Float32, s)?;
                let ramp = ops::clip(
                    &ops::divide(
                        &ops::subtract(&positions, &Array::from_f32(low as f32), s)?,
                        &Array::from_f32((high - low) as f32),
                        s,
                    )?,
                    &Array::from_f32(0.0),
                    &Array::from_f32(1.0),
                    s,
                )?;
                let freq_mask = ops::subtract(&Array::from_f32(1.0), &ramp, s)?;

                // (inter * extra) / (inter * mask + extra * (1 - mask)) —
                // `1 - mask` is a fresh op in the reference, not the ramp
                // reused; keep it that way for bit-identical rounding.
                let numer = ops::multiply(&freq_inter, &freq_extra, s)?;
                let denom = ops::add(
                    &ops::multiply(&freq_inter, &freq_mask, s)?,
                    &ops::multiply(
                        &freq_extra,
                        &ops::subtract(&Array::from_f32(1.0), &freq_mask, s)?,
                        s,
                    )?,
                    s,
                )?;
                let freqs = ops::divide(&numer, &denom, s)?;
                freqs.eval()?;
                Ok(Rope::Freqs {
                    freqs,
                    mscale: (m != 1.0).then_some(m as f32),
                })
            }
        }
    }

    /// Applies the rotation. Invariant at every call site: `dims` is the full
    /// last-axis width of `x` (all supported architectures rotate the whole
    /// head), so yarn's `x[..., :dims] *= mscale` is a whole-tensor multiply.
    pub(crate) fn apply(
        &self,
        x: &Array,
        dims: i32,
        traditional: bool,
        offset: i32,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        match self {
            Rope::Plain { base, scale } => {
                fast::rope_with_base(x, dims, traditional, *base, *scale, offset, s)
            }
            Rope::Freqs { freqs, mscale } => {
                let scaled;
                let x = match mscale {
                    Some(m) => {
                        scaled = ops::multiply(&Array::from_f32(*m), x, s)?;
                        &scaled
                    }
                    None => x,
                };
                fast::rope_with_freqs(x, dims, traditional, offset, freqs, s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// YarnRoPE parity with the pinned reference. The expected values were
    /// generated from the Python worker venv (mlx.core 0.31.1 / mlx-lm
    /// 0.31.2 — the B1-aligned stack, docs/decisions/0001-mlx-c-pin.md) via
    /// `initialize_rope(128, base=1e6, traditional=False, scaling_config=
    /// {"rope_type": "yarn", "factor": 4.0,
    ///  "original_max_position_embeddings": 32768})` — the documented Qwen3
    /// long-context recipe.
    ///
    /// The freq bar is <= [`YARN_FREQ_ULP_TOL`] ulp per element, NOT
    /// bit-exactness. That is deliberate — do not "fix" it back:
    ///
    /// - The freq table is an MLX float32 graph evaluated on the Metal
    ///   stream, and its transcendental (`power`) is neither correctly
    ///   rounded nor identical across GPU families / Metal compiler
    ///   versions. Measured at this pin for freq[3] = 1e6^0.046875 (true
    ///   value 1.91095297497…): local Metal 0x3FF49A1B, the macos-14 CI
    ///   runner's paravirtual GPU 0x3FF49A1A, MLX's CPU backend 0x3FF49A1C
    ///   — three faithful `pow` implementations spanning 2 ulp, each
    ///   deterministic per device. Fixture bits captured on one machine can
    ///   therefore never be a cross-machine bit-exact bar; this is ADR
    ///   0002's observation (bit-identity holds per device and kernel
    ///   class) at unit-test scale. Outside the interpolation blend the
    ///   multiply/divide chain around the `power` is bit-transparent
    ///   (verified in strict f32 emulation); inside the blend it AMPLIFIES
    ///   the pow spread — measured full spectrum on the macos-14 CI GPU:
    ///   every element within 2 ulp of the fixture bits except freq[39]
    ///   (deepest into the blend, mask = 1/17) at 4 ulp (2.6e-7 relative).
    /// - This does NOT loosen the golden-parity bar (SPEC §11.2 /
    ///   ADR 0002), which stays strictly bit-exact and is a different kind
    ///   of bar: it compares integer token ids from a reference generated
    ///   by mlx-lm running the SAME MLX kernels on the SAME device class,
    ///   where bit-equality is real and load-bearing. Never cite this
    ///   test's tolerance as precedent for relaxing a golden fixture.
    /// - `mscale` stays bit-exact: it is host-side f64 scalar math
    ///   (Python-float semantics), and the f64→f32 rounding at the
    ///   comparison absorbs any libm ulp differences; CI agrees
    ///   byte-for-byte across machines.
    #[test]
    fn yarn_freqs_match_reference_within_ulp_tol() {
        if !kiln_mlx::memory::metal_is_available() {
            eprintln!("skipping: no Metal device");
            return;
        }
        const EXPECTED_MSCALE: f64 = 1.138_629_436_111_989;
        const EXPECTED_FREQ_BITS: [u32; 64] = [
            0x3f800000, 0x3f9ed70c, 0x3fc51c50, 0x3ff49a1b, 0x4017c496, 0x403c55a5, 0x4069b621,
            0x409102bc, 0x40b3f300, 0x40df4e48, 0x410a8de7, 0x412beff0, 0x41555d09, 0x418462a8,
            0x41a44832, 0x41cbdd1f, 0x41fcfb72, 0x421cf7b5, 0x4242c979, 0x4271b7f4, 0x4295fa95,
            0x42ba1d4b, 0x42e6f4d6, 0x430f4d1f, 0x433a090f, 0x43720768, 0x439dcea7, 0x43ce51dc,
            0x440742da, 0x4431ebf3, 0x446ae1fb, 0x449bac8f, 0x44cf512c, 0x450aca02, 0x453afdb9,
            0x457dcc68, 0x45adc3b3, 0x45f082e9, 0x4628b1d0, 0x4670bd8c, 0x46afbb4e, 0x46da1273,
            0x47074e93, 0x4727e851, 0x47505cdc, 0x47814858, 0x47a06e81, 0x47c715f0, 0x47f70d8e,
            0x481949e6, 0x483e38c1, 0x486c0da4, 0x489276b6, 0x48b5c09b, 0x48e18b1a, 0x490bf150,
            0x492da8fc, 0x4957805b, 0x4985b63f, 0x49a5ed9b, 0x49cde812, 0x49ff8464, 0x4a1e8a5b,
            0x4a44bd24,
        ];

        kiln_mlx::init();
        let stream = Stream::gpu();
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32768.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        };
        let rope = Rope::new(&scaling, 128, 1_000_000.0, &stream).expect("yarn rope builds");
        let Rope::Freqs { freqs, mscale } = rope else {
            panic!("yarn must resolve to precomputed freqs");
        };
        assert_eq!(
            mscale,
            Some(EXPECTED_MSCALE as f32),
            "mscale diverged from the reference host-math value"
        );
        let values = freqs.data_f32().expect("freqs readable");
        assert_eq!(values.len(), EXPECTED_FREQ_BITS.len());
        let mut worst = 0u32;
        let mut offenders = Vec::new();
        for (i, (got, want)) in values
            .iter()
            .zip(EXPECTED_FREQ_BITS.iter().map(|b| f32::from_bits(*b)))
            .enumerate()
        {
            let d = ulp_distance(*got, want);
            worst = worst.max(d);
            if d > YARN_FREQ_ULP_TOL {
                offenders.push(format!(
                    "freq[{i}] = {got} ({:#010x}) vs reference {want} ({:#010x}): {d} ulp",
                    got.to_bits(),
                    want.to_bits()
                ));
            }
        }
        assert!(
            offenders.is_empty(),
            "yarn freqs off reference by > {YARN_FREQ_ULP_TOL} ulp (worst {worst}):\n{}",
            offenders.join("\n")
        );
    }

    /// Per-element bound for the yarn freq table vs the committed reference
    /// bits. Measured cross-device spectrum (macos-14 CI GPU vs the dev
    /// machine that generated the fixture bits): all elements within 2 ulp
    /// except freq[39] — deepest into the interpolation blend, where the
    /// divide chain amplifies the pow spread — at 4 ulp. 8 is 2x the
    /// observed worst (~1e-6 relative); any real defect (wrong exponent,
    /// mask, or factor) lands thousands of ulp out.
    const YARN_FREQ_ULP_TOL: u32 = 8;

    /// Ulp distance via bit-pattern difference — valid because all yarn
    /// freqs are positive finite f32 (monotone bit ordering); a NaN/inf/
    /// negative from a broken graph yields a huge distance and still fails.
    fn ulp_distance(a: f32, b: f32) -> u32 {
        (i64::from(a.to_bits()) - i64::from(b.to_bits())).unsigned_abs() as u32
    }
}
