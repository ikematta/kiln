//! Llama-family model (SPEC §7.2): Llama 2/3/3.x, Mistral, and llamafied
//! variants.
//!
//! Ported op-for-op from `mlx_lm.models.llama` (+ `rope_utils.Llama3RoPE`)
//! at the pinned reference version — same MLX kernels in the same order, so
//! greedy decoding is bit-identical to the golden fixtures. Do not "improve"
//! numerics here (e.g. fp32 accumulations, fused rewrites) without re-running
//! the golden harness; parity is the acceptance bar (SPEC §11.2).

use std::path::Path;

use kiln_mlx::fast::{self, SdpaMask};
use kiln_mlx::{Array, Dtype, MlxError, Stream, ops};

use crate::config::{ConfigError, LlamaConfig, Quantization, RopeScaling};
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
enum Linear {
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
    fn load(
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

    fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
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
}

/// Token embedding, quantized (mlx-lm `QuantizedEmbedding`) or dense; also
/// serves as the tied `lm_head` via [`Self::as_linear`].
#[derive(Debug)]
enum Embedding {
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
    fn load(
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
    fn lookup(&self, ids: &Array, s: &Stream) -> Result<Array, MlxError> {
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
    fn as_linear(&self, h: &Array, s: &Stream) -> Result<Array, MlxError> {
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

/// RoPE variant resolved from `rope_scaling` (mlx-lm `initialize_rope`).
#[derive(Debug)]
enum Rope {
    /// `nn.RoPE`: frequencies derived from `base` inside the kernel.
    Plain { base: f32, scale: f32 },
    /// `Llama3RoPE`: precomputed warped frequencies.
    Freqs { freqs: Array },
}

impl Rope {
    fn new(config: &LlamaConfig, s: &Stream) -> Result<Self, ModelError> {
        match config.rope_scaling()? {
            RopeScaling::Default => Ok(Rope::Plain {
                base: config.rope_theta,
                scale: 1.0,
            }),
            RopeScaling::Linear { factor } => Ok(Rope::Plain {
                base: config.rope_theta,
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
                let dims = config.head_dim() as f64;
                let half = ops::arange(0.0, dims, 2.0, Dtype::Float32, s)?;
                let exponents = ops::divide(&half, &Array::from_f32(dims as f32), s)?;
                let freqs = ops::power(&Array::from_f32(config.rope_theta), &exponents, s)?;
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
                Ok(Rope::Freqs { freqs })
            }
        }
    }

    fn apply(
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
            Rope::Freqs { freqs } => fast::rope_with_freqs(x, dims, traditional, offset, freqs, s),
        }
    }
}

#[derive(Debug)]
struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    traditional_rope: bool,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: Rope,
}

impl Attention {
    fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        mask: SdpaMask,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let (b, l) = (x.dim(0), x.dim(1));

        let queries = self.q_proj.forward(x, s)?;
        let keys = self.k_proj.forward(x, s)?;
        let values = self.v_proj.forward(x, s)?;

        // [B, L, H*D] -> [B, H, L, D]
        let queries = ops::reshape(&queries, &[b, l, self.n_heads, self.head_dim], s)?;
        let queries = ops::transpose(&queries, &[0, 2, 1, 3], s)?;
        let keys = ops::reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim], s)?;
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

        let out =
            fast::scaled_dot_product_attention(&queries, &keys, &values, self.scale, mask, s)?;
        let out = ops::transpose(&out, &[0, 2, 1, 3], s)?;
        let out = ops::reshape(&out, &[b, l, self.n_heads * self.head_dim], s)?;
        self.o_proj.forward(&out, s)
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
        // swiglu(gate, up) = silu(gate) * up ; silu(x) = x * sigmoid(x)
        let gate = self.gate_proj.forward(x, s)?;
        let up = self.up_proj.forward(x, s)?;
        let silu = ops::multiply(&gate, &ops::sigmoid(&gate, s)?, s)?;
        self.down_proj.forward(&ops::multiply(&silu, &up, s)?, s)
    }
}

#[derive(Debug)]
struct Block {
    input_layernorm: Array,
    post_attention_layernorm: Array,
    self_attn: Attention,
    mlp: Mlp,
    rms_eps: f32,
}

impl Block {
    fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        mask: SdpaMask,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let normed = fast::rms_norm(x, &self.input_layernorm, self.rms_eps, s)?;
        let attn = self.self_attn.forward(&normed, cache, mask, s)?;
        let h = ops::add(x, &attn, s)?;
        let normed = fast::rms_norm(&h, &self.post_attention_layernorm, self.rms_eps, s)?;
        let mlp = self.mlp.forward(&normed, s)?;
        ops::add(&h, &mlp, s)
    }
}

/// A loaded Llama-family model.
#[derive(Debug)]
pub struct LlamaModel {
    config: LlamaConfig,
    embed_tokens: Embedding,
    blocks: Vec<Block>,
    norm: Array,
    /// `None` when `tie_word_embeddings` (logits via `embed_tokens.as_linear`).
    lm_head: Option<Linear>,
}

impl LlamaModel {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = LlamaConfig::from_model_dir(dir)?;
        let mut store = WeightStore::from_model_dir(dir)?;
        let quant = config.quantization;

        let embed_tokens = Embedding::load(&mut store, "model.embed_tokens", quant)?;
        let head_dim = config.head_dim() as i32;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            blocks.push(Block {
                input_layernorm: store.take(&format!("{p}.input_layernorm.weight"))?,
                post_attention_layernorm: store
                    .take(&format!("{p}.post_attention_layernorm.weight"))?,
                self_attn: Attention {
                    n_heads: config.num_attention_heads as i32,
                    n_kv_heads: config.num_kv_heads() as i32,
                    head_dim,
                    scale: (config.head_dim() as f32).powf(-0.5),
                    traditional_rope: config.rope_traditional,
                    q_proj: Linear::load(&mut store, &format!("{p}.self_attn.q_proj"), quant)?,
                    k_proj: Linear::load(&mut store, &format!("{p}.self_attn.k_proj"), quant)?,
                    v_proj: Linear::load(&mut store, &format!("{p}.self_attn.v_proj"), quant)?,
                    o_proj: Linear::load(&mut store, &format!("{p}.self_attn.o_proj"), quant)?,
                    rope: Rope::new(&config, s)?,
                },
                mlp: Mlp {
                    gate_proj: Linear::load(&mut store, &format!("{p}.mlp.gate_proj"), quant)?,
                    up_proj: Linear::load(&mut store, &format!("{p}.mlp.up_proj"), quant)?,
                    down_proj: Linear::load(&mut store, &format!("{p}.mlp.down_proj"), quant)?,
                },
                rms_eps: config.rms_norm_eps,
            });
        }
        let norm = store.take("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            // mlx-lm sanitize drops a stray lm_head.weight on tied models.
            let _ = store.take_optional("lm_head.weight");
            let _ = store.take_optional("lm_head.scales");
            let _ = store.take_optional("lm_head.biases");
            None
        } else {
            Some(Linear::load(&mut store, "lm_head", quant)?)
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
            config,
            embed_tokens,
            blocks,
            norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// One fresh contiguous cache per layer.
    pub fn make_cache(&self) -> Vec<KvCache> {
        (0..self.blocks.len()).map(|_| KvCache::new()).collect()
    }

    /// Forward pass: `tokens [B, L]` (u32) -> logits `[B, L, vocab]`.
    /// Causal mask when `L > 1`, none for single-token decode — exactly
    /// mlx-lm's `create_attention_mask` for a cache without `make_mask`.
    pub fn forward(
        &self,
        tokens: &Array,
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
        let mask = if tokens.dim(1) > 1 {
            SdpaMask::Causal
        } else {
            SdpaMask::None
        };
        let mut h = self.embed_tokens.lookup(tokens, s)?;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            h = block.forward(&h, cache, mask, s)?;
        }
        let h = fast::rms_norm(&h, &self.norm, self.config.rms_norm_eps, s)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h, s)?,
            None => self.embed_tokens.as_linear(&h, s)?,
        };
        Ok(logits)
    }
}
