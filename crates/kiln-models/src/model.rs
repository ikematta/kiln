//! Architecture dispatch (SPEC §7.2): one enum over every implemented model
//! so the worker and test harnesses can load any supported checkpoint from
//! its `config.json` alone. Model-specific behavior stays in the per-arch
//! modules; this is routing only.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::ArchConfig;
use crate::gemma2::Gemma2Model;
use crate::gemma3::Gemma3Model;
use crate::llama::LlamaModel;
use crate::nn::ModelError;
use crate::qwen2::Qwen2Model;
use crate::qwen3::Qwen3Model;

/// A loaded model of any supported architecture, dispatched on
/// `config.json`'s `model_type`.
#[derive(Debug)]
pub enum AnyModel {
    Llama(LlamaModel),
    Qwen2(Qwen2Model),
    Qwen3(Qwen3Model),
    Gemma2(Gemma2Model),
    Gemma3(Gemma3Model),
}

impl AnyModel {
    /// Loads the model at `dir`, selecting the implementation by
    /// `model_type`. Unsupported architectures/rope/quantization fail here
    /// with the named [`crate::ConfigError`] reasons.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        match ArchConfig::from_model_dir(dir)? {
            ArchConfig::Llama(_) => Ok(Self::Llama(LlamaModel::load(dir, s)?)),
            ArchConfig::Qwen2(_) => Ok(Self::Qwen2(Qwen2Model::load(dir, s)?)),
            ArchConfig::Qwen3(_) => Ok(Self::Qwen3(Qwen3Model::load(dir, s)?)),
            ArchConfig::Gemma2(_) => Ok(Self::Gemma2(Gemma2Model::load(dir, s)?)),
            ArchConfig::Gemma3(_) => Ok(Self::Gemma3(Gemma3Model::load(dir, s)?)),
        }
    }

    pub fn model_type(&self) -> &str {
        match self {
            Self::Llama(m) => &m.config().model_type,
            Self::Qwen2(m) => &m.config().model_type,
            Self::Qwen3(m) => &m.config().model_type,
            Self::Gemma2(m) => &m.config().model_type,
            Self::Gemma3(m) => &m.config().model_type,
        }
    }

    /// EOS token ids from the model's `config.json` (worker stop set).
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Llama(m) => m.config().eos_token_ids(),
            Self::Qwen2(m) => m.config().eos_token_ids(),
            Self::Qwen3(m) => m.config().eos_token_ids(),
            Self::Gemma2(m) => m.config().eos_token_ids(),
            Self::Gemma3(m) => m.config().eos_token_ids(),
        }
    }

    /// KV geometry for the engine's paged pools.
    pub fn kv_dims(&self) -> KvDims {
        match self {
            Self::Llama(m) => m.kv_dims(),
            Self::Qwen2(m) => m.kv_dims(),
            Self::Qwen3(m) => m.kv_dims(),
            Self::Gemma2(m) => m.kv_dims(),
            Self::Gemma3(m) => m.kv_dims(),
        }
    }

    /// `true` when this model's prefill is only reference-defined for
    /// monolithic (offset-0, per-`prefill_chunk`) pieces. Engine builders
    /// honor it as `prefill_fine_chunk = prefill_chunk` (the single-tail
    /// schedule; Phase 5's fine grid stays on elsewhere). Two causes:
    /// - gemma2's manual softcapped attention (see gemma2.rs);
    /// - dense (unquantized) checkpoints: the ADR 0002 kernel-class pad
    ///   does not make fine-grid pieces bit-reproduce the reference's
    ///   single-piece pass for bf16 dense trunks at the pinned MLX — a
    ///   pure-mlx-lm replica of the padded 128+4→32 schedule flips the
    ///   same greedy token the Rust engine does, while the
    ///   reference-shaped replica matches the fixture (PROGRESS
    ///   2026-07-10, smollm2-135m-bf16/raw-tiny-remainder). The fine
    ///   grid was only ever validated on quantized checkpoints.
    pub fn monolithic_prefill_required(&self) -> bool {
        let dense = match self {
            Self::Llama(m) => m.config().quantization.is_none(),
            Self::Qwen2(m) => m.config().quantization.is_none(),
            Self::Qwen3(m) => m.config().quantization.is_none(),
            Self::Gemma2(m) => m.config().quantization.is_none(),
            Self::Gemma3(m) => m.config().quantization.is_none(),
        };
        match self {
            Self::Gemma2(m) => dense || m.monolithic_prefill_required(),
            _ => dense,
        }
    }

    /// Largest PROMPT length whose prefill op stream is reference-shaped
    /// (`None` = no architectural bound beyond pool/config limits).
    pub fn max_prompt_for_parity(&self) -> Option<usize> {
        match self {
            Self::Gemma2(m) => Some(m.max_prompt_for_parity()),
            _ => None,
        }
    }

    /// Largest TOTAL context (prompt + generated) whose op stream is
    /// reference-shaped (`None` = no architectural bound). gemma3's
    /// sliding window — see `gemma3.rs` module docs.
    pub fn max_context_for_parity(&self) -> Option<usize> {
        match self {
            Self::Gemma3(m) => Some(m.max_context_for_parity()),
            _ => None,
        }
    }

    /// The certified speculative-decoding envelope for THIS checkpoint
    /// (ADR 0005): the largest `gamma` whose gamma+1-row verify forward
    /// provably runs the same attention kernel class as plain (1-row)
    /// decode at the pinned MLX — `None` means speculation is
    /// unsupported for this model and a configured drafter must be
    /// rejected loudly at attach.
    ///
    /// Derived from the checkpoint's config, never hardcoded per model:
    /// - the architecture module must take the fused-SDPA decode path
    ///   (gemma2's manual softcapped attention has no kernel-class
    ///   certificate: its score/probs matmuls change gemv/gemm class
    ///   with the query row count);
    /// - the trunk must be quantized (dense trunks carry no fine-shape
    ///   bit guarantee — the ADR 0002 addendum precedent);
    /// - `head_dim` must be in the pinned fused-vector kernel's set
    ///   {64, 96, 128, 256} with equal Q/V head dims (outside it even
    ///   plain decode runs the unfused fallback, so no class equality
    ///   exists to preserve);
    /// - `gamma + 1 <= min(8, 32 / gqa_factor)` — the pinned
    ///   `supports_sdpa_vector` predicate (verified from
    ///   mlx/backend/metal/scaled_dot_product_attention.cpp, uniform
    ///   across devices and dtypes at the pin).
    ///
    /// The engine-side pieces of the envelope — the ADR 0002
    /// deterministic-width clamp and the 1-pass kv-length bound — live
    /// in kiln-engine; the worker combines all of them at drafter
    /// attachment. PRECONDITION FOR NEW ARCHITECTURES (ADR 0005): a new
    /// model family must be reviewed against this envelope AND pass the
    /// full spec_decode gate on the generating device before speculation
    /// is enabled for it; this method returning `Some` for an unreviewed
    /// geometry is not by itself permission.
    pub fn speculative_gamma_bound(&self) -> Option<usize> {
        // (n_heads, n_kv_heads, head_dim, quantized, fused-SDPA path)
        let (heads, kv_heads, head_dim, quantized, fused) = match self {
            Self::Llama(m) => {
                let c = m.config();
                (
                    c.num_attention_heads,
                    c.num_kv_heads(),
                    c.head_dim(),
                    c.quantization.is_some(),
                    true,
                )
            }
            Self::Qwen2(m) => {
                let c = m.config();
                (
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim(),
                    c.quantization.is_some(),
                    true,
                )
            }
            Self::Qwen3(m) => {
                let c = m.config();
                (
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.quantization.is_some(),
                    true,
                )
            }
            // gemma2 always runs the reference's manual softcapped
            // attention (see gemma2.rs) — never fused SDPA.
            Self::Gemma2(m) => {
                let c = m.config();
                (
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.quantization.is_some(),
                    false,
                )
            }
            Self::Gemma3(m) => {
                let c = m.config();
                (
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.quantization.is_some(),
                    true,
                )
            }
        };
        if !quantized || !fused || kv_heads == 0 {
            return None;
        }
        if !matches!(head_dim, 64 | 96 | 128 | 256) {
            return None;
        }
        // MLX computes gqa_factor by integer division of head counts.
        let gqa_factor = heads / kv_heads;
        if gqa_factor == 0 {
            return None;
        }
        let max_query_rows = (32 / gqa_factor).min(8);
        // gamma >= 1 needs at least two query rows in the vector class.
        (max_query_rows >= 2).then(|| max_query_rows - 1)
    }

    /// ADR 0002 B' startup calibration: the widest per-forward row count
    /// whose rows stay bit-identical to M=1 on this device, across every
    /// projection shape in the loaded model. Feeds
    /// `EngineConfig::deterministic_decode_width` and the informational
    /// `WorkerInfo.max_deterministic_decode_width`.
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        match self {
            Self::Llama(m) => m.calibrate_deterministic_width(s),
            Self::Qwen2(m) => m.calibrate_deterministic_width(s),
            Self::Qwen3(m) => m.calibrate_deterministic_width(s),
            Self::Gemma2(m) => m.calibrate_deterministic_width(s),
            Self::Gemma3(m) => m.calibrate_deterministic_width(s),
        }
    }
}

impl StepModel for AnyModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        match self {
            Self::Llama(m) => m.forward_step(batch, kv, s),
            Self::Qwen2(m) => m.forward_step(batch, kv, s),
            Self::Qwen3(m) => m.forward_step(batch, kv, s),
            Self::Gemma2(m) => m.forward_step(batch, kv, s),
            Self::Gemma3(m) => m.forward_step(batch, kv, s),
        }
    }
}
