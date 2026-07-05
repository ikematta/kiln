//! Gemma2 model (SPEC §7.2, Phase 6 Task 2).
//!
//! Ported op-for-op from `mlx_lm.models.gemma2` at the pinned reference
//! version: `1 + w` RMSNorms, the sandwich block (plain residual adds — no
//! f16 clipping), `gelu_approx` MLP, `sqrt(hidden_size)` embedding scaling
//! (a Python-float weak scalar), `1/sqrt(query_pre_attn_scalar)` attention
//! scaling, MANUAL attention with logit softcapping (`tanh(scores/cap)*cap`
//! — `mx.fast` SDPA is unusable under softcapping, so the reference matmuls
//! scores/probs explicitly), always-tied embeddings, and final logit
//! softcapping.
//!
//! **What the pinned reference does NOT implement, and Kiln therefore must
//! not either:** no sliding window (the config field is ignored; every
//! layer attends the full history), no `rope_scaling`.
//!
//! **Prefill shape constraint:** manual-attention score/prob matmul row
//! bits depend on the key-axis length, and the reference's boolean causal
//! mask is only defined for offset-0 pieces (`create_causal_mask(N)` with a
//! plain `KVCache` — no `make_mask` at the pin). Parity therefore requires
//! Kiln to prefill in the reference's own shape: one monolithic piece per
//! `prefill_chunk` (2048 = mlx-lm's `prefill_step_size`), no fine grid —
//! declared via [`Gemma2Model::monolithic_prefill_required`] and honored by
//! the worker/harness engine config; prompts longer than one chunk are
//! refused via [`Gemma2Model::max_prompt_for_parity`] (the reference itself
//! has no defined masking for a continuation chunk at this pin).

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::Gemma2Config;
use crate::generate::PREFILL_CHUNK;
use crate::nn::{Activation, AttentionShape, CausalLm, ModelError, NormStyle, Rope, TrunkOptions};
use crate::weights::WeightStore;

/// A loaded Gemma2 model.
#[derive(Debug)]
pub struct Gemma2Model {
    config: Gemma2Config,
    lm: CausalLm,
}

impl Gemma2Model {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = Gemma2Config::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            traditional_rope: config.rope_traditional,
            qk_norm_eps: None,
            // Reference: `1.0 / (args.query_pre_attn_scalar**0.5)` (f64,
            // narrowed at the FFI boundary).
            scale_override: Some(1.0 / config.query_pre_attn_scalar.sqrt()),
            attn_logit_softcapping: Some(config.attn_logit_softcapping),
        };
        // Reference: `h * (hidden_size**0.5)` — a weak Python float, so the
        // multiply runs in the hidden dtype.
        let embed_scale = Array::from_f32((config.hidden_size as f64).sqrt() as f32);
        let rope_theta = config.rope_theta;
        let head_dim = config.head_dim;
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            // gemma2 logits always come from `embed_tokens.as_linear`.
            true,
            TrunkOptions {
                norm_style: NormStyle::OnePlus,
                activation: Activation::GeluApprox,
                sandwich_norms: true,
                clip_residual_f16: false,
                embed_scale: Some(embed_scale),
                final_logit_softcapping: Some(config.final_logit_softcapping),
            },
            |_| {
                Rope::new(
                    &crate::config::RopeScaling::Default,
                    head_dim,
                    rope_theta,
                    s,
                )
            },
            s,
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &Gemma2Config {
        &self.config
    }

    /// KV geometry for the engine's paged pools.
    pub fn kv_dims(&self) -> KvDims {
        KvDims {
            layers: self.lm.num_layers(),
            kv_heads: self.config.num_key_value_heads as i32,
            head_dim: self.config.head_dim as i32,
        }
    }

    /// Manual softcapped attention: prefill must run in the reference's own
    /// monolithic shape (see module docs). The engine expresses that as
    /// `prefill_fine_chunk >= prefill_chunk` (single-tail schedule).
    pub fn monolithic_prefill_required(&self) -> bool {
        true
    }

    /// Largest prompt whose prefill is reference-shaped: one
    /// `prefill_step_size` chunk. See the module docs.
    pub fn max_prompt_for_parity(&self) -> usize {
        PREFILL_CHUNK
    }

    /// ADR 0002 B' startup calibration (see
    /// `CausalLm::calibrate_deterministic_width`).
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }
}

impl StepModel for Gemma2Model {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
