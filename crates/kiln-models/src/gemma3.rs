//! Gemma3 text-only model (SPEC §7.2, Phase 6 Task 2).
//!
//! Ported op-for-op from `mlx_lm.models.gemma3_text` at the pinned reference
//! version: `1 + w` RMSNorms, qk-norm, the sandwich block (post-attention /
//! pre- and post-feedforward norms around the sublayer outputs), f16
//! residual clipping (`clip_residual`), `gelu_approx` MLP,
//! `sqrt(hidden_size)` embedding scaling (a bf16 constant cast to the
//! hidden dtype), per-layer rope (local base on sliding layers,
//! `rope_theta` + optional `rope_scaling` on every
//! `sliding_window_pattern`-th layer), and `query_pre_attn_scalar**-0.5`
//! attention scaling.
//!
//! **Sliding-window status (parity envelope):** the reference implements the
//! window with a `RotatingKVCache` (`max_size = sliding_window`, `keep = 0`)
//! whose contents are exactly a plain temporal cache until the total
//! context EXCEEDS `sliding_window`; the window masks equal plain causal
//! masks in the same regime. Kiln currently implements exactly that regime
//! and refuses longer requests via [`Gemma3Model::max_context_for_parity`]
//! (worker-enforced): serving past the window without ring-order gather
//! would silently diverge from the reference AND from the model's own
//! semantics. Lifting the cap (ring-order decode gather + window-crossing
//! fixtures) is the recorded follow-up in PROGRESS.md.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, Dtype, MlxError, Stream, ops};

use crate::config::{Gemma3Config, RopeScaling};
use crate::nn::{Activation, AttentionShape, CausalLm, ModelError, NormStyle, Rope, TrunkOptions};
use crate::weights::WeightStore;

/// A loaded Gemma3 text model.
#[derive(Debug)]
pub struct Gemma3Model {
    config: Gemma3Config,
    lm: CausalLm,
}

impl Gemma3Model {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = Gemma3Config::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            // mlx_lm.models.gemma3_text hardcodes traditional=False.
            traditional_rope: false,
            qk_norm_eps: Some(config.rms_norm_eps),
            // Reference: `args.query_pre_attn_scalar**-0.5` (f64, narrowed
            // at the FFI boundary).
            scale_override: Some(config.query_pre_attn_scalar.powf(-0.5)),
            attn_logit_softcapping: None,
        };
        // Reference: `mx.array(hidden_size**0.5, mx.bfloat16)`, cast to the
        // hidden dtype per forward.
        let embed_scale = ops::astype(
            &Array::from_f32((config.hidden_size as f64).sqrt() as f32),
            Dtype::Bfloat16,
            s,
        )?;
        embed_scale.eval()?;
        // Reference `sanitize`: tied iff the checkpoint has no lm_head.
        let tie_word_embeddings = !store.contains("lm_head.weight");
        let scaling = config.rope_scaling()?;
        let pattern = config.sliding_window_pattern.max(1);
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            tie_word_embeddings,
            TrunkOptions {
                norm_style: NormStyle::OnePlus,
                activation: Activation::GeluApprox,
                sandwich_norms: true,
                clip_residual_f16: true,
                embed_scale: Some(embed_scale),
                final_logit_softcapping: None,
            },
            |layer| {
                // `is_sliding = (layer_idx + 1) % sliding_window_pattern != 0`
                if (layer + 1) % pattern != 0 {
                    Rope::new(
                        &RopeScaling::Default,
                        config.head_dim,
                        config.rope_local_base_freq,
                        s,
                    )
                } else {
                    Rope::new(&scaling, config.head_dim, config.rope_theta, s)
                }
            },
            s,
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &Gemma3Config {
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

    /// Largest total context (prompt + generated) whose op stream is
    /// reference-shaped: the sliding window. See the module docs.
    pub fn max_context_for_parity(&self) -> usize {
        self.config.sliding_window
    }

    /// ADR 0002 B' startup calibration (see
    /// `CausalLm::calibrate_deterministic_width`).
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }
}

impl StepModel for Gemma3Model {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
