//! Qwen3-family model (SPEC §7.2), including the GQA + qk-norm variants.
//!
//! Ported op-for-op from `mlx_lm.models.qwen3` at the pinned reference
//! version. Relative to Qwen2: `head_dim` is an explicit config field, the
//! attention projections drop their biases, and per-head RMSNorm (`q_norm` /
//! `k_norm`, shared across heads) applies to Q and K between the
//! `[B, L, H, D]` reshape and the head transpose — wired through
//! [`AttentionShape::qk_norm_eps`] into the shared [`CausalLm`] block math.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::Qwen3Config;
use crate::nn::{AttentionShape, CausalLm, ModelError, Rope};
use crate::weights::WeightStore;

/// A loaded Qwen3-family model.
#[derive(Debug)]
pub struct Qwen3Model {
    config: Qwen3Config,
    lm: CausalLm,
}

impl Qwen3Model {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = Qwen3Config::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            // mlx_lm.models.qwen3 hardcodes traditional=False.
            traditional_rope: false,
            qk_norm_eps: Some(config.rms_norm_eps),
        };
        let scaling = config.rope_scaling()?;
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            config.tie_word_embeddings,
            || Rope::new(&scaling, config.head_dim, config.rope_theta, s),
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &Qwen3Config {
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
}

impl StepModel for Qwen3Model {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
