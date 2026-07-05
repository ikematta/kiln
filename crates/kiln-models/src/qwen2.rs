//! Qwen2/2.5-family model (SPEC §7.2).
//!
//! Ported op-for-op from `mlx_lm.models.qwen2` at the pinned reference
//! version. The block math is byte-identical to Llama's ([`CausalLm`]); the
//! architecture differs only in config conventions: `head_dim` is always
//! `hidden_size / num_attention_heads` (no override field), `rope_theta`
//! defaults to 1e6, and the attention projections carry `.bias` vectors
//! (picked up from the checkpoint by the shared `Linear` loader).

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::Qwen2Config;
use crate::nn::{AttentionShape, CausalLm, ModelError, Rope};
use crate::weights::WeightStore;

/// A loaded Qwen2/2.5-family model.
#[derive(Debug)]
pub struct Qwen2Model {
    config: Qwen2Config,
    lm: CausalLm,
}

impl Qwen2Model {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = Qwen2Config::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim() as i32,
            traditional_rope: config.rope_traditional,
            qk_norm_eps: None,
        };
        let scaling = config.rope_scaling()?;
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            config.tie_word_embeddings,
            || Rope::new(&scaling, config.head_dim(), config.rope_theta, s),
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// KV geometry for the engine's paged pools.
    pub fn kv_dims(&self) -> KvDims {
        KvDims {
            layers: self.lm.num_layers(),
            kv_heads: self.config.num_key_value_heads as i32,
            head_dim: self.config.head_dim() as i32,
        }
    }

    /// ADR 0002 B' startup calibration (see
    /// `CausalLm::calibrate_deterministic_width`).
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }
}

impl StepModel for Qwen2Model {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
