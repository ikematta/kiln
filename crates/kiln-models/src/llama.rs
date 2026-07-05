//! Llama-family model (SPEC §7.2): Llama 2/3/3.x, Mistral, and llamafied
//! variants.
//!
//! Ported op-for-op from `mlx_lm.models.llama` (+ `rope_utils.Llama3RoPE`)
//! at the pinned reference version — same MLX kernels in the same order, so
//! greedy decoding is bit-identical to the golden fixtures. The decode paths
//! themselves live in [`crate::nn::CausalLm`] (llama/qwen2/qwen3 share the
//! block math exactly); this module contributes the config mapping and
//! loading rules. Do not "improve" numerics without re-running the golden
//! harness; parity is the acceptance bar (SPEC §11.2).

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::LlamaConfig;
use crate::kv_cache::KvCache;
use crate::nn::{AttentionShape, CausalLm, Rope, TrunkOptions};
use crate::weights::WeightStore;

pub use crate::nn::ModelError;

/// A loaded Llama-family model.
#[derive(Debug)]
pub struct LlamaModel {
    config: LlamaConfig,
    lm: CausalLm,
}

impl LlamaModel {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = LlamaConfig::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_kv_heads() as i32,
            head_dim: config.head_dim() as i32,
            traditional_rope: config.rope_traditional,
            qk_norm_eps: None,
            scale_override: None,
            attn_logit_softcapping: None,
        };
        let scaling = config.rope_scaling()?;
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            config.tie_word_embeddings,
            TrunkOptions::default(),
            |_| Rope::new(&scaling, config.head_dim(), config.rope_theta, s),
            s,
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// KV geometry for the engine's paged pools.
    pub fn kv_dims(&self) -> KvDims {
        KvDims {
            layers: self.lm.num_layers(),
            kv_heads: self.config.num_kv_heads() as i32,
            head_dim: self.config.head_dim() as i32,
        }
    }

    /// One fresh contiguous cache per layer.
    pub fn make_cache(&self) -> Vec<KvCache> {
        (0..self.lm.num_layers()).map(|_| KvCache::new()).collect()
    }

    /// ADR 0002 B' startup calibration (see
    /// `CausalLm::calibrate_deterministic_width`): feeds
    /// `EngineConfig::deterministic_decode_width`.
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }

    /// Forward pass: `tokens [B, L]` (u32) -> logits `[B, L, vocab]`
    /// (contiguous Phase-3 cache; see [`CausalLm::forward`]).
    pub fn forward(
        &self,
        tokens: &Array,
        caches: &mut [KvCache],
        s: &Stream,
    ) -> Result<Array, ModelError> {
        self.lm.forward(tokens, 0, caches, s)
    }

    /// [`Self::forward`] whose last `pad` positions are ADR 0002
    /// kernel-class pad rows (see [`CausalLm::forward`]); used by the
    /// `generate` prefill loop, which shares the engine's canonical
    /// schedule *and* its padding rule.
    pub(crate) fn forward_padded(
        &self,
        tokens: &Array,
        pad: i32,
        caches: &mut [KvCache],
        s: &Stream,
    ) -> Result<Array, ModelError> {
        self.lm.forward(tokens, pad, caches, s)
    }
}

impl StepModel for LlamaModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
