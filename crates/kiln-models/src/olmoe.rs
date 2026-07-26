//! OLMoE-family model (SPEC §7.2) — Kiln's first mixture-of-experts
//! architecture. Ported op-for-op from `mlx_lm.models.olmoe` at the pinned
//! reference version.
//!
//! Relative to the llama trunk: attention applies a FULL-WIDTH RMSNorm to
//! the raw q/k projection outputs before the head reshape
//! (`q_norm(q_proj(x))` — different reduction width than qwen3's per-head
//! norm), and every block's feed-forward is the sparse-MoE block from
//! `moe.rs` (softmax router, `argpartition` top-k, `SwitchGLU` expert
//! dispatch, routing-weighted combine), parameterized entirely by
//! `config.json`'s `num_experts` / `num_experts_per_tok` / `norm_topk_prob`
//! / `intermediate_size` — nothing is sized to the proxy checkpoint.
//!
//! MoE-specific engine posture (see `AnyModel`):
//! - **Monolithic prefill** (`monolithic_prefill_required` = true): the
//!   ADR 0002 bar-(3) kernel-class pad was validated on dense-MLP quantized
//!   trunks only. On a MoE trunk, pad rows would not merely raise the row
//!   count of a fixed matmul — they route through the gate and JOIN real
//!   rows' expert groups, changing the gather_qmm shapes (and possibly the
//!   sort-threshold branch) of REAL rows. That is outside the pad rule's
//!   empirical base, so MoE trunks take reference-shaped prefill pieces
//!   only — the ADR 0002 addendum precedent (dense trunks) applied to a
//!   new op family.
//! - **No speculation** (`speculative_gamma_bound` = None): ADR 0005's
//!   envelope certifies SDPA + trunk-matmul kernel classes; the expert
//!   gather path is a new, unreviewed kernel family, and per the ADR's
//!   decision (2) a new architecture needs a documented geometry review
//!   plus a green spec_decode gate on the generating device before any
//!   `Some` here is permission. Tracked in the SPEC §7.2 MoE backlog note.
//!
//! Verification scope: correctness is proven on the pinned OLMoE proxy on
//! M4-class hardware. Kernel-dispatch-class behavior on the large-MoE
//! deployment target (M3 Ultra / 128 GB) is explicitly NOT certified by
//! this module's tests — see `docs/decisions/0007-moe-kernel-dispatch-
//! hardware-scope.md`.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::OlmoeConfig;
use crate::moe::MoeOptions;
use crate::nn::{AttentionShape, CausalLm, ModelError, Rope, TrunkOptions};
use crate::weights::WeightStore;

/// A loaded OLMoE-family model.
#[derive(Debug)]
pub struct OlmoeModel {
    config: OlmoeConfig,
    lm: CausalLm,
}

impl OlmoeModel {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = OlmoeConfig::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_kv_heads() as i32,
            head_dim: config.head_dim() as i32,
            traditional_rope: config.rope_traditional,
            qk_norm_eps: Some(config.rms_norm_eps),
            // mlx_lm.models.olmoe: q_norm/k_norm are nn.RMSNorm over the
            // FULL projection width, applied before the head reshape.
            qk_norm_full_width: true,
            scale_override: None,
            attn_logit_softcapping: None,
        };
        let opts = TrunkOptions {
            moe: Some(MoeOptions {
                num_experts: config.num_experts,
                top_k: config.num_experts_per_tok,
                norm_topk_prob: config.norm_topk_prob,
            }),
            ..TrunkOptions::default()
        };
        let scaling = config.rope_scaling()?;
        let head_dim = config.head_dim();
        let lm = CausalLm::load(
            store,
            config.quantization,
            config.num_hidden_layers,
            &shape,
            config.rms_norm_eps,
            config.tie_word_embeddings,
            opts,
            |_| Rope::new(&scaling, head_dim, config.rope_theta, s),
            s,
        )?;
        Ok(Self { config, lm })
    }

    pub fn config(&self) -> &OlmoeConfig {
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

    /// ADR 0002 B' startup calibration. On MoE trunks this probes the full
    /// expert path (router, top-k, gather dispatch, combine) in addition
    /// to the plain projections — see
    /// `CausalLm::calibrate_deterministic_width`.
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }
}

impl StepModel for OlmoeModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
