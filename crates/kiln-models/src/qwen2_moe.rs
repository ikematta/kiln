//! Qwen2-MoE-family model (SPEC §7.2) — Kiln's SECOND mixture-of-experts
//! architecture, and the first with a shared expert. Ported op-for-op from
//! `mlx_lm.models.qwen2_moe` at the pinned reference version.
//!
//! Relative to the OLMoE trunk (Kiln's first MoE), three things differ and
//! all three are parity-relevant:
//!
//! - **Attention is plain qwen2**: q/k/v carry `.bias` vectors, `o_proj`
//!   does not, `head_dim` is always `hidden_size / num_attention_heads`,
//!   and there is NO qk-norm (olmoe applies a full-width one). So the
//!   attention shape here is byte-identical to [`crate::qwen2`]'s and the
//!   MoE-ness is confined to the feed-forward.
//! - **The feed-forward adds a shared expert.** Every token passes through
//!   `mlp.shared_expert` (a dense gated MLP of width
//!   `shared_expert_intermediate_size`) scaled by
//!   `sigmoid(mlp.shared_expert_gate(x))`, and that product is ADDED to the
//!   routed top-k sum. The routed experts are sized by the separate
//!   `moe_intermediate_size`; neither width is `intermediate_size`, which
//!   the reference's `ModelArgs` requires but never reads.
//! - **The block does not flatten its tokens.** `Qwen2MoeSparseMoeBlock`
//!   routes on `[B, L, D]` where `OlmoeSparseMoeBlock` reshapes to
//!   `[T, D]` first. That is carried by [`TokenLayout`], not smoothed
//!   over — see its docs for the op-stream shapes that actually diverge.
//!
//! Two reference fields are deliberately NOT honored — `norm_topk_prob`
//! (this family's block never normalizes its top-k scores) and
//! `decoder_sparse_step` / `mlp_only_layers` (the reference makes every
//! layer sparse unconditionally, so Kiln refuses the configs where the two
//! would disagree rather than serve a knowingly-wrong model). Both are
//! argued in [`Qwen2MoeConfig`]'s docs.
//!
//! MoE-specific engine posture (see `AnyModel`) is inherited unchanged from
//! the architecture, exactly as ADR 0007 decision (4) requires:
//! - **Monolithic prefill**: pad rows on a MoE piece route through the gate
//!   and join real rows' expert groups, changing REAL rows' `gather_qmm`
//!   shapes — outside the ADR 0002 pad rule's empirical base. The shared
//!   expert does not soften this; it is a dense MLP evaluated for every
//!   row, so a padded piece still perturbs the routed half.
//! - **No speculation** (`speculative_gamma_bound` = None): the
//!   `gather_qmm`/`SwitchGLU` family has no kernel-class certificate under
//!   ADR 0005, and per its decision (2) a new family needs a documented
//!   geometry review plus a green spec_decode gate before `Some` here is
//!   permission. That review does not exist for qwen2_moe either.
//!
//! Verification scope: as with OLMoE, correctness is proven on M4-class
//! hardware only — see `docs/decisions/0007-moe-kernel-dispatch-hardware-
//! scope.md` for the large-MoE deployment gap this does not close.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::Qwen2MoeConfig;
use crate::moe::{MoeOptions, TokenLayout};
use crate::nn::{AttentionShape, CausalLm, ModelError, Rope, TrunkOptions};
use crate::weights::WeightStore;

/// A loaded Qwen2-MoE-family model.
#[derive(Debug)]
pub struct Qwen2MoeModel {
    config: Qwen2MoeConfig,
    lm: CausalLm,
}

impl Qwen2MoeModel {
    /// Loads config + weights from a local model directory.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let config = Qwen2MoeConfig::from_model_dir(dir)?;
        let store = WeightStore::from_model_dir(dir)?;
        let shape = AttentionShape {
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_kv_heads() as i32,
            head_dim: config.head_dim() as i32,
            traditional_rope: config.rope_traditional,
            // qwen2 attention: no qk-norm of either placement.
            qk_norm_eps: None,
            qk_norm_full_width: false,
            scale_override: None,
            attn_logit_softcapping: None,
        };
        let opts = TrunkOptions {
            moe: Some(MoeOptions {
                num_experts: config.num_experts,
                top_k: config.num_experts_per_tok,
                // `Qwen2MoeSparseMoeBlock` has no `norm_topk_prob` knob and
                // never normalizes — hardcoded, NOT read from config.json,
                // so a checkpoint carrying `true` cannot silently diverge
                // from the reference (config.rs type docs).
                norm_topk_prob: false,
                layout: TokenLayout::Native,
                shared_expert: true,
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

    pub fn config(&self) -> &Qwen2MoeConfig {
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
    /// expert path (router, top-k, gather dispatch, combine) in addition to
    /// the plain projections — and here the shared expert rides along in
    /// the same probe, since it is part of the same block — see
    /// `CausalLm::calibrate_deterministic_width`.
    pub fn calibrate_deterministic_width(&self, s: &Stream) -> Result<usize, ModelError> {
        self.lm.calibrate_deterministic_width(s)
    }
}

impl StepModel for Qwen2MoeModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.lm.forward_step(batch, kv, s)
    }
}
