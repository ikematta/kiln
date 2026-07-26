//! Mixture-of-experts building blocks (SPEC §7.2), ported op-for-op from the
//! pinned reference: `mlx_lm.models.switch_layers` (`SwitchLinear`,
//! `QuantizedSwitchLinear`, `SwitchGLU`) and the sparse-MoE router block
//! shape shared by the OLMoE/Qwen-MoE families (`OlmoeSparseMoeBlock`).
//! Everything is parameterized by the checkpoint's config — expert count,
//! top-k, per-expert hidden width — never hardcoded to a model size.
//!
//! Parity-critical contracts (do not "improve" without re-running goldens):
//!
//! - **The sort threshold is part of the reference op stream.** `SwitchGLU`
//!   sorts token/expert pairs by expert id when `indices.size >= 64`
//!   (`do_sort`), and the `sorted_indices` hint changes gather-kernel
//!   scheduling. Kiln computes the same predicate over the same shapes, so
//!   a single-stream forward reproduces the reference's kernel sequence
//!   exactly. Batched steps change `indices.size` relative to M=1 — that is
//!   a new (MoE-specific) kernel-dispatch axis on top of ADR 0002's
//!   qmv/qmm boundary, and it is why `CausalLm::calibrate_deterministic_width`
//!   probes the whole MoE block (router + top-k + expert dispatch +
//!   combine), not just plain projections, on MoE trunks.
//! - **The reference's `swiglu` is an `mx.compile`d closure.** Measured at
//!   the pin (2026-07-25, M4 dev machine): the compiled kernel is
//!   bit-identical to the eager `silu(gate) * up` graph for f16 operands
//!   across the shapes this module produces, so Kiln issues the eager ops
//!   (same graph the dense [`crate::nn`] MLP uses). The golden harness is
//!   the standing proof; if a future pin breaks this equivalence the
//!   goldens catch it on the generating device.
//! - **Routing order ties bits.** `argpartition` leaves the top-k in
//!   implementation order (not sorted by weight), and the weighted combine
//!   sums experts in that order — identical op streams give identical
//!   float sums, so Kiln mirrors `argpartition(-weights)[..., :k]` exactly
//!   rather than substituting `topk`.

use kiln_mlx::{Array, MlxError, Stream, ops};

use crate::config::Quantization;
use crate::nn::{Linear, ModelError};
use crate::weights::WeightStore;

/// MoE geometry from the checkpoint's `config.json` (`num_experts`,
/// `num_experts_per_tok`, `norm_topk_prob`) — the routing/gating knobs of
/// the sparse block, resolved by the architecture module.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeOptions {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) norm_topk_prob: bool,
}

/// One per-expert projection stack: `weight [E, out, in]` (packed u32 when
/// quantized), applied via `gather_qmm`/`gather_mm` on per-token expert
/// indices. Quantized iff the checkpoint has `.scales` for it — the same
/// per-module rule as [`Linear`].
#[derive(Debug)]
pub(crate) enum SwitchLinear {
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

impl SwitchLinear {
    /// Loads `{mlp_prefix}.switch_mlp.{proj}` in either checkpoint form:
    /// already-stacked (`switch_mlp.{proj}.weight` = `[E, out, in]`, what
    /// mlx-lm itself saves after conversion) or per-expert
    /// (`experts.{e}.{proj}.weight`), stacked here exactly as the
    /// reference's `sanitize` does (`mx.stack` over experts, axis 0).
    fn load(
        store: &mut WeightStore,
        mlp_prefix: &str,
        proj: &str,
        num_experts: usize,
        quantization: Option<Quantization>,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        let stacked = format!("{mlp_prefix}.switch_mlp.{proj}");
        let per_expert = |name: &str| format!("{mlp_prefix}.experts.0.{proj}.{name}");

        if store.contains(&format!("{stacked}.scales")) || store.contains(&per_expert("scales")) {
            let q = quantization.ok_or_else(|| {
                ModelError::Mismatch(format!(
                    "{stacked} has quantized tensors but config.json has no quantization block"
                ))
            })?;
            let (weight, scales, biases) = if store.contains(&format!("{stacked}.scales")) {
                (
                    store.take(&format!("{stacked}.weight"))?,
                    store.take(&format!("{stacked}.scales"))?,
                    store.take(&format!("{stacked}.biases"))?,
                )
            } else {
                (
                    stack_experts(store, mlp_prefix, proj, "weight", num_experts, s)?,
                    stack_experts(store, mlp_prefix, proj, "scales", num_experts, s)?,
                    stack_experts(store, mlp_prefix, proj, "biases", num_experts, s)?,
                )
            };
            let bias = load_bias(store, mlp_prefix, proj, &stacked, num_experts, s)?;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                bias,
                group_size: q.group_size,
                bits: q.bits,
            })
        } else {
            let weight = if store.contains(&format!("{stacked}.weight")) {
                store.take(&format!("{stacked}.weight"))?
            } else {
                stack_experts(store, mlp_prefix, proj, "weight", num_experts, s)?
            };
            let bias = load_bias(store, mlp_prefix, proj, &stacked, num_experts, s)?;
            Ok(Self::Dense { weight, bias })
        }
    }

    /// `x @ W[indices]^T (+ bias[indices])` — mlx-lm
    /// `(Quantized)SwitchLinear.__call__`. `sorted_indices` must be exactly
    /// the caller's `do_sort` (it is a kernel-scheduling hint the reference
    /// passes through, part of the reproduced op stream).
    fn forward(
        &self,
        x: &Array,
        indices: &Array,
        sorted_indices: bool,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let (y, bias) = match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            } => (
                ops::gather_qmm(
                    x,
                    weight,
                    scales,
                    biases,
                    indices,
                    true,
                    *group_size,
                    *bits,
                    sorted_indices,
                    s,
                )?,
                bias,
            ),
            Self::Dense { weight, bias } => {
                // SwitchLinear: x @ weight.swapaxes(-1, -2)[indices].
                let wt = ops::transpose(weight, &[0, 2, 1], s)?;
                (ops::gather_mm(x, &wt, indices, sorted_indices, s)?, bias)
            }
        };
        match bias {
            Some(b) => {
                // `y + mx.expand_dims(bias[indices], -2)`.
                let gathered = ops::take(b, indices, 0, s)?;
                let mut shape = gathered.shape();
                shape.insert(shape.len() - 1, 1);
                ops::add(&y, &ops::reshape(&gathered, &shape, s)?, s)
            }
            None => Ok(y),
        }
    }
}

/// Stacks `experts.{0..E}.{proj}.{name}` along a new axis 0, materialized
/// at load (the reference evals all parameters after `sanitize`; keeping 64
/// mmap-backed sources per stack alive in the graph would also pin the
/// whole checkpoint mapping).
fn stack_experts(
    store: &mut WeightStore,
    mlp_prefix: &str,
    proj: &str,
    name: &str,
    num_experts: usize,
    s: &Stream,
) -> Result<Array, ModelError> {
    let parts: Vec<Array> = (0..num_experts)
        .map(|e| store.take(&format!("{mlp_prefix}.experts.{e}.{proj}.{name}")))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&Array> = parts.iter().collect();
    let out = ops::stack(&refs, 0, s)?;
    out.eval()?;
    Ok(out)
}

/// Optional `[E, out]` bias (`mlp_bias` checkpoints), in either form.
fn load_bias(
    store: &mut WeightStore,
    mlp_prefix: &str,
    proj: &str,
    stacked: &str,
    num_experts: usize,
    s: &Stream,
) -> Result<Option<Array>, ModelError> {
    if let Some(bias) = store.take_optional(&format!("{stacked}.bias")) {
        return Ok(Some(bias));
    }
    if store.contains(&format!("{mlp_prefix}.experts.0.{proj}.bias")) {
        return Ok(Some(stack_experts(
            store,
            mlp_prefix,
            proj,
            "bias",
            num_experts,
            s,
        )?));
    }
    Ok(None)
}

/// The gated expert MLP (`SwitchGLU`): gate/up/down per-expert stacks with
/// the `silu(gate) * up` activation, computed only for each token's
/// routed experts.
#[derive(Debug)]
pub(crate) struct SwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGlu {
    fn load(
        store: &mut WeightStore,
        mlp_prefix: &str,
        num_experts: usize,
        quantization: Option<Quantization>,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            gate_proj: SwitchLinear::load(
                store,
                mlp_prefix,
                "gate_proj",
                num_experts,
                quantization,
                s,
            )?,
            up_proj: SwitchLinear::load(
                store,
                mlp_prefix,
                "up_proj",
                num_experts,
                quantization,
                s,
            )?,
            down_proj: SwitchLinear::load(
                store,
                mlp_prefix,
                "down_proj",
                num_experts,
                quantization,
                s,
            )?,
        })
    }

    /// `x_flat [T, D]`, `indices [T, k]` -> `[T, k, D]` — mlx-lm
    /// `SwitchGLU.__call__`, including the `indices.size >= 64` sort
    /// (see the module docs: the threshold is part of the op stream).
    fn forward(&self, x_flat: &Array, indices: &Array, s: &Stream) -> Result<Array, MlxError> {
        let (t, d) = (x_flat.dim(0), x_flat.dim(1));
        let k = indices.dim(1);
        // mx.expand_dims(x, (-2, -3)).
        let x4 = ops::reshape(x_flat, &[t, 1, 1, d], s)?;

        let do_sort = (t as i64) * (k as i64) >= 64;
        let (xg, idx, inv_order) = if do_sort {
            // _gather_sort: group token/expert pairs by expert id.
            let idx_flat = ops::reshape(indices, &[t * k], s)?;
            let order = ops::argsort(&idx_flat, -1, s)?;
            let inv_order = ops::argsort(&order, -1, s)?;
            // x.flatten(0, -3)[order // k].
            let xf = ops::reshape(&x4, &[t, 1, d], s)?;
            let divisor = Array::from_u32_slice(&[k as u32], &[1])?;
            let rows = ops::floor_divide(&order, &divisor, s)?;
            let xg = ops::take(&xf, &rows, 0, s)?;
            let idx = ops::take(&idx_flat, &order, 0, s)?;
            (xg, idx, Some(inv_order))
        } else {
            (x4, indices.clone(), None)
        };

        let x_up = self.up_proj.forward(&xg, &idx, do_sort, s)?;
        let x_gate = self.gate_proj.forward(&xg, &idx, do_sort, s)?;
        // swiglu(gate, up) = silu(gate) * up, issued eagerly (measured
        // bit-identical to the reference's compiled closure at the pin —
        // module docs).
        let activated = ops::multiply(
            &ops::multiply(&x_gate, &ops::sigmoid(&x_gate, s)?, s)?,
            &x_up,
            s,
        )?;
        let y = self.down_proj.forward(&activated, &idx, do_sort, s)?;

        let y = match inv_order {
            // _scatter_unsort: x[inv_order], then unflatten to [T, k, ...].
            Some(inv) => {
                let y = ops::take(&y, &inv, 0, s)?;
                ops::reshape(&y, &[t, k, 1, d], s)?
            }
            None => y,
        };
        // squeeze(-2).
        ops::reshape(&y, &[t, k, d], s)
    }
}

/// The sparse-MoE feed-forward block (`OlmoeSparseMoeBlock`): a softmax
/// router over `num_experts`, top-k selection via `argpartition`, expert
/// evaluation through [`SwitchGlu`], and the routing-weighted combine.
#[derive(Debug)]
pub(crate) struct MoeBlock {
    /// The router (`mlp.gate`) — a plain [`Linear`], quantized or dense per
    /// the checkpoint like every other projection.
    pub(crate) gate: Linear,
    switch_mlp: SwitchGlu,
    top_k: i32,
    norm_topk_prob: bool,
}

impl MoeBlock {
    pub(crate) fn load(
        store: &mut WeightStore,
        mlp_prefix: &str,
        quantization: Option<Quantization>,
        opts: &MoeOptions,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            gate: Linear::load(store, &format!("{mlp_prefix}.gate"), quantization)?,
            switch_mlp: SwitchGlu::load(store, mlp_prefix, opts.num_experts, quantization, s)?,
            top_k: opts.top_k as i32,
            norm_topk_prob: opts.norm_topk_prob,
        })
    }

    /// `x [B, L, D] -> [B, L, D]` — mlx-lm `OlmoeSparseMoeBlock.__call__`.
    pub(crate) fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2));
        let t = b * l;
        let x_flat = ops::reshape(x, &[t, d], s)?;

        let router_logits = self.gate.forward(&x_flat, s)?;
        // Reference: mx.softmax(router_logits, axis=1, precise=True) on the
        // 2-D [T, E] — axis 1 is the last axis.
        let routing_weights = ops::softmax(&router_logits, -1, true, s)?;
        // argpartition(-weights, kth=k-1)[..., :k]: the top-k expert ids in
        // partition order (deliberately NOT value-sorted — see module docs).
        let neg = ops::negative(&routing_weights, s)?;
        let partitioned = ops::argpartition(&neg, self.top_k - 1, -1, s)?;
        let indices = ops::slice(&partitioned, &[0, 0], &[t, self.top_k], s)?;
        let mut scores = ops::take_along_axis(&routing_weights, &indices, -1, s)?;
        if self.norm_topk_prob {
            scores = ops::divide(&scores, &ops::sum(&scores, -1, true, s)?, s)?;
        }

        let y = self.switch_mlp.forward(&x_flat, &indices, s)?;
        // (y * scores[..., None]).sum(axis=-2).
        let scores3 = ops::reshape(&scores, &[t, self.top_k, 1], s)?;
        let y = ops::multiply(&y, &scores3, s)?;
        let y = ops::sum(&y, -2, false, s)?;
        ops::reshape(&y, &[b, l, d], s)
    }
}
