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
use crate::nn::{Activation, Linear, Mlp, ModelError};
use crate::weights::WeightStore;

/// MoE geometry from the checkpoint's `config.json` (`num_experts`,
/// `num_experts_per_tok`, `norm_topk_prob`) — the routing/gating knobs of
/// the sparse block, resolved by the architecture module.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeOptions {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) norm_topk_prob: bool,
    /// The token shape the reference block operates on — see
    /// [`TokenLayout`]. Not cosmetic: it changes the rank handed to
    /// `gather_qmm` on the unsorted branch.
    pub(crate) layout: TokenLayout,
    /// `true` loads the always-on shared expert and its sigmoid gate
    /// (`Qwen2MoeSparseMoeBlock`); olmoe has neither.
    pub(crate) shared_expert: bool,
}

/// Which shape the reference's sparse block keeps its tokens in. The two
/// MoE families differ here and it is NOT a free choice:
///
/// - `OlmoeSparseMoeBlock` flattens first (`x_flat = x.reshape(-1, D)`) and
///   runs the router, top-k and `SwitchGLU` on 2-D token rows.
/// - `Qwen2MoeSparseMoeBlock` never flattens; every op sees `[B, L, D]`.
///
/// Traced through `SwitchGLU`, the two converge on the SORTED branch:
/// `_gather_sort`'s `x.flatten(0, -3)` collapses `[T, 1, 1, D]` and
/// `[B, L, 1, 1, D]` to the same `[T, 1, D]`, so the three `gather_qmm`
/// calls get byte-identical shapes either way. They do NOT converge on the
/// unsorted branch (`indices.size < 64`), where `gather_qmm` receives a
/// rank-4 `x` + rank-2 `indices` under `Flattened` and rank-5 + rank-3
/// under `Native`, nor at the router matmul (`[T, D]` vs `[B, L, D]`).
/// With Kiln's `B == 1` those differ only by a leading unit axis and are
/// very likely bit-identical — but "very likely" is not the bar this
/// module is held to (see the module docs), so each architecture issues
/// the shapes its own reference issues and the goldens prove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenLayout {
    /// olmoe: flatten to `[T, D]` before the router.
    Flattened,
    /// qwen2_moe: keep `[B, L, D]` throughout.
    Native,
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

    /// mlx-lm `SwitchGLU.__call__`, including the `indices.size >= 64`
    /// sort (see the module docs: the threshold is part of the op stream).
    ///
    /// Shapes follow the caller's [`TokenLayout`]: `x [T, D]` with
    /// `indices [T, k]` -> `[T, k, D]` under `Flattened`, and
    /// `x [B, L, D]` with `indices [B, L, k]` -> `[B, L, k, D]` under
    /// `Native`. `expand_dims(x, (-2, -3))` and the trailing
    /// unflatten/squeeze carry the rank difference; the sorted branch
    /// collapses both to the same `[T, 1, D]` working shape.
    fn forward(&self, x: &Array, indices: &Array, s: &Stream) -> Result<Array, MlxError> {
        let rank = x.ndim() as i32;
        let d = x.dim(rank - 1);
        let token_dims: Vec<i32> = (0..rank - 1).map(|i| x.dim(i)).collect();
        let t: i32 = token_dims.iter().product();
        let k = indices.dim(indices.ndim() as i32 - 1);
        // mx.expand_dims(x, (-2, -3)) — [..tokens.., 1, 1, D].
        let expanded_shape: Vec<i32> = token_dims.iter().copied().chain([1, 1, d]).collect();
        let x4 = ops::reshape(x, &expanded_shape, s)?;

        let do_sort = (t as i64) * (k as i64) >= 64;
        let (xg, idx, inv_order) = if do_sort {
            // _gather_sort: group token/expert pairs by expert id.
            let idx_flat = ops::reshape(indices, &[t * k], s)?;
            let order = ops::argsort(&idx_flat, -1, s)?;
            let inv_order = ops::argsort(&order, -1, s)?;
            // x.flatten(0, -3)[order // k]. `flatten(0, -3)` yields
            // [T, 1, D] from BOTH layouts' expanded shapes — this is the
            // branch where Flattened and Native converge.
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
            // _scatter_unsort: x[inv_order], then
            // `unflatten(0, indices.shape)` -> [..tokens.., k, 1, D].
            Some(inv) => {
                let y = ops::take(&y, &inv, 0, s)?;
                let unflattened: Vec<i32> = token_dims.iter().copied().chain([k, 1, d]).collect();
                ops::reshape(&y, &unflattened, s)?
            }
            None => y,
        };
        // squeeze(-2) -> [..tokens.., k, D].
        let squeezed: Vec<i32> = token_dims.iter().copied().chain([k, d]).collect();
        ops::reshape(&y, &squeezed, s)
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
    layout: TokenLayout,
    /// `Qwen2MoeSparseMoeBlock`'s always-on expert; `None` for olmoe.
    /// Boxed so the family WITHOUT a shared expert does not carry its
    /// footprint in every block (and so `FeedForward`'s variants stay
    /// within clippy's size-difference bar).
    shared_expert: Option<Box<SharedExpert>>,
}

/// The shared (unrouted) expert every token passes through, gated by a
/// scalar sigmoid — `mlp.shared_expert` + `mlp.shared_expert_gate` in
/// `Qwen2MoeSparseMoeBlock`. Its width is the checkpoint's
/// `shared_expert_intermediate_size`, independent of the routed experts'
/// `moe_intermediate_size`; both come from the weight shapes.
#[derive(Debug)]
pub(crate) struct SharedExpert {
    mlp: Mlp,
    gate: Linear,
}

impl MoeBlock {
    pub(crate) fn load(
        store: &mut WeightStore,
        mlp_prefix: &str,
        quantization: Option<Quantization>,
        opts: &MoeOptions,
        s: &Stream,
    ) -> Result<Self, ModelError> {
        let shared_expert = if opts.shared_expert {
            Some(Box::new(SharedExpert {
                mlp: Mlp::load(
                    store,
                    &format!("{mlp_prefix}.shared_expert"),
                    quantization,
                    Activation::Silu,
                )?,
                gate: Linear::load(
                    store,
                    &format!("{mlp_prefix}.shared_expert_gate"),
                    quantization,
                )?,
            }))
        } else {
            None
        };
        Ok(Self {
            gate: Linear::load(store, &format!("{mlp_prefix}.gate"), quantization)?,
            switch_mlp: SwitchGlu::load(store, mlp_prefix, opts.num_experts, quantization, s)?,
            top_k: opts.top_k as i32,
            norm_topk_prob: opts.norm_topk_prob,
            layout: opts.layout,
            shared_expert,
        })
    }

    /// `x [B, L, D] -> [B, L, D]` — mlx-lm `OlmoeSparseMoeBlock.__call__`
    /// under [`TokenLayout::Flattened`], `Qwen2MoeSparseMoeBlock.__call__`
    /// under [`TokenLayout::Native`]. The two references share every op in
    /// this body; they differ in the token shape those ops see (see
    /// [`TokenLayout`]) and in the shared expert, which only qwen2_moe has.
    pub(crate) fn forward(&self, x: &Array, s: &Stream) -> Result<Array, MlxError> {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2));
        let t = b * l;
        // olmoe flattens to [T, D] first; qwen2_moe routes on [B, L, D].
        let routed_in = match self.layout {
            TokenLayout::Flattened => ops::reshape(x, &[t, d], s)?,
            TokenLayout::Native => x.clone(),
        };
        // Token axes of whatever layout we are in: [T] or [B, L].
        let token_dims: Vec<i32> = match self.layout {
            TokenLayout::Flattened => vec![t],
            TokenLayout::Native => vec![b, l],
        };

        let router_logits = self.gate.forward(&routed_in, s)?;
        // Reference: `softmax(..., axis=1, precise=True)` on olmoe's 2-D
        // [T, E] and `axis=-1` on qwen2_moe's [B, L, E] — the last axis in
        // both, i.e. over the experts.
        let routing_weights = ops::softmax(&router_logits, -1, true, s)?;
        // argpartition(-weights, kth=k-1)[..., :k]: the top-k expert ids in
        // partition order (deliberately NOT value-sorted — see module docs).
        let neg = ops::negative(&routing_weights, s)?;
        let partitioned = ops::argpartition(&neg, self.top_k - 1, -1, s)?;
        let start = vec![0; token_dims.len() + 1];
        let stop: Vec<i32> = token_dims.iter().copied().chain([self.top_k]).collect();
        let indices = ops::slice(&partitioned, &start, &stop, s)?;
        let mut scores = ops::take_along_axis(&routing_weights, &indices, -1, s)?;
        // olmoe only: qwen2_moe's block has no `norm_topk_prob` knob and
        // never normalizes (config.rs type docs).
        if self.norm_topk_prob {
            scores = ops::divide(&scores, &ops::sum(&scores, -1, true, s)?, s)?;
        }

        let y = self.switch_mlp.forward(&routed_in, &indices, s)?;
        // (y * scores[..., None]).sum(axis=-2).
        let scores_shape: Vec<i32> = token_dims.iter().copied().chain([self.top_k, 1]).collect();
        let scores_b = ops::reshape(&scores, &scores_shape, s)?;
        let y = ops::multiply(&y, &scores_b, s)?;
        let y = ops::sum(&y, -2, false, s)?;
        // Flattened reduces to [T, D] and owes the caller [B, L, D];
        // Native is already there and the reference issues no reshape.
        let y = match self.layout {
            TokenLayout::Flattened => ops::reshape(&y, &[b, l, d], s)?,
            TokenLayout::Native => y,
        };

        match &self.shared_expert {
            None => Ok(y),
            // `y + sigmoid(shared_expert_gate(x)) * shared_expert(x)`, in
            // the reference's operand order (the gated product is formed
            // first, then added to the routed sum).
            Some(shared) => {
                // Issued in the reference's order — the expert first, then
                // its gate — so the graph is built the same way round. The
                // two are independent subgraphs, so this cannot change the
                // arithmetic; it costs nothing and leaves one less
                // difference to reason about.
                let out = shared.mlp.forward(x, s)?;
                let gate = ops::sigmoid(&shared.gate.forward(x, s)?, s)?;
                ops::add(&y, &ops::multiply(&gate, &out, s)?, s)
            }
        }
    }
}
