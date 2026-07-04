//! Simple contiguous per-layer KV cache — the Phase 3 v0 cache (paged KV
//! replaces this in Phase 4, SPEC §12).
//!
//! Mechanics mirror mlx-lm's `KVCache` exactly (256-token step growth,
//! trim-to-offset before growing, functional slice writes): the buffer
//! layout does not affect numerics, but keeping the same amortized-growth
//! strategy keeps decode throughput comparable with the reference.

use kiln_mlx::{Array, MlxError, Stream, ops};

const STEP: i32 = 256;

/// KV history for one transformer layer, `[B, n_kv_heads, T, head_dim]`.
#[derive(Debug, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokens currently cached — the RoPE offset for the next step.
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Appends this step's keys/values (`[B, H, L, D]`) and returns the full
    /// `[B, H, offset+L, D]` views to attend over.
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
        s: &Stream,
    ) -> Result<(Array, Array), MlxError> {
        let (b, h, added, kd) = (keys.dim(0), keys.dim(1), keys.dim(2), keys.dim(3));
        let vd = values.dim(3);
        let prev = self.offset;

        let needs_growth = match &self.keys {
            None => true,
            Some(k) => prev + added > k.dim(2),
        };
        if needs_growth {
            let n_steps = (STEP + added - 1) / STEP;
            let dtype = keys.dtype().ok_or_else(|| MlxError {
                message: "KV arrays have a dtype outside the supported set".to_owned(),
            })?;
            let new_k = ops::zeros(&[b, h, n_steps * STEP, kd], dtype, s)?;
            let new_v = ops::zeros(&[b, h, n_steps * STEP, vd], dtype, s)?;
            match (self.keys.take(), self.values.take()) {
                (Some(mut old_k), Some(mut old_v)) => {
                    if prev % STEP != 0 {
                        old_k = ops::slice(&old_k, &[0, 0, 0, 0], &[b, h, prev, kd], s)?;
                        old_v = ops::slice(&old_v, &[0, 0, 0, 0], &[b, h, prev, vd], s)?;
                    }
                    self.keys = Some(ops::concatenate(&[&old_k, &new_k], 2, s)?);
                    self.values = Some(ops::concatenate(&[&old_v, &new_v], 2, s)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }

        self.offset += added;
        // Functional writes: the buffers are replaced by versions with this
        // step's K/V spliced in (mlx-lm's `self.keys[..., prev:offset, :] = keys`).
        let k_buf = self.keys.as_ref().ok_or_else(|| MlxError {
            message: "KV cache buffer missing after growth".to_owned(),
        })?;
        let v_buf = self.values.as_ref().ok_or_else(|| MlxError {
            message: "KV cache buffer missing after growth".to_owned(),
        })?;
        let k_buf = ops::slice_update(k_buf, keys, &[0, 0, prev, 0], &[b, h, self.offset, kd], s)?;
        let v_buf =
            ops::slice_update(v_buf, values, &[0, 0, prev, 0], &[b, h, self.offset, vd], s)?;

        let k_view = ops::slice(&k_buf, &[0, 0, 0, 0], &[b, h, self.offset, kd], s)?;
        let v_view = ops::slice(&v_buf, &[0, 0, 0, 0], &[b, h, self.offset, vd], s)?;
        self.keys = Some(k_buf);
        self.values = Some(v_buf);
        Ok((k_view, v_view))
    }

    /// The arrays to evaluate at a prefill-chunk boundary (mlx-lm's
    /// `mx.eval([c.state for c in cache])`).
    pub fn state(&self) -> Vec<&Array> {
        [self.keys.as_ref(), self.values.as_ref()]
            .into_iter()
            .flatten()
            .collect()
    }
}
