//! Physical paged-KV storage (SPEC §6.3): per-layer K and V pools as
//! preallocated MLX arrays, addressed by [`BlockId`]s from the block
//! manager. The gather-based paged-attention strategy (SPEC §7.4 v0) reads
//! a request's history by gathering its blocks into a contiguous
//! per-request view; the custom block-table-aware Metal kernel is Phase 7.
//!
//! Layout: each pool is `[num_blocks, kv_heads, block_size, head_dim]`
//! (head-major inside a block), so
//! - writes are plain `slice_update`s of `[1, H, run, D]` segments straight
//!   out of the step's post-RoPE K/V (no transpose), and
//! - a gathered view reshapes to `[1, H, tokens, head_dim]` with the same
//!   "row-contiguous per head, strided head planes" stride pattern the
//!   Phase-3 contiguous cache handed to fused SDPA — keeping the kernel's
//!   input layout, and therefore its arithmetic, identical to the path the
//!   golden fixtures were validated against.
//!
//! Updates are functional (mlx `slice_update`), chained onto the pool
//! handles this struct owns. All writes for a step happen before any
//! gather references the final pool version, so the intermediate versions
//! have exactly one consumer each and MLX's donation turns the chain into
//! in-place writes at eval.

use kiln_mlx::{Array, Dtype, MlxError, Stream, ops};

use crate::block::{BlockId, CowCopy};
use crate::paged_attn::{PagedAttn, PagedAttnInputs};

/// Static dimensions of one paged pool set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSpec {
    /// Transformer layers (one K and one V pool each).
    pub layers: usize,
    /// KV heads per layer.
    pub kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Blocks per pool (the block manager's capacity).
    pub num_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
}

impl KvSpec {
    /// Bytes one block occupies across all layers (K and V) at the given
    /// element size — the single source of the pool-cost formula
    /// (SPEC §2.3/§6.4 admission projections). [`PagedKv::bytes_per_block`]
    /// applies it with the materialized dtype; the worker applies it at
    /// load with the 16-bit element size every rust-servable checkpoint
    /// computes in (SPEC §7.3: quantized-with-16-bit-scales or f16/bf16
    /// dense) to report pool geometry before any traffic arrives.
    pub fn bytes_per_block(&self, element_size: usize) -> u64 {
        element_size as u64
            * 2
            * self.layers as u64
            * self.kv_heads as u64
            * self.block_size as u64
            * self.head_dim as u64
    }
}

/// One contiguous run of token rows to write into a single block.
///
/// A step segment of `len` tokens covers at most `len / block_size + 1`
/// runs; the engine derives them from the request's block table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteRun {
    /// Destination block.
    pub block: BlockId,
    /// First row inside the block.
    pub row_start: i32,
    /// First source token within the step segment.
    pub src_start: i32,
    /// Rows written.
    pub len: i32,
}

/// Per-layer K/V pools. Holds no ownership state — that is the block
/// manager's job; this type only moves bytes.
#[derive(Debug)]
pub struct PagedKv {
    spec: KvSpec,
    /// `(keys, values)` per layer; `None` until the first write fixes the
    /// dtype (the model's activation dtype, known only at runtime).
    pools: Vec<Option<(Array, Array)>>,
    /// Custom block-table-aware attention kernels (SPEC §7.4 Phase 7),
    /// present iff the config flag enabled them. `None` = gather path only.
    attn: Option<PagedAttn>,
}

impl PagedKv {
    pub fn new(spec: KvSpec) -> Self {
        let mut pools = Vec::with_capacity(spec.layers);
        pools.resize_with(spec.layers, || None);
        Self {
            spec,
            pools,
            attn: None,
        }
    }

    pub fn spec(&self) -> &KvSpec {
        &self.spec
    }

    /// Builds the custom paged-attention kernels (SPEC §7.4 flag ON). The
    /// gather path stays available; models route decode-shaped steps here
    /// via [`Self::paged_sdpa`] only when this succeeded.
    pub fn enable_attention_kernel(&mut self) -> Result<(), MlxError> {
        if self.attn.is_none() {
            self.attn = Some(PagedAttn::new()?);
        }
        Ok(())
    }

    pub fn attention_kernel_enabled(&self) -> bool {
        self.attn.is_some()
    }

    /// Block-table-aware decode attention over this sequence's pooled
    /// history: `q` `[1, H, 1, D]` (post-RoPE, this step's K/V already
    /// written), output `[1, H, 1, D]` — same contract as gathering
    /// `inputs.context_len` tokens and calling fused SDPA, without the
    /// copy. Callers gate on [`Self::attention_kernel_enabled`].
    pub fn paged_sdpa(
        &self,
        layer: usize,
        q: &Array,
        inputs: &PagedAttnInputs,
        scale: &Array,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let attn = self.attn.as_ref().ok_or_else(|| MlxError {
            message: "paged_sdpa called with the attention kernel disabled".to_owned(),
        })?;
        let (k_pool, v_pool) = self
            .pools
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| MlxError {
                message: format!("paged sdpa on unwritten layer {layer}"),
            })?;
        attn.sdpa(&self.spec, k_pool, v_pool, q, inputs, scale, s)
    }

    /// Writes one step segment's post-RoPE `keys`/`values`
    /// (`[1, H, len, D]`) into the layer's pools at `runs`.
    pub fn write(
        &mut self,
        layer: usize,
        runs: &[WriteRun],
        keys: &Array,
        values: &Array,
        s: &Stream,
    ) -> Result<(), MlxError> {
        let (h, d) = (self.spec.kv_heads, self.spec.head_dim);
        let shape = [
            self.spec.num_blocks as i32,
            h,
            self.spec.block_size as i32,
            d,
        ];
        let slot = self.pools.get_mut(layer).ok_or_else(|| MlxError {
            message: format!("paged KV write to layer {layer} of {}", self.spec.layers),
        })?;
        let (mut k_pool, mut v_pool) = match slot.take() {
            Some(pools) => pools,
            None => {
                let dtype = keys.dtype().ok_or_else(|| MlxError {
                    message: "step keys have a dtype outside the supported set".to_owned(),
                })?;
                (ops::zeros(&shape, dtype, s)?, ops::zeros(&shape, dtype, s)?)
            }
        };
        for run in runs {
            let b = run.block.index() as i32;
            let start = [b, 0, run.row_start, 0];
            let stop = [b + 1, h, run.row_start + run.len, d];
            let src = [0, 0, run.src_start, 0];
            let src_stop = [1, h, run.src_start + run.len, d];
            let k_src = ops::slice(keys, &src, &src_stop, s)?;
            let v_src = ops::slice(values, &src, &src_stop, s)?;
            k_pool = ops::slice_update(&k_pool, &k_src, &start, &stop, s)?;
            v_pool = ops::slice_update(&v_pool, &v_src, &start, &stop, s)?;
        }
        *slot = Some((k_pool, v_pool));
        Ok(())
    }

    /// Gathers `blocks` into contiguous per-request K/V views trimmed to
    /// `len` tokens: `([1, H, len, D], [1, H, len, D])`.
    pub fn gather(
        &self,
        layer: usize,
        blocks: &[BlockId],
        len: i32,
        s: &Stream,
    ) -> Result<(Array, Array), MlxError> {
        let (h, d, bs) = (
            self.spec.kv_heads,
            self.spec.head_dim,
            self.spec.block_size as i32,
        );
        let (k_pool, v_pool) = self
            .pools
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| MlxError {
                message: format!("paged KV gather from unwritten layer {layer}"),
            })?;
        let n = blocks.len() as i32;
        if len < 1 || len > n * bs {
            return Err(MlxError {
                message: format!("paged KV gather of {len} tokens from {n} block(s)"),
            });
        }
        let ids: Vec<u32> = blocks.iter().map(|b| b.index() as u32).collect();
        let ids = Array::from_u32_slice(&ids, &[n])?;
        let gather_one = |pool: &Array| -> Result<Array, MlxError> {
            let g = ops::take(pool, &ids, 0, s)?; // [n, H, bs, D]
            let g = ops::transpose(&g, &[1, 0, 2, 3], s)?; // [H, n, bs, D]
            let g = ops::reshape(&g, &[1, h, n * bs, d], s)?; // contiguous
            ops::slice(&g, &[0, 0, 0, 0], &[1, h, len, d], s)
        };
        Ok((gather_one(k_pool)?, gather_one(v_pool)?))
    }

    /// Executes a copy-on-write instruction: `src` block's rows into `dst`,
    /// in every allocated layer (must run before the step's writes chain
    /// onto the pools, which the engine's build order guarantees).
    pub fn copy_block(&mut self, cow: CowCopy, s: &Stream) -> Result<(), MlxError> {
        let (h, d, bs) = (
            self.spec.kv_heads,
            self.spec.head_dim,
            self.spec.block_size as i32,
        );
        let (src, dst) = (cow.src.index() as i32, cow.dst.index() as i32);
        for slot in &mut self.pools {
            if let Some((k_pool, v_pool)) = slot.take() {
                let k_rows = ops::slice(&k_pool, &[src, 0, 0, 0], &[src + 1, h, bs, d], s)?;
                let v_rows = ops::slice(&v_pool, &[src, 0, 0, 0], &[src + 1, h, bs, d], s)?;
                let start = [dst, 0, 0, 0];
                let stop = [dst + 1, h, bs, d];
                *slot = Some((
                    ops::slice_update(&k_pool, &k_rows, &start, &stop, s)?,
                    ops::slice_update(&v_pool, &v_rows, &start, &stop, s)?,
                ));
            }
        }
        Ok(())
    }

    /// The pools' element dtype, once fixed by the first write (or by
    /// [`Self::ensure_pools`]).
    pub fn dtype(&self) -> Option<Dtype> {
        self.pools
            .iter()
            .flatten()
            .next()
            .and_then(|(k, _)| k.dtype())
    }

    /// Materializes every layer's pools at `dtype` — the SSD warm-load
    /// path, where a block may be uploaded before any step has written
    /// (and therefore before the dtype would be discovered).
    pub fn ensure_pools(&mut self, dtype: Dtype, s: &Stream) -> Result<(), MlxError> {
        let shape = [
            self.spec.num_blocks as i32,
            self.spec.kv_heads,
            self.spec.block_size as i32,
            self.spec.head_dim,
        ];
        for slot in &mut self.pools {
            if slot.is_none() {
                *slot = Some((ops::zeros(&shape, dtype, s)?, ops::zeros(&shape, dtype, s)?));
            }
        }
        Ok(())
    }

    /// Serializes one block: for each layer, the K rows then the V rows
    /// (`kv_heads * block_size * head_dim` elements each) as raw bytes —
    /// the SSD slab payload (see `crate::ssd` for the layout). Forces
    /// evaluation of the pool chain, so callers only capture settled
    /// blocks outside the pipelined fast path.
    pub fn read_block_bytes(&self, block: BlockId, s: &Stream) -> Result<Vec<u8>, MlxError> {
        let (h, d, bs) = (
            self.spec.kv_heads,
            self.spec.head_dim,
            self.spec.block_size as i32,
        );
        let b = block.index() as i32;
        let mut bytes = Vec::new();
        for (layer, slot) in self.pools.iter().enumerate() {
            let (k_pool, v_pool) = slot.as_ref().ok_or_else(|| MlxError {
                message: format!("block capture from unwritten layer {layer}"),
            })?;
            for pool in [k_pool, v_pool] {
                let rows = ops::slice(pool, &[b, 0, 0, 0], &[b + 1, h, bs, d], s)?;
                let rows = ops::contiguous(&rows, s)?;
                bytes.extend_from_slice(&rows.data_raw_bytes()?);
            }
        }
        Ok(bytes)
    }

    /// Uploads one serialized block (the inverse of
    /// [`Self::read_block_bytes`]) into the pools at `dtype`.
    pub fn write_block_bytes(
        &mut self,
        block: BlockId,
        bytes: &[u8],
        dtype: Dtype,
        s: &Stream,
    ) -> Result<(), MlxError> {
        self.ensure_pools(dtype, s)?;
        let (h, d, bs) = (
            self.spec.kv_heads,
            self.spec.head_dim,
            self.spec.block_size as i32,
        );
        let plane = (h * bs * d) as usize * dtype.size();
        if bytes.len() != plane * 2 * self.spec.layers {
            return Err(MlxError {
                message: format!(
                    "block payload of {} bytes does not match the pool geometry ({} expected)",
                    bytes.len(),
                    plane * 2 * self.spec.layers
                ),
            });
        }
        let run = [WriteRun {
            block,
            row_start: 0,
            src_start: 0,
            len: bs,
        }];
        for layer in 0..self.spec.layers {
            let at = layer * 2 * plane;
            let keys = Array::from_raw_bytes(&bytes[at..at + plane], &[1, h, bs, d], dtype)?;
            let values =
                Array::from_raw_bytes(&bytes[at + plane..at + 2 * plane], &[1, h, bs, d], dtype)?;
            self.write(layer, &run, &keys, &values, s)?;
        }
        Ok(())
    }

    /// The arrays to evaluate at a step boundary (mlx-lm's
    /// `mx.eval([c.state for c in cache])`).
    pub fn state(&self) -> Vec<&Array> {
        self.pools
            .iter()
            .flatten()
            .flat_map(|(k, v)| [k, v])
            .collect()
    }

    /// Bytes currently backing the pools (0 until first write).
    pub fn allocated_bytes(&self) -> u64 {
        self.pools
            .iter()
            .flatten()
            .flat_map(|(k, v)| [k, v])
            .map(|a| (a.size() * a.dtype().map_or(0, |d| d.size())) as u64)
            .sum()
    }

    /// Bytes one block occupies across all layers (K and V), once the
    /// dtype is known.
    pub fn bytes_per_block(&self) -> u64 {
        let element = self
            .pools
            .iter()
            .flatten()
            .next()
            .map_or(0, |(k, _)| k.dtype().map_or(0, |d| d.size()));
        self.spec.bytes_per_block(element)
    }

    /// Drops all pool storage (engine fault recovery); pools reallocate
    /// lazily on the next write.
    pub fn reset(&mut self) {
        for slot in &mut self.pools {
            *slot = None;
        }
    }
}
