//! Block-table-aware paged attention (SPEC §7.4 Phase 7): custom Metal
//! kernels that read K/V straight out of the paged pools, eliminating the
//! per-step gather copy of the v0 path.
//!
//! # Parity design (read before touching ANY line of the kernel sources)
//!
//! These kernels are ports of the EXACT kernels the gather path executes at
//! the pin (MLX v0.31.1, `mlx/backend/metal/kernels/sdpa_vector.h`:
//! `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`, specialized
//! to the decode case: query length 1, no mask, no sinks, row-contiguous
//! queries). The ONLY change is addressing: token `i` of kv-head `h` reads
//! `pool[tbl[i / BS], h, i % BS, :]` instead of a contiguous gathered view —
//! the same values the gather would copy, loaded in the same iteration
//! order, accumulated with the same algebra in the same dtypes. Reduction
//! order is preserved BY CONSTRUCTION; that is the entire bit-exactness
//! argument (ADR 0002: different reduction orders = different kernel class
//! = ulp-level divergence). Both compile paths are non-fast-math at the pin
//! (builtin metallib: `-fno-fast-math`; custom kernels:
//! `setFastMathEnabled(false)`).
//!
//! The variant DISPATCH is part of the port: the reference routes decode
//! (qL=1) to the 2-pass kernels iff
//! `((devc=='d'||devc=='s') && N>=1024) || (GQA && N>=4096)` where `devc` is
//! the last character of the GPU architecture string, and quantizes the
//! 2-pass `blocks` count by device class and N
//! (`scaled_dot_product_attention.cpp` at the pin — table replicated in
//! [`two_pass_blocks`]). Running a different variant than the reference at
//! any (device, N) is a guaranteed kernel-class boundary, so both the
//! predicate and the table are replicated bit-for-bit.
//!
//! Any edit to the sources below that changes a floating-point operation,
//! its order, or a dispatch boundary invalidates the parity argument and
//! must re-run the full golden suite plus the kernel-vs-gather bit probe.

use kiln_mlx::fast::{KernelInvocation, KernelOutput, MetalKernel};
use kiln_mlx::{Array, Dtype, MlxError, Stream, device};

use crate::paged::KvSpec;

/// Header shared by all three kernels (the reference `sdpa_vector.h`
/// includes `<metal_simdgroup>`; MLX prepends its kernel `utils.h`, which
/// supplies `Limits<>`, the 16-bit dtype typedefs, and
/// `using namespace metal`).
const HEADER: &str = "#include <metal_simdgroup>\n";

/// Port of `sdpa_vector<T, D, V>` (1-pass). One threadgroup per query head:
/// 32 simdgroups stride the keys (`i += BN`), each lane owns `D/32` query
/// features, and the threadgroup combine reduces the 32 partial softmax
/// states — all verbatim from the pin. Template args: `T` element dtype,
/// `D` qk head dim, `V` v head dim, `GQA` heads-per-kv-head, `HK` kv heads,
/// `BS` tokens per block.
///
/// Buffers: `queries` `[1, H, 1, D]` row-contiguous (post-RoPE, exactly the
/// array the gather path hands SDPA), `kpool`/`vpool`
/// `[num_blocks, HK, BS, D|V]`, `tbl` u32 block table (padded to >= 8
/// entries so its address space class never flips the generated source),
/// `nkeys` i32 scalar (history length INCLUDING this step's token),
/// `scale_val` f32 scalar.
const ONE_PASS_SRC: &str = r#"
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int head_idx = int(threadgroup_position_in_grid.x);
  const int kv_head_idx = head_idx / GQA;

  const device T* q_head = queries + head_idx * D + simd_lid * qk_per_thread;
  device T* out_head = out + head_idx * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale_val) * q_head[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // For each key
  for (int i = int(simd_gid); i < nkeys; i += BN) {
    const int64_t row =
        (int64_t(tbl[i / BS]) * HK + kv_head_idx) * BS + (i % BS);
    const device T* keys_i = kpool + row * D + simd_lid * qk_per_thread;
    const device T* values_i = vpool + row * V + simd_lid * v_per_thread;

    // Read the key
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys_i[j];
    }

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values_i[j];
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out_head[i] = static_cast<T>(o[i]);
    }
  }
"#;

/// Port of `sdpa_vector_2pass_1<T, D, V>`. One simdgroup per (query head,
/// key partition): partition `block_idx` covers keys `block_idx, block_idx
/// + BLOCKS, ...` and writes an unnormalized partial plus its softmax
/// running (sum, max). NOTE the reference reads keys straight into the
/// score product here (no thread-local `k[]` staging, unlike the 1-pass
/// kernel) — kept faithfully. Extra template arg: `BLOCKS` (the
/// device-class- and N-quantized partition count, [`two_pass_blocks`]).
const TWO_PASS_1_SRC: &str = r#"
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  const uint simd_lid = thread_index_in_simdgroup;
  const int kv_head_idx = int(threadgroup_position_in_grid.x);
  const int block_idx = int(threadgroup_position_in_grid.z);
  const int q_head_idx =
      GQA * kv_head_idx + int(thread_position_in_threadgroup.y);

  const device T* q_head = queries + q_head_idx * D + simd_lid * qk_per_thread;
  device T* out_part = partials + q_head_idx * BLOCKS * V + block_idx * V +
      simd_lid * v_per_thread;
  device float* sums_out = sums + q_head_idx * BLOCKS + block_idx;
  device float* maxs_out = maxs + q_head_idx * BLOCKS + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale_val) * q_head[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx; i < nkeys; i += BLOCKS) {
    const int64_t row =
        (int64_t(tbl[i / BS]) * HK + kv_head_idx) * BS + (i % BS);
    const device T* keys_i = kpool + row * D + simd_lid * qk_per_thread;
    const device T* values_i = vpool + row * V + simd_lid * v_per_thread;

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * keys_i[j];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values_i[j];
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums_out[0] = sum_exp_score;
    maxs_out[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out_part[i] = static_cast<T>(o[i]);
  }
"#;

/// Port of `sdpa_vector_2pass_2<T, D>` — combines the pass-1 partials. No
/// paging here (partials/sums/maxs are contiguous); the source differs from
/// the pin only in `blocks` being the `BLOCKS` template constant instead of
/// a runtime buffer (compile-time constant either way: the reference bakes
/// it into pass 1 as a function constant). `D` is the VALUE head dim.
const TWO_PASS_2_SRC: &str = r#"
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int head_idx = int(threadgroup_position_in_grid.x);

  const device T* part_head = partials + head_idx * BLOCKS * D +
      simd_gid * D + simd_lid * elem_per_thread;
  const device float* sums_head = sums + head_idx * BLOCKS;
  const device float* maxs_head = maxs + head_idx * BLOCKS;
  device T* out_head = out + head_idx * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < BLOCKS / BN; ++b) {
    max_score = max(max_score, maxs_head[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(maxs_head[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums_head[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(maxs_head[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(part_head[i]);
    }
    maxs_head += BN;
    sums_head += BN;
    part_head += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out_head[i] = static_cast<T>(o[i]);
    }
  }
"#;

/// Per-sequence per-step inputs for the kernel path, prepared once by the
/// engine and shared across every layer's call (28+ layers would otherwise
/// rebuild identical host arrays each step).
#[derive(Debug)]
pub struct PagedAttnInputs {
    /// u32 block ids in token order, zero-padded to >= 8 entries (address-
    /// space-class stability of the generated kernel source; the pad
    /// entries are never dereferenced).
    pub(crate) table: Array,
    /// History length including this step's token (`offset + 1`).
    pub(crate) context_len: i32,
    /// `context_len` as an ndim-0 i32 array (the `nkeys` kernel input).
    pub(crate) context_len_arr: Array,
}

impl PagedAttnInputs {
    /// Minimum block-table entries (see `table` docs).
    const MIN_TABLE_LEN: usize = 8;

    /// `blocks` in token order; `context_len` counts the history INCLUDING
    /// the token whose K/V this step just wrote (`offset + 1`). Public for
    /// the parity test harness; the engine builds these in
    /// `build_seq_step`.
    pub fn build(
        blocks: impl ExactSizeIterator<Item = u32>,
        context_len: i32,
    ) -> Result<Self, MlxError> {
        let mut ids: Vec<u32> = Vec::with_capacity(blocks.len().max(Self::MIN_TABLE_LEN));
        ids.extend(blocks);
        while ids.len() < Self::MIN_TABLE_LEN {
            ids.push(0);
        }
        let len = ids.len() as i32;
        Ok(Self {
            table: Array::from_u32_slice(&ids, &[len])?,
            context_len,
            context_len_arr: Array::from_i32(context_len),
        })
    }
}

/// The three kernel handles plus the device dispatch class. Owned by
/// `PagedKv` when the SPEC §7.4 flag enables the kernel path.
#[derive(Debug)]
pub(crate) struct PagedAttn {
    one_pass: MetalKernel,
    two_pass_1: MetalKernel,
    two_pass_2: MetalKernel,
    /// Last byte of the GPU architecture string — the reference dispatch's
    /// device class key (`Device::get_architecture().back()` at the pin).
    devc: u8,
}

/// The reference's 2-pass routing predicate for decode (qL = 1) shapes
/// (`ScaledDotProductAttention::eval_gpu` at the pin).
fn use_two_pass(devc: u8, n: i32, gqa: i32) -> bool {
    ((devc == b'd' || devc == b's') && n >= 1024) || (gqa > 1 && n >= 4096)
}

/// The reference's 2-pass partition-count table (`sdpa_vector_2pass` at the
/// pin), specialized to qL = 1 (`n_simds = gqa_factor`).
fn two_pass_blocks(devc: u8, n: i32, n_simds: i32) -> i32 {
    if devc == b's' {
        if n > 1024 && n_simds > 4 {
            if n <= 8192 {
                128
            } else if n <= 32768 {
                256
            } else if n <= 65536 {
                512
            } else {
                1024
            }
        } else {
            64
        }
    } else if devc == b'd' {
        if n_simds <= 2 && n > 8192 {
            256
        } else if n_simds >= 6 {
            // Reference: `N >= 16384 && N < 65536`.
            if (16384..65536).contains(&n) {
                512
            } else if n >= 65536 {
                1024
            } else {
                128
            }
        } else {
            128
        }
    } else if n_simds >= 4 {
        64
    } else {
        32
    }
}

impl PagedAttn {
    pub(crate) fn new() -> Result<Self, MlxError> {
        let arch = device::gpu_architecture()?;
        let devc = *arch.as_bytes().last().ok_or_else(|| MlxError {
            message: "empty GPU architecture string".to_owned(),
        })?;
        let inputs = &["queries", "kpool", "vpool", "tbl", "nkeys", "scale_val"];
        Ok(Self {
            one_pass: MetalKernel::new(
                "kiln_paged_sdpa_vector",
                inputs,
                &["out"],
                ONE_PASS_SRC,
                HEADER,
            )?,
            two_pass_1: MetalKernel::new(
                "kiln_paged_sdpa_vector_2pass_1",
                inputs,
                &["partials", "sums", "maxs"],
                TWO_PASS_1_SRC,
                HEADER,
            )?,
            two_pass_2: MetalKernel::new(
                "kiln_paged_sdpa_vector_2pass_2",
                &["partials", "sums", "maxs"],
                &["out"],
                TWO_PASS_2_SRC,
                HEADER,
            )?,
            devc,
        })
    }

    /// Decode-shaped paged SDPA: `q` is `[1, H, 1, D]` (row-contiguous,
    /// post-RoPE), and the sequence's history — including this step's
    /// already-written token — spans `inputs.context_len` tokens of the
    /// pools. Returns `[1, H, 1, D]`, the same shape/dtype the gather+SDPA
    /// path produces.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sdpa(
        &self,
        spec: &KvSpec,
        k_pool: &Array,
        v_pool: &Array,
        q: &Array,
        inputs: &PagedAttnInputs,
        scale: &Array,
        s: &Stream,
    ) -> Result<Array, MlxError> {
        let h = q.dim(1);
        let (hk, d) = (spec.kv_heads, spec.head_dim);
        let n = inputs.context_len;
        let dtype = q.dtype().ok_or_else(|| MlxError {
            message: "paged sdpa on queries with an unsupported dtype".to_owned(),
        })?;
        let gqa = h / hk;
        debug_assert_eq!(q.dim(2), 1, "paged sdpa is decode-only (qL = 1)");
        debug_assert_eq!(gqa * hk, h, "query heads not a multiple of kv heads");
        debug_assert!(
            n as i64 <= inputs.table.size() as i64 * spec.block_size as i64,
            "context of {n} tokens exceeds the block table"
        );

        let template_dtypes = &[("T", dtype)];
        let bs = spec.block_size as i32;
        let kernel_args = |extra: &[(&'static str, i32)]| -> Vec<(&'static str, i32)> {
            let mut args = vec![("D", d), ("V", d), ("GQA", gqa), ("HK", hk), ("BS", bs)];
            args.extend_from_slice(extra);
            args
        };
        let buffers: [&Array; 6] = [
            q,
            k_pool,
            v_pool,
            &inputs.table,
            &inputs.context_len_arr,
            scale,
        ];

        if !use_two_pass(self.devc, n, gqa) {
            let mut outputs = self.one_pass.apply(
                &buffers,
                &KernelInvocation {
                    template_dtypes,
                    template_ints: &kernel_args(&[]),
                    grid: (h * 1024, 1, 1),
                    threadgroup: (1024, 1, 1),
                    outputs: &[KernelOutput {
                        shape: vec![1, h, 1, d],
                        dtype,
                    }],
                },
                s,
            )?;
            return outputs.pop().ok_or_else(|| MlxError {
                message: "paged sdpa kernel returned no output".to_owned(),
            });
        }

        let blocks = two_pass_blocks(self.devc, n, gqa);
        let pass1 = self.two_pass_1.apply(
            &buffers,
            &KernelInvocation {
                template_dtypes,
                template_ints: &kernel_args(&[("BLOCKS", blocks)]),
                grid: (hk * 32, gqa, blocks),
                threadgroup: (32, gqa, 1),
                outputs: &[
                    KernelOutput {
                        shape: vec![1, h, 1, blocks, d],
                        dtype,
                    },
                    KernelOutput {
                        shape: vec![1, h, 1, blocks],
                        dtype: Dtype::Float32,
                    },
                    KernelOutput {
                        shape: vec![1, h, 1, blocks],
                        dtype: Dtype::Float32,
                    },
                ],
            },
            s,
        )?;
        let mut outputs = self.two_pass_2.apply(
            &pass1.iter().collect::<Vec<_>>(),
            &KernelInvocation {
                template_dtypes,
                template_ints: &[("D", d), ("BLOCKS", blocks)],
                grid: (h * 1024, 1, 1),
                threadgroup: (1024, 1, 1),
                outputs: &[KernelOutput {
                    shape: vec![1, h, 1, d],
                    dtype,
                }],
            },
            s,
        )?;
        outputs.pop().ok_or_else(|| MlxError {
            message: "paged sdpa pass-2 kernel returned no output".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{two_pass_blocks, use_two_pass};

    /// The routing predicate, checked against
    /// `ScaledDotProductAttention::eval_gpu` at the pin (qL = 1 decode).
    #[test]
    fn two_pass_predicate_matches_the_pin() {
        // 'd'/'s' devices: threshold 1024 regardless of GQA.
        for devc in [b'd', b's'] {
            assert!(!use_two_pass(devc, 1023, 1));
            assert!(use_two_pass(devc, 1024, 1));
            assert!(use_two_pass(devc, 1024, 4));
        }
        // Everything else: GQA-only threshold at 4096.
        for devc in [b'p', b'g'] {
            assert!(!use_two_pass(devc, 4096, 1), "MHA never routes 2-pass");
            assert!(!use_two_pass(devc, 4095, 4));
            assert!(use_two_pass(devc, 4096, 4));
            assert!(use_two_pass(devc, 4096, 2));
        }
    }

    /// The partition-count table, transcribed case-by-case from
    /// `sdpa_vector_2pass` at the pin (`n_simds = gqa_factor` at qL = 1).
    #[test]
    fn two_pass_blocks_matches_the_pin() {
        // devc == 's': 64 unless (N > 1024 && n_simds > 4).
        assert_eq!(two_pass_blocks(b's', 1024, 8), 64);
        assert_eq!(two_pass_blocks(b's', 2048, 4), 64);
        assert_eq!(two_pass_blocks(b's', 2048, 5), 128);
        assert_eq!(two_pass_blocks(b's', 8192, 8), 128);
        assert_eq!(two_pass_blocks(b's', 8193, 8), 256);
        assert_eq!(two_pass_blocks(b's', 32768, 8), 256);
        assert_eq!(two_pass_blocks(b's', 32769, 8), 512);
        assert_eq!(two_pass_blocks(b's', 65536, 8), 512);
        assert_eq!(two_pass_blocks(b's', 65537, 8), 1024);
        // devc == 'd': 128; 256 for narrow-simd long-N; 512/1024 for wide.
        assert_eq!(two_pass_blocks(b'd', 4096, 4), 128);
        assert_eq!(two_pass_blocks(b'd', 8192, 2), 128);
        assert_eq!(two_pass_blocks(b'd', 8193, 2), 256);
        assert_eq!(two_pass_blocks(b'd', 8193, 3), 128);
        assert_eq!(two_pass_blocks(b'd', 16383, 6), 128);
        assert_eq!(two_pass_blocks(b'd', 16384, 6), 512);
        assert_eq!(two_pass_blocks(b'd', 65535, 6), 512);
        assert_eq!(two_pass_blocks(b'd', 65536, 6), 1024);
        // Every other class: 64 iff n_simds >= 4, else 32.
        assert_eq!(two_pass_blocks(b'p', 8192, 4), 64);
        assert_eq!(two_pass_blocks(b'p', 8192, 2), 32);
        assert_eq!(two_pass_blocks(b'g', 100000, 3), 32);
    }
}
