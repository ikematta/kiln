//! Kernel-vs-gather BIT-exactness probe (SPEC §7.4 Phase 7).
//!
//! The paged-attention kernel is a port of the exact `sdpa_vector` kernels
//! the gather path runs, changed only in addressing — so on identical pool
//! state and queries its output must be BIT-identical to gather + fused
//! SDPA. That claim has one untestable-from-source residual: the builtin
//! kernels are compiled offline into MLX's metallib while custom kernels
//! JIT-compile at runtime, and two compiler builds may e.g. contract
//! fused-multiply-adds differently. This probe measures exactly that, on
//! this device, at every dispatch variant the reference can take: a
//! context-length grid straddling the 1-pass/2-pass boundaries, the
//! fixture models' GQA geometries, both 16-bit activation dtypes,
//! outlier-heavy values, ragged block tails, and a permuted
//! (non-identity) block table.
//!
//! A failure here is NOT a broken test: it means the kernel is a new
//! kernel class on this device (ADR 0002) and the parity bar drops to
//! token-id equality — characterize the failing (dtype, geometry, N),
//! record it in PROGRESS.md under DECISION NEEDED, and stop. Do not relax
//! the assertion.
//!
//! Same-device scope per ADR 0004: on the fixture-generating device class
//! this is a hard gate; on foreign devices (CI) it still runs — both paths
//! execute on the SAME device, so bit-equality is a device-independent
//! invariant of the port, not a cross-device fixture comparison.
//!
//! Single `#[test]`: the kiln-mlx live-object counter is process-global.

#![cfg(feature = "metal")]

use kiln_engine::{BlockManager, KvSpec, PagedAttnInputs, PagedKv, WriteRun};
use kiln_mlx::fast::{self, SdpaMask};
use kiln_mlx::{Array, Dtype, Stream, debug, ops};

const BLOCK_SIZE: usize = 32;
/// 8192 tokens of history + 1 decode slot.
const MAX_TOKENS: usize = 8193;

/// Fixture-model attention geometries (q heads, kv heads, head dim).
const GEOMETRIES: [(i32, i32, i32); 3] = [
    (32, 8, 64),  // llama-3.2-1b
    (16, 8, 128), // qwen3-0.6b
    (4, 1, 256),  // gemma-3-1b (MHA-free sanity: GQA 4 over 1 kv head)
];

/// Context lengths (INCLUDING the current token) straddling every dispatch
/// boundary of the pinned reference: the 'd'/'s' 2-pass threshold (1024),
/// the GQA 2-pass threshold (4096), block-size raggedness, and the 8k
/// acceptance point.
const CONTEXT_GRID: [i32; 14] = [
    1, 2, 31, 32, 33, 255, 1023, 1024, 1025, 2048, 4095, 4096, 4097, 8193,
];

/// Deterministic value source (xorshift64*): uniform-ish in [-1, 1), with
/// every 97th value scaled x64 — the outlier-heavy pattern ADR 0002's
/// addendum found necessary to expose real kernel-class divergences that
/// Gaussian data false-negatived.
struct Values {
    state: u64,
    emitted: usize,
}

impl Values {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            emitted: 0,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        let unit = ((bits >> 40) as f32) / f32::from_bits(0x4B800000); // 2^24
        let value = 2.0 * unit - 1.0;
        self.emitted += 1;
        if self.emitted.is_multiple_of(97) {
            value * 64.0
        } else {
            value
        }
    }

    fn array(&mut self, shape: &[i32], dtype: Dtype, s: &Stream) -> Array {
        let n: i64 = shape.iter().map(|&d| i64::from(d)).product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
        let f32s = Array::from_f32_slice(&data, shape).unwrap();
        ops::astype(&f32s, dtype, s).unwrap()
    }
}

fn raw_bytes(a: &Array, s: &Stream) -> Vec<u8> {
    let c = ops::contiguous(a, s).unwrap();
    c.eval().unwrap();
    c.data_raw_bytes().unwrap()
}

#[test]
fn kernel_output_is_bit_identical_to_gather_sdpa() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: paged-attention kernels need Metal");
        return;
    }
    let baseline = debug::live_objects();
    {
        let s = Stream::gpu();
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            for (h, hk, d) in GEOMETRIES {
                probe_geometry(h, hk, d, dtype, &s);
            }
        }
    }
    assert_eq!(debug::live_objects(), baseline, "parity probe leaked");
}

fn probe_geometry(h: i32, hk: i32, d: i32, dtype: Dtype, s: &Stream) {
    let num_blocks = MAX_TOKENS.div_ceil(BLOCK_SIZE);
    let spec = KvSpec {
        layers: 1,
        kv_heads: hk,
        head_dim: d,
        num_blocks,
        block_size: BLOCK_SIZE,
    };
    let mut mgr = BlockManager::new(num_blocks, BLOCK_SIZE).unwrap();
    let mut kv = PagedKv::new(spec);
    kv.enable_attention_kernel().unwrap();

    // Allocate every block, then use them in a rotated order so the block
    // table is NOT the identity mapping — the kernel's indirection must
    // not depend on physical adjacency.
    let mut blocks: Vec<_> = (0..num_blocks).map(|_| mgr.allocate().unwrap()).collect();
    blocks.rotate_left(3);

    // One functional write per block (realistic pool history: many chained
    // slice_updates), values from the deterministic outlier-heavy source.
    let mut values = Values::new(0x5EED ^ ((h as u64) << 32 | (d as u64) << 8 | dtype as u64));
    for (i, block) in blocks.iter().enumerate() {
        let len = (MAX_TOKENS - i * BLOCK_SIZE).min(BLOCK_SIZE) as i32;
        let run = [WriteRun {
            block: *block,
            row_start: 0,
            src_start: 0,
            len,
        }];
        let keys = values.array(&[1, hk, len, d], dtype, s);
        let vals = values.array(&[1, hk, len, d], dtype, s);
        kv.write(0, &run, &keys, &vals, s).unwrap();
    }
    // Settle the pools once so per-N comparisons don't re-run the write
    // chain (identical either way; this is a test-time optimization).
    kiln_mlx::eval(&kv.state()).unwrap();

    let scale = (f64::from(d).powf(-0.5)) as f32;
    let scale_arr = Array::from_f32(scale);

    for n in CONTEXT_GRID {
        let q = values.array(&[1, h, 1, d], dtype, s);
        let n_blocks = (n as usize).div_ceil(BLOCK_SIZE);
        let table = &blocks[..n_blocks];

        // Reference: the gather path exactly as Attention::forward_step
        // runs it for a decode step (mask None at qL = 1).
        let (k, v) = kv.gather(0, table, n, s).unwrap();
        let reference =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, SdpaMask::None, s).unwrap();

        // Kernel path.
        let inputs = PagedAttnInputs::build(table.iter().map(|b| b.index() as u32), n).unwrap();
        let kernel = kv.paged_sdpa(0, &q, &inputs, &scale_arr, s).unwrap();

        assert_eq!(kernel.shape(), reference.shape(), "shape mismatch");
        assert_eq!(
            raw_bytes(&kernel, s),
            raw_bytes(&reference, s),
            "BIT DIVERGENCE kernel-vs-gather at dtype={dtype:?} heads={h}/{hk} \
             head_dim={d} context={n}: the custom kernel is not in the \
             reference kernel class on this device — see the module docs \
             (do NOT weaken this assert; characterize and stop)"
        );
    }
}
