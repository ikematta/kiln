//! PagedKv storage tests (no model): write/gather round-trips across block
//! boundaries, copy-on-write block copies, and pool accounting.
//!
//! Single `#[test]` because the kiln-mlx live-object counter is
//! process-global (see kiln-mlx/tests/wrappers.rs).

#![cfg(feature = "metal")]

use kiln_engine::{BlockManager, CowCopy, KvSpec, PagedKv, WriteRun};
use kiln_mlx::{Array, Dtype, Stream, debug, ops};

const LAYERS: usize = 2;
const HEADS: i32 = 2;
const DIM: i32 = 3;
const BLOCK_SIZE: usize = 4;
const NUM_BLOCKS: usize = 4;

fn stream() -> Stream {
    if kiln_mlx::memory::metal_is_available() {
        Stream::gpu()
    } else {
        Stream::cpu()
    }
}

/// `[1, HEADS, len, DIM]` filled with `base, base+1, ...` per position.
fn step_kv(len: i32, base: f32, s: &Stream) -> Array {
    let n = f64::from(HEADS * len * DIM);
    let a = ops::arange(f64::from(base), f64::from(base) + n, 1.0, Dtype::Float32, s).unwrap();
    ops::reshape(&a, &[1, HEADS, len, DIM], s).unwrap()
}

fn read(a: &Array, s: &Stream) -> Vec<f32> {
    let c = ops::contiguous(a, s).unwrap();
    c.eval().unwrap();
    c.data_f32().unwrap()
}

#[test]
fn paged_kv_behavior() {
    let baseline = debug::live_objects();
    {
        let s = stream();
        let spec = KvSpec {
            layers: LAYERS,
            kv_heads: HEADS,
            head_dim: DIM,
            num_blocks: NUM_BLOCKS,
            block_size: BLOCK_SIZE,
        };
        let mut mgr = BlockManager::new(NUM_BLOCKS, BLOCK_SIZE).unwrap();
        let mut kv = PagedKv::new(spec);

        assert!(kv.state().is_empty());
        assert_eq!(kv.allocated_bytes(), 0);

        // 6 tokens spanning two blocks, then 2 more into the partial tail.
        let b0 = mgr.allocate().unwrap();
        let b1 = mgr.allocate().unwrap();
        let first_runs = [
            WriteRun {
                block: b0,
                row_start: 0,
                src_start: 0,
                len: 4,
            },
            WriteRun {
                block: b1,
                row_start: 0,
                src_start: 4,
                len: 2,
            },
        ];
        let k0 = step_kv(6, 0.0, &s);
        let v0 = step_kv(6, 1000.0, &s);
        for layer in 0..LAYERS {
            kv.write(layer, &first_runs, &k0, &v0, &s).unwrap();
        }
        let (gk, gv) = kv.gather(0, &[b0, b1], 6, &s).unwrap();
        assert_eq!(gk.shape(), vec![1, HEADS, 6, DIM]);
        assert_eq!(read(&gk, &s), read(&k0, &s), "gathered K != written K");
        assert_eq!(read(&gv, &s), read(&v0, &s), "gathered V != written V");

        let second_runs = [WriteRun {
            block: b1,
            row_start: 2,
            src_start: 0,
            len: 2,
        }];
        let k1 = step_kv(2, 500.0, &s);
        let v1 = step_kv(2, 1500.0, &s);
        for layer in 0..LAYERS {
            kv.write(layer, &second_runs, &k1, &v1, &s).unwrap();
        }
        let (gk, _) = kv.gather(1, &[b0, b1], 8, &s).unwrap();
        let got = read(&gk, &s);
        let (old, new) = (read(&k0, &s), read(&k1, &s));
        // Per head: 6 old positions then 2 new ones.
        for h in 0..HEADS as usize {
            let head = &got[h * 8 * DIM as usize..(h + 1) * 8 * DIM as usize];
            assert_eq!(
                &head[..6 * DIM as usize],
                &old[h * 6 * DIM as usize..(h + 1) * 6 * DIM as usize],
                "append disturbed earlier tokens (head {h})"
            );
            assert_eq!(
                &head[6 * DIM as usize..],
                &new[h * 2 * DIM as usize..(h + 1) * 2 * DIM as usize],
                "appended tokens misplaced (head {h})"
            );
        }

        // COW copy: block b2 becomes a byte-identical clone of b0, and
        // writing to the clone leaves the original untouched.
        let b2 = mgr.allocate().unwrap();
        kv.copy_block(CowCopy { src: b0, dst: b2 }, &s).unwrap();
        let (orig, _) = kv.gather(0, &[b0], 4, &s).unwrap();
        let (copy, _) = kv.gather(0, &[b2], 4, &s).unwrap();
        assert_eq!(read(&orig, &s), read(&copy, &s), "COW copy differs");
        let overwrite = [WriteRun {
            block: b2,
            row_start: 0,
            src_start: 0,
            len: 4,
        }];
        let k2 = step_kv(4, 9000.0, &s);
        kv.write(0, &overwrite, &k2, &k2, &s).unwrap();
        let (orig_after, _) = kv.gather(0, &[b0], 4, &s).unwrap();
        assert_eq!(
            read(&orig, &s),
            read(&orig_after, &s),
            "write-after-COW mutated the shared source block"
        );

        // Accounting: f32 pools, LAYERS x (K+V) x [NUM_BLOCKS, H, BS, D].
        let expected = (LAYERS * 2 * NUM_BLOCKS * BLOCK_SIZE) as u64
            * HEADS as u64
            * DIM as u64
            * Dtype::Float32.size() as u64;
        assert_eq!(kv.allocated_bytes(), expected);
        assert_eq!(
            kv.bytes_per_block() * NUM_BLOCKS as u64,
            expected,
            "bytes_per_block inconsistent with allocated_bytes"
        );
        assert_eq!(kv.state().len(), LAYERS * 2);

        // Errors: unwritten layer, over-long gather, foreign block id.
        let fresh = PagedKv::new(spec);
        assert!(fresh.gather(0, &[b0], 1, &s).is_err());
        assert!(kv.gather(0, &[b0], 5, &s).is_err());
        assert!(kv.write(LAYERS, &overwrite, &k2, &k2, &s).is_err());
    }
    assert_eq!(debug::live_objects(), baseline, "paged kv leaked handles");
}
