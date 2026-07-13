//! Wrapper-level FFI discipline tests: leak counter returns to baseline,
//! the error handler turns MLX faults into `Err` instead of `exit()`,
//! clone aliasing behaves, and data reads round-trip.
//!
//! Single `#[test]` on purpose: the live-object counter is process-global,
//! so the baseline assertions must not interleave with a concurrently
//! running test that also constructs arrays.

#![cfg(feature = "metal")]

use kiln_mlx::{Array, Dtype, Stream, debug, eval, ops};

fn gpu_or_cpu() -> Stream {
    if kiln_mlx::memory::metal_is_available() {
        Stream::gpu()
    } else {
        Stream::cpu()
    }
}

#[test]
fn wrapper_discipline() {
    add_eval_read_and_leak_baseline();
    mlx_errors_are_results_not_exits();
    clone_aliases_and_frees_independently();
    f16_raw_bytes_round_trip();
    shape_mismatch_is_checked();
    host_reads_reject_wrong_dtype();
    host_reads_reject_non_contiguous_views();
    custom_metal_kernel_smoke();
}

/// End-to-end custom-kernel check (SPEC §7.4 machinery): a templated
/// gather-by-index kernel JIT-compiles, respects template args and thread
/// geometry, produces correct values, and leaks nothing. Metal-only: the
/// primitive has no CPU implementation.
fn custom_metal_kernel_smoke() {
    if !kiln_mlx::memory::metal_is_available() {
        return;
    }
    use kiln_mlx::fast::{KernelInvocation, KernelOutput, MetalKernel};
    let baseline = debug::live_objects();
    {
        let s = Stream::gpu();
        let kernel = MetalKernel::new(
            "kiln_wrapper_smoke",
            &["src", "idx"],
            &["out"],
            // One thread per (row, column): out[r, c] = src[idx[r], c] * W.
            "  uint r = thread_position_in_grid.y;\n\
             uint c = thread_position_in_grid.x;\n\
             out[r * D + c] = src[idx[r] * D + c] * static_cast<T>(W);\n",
            "",
        )
        .unwrap();
        let src = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let idx = Array::from_u32_slice(&[2, 0, 2, 1, 1, 0, 2, 0], &[8]).unwrap();
        let outputs = kernel
            .apply(
                &[&src, &idx],
                &KernelInvocation {
                    template_dtypes: &[("T", Dtype::Float32)],
                    template_ints: &[("D", 2), ("W", 3)],
                    grid: (2, 8, 1),
                    threadgroup: (2, 8, 1),
                    outputs: &[KernelOutput {
                        shape: vec![8, 2],
                        dtype: Dtype::Float32,
                    }],
                },
                &s,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let got = outputs[0].data_f32().unwrap();
        let want: Vec<f32> = [2, 0, 2, 1, 1, 0, 2, 0]
            .iter()
            .flat_map(|&r: &usize| [src_row(r).0 * 3.0, src_row(r).1 * 3.0])
            .collect();
        assert_eq!(got, want);
    }
    assert_eq!(debug::live_objects(), baseline);
}

fn src_row(r: usize) -> (f32, f32) {
    let flat = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    (flat[r * 2], flat[r * 2 + 1])
}

/// MLX's item<T>/data<T> are raw reinterpreting accessors; the safe wrappers
/// must refuse a mismatched T instead of reading the buffer at the wrong
/// element width (OOB when T is wider — e.g. u32 reads of an f16 buffer).
fn host_reads_reject_wrong_dtype() {
    let baseline = debug::live_objects();
    {
        let floats = Array::from_f32_slice(&[1.0, 2.0], &[2]).unwrap();
        let err = floats.data_u32().unwrap_err();
        assert!(err.message.contains("data_u32"), "got: {}", err.message);

        let scalar_f16 = Array::from_raw_bytes(&[0x00, 0x3C], &[], Dtype::Float16).unwrap();
        assert!(scalar_f16.item_u32().is_err());
        assert!(scalar_f16.item_f32().is_err());

        let ids = Array::from_u32_slice(&[7], &[1]).unwrap();
        assert!(ids.data_f32().is_err());
        assert_eq!(ids.data_u32().unwrap(), vec![7]);
    }
    assert_eq!(debug::live_objects(), baseline);
}

/// data<T> returns the raw base pointer, so strided views read as size()
/// consecutive elements would be wrong/out-of-bounds; the wrapper must
/// refuse them and the ops::contiguous escape hatch must fix them up.
fn host_reads_reject_non_contiguous_views() {
    let baseline = debug::live_objects();
    {
        let s = gpu_or_cpu();
        let a = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t = ops::transpose(&a, &[1, 0], &s).unwrap(); // [3, 2], strided
        t.eval().unwrap();
        let err = t.data_f32().unwrap_err();
        assert!(
            err.message.contains("non-contiguous"),
            "got: {}",
            err.message
        );

        let fixed = ops::contiguous(&t, &s).unwrap();
        assert_eq!(
            fixed.data_f32().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );

        // Interior column slice: strided view over a larger parent buffer.
        let col = ops::slice(&a, &[0, 1], &[2, 2], &s).unwrap(); // [2, 1]
        col.eval().unwrap();
        assert!(col.data_f32().is_err());
        let col = ops::contiguous(&col, &s).unwrap();
        assert_eq!(col.data_f32().unwrap(), vec![2.0, 5.0]);
    }
    assert_eq!(debug::live_objects(), baseline);
}

fn add_eval_read_and_leak_baseline() {
    let baseline = debug::live_objects();
    {
        let s = gpu_or_cpu();
        let a = Array::from_f32_slice(&[1.0, 2.0], &[2]).unwrap();
        let b = Array::from_f32_slice(&[10.0, 20.0], &[2]).unwrap();
        let sum = ops::add(&a, &b, &s).unwrap();
        eval(&[&sum]).unwrap();
        assert_eq!(sum.data_f32().unwrap(), vec![11.0, 22.0]);
        assert_eq!(sum.shape(), vec![2]);
        assert_eq!(sum.dtype(), Some(Dtype::Float32));
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "wrapper handles leaked (live-object counter did not return to baseline)"
    );
}

fn mlx_errors_are_results_not_exits() {
    let baseline = debug::live_objects();
    {
        let s = gpu_or_cpu();
        let a = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        // 3 elements cannot reshape to [2, 2]: must surface as Err via the
        // installed handler — with the mlx-c default handler this would
        // exit() the test process.
        let err = ops::reshape(&a, &[2, 2], &s).unwrap_err();
        assert!(
            err.message.to_lowercase().contains("reshape"),
            "unexpected message: {}",
            err.message
        );
        // The stream still works after a recorded error.
        let ok = ops::add(&a, &a, &s).unwrap();
        assert_eq!(ok.data_f32().unwrap(), vec![2.0, 4.0, 6.0]);
    }
    assert_eq!(debug::live_objects(), baseline);
}

fn clone_aliases_and_frees_independently() {
    let baseline = debug::live_objects();
    {
        let a = Array::from_u32_slice(&[7, 8, 9], &[3]).unwrap();
        let b = a.clone();
        drop(a);
        assert_eq!(b.data_u32().unwrap(), vec![7, 8, 9]);
    }
    assert_eq!(debug::live_objects(), baseline);
}

fn f16_raw_bytes_round_trip() {
    let baseline = debug::live_objects();
    {
        let s = gpu_or_cpu();
        // f16 bit patterns: 1.0 = 0x3C00, -2.0 = 0xC000 (little-endian).
        let bytes = [0x00_u8, 0x3C, 0x00, 0xC0];
        let a = Array::from_raw_bytes(&bytes, &[2], Dtype::Float16).unwrap();
        let f = ops::astype(&a, Dtype::Float32, &s).unwrap();
        assert_eq!(f.data_f32().unwrap(), vec![1.0, -2.0]);

        // Length mismatch is a checked error.
        assert!(Array::from_raw_bytes(&bytes, &[3], Dtype::Float16).is_err());
    }
    assert_eq!(debug::live_objects(), baseline);
}

fn shape_mismatch_is_checked() {
    assert!(Array::from_f32_slice(&[1.0, 2.0], &[3]).is_err());
}
