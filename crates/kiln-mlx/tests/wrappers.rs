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
