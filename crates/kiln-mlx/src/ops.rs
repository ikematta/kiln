//! Safe wrappers over the mlx-c ops needed by the Llama path and the sampler.
//!
//! All ops are lazy (they build graph nodes); nothing computes until
//! [`crate::eval`]/[`crate::async_eval`]. Every op takes the [`Stream`] it is
//! issued on — always the engine thread's stream.
//!
//! Only the ops Kiln actually uses are wrapped; extend as models require.

#![allow(unsafe_code)]

use crate::array::{Array, Dtype, VectorArray};
use crate::error::{MlxError, check};
use crate::stream::Stream;
use crate::sys;

type Result<T> = std::result::Result<T, MlxError>;

fn opt_int(value: Option<i32>) -> sys::mlx_optional_int {
    sys::mlx_optional_int {
        value: value.unwrap_or_default(),
        has_value: value.is_some(),
    }
}

/// One output-handle + checked-call round trip. SAFETY (applies to every op
/// below): all `Array`/`Stream` arguments hold live handles for the duration
/// of the call, output handles come from `Array::new_handle`, and non-zero
/// statuses are surfaced as `MlxError` by `check` (the array drops cleanly).
macro_rules! unary_op {
    ($(#[$doc:meta])* $name:ident, $sys:ident) => {
        $(#[$doc])*
        pub fn $name(a: &Array, s: &Stream) -> Result<Array> {
            let mut out = Array::new_handle();
            check(unsafe { sys::$sys(out.raw_out(), a.raw(), s.raw()) })?;
            Ok(out)
        }
    };
}

macro_rules! binary_op {
    ($(#[$doc:meta])* $name:ident, $sys:ident) => {
        $(#[$doc])*
        pub fn $name(a: &Array, b: &Array, s: &Stream) -> Result<Array> {
            let mut out = Array::new_handle();
            check(unsafe { sys::$sys(out.raw_out(), a.raw(), b.raw(), s.raw()) })?;
            Ok(out)
        }
    };
}

unary_op!(
    /// Elementwise logistic sigmoid.
    sigmoid, mlx_sigmoid
);
unary_op!(
    /// Elementwise e^x.
    exp, mlx_exp
);
unary_op!(
    /// Elementwise -x.
    negative, mlx_negative
);

binary_op!(
    /// Elementwise a + b (with broadcasting).
    add, mlx_add
);
binary_op!(
    /// Elementwise a - b (with broadcasting).
    subtract, mlx_subtract
);
binary_op!(
    /// Elementwise a * b (with broadcasting).
    multiply, mlx_multiply
);
binary_op!(
    /// Elementwise a / b (with broadcasting).
    divide, mlx_divide
);
binary_op!(
    /// Elementwise a ^ b (with broadcasting).
    power, mlx_power
);
binary_op!(
    /// Elementwise a > b, boolean result.
    greater, mlx_greater
);
binary_op!(
    /// Elementwise a < b, boolean result.
    less, mlx_less
);
binary_op!(
    /// Elementwise logical AND, boolean result.
    logical_and, mlx_logical_and
);
binary_op!(
    /// Dense matrix multiplication.
    matmul, mlx_matmul
);

/// Elementwise clamp to `[a_min, a_max]` (both bounds required — mlx-c
/// accepts null bounds, but every Kiln call site clips on both sides).
pub fn clip(a: &Array, a_min: &Array, a_max: &Array, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; checked status (see the macro safety note).
    check(unsafe { sys::mlx_clip(out.raw_out(), a.raw(), a_min.raw(), a_max.raw(), s.raw()) })?;
    Ok(out)
}

/// `condition ? x : y` elementwise.
pub fn where_cond(condition: &Array, x: &Array, y: &Array, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_where(out.raw_out(), condition.raw(), x.raw(), y.raw(), s.raw()) })?;
    Ok(out)
}

/// Half-open range `[start, stop)` with step, in `dtype`.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_arange(out.raw_out(), start, stop, step, dtype.to_sys(), s.raw()) })?;
    Ok(out)
}

/// Zero-filled array of `shape`.
pub fn zeros(shape: &[i32], dtype: Dtype, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_zeros(
            out.raw_out(),
            shape.as_ptr(),
            shape.len(),
            dtype.to_sys(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Reshape (element count must be preserved; `-1` infers one dimension).
pub fn reshape(a: &Array, shape: &[i32], s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_reshape(out.raw_out(), a.raw(), shape.as_ptr(), shape.len(), s.raw())
    })?;
    Ok(out)
}

/// Permute dimensions by `axes`.
pub fn transpose(a: &Array, axes: &[i32], s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_transpose_axes(out.raw_out(), a.raw(), axes.as_ptr(), axes.len(), s.raw())
    })?;
    Ok(out)
}

/// Concatenate along `axis`.
pub fn concatenate(arrays: &[&Array], axis: i32, s: &Stream) -> Result<Array> {
    let vec = VectorArray::from_arrays(arrays)?;
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_concatenate_axis(out.raw_out(), vec.raw(), axis, s.raw()) })?;
    Ok(out)
}

/// Basic slice `[start, stop)` per dimension, all strides 1.
pub fn slice(a: &Array, start: &[i32], stop: &[i32], s: &Stream) -> Result<Array> {
    debug_assert_eq!(start.len(), stop.len());
    let strides = vec![1_i32; start.len()];
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_slice(
            out.raw_out(),
            a.raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Functional slice assignment: returns `src` with `[start, stop)` replaced
/// by `update` (strides 1) — the KV-cache write.
pub fn slice_update(
    src: &Array,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    s: &Stream,
) -> Result<Array> {
    debug_assert_eq!(start.len(), stop.len());
    let strides = vec![1_i32; start.len()];
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_slice_update(
            out.raw_out(),
            src.raw(),
            update.raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Cast to `dtype`.
pub fn astype(a: &Array, dtype: Dtype, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_astype(out.raw_out(), a.raw(), dtype.to_sys(), s.raw()) })?;
    Ok(out)
}

/// Row-contiguous copy of `a` (no-op graph node when already contiguous) —
/// the escape hatch for `Array::data_*` on strided views.
pub fn contiguous(a: &Array, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_contiguous(out.raw_out(), a.raw(), false, s.raw()) })?;
    Ok(out)
}

/// Gather rows/elements of `a` at `indices` along `axis` (embedding lookup).
pub fn take(a: &Array, indices: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_take_axis(out.raw_out(), a.raw(), indices.raw(), axis, s.raw()) })?;
    Ok(out)
}

/// Gather along `axis` with positional `indices` (same shape semantics as
/// `mx.take_along_axis`).
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_take_along_axis(out.raw_out(), a.raw(), indices.raw(), axis, s.raw())
    })?;
    Ok(out)
}

/// Scatter `values` into `a` at `indices` along `axis` (functional).
pub fn put_along_axis(
    a: &Array,
    indices: &Array,
    values: &Array,
    axis: i32,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_put_along_axis(
            out.raw_out(),
            a.raw(),
            indices.raw(),
            values.raw(),
            axis,
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Index of the maximum along `axis` — greedy sampling.
pub fn argmax(a: &Array, axis: i32, keepdims: bool, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_argmax_axis(out.raw_out(), a.raw(), axis, keepdims, s.raw()) })?;
    Ok(out)
}

/// Maximum along `axis`.
pub fn max(a: &Array, axis: i32, keepdims: bool, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_max_axis(out.raw_out(), a.raw(), axis, keepdims, s.raw()) })?;
    Ok(out)
}

/// Partition indices along `axis`: indices of elements `<=` the `kth`
/// element come first (order within partitions undefined).
pub fn argpartition(a: &Array, kth: i32, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_argpartition_axis(out.raw_out(), a.raw(), kth, axis, s.raw()) })?;
    Ok(out)
}

/// Ascending sort indices along `axis`.
pub fn argsort(a: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_argsort_axis(out.raw_out(), a.raw(), axis, s.raw()) })?;
    Ok(out)
}

/// Ascending sort along `axis`.
pub fn sort(a: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_sort_axis(out.raw_out(), a.raw(), axis, s.raw()) })?;
    Ok(out)
}

/// Top-`k` values along `axis` (unordered within the k).
pub fn topk(a: &Array, k: i32, axis: i32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_topk_axis(out.raw_out(), a.raw(), k, axis, s.raw()) })?;
    Ok(out)
}

/// Cumulative sum along `axis`.
pub fn cumsum(a: &Array, axis: i32, reverse: bool, inclusive: bool, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_cumsum(out.raw_out(), a.raw(), axis, reverse, inclusive, s.raw()) })?;
    Ok(out)
}

/// Softmax along `axis`.
pub fn softmax(a: &Array, axis: i32, precise: bool, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_softmax_axis(out.raw_out(), a.raw(), axis, precise, s.raw()) })?;
    Ok(out)
}

/// log(sum(exp(a))) over all axes, matching `mx.logsumexp(a, keepdims=...)`.
pub fn logsumexp(a: &Array, keepdims: bool, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe { sys::mlx_logsumexp(out.raw_out(), a.raw(), keepdims, s.raw()) })?;
    Ok(out)
}

/// Affine-quantized matmul (mlx-lm weight format, SPEC §7.3):
/// `x @ dequant(w, scales, biases)^T?`.
#[allow(clippy::too_many_arguments)] // mirrors the mlx-c parameter list
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_quantized_matmul(
            out.raw_out(),
            x.raw(),
            w.raw(),
            scales.raw(),
            biases.raw(),
            transpose,
            opt_int(Some(group_size)),
            opt_int(Some(bits)),
            c"affine".as_ptr(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Dequantize packed affine-quantized rows (quantized embedding lookup).
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    check(unsafe {
        sys::mlx_dequantize(
            out.raw_out(),
            w.raw(),
            scales.raw(),
            biases.raw(),
            opt_int(Some(group_size)),
            opt_int(Some(bits)),
            c"affine".as_ptr(),
            Array::null_raw(),
            sys::mlx_optional_dtype {
                value: sys::mlx_dtype::MLX_FLOAT16,
                has_value: false,
            },
            s.raw(),
        )
    })?;
    Ok(out)
}
