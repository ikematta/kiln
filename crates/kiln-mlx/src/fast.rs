//! Safe wrappers over `mlx::fast` fused kernels (SPEC §7.1: attention goes
//! through `mlx_fast_scaled_dot_product_attention`) and custom Metal
//! kernels (`mlx_fast_metal_kernel_new`, SPEC §7.4 Phase 7).

#![allow(unsafe_code)]

use std::ffi::CString;

use crate::array::{Array, Dtype, VectorArray};
use crate::error::{MlxError, check};
use crate::stream::Stream;
use crate::{debug, sys};

type Result<T> = std::result::Result<T, MlxError>;

/// Attention mask modes accepted by MLX's fused SDPA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaMask {
    /// No mask (single-token decode step).
    None,
    /// Causal mask, bottom-right aligned when keys are longer than queries
    /// (prefill over an existing cache).
    Causal,
}

impl SdpaMask {
    fn as_cstr(self) -> &'static std::ffi::CStr {
        match self {
            SdpaMask::None => c"",
            SdpaMask::Causal => c"causal",
        }
    }
}

/// Fused RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm(x: &Array, weight: &Array, eps: f32, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; checked status (see ops.rs safety note).
    check(unsafe { sys::mlx_fast_rms_norm(out.raw_out(), x.raw(), weight.raw(), eps, s.raw()) })?;
    Ok(out)
}

/// Fused RoPE over the first `dims` features with explicit per-pair `freqs`
/// (the mlx-lm `Llama3RoPE` calling convention: `base=None, scale=1.0`).
pub fn rope_with_freqs(
    x: &Array,
    dims: i32,
    traditional: bool,
    offset: i32,
    freqs: &Array,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; base passed as "no value" mirrors Python's None.
    check(unsafe {
        sys::mlx_fast_rope(
            out.raw_out(),
            x.raw(),
            dims,
            traditional,
            sys::mlx_optional_float {
                value: 0.0,
                has_value: false,
            },
            1.0,
            offset,
            freqs.raw(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Fused RoPE with an implicit `base`-derived frequency ladder (plain
/// `nn.RoPE`, used when a model has no `rope_scaling`).
pub fn rope_with_base(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; freqs passed as mlx-c's documented null array.
    check(unsafe {
        sys::mlx_fast_rope(
            out.raw_out(),
            x.raw(),
            dims,
            traditional,
            sys::mlx_optional_float {
                value: base,
                has_value: true,
            },
            scale,
            offset,
            Array::null_raw(),
            s.raw(),
        )
    })?;
    Ok(out)
}

/// Internal RAII wrapper for `mlx_vector_string` (custom-kernel name lists).
struct VectorString {
    raw: sys::mlx_vector_string,
}

impl VectorString {
    fn from_strs(strs: &[&str]) -> Result<Self> {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        let vec = Self {
            raw: unsafe { sys::mlx_vector_string_new() },
        };
        for s in strs {
            let c = cstring(s)?;
            // SAFETY: vector handle is live; append copies the C string.
            check(unsafe { sys::mlx_vector_string_append_value(vec.raw, c.as_ptr()) })?;
        }
        Ok(vec)
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        // SAFETY: exactly one free per `*_new`.
        let _ = unsafe { sys::mlx_vector_string_free(self.raw) };
        debug::track_free();
    }
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| MlxError {
        message: format!("string {s:?} contains an interior NUL"),
    })
}

/// One output an invocation produces: shape + dtype (custom kernels write
/// into preallocated buffers; MLX allocates them from this description).
#[derive(Debug, Clone)]
pub struct KernelOutput {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
}

/// Per-invocation parameters for [`MetalKernel::apply`].
///
/// `grid` counts THREADS (mlx custom kernels use `dispatch_threads`, not
/// threadgroup counts — MLX's builtin kernels' `dispatch_threadgroups(g, t)`
/// translates to `grid = (g.x*t.x, g.y*t.y, g.z*t.z)` here). Template
/// arguments are appended dtype-first then ints; the generated `template
/// <...>` header declares them in exactly that order.
#[derive(Debug, Clone)]
pub struct KernelInvocation<'a> {
    pub template_dtypes: &'a [(&'a str, Dtype)],
    pub template_ints: &'a [(&'a str, i32)],
    pub grid: (i32, i32, i32),
    pub threadgroup: (i32, i32, i32),
    pub outputs: &'a [KernelOutput],
}

/// Internal RAII wrapper for one `mlx_fast_metal_kernel_config`.
struct KernelConfig {
    raw: sys::mlx_fast_metal_kernel_config,
}

impl KernelConfig {
    fn build(call: &KernelInvocation) -> Result<Self> {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        let config = Self {
            raw: unsafe { sys::mlx_fast_metal_kernel_config_new() },
        };
        for out in call.outputs {
            // SAFETY: live handle; shape pointer valid for the call.
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config.raw,
                    out.shape.as_ptr(),
                    out.shape.len(),
                    out.dtype.to_sys(),
                )
            })?;
        }
        let (gx, gy, gz) = call.grid;
        let (tx, ty, tz) = call.threadgroup;
        // SAFETY: live handle; plain setters.
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_grid(config.raw, gx, gy, gz) })?;
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_set_thread_group(config.raw, tx, ty, tz)
        })?;
        for (name, dtype) in call.template_dtypes {
            let c = cstring(name)?;
            // SAFETY: live handle; name valid for the call (copied by mlx-c).
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                    config.raw,
                    c.as_ptr(),
                    dtype.to_sys(),
                )
            })?;
        }
        for (name, value) in call.template_ints {
            let c = cstring(name)?;
            // SAFETY: live handle; name valid for the call (copied by mlx-c).
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config.raw,
                    c.as_ptr(),
                    *value,
                )
            })?;
        }
        Ok(config)
    }
}

impl Drop for KernelConfig {
    fn drop(&mut self) {
        // SAFETY: exactly one free per `*_new`.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.raw) };
        debug::track_free();
    }
}

/// A custom Metal kernel (`mlx.fast.metal_kernel` analogue). The handle is
/// a reusable graph-builder: `apply` adds a lazy node like any other op.
/// MLX JIT-compiles one pipeline per distinct template instantiation on
/// first evaluation and caches it for the process lifetime.
///
/// Kiln policy: `ensure_row_contiguous` is always OFF — callers pass
/// explicit geometry and must feed row-contiguous arrays (MLX would
/// otherwise silently insert full copies of pool-sized inputs), and
/// `atomic_outputs` is always OFF (atomics have non-deterministic
/// accumulation order, which the determinism bars forbid).
#[derive(Debug)]
pub struct MetalKernel {
    raw: sys::mlx_fast_metal_kernel,
    n_inputs: usize,
    n_outputs: usize,
}

impl MetalKernel {
    /// Declares a kernel whose BODY is `source` (the signature is generated
    /// by MLX from `input_names`/`output_names` and the per-call template
    /// args). `header` is pasted above the generated signature — helper
    /// functions and includes go there. MLX prepends its kernel `utils.h`
    /// (so `Limits<T>`, `float16_t`/`bfloat16_t`, and `using namespace
    /// metal` are in scope).
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
    ) -> Result<Self> {
        crate::init();
        let c_name = cstring(name)?;
        let c_source = cstring(source)?;
        let c_header = cstring(header)?;
        let inputs = VectorString::from_strs(input_names)?;
        let outputs = VectorString::from_strs(output_names)?;
        debug::track_new();
        // SAFETY: all handles/strings are live for the call; mlx-c copies
        // them. The returned handle is freed in Drop.
        let raw = unsafe {
            sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                inputs.raw,
                outputs.raw,
                c_source.as_ptr(),
                c_header.as_ptr(),
                false, // ensure_row_contiguous: see type-level policy note
                false, // atomic_outputs
            )
        };
        if raw.ctx.is_null() {
            debug::track_free();
            return Err(MlxError {
                message: format!("mlx_fast_metal_kernel_new({name}) returned null"),
            });
        }
        Ok(Self {
            raw,
            n_inputs: input_names.len(),
            n_outputs: output_names.len(),
        })
    }

    /// Adds one lazy invocation to the graph, returning the output arrays
    /// in `output_names` order.
    pub fn apply(
        &self,
        inputs: &[&Array],
        call: &KernelInvocation,
        s: &Stream,
    ) -> Result<Vec<Array>> {
        if inputs.len() != self.n_inputs || call.outputs.len() != self.n_outputs {
            return Err(MlxError {
                message: format!(
                    "kernel apply with {} input(s)/{} output(s), declared {}/{}",
                    inputs.len(),
                    call.outputs.len(),
                    self.n_inputs,
                    self.n_outputs
                ),
            });
        }
        let config = KernelConfig::build(call)?;
        let inputs = VectorArray::from_arrays(inputs)?;
        let mut outputs = VectorArray::new_handle();
        // SAFETY: all handles live; `outputs` is a fresh out-vector the call
        // fills (elements extracted as owning handles below).
        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                outputs.raw_out(),
                self.raw,
                inputs.raw(),
                config.raw,
                s.raw(),
            )
        })?;
        (0..outputs.len()).map(|i| outputs.get(i)).collect()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // SAFETY: exactly one free per `*_new`.
        unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
        debug::track_free();
    }
}

/// Fused scaled-dot-product attention over `[B, heads, L, head_dim]` inputs
/// (GQA handled natively when `keys`/`values` have fewer heads).
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: SdpaMask,
    s: &Stream,
) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; optional mask/sinks arrays passed as documented
    // null arrays.
    check(unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            out.raw_out(),
            queries.raw(),
            keys.raw(),
            values.raw(),
            scale,
            mask.as_cstr().as_ptr(),
            Array::null_raw(),
            Array::null_raw(),
            s.raw(),
        )
    })?;
    Ok(out)
}
