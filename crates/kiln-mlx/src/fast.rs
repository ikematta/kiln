//! Safe wrappers over `mlx::fast` fused kernels (SPEC §7.1: attention goes
//! through `mlx_fast_scaled_dot_product_attention`).

#![allow(unsafe_code)]

use crate::array::Array;
use crate::error::{MlxError, check};
use crate::stream::Stream;
use crate::sys;

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
