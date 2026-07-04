//! MLX memory introspection — feeds the worker `MemoryReport` (SPEC §2.3)
//! and the decode loop's periodic cache trim.

#![allow(unsafe_code)]

use crate::error::{MlxError, check};
use crate::sys;

type Result<T> = std::result::Result<T, MlxError>;

/// Bytes actively used by MLX allocations.
pub fn active_memory() -> Result<usize> {
    let mut out = 0_usize;
    // SAFETY: out pointer valid for the call; checked status.
    check(unsafe { sys::mlx_get_active_memory(&mut out) })?;
    Ok(out)
}

/// Bytes held in MLX's buffer cache (reusable, not leaked).
pub fn cache_memory() -> Result<usize> {
    let mut out = 0_usize;
    // SAFETY: as above.
    check(unsafe { sys::mlx_get_cache_memory(&mut out) })?;
    Ok(out)
}

/// Peak bytes since process start (or the last reset).
pub fn peak_memory() -> Result<usize> {
    let mut out = 0_usize;
    // SAFETY: as above.
    check(unsafe { sys::mlx_get_peak_memory(&mut out) })?;
    Ok(out)
}

/// Releases MLX's buffer cache back to the OS.
pub fn clear_cache() -> Result<()> {
    // SAFETY: no arguments; checked status.
    check(unsafe { sys::mlx_clear_cache() })
}

/// Whether a Metal device is available (tests auto-skip when not).
pub fn metal_is_available() -> bool {
    let mut out = false;
    // SAFETY: out pointer valid for the call.
    let status = unsafe { sys::mlx_metal_is_available(&mut out) };
    status == 0 && out
}
