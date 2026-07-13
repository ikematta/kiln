//! Device introspection: the GPU architecture identifier.
//!
//! The pinned MLX selects Metal kernel variants (and grid quantizations) by
//! the LAST CHARACTER of this string (`mlx::core::metal::Device::
//! get_architecture().back()` — e.g. `"applegpu_g16p"` → `'p'`). Kiln's
//! paged-attention port must replicate that dispatch exactly to stay in the
//! reference kernel class (ADR 0002), so the string is exposed here.

#![allow(unsafe_code)]

use std::ffi::CStr;

use crate::error::{MlxError, check};
use crate::{debug, sys};

/// The Metal GPU architecture string reported by MLX (the same value
/// `Device::get_architecture()` uses for kernel dispatch decisions).
pub fn gpu_architecture() -> Result<String, MlxError> {
    crate::init();
    debug::track_new();
    // SAFETY: constructor with no preconditions; freed below (exactly once
    // on every path).
    let dev = unsafe { sys::mlx_device_new_type(sys::mlx_device_type_::MLX_GPU, 0) };
    let result = gpu_architecture_of(dev);
    // SAFETY: live handle, freed exactly once.
    let _ = unsafe { sys::mlx_device_free(dev) };
    debug::track_free();
    result
}

fn gpu_architecture_of(dev: sys::mlx_device) -> Result<String, MlxError> {
    debug::track_new();
    // SAFETY: constructor with no preconditions; freed below.
    let mut info = unsafe { sys::mlx_device_info_new() };
    let result = (|| {
        // SAFETY: live handles; `info` is an out-parameter the call fills.
        check(unsafe { sys::mlx_device_info_get(&mut info, dev) })?;
        let mut value: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: live handle; key is a NUL-terminated literal; on status 0
        // `value` points into `info`'s storage (copied before the free).
        let status =
            unsafe { sys::mlx_device_info_get_string(&mut value, info, c"architecture".as_ptr()) };
        if status != 0 || value.is_null() {
            return Err(MlxError {
                message: "device info has no architecture string".to_owned(),
            });
        }
        // SAFETY: non-null NUL-terminated string owned by `info`, which is
        // alive until after the copy below.
        Ok(unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned())
    })();
    // SAFETY: live handle, freed exactly once.
    let _ = unsafe { sys::mlx_device_info_free(info) };
    debug::track_free();
    result
}
