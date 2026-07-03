//! Raw `extern "C"` declarations against the vendored mlx-c (pinned at
//! v0.6.0 — see `docs/decisions/0001-mlx-c-pin.md`).
//!
//! Phase 0 carries only the handful of symbols the smoke test needs, written
//! by hand against the pinned headers. The full bindgen-generated surface
//! replaces this file in Phase 3 (SPEC §7.1).
//!
//! All mlx-c handles are single-pointer structs passed by value; fallible
//! calls return a non-zero `c_int` on error.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// `mlx_array` handle (`mlx/c/array.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct mlx_array {
    pub ctx: *mut c_void,
}

/// `mlx_stream` handle (`mlx/c/stream.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct mlx_stream {
    pub ctx: *mut c_void,
}

/// `mlx_vector_array` handle (`mlx/c/vector.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct mlx_vector_array {
    pub ctx: *mut c_void,
}

/// `mlx_error_handler_func` (`mlx/c/error.h`).
pub type mlx_error_handler_func = unsafe extern "C" fn(msg: *const c_char, data: *mut c_void);

unsafe extern "C" {
    // array.h
    pub fn mlx_array_new() -> mlx_array;
    pub fn mlx_array_new_float(val: f32) -> mlx_array;
    pub fn mlx_array_free(arr: mlx_array) -> c_int;
    pub fn mlx_array_item_float32(res: *mut f32, arr: mlx_array) -> c_int;

    // stream.h
    pub fn mlx_default_cpu_stream_new() -> mlx_stream;
    pub fn mlx_stream_free(stream: mlx_stream) -> c_int;

    // vector.h
    pub fn mlx_vector_array_new_value(val: mlx_array) -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;

    // ops.h
    pub fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;

    // transforms.h
    pub fn mlx_eval(outputs: mlx_vector_array) -> c_int;

    // error.h — the default handler calls exit(); workers MUST install a
    // custom one at startup (CLAUDE.md). Wired up in Phase 3.
    pub fn mlx_set_error_handler(
        handler: Option<mlx_error_handler_func>,
        data: *mut c_void,
        dtor: Option<unsafe extern "C" fn(data: *mut c_void)>,
    );
}
