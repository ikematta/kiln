//! Raw `extern "C"` bindings against the vendored mlx-c, bindgen-generated at
//! build time from the pinned headers (v0.6.0 — see
//! `docs/decisions/0001-mlx-c-pin.md`), per SPEC §7.1.
//!
//! `build.rs` allowlists the `mlx_*` functions/types (and `MLX_*` constants)
//! and writes the bindings to `$OUT_DIR/mlx_sys.rs`. Because the headers are
//! frozen with the submodule pin, the generated surface is stable; nothing in
//! this file is hand-maintained.
//!
//! All mlx-c handles are single-pointer structs passed by value; fallible
//! calls return a non-zero `c_int` on error and report the message through
//! the installed error handler (see [`crate::error`]).

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/mlx_sys.rs"));
