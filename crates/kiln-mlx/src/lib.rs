//! `kiln-mlx`: FFI bindings and safe wrappers over the vendored mlx-c.
//!
//! This is the ONLY crate in the workspace permitted to contain `unsafe`
//! (CLAUDE.md). All raw bindings live in [`sys`]; everything else must go
//! through safe wrapper modules of this crate. The real wrapper types
//! (`Array`, `Stream`, error handler, leak counter) land in Phase 3.

#[cfg(feature = "metal")]
pub mod smoke;
#[cfg(feature = "metal")]
pub mod sys;
