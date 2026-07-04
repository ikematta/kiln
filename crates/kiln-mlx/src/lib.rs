//! `kiln-mlx`: FFI bindings and safe wrappers over the vendored mlx-c.
//!
//! This is the ONLY crate in the workspace permitted to contain `unsafe`
//! (CLAUDE.md). All raw bindings live in [`sys`] (bindgen-generated at build
//! time from the pinned headers); everything else goes through the safe
//! wrapper modules: [`Array`]/[`Stream`] RAII handles, [`ops`]/[`fast`]/
//! [`random`] lazy graph builders, [`memory`] introspection, the recording
//! [`error`] handler, and the [`debug`] leak counter.
//!
//! Threading model: all MLX operations are issued from the single engine
//! thread. [`Stream`] and [`Array`] are `!Send`/`!Sync` to enforce it.
//!
//! Everything is gated on the `metal` feature; without it (the Linux CI
//! compile-check) this crate is empty.

#[cfg(feature = "metal")]
mod array;
#[cfg(feature = "metal")]
pub mod debug;
#[cfg(feature = "metal")]
pub mod error;
#[cfg(feature = "metal")]
pub mod fast;
#[cfg(feature = "metal")]
pub mod memory;
#[cfg(feature = "metal")]
pub mod ops;
#[cfg(feature = "metal")]
pub mod random;
#[cfg(feature = "metal")]
pub mod smoke;
#[cfg(feature = "metal")]
mod stream;
#[cfg(feature = "metal")]
pub mod sys;

#[cfg(feature = "metal")]
pub use array::{Array, Dtype, async_eval, eval};
#[cfg(feature = "metal")]
pub use error::MlxError;
#[cfg(feature = "metal")]
pub use stream::Stream;

/// Installs the recording MLX error handler (idempotent). Called lazily by
/// every wrapper constructor; worker binaries should also call it explicitly
/// at startup, before any MLX work (CLAUDE.md: the default handler `exit()`s
/// the process on any MLX error).
#[cfg(feature = "metal")]
pub fn init() {
    error::install_error_handler();
}
