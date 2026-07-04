//! Seeded randomness for sampling (SPEC §6.6): categorical draws with
//! per-request keys derived from the request seed — never MLX's global
//! random state, so requests are deterministic given their seed.

#![allow(unsafe_code)]

use crate::array::Array;
use crate::error::{MlxError, check};
use crate::stream::Stream;
use crate::sys;

type Result<T> = std::result::Result<T, MlxError>;

/// A PRNG key derived from `seed` (`mx.random.key`).
pub fn key(seed: u64) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live output handle; checked status.
    check(unsafe { sys::mlx_random_key(out.raw_out(), seed) })?;
    Ok(out)
}

/// Splits `key` into two independent keys (`mx.random.split`).
pub fn split(key: &Array, s: &Stream) -> Result<(Array, Array)> {
    let mut first = Array::new_handle();
    let mut second = Array::new_handle();
    // SAFETY: live handles; checked status.
    check(unsafe { sys::mlx_random_split(first.raw_out(), second.raw_out(), key.raw(), s.raw()) })?;
    Ok((first, second))
}

/// Draws from a categorical distribution over unnormalized `logits` along
/// the last axis, using `key` (`mx.random.categorical`).
pub fn categorical(logits: &Array, key: &Array, s: &Stream) -> Result<Array> {
    let mut out = Array::new_handle();
    // SAFETY: live handles; checked status.
    check(unsafe {
        sys::mlx_random_categorical(out.raw_out(), logits.raw(), -1, key.raw(), s.raw())
    })?;
    Ok(out)
}
