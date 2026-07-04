//! Debug-build allocation counter (CLAUDE.md FFI rules).
//!
//! Every safe wrapper that owns an mlx-c handle (`Array`, `Stream`, the
//! internal vector wrapper) increments the counter on `*_new` and decrements
//! it in `Drop`. Tests that construct arrays must end with [`live_objects`]
//! back at their starting baseline — that is the leak gate.
//!
//! In release builds the counter is compiled out and [`live_objects`] always
//! returns 0.

#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicI64, Ordering};

#[cfg(debug_assertions)]
static LIVE_OBJECTS: AtomicI64 = AtomicI64::new(0);

#[inline]
pub(crate) fn track_new() {
    #[cfg(debug_assertions)]
    LIVE_OBJECTS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn track_free() {
    #[cfg(debug_assertions)]
    LIVE_OBJECTS.fetch_sub(1, Ordering::Relaxed);
}

/// Number of currently live wrapper-owned mlx-c handles (debug builds).
pub fn live_objects() -> i64 {
    #[cfg(debug_assertions)]
    {
        LIVE_OBJECTS.load(Ordering::Relaxed)
    }
    #[cfg(not(debug_assertions))]
    {
        0
    }
}
