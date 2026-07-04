//! `Stream`: a safe wrapper over `mlx_stream`.
//!
//! All MLX operations are issued from the single engine thread (SPEC §6.2).
//! `Stream` is deliberately `!Send`/`!Sync` (raw-pointer field) so the type
//! system confines it to the thread that created it — do NOT "fix" that with
//! a `Mutex` (CLAUDE.md).

use std::marker::PhantomData;

use crate::error::{MlxError, check};
use crate::{debug, sys};

/// An MLX execution stream bound to the creating thread.
#[derive(Debug)]
pub struct Stream {
    raw: sys::mlx_stream,
    /// Raw pointer marker: keeps `Stream` `!Send + !Sync`.
    _not_send: PhantomData<*mut ()>,
}

impl Stream {
    fn from_raw(raw: sys::mlx_stream) -> Self {
        crate::init();
        debug::track_new();
        Self {
            raw,
            _not_send: PhantomData,
        }
    }

    /// The MLX default GPU stream — the stream every inference op runs on.
    #[allow(unsafe_code)]
    pub fn gpu() -> Self {
        // SAFETY: constructor with no preconditions; handle freed in Drop.
        Self::from_raw(unsafe { sys::mlx_default_gpu_stream_new() })
    }

    /// The MLX default CPU stream.
    #[allow(unsafe_code)]
    pub fn cpu() -> Self {
        // SAFETY: constructor with no preconditions; handle freed in Drop.
        Self::from_raw(unsafe { sys::mlx_default_cpu_stream_new() })
    }

    /// Blocks until all work queued on this stream has completed.
    #[allow(unsafe_code)]
    pub fn synchronize(&self) -> Result<(), MlxError> {
        // SAFETY: `self.raw` is a live stream handle.
        check(unsafe { sys::mlx_synchronize(self.raw) })
    }

    pub(crate) fn raw(&self) -> sys::mlx_stream {
        self.raw
    }
}

impl Drop for Stream {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: exactly one free per `*_new`; handle not used after drop.
        let _ = unsafe { sys::mlx_stream_free(self.raw) };
        debug::track_free();
    }
}
