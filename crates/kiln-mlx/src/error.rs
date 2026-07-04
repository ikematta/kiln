//! MLX error handling.
//!
//! mlx-c's *default* error handler prints and calls `exit()` — a worker that
//! dies on a bad shape is a bug (CLAUDE.md). [`install_error_handler`] swaps
//! in a handler that records the message in a thread-local instead; every
//! fallible mlx-c call (non-zero `c_int`) is then turned into a proper
//! [`MlxError`] by [`check`].
//!
//! mlx-c invokes the handler synchronously on the thread that made the
//! failing call (its C shims catch the C++ exception at the boundary), so a
//! thread-local is the correct storage.
//!
//! Installation is idempotent and performed lazily by every wrapper
//! constructor via [`crate::init`]; worker binaries should still call
//! `kiln_mlx::init()` explicitly at startup so the swap happens before any
//! MLX work.

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Once;

use crate::sys;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// An error reported by MLX through the installed error handler.
#[derive(Debug, Clone, thiserror::Error)]
#[error("mlx error: {message}")]
pub struct MlxError {
    pub message: String,
}

#[allow(unsafe_code)]
unsafe extern "C" fn record_error(msg: *const c_char, _data: *mut c_void) {
    let message = if msg.is_null() {
        "unknown mlx error (null message)".to_owned()
    } else {
        // SAFETY: mlx-c passes a NUL-terminated C string owned by the caller
        // for the duration of this call; we copy it out immediately.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message));
}

static INSTALL: Once = Once::new();

/// Replaces mlx-c's exit-on-error default handler with the recording handler.
/// Idempotent; safe to call from any thread.
#[allow(unsafe_code)]
pub fn install_error_handler() {
    INSTALL.call_once(|| {
        // SAFETY: `record_error` is a valid handler for the whole program
        // lifetime; no user data, no destructor.
        unsafe { sys::mlx_set_error_handler(Some(record_error), std::ptr::null_mut(), None) }
    });
}

/// Maps an mlx-c status code to `Result`, consuming the thread-local message
/// recorded by the handler for this failing call.
pub(crate) fn check(status: c_int) -> Result<(), MlxError> {
    if status == 0 {
        return Ok(());
    }
    let message = LAST_ERROR
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_else(|| "mlx call failed but no error message was recorded".to_owned());
    Err(MlxError { message })
}
