#![deny(unsafe_code)]
//! `kiln-jobs`: download/quantize job runner (SPEC §9.1).
//!
//! Three entry points share one execution path:
//! - `kiln-jobs download <hf_repo>` — resumable Hugging Face download,
//!   progress as JSON lines on stdout.
//! - `kiln-jobs quantize <path>` — wraps `python -m mlx_lm convert` in the
//!   jobs venv (`python/kiln_jobs_py`); quantization is never reimplemented.
//! - `kiln-jobs serve --socket <uds>` — the long-running job server the
//!   gateway's `/admin/jobs/*` endpoints proxy to (gRPC over UDS,
//!   `proto/kiln/v1/jobs.proto`).
//!
//! Every job — CLI or server-submitted — is recorded in the SQLite job store.

pub mod events;
pub mod hub;
pub mod quantize;
pub mod runner;
pub mod serve;
pub mod store;

use std::path::{Path, PathBuf};

/// Expands a leading `~/` using `$HOME`; other paths pass through verbatim.
/// (Same rules as the gateway's `registry::expand_tilde`; duplicated because
/// no shared util crate exists and neither binary should link the other.)
pub fn expand_tilde(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if let Some(rest) = text.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path.to_path_buf()
}
