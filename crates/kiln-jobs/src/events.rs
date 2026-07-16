//! Progress events. One shape for every consumer: `kiln-jobs download`
//! prints them as JSON lines on stdout (SPEC §9.1), and the job store keeps
//! the latest one per job as `detail_json` for the admin API to poll.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::store::JobStore;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// Repo listing resolved: what the download is about to do.
    Plan {
        repo: String,
        revision: String,
        files: usize,
        total_bytes: u64,
    },
    /// A file transfer (re)starting; `resume_from > 0` means a `.part` file
    /// is being continued via HTTP Range.
    File {
        path: String,
        size: u64,
        resume_from: u64,
    },
    /// Byte counts for the file in flight (throttled to ~1/s) plus the
    /// job-level totals.
    Progress {
        path: String,
        received: u64,
        size: u64,
        done_bytes: u64,
        total_bytes: u64,
    },
    /// File already present and verified; not re-downloaded.
    Skip { path: String },
    /// A retryable failure; the attempt counter is per file / per API call.
    Retry {
        context: String,
        attempt: u32,
        max_attempts: u32,
        error: String,
    },
    /// A line of converter output (`mlx_lm convert` stdout/stderr).
    Log { line: String },
    /// Terminal success; `dest` is the produced directory.
    Done { dest: String },
    /// Terminal failure.
    Error { message: String },
}

impl Event {
    pub fn to_json(&self) -> String {
        // Serialization of this enum cannot fail (no maps with non-string
        // keys, no non-finite floats), but never panic in library code.
        serde_json::to_string(self)
            .unwrap_or_else(|err| format!(r#"{{"event":"error","message":"serialize: {err}"}}"#))
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Event::Done { .. } | Event::Error { .. })
    }
}

pub trait Sink: Send + Sync {
    fn emit(&self, event: &Event);
}

/// JSON lines on stdout — the `kiln-jobs download` contract (SPEC §9.1).
pub struct StdoutSink;

impl Sink for StdoutSink {
    fn emit(&self, event: &Event) {
        println!("{}", event.to_json());
    }
}

/// Mirrors events into the job's `detail_json` column. `Progress` updates
/// are additionally throttled here (one write per second) so a fast local
/// mirror cannot turn the store into a write hot spot; state-changing
/// events always land.
pub struct StoreSink {
    store: Arc<JobStore>,
    job_id: String,
    last_progress_unix: AtomicI64,
}

impl StoreSink {
    pub fn new(store: Arc<JobStore>, job_id: String) -> Self {
        Self {
            store,
            job_id,
            last_progress_unix: AtomicI64::new(0),
        }
    }
}

impl Sink for StoreSink {
    fn emit(&self, event: &Event) {
        if matches!(event, Event::Progress { .. }) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let last = self.last_progress_unix.load(Ordering::Relaxed);
            if now <= last {
                return;
            }
            self.last_progress_unix.store(now, Ordering::Relaxed);
        }
        if let Err(err) = self.store.update_detail(&self.job_id, &event.to_json()) {
            tracing::warn!(job = %self.job_id, error = %err, "failed to persist progress event");
        }
    }
}

pub struct Fanout(pub Vec<Box<dyn Sink>>);

impl Sink for Fanout {
    fn emit(&self, event: &Event) {
        for sink in &self.0 {
            sink.emit(event);
        }
    }
}
