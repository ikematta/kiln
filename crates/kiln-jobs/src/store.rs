//! SQLite job store (SPEC §9.1: "Job state in a SQLite file").
//!
//! One process owns the file at a time (the CLI or `kiln-jobs serve`);
//! access within the process is serialized by a mutex — job bookkeeping is
//! a handful of sub-millisecond writes per job, not a throughput surface.

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("job store poisoned by a panicked writer")]
    Poisoned,
    #[error("failed to create job store directory: {0}")]
    CreateDir(#[source] std::io::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Download,
    Quantize,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Download => "download",
            JobKind::Quantize => "quantize",
        }
    }

    fn from_str(text: &str) -> Option<Self> {
        match text {
            "download" => Some(JobKind::Download),
            "quantize" => Some(JobKind::Quantize),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Succeeded => "succeeded",
            JobState::Failed => "failed",
        }
    }

    fn from_str(text: &str) -> Option<Self> {
        match text {
            "queued" => Some(JobState::Queued),
            "running" => Some(JobState::Running),
            "succeeded" => Some(JobState::Succeeded),
            "failed" => Some(JobState::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: String,
    pub kind: JobKind,
    pub state: JobState,
    pub spec_json: String,
    pub detail_json: String,
    pub created_unix: i64,
    pub updated_unix: i64,
}

pub struct JobStore {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    state        TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    detail_json  TEXT NOT NULL DEFAULT '',
    created_unix INTEGER NOT NULL,
    updated_unix INTEGER NOT NULL
);
";

/// What an interrupted job's detail says after crash recovery. Downloads are
/// safe to resubmit: completed files verify-and-skip, partial files resume
/// from their `.part` offset.
const INTERRUPTED_DETAIL: &str =
    r#"{"event":"error","message":"interrupted: job runner exited mid-job; resubmit to resume"}"#;

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl JobStore {
    /// Opens (creating if needed) the store and runs crash recovery: any job
    /// still marked `running` was interrupted by a previous process exit and
    /// is moved to `failed` so pollers never see a phantom in-flight job.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(StoreError::CreateDir)?;
        }
        let conn = Connection::open(path)?;
        // WAL keeps the file readable by forensic tooling mid-write; the
        // busy timeout covers the brief window where a previous owner's WAL
        // checkpoint is still completing.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.execute_batch(SCHEMA)?;
        conn.execute(
            "UPDATE jobs SET state = 'failed', detail_json = ?1, updated_unix = ?2
             WHERE state = 'running'",
            params![INTERRUPTED_DETAIL, now_unix()],
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn with_conn<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, StoreError> {
        let conn = self.conn.lock().map_err(|_| StoreError::Poisoned)?;
        Ok(f(&conn)?)
    }

    pub fn insert(&self, kind: JobKind, spec_json: &str) -> Result<JobRecord, StoreError> {
        let record = JobRecord {
            id: uuid::Uuid::now_v7().to_string(),
            kind,
            state: JobState::Queued,
            spec_json: spec_json.to_string(),
            detail_json: String::new(),
            created_unix: now_unix(),
            updated_unix: now_unix(),
        };
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO jobs (id, kind, state, spec_json, detail_json, created_unix, updated_unix)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    record.id,
                    record.kind.as_str(),
                    record.state.as_str(),
                    record.spec_json,
                    record.detail_json,
                    record.created_unix,
                    record.updated_unix,
                ],
            )
            .map(|_| ())
        })?;
        Ok(record)
    }

    pub fn set_state(
        &self,
        id: &str,
        state: JobState,
        detail_json: Option<&str>,
    ) -> Result<(), StoreError> {
        self.with_conn(|conn| {
            match detail_json {
                Some(detail) => conn.execute(
                    "UPDATE jobs SET state = ?2, detail_json = ?3, updated_unix = ?4 WHERE id = ?1",
                    params![id, state.as_str(), detail, now_unix()],
                ),
                None => conn.execute(
                    "UPDATE jobs SET state = ?2, updated_unix = ?3 WHERE id = ?1",
                    params![id, state.as_str(), now_unix()],
                ),
            }
            .map(|_| ())
        })
    }

    pub fn update_detail(&self, id: &str, detail_json: &str) -> Result<(), StoreError> {
        self.with_conn(|conn| {
            conn.execute(
                "UPDATE jobs SET detail_json = ?2, updated_unix = ?3 WHERE id = ?1",
                params![id, detail_json, now_unix()],
            )
            .map(|_| ())
        })
    }

    pub fn get(&self, id: &str) -> Result<Option<JobRecord>, StoreError> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT id, kind, state, spec_json, detail_json, created_unix, updated_unix
                 FROM jobs WHERE id = ?1",
                params![id],
                row_to_record,
            )
            .optional()
        })
    }

    /// Newest first, capped — the admin list endpoint is a dashboard, not an
    /// archive query.
    pub fn list(&self, limit: usize) -> Result<Vec<JobRecord>, StoreError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, kind, state, spec_json, detail_json, created_unix, updated_unix
                 FROM jobs ORDER BY id DESC LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], row_to_record)?;
            rows.collect()
        })
    }

    /// Jobs accepted but not yet started, oldest first — re-enqueued when
    /// `kiln-jobs serve` starts.
    pub fn queued(&self) -> Result<Vec<JobRecord>, StoreError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, kind, state, spec_json, detail_json, created_unix, updated_unix
                 FROM jobs WHERE state = 'queued' ORDER BY id ASC",
            )?;
            let rows = stmt.query_map([], row_to_record)?;
            rows.collect()
        })
    }
}

fn row_to_record(row: &rusqlite::Row<'_>) -> Result<JobRecord, rusqlite::Error> {
    let kind_text: String = row.get(1)?;
    let state_text: String = row.get(2)?;
    let parse_err = |what: &str, value: &str| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            format!("unknown {what} '{value}' in job store").into(),
        )
    };
    Ok(JobRecord {
        id: row.get(0)?,
        kind: JobKind::from_str(&kind_text).ok_or_else(|| parse_err("job kind", &kind_text))?,
        state: JobState::from_str(&state_text)
            .ok_or_else(|| parse_err("job state", &state_text))?,
        spec_json: row.get(3)?,
        detail_json: row.get(4)?,
        created_unix: row.get(5)?,
        updated_unix: row.get(6)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("kiln-jobs-store-{}.sqlite", uuid::Uuid::now_v7()))
    }

    #[test]
    fn insert_get_list_roundtrip() {
        let path = temp_db();
        let store = JobStore::open(&path).expect("open");
        let a = store
            .insert(JobKind::Download, r#"{"repo":"org/a"}"#)
            .expect("insert");
        let b = store
            .insert(JobKind::Quantize, r#"{"src":"/m"}"#)
            .expect("insert");

        let got = store.get(&a.id).expect("get").expect("present");
        assert_eq!(got.kind, JobKind::Download);
        assert_eq!(got.state, JobState::Queued);
        assert_eq!(got.spec_json, r#"{"repo":"org/a"}"#);

        // Newest first (uuid v7 ids are time-ordered).
        let listed = store.list(10).expect("list");
        assert_eq!(
            listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec![b.id.as_str(), a.id.as_str()]
        );
        assert!(store.get("missing").expect("get").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn state_transitions_and_detail() {
        let path = temp_db();
        let store = JobStore::open(&path).expect("open");
        let job = store.insert(JobKind::Download, "{}").expect("insert");

        store
            .set_state(&job.id, JobState::Running, None)
            .expect("running");
        store
            .update_detail(&job.id, r#"{"event":"progress"}"#)
            .expect("detail");
        let got = store.get(&job.id).expect("get").expect("present");
        assert_eq!(got.state, JobState::Running);
        assert_eq!(got.detail_json, r#"{"event":"progress"}"#);

        store
            .set_state(&job.id, JobState::Succeeded, Some(r#"{"event":"done"}"#))
            .expect("done");
        let got = store.get(&job.id).expect("get").expect("present");
        assert_eq!(got.state, JobState::Succeeded);
        assert_eq!(got.detail_json, r#"{"event":"done"}"#);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopen_fails_interrupted_running_jobs_and_requeues_nothing() {
        let path = temp_db();
        {
            let store = JobStore::open(&path).expect("open");
            let running = store.insert(JobKind::Download, "{}").expect("insert");
            store
                .set_state(&running.id, JobState::Running, None)
                .expect("running");
            store.insert(JobKind::Quantize, "{}").expect("insert");
        }
        let store = JobStore::open(&path).expect("reopen");
        let listed = store.list(10).expect("list");
        let states: Vec<JobState> = listed.iter().map(|r| r.state).collect();
        assert_eq!(states, vec![JobState::Queued, JobState::Failed]);
        let failed = &listed[1];
        assert!(failed.detail_json.contains("interrupted"));
        // The queued job survives recovery and is re-enqueued by serve.
        assert_eq!(store.queued().expect("queued").len(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
