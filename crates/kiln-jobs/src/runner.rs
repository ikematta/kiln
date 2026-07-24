//! Job execution: dispatches a stored job record to the download or quantize
//! engine, mirroring progress into the store (and stdout for the CLI).

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::events::{Event, Fanout, Sink, StdoutSink, StoreSink};
use crate::hub::HubClient;
use crate::quantize::{self, ConvertSpec};
use crate::store::{JobKind, JobRecord, JobState, JobStore};

/// Runner-level defaults shared by the CLI and `serve`.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Where downloads and quantized outputs land when a job names no
    /// explicit destination (the gateway passes its `model_dir`).
    pub dest_root: PathBuf,
    /// uv project directory of the jobs venv (quantize only).
    pub venv: PathBuf,
}

/// The concrete, defaults-filled parameters stored as `spec_json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub repo: String,
    pub revision: String,
    pub dest: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeJob {
    pub src: String,
    pub bits: u32,
    pub group_size: u32,
    pub out: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("{0}")]
    Invalid(String),
    #[error("spec_json does not parse: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Validates and fills defaults for a download submission.
pub fn download_job(
    repo: &str,
    revision: Option<&str>,
    dest: Option<PathBuf>,
    config: &RunnerConfig,
) -> Result<DownloadJob, SpecError> {
    if repo.is_empty() {
        return Err(SpecError::Invalid("repo must be non-empty".into()));
    }
    Ok(DownloadJob {
        repo: repo.to_string(),
        revision: revision
            .filter(|rev| !rev.is_empty())
            .unwrap_or("main")
            .to_string(),
        dest: dest.unwrap_or_else(|| config.dest_root.join(repo.replace('/', "--"))),
    })
}

/// Validates and fills defaults for a quantize submission.
pub fn quantize_job(
    src: &str,
    bits: Option<u32>,
    group_size: Option<u32>,
    out: Option<PathBuf>,
    config: &RunnerConfig,
) -> Result<QuantizeJob, SpecError> {
    if src.is_empty() {
        return Err(SpecError::Invalid("src must be non-empty".into()));
    }
    let bits = bits.unwrap_or(quantize::DEFAULT_BITS);
    let group_size = group_size.unwrap_or(quantize::DEFAULT_GROUP_SIZE);
    quantize::validate(bits, group_size).map_err(|err| SpecError::Invalid(err.to_string()))?;
    Ok(QuantizeJob {
        src: src.to_string(),
        bits,
        group_size,
        out: out.unwrap_or_else(|| quantize::default_out(&config.dest_root, src, bits, group_size)),
    })
}

pub fn to_spec_json(job: &impl Serialize) -> String {
    serde_json::to_string(job).unwrap_or_else(|_| "{}".to_string())
}

/// Runs one job to a terminal state. Every outcome — including a spec that
/// no longer parses — lands in the store; this function never panics and
/// never leaves a job `running`.
pub async fn execute(
    store: &Arc<JobStore>,
    record: &JobRecord,
    config: &RunnerConfig,
    mirror_stdout: bool,
) -> JobState {
    let mut sinks: Vec<Box<dyn Sink>> = vec![Box::new(StoreSink::new(
        Arc::clone(store),
        record.id.clone(),
    ))];
    if mirror_stdout {
        sinks.push(Box::new(StdoutSink));
    }
    let sink: Arc<dyn Sink> = Arc::new(Fanout(sinks));

    if let Err(err) = store.set_state(&record.id, JobState::Running, None) {
        tracing::error!(job = %record.id, error = %err, "failed to mark job running");
    }

    let result: Result<String, String> = match record.kind {
        JobKind::Download => run_download(&record.spec_json, sink.as_ref()).await,
        JobKind::Quantize => run_quantize(&record.spec_json, config, Arc::clone(&sink)).await,
    };

    let (state, detail) = match result {
        Ok(dest) => (JobState::Succeeded, Event::Done { dest }.to_json()),
        Err(message) => {
            let event = Event::Error {
                message: message.clone(),
            };
            sink.emit(&event);
            tracing::warn!(job = %record.id, error = %message, "job failed");
            (JobState::Failed, event.to_json())
        }
    };
    if let Err(err) = store.set_state(&record.id, state, Some(&detail)) {
        tracing::error!(job = %record.id, error = %err, "failed to record job outcome");
    }
    state
}

async fn run_download(spec_json: &str, sink: &dyn Sink) -> Result<String, String> {
    let job: DownloadJob = serde_json::from_str(spec_json).map_err(|err| err.to_string())?;
    // Hub credentials (HF_TOKEN) are read from the environment at execution
    // time, never from spec_json — persisted job state must stay secret-free.
    let hub = HubClient::from_env().map_err(|err| err.to_string())?;
    hub.download_repo(&job.repo, &job.revision, &job.dest, sink)
        .await
        .map_err(|err| err.to_string())?;
    Ok(job.dest.display().to_string())
}

async fn run_quantize(
    spec_json: &str,
    config: &RunnerConfig,
    sink: Arc<dyn Sink>,
) -> Result<String, String> {
    let job: QuantizeJob = serde_json::from_str(spec_json).map_err(|err| err.to_string())?;
    let spec = ConvertSpec {
        src: job.src,
        out: job.out.clone(),
        bits: job.bits,
        group_size: job.group_size,
        venv: config.venv.clone(),
    };
    quantize::run_convert(&spec, sink)
        .await
        .map_err(|err| err.to_string())?;
    Ok(job.out.display().to_string())
}
