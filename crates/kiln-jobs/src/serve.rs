//! The long-running job server (SPEC §9.1): gRPC over a Unix domain socket,
//! `proto/kiln/v1/jobs.proto`. The gateway's `/admin/jobs/*` endpoints are a
//! thin proxy over this service.
//!
//! Submissions are recorded `queued` and executed by a single sequential
//! runner task — downloads and quantizations are bandwidth/GPU-heavy; v1
//! deliberately runs one at a time. Jobs still `queued` from a previous
//! process (accepted but never started) are re-enqueued at startup; jobs
//! that were mid-flight are failed by the store's crash recovery and can be
//! resubmitted (downloads resume from their `.part` files).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kiln_proto::v1::jobs_server::{Jobs, JobsServer};
use kiln_proto::v1::{
    DownloadSpec, JobKind as ProtoJobKind, JobList, JobRef, JobState as ProtoJobState, JobStatus,
    ListJobsRequest, QuantizeSpec,
};
use tokio::sync::mpsc;
use tonic::{Request, Response, Status};

use crate::runner::{self, RunnerConfig};
use crate::store::{JobKind, JobRecord, JobState, JobStore};

/// The admin list endpoint is a dashboard, not an archive query.
const LIST_LIMIT: usize = 200;

pub fn record_to_proto(record: &JobRecord) -> JobStatus {
    JobStatus {
        id: record.id.clone(),
        kind: match record.kind {
            JobKind::Download => ProtoJobKind::Download,
            JobKind::Quantize => ProtoJobKind::Quantize,
        } as i32,
        state: match record.state {
            JobState::Queued => ProtoJobState::Queued,
            JobState::Running => ProtoJobState::Running,
            JobState::Succeeded => ProtoJobState::Succeeded,
            JobState::Failed => ProtoJobState::Failed,
        } as i32,
        spec_json: record.spec_json.clone(),
        detail_json: record.detail_json.clone(),
        created_unix: record.created_unix,
        updated_unix: record.updated_unix,
    }
}

fn store_error(err: crate::store::StoreError) -> Status {
    Status::internal(format!("job store: {err}"))
}

pub struct JobsService {
    store: Arc<JobStore>,
    queue: mpsc::UnboundedSender<JobRecord>,
    config: RunnerConfig,
}

impl JobsService {
    fn submit(&self, kind: JobKind, spec_json: String) -> Result<Response<JobStatus>, Status> {
        let record = self.store.insert(kind, &spec_json).map_err(store_error)?;
        // A closed queue means the runner task is gone — refuse rather than
        // accept a job that would sit `queued` forever.
        self.queue
            .send(record.clone())
            .map_err(|_| Status::unavailable("job runner is shutting down"))?;
        Ok(Response::new(record_to_proto(&record)))
    }
}

#[tonic::async_trait]
impl Jobs for JobsService {
    async fn submit_download(
        &self,
        request: Request<DownloadSpec>,
    ) -> Result<Response<JobStatus>, Status> {
        let spec = request.into_inner();
        let job = runner::download_job(
            &spec.repo,
            Some(spec.revision.as_str()),
            (!spec.dest.is_empty()).then(|| PathBuf::from(&spec.dest)),
            &self.config,
        )
        .map_err(|err| Status::invalid_argument(err.to_string()))?;
        self.submit(JobKind::Download, runner::to_spec_json(&job))
    }

    async fn submit_quantize(
        &self,
        request: Request<QuantizeSpec>,
    ) -> Result<Response<JobStatus>, Status> {
        let spec = request.into_inner();
        let job = runner::quantize_job(
            &spec.src,
            (spec.bits != 0).then_some(spec.bits),
            (spec.group_size != 0).then_some(spec.group_size),
            (!spec.out.is_empty()).then(|| PathBuf::from(&spec.out)),
            &self.config,
        )
        .map_err(|err| Status::invalid_argument(err.to_string()))?;
        self.submit(JobKind::Quantize, runner::to_spec_json(&job))
    }

    async fn get_job(&self, request: Request<JobRef>) -> Result<Response<JobStatus>, Status> {
        let id = request.into_inner().id;
        match self.store.get(&id).map_err(store_error)? {
            Some(record) => Ok(Response::new(record_to_proto(&record))),
            None => Err(Status::not_found(format!("no job with id {id}"))),
        }
    }

    async fn list_jobs(
        &self,
        _request: Request<ListJobsRequest>,
    ) -> Result<Response<JobList>, Status> {
        let jobs = self
            .store
            .list(LIST_LIMIT)
            .map_err(store_error)?
            .iter()
            .map(record_to_proto)
            .collect();
        Ok(Response::new(JobList { jobs }))
    }
}

/// Binds the UDS, re-enqueues persisted `queued` jobs, and serves until
/// SIGTERM/ctrl-c. The runner task executes jobs one at a time.
pub async fn serve(
    socket: &Path,
    store: Arc<JobStore>,
    config: RunnerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let (queue_tx, mut queue_rx) = mpsc::unbounded_channel::<JobRecord>();
    for record in store.queued()? {
        // Startup re-enqueue cannot fail: the receiver is alive right here.
        let _ = queue_tx.send(record);
    }

    let runner_store = Arc::clone(&store);
    let runner_config = config.clone();
    tokio::spawn(async move {
        while let Some(record) = queue_rx.recv().await {
            tracing::info!(job = %record.id, kind = record.kind.as_str(), "job starting");
            let state = runner::execute(&runner_store, &record, &runner_config, true).await;
            tracing::info!(job = %record.id, state = state.as_str(), "job finished");
        }
    });

    // A stale socket from a crashed predecessor would fail the bind.
    if socket.exists() {
        std::fs::remove_file(socket)?;
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = tokio::net::UnixListener::bind(socket)?;
    tracing::info!(socket = %socket.display(), "kiln-jobs listening");
    let incoming = async_stream::stream! {
        loop {
            yield listener.accept().await.map(|(stream, _addr)| stream);
        }
    };
    let service = JobsService {
        store,
        queue: queue_tx,
        config,
    };
    tonic::transport::Server::builder()
        .add_service(JobsServer::new(service))
        .serve_with_incoming_shutdown(incoming, shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let ctrl_c = tokio::signal::ctrl_c();
    match signal(SignalKind::terminate()) {
        Ok(mut sigterm) => {
            tokio::select! {
                _ = ctrl_c => {}
                _ = sigterm.recv() => {}
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "SIGTERM handler unavailable; ctrl-c only");
            let _ = ctrl_c.await;
        }
    }
}
