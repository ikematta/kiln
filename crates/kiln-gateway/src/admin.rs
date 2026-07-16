//! Admin jobs API (SPEC §8.1: `POST /admin/jobs/*` "proxied to kiln-jobs";
//! SPEC §9.1: "Gateway admin API proxies job submission/status").
//!
//! The gateway never runs jobs itself: `kiln-jobs serve` is spawned on
//! demand (SPEC §2.1: "one process, on demand") on a UDS under the runtime
//! dir, and these handlers translate admin JSON to the `kiln.v1.Jobs` gRPC
//! service and back. Bare minimum for Phase 10 part 1 — submit + poll; the
//! full admin surface (models table, stats SSE) is part 2.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::jobs_client::JobsClient;
use kiln_proto::v1::{
    DownloadSpec, JobKind, JobRef, JobState, JobStatus, ListJobsRequest, QuantizeSpec,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Child;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::app::AppState;
use crate::config::KilnConfig;
use crate::error::ApiError;
use crate::registry::expand_tilde;
use crate::uds;

/// How long a freshly spawned kiln-jobs gets to bind its socket.
const SPAWN_READY_TIMEOUT: Duration = Duration::from_secs(10);

/// Lazily spawns and talks to the kiln-jobs server.
pub struct JobsProxy {
    argv: Vec<String>,
    socket: PathBuf,
    db: PathBuf,
    dest_root: PathBuf,
    channel: Channel,
    /// The spawned child, if this proxy owns one. `kill_on_drop` reaps it
    /// with the gateway; the SQLite store's crash recovery handles whatever
    /// was mid-flight.
    child: Mutex<Option<Child>>,
    /// Test hook: `external()` proxies to an already-running server.
    spawn: bool,
}

impl JobsProxy {
    pub fn from_config(config: &KilnConfig) -> Result<Self, tonic::transport::Error> {
        let runtime_dir = expand_tilde(&config.server.runtime_dir);
        let socket = runtime_dir.join("jobs.sock");
        Ok(Self {
            argv: config.server.jobs_argv.clone(),
            db: expand_tilde(&config.server.jobs_db),
            dest_root: expand_tilde(&config.server.model_dir),
            channel: uds::uds_channel(socket.clone())?,
            socket,
            child: Mutex::new(None),
            spawn: true,
        })
    }

    /// Proxy to an existing server on `socket` without spawning (tests).
    pub fn external(socket: PathBuf) -> Result<Self, tonic::transport::Error> {
        Ok(Self {
            argv: Vec::new(),
            db: PathBuf::new(),
            dest_root: PathBuf::new(),
            channel: uds::uds_channel(socket.clone())?,
            socket,
            child: Mutex::new(None),
            spawn: false,
        })
    }

    /// A connected client, spawning the job server first if needed.
    async fn client(&self) -> Result<JobsClient<Channel>, ApiError> {
        self.ensure_running().await?;
        Ok(JobsClient::new(self.channel.clone()))
    }

    async fn ensure_running(&self) -> Result<(), ApiError> {
        if !self.spawn {
            return Ok(());
        }
        let mut child = self.child.lock().await;
        if let Some(running) = child.as_mut() {
            match running.try_wait() {
                Ok(None) => return Ok(()), // still alive
                Ok(Some(status)) => {
                    tracing::warn!(status = %status, "kiln-jobs exited; respawning");
                }
                Err(err) => {
                    tracing::warn!(error = %err, "kiln-jobs status unknown; respawning");
                }
            }
            *child = None;
        }

        let mut argv = self.argv.clone();
        argv.extend(
            [
                "serve",
                "--socket",
                &self.socket.display().to_string(),
                "--db",
                &self.db.display().to_string(),
                "--dest-root",
                &self.dest_root.display().to_string(),
            ]
            .map(String::from),
        );
        tracing::info!(argv = ?argv, "spawning kiln-jobs");
        let spawned = tokio::process::Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|err| {
                ApiError::jobs_unavailable(format!(
                    "failed to spawn kiln-jobs ({}): {err}",
                    self.argv.join(" ")
                ))
            })?;
        *child = Some(spawned);

        // Wait for the socket to accept — the bind happens after the store
        // opens, so acceptance implies the server is serving.
        let deadline = tokio::time::Instant::now() + SPAWN_READY_TIMEOUT;
        loop {
            if tokio::net::UnixStream::connect(&self.socket).await.is_ok() {
                return Ok(());
            }
            if let Some(running) = child.as_mut()
                && let Ok(Some(status)) = running.try_wait()
            {
                *child = None;
                return Err(ApiError::jobs_unavailable(format!(
                    "kiln-jobs exited during startup ({status}); check gateway logs"
                )));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(ApiError::jobs_unavailable(
                    "kiln-jobs did not bind its socket in time",
                ));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

fn status_to_api_error(status: tonic::Status) -> ApiError {
    match status.code() {
        tonic::Code::InvalidArgument => ApiError::invalid_request(status.message().to_string()),
        tonic::Code::NotFound => ApiError::not_found(status.message().to_string()),
        code => {
            ApiError::jobs_unavailable(format!("job runner error ({code:?}): {}", status.message()))
        }
    }
}

/// The admin JSON shape: proto enums as lowercase strings, the stored
/// spec/detail JSON inlined as objects.
fn job_json(status: &JobStatus) -> Value {
    let parse = |text: &str| -> Value {
        if text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
        }
    };
    json!({
        "id": status.id,
        "kind": match status.kind() {
            JobKind::Download => "download",
            JobKind::Quantize => "quantize",
            JobKind::Unspecified => "unspecified",
        },
        "state": match status.state() {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Succeeded => "succeeded",
            JobState::Failed => "failed",
            JobState::Unspecified => "unspecified",
        },
        "spec": parse(&status.spec_json),
        "detail": parse(&status.detail_json),
        "created_unix": status.created_unix,
        "updated_unix": status.updated_unix,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadBody {
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub dest: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizeBody {
    pub path: String,
    #[serde(default)]
    pub bits: Option<u32>,
    #[serde(default)]
    pub group_size: Option<u32>,
    #[serde(default)]
    pub out: Option<String>,
}

pub async fn submit_download(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DownloadBody>,
) -> Result<Response, ApiError> {
    let mut client = state.jobs.client().await?;
    let status = client
        .submit_download(DownloadSpec {
            repo: body.repo,
            revision: body.revision.unwrap_or_default(),
            dest: body.dest.unwrap_or_default(),
        })
        .await
        .map_err(status_to_api_error)?
        .into_inner();
    Ok((StatusCode::ACCEPTED, Json(job_json(&status))).into_response())
}

pub async fn submit_quantize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<QuantizeBody>,
) -> Result<Response, ApiError> {
    let mut client = state.jobs.client().await?;
    let status = client
        .submit_quantize(QuantizeSpec {
            src: body.path,
            bits: body.bits.unwrap_or(0),
            group_size: body.group_size.unwrap_or(0),
            out: body.out.unwrap_or_default(),
        })
        .await
        .map_err(status_to_api_error)?
        .into_inner();
    Ok((StatusCode::ACCEPTED, Json(job_json(&status))).into_response())
}

pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let mut client = state.jobs.client().await?;
    let status = client
        .get_job(JobRef { id })
        .await
        .map_err(status_to_api_error)?
        .into_inner();
    Ok(Json(job_json(&status)).into_response())
}

pub async fn list_jobs(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    let mut client = state.jobs.client().await?;
    let list = client
        .list_jobs(ListJobsRequest {})
        .await
        .map_err(status_to_api_error)?
        .into_inner();
    let jobs: Vec<Value> = list.jobs.iter().map(job_json).collect();
    Ok(Json(json!({ "jobs": jobs })).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{self, AppState};
    use crate::auth::Auth;
    use crate::config::KilnConfig;
    use crate::metrics::Metrics;
    use crate::supervisor::Supervisor;
    use axum::body::Body;
    use axum::http::Request;
    use kiln_proto::v1::jobs_server::{Jobs, JobsServer};
    use tower::ServiceExt as _;

    const ADMIN_TOKEN: &str = "admin-token";

    /// Canned kiln.v1.Jobs service: enough behavior to exercise the proxy's
    /// translation (JSON in/out, status mapping). The real server is covered
    /// by kiln-jobs' own tests and the e2e suite.
    struct StubJobs;

    fn canned(kind: JobKind, spec_json: &str) -> JobStatus {
        JobStatus {
            id: "job-1".to_string(),
            kind: kind as i32,
            state: JobState::Queued as i32,
            spec_json: spec_json.to_string(),
            detail_json: String::new(),
            created_unix: 1,
            updated_unix: 1,
        }
    }

    #[tonic::async_trait]
    impl Jobs for StubJobs {
        async fn submit_download(
            &self,
            request: tonic::Request<DownloadSpec>,
        ) -> Result<tonic::Response<JobStatus>, tonic::Status> {
            let spec = request.into_inner();
            if spec.repo.is_empty() {
                return Err(tonic::Status::invalid_argument("repo must be non-empty"));
            }
            Ok(tonic::Response::new(canned(
                JobKind::Download,
                &format!(r#"{{"repo":"{}"}}"#, spec.repo),
            )))
        }

        async fn submit_quantize(
            &self,
            request: tonic::Request<QuantizeSpec>,
        ) -> Result<tonic::Response<JobStatus>, tonic::Status> {
            let spec = request.into_inner();
            if spec.bits == 5 {
                return Err(tonic::Status::invalid_argument("unsupported bits 5"));
            }
            Ok(tonic::Response::new(canned(
                JobKind::Quantize,
                &format!(r#"{{"src":"{}"}}"#, spec.src),
            )))
        }

        async fn get_job(
            &self,
            request: tonic::Request<JobRef>,
        ) -> Result<tonic::Response<JobStatus>, tonic::Status> {
            let id = request.into_inner().id;
            if id == "missing" {
                return Err(tonic::Status::not_found(format!("no job with id {id}")));
            }
            let mut status = canned(JobKind::Download, r#"{"repo":"org/m"}"#);
            status.detail_json = r#"{"event":"done","dest":"/models/org--m"}"#.to_string();
            status.state = JobState::Succeeded as i32;
            Ok(tonic::Response::new(status))
        }

        async fn list_jobs(
            &self,
            _request: tonic::Request<ListJobsRequest>,
        ) -> Result<tonic::Response<kiln_proto::v1::JobList>, tonic::Status> {
            Ok(tonic::Response::new(kiln_proto::v1::JobList {
                jobs: vec![canned(JobKind::Download, r#"{"repo":"org/m"}"#)],
            }))
        }
    }

    /// Serves the stub on a fresh UDS (under /tmp — macOS 104-byte limit)
    /// and returns a router whose JobsProxy points at it.
    async fn stub_router(admin_hash: Option<String>) -> axum::Router {
        let socket = PathBuf::from(format!(
            "/tmp/kiln-admin-test-{}.sock",
            uuid::Uuid::now_v7()
        ));
        let listener = tokio::net::UnixListener::bind(&socket).expect("bind stub uds");
        tokio::spawn(async move {
            let incoming = async_stream::stream! {
                loop {
                    yield listener.accept().await.map(|(stream, _addr)| stream);
                }
            };
            let _ = tonic::transport::Server::builder()
                .add_service(JobsServer::new(StubJobs))
                .serve_with_incoming(incoming)
                .await;
        });

        let mut config = KilnConfig::default();
        config.auth.admin_token_hash = admin_hash;
        let metrics = Arc::new(Metrics::new().expect("metrics"));
        let (registry, lifecycle, _supervisor) =
            Supervisor::start(&config, Arc::clone(&metrics)).expect("supervisor with no models");
        let state = Arc::new(AppState {
            registry,
            lifecycle,
            metrics,
            auth: Auth::from_config(&config.auth).expect("valid auth config"),
            jobs: JobsProxy::external(socket).expect("proxy"),
        });
        app::router(state)
    }

    fn admin_hash() -> String {
        use argon2::PasswordHasher;
        use argon2::password_hash::{SaltString, rand_core::OsRng};
        argon2::Argon2::default()
            .hash_password(ADMIN_TOKEN.as_bytes(), &SaltString::generate(&mut OsRng))
            .expect("hashing works")
            .to_string()
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn request(method: &str, uri: &str, token: Option<&str>, body: Option<Value>) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        match body {
            Some(value) => builder
                .header("content-type", "application/json")
                .body(Body::from(value.to_string())),
            None => builder.body(Body::empty()),
        }
        .expect("request")
    }

    #[tokio::test]
    async fn admin_is_fail_closed_without_a_configured_token() {
        let router = stub_router(None).await;
        let response = router
            .oneshot(request("GET", "/admin/jobs", Some(ADMIN_TOKEN), None))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "admin_disabled");
    }

    #[tokio::test]
    async fn admin_rejects_missing_and_wrong_tokens() {
        let router = stub_router(Some(admin_hash())).await;
        for token in [None, Some("wrong")] {
            let response = router
                .clone()
                .oneshot(request("GET", "/admin/jobs", token, None))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn download_submission_proxies_and_translates() {
        let router = stub_router(Some(admin_hash())).await;
        let response = router
            .oneshot(request(
                "POST",
                "/admin/jobs/download",
                Some(ADMIN_TOKEN),
                Some(json!({"repo": "org/m"})),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = body_json(response).await;
        assert_eq!(body["id"], "job-1");
        assert_eq!(body["kind"], "download");
        assert_eq!(body["state"], "queued");
        // spec_json arrives inlined as an object, not a string.
        assert_eq!(body["spec"]["repo"], "org/m");
    }

    #[tokio::test]
    async fn invalid_argument_maps_to_400_and_not_found_to_404() {
        let router = stub_router(Some(admin_hash())).await;
        let response = router
            .clone()
            .oneshot(request(
                "POST",
                "/admin/jobs/quantize",
                Some(ADMIN_TOKEN),
                Some(json!({"path": "/m", "bits": 5})),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let response = router
            .oneshot(request(
                "GET",
                "/admin/jobs/missing",
                Some(ADMIN_TOKEN),
                None,
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn get_and_list_translate_detail_and_enums() {
        let router = stub_router(Some(admin_hash())).await;
        let response = router
            .clone()
            .oneshot(request("GET", "/admin/jobs/job-1", Some(ADMIN_TOKEN), None))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["state"], "succeeded");
        assert_eq!(body["detail"]["event"], "done");

        let response = router
            .oneshot(request("GET", "/admin/jobs", Some(ADMIN_TOKEN), None))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["jobs"][0]["kind"], "download");
    }
}
