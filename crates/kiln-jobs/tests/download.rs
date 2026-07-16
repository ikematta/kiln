//! Download engine tests against a local stub hub — deterministic, no
//! network. Each failure mode the fetch-test-model.sh hardening proved out
//! is reproduced here: mid-transfer connection loss (Range resume), a server
//! that ignores Range, a refused resume window (416), sha256 mismatch, and
//! the retryable/fatal HTTP status split.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use kiln_jobs::events::{Event, Sink};
use kiln_jobs::hub::{HubClient, HubError, REVISION_MARKER};
use sha2::{Digest, Sha256};

const SHA: &str = "e2e0e2e0e2e0e2e0e2e0e2e0e2e0e2e0e2e0e2e0";

/// Deterministic pseudo-random content, incompressible enough to be honest.
fn file_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8)
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Per-file failure injection, applied on the Nth resolve request for that
/// file (1-based), once.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Inject {
    None,
    /// Drop the connection after sending this many body bytes.
    DropAfter(usize),
    /// Ignore any Range header; reply 200 with the full body.
    IgnoreRange,
    /// Refuse the resume window.
    Refuse416,
    /// Serve a corrupted body (right length, wrong bytes).
    Corrupt,
    /// Reply 503.
    Retryable503,
    /// Reply 404.
    Fatal404,
}

struct StubFile {
    bytes: Vec<u8>,
    /// Advertise the content sha as an LFS oid in the tree listing.
    lfs: bool,
    inject_on_first_request: Inject,
}

struct StubState {
    files: HashMap<String, StubFile>,
    /// Range header (if any) per resolve request, in order.
    resolve_requests: Mutex<Vec<Option<String>>>,
    /// 503s to serve on the tree endpoint before succeeding.
    tree_failures: AtomicU32,
}

impl StubState {
    fn requests_for_file(&self) -> Vec<Option<String>> {
        self.resolve_requests.lock().expect("lock").clone()
    }
}

async fn revision(AxumPath((_org, _name, _rev)): AxumPath<(String, String, String)>) -> Response {
    axum::Json(serde_json::json!({ "sha": SHA })).into_response()
}

async fn tree(
    State(state): State<Arc<StubState>>,
    AxumPath((_org, _name, _rev)): AxumPath<(String, String, String)>,
) -> Response {
    if state
        .tree_failures
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
        .is_ok()
    {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let entries: Vec<serde_json::Value> = state
        .files
        .iter()
        .map(|(path, file)| {
            let mut entry = serde_json::json!({
                "type": "file",
                "path": path,
                "size": file.bytes.len(),
            });
            if file.lfs {
                entry["lfs"] = serde_json::json!({ "oid": sha256_hex(&file.bytes) });
            }
            entry
        })
        .collect();
    axum::Json(entries).into_response()
}

async fn resolve(
    State(state): State<Arc<StubState>>,
    AxumPath((_org, _name, _rev, path)): AxumPath<(String, String, String, String)>,
    headers: HeaderMap,
) -> Response {
    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let request_index = {
        let mut log = state.resolve_requests.lock().expect("lock");
        log.push(range.clone());
        log.len()
    };
    let Some(file) = state.files.get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let inject = if request_index == 1 {
        file.inject_on_first_request
    } else {
        Inject::None
    };

    match inject {
        Inject::Retryable503 => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Inject::Fatal404 => return StatusCode::NOT_FOUND.into_response(),
        Inject::Refuse416 => return StatusCode::RANGE_NOT_SATISFIABLE.into_response(),
        _ => {}
    }

    let offset = match (&range, inject) {
        (_, Inject::IgnoreRange) | (None, _) => None,
        (Some(range), _) => range
            .strip_prefix("bytes=")
            .and_then(|r| r.strip_suffix('-'))
            .and_then(|n| n.parse::<usize>().ok()),
    };
    let body_bytes = match inject {
        Inject::Corrupt => file_bytes(file.bytes.len()).iter().map(|b| !b).collect(),
        _ => file.bytes.clone(),
    };
    let (status, slice) = match offset {
        Some(offset) if offset <= body_bytes.len() => {
            (StatusCode::PARTIAL_CONTENT, body_bytes[offset..].to_vec())
        }
        _ => (StatusCode::OK, body_bytes),
    };

    let body = match inject {
        Inject::DropAfter(n) if n < slice.len() => {
            // Declared Content-Length with an erroring stream aborts the
            // connection mid-body — the client sees a transport error after
            // banking `n` bytes.
            let prefix = axum::body::Bytes::from(slice[..n].to_vec());
            let stream = async_stream::stream! {
                yield Ok::<_, std::io::Error>(prefix);
                yield Err(std::io::Error::other("injected connection loss"));
            };
            let mut response = Response::new(Body::from_stream(stream));
            response.headers_mut().insert(
                header::CONTENT_LENGTH,
                header::HeaderValue::from_str(&slice.len().to_string()).expect("len"),
            );
            *response.status_mut() = status;
            return response;
        }
        _ => Body::from(slice),
    };
    (status, body).into_response()
}

/// Starts the stub hub; returns its endpoint URL and shared state.
async fn start_stub(files: Vec<(&str, StubFile)>, tree_failures: u32) -> (String, Arc<StubState>) {
    let state = Arc::new(StubState {
        files: files
            .into_iter()
            .map(|(path, file)| (path.to_string(), file))
            .collect(),
        resolve_requests: Mutex::new(Vec::new()),
        tree_failures: AtomicU32::new(tree_failures),
    });
    let router = Router::new()
        .route("/api/models/{org}/{name}/revision/{rev}", get(revision))
        .route("/api/models/{org}/{name}/tree/{rev}", get(tree))
        .route("/{org}/{name}/resolve/{rev}/{*path}", get(resolve))
        .with_state(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub hub");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    (endpoint, state)
}

#[derive(Default)]
struct CaptureSink(Mutex<Vec<Event>>);

impl Sink for CaptureSink {
    fn emit(&self, event: &Event) {
        self.0.lock().expect("lock").push(event.clone());
    }
}

impl CaptureSink {
    fn events(&self) -> Vec<Event> {
        self.0.lock().expect("lock").clone()
    }
}

fn client(endpoint: &str) -> HubClient {
    HubClient::new(endpoint)
        .expect("client")
        .with_backoff_base(Duration::from_millis(10))
}

fn temp_dest(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("kiln-dl-{tag}-{}", uuid::Uuid::now_v7()))
}

fn plain(bytes: Vec<u8>) -> StubFile {
    StubFile {
        bytes,
        lfs: false,
        inject_on_first_request: Inject::None,
    }
}

fn lfs(bytes: Vec<u8>, inject: Inject) -> StubFile {
    StubFile {
        bytes,
        lfs: true,
        inject_on_first_request: inject,
    }
}

#[tokio::test]
async fn full_download_then_verified_skip_on_rerun() {
    let weights = file_bytes(3 << 20);
    let (endpoint, state) = start_stub(
        vec![
            ("config.json", plain(br#"{"model_type":"llama"}"#.to_vec())),
            ("model.safetensors", lfs(weights.clone(), Inject::None)),
            ("sub/tokenizer.json", plain(b"{}".to_vec())),
        ],
        0,
    )
    .await;
    let dest = temp_dest("full");
    let sink = CaptureSink::default();

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("download succeeds");

    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    assert!(dest.join("sub/tokenizer.json").is_file());
    assert_eq!(
        std::fs::read_to_string(dest.join(REVISION_MARKER)).expect("marker"),
        format!("org/tiny@{SHA}\n")
    );
    let events = sink.events();
    assert!(matches!(events.first(), Some(Event::Plan { files: 3, .. })));
    assert!(matches!(events.last(), Some(Event::Done { .. })));

    // Second run: everything present and verified — no transfers at all.
    let before = state.requests_for_file().len();
    let sink = CaptureSink::default();
    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("rerun succeeds");
    assert_eq!(state.requests_for_file().len(), before);
    let skips = sink
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Skip { .. }))
        .count();
    assert_eq!(skips, 3);
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn interrupted_transfer_resumes_with_range_and_verifies() {
    let weights = file_bytes(4 << 20);
    let dropped_at = 1 << 20;
    let (endpoint, state) = start_stub(
        vec![(
            "model.safetensors",
            lfs(weights.clone(), Inject::DropAfter(dropped_at)),
        )],
        0,
    )
    .await;
    let dest = temp_dest("resume");
    let sink = CaptureSink::default();

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("download succeeds after resume");

    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    // The second resolve request carried a Range picking up the banked bytes.
    let requests = state.requests_for_file();
    assert_eq!(requests.len(), 2, "one drop, one resume: {requests:?}");
    assert_eq!(requests[0], None);
    let resumed_from: u64 = requests[1]
        .as_deref()
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.strip_suffix('-'))
        .and_then(|n| n.parse().ok())
        .expect("second request has a Range header");
    // The client resumes from the bytes it actually banked — nonzero, and
    // never more than the server sent before dropping (the abort races
    // client reads, so it may be less).
    assert!(
        resumed_from > 0 && resumed_from <= dropped_at as u64,
        "resume offset {resumed_from} outside (0, {dropped_at}]"
    );
    // And the events said so.
    assert!(
        sink.events()
            .iter()
            .any(|e| matches!(e, Event::File { resume_from, .. } if *resume_from == resumed_from))
    );
    assert!(
        sink.events()
            .iter()
            .any(|e| matches!(e, Event::Retry { .. }))
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn server_ignoring_range_restarts_the_file_from_zero() {
    let weights = file_bytes(2 << 20);
    let (endpoint, _state) = start_stub(
        vec![(
            "model.safetensors",
            lfs(weights.clone(), Inject::IgnoreRange),
        )],
        0,
    )
    .await;
    let dest = temp_dest("norange");
    // A stale .part with WRONG bytes: if the client appended the 200 body
    // to it, the size/sha checks would fail; restart-from-zero succeeds.
    std::fs::create_dir_all(&dest).expect("dest");
    std::fs::write(dest.join("model.safetensors.part"), vec![0xAA; 512 << 10]).expect("part");

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &CaptureSink::default())
        .await
        .expect("download succeeds despite ignored Range");
    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn refused_resume_window_discards_partial_and_refetches() {
    let weights = file_bytes(2 << 20);
    let (endpoint, state) = start_stub(
        vec![("model.safetensors", lfs(weights.clone(), Inject::Refuse416))],
        0,
    )
    .await;
    let dest = temp_dest("refuse");
    std::fs::create_dir_all(&dest).expect("dest");
    std::fs::write(dest.join("model.safetensors.part"), &weights[..256 << 10]).expect("part");
    let sink = CaptureSink::default();

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("download succeeds after 416");
    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    let requests = state.requests_for_file();
    assert_eq!(requests.len(), 2, "{requests:?}");
    assert!(requests[0].is_some(), "first request tried to resume");
    assert_eq!(requests[1], None, "after 416 the partial is discarded");
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn sha_mismatch_discards_partial_and_retries() {
    let weights = file_bytes(1 << 20);
    let (endpoint, state) = start_stub(
        vec![("model.safetensors", lfs(weights.clone(), Inject::Corrupt))],
        0,
    )
    .await;
    let dest = temp_dest("corrupt");
    let sink = CaptureSink::default();

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("download succeeds after discarding corrupt body");
    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    let requests = state.requests_for_file();
    assert_eq!(requests.len(), 2, "{requests:?}");
    // Never resume on top of a corrupt prefix: the retry starts from zero.
    assert_eq!(requests[1], None);
    assert!(sink.events().iter().any(|e| matches!(
        e, Event::Retry { error, .. } if error.contains("sha256 mismatch")
    )));
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn non_retryable_status_fails_immediately() {
    let (endpoint, state) = start_stub(
        vec![("gone.bin", lfs(file_bytes(1024), Inject::Fatal404))],
        0,
    )
    .await;
    let dest = temp_dest("fatal");

    let err = client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &CaptureSink::default())
        .await
        .expect_err("404 is fatal");
    assert!(matches!(err, HubError::Status { status: 404, .. }), "{err}");
    assert_eq!(state.requests_for_file().len(), 1, "no retries on 404");
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn retryable_file_status_is_retried_then_succeeds() {
    let weights = file_bytes(1 << 20);
    let (endpoint, state) = start_stub(
        vec![(
            "model.safetensors",
            lfs(weights.clone(), Inject::Retryable503),
        )],
        0,
    )
    .await;
    let dest = temp_dest("file503");

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &CaptureSink::default())
        .await
        .expect("download succeeds after file 503");
    assert_eq!(
        std::fs::read(dest.join("model.safetensors")).expect("weights"),
        weights
    );
    assert_eq!(state.requests_for_file().len(), 2);
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test]
async fn retryable_api_status_is_retried_then_succeeds() {
    let (endpoint, _state) = start_stub(
        vec![("config.json", plain(b"{}".to_vec()))],
        2, // two 503s on the tree endpoint, then success
    )
    .await;
    let dest = temp_dest("api503");
    let sink = CaptureSink::default();

    client(&endpoint)
        .download_repo("org/tiny", "main", &dest, &sink)
        .await
        .expect("download succeeds after tree 503s");
    let retries = sink
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Retry { error, .. } if error.contains("503")))
        .count();
    assert_eq!(retries, 2);
    let _ = std::fs::remove_dir_all(&dest);
}
