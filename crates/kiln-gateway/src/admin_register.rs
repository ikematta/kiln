//! Runtime model registration (`POST /admin/models`) and the add-model
//! memory estimate (`GET /admin/models/estimate`).
//!
//! The single "Add Model" flow the admin UI drives: HF repo id in →
//! loaded and servable with zero gateway restart. Registration does two
//! things as one operation under [`Registrar::add_lock`]:
//! (a) makes the model live — registry entry, lifecycle slot, supervision
//!     task, all through the same code paths a configured model uses, so
//!     the existing load/unload/pin machinery applies immediately; and
//! (b) persists the same `[[model]]` block into kiln.toml on disk
//!     (crate::config_write: fresh read, in-place edit, comments and hand
//!     edits preserved, loud failure over corruption) so the model
//!     survives the next restart.
//! Persistence runs FIRST: a failed disk write leaves the gateway exactly
//! as it was, and a crash between the write and the live insert costs an
//! extra restart-time entry, never a lost one.
//!
//! A model that is not on disk yet is NOT registered: the handler answers
//! a structured 409 `model_not_downloaded` carrying the exact download
//! coordinates (repo + the dest the resolver will find it at), the UI
//! runs the existing Phase 10 download-job flow against them, and the
//! retried add finds the files in place — one continuous flow.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::AppState;
use crate::config::{KilnConfig, ModelConfig, WorkerKind};
use crate::config_write::{self, PersistError};
use crate::error::ApiError;
use crate::lifecycle;
use crate::registry::{RegistryError, UnloadReason, WorkerStatus, expand_tilde};
use crate::supervisor::{LOAD_OVERHEAD_MARGIN_BYTES, ModelSpawner};

/// Hub API budget for the size probe: an estimate must never hang the
/// dashboard.
const HUB_TIMEOUT: Duration = Duration::from_secs(20);

pub struct Registrar {
    spawner: ModelSpawner,
    config_path: PathBuf,
    /// Expanded `server.runtime_dir` (socket paths for new entries).
    runtime_dir: PathBuf,
    /// Expanded `server.model_dir`: where the default download dest for
    /// an HF repo lives (`<model_dir>/<org>--<name>`, kiln-jobs'
    /// derivation).
    model_dir: PathBuf,
    /// Serializes the whole add flow — live duplicate check, config
    /// write, registry insert — so two concurrent adds cannot interleave.
    add_lock: tokio::sync::Mutex<()>,
    http: reqwest::Client,
}

impl Registrar {
    pub fn new(
        spawner: ModelSpawner,
        config: &KilnConfig,
        config_path: PathBuf,
    ) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(HUB_TIMEOUT)
            .user_agent("kiln-gateway")
            .build()
            .map_err(|err| format!("failed to build hub http client: {err}"))?;
        Ok(Self {
            spawner,
            config_path,
            runtime_dir: expand_tilde(&config.server.runtime_dir),
            model_dir: expand_tilde(&config.server.model_dir),
            add_lock: tokio::sync::Mutex::new(()),
            http,
        })
    }

    /// Where `path` leads right now: a servable local directory, a known
    /// HF repo whose files are not on disk yet, or nothing usable.
    fn resolve(&self, path: &str) -> Resolution {
        let local = expand_tilde(Path::new(path));
        if local.join("config.json").is_file() {
            return Resolution::Local(local);
        }
        if let Some(repo) = hf_repo_shaped(path) {
            // kiln-jobs' default dest derivation (`org/name` → `org--name`
            // under model_dir): a repo the job runner already fetched is
            // local without the operator retyping the path.
            let dest = self.model_dir.join(repo.replace('/', "--"));
            if dest.join("config.json").is_file() {
                return Resolution::Local(dest);
            }
            return Resolution::NotDownloaded {
                repo: repo.to_string(),
                dest,
            };
        }
        Resolution::Invalid(format!(
            "path '{path}' is neither a local model directory (no config.json there) \
             nor an org/name Hugging Face repo id"
        ))
    }

    /// Sum of `*.safetensors` sizes the hub lists for `repo` — the same
    /// weight-bytes basis the Phase 9 load projection uses for local
    /// dirs. `HF_ENDPOINT` overrides the hub exactly as it does for
    /// kiln-jobs downloads.
    async fn hub_weights_bytes(&self, repo: &str) -> Result<u64, ApiError> {
        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
        let endpoint = endpoint.trim_end_matches('/');

        let revision: Value = self
            .hub_json(&format!("{endpoint}/api/models/{repo}/revision/main"), repo)
            .await?;
        let Some(sha) = revision["sha"].as_str() else {
            return Err(hub_error(format!(
                "hub revision response for '{repo}' has no sha"
            )));
        };
        // Single page: the hub lists up to 1000 entries per page and
        // weight shards live at the repo root in the dozens — an estimate
        // does not need pagination.
        let tree: Value = self
            .hub_json(
                &format!("{endpoint}/api/models/{repo}/tree/{sha}?recursive=true"),
                repo,
            )
            .await?;
        let Some(entries) = tree.as_array() else {
            return Err(hub_error(format!(
                "hub tree response for '{repo}' is not a list"
            )));
        };
        Ok(entries
            .iter()
            .filter(|entry| {
                entry["type"].as_str() == Some("file")
                    && entry["path"]
                        .as_str()
                        .is_some_and(|p| p.ends_with(".safetensors"))
            })
            .map(|entry| entry["size"].as_u64().unwrap_or(0))
            .sum())
    }

    async fn hub_json(&self, url: &str, repo: &str) -> Result<Value, ApiError> {
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|err| hub_error(format!("hub request for '{repo}' failed: {err}")))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(ApiError::invalid_request(format!(
                "repo '{repo}' not found on the hub (HTTP 404); check the org/name id"
            )));
        }
        if !response.status().is_success() {
            return Err(hub_error(format!(
                "hub answered HTTP {} for '{repo}'",
                response.status().as_u16()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|err| hub_error(format!("bad hub response for '{repo}': {err}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|err| hub_error(format!("bad hub response for '{repo}': {err}")))
    }
}

enum Resolution {
    Local(PathBuf),
    NotDownloaded { repo: String, dest: PathBuf },
    Invalid(String),
}

/// `org/name` shape a Hugging Face repo id has: exactly one slash, both
/// halves non-empty, and nothing path-like about it.
fn hf_repo_shaped(path: &str) -> Option<&str> {
    let (org, name) = path.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    if path.starts_with(['.', '~']) {
        return None;
    }
    Some(path)
}

fn hub_error(message: String) -> ApiError {
    ApiError {
        status: StatusCode::BAD_GATEWAY,
        error_type: "server_error",
        code: "hub_unavailable",
        message,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddModelBody {
    pub id: String,
    /// HF repo id or local path — the same latitude a `[[model]]` block
    /// has, except the files must already be on disk to register.
    pub path: String,
    #[serde(default)]
    pub worker: WorkerKind,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub ttl_seconds: u64,
}

/// `POST /admin/models`: register a new model — live in the supervisor
/// AND persisted to kiln.toml — without a restart. 201 = registered and
/// immediately loadable via the existing load endpoint; 409
/// `model_exists` for a duplicate id (never an overwrite); 409
/// `model_not_downloaded` with download coordinates when the files are
/// not on disk yet.
pub async fn add_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddModelBody>,
) -> Result<Response, ApiError> {
    if body.id.trim().is_empty() {
        return Err(ApiError::invalid_request("id must be non-empty"));
    }
    if body.path.trim().is_empty() {
        return Err(ApiError::invalid_request("path must be non-empty"));
    }

    let registrar = &state.registrar;
    let _guard = registrar.add_lock.lock().await;

    if state.registry.get(&body.id).is_some() {
        return Err(ApiError::conflict(
            "model_exists",
            format!(
                "a model with id '{}' is already registered; removing or replacing \
                 one requires editing kiln.toml and restarting",
                body.id
            ),
        ));
    }

    let local_dir = match registrar.resolve(&body.path) {
        Resolution::Local(dir) => dir,
        Resolution::NotDownloaded { repo, dest } => {
            // Structured 409 the UI turns into the download-job flow: the
            // standard error envelope plus the exact coordinates to
            // download with and retry.
            let body = json!({
                "error": {
                    "message": format!(
                        "model '{repo}' is not downloaded; submit \
                         POST /admin/jobs/download {{\"repo\": \"{repo}\", \"dest\": {:?}}} \
                         and retry this request when the job succeeds",
                        dest.display().to_string()
                    ),
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "model_not_downloaded",
                },
                "download": {
                    "repo": repo,
                    "dest": dest.display().to_string(),
                },
            });
            return Ok((StatusCode::CONFLICT, Json(body)).into_response());
        }
        Resolution::Invalid(message) => return Err(ApiError::invalid_request(message)),
    };

    let model = ModelConfig {
        id: body.id.clone(),
        // The RESOLVED local path is what registers and persists: kiln.toml
        // must describe something the next boot can load.
        path: local_dir.display().to_string(),
        worker: body.worker,
        pinned: body.pinned,
        ttl_seconds: body.ttl_seconds,
        speculative: None,
    };

    // Build the entry before anything is persisted or published: this is
    // where an unservable model (explicit rust on an unsupported arch,
    // broken tokenizer, over-long socket path) fails, leaving no trace.
    // Initial status Unloaded(Registered): nothing runs until asked, the
    // load button lights up, and /readyz keeps counting the fleet settled.
    let (entry, status_tx) = crate::registry::build_entry(
        &model,
        &registrar.runtime_dir,
        WorkerStatus::Unloaded {
            reason: UnloadReason::Registered,
        },
    )
    .map_err(registry_error_to_api)?;

    // Disk first (fresh-read + in-place edit + re-verify in config_write);
    // failure here leaves the gateway exactly as it was.
    config_write::append_model(&registrar.config_path, &model).map_err(persist_error_to_api)?;

    state
        .registry
        .insert(Arc::clone(&entry))
        .map_err(registry_error_to_api)?;
    let Some(cmd_rx) = state.lifecycle.add_slot(&model, entry.status.clone()) else {
        // Unreachable while the add lock serializes registration; loud
        // anyway rather than a model with no supervision.
        return Err(ApiError::internal(format!(
            "lifecycle slot for '{}' already exists",
            model.id
        )));
    };
    registrar
        .spawner
        .spawn(Arc::clone(&entry), status_tx, cmd_rx);

    tracing::info!(model = %model.id, path = %model.path,
        config = %registrar.config_path.display(),
        "model registered at runtime and persisted to kiln.toml");
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "model": crate::admin_models::model_json(&state, &entry),
            "persisted_to": registrar.config_path.display().to_string(),
        })),
    )
        .into_response())
}

fn registry_error_to_api(err: RegistryError) -> ApiError {
    match err {
        RegistryError::Duplicate(id) => ApiError::conflict(
            "model_exists",
            format!("a model with id '{id}' is already registered"),
        ),
        RegistryError::Channel { .. } => ApiError::internal(err.to_string()),
        // Everything else is the operator asking for something this
        // machine cannot serve — say exactly why.
        other => ApiError::invalid_request(other.to_string()),
    }
}

fn persist_error_to_api(err: PersistError) -> ApiError {
    match err {
        PersistError::DuplicateInFile(_) => ApiError::conflict("config_conflict", err.to_string()),
        // Unparseable/Reverify/IO: fail loudly, nothing was written — the
        // message says so.
        other => ApiError::internal(other.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct EstimateQuery {
    pub path: String,
}

/// `GET /admin/models/estimate?path=…`: the plain memory answer the UI
/// shows before a download — how many bytes this model will want
/// (weights on disk or the hub's listing, plus the same load-overhead
/// margin the Phase 9 projection charges) against the live budget
/// ledger's real headroom AND the live system-memory gate the load
/// itself will face (fresh probe): `fits` is the conjunction, with
/// `fits_budget`/`fits_system` splitting the answer for the UI.
pub async fn estimate_model(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EstimateQuery>,
) -> Result<Json<Value>, ApiError> {
    let registrar = &state.registrar;
    let (weights_bytes, source) = match registrar.resolve(&query.path) {
        Resolution::Local(dir) => (lifecycle::weights_bytes_on_disk(&dir), "local"),
        Resolution::NotDownloaded { repo, .. } => {
            (registrar.hub_weights_bytes(&repo).await?, "hub")
        }
        Resolution::Invalid(message) => return Err(ApiError::invalid_request(message)),
    };
    let estimated_bytes = match weights_bytes {
        0 => 0,
        bytes => bytes + LOAD_OVERHEAD_MARGIN_BYTES,
    };
    let budget_bytes = state.lifecycle.budget_bytes();
    let charged_bytes = state.lifecycle.charged_bytes();
    let headroom_bytes = budget_bytes.saturating_sub(charged_bytes);
    // The same fresh-probe verdict the load path will reach (fails open
    // like it too — a probe failure never blocks the estimate).
    let fits_system = {
        let lifecycle = Arc::clone(&state.lifecycle);
        tokio::task::spawn_blocking(move || lifecycle.admit_load_system(estimated_bytes))
            .await
            .unwrap_or(Ok(()))
            .is_ok()
    };
    let system = state.lifecycle.system_memory();
    Ok(Json(json!({
        "path": query.path,
        "source": source,
        "weights_bytes": weights_bytes,
        "estimated_bytes": estimated_bytes,
        "budget_bytes": budget_bytes,
        "charged_bytes": charged_bytes,
        "headroom_bytes": headroom_bytes,
        "system_available_bytes": system.map(|m| m.available_bytes),
        "pressure_level": system.map(|m| m.pressure_level),
        "min_available_bytes": state.lifecycle.min_available_bytes(),
        "fits_budget": estimated_bytes <= headroom_bytes,
        "fits_system": fits_system,
        "fits": estimated_bytes <= headroom_bytes && fits_system,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Auth;
    use crate::metrics::Metrics;
    use crate::registry::Registry;
    use axum::extract::{Query, State};

    /// Registrar-backed AppState rooted in a throwaway /tmp dir (short
    /// paths: registered sockets must clear the macOS 104-byte UDS limit).
    fn state_for_add(tag: &str) -> (Arc<AppState>, std::path::PathBuf) {
        let root = PathBuf::from(format!(
            "/tmp/kiln-regadd-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(root.join("models")).expect("model dir");
        std::fs::create_dir_all(root.join("run")).expect("runtime dir");
        let config_path = root.join("kiln.toml");
        std::fs::write(
            &config_path,
            "# operator note: hands off\n[server]\nport = 8080\n",
        )
        .expect("config file");

        let mut config = crate::config::KilnConfig::default();
        config.server.runtime_dir = root.join("run");
        config.server.model_dir = root.join("models");
        let metrics = Arc::new(Metrics::new().expect("metrics"));
        let (registry, _senders) = Registry::from_config(&config).expect("empty registry");
        let registry = Arc::new(registry);
        let (lifecycle, _receivers) =
            crate::lifecycle::Lifecycle::new(&config, &registry, Arc::clone(&metrics))
                .expect("lifecycle");
        let lifecycle = Arc::new(lifecycle);
        let registrar = Registrar::new(
            ModelSpawner::test_stub(
                Arc::new(config.clone()),
                Arc::clone(&metrics),
                Arc::clone(&lifecycle),
            ),
            &config,
            config_path,
        )
        .expect("registrar");
        let state = Arc::new(AppState {
            registry,
            lifecycle,
            metrics,
            auth: Auth::from_config(&config.auth).expect("auth"),
            jobs: crate::admin::JobsProxy::external(PathBuf::from("/tmp/kiln-regadd.sock"))
                .expect("proxy"),
            registrar,
            shutdown: tokio::sync::watch::channel(false).1,
        });
        (state, root)
    }

    /// A python-servable model dir (phi3 keeps the rust matrix out of it).
    fn model_dir_in(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("model dir");
        std::fs::write(dir.join("config.json"), b"{\"model_type\": \"phi3\"}").expect("config");
        dir
    }

    fn body(id: &str, path: &str) -> AddModelBody {
        AddModelBody {
            id: id.into(),
            path: path.into(),
            worker: WorkerKind::Python,
            pinned: false,
            ttl_seconds: 0,
        }
    }

    async fn response_json(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    #[tokio::test]
    async fn add_local_model_registers_live_and_persists() {
        let (state, root) = state_for_add("local");
        let dir = model_dir_in(&root, "m1-src");
        let response = add_model(
            State(Arc::clone(&state)),
            Json(body("m1", &dir.display().to_string())),
        )
        .await
        .expect("add succeeds");
        assert_eq!(response.status(), StatusCode::CREATED);
        let json = response_json(response).await;
        assert_eq!(json["model"]["id"], "m1");
        assert_eq!(json["model"]["status"], "unloaded (registered)");

        // Live: in the registry (request path would find it), slot in the
        // lifecycle ledger, listed by the admin models table.
        let entry = state.registry.get("m1").expect("registered");
        assert_eq!(
            entry.status(),
            WorkerStatus::Unloaded {
                reason: UnloadReason::Registered
            }
        );
        assert_eq!(state.lifecycle.pinned("m1"), Some(false));
        let Json(list) = crate::admin_models::list_models(State(Arc::clone(&state))).await;
        assert_eq!(list["models"][0]["id"], "m1");

        // Persisted: the operator comment survives and the file re-parses
        // with exactly one [[model]].
        let text = std::fs::read_to_string(root.join("kiln.toml")).expect("config readable");
        assert!(text.contains("# operator note: hands off"), "{text}");
        let parsed = KilnConfig::parse_str(&text).expect("file still valid");
        assert_eq!(parsed.models.len(), 1);
        assert_eq!(parsed.models[0].id, "m1");
        assert_eq!(parsed.models[0].path, dir.display().to_string());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn duplicate_ids_conflict_never_overwrite() {
        let (state, root) = state_for_add("dup");
        let dir = model_dir_in(&root, "dup-src");
        let path = dir.display().to_string();
        add_model(State(Arc::clone(&state)), Json(body("dup", &path)))
            .await
            .expect("first add");
        // Live duplicate: clear 409, nothing changed.
        let err = add_model(State(Arc::clone(&state)), Json(body("dup", &path)))
            .await
            .expect_err("duplicate rejected");
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "model_exists");
        let text = std::fs::read_to_string(root.join("kiln.toml")).expect("read");
        assert_eq!(
            KilnConfig::parse_str(&text).expect("valid").models.len(),
            1,
            "duplicate must not write a second block"
        );

        // File-only duplicate (hand-edited since boot): also a 409, naming
        // the situation.
        std::fs::write(
            root.join("kiln.toml"),
            format!("{text}\n[[model]]\nid = \"hand-added\"\npath = \"{path}\"\n"),
        )
        .expect("hand edit");
        let err = add_model(State(Arc::clone(&state)), Json(body("hand-added", &path)))
            .await
            .expect_err("file conflict rejected");
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "config_conflict");
        assert!(err.message.contains("restart"), "{}", err.message);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn missing_model_answers_structured_not_downloaded() {
        let (state, root) = state_for_add("notdl");
        let response = add_model(State(Arc::clone(&state)), Json(body("fresh", "stub/tiny")))
            .await
            .expect("structured 409, not an ApiError");
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let json = response_json(response).await;
        assert_eq!(json["error"]["code"], "model_not_downloaded");
        assert_eq!(json["download"]["repo"], "stub/tiny");
        let dest = json["download"]["dest"].as_str().expect("dest");
        assert!(dest.ends_with("models/stub--tiny"), "{dest}");
        // Nothing registered, nothing persisted.
        assert!(state.registry.get("fresh").is_none());
        let text = std::fs::read_to_string(root.join("kiln.toml")).expect("read");
        assert!(!text.contains("[[model]]"), "{text}");

        // Once the files exist at the derived dest, the same request
        // resolves and registers against it.
        model_dir_in(&root.join("models"), "stub--tiny");
        let response = add_model(State(Arc::clone(&state)), Json(body("fresh", "stub/tiny")))
            .await
            .expect("resolves after download");
        assert_eq!(response.status(), StatusCode::CREATED);
        let text = std::fs::read_to_string(root.join("kiln.toml")).expect("read");
        assert!(text.contains("stub--tiny"), "{text}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn unservable_and_invalid_requests_are_400s() {
        let (state, root) = state_for_add("invalid");
        // Explicit rust on an arch the rust worker cannot serve.
        let dir = model_dir_in(&root, "phi-src");
        let mut rust_body = body("phi", &dir.display().to_string());
        rust_body.worker = WorkerKind::Rust;
        let err = add_model(State(Arc::clone(&state)), Json(rust_body))
            .await
            .expect_err("rust-unservable");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);

        // A path that is neither local nor repo-shaped.
        let err = add_model(
            State(Arc::clone(&state)),
            Json(body("ghost", "/no/such/dir")),
        )
        .await
        .expect_err("invalid path");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);

        for empty in [body("", "x/y"), body("id", "")] {
            let err = add_model(State(Arc::clone(&state)), Json(empty))
                .await
                .expect_err("empty field");
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn estimate_prices_local_dirs_against_the_live_budget() {
        let (state, root) = state_for_add("estimate");
        let dir = model_dir_in(&root, "est-src");
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 128]).expect("weights");
        std::fs::write(dir.join("model-2.safetensors"), vec![0u8; 64]).expect("weights");
        let Json(json) = estimate_model(
            State(Arc::clone(&state)),
            Query(EstimateQuery {
                path: dir.display().to_string(),
            }),
        )
        .await
        .expect("estimate");
        assert_eq!(json["source"], "local");
        assert_eq!(json["weights_bytes"], 192);
        assert_eq!(
            json["estimated_bytes"].as_u64(),
            Some(192 + LOAD_OVERHEAD_MARGIN_BYTES)
        );
        assert!(json["budget_bytes"].as_u64().unwrap_or(0) > 0);
        assert_eq!(
            json["headroom_bytes"].as_u64(),
            json["budget_bytes"].as_u64(),
            "nothing loaded: headroom == budget"
        );
        assert_eq!(json["fits"], true);

        let err = estimate_model(
            State(Arc::clone(&state)),
            Query(EstimateQuery {
                path: "/no/such/dir".into(),
            }),
        )
        .await
        .expect_err("invalid estimate path");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn repo_shape_detection() {
        assert!(hf_repo_shaped("mlx-community/Qwen3-0.6B-4bit").is_some());
        assert!(hf_repo_shaped("stub/tiny").is_some());
        for not_repo in [
            "",
            "no-slash",
            "/leading",
            "trailing/",
            "a/b/c",
            "~/models/x",
            "./rel/x",
        ] {
            assert!(hf_repo_shaped(not_repo).is_none(), "{not_repo}");
        }
    }
}
