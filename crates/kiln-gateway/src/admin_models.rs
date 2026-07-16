//! Admin models + stats surface (SPEC §8.1: `GET/POST /admin/models`
//! list/load/unload/pin, `GET /admin/stats`; SPEC §12 Phase 10 "models
//! table, load/unload/pin, live stats via SSE").
//!
//! Actions are thin translations onto the Phase 9 lifecycle: load/unload
//! return 202 and complete asynchronously in the supervision task — the
//! admin UI (and any operator with `curl`) watches progress through the
//! same status the request path uses, streamed by `GET /admin/stats`
//! (1s-tick SSE snapshots assembled from the lifecycle ledger plus live
//! worker `Health`/`Stats` RPCs over the existing channels).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{HealthRequest, StatsRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::timeout;

use crate::app::AppState;
use crate::config::WorkerKind;
use crate::error::ApiError;
use crate::registry::{ModelEntry, WorkerStatus};

/// SSE snapshot cadence; matches the supervisor's health poll (SPEC §5).
const STATS_TICK: Duration = Duration::from_secs(1);
/// Per-tick RPC deadline: a slow worker costs one stale sample, never a
/// stalled stream.
const STATS_RPC_TIMEOUT: Duration = Duration::from_millis(500);

fn worker_label(kind: WorkerKind) -> &'static str {
    match kind {
        WorkerKind::Rust => "rust",
        WorkerKind::Python => "python",
        // Resolved at registry build; kept total for completeness.
        WorkerKind::Auto => "auto",
    }
}

/// One row of the models table (the static part; live numbers ride SSE).
fn model_json(state: &AppState, entry: &ModelEntry) -> Value {
    json!({
        "id": entry.id,
        "path": entry.model_path.display().to_string(),
        "worker": worker_label(entry.worker_kind),
        "status": entry.status().to_string(),
        "pinned": state.lifecycle.pinned(&entry.id).unwrap_or(entry.config.pinned),
        "ttl_seconds": entry.config.ttl_seconds,
        "usage_bytes": state.lifecycle.model_usage_bytes(&entry.id),
        "created_unix": entry.created_unix,
    })
}

fn memory_json(state: &AppState) -> Value {
    json!({
        "budget_bytes": state.lifecycle.budget_bytes(),
        "used_bytes": state.lifecycle.used_bytes(),
        "reserved_bytes": state.lifecycle.reserved_bytes(),
        "total_bytes": state.lifecycle.total_bytes(),
    })
}

/// `GET /admin/models`: the models table plus the machine memory ledger.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let models: Vec<Value> = state
        .registry
        .iter()
        .map(|entry| model_json(&state, entry))
        .collect();
    Json(json!({ "models": models, "memory": memory_json(&state) }))
}

fn entry_or_404<'a>(state: &'a AppState, id: &str) -> Result<&'a Arc<ModelEntry>, ApiError> {
    state
        .registry
        .get(id)
        .ok_or_else(|| ApiError::not_found(format!("no configured model with id '{id}'")))
}

/// `POST /admin/models/{id}/load`: ask the supervisor to (re)load an
/// unloaded model. 202 = requested; completion is observable via status.
pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let entry = entry_or_404(&state, &id)?;
    let status = entry.status();
    match status {
        WorkerStatus::Unloaded { .. } => {
            state.lifecycle.request_load(&id);
            Ok((
                StatusCode::ACCEPTED,
                Json(json!({"id": id, "action": "load"})),
            )
                .into_response())
        }
        // Already loaded or on its way: idempotent no-op.
        WorkerStatus::Ready | WorkerStatus::Starting | WorkerStatus::Restarting { .. } => Ok(Json(
            json!({"id": id, "action": "none", "status": status.to_string()}),
        )
        .into_response()),
        WorkerStatus::Failed => Err(ApiError::conflict(
            "model_failed",
            format!(
                "model '{id}' exceeded its restart budget and requires a manual \
                 reset (restart the gateway)"
            ),
        )),
        WorkerStatus::Draining | WorkerStatus::Stopped => Err(ApiError::conflict(
            "model_busy",
            format!("model '{id}' is {status}; retry shortly"),
        )),
    }
}

/// `POST /admin/models/{id}/unload`: drain → SIGTERM → SIGKILL via the
/// supervision task (SPEC §2.2). 202 = requested.
pub async fn unload_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let entry = entry_or_404(&state, &id)?;
    let status = entry.status();
    match status {
        // Restarting included: an unload during crash backoff cancels the
        // pending restart (the supervisor handles the race).
        WorkerStatus::Ready | WorkerStatus::Restarting { .. } => {
            if !state.lifecycle.request_unload(&id) {
                return Err(ApiError::internal(format!(
                    "supervision task for model '{id}' is gone"
                )));
            }
            Ok((
                StatusCode::ACCEPTED,
                Json(json!({"id": id, "action": "unload"})),
            )
                .into_response())
        }
        // Already there or already on its way down.
        WorkerStatus::Unloaded { .. } | WorkerStatus::Draining => Ok(Json(
            json!({"id": id, "action": "none", "status": status.to_string()}),
        )
        .into_response()),
        WorkerStatus::Failed => Err(ApiError::conflict(
            "model_failed",
            format!("model '{id}' is failed; nothing is running"),
        )),
        WorkerStatus::Starting | WorkerStatus::Stopped => Err(ApiError::conflict(
            "model_busy",
            format!("model '{id}' is {status}; retry when it settles"),
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinBody {
    pub pinned: bool,
}

/// `POST /admin/models/{id}/pin`: toggle LRU-eviction pinning at runtime.
/// Runtime-only — kiln.toml stays the boot-time source of truth.
pub async fn pin_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<PinBody>,
) -> Result<Response, ApiError> {
    entry_or_404(&state, &id)?;
    if !state.lifecycle.set_pinned(&id, body.pinned) {
        return Err(ApiError::not_found(format!(
            "no lifecycle slot for model '{id}'"
        )));
    }
    Ok(Json(json!({"id": id, "pinned": body.pinned})).into_response())
}

/// Live per-model numbers for one SSE tick: worker `Health` + `Stats` over
/// the entry's existing channel. `None` fields when the worker is not up
/// (or does not implement Stats — the python worker today).
async fn live_worker_json(entry: &ModelEntry) -> (Value, Value) {
    let mut client = WorkerClient::new(entry.channel.clone());
    let health = match timeout(STATS_RPC_TIMEOUT, client.health(HealthRequest {})).await {
        Ok(Ok(resp)) => {
            let health = resp.into_inner();
            let memory = health.memory.unwrap_or_default();
            json!({
                "state": format!("{:?}", health.state()),
                "requests_running": health.requests_running,
                "requests_waiting": health.requests_waiting,
                "uptime_ms": health.uptime_ms,
                "weights_bytes": memory.weights_bytes,
                "kv_pool_allocated_bytes": memory.kv_pool_allocated_bytes,
                "kv_pool_used_bytes": memory.kv_pool_used_bytes,
                "mlx_active_bytes": memory.mlx_active_bytes,
                "mlx_cache_bytes": memory.mlx_cache_bytes,
            })
        }
        _ => Value::Null,
    };
    let stats = match timeout(STATS_RPC_TIMEOUT, client.stats(StatsRequest {})).await {
        Ok(Ok(resp)) => {
            let stats = resp.into_inner();
            json!({
                "requests_total": stats.requests_total,
                "requests_failed": stats.requests_failed,
                "tokens_prefilled_total": stats.tokens_prefilled_total,
                "tokens_generated_total": stats.tokens_generated_total,
                "prefix_tokens_reused_total": stats.prefix_tokens_reused_total,
                "kv_blocks_allocated": stats.kv_blocks_allocated,
                "kv_blocks_free": stats.kv_blocks_free,
                "spec_tokens_proposed_total": stats.spec_tokens_proposed_total,
                "spec_tokens_accepted_total": stats.spec_tokens_accepted_total,
                "engine_steps_total": stats.engine_steps_total,
            })
        }
        _ => Value::Null,
    };
    (health, stats)
}

/// One `event: stats` SSE payload: models table + memory ledger + live
/// worker numbers.
async fn stats_snapshot(state: &AppState) -> Value {
    let mut models = Vec::new();
    for entry in state.registry.iter() {
        let mut row = model_json(state, entry);
        let status = entry.status();
        let (health, stats) = if matches!(status, WorkerStatus::Ready | WorkerStatus::Draining) {
            live_worker_json(entry).await
        } else {
            (Value::Null, Value::Null)
        };
        row["health"] = health;
        row["stats"] = stats;
        models.push(row);
    }
    json!({ "models": models, "memory": memory_json(state) })
}

/// `GET /admin/stats`: SSE stream of 1s-tick snapshots. The browser
/// `EventSource` API cannot set an `Authorization` header, so the admin UI
/// consumes this with a streaming `fetch` instead — same frames.
pub async fn stats_sse(State(state): State<Arc<AppState>>) -> Response {
    let stream = async_stream::stream! {
        loop {
            let snapshot = stats_snapshot(&state).await;
            yield Ok::<Event, Infallible>(
                Event::default().event("stats").data(snapshot.to_string()),
            );
            tokio::time::sleep(STATS_TICK).await;
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Auth;
    use crate::config::KilnConfig;
    use crate::lifecycle::{Command, Lifecycle};
    use crate::metrics::Metrics;
    use crate::registry::{Registry, UnloadReason};
    use std::path::PathBuf;
    use tokio::sync::{mpsc, watch};

    /// A one-model AppState with NO supervision tasks: the returned watch
    /// sender drives the status by hand and the command receiver observes
    /// what the handlers asked the (absent) supervisor to do.
    fn state_with_model() -> (
        Arc<AppState>,
        watch::Sender<WorkerStatus>,
        mpsc::UnboundedReceiver<Command>,
        PathBuf,
    ) {
        let dir = std::env::temp_dir().join(format!(
            "kiln-admin-models-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        std::fs::write(dir.join("config.json"), b"{\"model_type\": \"phi3\"}").expect("config");

        let mut config = KilnConfig::default();
        config.models.push(crate::config::ModelConfig {
            id: "m".into(),
            path: dir.to_string_lossy().into_owned(),
            worker: WorkerKind::Python,
            pinned: false,
            ttl_seconds: 0,
            speculative: None,
        });
        let metrics = Arc::new(Metrics::new().expect("metrics"));
        let (registry, mut senders) = Registry::from_config(&config).expect("registry");
        let registry = Arc::new(registry);
        let (lifecycle, mut receivers) =
            Lifecycle::new(&config, &registry, Arc::clone(&metrics)).expect("lifecycle");
        let state = Arc::new(AppState {
            registry,
            lifecycle: Arc::new(lifecycle),
            metrics,
            auth: Auth::from_config(&config.auth).expect("auth"),
            jobs: crate::admin::JobsProxy::external(PathBuf::from("/tmp/kiln-admin-models.sock"))
                .expect("proxy"),
        });
        (state, senders.remove(0), receivers.remove(0), dir)
    }

    #[tokio::test]
    async fn list_reports_status_pin_and_memory() {
        let (state, status_tx, _cmd_rx, dir) = state_with_model();
        let Json(body) = list_models(State(Arc::clone(&state))).await;
        assert_eq!(body["models"][0]["id"], "m");
        assert_eq!(body["models"][0]["status"], "starting");
        assert_eq!(body["models"][0]["pinned"], false);
        assert_eq!(body["models"][0]["worker"], "python");
        assert!(body["memory"]["budget_bytes"].as_u64().unwrap_or(0) > 0);

        status_tx.send_replace(WorkerStatus::Ready);
        let Json(body) = list_models(State(state)).await;
        assert_eq!(body["models"][0]["status"], "ready");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_and_unload_translate_status_to_supervisor_commands() {
        let (state, status_tx, mut cmd_rx, dir) = state_with_model();

        // Starting: load is an idempotent no-op, unload a 409.
        let response = load_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect("no-op load");
        assert_eq!(response.status(), StatusCode::OK);
        let err = unload_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect_err("cannot unload mid-start");
        assert_eq!(err.status, StatusCode::CONFLICT);
        assert_eq!(err.code, "model_busy");

        // Unloaded: load sends Command::Load.
        status_tx.send_replace(WorkerStatus::Unloaded {
            reason: UnloadReason::IdleTtl,
        });
        let response = load_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect("load accepted");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert!(matches!(cmd_rx.try_recv(), Ok(Command::Load)));

        // Ready: unload sends Command::Unload{Admin}.
        status_tx.send_replace(WorkerStatus::Ready);
        let response = unload_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect("unload accepted");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        match cmd_rx.try_recv() {
            Ok(Command::Unload { reason, .. }) => assert_eq!(reason, UnloadReason::Admin),
            other => panic!("expected Unload command, got {:?}", other.is_ok()),
        }

        // Failed: both conflict, naming the manual reset.
        status_tx.send_replace(WorkerStatus::Failed);
        let err = load_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect_err("failed model refuses load");
        assert_eq!(err.code, "model_failed");
        assert!(err.message.contains("manual reset"), "{}", err.message);
        let err = unload_model(State(Arc::clone(&state)), Path("m".into()))
            .await
            .expect_err("failed model refuses unload");
        assert_eq!(err.code, "model_failed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pin_toggles_and_unknown_models_404() {
        let (state, _status_tx, _cmd_rx, dir) = state_with_model();
        let response = pin_model(
            State(Arc::clone(&state)),
            Path("m".into()),
            Json(PinBody { pinned: true }),
        )
        .await
        .expect("pin");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.lifecycle.pinned("m"), Some(true));
        let Json(body) = list_models(State(Arc::clone(&state))).await;
        assert_eq!(body["models"][0]["pinned"], true);

        for err in [
            load_model(State(Arc::clone(&state)), Path("ghost".into()))
                .await
                .expect_err("404"),
            unload_model(State(Arc::clone(&state)), Path("ghost".into()))
                .await
                .expect_err("404"),
            pin_model(
                State(Arc::clone(&state)),
                Path("ghost".into()),
                Json(PinBody { pinned: true }),
            )
            .await
            .expect_err("404"),
        ] {
            assert_eq!(err.status, StatusCode::NOT_FOUND);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn stats_snapshot_carries_nulls_for_down_workers() {
        let (state, _status_tx, _cmd_rx, dir) = state_with_model();
        // Starting: no RPCs are attempted, health/stats are null, the
        // ledger numbers are present anyway.
        let snapshot = stats_snapshot(&state).await;
        assert_eq!(snapshot["models"][0]["id"], "m");
        assert!(snapshot["models"][0]["health"].is_null());
        assert!(snapshot["models"][0]["stats"].is_null());
        assert!(snapshot["memory"]["budget_bytes"].as_u64().unwrap_or(0) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
