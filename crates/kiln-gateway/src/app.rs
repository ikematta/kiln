//! Router assembly, shared state, and the observability middleware
//! (request-id, structured request logs, HTTP metrics).

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::auth::Auth;
use crate::error::ApiError;
use crate::lifecycle::Lifecycle;
use crate::metrics::Metrics;
use crate::openai::ModelObject;
use crate::registry::{Registry, WorkerStatus};

pub struct AppState {
    pub registry: Arc<Registry>,
    pub lifecycle: Arc<Lifecycle>,
    pub metrics: Arc<Metrics>,
    pub auth: Auth,
    pub jobs: crate::admin::JobsProxy,
}

/// Per-request UUIDv7, generated in [`observe`]; reused as the worker
/// `request_id` and echoed back as `x-request-id`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

pub fn router(state: Arc<AppState>) -> Router {
    let api = Router::new()
        .route("/v1/chat/completions", post(crate::chat::chat_completions))
        .route("/v1/completions", post(crate::completions::completions))
        .route("/v1/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            crate::auth::require_api_key,
        ));
    // Anthropic surface: same keys, `x-api-key` convention and Anthropic
    // error envelope (SPEC §8.1).
    let anthropic_api = Router::new()
        .route("/v1/messages", post(crate::messages::messages))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            crate::auth::require_api_key_anthropic,
        ));
    // Admin surface (SPEC §8.1): separate bearer token, fail-closed when
    // unconfigured. Jobs proxy (part 1) + models/stats (part 2).
    let admin = Router::new()
        .route("/admin/jobs/download", post(crate::admin::submit_download))
        .route("/admin/jobs/quantize", post(crate::admin::submit_quantize))
        .route("/admin/jobs", get(crate::admin::list_jobs))
        .route("/admin/jobs/{id}", get(crate::admin::get_job))
        .route("/admin/models", get(crate::admin_models::list_models))
        .route(
            "/admin/models/{id}/load",
            post(crate::admin_models::load_model),
        )
        .route(
            "/admin/models/{id}/unload",
            post(crate::admin_models::unload_model),
        )
        .route(
            "/admin/models/{id}/pin",
            post(crate::admin_models::pin_model),
        )
        .route("/admin/stats", get(crate::admin_models::stats_sse))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            crate::auth::require_admin,
        ));

    Router::new()
        .merge(api)
        .merge(anthropic_api)
        .merge(admin)
        // Unauthenticated by design: the gateway binds localhost by default
        // (SPEC §8.1); operators exposing it terminate auth upstream. The
        // /ui shell is static code — all data behind it rides the
        // bearer-gated /admin API above.
        .route("/ui", get(crate::ui::index))
        .route("/ui/", get(crate::ui::index))
        .route("/ui/{*path}", get(crate::ui::asset))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_endpoint))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), observe))
        .with_state(state)
}

async fn observe(
    State(state): State<Arc<AppState>>,
    matched: Option<MatchedPath>,
    mut request: Request,
    next: Next,
) -> Response {
    let started = Instant::now();
    let method = request.method().to_string();
    // Matched route pattern, not the raw URI: bounded metric cardinality.
    let path = matched
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "(unmatched)".to_string());
    let request_id = uuid::Uuid::now_v7().to_string();
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let mut response = next.run(request).await;

    let status = response.status().as_u16();
    let elapsed = started.elapsed();
    state
        .metrics
        .http_requests_total
        .with_label_values(&[&method, &path, &status.to_string()])
        .inc();
    state
        .metrics
        .http_request_duration_seconds
        .with_label_values(&[&path])
        .observe(elapsed.as_secs_f64());
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    tracing::info!(target: "kiln::http",
        method = %method, path = %path, status,
        duration_ms = elapsed.as_millis() as u64,
        request_id = %request_id,
        "request handled");
    response
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// 200 once every configured worker has settled: Ready, or deliberately
/// Unloaded (idle TTL / evicted / over budget — SPEC §2.2 lifecycle states,
/// reloaded on demand). Everything transitional or broken is 503 with
/// per-model states.
async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    let mut models = serde_json::Map::new();
    let mut all_ready = true;
    for entry in state.registry.iter() {
        let status = entry.status();
        all_ready &= matches!(status, WorkerStatus::Ready | WorkerStatus::Unloaded { .. });
        models.insert(entry.id.clone(), json!(status.to_string()));
    }
    let status = if all_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = json!({
        "status": if all_ready { "ready" } else { "unavailable" },
        "models": models,
    });
    (status, Json(body)).into_response()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let data: Vec<ModelObject> = state
        .registry
        .iter()
        .map(|entry| ModelObject {
            id: entry.id.clone(),
            object: "model",
            created: entry.created_unix,
            owned_by: "kiln",
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> Response {
    match state.metrics.encode() {
        Ok(text) => ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], text).into_response(),
        Err(err) => ApiError::internal(format!("metrics encoding failed: {err}")).into_response(),
    }
}
