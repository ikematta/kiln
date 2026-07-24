//! OpenAI-compatible error responses. Every error leaving the gateway has
//! the `{"error": {message, type, param, code}}` shape the client SDKs parse.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use kiln_proto::v1::{Finished, WorkerErrorCode};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub error_type: &'static str,
    pub code: &'static str,
    pub message: String,
    /// Emitted as a `Retry-After` header (whole seconds) on both the OpenAI
    /// and Anthropic envelopes; set by the rate-limit rejections.
    pub retry_after_secs: Option<u64>,
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: "invalid_request",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    pub fn invalid_api_key() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            error_type: "invalid_request_error",
            code: "invalid_api_key",
            message: "Incorrect or missing API key.".into(),
            retry_after_secs: None,
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            code: "model_not_found",
            message: format!("The model '{model}' does not exist or is not configured."),
            retry_after_secs: None,
        }
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: "context_length_exceeded",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// Worker still loading the model: retriable, distinct from a crash.
    pub fn model_loading(model: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "server_error",
            code: "model_loading",
            message: format!("The model '{model}' is still loading; retry shortly."),
            retry_after_secs: None,
        }
    }

    /// Worker mid-unload (eviction or idle TTL): retriable, distinct from
    /// loading — the model may come back only on a later request.
    pub fn model_unloading(model: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "server_error",
            code: "model_unloading",
            message: format!("The model '{model}' is being unloaded; retry shortly."),
            retry_after_secs: None,
        }
    }

    /// Per-request admission rejection (SPEC §2.3/§8.2, Phase 9 part 2):
    /// serving the request would grow the worker past the machine's
    /// remaining memory headroom — the configured budget, or what the OS
    /// can actually grant right now (`denial.constraint`). Retriable —
    /// headroom recovers when a model unloads, usage shrinks, or system
    /// pressure clears.
    pub fn insufficient_memory(model: &str, denial: crate::lifecycle::MemoryDenial) -> Self {
        use crate::lifecycle::AdmitConstraint;
        let bound = match denial.constraint {
            AdmitConstraint::Budget => "of the machine memory budget remain",
            AdmitConstraint::SystemAvailability => {
                "of real system memory remain above the configured floor \
                 (other processes hold the rest)"
            }
            AdmitConstraint::SystemPressure => {
                "can be granted while the OS reports elevated memory pressure"
            }
        };
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "server_error",
            code: "insufficient_memory",
            message: format!(
                "admission rejected: serving this request could grow model '{model}' by \
                 {} bytes (KV pool materialization) but only {} bytes {bound}; \
                 retry later or free memory",
                denial.needed_bytes, denial.headroom_bytes
            ),
            retry_after_secs: None,
        }
    }

    /// Per-key requests-per-minute rejection (SPEC §8.3). OpenAI's
    /// rate-limit shape: `type` names the exhausted scope ("requests"),
    /// `code` is "rate_limit_exceeded", and `Retry-After` says when one
    /// request's worth of budget will have refilled.
    pub fn rate_limited_requests(key: &str, limit_rpm: u32, retry_after_secs: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            error_type: "requests",
            code: "rate_limit_exceeded",
            message: format!(
                "Rate limit reached for API key '{key}': {limit_rpm} requests per minute. \
                 Retry after {retry_after_secs}s."
            ),
            retry_after_secs: Some(retry_after_secs),
        }
    }

    /// Per-key tokens-per-minute rejection (SPEC §8.3): the worst-case
    /// reservation (`prompt + max_tokens`, crate::ratelimit module docs) did
    /// not fit the key's budget. `retry_after_secs: None` means the request
    /// alone exceeds the whole per-minute budget — waiting can never help,
    /// so no `Retry-After` is emitted and the message says what to change.
    pub fn rate_limited_tokens(
        key: &str,
        limit_tpm: u32,
        prompt_tokens: u32,
        max_tokens: u32,
        retry_after_secs: Option<u64>,
    ) -> Self {
        let needed = u64::from(prompt_tokens) + u64::from(max_tokens);
        let message = match retry_after_secs {
            Some(secs) => format!(
                "Rate limit reached for API key '{key}': {limit_tpm} tokens per minute. \
                 This request reserves {needed} tokens ({prompt_tokens} prompt + up to \
                 {max_tokens} completion); unused reservation is refunded when the \
                 response finishes. Retry after {secs}s."
            ),
            None => format!(
                "Request exceeds the rate limit for API key '{key}': it could consume up \
                 to {needed} tokens ({prompt_tokens} prompt + up to {max_tokens} \
                 completion), more than the key's whole budget of {limit_tpm} tokens per \
                 minute. Lower max_tokens or raise the key's tpm."
            ),
        };
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            error_type: "tokens",
            code: "rate_limit_exceeded",
            message,
            retry_after_secs,
        }
    }

    /// Worker died (mid-request or before it): the SPEC §2.2 structured 502
    /// with a retriable code.
    pub fn worker_crashed(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            error_type: "server_error",
            code: "worker_crashed",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// Worker exceeded its restart budget; not retriable without operator
    /// action.
    pub fn worker_failed(model: &str) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            error_type: "server_error",
            code: "worker_failed",
            message: format!(
                "The worker for model '{model}' crashed repeatedly and requires manual reset."
            ),
            retry_after_secs: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            code: "internal_error",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// Admin surface is configured off: no `auth.admin_token_hash` set.
    /// Deliberately fail-closed (unlike API keys, admin endpoints trigger
    /// downloads and subprocesses — SPEC §8.1 says "bearer-token gated").
    pub fn admin_disabled() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            error_type: "invalid_request_error",
            code: "admin_disabled",
            message: "The admin API is disabled: set auth.admin_token_hash in kiln.toml \
                      (hash a token with `kiln-gateway hash-key`)."
                .into(),
            retry_after_secs: None,
        }
    }

    /// An admin action conflicts with the model's current lifecycle state
    /// (e.g. loading a FAILED model, unloading one that is still starting).
    pub fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            error_type: "invalid_request_error",
            code,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// A named admin resource (e.g. a job id) does not exist.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            code: "not_found",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// The kiln-jobs runner is unreachable or errored below the proxy.
    pub fn jobs_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            error_type: "server_error",
            code: "jobs_unavailable",
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// Maps a worker `Finished{finish_reason=ERROR}` event.
    pub fn from_worker_finished(finished: &Finished) -> Self {
        let detail = &finished.error_detail;
        match finished.error_code() {
            WorkerErrorCode::WorkerErrorInvalidRequest => {
                Self::invalid_request(format!("worker rejected request: {detail}"))
            }
            WorkerErrorCode::WorkerErrorCtxOverflow => {
                Self::context_length_exceeded(detail.clone())
            }
            WorkerErrorCode::WorkerErrorGrammarUnsupported
            | WorkerErrorCode::WorkerErrorGrammarCompile => Self {
                status: StatusCode::BAD_REQUEST,
                error_type: "invalid_request_error",
                code: "grammar_unsupported",
                message: detail.clone(),
                retry_after_secs: None,
            },
            WorkerErrorCode::WorkerErrorOomRejected => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error_type: "server_error",
                code: "overloaded",
                message: format!("worker rejected request for memory headroom: {detail}"),
                retry_after_secs: None,
            },
            WorkerErrorCode::WorkerErrorDraining => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error_type: "server_error",
                code: "worker_draining",
                message: "the worker is draining; retry shortly".into(),
                retry_after_secs: None,
            },
            WorkerErrorCode::WorkerErrorInternal | WorkerErrorCode::WorkerErrorUnspecified => {
                Self::internal(format!("worker error: {detail}"))
            }
        }
    }

    /// Maps a failed worker RPC (transport/socket level). During a crash or
    /// restart window the UDS connect fails with `Unavailable`.
    pub fn from_worker_status(status: &tonic::Status) -> Self {
        match status.code() {
            tonic::Code::Unavailable => Self::worker_crashed(
                "the model worker is unavailable (crashed or restarting); retry shortly",
            ),
            tonic::Code::InvalidArgument => Self::invalid_request(status.message().to_string()),
            code => Self::worker_crashed(format!("worker RPC failed: {code}")),
        }
    }

    /// Outcome label for `kiln_chat_completions_total`.
    pub fn outcome(&self) -> &'static str {
        if self.code == "worker_crashed" || self.code == "worker_failed" {
            "worker_crashed"
        } else if self.status.is_client_error() {
            "client_error"
        } else if self.status == StatusCode::SERVICE_UNAVAILABLE {
            "unavailable"
        } else {
            "worker_error"
        }
    }

    /// The `{"error": ...}` JSON object (also used for SSE error events).
    pub fn body(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": null,
                "code": self.code,
            }
        })
    }

    /// The Anthropic error type for this error's status (their taxonomy is
    /// status-keyed; the `anthropic` SDK maps it to a typed exception).
    fn anthropic_error_type(&self) -> &'static str {
        match self.status.as_u16() {
            400 => "invalid_request_error",
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            // Retriable capacity states (model loading, draining, OOM
            // admission rejects) — Anthropic's "try again" type.
            503 | 529 => "overloaded_error",
            _ => "api_error",
        }
    }

    /// The Anthropic `{"type": "error", "error": {...}}` envelope (also the
    /// payload of a mid-stream `event: error` on `/v1/messages` SSE).
    pub fn anthropic_body(&self) -> serde_json::Value {
        json!({
            "type": "error",
            "error": {
                "type": self.anthropic_error_type(),
                "message": self.message,
            }
        })
    }

    /// [`IntoResponse`] with the Anthropic envelope instead of the OpenAI
    /// one; same status code and headers.
    pub fn into_anthropic_response(self) -> Response {
        let retry_after_secs = self.retry_after_secs;
        let response = (self.status, Json(self.anthropic_body())).into_response();
        with_retry_after(response, retry_after_secs)
    }
}

fn with_retry_after(mut response: Response, retry_after_secs: Option<u64>) -> Response {
    if let Some(secs) = retry_after_secs {
        response
            .headers_mut()
            .insert(axum::http::header::RETRY_AFTER, secs.into());
    }
    response
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry_after_secs = self.retry_after_secs;
        let response = (self.status, Json(self.body())).into_response();
        with_retry_after(response, retry_after_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_shape_matches_openai() {
        let err = ApiError::model_not_found("m");
        let body = err.body();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(body["error"]["param"].is_null());
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn anthropic_body_shape() {
        // The `anthropic` SDK keys typed exceptions on this envelope.
        let err = ApiError::invalid_request("bad temperature");
        let body = err.anthropic_body();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "bad temperature");

        assert_eq!(
            ApiError::invalid_api_key().anthropic_body()["error"]["type"],
            "authentication_error"
        );
        assert_eq!(
            ApiError::model_not_found("m").anthropic_body()["error"]["type"],
            "not_found_error"
        );
        assert_eq!(
            ApiError::model_loading("m").anthropic_body()["error"]["type"],
            "overloaded_error"
        );
        assert_eq!(
            ApiError::worker_crashed("x").anthropic_body()["error"]["type"],
            "api_error"
        );
    }

    /// SPEC §8.3: rate-limit rejections are OpenAI's 429 shape (`type` =
    /// exhausted scope, `code` = "rate_limit_exceeded") plus a `Retry-After`
    /// header; the Anthropic envelope maps them to `rate_limit_error`.
    #[test]
    fn rate_limit_shape_and_retry_after_header() {
        let err = ApiError::rate_limited_requests("alice", 60, 17);
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        let body = err.body();
        assert_eq!(body["error"]["type"], "requests");
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
        assert_eq!(err.anthropic_body()["error"]["type"], "rate_limit_error");
        let response = err.into_response();
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&axum::http::HeaderValue::from(17u64))
        );

        let err = ApiError::rate_limited_tokens("alice", 1000, 40, 200, Some(9));
        assert_eq!(err.body()["error"]["type"], "tokens");
        assert_eq!(err.retry_after_secs, Some(9));
        let response = err.into_anthropic_response();
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&axum::http::HeaderValue::from(9u64))
        );

        // A request that can never fit gets no Retry-After — waiting cannot
        // help; the message says what to change instead.
        let err = ApiError::rate_limited_tokens("alice", 100, 40, 200, None);
        assert!(err.message.contains("max_tokens"), "{}", err.message);
        let response = err.into_response();
        assert!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none()
        );
    }

    #[test]
    fn outcomes() {
        assert_eq!(ApiError::invalid_request("x").outcome(), "client_error");
        assert_eq!(ApiError::worker_crashed("x").outcome(), "worker_crashed");
        assert_eq!(ApiError::model_loading("m").outcome(), "unavailable");
        assert_eq!(ApiError::internal("x").outcome(), "worker_error");
    }
}
