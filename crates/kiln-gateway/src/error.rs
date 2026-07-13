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
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: "invalid_request",
            message: message.into(),
        }
    }

    pub fn invalid_api_key() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            error_type: "invalid_request_error",
            code: "invalid_api_key",
            message: "Incorrect or missing API key.".into(),
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            code: "model_not_found",
            message: format!("The model '{model}' does not exist or is not configured."),
        }
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: "context_length_exceeded",
            message: message.into(),
        }
    }

    /// Worker still loading the model: retriable, distinct from a crash.
    pub fn model_loading(model: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error_type: "server_error",
            code: "model_loading",
            message: format!("The model '{model}' is still loading; retry shortly."),
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
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            code: "internal_error",
            message: message.into(),
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
            },
            WorkerErrorCode::WorkerErrorOomRejected => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error_type: "server_error",
                code: "overloaded",
                message: format!("worker rejected request for memory headroom: {detail}"),
            },
            WorkerErrorCode::WorkerErrorDraining => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                error_type: "server_error",
                code: "worker_draining",
                message: "the worker is draining; retry shortly".into(),
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
    /// one; same status code.
    pub fn into_anthropic_response(self) -> Response {
        (self.status, Json(self.anthropic_body())).into_response()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
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

    #[test]
    fn outcomes() {
        assert_eq!(ApiError::invalid_request("x").outcome(), "client_error");
        assert_eq!(ApiError::worker_crashed("x").outcome(), "worker_crashed");
        assert_eq!(ApiError::model_loading("m").outcome(), "unavailable");
        assert_eq!(ApiError::internal("x").outcome(), "worker_error");
    }
}
