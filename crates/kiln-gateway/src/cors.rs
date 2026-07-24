//! Opt-in CORS for browser clients (SPEC §8.3, `server.cors_origins`).
//!
//! # Policy
//!
//! The gateway historically sent no CORS headers, which means browser JS
//! on any other origin cannot call it — the fetch itself is blocked by
//! the browser, not by Kiln. That stays the shipped default: with
//! `cors_origins` empty, [`layer`] returns `None` and the router carries
//! no CORS machinery at all. Listing origins opts exactly those pages
//! into cross-origin access; the single entry `"*"` is the explicit
//! wildcard (config validation forbids mixing it with specific origins).
//!
//! # Shape of the layer
//!
//! - **Origins** are the security boundary: exact matches against the
//!   browser's `Origin` header, config entries ASCII-lowercased once at
//!   startup (browsers normalize scheme and host to lowercase).
//! - **Methods and headers mirror the preflight request** rather than
//!   enumerating the API surface. Once an origin is allowlisted there is
//!   nothing to gain by second-guessing which header it sends
//!   (`authorization`, `x-api-key`, `anthropic-version`, ...), and a
//!   maintained list would silently desync from new routes or SDK header
//!   conventions — the naive-CORS failure mode where preflights break
//!   while simple requests keep working.
//! - **Credentials stay off** (the tower-http default). Kiln's auth is
//!   bearer/`x-api-key` request headers, which cross-origin `fetch`
//!   sends without credentials mode; cookies are not part of the API.
//!   This is also what makes the `"*"` opt-in coherent — wildcard +
//!   credentials is forbidden by the CORS spec (tower-http panics on
//!   that combination).
//! - **`x-request-id` and `Retry-After` are exposed**: browsers let page
//!   JS read only safelisted response headers, and these two are the
//!   ones a client can act on (error correlation, 429 backoff).
//!
//! # Placement in the router
//!
//! [`crate::app::router`] applies this layer *inside* the `observe`
//! middleware (preflights get request ids, logs, and
//! `kiln_http_requests_total{method="OPTIONS",...}` — the e2e suite
//! asserts on that counter) but *outside* every auth `route_layer`:
//! tower-http answers a preflight without calling the inner service, and
//! it must, because browsers send no `Authorization` header on the
//! preflight `OPTIONS` — auth-first ordering would 401 every preflight,
//! the other classic naive-CORS breakage.

use std::time::Duration;

use axum::http::{HeaderName, header};
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use crate::config::ServerConfig;

/// Preflight results change only when the config changes, so let
/// browsers cache them (they clamp as they see fit; Chrome caps at 2h).
const MAX_AGE: Duration = Duration::from_secs(3600);

/// Builds the CORS layer from `[server]`. `None` when `cors_origins` is
/// empty — the default, meaning no CORS headers are sent at all.
pub fn layer(server: &ServerConfig) -> Option<CorsLayer> {
    if server.cors_origins.is_empty() {
        return None;
    }
    let origin = if server.cors_origins.iter().any(|entry| entry == "*") {
        AllowOrigin::any()
    } else {
        AllowOrigin::list(
            server
                .cors_origins
                .iter()
                // Config validation guarantees ASCII with no separators,
                // so lowercasing is byte-safe and parsing cannot fail.
                .filter_map(|entry| entry.to_ascii_lowercase().parse().ok()),
        )
    };
    Some(
        CorsLayer::new()
            .allow_origin(origin)
            .allow_methods(AllowMethods::mirror_request())
            .allow_headers(AllowHeaders::mirror_request())
            .expose_headers([HeaderName::from_static("x-request-id"), header::RETRY_AFTER])
            .max_age(MAX_AGE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Method, Request, Response, StatusCode};
    use axum::middleware::{self, Next};
    use axum::routing::post;
    use tower::ServiceExt;

    const ALLOWED: &str = "http://localhost:5173";

    fn server(origins: &[&str]) -> ServerConfig {
        ServerConfig {
            cors_origins: origins.iter().map(|s| s.to_string()).collect(),
            ..ServerConfig::default()
        }
    }

    /// Mini app with the production layer nesting: an auth-shaped
    /// `route_layer` that 401s anything without `Authorization`, wrapped
    /// by the CORS layer — exactly the ordering question preflights
    /// hinge on, since browsers send no credentials on `OPTIONS`.
    fn app(origins: &[&str]) -> Router {
        let router = Router::new()
            .route("/v1/chat/completions", post(|| async { "handled" }))
            .route_layer(middleware::from_fn(
                |request: Request<Body>, next: Next| async move {
                    if request.headers().contains_key(header::AUTHORIZATION) {
                        next.run(request).await
                    } else {
                        Response::builder()
                            .status(StatusCode::UNAUTHORIZED)
                            .body(Body::empty())
                            .unwrap()
                    }
                },
            ));
        match layer(&server(origins)) {
            Some(cors) => router.layer(cors),
            None => router,
        }
    }

    fn preflight(origin: &str) -> Request<Body> {
        Request::builder()
            .method(Method::OPTIONS)
            .uri("/v1/chat/completions")
            .header(header::ORIGIN, origin)
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(
                header::ACCESS_CONTROL_REQUEST_HEADERS,
                "authorization,content-type",
            )
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn empty_config_builds_no_layer() {
        assert!(layer(&ServerConfig::default()).is_none());
    }

    #[tokio::test]
    async fn preflight_from_allowed_origin_short_circuits_before_auth() {
        let response = app(&[ALLOWED]).oneshot(preflight(ALLOWED)).await.unwrap();
        // 200, not the route_layer's 401: the preflight never reached auth.
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers();
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            ALLOWED,
            "allow-origin must echo the configured origin"
        );
        // Mirrored, not enumerated: what the browser asked for comes back.
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_METHODS], "POST");
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization,content-type"
        );
        assert_eq!(headers[header::ACCESS_CONTROL_MAX_AGE], "3600");
    }

    #[tokio::test]
    async fn preflight_from_unknown_origin_gets_no_allow_origin() {
        let response = app(&[ALLOWED])
            .oneshot(preflight("http://evil.example"))
            .await
            .unwrap();
        // tower-http still answers the OPTIONS, but without allow-origin
        // the browser blocks the real request client-side.
        assert!(
            !response
                .headers()
                .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            "unknown origin must not be allowed"
        );
    }

    #[tokio::test]
    async fn actual_request_gets_allow_origin_and_exposed_headers() {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(header::ORIGIN, ALLOWED)
            .header(header::AUTHORIZATION, "Bearer k")
            .body(Body::empty())
            .unwrap();
        let response = app(&[ALLOWED]).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "the handler ran");
        let headers = response.headers();
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], ALLOWED);
        let exposed = headers[header::ACCESS_CONTROL_EXPOSE_HEADERS]
            .to_str()
            .unwrap();
        assert!(exposed.contains("x-request-id"), "{exposed}");
        assert!(exposed.contains("retry-after"), "{exposed}");
    }

    /// Browsers normalize `Origin` to lowercase; a mixed-case config
    /// entry must still match.
    #[tokio::test]
    async fn config_origins_are_lowercased_to_match_browser_origins() {
        let response = app(&["http://LocalHost:5173"])
            .oneshot(preflight(ALLOWED))
            .await
            .unwrap();
        assert_eq!(
            response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            ALLOWED
        );
    }

    #[tokio::test]
    async fn lone_wildcard_allows_any_origin() {
        let response = app(&["*"])
            .oneshot(preflight("http://anything.example"))
            .await
            .unwrap();
        assert_eq!(response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
    }
}
