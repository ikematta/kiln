//! Embedded admin SPA (SPEC §3: SvelteKit static adapter served from
//! embedded assets via rust-embed; SPEC §12 Phase 10).
//!
//! Served unauthenticated at `/ui`: the shell is static code with no
//! secrets — every piece of data it renders comes from the bearer-gated
//! `/admin/*` API, and the page itself surfaces the API's 401/403
//! messages verbatim (including the fail-closed "admin API is disabled"
//! 403 that names the fix).
//!
//! Debug builds read `admin/build/` from disk at runtime; release builds
//! embed the files (rust-embed's default), keeping the SPEC §1.1 "single
//! static gateway binary". A checkout that never ran `npm --prefix admin
//! run build` gets a 503 naming that command instead of a bare 404.

use axum::extract::{Path, RawQuery};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "../../admin/build/"]
struct Assets;

fn not_built() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        "admin UI not built: run `npm --prefix admin install && npm --prefix admin run build` \
         and restart (release builds embed it at compile time)",
    )
        .into_response()
}

fn serve(path: &str) -> Response {
    let Some(file) = Assets::get(path) else {
        return if Assets::get("index.html").is_none() {
            not_built()
        } else {
            (StatusCode::NOT_FOUND, "no such asset").into_response()
        };
    };
    // SvelteKit content-hashes everything under _app/immutable; the shell
    // itself must always revalidate so a rebuilt UI is picked up.
    let cache = if path.contains("immutable") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        [
            (header::CONTENT_TYPE, file.metadata.mimetype()),
            (header::CACHE_CONTROL, cache),
        ],
        file.data,
    )
        .into_response()
}

/// `GET /ui` (no trailing slash): 308 to `/ui/`. The shell's asset links
/// are relative (`./_app/...`), so they only resolve to `/ui/_app/...`
/// when the document URL ends in a slash — served directly at `/ui` the
/// browser requests `/_app/...`, every asset 404s, and the page renders
/// blank. Redirecting before serving is the fix, not an optimization.
pub async fn redirect_to_slash(RawQuery(query): RawQuery) -> Redirect {
    match query {
        Some(query) => Redirect::permanent(&format!("/ui/?{query}")),
        None => Redirect::permanent("/ui/"),
    }
}

/// `GET /ui/`: the prerendered shell.
pub async fn index() -> Response {
    serve("index.html")
}

/// `GET /ui/{*path}`: build assets (`_app/...`).
pub async fn asset(Path(path): Path<String>) -> Response {
    serve(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Presence-agnostic: whether or not the SPA has been built, /ui must
    /// answer something actionable — the shell, or the 503 naming the
    /// build command. (The full UI is exercised by the browser e2e test.)
    #[tokio::test]
    async fn ui_root_is_shell_or_actionable_503() {
        let response = index().await;
        match response.status() {
            StatusCode::OK => {
                let content_type = response
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                assert!(content_type.contains("html"), "{content_type}");
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
                    .await
                    .expect("body");
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("npm --prefix admin"), "{text}");
            }
            other => panic!("unexpected status {other}"),
        }
    }

    #[tokio::test]
    async fn unknown_asset_is_404_when_built() {
        let response = asset(Path("no/such/file.js".into())).await;
        assert!(
            response.status() == StatusCode::NOT_FOUND
                || response.status() == StatusCode::SERVICE_UNAVAILABLE
        );
    }

    /// /ui without the trailing slash must redirect, never serve the
    /// shell in place: the shell's relative asset links only resolve
    /// under /ui/. (The full chain — redirect followed, assets fetched —
    /// is asserted by the e2e regression test.)
    #[tokio::test]
    async fn ui_bare_redirects_to_slash() {
        let response = redirect_to_slash(RawQuery(None)).await.into_response();
        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(
            response
                .headers()
                .get(header::LOCATION)
                .and_then(|v| v.to_str().ok()),
            Some("/ui/")
        );
    }

    #[tokio::test]
    async fn ui_bare_redirect_preserves_query() {
        let response = redirect_to_slash(RawQuery(Some("a=1&b=2".into())))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(
            response
                .headers()
                .get(header::LOCATION)
                .and_then(|v| v.to_str().ok()),
            Some("/ui/?a=1&b=2")
        );
    }
}
