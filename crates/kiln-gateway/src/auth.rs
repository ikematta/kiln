//! API-key auth (SPEC §8.3): keys live in config as argon2 hashes, presented
//! as `Authorization: Bearer <key>` (OpenAI convention) or `x-api-key`
//! (Anthropic convention — the `anthropic` SDK sends only this header).
//! Verified keys are cached (by sha256 of the raw key — never the raw key
//! itself) so the argon2 cost is paid once per key, not per request.

use std::collections::HashMap;
use std::sync::Arc;

use argon2::password_hash::PasswordHash;
use argon2::{Argon2, PasswordVerifier};
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::app::AppState;
use crate::config::AuthConfig;
use crate::error::ApiError;

pub struct Auth {
    /// (key name, PHC hash string) pairs; only entries whose hash parses.
    keys: Vec<(String, String)>,
    /// sha256(raw key) → key name for keys that already verified.
    cache: RwLock<HashMap<[u8; 32], String>>,
}

impl Auth {
    /// Builds the verifier from config. Entries with empty or unparseable
    /// hashes are skipped with a warning. With no usable keys, auth is
    /// DISABLED (the gateway binds localhost by default; real deployments
    /// must configure keys).
    pub fn from_config(config: &AuthConfig) -> Self {
        let mut keys = Vec::new();
        for entry in &config.api_keys {
            if entry.key_hash.is_empty() {
                tracing::warn!(key = %entry.name, "api key has empty key_hash; ignoring");
                continue;
            }
            if let Err(err) = PasswordHash::new(&entry.key_hash) {
                tracing::warn!(key = %entry.name, error = %err,
                    "api key hash is not a valid PHC string; ignoring");
                continue;
            }
            keys.push((entry.name.clone(), entry.key_hash.clone()));
        }
        if keys.is_empty() {
            tracing::warn!(
                "no usable API keys configured; /v1 endpoints are UNAUTHENTICATED \
                 (add [[auth.api_keys]] entries; hash keys with `kiln-gateway hash-key`)"
            );
        }
        Self {
            keys,
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Verifies a presented key; returns the key's configured name.
    pub async fn verify(&self, presented: &str) -> Option<String> {
        let digest: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
        if let Some(name) = self.cache.read().await.get(&digest) {
            return Some(name.clone());
        }
        // argon2 verification is CPU-heavy by design; keep it off the
        // async workers.
        let presented = presented.to_string();
        let candidates = self.keys.clone();
        let verified = tokio::task::spawn_blocking(move || {
            candidates.into_iter().find_map(|(name, phc)| {
                let parsed = PasswordHash::new(&phc).ok()?;
                Argon2::default()
                    .verify_password(presented.as_bytes(), &parsed)
                    .ok()
                    .map(|()| name)
            })
        })
        .await
        .ok()??;
        self.cache.write().await.insert(digest, verified.clone());
        Some(verified)
    }
}

/// The presented key: `Authorization: Bearer` or Anthropic's `x-api-key`.
/// Both are accepted on every authenticated route — the credential set is
/// one, only the SDK conventions differ.
fn presented_key(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            request
                .headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
        })
}

/// Ok(request) when authenticated (or auth is disabled), Err(()) otherwise;
/// the route-shape-specific middlewares below own the error envelope.
async fn authenticate(state: &AppState, request: Request) -> Result<Request, ()> {
    if !state.auth.enabled() {
        return Ok(request);
    }
    let Some(presented) = presented_key(&request) else {
        return Err(());
    };
    match state.auth.verify(presented).await {
        Some(name) => {
            tracing::debug!(api_key = %name, "authenticated");
            Ok(request)
        }
        None => Err(()),
    }
}

/// Route-layer middleware for the OpenAI-shaped `/v1/*` endpoints.
pub async fn require_api_key(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    match authenticate(&state, request).await {
        Ok(request) => next.run(request).await,
        Err(()) => ApiError::invalid_api_key().into_response(),
    }
}

/// Route-layer middleware for `/v1/messages`: same credentials, Anthropic
/// error envelope (the `anthropic` SDK parses `{"type": "error", ...}`).
pub async fn require_api_key_anthropic(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    match authenticate(&state, request).await {
        Ok(request) => next.run(request).await,
        Err(()) => ApiError::invalid_api_key().into_anthropic_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiKeyConfig;

    fn hash(key: &str) -> String {
        use argon2::PasswordHasher;
        use argon2::password_hash::{SaltString, rand_core::OsRng};
        Argon2::default()
            .hash_password(key.as_bytes(), &SaltString::generate(&mut OsRng))
            .expect("hashing works")
            .to_string()
    }

    #[tokio::test]
    async fn verifies_and_caches() {
        let auth = Auth::from_config(&AuthConfig {
            admin_token_hash: None,
            api_keys: vec![ApiKeyConfig {
                name: "alice".into(),
                key_hash: hash("s3cret"),
                rpm: None,
                tpm: None,
            }],
        });
        assert!(auth.enabled());
        assert_eq!(auth.verify("s3cret").await.as_deref(), Some("alice"));
        // Second call hits the cache (still must succeed).
        assert_eq!(auth.verify("s3cret").await.as_deref(), Some("alice"));
        assert_eq!(auth.verify("wrong").await, None);
    }

    #[tokio::test]
    async fn empty_and_malformed_hashes_disable_auth() {
        let auth = Auth::from_config(&AuthConfig {
            admin_token_hash: None,
            api_keys: vec![
                ApiKeyConfig {
                    name: "empty".into(),
                    key_hash: String::new(),
                    rpm: None,
                    tpm: None,
                },
                ApiKeyConfig {
                    name: "garbage".into(),
                    key_hash: "not-a-phc-string".into(),
                    rpm: None,
                    tpm: None,
                },
            ],
        });
        assert!(!auth.enabled());
    }
}
