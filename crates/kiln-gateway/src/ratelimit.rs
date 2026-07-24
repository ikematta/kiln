//! Per-key rate limiting (SPEC §8.3): token buckets for requests/minute and
//! tokens/minute, configured per API key in kiln.toml
//! (`[[auth.api_keys]] ... rpm = 600, tpm = 500000`).
//!
//! # Enforcement model
//!
//! - **rpm** is checked before the request is processed: a `tower`
//!   route-layer middleware ([`enforce_rpm`] / [`enforce_rpm_anthropic`],
//!   same `axum::middleware::from_fn_with_state` pattern as auth) takes one
//!   token from the key's request bucket. The auth middleware stamps the
//!   request with the verified key's [`RateLimitHandle`] so this layer and
//!   the handlers know which buckets apply. All authenticated `/v1` routes
//!   count, including `GET /v1/models`.
//! - **tpm** cannot be fully checked up front — completion tokens are
//!   unknown until generation ends. The gateway RESERVES the worst case
//!   (`prompt_tokens + max_tokens`) before submitting to the worker, then
//!   RECONCILES when the request settles by refunding the unused remainder
//!   (`reserved − client-visible usage`) — the same reserve-then-reconcile
//!   shape as the Phase 9 memory-admission ledger. Consequences, all
//!   deliberate:
//!   - Concurrent requests can never over-commit a budget: a bucket's
//!     check-and-take is one atomic operation under its lock.
//!   - A request whose worst case exceeds the tpm limit outright is
//!     rejected immediately with an actionable message (lower
//!     `max_tokens`), because no amount of waiting makes it admissible —
//!     and the failed attempt consumes nothing.
//!   - Conservative in flight: a large `max_tokens` holds budget until the
//!     response settles; the refund then restores it.
//!   - Abandoned requests forfeit: a client that disconnects mid-stream, or
//!     a request that errors after submission, never settles, so its full
//!     reservation stays charged until the bucket refills. Refunding those
//!     would let a client burn GPU without ever depleting its budget; the
//!     forfeit self-heals within a minute.
//!   - Usage is the client-visible count (`Finished` after
//!     `TextPipeline::apply_usage`), identical across worker kinds — never
//!     the worker's cancel-overshoot total.
//!
//! Buckets start full and refill continuously (limit/60 per second). State
//! is in-memory: a gateway restart forgives spent budget (fine for a
//! single-host inference server — there is no distributed state). Keys
//! without `rpm`/`tpm`, and all requests when auth is disabled, are
//! unlimited. Admin endpoints are not rate-limited (separate operator
//! token). Rejections are OpenAI's rate-limit error shape (`"type":
//! "requests" | "tokens"`, `"code": "rate_limit_exceeded"`) with a
//! `Retry-After` header, or the Anthropic `rate_limit_error` envelope on
//! `/v1/messages`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::app::AppState;
use crate::config::AuthConfig;
use crate::error::ApiError;

/// All per-key limiters, keyed by the configured key name. Built once at
/// startup; entries exist only for keys with at least one limit.
pub struct RateLimiter {
    keys: HashMap<String, Arc<KeyLimiter>>,
}

impl RateLimiter {
    pub fn from_config(config: &AuthConfig) -> Self {
        let keys = config
            .api_keys
            .iter()
            .filter(|key| key.rpm.is_some() || key.tpm.is_some())
            .map(|key| {
                (
                    key.name.clone(),
                    Arc::new(KeyLimiter {
                        name: key.name.clone(),
                        rpm: key.rpm.map(Limit::per_minute),
                        tpm: key.tpm.map(Limit::per_minute),
                    }),
                )
            })
            .collect();
        Self { keys }
    }

    /// The limiter for a verified key name; `None` means unlimited.
    pub fn handle(&self, key_name: &str) -> Option<Arc<KeyLimiter>> {
        self.keys.get(key_name).cloned()
    }
}

/// Stamped into request extensions by the auth middleware when the
/// authenticated key has any limit configured; read by the rpm middleware
/// and by the completion handlers (tpm reservation).
#[derive(Clone)]
pub struct RateLimitHandle(pub Arc<KeyLimiter>);

/// One key's buckets.
#[derive(Debug)]
pub struct KeyLimiter {
    name: String,
    rpm: Option<Limit>,
    tpm: Option<Limit>,
}

/// A configured per-minute limit and its bucket.
#[derive(Debug)]
struct Limit {
    per_minute: u32,
    bucket: Bucket,
}

impl Limit {
    fn per_minute(limit: u32) -> Self {
        Self {
            per_minute: limit,
            bucket: Bucket::new(f64::from(limit), f64::from(limit) / 60.0),
        }
    }
}

/// Why a take was refused.
#[derive(Debug)]
pub(crate) struct Denied {
    /// Seconds until the bucket will have refilled enough for this take
    /// (>= 1). Meaningless when `exceeds_capacity`.
    pub(crate) retry_after_secs: u64,
    /// The amount is larger than the whole bucket: this take can NEVER
    /// succeed at the configured limit.
    pub(crate) exceeds_capacity: bool,
}

/// Continuous-refill token bucket. The check-and-take under one lock is
/// what makes concurrent admission race-free: two racers serialize on the
/// mutex, and the loser sees the winner's deduction (the over-admission
/// class of bug the Phase 9 reservation ledger once had).
#[derive(Debug)]
struct Bucket {
    capacity: f64,
    rate_per_sec: f64,
    state: Mutex<BucketState>,
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    updated: Instant,
}

impl Bucket {
    fn new(capacity: f64, rate_per_sec: f64) -> Self {
        Self {
            capacity,
            rate_per_sec,
            state: Mutex::new(BucketState {
                tokens: capacity,
                updated: Instant::now(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BucketState> {
        // A poisoned bucket mutex means a panic while holding it; the state
        // (two floats) is always consistent, so keep serving.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn try_take_at(&self, amount: f64, now: Instant) -> Result<(), Denied> {
        let mut state = self.lock();
        // Refill only moves forward: `now` may be stale relative to another
        // racer that locked first with a later timestamp.
        let elapsed = now.saturating_duration_since(state.updated);
        if elapsed > Duration::ZERO {
            state.tokens =
                (state.tokens + elapsed.as_secs_f64() * self.rate_per_sec).min(self.capacity);
            state.updated = now;
        }
        if state.tokens >= amount {
            state.tokens -= amount;
            Ok(())
        } else {
            let deficit = amount - state.tokens;
            Err(Denied {
                retry_after_secs: (deficit / self.rate_per_sec).ceil().max(1.0) as u64,
                exceeds_capacity: amount > self.capacity,
            })
        }
    }

    /// Returns unused reservation to the bucket, capped at capacity (refill
    /// may have accrued while the reservation was out).
    fn put_back(&self, amount: f64) {
        let mut state = self.lock();
        state.tokens = (state.tokens + amount).min(self.capacity);
    }
}

impl KeyLimiter {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn rpm_limit(&self) -> Option<u32> {
        self.rpm.as_ref().map(|limit| limit.per_minute)
    }

    pub(crate) fn tpm_limit(&self) -> Option<u32> {
        self.tpm.as_ref().map(|limit| limit.per_minute)
    }

    /// Takes one request from the rpm bucket (no-op without an rpm limit).
    pub(crate) fn take_request(&self) -> Result<(), Denied> {
        self.take_request_at(Instant::now())
    }

    fn take_request_at(&self, now: Instant) -> Result<(), Denied> {
        match &self.rpm {
            Some(limit) => limit.bucket.try_take_at(1.0, now),
            None => Ok(()),
        }
    }

    /// Reserves `needed` tokens from the tpm bucket. `Ok(None)` when the key
    /// has no tpm limit; the reservation must be [`TpmReservation::settle`]d
    /// with actual usage once known (unsettled = forfeited, module docs).
    pub(crate) fn reserve_tokens(
        self: &Arc<Self>,
        needed: u64,
    ) -> Result<Option<TpmReservation>, Denied> {
        self.reserve_tokens_at(needed, Instant::now())
    }

    fn reserve_tokens_at(
        self: &Arc<Self>,
        needed: u64,
        now: Instant,
    ) -> Result<Option<TpmReservation>, Denied> {
        let Some(limit) = &self.tpm else {
            return Ok(None);
        };
        limit.bucket.try_take_at(needed as f64, now)?;
        Ok(Some(TpmReservation {
            limiter: Arc::clone(self),
            reserved: needed,
            settled: AtomicBool::new(false),
        }))
    }
}

/// A worst-case tpm hold (`prompt + max_tokens`), taken before `Submit`.
/// Settling with actual usage refunds the remainder; dropping unsettled
/// forfeits the whole hold (deliberate — module docs).
#[derive(Debug)]
pub(crate) struct TpmReservation {
    limiter: Arc<KeyLimiter>,
    reserved: u64,
    settled: AtomicBool,
}

impl TpmReservation {
    /// Reconciles the hold against what the request actually consumed
    /// (client-visible prompt + completion tokens). Idempotent.
    pub(crate) fn settle(&self, actual_tokens: u64) {
        if self.settled.swap(true, Ordering::AcqRel) {
            return;
        }
        let refund = self.reserved.saturating_sub(actual_tokens);
        if refund > 0
            && let Some(limit) = &self.limiter.tpm
        {
            limit.bucket.put_back(refund as f64);
        }
    }
}

/// The tpm check used by the completion handlers, once `prompt_tokens` and
/// the effective `max_tokens` are known. `Ok(None)` = nothing to settle
/// (no key identity or no tpm limit).
pub(crate) fn reserve_completion_tokens(
    state: &AppState,
    handle: Option<&RateLimitHandle>,
    prompt_tokens: u32,
    max_tokens: u32,
) -> Result<Option<TpmReservation>, ApiError> {
    let Some(RateLimitHandle(limiter)) = handle else {
        return Ok(None);
    };
    let needed = u64::from(prompt_tokens) + u64::from(max_tokens);
    match limiter.reserve_tokens(needed) {
        Ok(reservation) => Ok(reservation),
        Err(denied) => {
            state
                .metrics
                .rate_limited_total
                .with_label_values(&[limiter.name(), "tokens"])
                .inc();
            tracing::info!(target: "kiln::ratelimit", api_key = %limiter.name(),
                limit_tpm = limiter.tpm_limit().unwrap_or(0), needed_tokens = needed,
                retry_after_secs = denied.retry_after_secs,
                exceeds_capacity = denied.exceeds_capacity,
                "request rejected: tokens-per-minute limit");
            Err(ApiError::rate_limited_tokens(
                limiter.name(),
                limiter.tpm_limit().unwrap_or(0),
                prompt_tokens,
                max_tokens,
                (!denied.exceeds_capacity).then_some(denied.retry_after_secs),
            ))
        }
    }
}

/// Shared by both rpm middlewares; the wrappers own the error envelope.
fn check_rpm(state: &AppState, request: &Request) -> Result<(), ApiError> {
    let Some(RateLimitHandle(limiter)) = request.extensions().get::<RateLimitHandle>() else {
        return Ok(());
    };
    match limiter.take_request() {
        Ok(()) => Ok(()),
        Err(denied) => {
            state
                .metrics
                .rate_limited_total
                .with_label_values(&[limiter.name(), "requests"])
                .inc();
            tracing::info!(target: "kiln::ratelimit", api_key = %limiter.name(),
                limit_rpm = limiter.rpm_limit().unwrap_or(0),
                retry_after_secs = denied.retry_after_secs,
                "request rejected: requests-per-minute limit");
            Err(ApiError::rate_limited_requests(
                limiter.name(),
                limiter.rpm_limit().unwrap_or(0),
                denied.retry_after_secs,
            ))
        }
    }
}

/// Route-layer middleware for the OpenAI-shaped `/v1/*` endpoints; layered
/// inside auth (which stamps the [`RateLimitHandle`]).
pub async fn enforce_rpm(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    match check_rpm(&state, &request) {
        Ok(()) => next.run(request).await,
        Err(err) => err.into_response(),
    }
}

/// Same check for `/v1/messages`, Anthropic error envelope.
pub async fn enforce_rpm_anthropic(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    match check_rpm(&state, &request) {
        Ok(()) => next.run(request).await,
        Err(err) => err.into_anthropic_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiKeyConfig;

    fn limiter(rpm: Option<u32>, tpm: Option<u32>) -> Arc<KeyLimiter> {
        RateLimiter::from_config(&AuthConfig {
            admin_token_hash: None,
            api_keys: vec![ApiKeyConfig {
                name: "k".into(),
                key_hash: "unused".into(),
                rpm,
                tpm,
            }],
        })
        .handle("k")
        .expect("key has limits")
    }

    #[test]
    fn no_handle_for_unlimited_or_unknown_keys() {
        let limits = RateLimiter::from_config(&AuthConfig {
            admin_token_hash: None,
            api_keys: vec![ApiKeyConfig {
                name: "free".into(),
                key_hash: "unused".into(),
                rpm: None,
                tpm: None,
            }],
        });
        assert!(limits.handle("free").is_none());
        assert!(limits.handle("nobody").is_none());
    }

    #[test]
    fn rpm_bucket_denies_past_limit_and_refills_on_schedule() {
        let limiter = limiter(Some(2), None);
        let t0 = Instant::now();
        assert!(limiter.take_request_at(t0).is_ok());
        assert!(limiter.take_request_at(t0).is_ok());
        let denied = limiter
            .take_request_at(t0)
            .expect_err("third request in the window must be denied");
        // 2/min = 1 token per 30s from an empty bucket.
        assert_eq!(denied.retry_after_secs, 30);
        assert!(!denied.exceeds_capacity);
        // One second before the refill: still denied.
        assert!(
            limiter
                .take_request_at(t0 + Duration::from_secs(29))
                .is_err()
        );
        // At the advertised Retry-After: admitted again.
        assert!(
            limiter
                .take_request_at(t0 + Duration::from_secs(30))
                .is_ok()
        );
    }

    #[test]
    fn rpm_only_key_never_blocks_token_reservations() {
        let limiter = limiter(Some(1), None);
        let reservation = limiter
            .reserve_tokens(1_000_000)
            .expect("no tpm limit configured");
        assert!(reservation.is_none());
    }

    #[test]
    fn tpm_reserve_settle_refunds_only_unused() {
        let limiter = limiter(None, Some(100));
        let t0 = Instant::now();
        let reservation = limiter
            .reserve_tokens_at(80, t0)
            .expect("fits the fresh bucket")
            .expect("tpm limit configured");
        // 80 held: a 30-token reservation no longer fits.
        assert!(limiter.reserve_tokens_at(30, t0).is_err());
        // Actual usage 10 → 70 refunded, 90 available again.
        reservation.settle(10);
        assert!(limiter.reserve_tokens_at(90, t0).is_ok());
    }

    #[test]
    fn settle_is_idempotent_and_capped_at_capacity() {
        let limiter = limiter(None, Some(100));
        let t0 = Instant::now();
        let reservation = limiter
            .reserve_tokens_at(80, t0)
            .expect("fits")
            .expect("tpm limit configured");
        reservation.settle(0);
        reservation.settle(0); // double-settle must not refund twice
        // Exactly the full capacity is back — no more, no less.
        assert!(limiter.reserve_tokens_at(100, t0).is_ok());
        assert!(limiter.reserve_tokens_at(1, t0).is_err());
    }

    #[test]
    fn reservation_larger_than_capacity_flagged_and_consumes_nothing() {
        let limiter = limiter(None, Some(100));
        let t0 = Instant::now();
        let denied = limiter
            .reserve_tokens_at(150, t0)
            .expect_err("cannot ever fit");
        assert!(denied.exceeds_capacity);
        // The failed attempt took nothing: the whole bucket is still there.
        assert!(limiter.reserve_tokens_at(100, t0).is_ok());
    }

    /// The task-mandated race test: many threads hitting one key's rpm
    /// bucket simultaneously admit EXACTLY the configured limit — the
    /// over-admission class of bug the Phase 9 reservation ledger once had.
    #[test]
    fn concurrent_rpm_burst_admits_exactly_the_limit() {
        let limiter = limiter(Some(8), None);
        let barrier = Arc::new(std::sync::Barrier::new(32));
        let admitted: Vec<bool> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..32)
                .map(|_| {
                    let limiter = Arc::clone(&limiter);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        limiter.take_request().is_ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });
        assert_eq!(
            admitted.iter().filter(|ok| **ok).count(),
            8,
            "burst must admit exactly the rpm limit"
        );
    }

    /// Same property for tpm reservations: two racers whose combined
    /// reservations exceed the budget — exactly one wins.
    #[test]
    fn concurrent_tpm_reservations_cannot_overcommit() {
        let limiter = limiter(None, Some(400));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let outcomes: Vec<bool> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let limiter = Arc::clone(&limiter);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        limiter.reserve_tokens(330).is_ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });
        assert_eq!(
            outcomes.iter().filter(|ok| **ok).count(),
            1,
            "330 + 330 > 400: exactly one reservation may win"
        );
    }
}
