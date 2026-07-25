//! TTFT and total request timeouts (SPEC §8.3).
//!
//! # Enforcement model
//!
//! Two independent, separately configurable budgets, both anchored at
//! **request arrival at the gateway** (handler entry, before body read):
//!
//! - **`server.ttft_timeout_secs`** — arrival → first generated token.
//!   The clock deliberately starts at arrival, not at prefill start:
//!   under Phase 9 admission pressure a request can sit in the worker's
//!   queue indefinitely, and queue wait is exactly what a client
//!   experiences as time-to-first-token. A timer starting at prefill
//!   would protect nothing for a queued request. Only a `Tokens` event
//!   satisfies the budget — the worker's `Admitted` event is an enqueue
//!   acknowledgement (the python worker emits it before any compute),
//!   not progress a client can see.
//! - **`server.total_timeout_secs`** — arrival → terminal event. Bounds
//!   the whole request, including generation time after the first token.
//!
//! These are NOT a queue-wait budget plus a compute budget: SPEC §8.3
//! names TTFT and total explicitly, and both are client-experience
//! bounds. A queue-only budget would let any amount of prefill run
//! after admission (invisible to the client until the first token), and
//! an operator tuning split budgets would have to sum them back into
//! the client-visible numbers anyway. Absent = disabled (the historical
//! behavior), matching the rpm/tpm "absent = unlimited" convention.
//!
//! # On expiry
//!
//! The request is cancelled **through the worker's existing Cancel RPC**
//! ([`crate::chat::CompletionCtx::abort_for_timeout`]) — the same path a
//! gateway-side stop-string match and a client disconnect use, with the
//! same ≤2-engine-step stop bound the workers promise for Cancel. There
//! is no separate teardown mechanism. The client gets a 504 with
//! `type: "timeout_error"` (OpenAI envelope; the Anthropic envelope's
//! status-keyed taxonomy has no timeout type, so it reports its 5xx
//! catch-all `api_error` — the message carries the detail). Partial
//! output on a non-streaming request is discarded; on a streaming
//! request the tokens already sent stand, and the stream ends with the
//! terminal error event instead of a completion. The request's tpm
//! reservation is left unsettled — it forfeits exactly like a client
//! disconnect (crate::ratelimit module docs) and self-heals via refill.
//!
//! # What the timers do NOT cover
//!
//! The pre-stream RPCs (`Tokenize`, `Submit`) are not wrapped: they are
//! cheap enqueue-shaped calls on a live UDS channel, and a worker whose
//! gRPC layer stops answering is already detected and restarted by the
//! supervisor's 1s health poll (2s RPC deadline, 3s missed-deadline
//! budget), which fails the in-flight request through the normal
//! crash path. Time spent there still counts against the budgets —
//! deadlines are absolute Instants from arrival.

use std::time::Duration;

use kiln_proto::v1::{TokenEvent, token_event};
use tokio::time::Instant;
use tonic::Streaming;

use crate::config::ServerConfig;

/// Configured budgets, built once at startup from `[server]`.
pub struct Timeouts {
    pub ttft: Option<Duration>,
    pub total: Option<Duration>,
}

impl Timeouts {
    pub fn from_config(server: &ServerConfig) -> Self {
        Self {
            ttft: server.ttft_timeout_secs.map(Duration::from_secs),
            total: server.total_timeout_secs.map(Duration::from_secs),
        }
    }
}

/// Which budget expired; labels `kiln_timeout_total{scope}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeoutScope {
    Ttft,
    Total,
}

impl TimeoutScope {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Ttft => "ttft",
            Self::Total => "total",
        }
    }
}

/// One request's live deadlines. The TTFT deadline retires when the first
/// `Tokens` event arrives; the total deadline lasts the whole request.
pub(crate) struct Deadlines {
    ttft: Option<Instant>,
    total: Option<Instant>,
}

impl Deadlines {
    /// Budgets anchored at `arrival` — the handler-entry timestamp, so
    /// body read, validation, tokenization, and Submit all spend budget.
    pub(crate) fn start(timeouts: &Timeouts, arrival: Instant) -> Self {
        Self {
            ttft: timeouts.ttft.map(|budget| arrival + budget),
            total: timeouts.total.map(|budget| arrival + budget),
        }
    }

    /// The earliest live deadline. Config validation guarantees
    /// ttft <= total, so while both are armed the TTFT one fires first.
    fn next(&self) -> Option<(Instant, TimeoutScope)> {
        match (self.ttft, self.total) {
            (Some(ttft), _) => Some((ttft, TimeoutScope::Ttft)),
            (None, Some(total)) => Some((total, TimeoutScope::Total)),
            (None, None) => None,
        }
    }

    fn first_token(&mut self) {
        self.ttft = None;
    }

    /// The absolute total-budget deadline, if configured — the bound MCP
    /// tool execution runs under between generation rounds (crate::mcp
    /// module docs: MCP round trips spend total budget).
    pub(crate) fn total(&self) -> Option<Instant> {
        self.total
    }
}

/// Next worker event under the request's deadlines, flattened for a
/// single-level match at the six stream-consumption sites.
pub(crate) enum WorkerEvent {
    Event(TokenEvent),
    /// Stream ended without a terminal event (worker crashed mid-request).
    Closed,
    Rpc(tonic::Status),
    TimedOut(TimeoutScope),
}

/// Awaits the next event, bounded by the earliest live deadline. A
/// `Tokens` event retires the TTFT deadline on the way through.
pub(crate) async fn next_event(
    events: &mut Streaming<TokenEvent>,
    deadlines: &mut Deadlines,
) -> WorkerEvent {
    let message = match deadlines.next() {
        None => events.message().await,
        Some((at, scope)) => match tokio::time::timeout_at(at, events.message()).await {
            Ok(message) => message,
            Err(_elapsed) => return WorkerEvent::TimedOut(scope),
        },
    };
    match message {
        Ok(Some(event)) => {
            if matches!(event.event, Some(token_event::Event::Tokens(_))) {
                deadlines.first_token();
            }
            WorkerEvent::Event(event)
        }
        Ok(None) => WorkerEvent::Closed,
        Err(status) => WorkerEvent::Rpc(status),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timeouts(ttft: Option<u64>, total: Option<u64>) -> Timeouts {
        Timeouts {
            ttft: ttft.map(Duration::from_secs),
            total: total.map(Duration::from_secs),
        }
    }

    #[test]
    fn ttft_deadline_fires_first_then_retires_to_total() {
        let arrival = Instant::now();
        let mut deadlines = Deadlines::start(&timeouts(Some(5), Some(60)), arrival);

        let (at, scope) = deadlines.next().expect("both budgets armed");
        assert_eq!(scope, TimeoutScope::Ttft);
        assert_eq!(at, arrival + Duration::from_secs(5));

        deadlines.first_token();
        let (at, scope) = deadlines.next().expect("total budget remains");
        assert_eq!(scope, TimeoutScope::Total);
        assert_eq!(at, arrival + Duration::from_secs(60));
    }

    #[test]
    fn single_budget_configurations() {
        let arrival = Instant::now();

        // TTFT only: nothing bounds the stream after the first token.
        let mut ttft_only = Deadlines::start(&timeouts(Some(5), None), arrival);
        assert_eq!(ttft_only.next().expect("armed").1, TimeoutScope::Ttft);
        ttft_only.first_token();
        assert!(ttft_only.next().is_none());

        // Total only: the same deadline before and after the first token.
        let mut total_only = Deadlines::start(&timeouts(None, Some(60)), arrival);
        assert_eq!(total_only.next().expect("armed").1, TimeoutScope::Total);
        total_only.first_token();
        assert_eq!(
            total_only.next().expect("still armed").1,
            TimeoutScope::Total
        );

        // Neither: never a deadline (the historical unbounded behavior).
        let none = Deadlines::start(&timeouts(None, None), arrival);
        assert!(none.next().is_none());
    }

    #[test]
    fn budgets_anchor_at_arrival_not_at_poll_time() {
        // The deadline is absolute: time spent before the event loop
        // (tokenize, Submit) is budget spent, not budget deferred.
        let arrival = Instant::now() - Duration::from_secs(4);
        let deadlines = Deadlines::start(&timeouts(Some(5), None), arrival);
        let (at, _) = deadlines.next().expect("armed");
        assert!(at <= Instant::now() + Duration::from_secs(1));
    }
}
