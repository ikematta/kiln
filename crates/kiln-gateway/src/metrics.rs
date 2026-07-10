//! Prometheus metrics (SPEC §1.7, §8.1): a gateway-owned registry re-exported
//! at `GET /metrics`. Label cardinality is bounded: `path` is the matched
//! route pattern (never the raw URI), `model` the configured model id.
//!
//! Two families live here: gateway-observed HTTP/request counters, and
//! worker-reported `Stats` values (SPEC §5) the supervisor polls and
//! re-exports with a `model` label (SPEC §2.3). The latter mirror
//! worker-lifetime totals, so they are exported as gauges (`set`) — they
//! reset when a worker restarts, exactly like a scraped counter would.

use kiln_proto::v1::WorkerStats;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry, TextEncoder,
};

pub struct Metrics {
    registry: Registry,
    /// HTTP requests by method, matched route, and status code.
    pub http_requests_total: IntCounterVec,
    /// HTTP latency by matched route.
    pub http_request_duration_seconds: HistogramVec,
    /// Chat completion requests by model and outcome
    /// (`ok | client_error | worker_error | worker_crashed | unavailable`).
    pub chat_completions_total: IntCounterVec,
    /// Legacy text completion requests (`/v1/completions`) by model and
    /// outcome; same outcome values as `chat_completions_total`.
    pub completions_total: IntCounterVec,
    /// Prompt tokens processed, per model (from worker `Finished` events).
    pub prompt_tokens_total: IntCounterVec,
    /// Completion tokens generated, per model.
    pub completion_tokens_total: IntCounterVec,
    /// Worker crash-restarts, per model.
    pub worker_restarts_total: IntCounterVec,
    /// 1 while the worker is Ready, else 0.
    pub worker_up: IntGaugeVec,
    /// Worker `Stats` mirrors, per model.
    pub worker_stats: WorkerStatGauges,
}

/// One gauge per re-exported `WorkerStats` field (all labeled `model`).
pub struct WorkerStatGauges {
    requests_total: IntGaugeVec,
    requests_failed_total: IntGaugeVec,
    requests_cancelled_total: IntGaugeVec,
    requests_preempted_total: IntGaugeVec,
    tokens_prefilled_total: IntGaugeVec,
    tokens_generated_total: IntGaugeVec,
    prefix_tokens_reused_total: IntGaugeVec,
    kv_blocks_allocated: IntGaugeVec,
    kv_blocks_free: IntGaugeVec,
    ssd_blocks_stored: IntGaugeVec,
    ssd_reads_total: IntGaugeVec,
    ssd_writes_total: IntGaugeVec,
    ssd_fingerprint_rejects_total: IntGaugeVec,
    engine_steps_total: IntGaugeVec,
}

impl WorkerStatGauges {
    /// Applies one polled `WorkerStats` snapshot for `model`.
    pub fn record(&self, model: &str, stats: &WorkerStats) {
        let set = |gauge: &IntGaugeVec, value: u64| {
            gauge
                .with_label_values(&[model])
                .set(i64::try_from(value).unwrap_or(i64::MAX));
        };
        set(&self.requests_total, stats.requests_total);
        set(&self.requests_failed_total, stats.requests_failed);
        set(&self.requests_cancelled_total, stats.requests_cancelled);
        set(&self.requests_preempted_total, stats.requests_preempted);
        set(&self.tokens_prefilled_total, stats.tokens_prefilled_total);
        set(&self.tokens_generated_total, stats.tokens_generated_total);
        set(
            &self.prefix_tokens_reused_total,
            stats.prefix_tokens_reused_total,
        );
        set(&self.kv_blocks_allocated, stats.kv_blocks_allocated);
        set(&self.kv_blocks_free, stats.kv_blocks_free);
        set(&self.ssd_blocks_stored, stats.ssd_blocks_stored);
        set(&self.ssd_reads_total, stats.ssd_reads_total);
        set(&self.ssd_writes_total, stats.ssd_writes_total);
        set(
            &self.ssd_fingerprint_rejects_total,
            stats.ssd_fingerprint_rejects_total,
        );
        set(&self.engine_steps_total, stats.engine_steps_total);
    }
}

impl Metrics {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        let counter = |name: &str, help: &str, labels: &[&str]| {
            let vec = IntCounterVec::new(Opts::new(name, help), labels)?;
            registry.register(Box::new(vec.clone()))?;
            Ok::<_, prometheus::Error>(vec)
        };
        let gauge = |name: &str, help: &str, labels: &[&str]| {
            let vec = IntGaugeVec::new(Opts::new(name, help), labels)?;
            registry.register(Box::new(vec.clone()))?;
            Ok::<_, prometheus::Error>(vec)
        };

        let http_requests_total = counter(
            "kiln_http_requests_total",
            "HTTP requests handled, by method, matched route, and status",
            &["method", "path", "status"],
        )?;
        let http_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "kiln_http_request_duration_seconds",
                "HTTP request latency by matched route",
            ),
            &["path"],
        )?;
        registry.register(Box::new(http_request_duration_seconds.clone()))?;
        let chat_completions_total = counter(
            "kiln_chat_completions_total",
            "Chat completion requests by model and outcome",
            &["model", "outcome"],
        )?;
        let completions_total = counter(
            "kiln_completions_total",
            "Legacy text completion requests by model and outcome",
            &["model", "outcome"],
        )?;
        let prompt_tokens_total = counter(
            "kiln_prompt_tokens_total",
            "Prompt tokens processed per model",
            &["model"],
        )?;
        let completion_tokens_total = counter(
            "kiln_completion_tokens_total",
            "Completion tokens generated per model",
            &["model"],
        )?;
        let worker_restarts_total = counter(
            "kiln_worker_restarts_total",
            "Worker crash-restarts per model",
            &["model"],
        )?;
        let worker_up = gauge(
            "kiln_worker_up",
            "1 while the model's worker is Ready, else 0",
            &["model"],
        )?;
        // Worker Stats mirrors (SPEC §5/§2.3). Gauge-typed: worker-lifetime
        // totals that reset with the worker process.
        let stat = |name: &str, help: &str| gauge(name, help, &["model"]);
        let worker_stats = WorkerStatGauges {
            requests_total: stat(
                "kiln_worker_requests_total",
                "Requests accepted by the worker (worker lifetime)",
            )?,
            requests_failed_total: stat(
                "kiln_worker_requests_failed_total",
                "Requests finished with an error (worker lifetime)",
            )?,
            requests_cancelled_total: stat(
                "kiln_worker_requests_cancelled_total",
                "Requests cancelled (worker lifetime)",
            )?,
            requests_preempted_total: stat(
                "kiln_worker_requests_preempted_total",
                "Preemption events under memory pressure (worker lifetime)",
            )?,
            tokens_prefilled_total: stat(
                "kiln_worker_tokens_prefilled_total",
                "Prompt tokens prefilled (worker lifetime)",
            )?,
            tokens_generated_total: stat(
                "kiln_worker_tokens_generated_total",
                "Tokens generated (worker lifetime)",
            )?,
            prefix_tokens_reused_total: stat(
                "kiln_worker_prefix_tokens_reused_total",
                "Prompt tokens served from the radix prefix cache (worker lifetime)",
            )?,
            kv_blocks_allocated: stat(
                "kiln_worker_kv_blocks_allocated",
                "KV pool blocks currently owned by requests or the prefix cache",
            )?,
            kv_blocks_free: stat(
                "kiln_worker_kv_blocks_free",
                "KV pool blocks currently free",
            )?,
            ssd_blocks_stored: stat(
                "kiln_worker_ssd_blocks_stored",
                "KV blocks persisted in the SSD tier",
            )?,
            ssd_reads_total: stat(
                "kiln_worker_ssd_reads_total",
                "KV blocks loaded from the SSD tier (worker lifetime)",
            )?,
            ssd_writes_total: stat(
                "kiln_worker_ssd_writes_total",
                "KV blocks flushed to the SSD tier (worker lifetime)",
            )?,
            ssd_fingerprint_rejects_total: stat(
                "kiln_worker_ssd_fingerprint_rejects_total",
                "SSD slabs/slots rejected by fingerprint or verification (worker lifetime)",
            )?,
            engine_steps_total: stat(
                "kiln_worker_engine_steps_total",
                "Engine iterations (worker lifetime)",
            )?,
        };

        Ok(Self {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            chat_completions_total,
            completions_total,
            prompt_tokens_total,
            completion_tokens_total,
            worker_restarts_total,
            worker_up,
            worker_stats,
        })
    }

    /// Text-format exposition for `GET /metrics`.
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let mut buf = Vec::new();
        TextEncoder::new().encode(&self.registry.gather(), &mut buf)?;
        String::from_utf8(buf).map_err(|e| prometheus::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_and_encodes() {
        let metrics = Metrics::new().expect("metrics build");
        metrics
            .http_requests_total
            .with_label_values(&["POST", "/v1/chat/completions", "200"])
            .inc();
        metrics.worker_up.with_label_values(&["m"]).set(1);
        let text = metrics.encode().expect("encode");
        assert!(text.contains("kiln_http_requests_total"), "{text}");
        assert!(text.contains("kiln_worker_up{model=\"m\"} 1"), "{text}");
    }

    #[test]
    fn worker_stats_reexport_with_model_label() {
        let metrics = Metrics::new().expect("metrics build");
        let stats = WorkerStats {
            requests_total: 7,
            prefix_tokens_reused_total: 2016,
            kv_blocks_free: 448,
            ssd_writes_total: 63,
            engine_steps_total: 1234,
            ..WorkerStats::default()
        };
        metrics.worker_stats.record("m", &stats);
        let text = metrics.encode().expect("encode");
        for needle in [
            "kiln_worker_requests_total{model=\"m\"} 7",
            "kiln_worker_prefix_tokens_reused_total{model=\"m\"} 2016",
            "kiln_worker_kv_blocks_free{model=\"m\"} 448",
            "kiln_worker_ssd_writes_total{model=\"m\"} 63",
            "kiln_worker_engine_steps_total{model=\"m\"} 1234",
        ] {
            assert!(text.contains(needle), "missing {needle} in:\n{text}");
        }
    }
}
