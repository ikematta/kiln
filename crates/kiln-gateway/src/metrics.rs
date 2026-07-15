//! Prometheus metrics (SPEC §1.7, §8.1): a gateway-owned registry re-exported
//! at `GET /metrics`. Label cardinality is bounded: `path` is the matched
//! route pattern (never the raw URI), `model` the configured model id.
//!
//! Two families live here: gateway-observed HTTP/request counters, and
//! worker-reported `Stats` values (SPEC §5) the supervisor polls and
//! re-exports with a `model` label (SPEC §2.3). The latter mirror
//! worker-lifetime totals, so they are exported as gauges (`set`) — they
//! reset when a worker restarts, exactly like a scraped counter would.

use kiln_proto::v1::{MemoryReport, WorkerStats};
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
    TextEncoder,
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
    /// Anthropic Messages requests (`/v1/messages`) by model and outcome;
    /// same outcome values as `chat_completions_total`.
    pub messages_total: IntCounterVec,
    /// Prompt tokens processed, per model (from worker `Finished` events).
    pub prompt_tokens_total: IntCounterVec,
    /// Completion tokens generated, per model.
    pub completion_tokens_total: IntCounterVec,
    /// Worker crash-restarts, per model.
    pub worker_restarts_total: IntCounterVec,
    /// 1 while the worker is Ready, else 0.
    pub worker_up: IntGaugeVec,
    /// Deliberate unloads by model and reason (`evicted | idle_ttl |
    /// over_budget`) — SPEC §2.2 memory governance, distinct from crashes.
    pub worker_unloads_total: IntCounterVec,
    /// Machine memory budget (SPEC §2.3): total unified memory ×
    /// `memory.budget_fraction`, or the explicit `memory.budget_bytes`.
    pub memory_budget_bytes: IntGauge,
    /// Bytes currently charged against the budget across all workers.
    pub memory_used_bytes: IntGauge,
    /// Worker `Stats` mirrors, per model.
    pub worker_stats: WorkerStatGauges,
    /// Heartbeat `MemoryReport` mirrors, per model (SPEC §2.3).
    pub worker_memory: MemoryGauges,
}

/// Per-model memory gauges from the heartbeat `MemoryReport`, plus the
/// derived budget footprint (`crate::lifecycle::footprint_bytes`).
pub struct MemoryGauges {
    footprint: IntGaugeVec,
    weights: IntGaugeVec,
    kv_pool_allocated: IntGaugeVec,
    mlx_active: IntGaugeVec,
    mlx_cache: IntGaugeVec,
    process_rss: IntGaugeVec,
}

impl MemoryGauges {
    /// Applies one heartbeat snapshot for `model`.
    pub fn record(&self, model: &str, report: &MemoryReport, footprint_bytes: u64) {
        let set = |gauge: &IntGaugeVec, value: u64| {
            gauge
                .with_label_values(&[model])
                .set(i64::try_from(value).unwrap_or(i64::MAX));
        };
        set(&self.footprint, footprint_bytes);
        set(&self.weights, report.weights_bytes);
        set(&self.kv_pool_allocated, report.kv_pool_allocated_bytes);
        set(&self.mlx_active, report.mlx_active_bytes);
        set(&self.mlx_cache, report.mlx_cache_bytes);
        set(&self.process_rss, report.process_rss_bytes);
    }

    /// Zeroes every gauge for `model` (worker exited: unload or crash).
    pub fn clear(&self, model: &str) {
        self.record(model, &MemoryReport::default(), 0);
    }
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
        let messages_total = counter(
            "kiln_messages_total",
            "Anthropic Messages requests by model and outcome",
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
        let worker_unloads_total = counter(
            "kiln_worker_unloads_total",
            "Deliberate worker unloads per model and reason (evicted | idle_ttl | over_budget)",
            &["model", "reason"],
        )?;
        let plain_gauge = |name: &str, help: &str| {
            let gauge = IntGauge::new(name, help)?;
            registry.register(Box::new(gauge.clone()))?;
            Ok::<_, prometheus::Error>(gauge)
        };
        let memory_budget_bytes = plain_gauge(
            "kiln_memory_budget_bytes",
            "Machine memory budget for all workers (SPEC 2.3)",
        )?;
        let memory_used_bytes = plain_gauge(
            "kiln_memory_used_bytes",
            "Bytes currently charged against the machine budget",
        )?;
        // Heartbeat MemoryReport mirrors (SPEC §2.3), per model.
        let mem = |name: &str, help: &str| gauge(name, help, &["model"]);
        let worker_memory = MemoryGauges {
            footprint: mem(
                "kiln_worker_memory_bytes",
                "Budgeted footprint of the worker: max(mlx active, weights + KV pool) + mlx cache",
            )?,
            weights: mem(
                "kiln_worker_weights_bytes",
                "Weight bytes held by the worker (target + draft)",
            )?,
            kv_pool_allocated: mem(
                "kiln_worker_kv_pool_allocated_bytes",
                "Preallocated KV pool bytes (target + draft)",
            )?,
            mlx_active: mem(
                "kiln_worker_mlx_active_bytes",
                "MLX active memory reported by the worker",
            )?,
            mlx_cache: mem(
                "kiln_worker_mlx_cache_bytes",
                "MLX buffer-cache memory reported by the worker",
            )?,
            process_rss: mem(
                "kiln_worker_process_rss_bytes",
                "Worker process resident set size",
            )?,
        };
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
            messages_total,
            prompt_tokens_total,
            completion_tokens_total,
            worker_restarts_total,
            worker_up,
            worker_unloads_total,
            memory_budget_bytes,
            memory_used_bytes,
            worker_stats,
            worker_memory,
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
    fn memory_gauges_record_and_clear() {
        let metrics = Metrics::new().expect("metrics build");
        metrics.memory_budget_bytes.set(1000);
        metrics.memory_used_bytes.set(700);
        let report = MemoryReport {
            weights_bytes: 500,
            kv_pool_allocated_bytes: 100,
            mlx_active_bytes: 620,
            mlx_cache_bytes: 80,
            process_rss_bytes: 900,
            ..MemoryReport::default()
        };
        metrics.worker_memory.record("m", &report, 700);
        metrics
            .worker_unloads_total
            .with_label_values(&["m", "evicted"])
            .inc();
        let text = metrics.encode().expect("encode");
        for needle in [
            "kiln_memory_budget_bytes 1000",
            "kiln_memory_used_bytes 700",
            "kiln_worker_memory_bytes{model=\"m\"} 700",
            "kiln_worker_weights_bytes{model=\"m\"} 500",
            "kiln_worker_mlx_cache_bytes{model=\"m\"} 80",
            "kiln_worker_unloads_total{model=\"m\",reason=\"evicted\"} 1",
        ] {
            assert!(text.contains(needle), "missing {needle} in:\n{text}");
        }
        metrics.worker_memory.clear("m");
        let text = metrics.encode().expect("encode");
        assert!(
            text.contains("kiln_worker_memory_bytes{model=\"m\"} 0"),
            "{text}"
        );
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
