//! Prometheus metrics (SPEC §1.7, §8.1): a gateway-owned registry re-exported
//! at `GET /metrics`. Label cardinality is bounded: `path` is the matched
//! route pattern (never the raw URI), `model` the configured model id.

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
    /// Prompt tokens processed, per model (from worker `Finished` events).
    pub prompt_tokens_total: IntCounterVec,
    /// Completion tokens generated, per model.
    pub completion_tokens_total: IntCounterVec,
    /// Worker crash-restarts, per model.
    pub worker_restarts_total: IntCounterVec,
    /// 1 while the worker is Ready, else 0.
    pub worker_up: IntGaugeVec,
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

        Ok(Self {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            chat_completions_total,
            prompt_tokens_total,
            completion_tokens_total,
            worker_restarts_total,
            worker_up,
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
}
