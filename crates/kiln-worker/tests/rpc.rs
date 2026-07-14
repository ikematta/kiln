//! Black-box worker gRPC tests (SPEC §11.3: "submit/stream/cancel/drain"):
//! spawn the real `kiln-worker` binary on a Unix domain socket and drive
//! the frozen `worker.proto` with the tonic client, exactly as the gateway
//! does.
//!
//! Covers the Phase 4 part 3 RPC semantics:
//! - Cancel mid-stream → `Finished{CANCELLED}` on the Submit stream,
//!   `CancelAck{found}` true then false;
//! - Drain GRACEFUL: Health reports DRAINING, new Submits are rejected
//!   in-band with `WORKER_ERROR_DRAINING`, in-flight requests finish, and
//!   the optional deadline escalates stragglers to cancellation;
//! - Drain IMMEDIATE: in-flight requests are cancelled.
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable.

#![cfg(feature = "metal")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use hyper_util::rt::TokioIo;
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    CancelRequest, DrainMode, DrainRequest, FinishReason, Finished, HealthRequest, Priority,
    SamplingParams, StoppingParams, SubmitRequest, TokenEvent, TokenIds, WorkerErrorCode,
    WorkerState, submit_request, token_event,
};
use tonic::transport::{Channel, Endpoint, Uri};

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// Generous cap: model load on a cold CI runner dominates.
const READY_TIMEOUT: Duration = Duration::from_secs(180);
/// Cap on any single stream read; real events arrive per decode step.
const EVENT_TIMEOUT: Duration = Duration::from_secs(60);

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

/// The worker subprocess; killed (and its socket removed) on drop so a
/// failing assertion cannot leak a child process.
struct Worker {
    child: Child,
    socket: PathBuf,
}

impl Worker {
    fn spawn(model: &PathBuf, tag: &str) -> Worker {
        Self::spawn_with(model, tag, &[])
    }

    fn spawn_with(model: &PathBuf, tag: &str, extra: &[&str]) -> Worker {
        let socket =
            std::env::temp_dir().join(format!("kiln-rpc-{tag}-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let child = Command::new(env!("CARGO_BIN_EXE_kiln-worker"))
            .arg("--model")
            .arg(model)
            .arg("--socket")
            .arg(&socket)
            .arg("--model-id")
            .arg(format!("rpc-test-{tag}"))
            .args(extra)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("kiln-worker spawns");
        Worker { child, socket }
    }

    /// Lazy UDS channel (same shape as kiln-gateway/src/uds.rs).
    fn channel(&self) -> Channel {
        let path = self.socket.clone();
        Endpoint::try_from("http://kiln-worker.invalid")
            .expect("static endpoint uri")
            .connect_with_connector_lazy(tower::service_fn(move |_: Uri| {
                let path = path.clone();
                async move {
                    Ok::<_, std::io::Error>(TokioIo::new(
                        tokio::net::UnixStream::connect(path).await?,
                    ))
                }
            }))
    }

    async fn client_when_ready(&self) -> WorkerClient<Channel> {
        let mut client = WorkerClient::new(self.channel());
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Ok(response) = client.health(HealthRequest {}).await {
                let status = response.into_inner();
                match status.state() {
                    WorkerState::Ready => return client,
                    WorkerState::Unhealthy => panic!("worker unhealthy: {}", status.detail),
                    _ => {}
                }
            }
            assert!(
                Instant::now() < deadline,
                "worker did not become ready in {READY_TIMEOUT:?}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// Greedy request over fixed token ids; `ignore_eos` so streams run to
/// `max_tokens` deterministically.
fn submission(id: &str, max_tokens: u32) -> SubmitRequest {
    SubmitRequest {
        request_id: id.to_owned(),
        input: Some(submit_request::Input::TokenIds(TokenIds {
            ids: (1..=8).collect(),
        })),
        sampling: Some(SamplingParams::default()),
        stopping: Some(StoppingParams {
            max_tokens,
            ignore_eos: true,
            ..StoppingParams::default()
        }),
        grammar: None,
        priority: Priority::Interactive as i32,
        prefix_hint: 0,
        echo_prompt: false,
    }
}

type EventStream = tonic::Streaming<TokenEvent>;

async fn next_event(stream: &mut EventStream) -> Option<token_event::Event> {
    tokio::time::timeout(EVENT_TIMEOUT, stream.message())
        .await
        .expect("stream read timed out")
        .expect("stream errored")
        .and_then(|event| event.event)
}

/// Reads until the terminal `Finished`, returning it plus the number of
/// token chunks seen on the way.
async fn read_to_finished(stream: &mut EventStream) -> (Finished, usize) {
    let mut chunks = 0;
    loop {
        match next_event(stream).await {
            Some(token_event::Event::Finished(finished)) => {
                assert!(
                    next_event(stream).await.is_none(),
                    "events after Finished on a Submit stream"
                );
                return (finished, chunks);
            }
            Some(token_event::Event::Tokens(_)) => chunks += 1,
            Some(_) => {}
            None => panic!("stream ended without a Finished event"),
        }
    }
}

/// Reads events until `n` token chunks have been seen (stream stays open).
async fn read_chunks(stream: &mut EventStream, n: usize) {
    let mut chunks = 0;
    while chunks < n {
        match next_event(stream).await {
            Some(token_event::Event::Tokens(_)) => chunks += 1,
            Some(token_event::Event::Finished(finished)) => {
                panic!("stream finished early: {finished:?}")
            }
            Some(_) => {}
            None => panic!("stream ended while waiting for chunks"),
        }
    }
}

/// Times `min_tokens` generated tokens on an open stream, returning the
/// observed wall-clock per-token period. Call only after at least one
/// chunk has been read so prefill and kernel warmup don't inflate it.
async fn measured_token_period(stream: &mut EventStream, min_tokens: u32) -> Duration {
    let started = Instant::now();
    let mut tokens = 0;
    while tokens < min_tokens {
        match next_event(stream).await {
            Some(token_event::Event::Tokens(chunk)) => tokens += chunk.token_ids.len() as u32,
            Some(token_event::Event::Finished(finished)) => {
                panic!("stream finished while measuring decode rate: {finished:?}")
            }
            Some(_) => {}
            None => panic!("stream ended while measuring decode rate"),
        }
    }
    started
        .elapsed()
        .div_f64(f64::from(tokens))
        .max(Duration::from_millis(1))
}

async fn cancel_and_drain_rpc_semantics() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    // --- Worker 1: Cancel, then GRACEFUL drain with deadline escalation.
    {
        let worker = Worker::spawn(&model, "graceful");
        let mut client = worker.client_when_ready().await;

        // Cancel mid-stream: ack found, stream ends CANCELLED, second
        // cancel reports the request gone. The first chunk absorbs
        // prefill and kernel warmup; the next 12 tokens are timed to
        // size the drain deadline below — hosted runners have decoded
        // this model at ~4 tok/s under shared-GPU contention (run
        // 29127458930) vs ~125 tok/s on a dev M-series, so no fixed
        // deadline suits both.
        let mut stream = client
            .submit(submission("cancel-1", 400))
            .await
            .expect("submit ok")
            .into_inner();
        read_chunks(&mut stream, 1).await;
        let per_token = measured_token_period(&mut stream, 12).await;
        let ack = client
            .cancel(CancelRequest {
                request_id: "cancel-1".to_owned(),
            })
            .await
            .expect("cancel ok")
            .into_inner();
        assert!(ack.found, "live request not found by Cancel");
        let (finished, _) = read_to_finished(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Cancelled);
        assert!(
            finished.completion_tokens < 400,
            "cancel did not stop the stream early"
        );
        let ack = client
            .cancel(CancelRequest {
                request_id: "cancel-1".to_owned(),
            })
            .await
            .expect("cancel ok")
            .into_inner();
        assert!(!ack.found, "finished request still cancellable");

        // GRACEFUL drain: the short request finishes, the long one is
        // escalated to CANCELLED at the deadline, new Submits get the
        // in-band DRAINING error, Health reports DRAINING.
        //
        // The escalation contract is about ordering, not speed: the
        // deadline must outlive the short request yet expire well
        // before the long one could finish. Contention on a shared
        // runner is non-stationary — PR #14 run 29139947140 measured
        // 188 ms/token while a sibling test held the GPU, then decoded
        // the drain phase 11x faster once it finished — so the long
        // side cannot be a predicted-rate multiple. Instead its
        // max_tokens is a constant near the KV-pool admission ceiling
        // (prompt + max_tokens < 512 blocks x 32): finishing 12000
        // tokens inside the deadline would take a >=37x
        // measurement-to-execution rate swing, or >=2400 tok/s in the
        // floor regime. Only the deadline scales with the measured
        // period — 20x the short request's decode time, floored at
        // 5000ms — where over-estimating the period only adds margin.
        const SHORT_TOKENS: u32 = 16;
        const LONG_TOKENS: u32 = 12_000;
        let deadline = (per_token * (20 * SHORT_TOKENS)).max(Duration::from_millis(5000));
        eprintln!(
            "measured decode {:.1} ms/token -> drain deadline {} ms",
            per_token.as_secs_f64() * 1e3,
            deadline.as_millis()
        );
        let mut short = client
            .submit(submission("drain-short", SHORT_TOKENS))
            .await
            .expect("submit ok")
            .into_inner();
        let mut long = client
            .submit(submission("drain-long", LONG_TOKENS))
            .await
            .expect("submit ok")
            .into_inner();
        read_chunks(&mut short, 2).await;
        let ack = client
            .drain(DrainRequest {
                mode: DrainMode::Graceful as i32,
                deadline_ms: deadline.as_millis() as u64,
            })
            .await
            .expect("drain ok")
            .into_inner();
        assert!(
            (1..=2).contains(&ack.requests_remaining),
            "unexpected requests_remaining: {}",
            ack.requests_remaining
        );
        let health = client
            .health(HealthRequest {})
            .await
            .expect("health ok")
            .into_inner();
        assert_eq!(health.state(), WorkerState::Draining);
        let mut rejected = client
            .submit(submission("drain-rejected", 8))
            .await
            .expect("submit rpc itself succeeds")
            .into_inner();
        let (finished, chunks) = read_to_finished(&mut rejected).await;
        assert_eq!(finished.finish_reason(), FinishReason::Error);
        assert_eq!(finished.error_code(), WorkerErrorCode::WorkerErrorDraining);
        assert_eq!(chunks, 0, "rejected request must not stream tokens");
        let (finished, _) = read_to_finished(&mut short).await;
        assert_eq!(
            finished.finish_reason(),
            FinishReason::Length,
            "graceful drain must let in-flight requests finish \
             (deadline {}ms at {:.1} ms/token): {finished:?}",
            deadline.as_millis(),
            per_token.as_secs_f64() * 1e3,
        );
        assert_eq!(finished.completion_tokens, SHORT_TOKENS);
        let (finished, _) = read_to_finished(&mut long).await;
        assert_eq!(
            finished.finish_reason(),
            FinishReason::Cancelled,
            "drain deadline must escalate stragglers \
             (deadline {}ms at {:.1} ms/token): {finished:?}",
            deadline.as_millis(),
            per_token.as_secs_f64() * 1e3,
        );
        assert!(finished.completion_tokens < LONG_TOKENS);
        eprintln!("worker 1: cancel + graceful drain (deadline escalation) ok");
    }

    // --- Worker 2: IMMEDIATE drain cancels in-flight work.
    {
        let worker = Worker::spawn(&model, "immediate");
        let mut client = worker.client_when_ready().await;
        let mut stream = client
            .submit(submission("immediate-1", 2000))
            .await
            .expect("submit ok")
            .into_inner();
        read_chunks(&mut stream, 2).await;
        client
            .drain(DrainRequest {
                mode: DrainMode::Immediate as i32,
                deadline_ms: 0,
            })
            .await
            .expect("drain ok");
        let (finished, _) = read_to_finished(&mut stream).await;
        assert_eq!(
            finished.finish_reason(),
            FinishReason::Cancelled,
            "IMMEDIATE drain must cancel in-flight requests: {finished:?}"
        );
        assert!(finished.completion_tokens < 2000);
        let health = client
            .health(HealthRequest {})
            .await
            .expect("health ok")
            .into_inner();
        assert_eq!(health.state(), WorkerState::Draining);
        let mut rejected = client
            .submit(submission("immediate-rejected", 8))
            .await
            .expect("submit rpc itself succeeds")
            .into_inner();
        let (finished, _) = read_to_finished(&mut rejected).await;
        assert_eq!(finished.error_code(), WorkerErrorCode::WorkerErrorDraining);
        eprintln!("worker 2: immediate drain ok");
    }
}

/// Phase 5 RPC surface: `PrefixCacheHit` events, `Finished.cached_prompt_tokens`,
/// the `Stats` RPC (SPEC §5), and the SSD tier surviving a worker restart
/// (SPEC §6.4 persistence) — all over the frozen proto, exactly as the
/// gateway consumes them.
async fn prefix_cache_stats_and_ssd_restart() {
    use kiln_proto::v1::{Capability, InfoRequest, StatsRequest};

    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let ssd_dir = std::env::temp_dir().join(format!("kiln-rpc-ssd-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ssd_dir);
    std::fs::create_dir_all(&ssd_dir).expect("ssd dir");
    let ssd_arg = ssd_dir.display().to_string();

    // 64-token prompt: two full 32-token blocks become cache-eligible,
    // and (p-1) % 32 == 31 keeps the tail chunk hash-discoverable after a
    // restart (the containment probe needs a full chunk's tokens).
    let submission_64 = |id: &str| SubmitRequest {
        request_id: id.to_owned(),
        input: Some(submit_request::Input::TokenIds(TokenIds {
            ids: (1..=64).collect(),
        })),
        sampling: Some(SamplingParams::default()),
        stopping: Some(StoppingParams {
            max_tokens: 8,
            ignore_eos: true,
            ..StoppingParams::default()
        }),
        grammar: None,
        priority: Priority::Interactive as i32,
        prefix_hint: 0,
        echo_prompt: false,
    };

    /// Reads to Finished, returning it plus any PrefixCacheHit event.
    async fn read_with_cache(
        stream: &mut EventStream,
    ) -> (Finished, Option<kiln_proto::v1::PrefixCacheHit>) {
        let mut cache = None;
        loop {
            match next_event(stream).await {
                Some(token_event::Event::Finished(finished)) => return (finished, cache),
                Some(token_event::Event::Cache(hit)) => cache = Some(hit),
                Some(_) => {}
                None => panic!("stream ended without a Finished event"),
            }
        }
    }

    {
        let worker = Worker::spawn_with(&model, "cache", &["--ssd-dir", &ssd_arg]);
        let mut client = worker.client_when_ready().await;

        let info = client
            .get_info(InfoRequest {})
            .await
            .expect("info ok")
            .into_inner();
        assert!(
            info.capabilities
                .contains(&(Capability::PrefixCache as i32)),
            "PREFIX_CACHE must be advertised: {:?}",
            info.capabilities
        );
        assert!(
            info.capabilities.contains(&(Capability::SsdTier as i32)),
            "SSD_TIER must be advertised with --ssd-dir: {:?}",
            info.capabilities
        );

        // Cold, then warm: the second stream must carry PrefixCacheHit
        // and account the reuse in Finished.
        let mut stream = client
            .submit(submission_64("cache-cold"))
            .await
            .expect("submit ok")
            .into_inner();
        let (finished, cache) = read_with_cache(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Length);
        assert!(cache.is_none(), "first run cannot hit: {cache:?}");
        assert_eq!(finished.cached_prompt_tokens, 0);

        let mut stream = client
            .submit(submission_64("cache-warm"))
            .await
            .expect("submit ok")
            .into_inner();
        let (finished, cache) = read_with_cache(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Length);
        let hit = cache.expect("resubmit must emit PrefixCacheHit");
        // Containment: every prefill position (p - 1 = 63) is served —
        // one full block plus 31 rows of the next (copy-on-write tail).
        assert_eq!(hit.tokens_reused, 63, "resubmit must reuse all prefill");
        assert!(!hit.from_ssd, "pool-resident hit");
        assert_eq!(finished.cached_prompt_tokens, 63);

        // Stats (SPEC §5): totals + gauges, and the idle write-behind
        // flush persisting the donated block.
        let deadline = Instant::now() + Duration::from_secs(30);
        let stats = loop {
            let stats = client
                .stats(StatsRequest {})
                .await
                .expect("stats ok")
                .into_inner();
            if stats.ssd_writes_total >= 2 || Instant::now() > deadline {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(stats.requests_total, 2);
        assert!(stats.engine_steps_total > 0);
        assert_eq!(stats.prefix_tokens_reused_total, 63);
        assert_eq!(
            stats.kv_blocks_allocated + stats.kv_blocks_free,
            512,
            "block gauges must cover the pool: {stats:?}"
        );
        assert_eq!(stats.ssd_writes_total, 2, "two unique full blocks flushed");
        assert_eq!(stats.ssd_blocks_stored, 2);
        assert_eq!(stats.ssd_fingerprint_rejects_total, 0);
        eprintln!("stats + prefix cache over RPC ok: {stats:?}");
    }

    // Restart (SPEC §6.4 persistence): a fresh worker over the same cache
    // directory serves the prefix from SSD.
    {
        let worker = Worker::spawn_with(&model, "cache2", &["--ssd-dir", &ssd_arg]);
        let mut client = worker.client_when_ready().await;
        let mut stream = client
            .submit(submission_64("cache-restart"))
            .await
            .expect("submit ok")
            .into_inner();
        let (finished, cache) = read_with_cache(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Length);
        let hit = cache.expect("restart must hit from SSD");
        assert_eq!(hit.tokens_reused, 63, "containment across the restart");
        assert!(hit.from_ssd, "hit must come from the cold tier");
        let stats = client
            .stats(StatsRequest {})
            .await
            .expect("stats ok")
            .into_inner();
        assert!(stats.ssd_reads_total >= 1, "{stats:?}");
        eprintln!("worker restart served the prefix from SSD: {hit:?}");
    }
    let _ = std::fs::remove_dir_all(&ssd_dir);
}

/// One `#[test]` because cases in a binary run concurrently and both of
/// the above spawn kiln-worker child processes that drive the GPU — the
/// process-level counterpart of the single-engine-thread discipline every
/// Metal suite follows (see spec_probe.rs for the in-process precedent).
/// Concurrent workers on the CI runner's shared paravirtual GPU killed one
/// mid-stream (run 29364413227 attempt 1: h2 BrokenPipe from the
/// ssd-restart worker while the cancel test's worker was live; attempt 2
/// of the identical commit passed). Cases run in the old in-file order.
#[tokio::test(flavor = "multi_thread")]
async fn worker_rpc_semantics() {
    cancel_and_drain_rpc_semantics().await;
    prefix_cache_stats_and_ssd_restart().await;
}
