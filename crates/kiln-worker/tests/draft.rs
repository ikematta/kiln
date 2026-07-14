//! Speculative decoding over the frozen proto (SPEC §6.5, Phase 8
//! part 2): `--draft-model` attaches a drafter to the worker, gated by
//! the tokenizer-compatibility check.
//!
//! Two workers are exercised:
//! - an INCOMPATIBLE pair (llama-3.2-1b target, qwen3-0.6b draft — the
//!   cross-tokenizer pair part 1 used to prove loading isolation) must be
//!   rejected LOUDLY at load: the worker reports UNHEALTHY with the
//!   incompatibility in its detail, and never serves;
//! - a COMPATIBLE pair (qwen3-0.6b-8bit target, qwen3-0.6b-4bit draft —
//!   byte-identical tokenizers) must reach READY, report worker-total
//!   weights (SPEC §2.3), advertise CAPABILITY_SPECULATIVE (gated on the
//!   correctness attach gates only — compat + the ADR 0005 envelope —
//!   never on a throughput expectation, per ADR 0006; a draft-less worker
//!   must NOT advertise it), stream greedy output IDENTICAL to a
//!   draft-less worker on the same device (the §6.5 correctness
//!   invariant, over RPC, with the verify loop live), and record
//!   acceptance metrics in `Timings` and `Stats`.
//!
//! Plus the `--draft-gamma` argv contract (the gateway's
//! `[model.speculative]` wiring emits it): 0 and gamma-without-draft are
//! rejected at parse, before any MLX work.
//!
//! Its own test binary, NOT a case in rpc.rs: test binaries run
//! sequentially under `cargo test`, while cases inside one binary run
//! concurrently — and rpc.rs's drain-deadline test calibrates a decode
//! rate under a contention profile that a third GPU-heavy sibling
//! breaks (measured on CI run 29294575202: the rate swung 60x once
//! this test's workers finished, past that test's 37x design margin).
//!
//! Engine-level invariance (all golden fixtures, self-draft and
//! adversarial drafters, rollback measurement) lives in
//! kiln-models/tests/spec_decode.rs; this suite covers the worker binary
//! and proto surface. Skips (with a note) when `KILN_TEST_MODELS` is
//! unset or Metal is unavailable.

#![cfg(feature = "metal")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use hyper_util::rt::TokioIo;
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    Capability, FinishReason, Finished, HealthRequest, InfoRequest, Priority, SamplingParams,
    StatsRequest, StoppingParams, SubmitRequest, TokenEvent, TokenIds, WorkerState, submit_request,
    token_event,
};
use tonic::transport::{Channel, Endpoint, Uri};

const TARGET_NAME: &str = "qwen3-0.6b-8bit";
const DRAFT_NAME: &str = "qwen3-0.6b-4bit";
const INCOMPATIBLE_TARGET_NAME: &str = "llama-3.2-1b-4bit";
/// Tokenizer-compatible with itself, but its manual softcapped attention
/// puts it outside the ADR 0005 speculation envelope.
const OUT_OF_ENVELOPE_NAME: &str = "gemma-2-2b-it-4bit";
/// Generous cap: model load on a cold CI runner dominates.
const READY_TIMEOUT: Duration = Duration::from_secs(180);
/// Cap on any single stream read; real events arrive per decode step.
const EVENT_TIMEOUT: Duration = Duration::from_secs(60);

fn named_model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(name);
    dir.join("config.json").is_file().then_some(dir)
}

/// `.safetensors` bytes — the `StaticInfo.weights_bytes` convention both
/// sides of the assertion are built from.
fn fs_weights_bytes(dir: &PathBuf) -> u64 {
    std::fs::read_dir(dir)
        .expect("model dir readable")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
                .then(|| entry.metadata().ok().map(|meta| meta.len()))?
        })
        .sum()
}

/// The worker subprocess; killed (and its socket removed) on drop so a
/// failing assertion cannot leak a child process.
struct Worker {
    child: Child,
    socket: PathBuf,
}

impl Worker {
    fn spawn(model: &PathBuf, tag: &str, extra: &[&str]) -> Worker {
        let socket =
            std::env::temp_dir().join(format!("kiln-draft-{tag}-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let child = Command::new(env!("CARGO_BIN_EXE_kiln-worker"))
            .arg("--model")
            .arg(model)
            .arg("--socket")
            .arg(&socket)
            .arg("--model-id")
            .arg(format!("draft-test-{tag}"))
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

    /// Polls Health until the worker settles: READY, or UNHEALTHY with a
    /// detail string. Panics on timeout.
    async fn settled_state(&self) -> (WorkerClient<Channel>, WorkerState, String) {
        let mut client = WorkerClient::new(self.channel());
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Ok(response) = client.health(HealthRequest {}).await {
                let status = response.into_inner();
                match status.state() {
                    WorkerState::Ready => return (client, WorkerState::Ready, status.detail),
                    WorkerState::Unhealthy => {
                        return (client, WorkerState::Unhealthy, status.detail);
                    }
                    _ => {}
                }
            }
            assert!(
                Instant::now() < deadline,
                "worker did not settle in {READY_TIMEOUT:?}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn client_when_ready(&self) -> WorkerClient<Channel> {
        let (client, state, detail) = self.settled_state().await;
        assert_eq!(state, WorkerState::Ready, "worker unhealthy: {detail}");
        client
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

/// Reads to the terminal `Finished`, collecting every streamed token id.
async fn read_token_ids(stream: &mut tonic::Streaming<TokenEvent>) -> (Finished, Vec<u32>) {
    let mut ids = Vec::new();
    loop {
        let event = tokio::time::timeout(EVENT_TIMEOUT, stream.message())
            .await
            .expect("stream read timed out")
            .expect("stream errored")
            .and_then(|event| event.event);
        match event {
            Some(token_event::Event::Finished(finished)) => return (finished, ids),
            Some(token_event::Event::Tokens(chunk)) => ids.extend(chunk.token_ids),
            Some(_) => {}
            None => panic!("stream ended without a Finished event"),
        }
    }
}

/// Spawns the worker binary with `args` and asserts it exits with a
/// failure whose stderr names `expect`. Argv parsing happens before any
/// MLX init, so this needs neither a GPU nor models.
fn assert_rejected_argv(args: &[&str], expect: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_kiln-worker"))
        .args(args)
        .output()
        .expect("kiln-worker spawns");
    assert!(
        !output.status.success(),
        "argv {args:?} must be rejected, got exit {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expect),
        "rejection of {args:?} must name its cause ({expect:?}), got: {stderr}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn draft_verify_over_rpc_with_compat_gate() {
    // --- argv contract first (no GPU, no models): the gateway's
    // [model.speculative] wiring appends `--draft-gamma`; a zero gamma
    // (speculation requested but never proposing) and a gamma without a
    // draft are both misconfigurations, rejected at parse.
    assert_rejected_argv(
        &[
            "--model",
            "/nonexistent",
            "--socket",
            "/tmp/x.sock",
            "--draft-gamma",
            "0",
        ],
        "--draft-gamma needs an integer >= 1",
    );
    assert_rejected_argv(
        &[
            "--model",
            "/nonexistent",
            "--socket",
            "/tmp/x.sock",
            "--draft-gamma",
            "4",
        ],
        "--draft-gamma requires --draft-model",
    );
    eprintln!("draft-gamma argv rejections ok");

    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let (Some(target), Some(draft), Some(incompatible_target)) = (
        named_model_dir(TARGET_NAME),
        named_model_dir(DRAFT_NAME),
        named_model_dir(INCOMPATIBLE_TARGET_NAME),
    ) else {
        eprintln!(
            "skipping: KILN_TEST_MODELS not set or {TARGET_NAME}/{DRAFT_NAME}/\
             {INCOMPATIBLE_TARGET_NAME} missing"
        );
        return;
    };
    let draft_arg = draft.display().to_string();

    // --- Loud rejection: a cross-tokenizer pair never reaches serving.
    {
        let worker = Worker::spawn(
            &incompatible_target,
            "reject",
            &["--draft-model", &draft_arg],
        );
        let (_, state, detail) = worker.settled_state().await;
        assert_eq!(
            state,
            WorkerState::Unhealthy,
            "an incompatible draft/target pair must fail the load, not serve: {detail}"
        );
        assert!(
            detail.contains("incompatible"),
            "the rejection must name its cause in the health detail: {detail}"
        );
        eprintln!("incompatible pair rejected as UNHEALTHY: {detail}");
    }

    // --- Loud rejection #2 (ADR 0005): a tokenizer-COMPATIBLE pair whose
    // target has no certified verify kernel class (gemma2's manual
    // softcapped attention) must also fail the load — never serve with
    // the requested speculation silently inert.
    if let Some(gemma2) = named_model_dir(OUT_OF_ENVELOPE_NAME) {
        let self_draft = gemma2.display().to_string();
        let worker = Worker::spawn(&gemma2, "envelope", &["--draft-model", &self_draft]);
        let (_, state, detail) = worker.settled_state().await;
        assert_eq!(
            state,
            WorkerState::Unhealthy,
            "an out-of-envelope target with a configured draft must fail the load: {detail}"
        );
        assert!(
            detail.contains("envelope"),
            "the rejection must name the ADR 0005 envelope: {detail}"
        );
        eprintln!("out-of-envelope target rejected as UNHEALTHY: {detail}");
    } else {
        eprintln!("skipping envelope-rejection case: {OUT_OF_ENVELOPE_NAME} missing");
    }

    // --- Baseline: compatible target, no draft.
    let (base_weights, base_tokens) = {
        let worker = Worker::spawn(&target, "nodraft", &[]);
        let mut client = worker.client_when_ready().await;
        let info = client
            .get_info(InfoRequest {})
            .await
            .expect("info ok")
            .into_inner();
        assert!(
            !info
                .capabilities
                .contains(&(Capability::Speculative as i32)),
            "a worker without a draft must not advertise SPECULATIVE: {:?}",
            info.capabilities
        );
        let health = client
            .health(HealthRequest {})
            .await
            .expect("health ok")
            .into_inner();
        let memory = health.memory.expect("memory report");
        assert_eq!(memory.weights_bytes, fs_weights_bytes(&target));
        let mut stream = client
            .submit(submission("nodraft-1", 24))
            .await
            .expect("submit ok")
            .into_inner();
        let (finished, tokens) = read_token_ids(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Length);
        let timings = finished.timings.expect("timings present");
        assert_eq!(
            (timings.spec_tokens_proposed, timings.spec_tokens_accepted),
            (0, 0),
            "no drafter, no speculation metrics"
        );
        (memory.weights_bytes, tokens)
    };

    // --- Same target with the compatible draft attached (through the
    // full gateway argv shape, --draft-gamma included): the verify loop
    // is live, and greedy output must not move by a token.
    let worker = Worker::spawn(
        &target,
        "draft",
        &["--draft-model", &draft_arg, "--draft-gamma", "4"],
    );
    let mut client = worker.client_when_ready().await;

    let info = client
        .get_info(InfoRequest {})
        .await
        .expect("info ok")
        .into_inner();
    assert!(
        info.capabilities
            .contains(&(Capability::Speculative as i32)),
        "a compatible attached draft must advertise SPECULATIVE \
         (correctness-gated only, ADR 0006): {:?}",
        info.capabilities
    );

    let health = client
        .health(HealthRequest {})
        .await
        .expect("health ok")
        .into_inner();
    assert_eq!(health.state(), WorkerState::Ready);
    let memory = health.memory.expect("memory report");
    assert_eq!(
        memory.weights_bytes,
        base_weights + fs_weights_bytes(&draft),
        "weights_bytes must be the worker total: target + draft"
    );

    let mut stream = client
        .submit(submission("draft-1", 24))
        .await
        .expect("submit ok")
        .into_inner();
    let (finished, tokens) = read_token_ids(&mut stream).await;
    assert_eq!(finished.finish_reason(), FinishReason::Length);
    assert_eq!(
        tokens, base_tokens,
        "greedy output changed with the draft/verify loop active (SPEC §6.5 invariant)"
    );
    let timings = finished.timings.expect("timings present");
    assert!(
        timings.spec_tokens_proposed > 0,
        "speculation never engaged for the drafted request: {timings:?}"
    );
    let stats = client
        .stats(StatsRequest {})
        .await
        .expect("stats ok")
        .into_inner();
    assert_eq!(
        (
            stats.spec_tokens_proposed_total as u32,
            stats.spec_tokens_accepted_total as u32
        ),
        (timings.spec_tokens_proposed, timings.spec_tokens_accepted),
        "worker stats totals must mirror the single request's metrics"
    );
    eprintln!(
        "draft/verify over RPC ok: weights {} -> {} bytes, {} identical tokens, \
         {}/{} draft tokens accepted",
        base_weights,
        memory.weights_bytes,
        tokens.len(),
        timings.spec_tokens_accepted,
        timings.spec_tokens_proposed,
    );
}
