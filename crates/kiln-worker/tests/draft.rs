//! Draft-model coexistence over the frozen proto (SPEC §6.5, Phase 8
//! part 1): `--draft-model` loads a second model into the worker
//! process. The worker must reach READY, `MemoryReport.weights_bytes`
//! must grow by exactly the draft checkpoint's bytes (worker totals,
//! SPEC §2.3 — the gateway budgets whole workers),
//! CAPABILITY_SPECULATIVE must stay un-advertised (no verify loop
//! exists yet), and target greedy output must be IDENTICAL to a
//! draft-less worker on the same device (loading isolation; greedy is
//! bit-reproducible run-to-run on one build per CLAUDE.md).
//!
//! Deliberately a cross-tokenizer pair — part 1 proves loading, not
//! drafting compatibility (the vocab check belongs to the verify loop).
//!
//! Its own test binary, NOT a case in rpc.rs: test binaries run
//! sequentially under `cargo test`, while cases inside one binary run
//! concurrently — and rpc.rs's drain-deadline test calibrates a decode
//! rate under a contention profile that a third GPU-heavy sibling
//! breaks (measured on CI run 29294575202: the rate swung 60x once
//! this test's workers finished, past that test's 37x design margin).
//!
//! The engine-level coexistence invariants (bit-identical target
//! output with the draft's pool materialized, sentinel bytes, leak
//! gate) live in kiln-models/tests/draft.rs; this suite covers the
//! worker binary + proto surface. Skips (with a note) when
//! `KILN_TEST_MODELS` is unset or Metal is unavailable.

#![cfg(feature = "metal")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use hyper_util::rt::TokioIo;
use kiln_proto::v1::worker_client::WorkerClient;
use kiln_proto::v1::{
    Capability, FinishReason, Finished, HealthRequest, InfoRequest, Priority, SamplingParams,
    StoppingParams, SubmitRequest, TokenEvent, TokenIds, WorkerState, submit_request, token_event,
};
use tonic::transport::{Channel, Endpoint, Uri};

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
const DRAFT_MODEL_NAME: &str = "qwen3-0.6b-4bit";
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

#[tokio::test(flavor = "multi_thread")]
async fn draft_model_loads_alongside_target() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let (Some(model), Some(draft)) = (
        named_model_dir(MODEL_NAME),
        named_model_dir(DRAFT_MODEL_NAME),
    ) else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME}/{DRAFT_MODEL_NAME} missing");
        return;
    };

    // --- Baseline: no draft.
    let (base_weights, base_tokens) = {
        let worker = Worker::spawn(&model, "nodraft", &[]);
        let mut client = worker.client_when_ready().await;
        let health = client
            .health(HealthRequest {})
            .await
            .expect("health ok")
            .into_inner();
        let memory = health.memory.expect("memory report");
        assert_eq!(memory.weights_bytes, fs_weights_bytes(&model));
        let mut stream = client
            .submit(submission("nodraft-1", 24))
            .await
            .expect("submit ok")
            .into_inner();
        let (finished, tokens) = read_token_ids(&mut stream).await;
        assert_eq!(finished.finish_reason(), FinishReason::Length);
        (memory.weights_bytes, tokens)
    };

    // --- Same target with the draft loaded alongside.
    let draft_arg = draft.display().to_string();
    let worker = Worker::spawn(&model, "draft", &["--draft-model", &draft_arg]);
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
        "SPECULATIVE must not be advertised before the verify loop exists: {:?}",
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
    assert_eq!(
        memory.kv_pool_allocated_bytes, 0,
        "no pool (target or draft) is materialized before the first request"
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
        "greedy output changed when the draft model was loaded alongside"
    );
    eprintln!(
        "draft coexistence over RPC ok: weights {} -> {} bytes, {} identical tokens",
        base_weights,
        memory.weights_bytes,
        tokens.len()
    );
}
