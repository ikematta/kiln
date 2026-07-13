//! Structured-output acceptance (SPEC §12 Phase 7, §11.3): drive the real
//! `kiln-worker` binary over the frozen `worker.proto` with `GrammarSpec`
//! requests against the pinned test model, and assert every constrained
//! generation is schema-valid — 100/100 across three schema shapes
//! (nested objects, enums, arrays) at temperature 1.0 with distinct seeds,
//! so the masking (not the model's inclinations) carries the guarantee.
//!
//! Grammar validity is a device-independent invariant: the mask makes
//! schema conformance a hard guarantee regardless of logits numerics, so
//! this suite is CI-blocking (unlike golden parity).
//!
//! Also covers the proto's grammar error semantics: lark specs are
//! GRAMMAR_UNSUPPORTED (json_schema + regex only, per the Phase 7 scope),
//! uncompilable specs are GRAMMAR_COMPILE, and `GetInfo` advertises
//! CAPABILITY_GRAMMAR.
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
    Capability, FinishReason, GrammarSpec, HealthRequest, InfoRequest, Priority, SamplingParams,
    StoppingParams, SubmitRequest, TokenEvent, TokenIds, WorkerErrorCode, WorkerState,
    grammar_spec, submit_request, token_event,
};
use tonic::transport::{Channel, Endpoint, Uri};

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// Generous cap: model load on a cold CI runner dominates.
const READY_TIMEOUT: Duration = Duration::from_secs(180);
/// Cap on any single stream read; real events arrive per decode step.
const EVENT_TIMEOUT: Duration = Duration::from_secs(120);
/// Concurrent constrained requests per wave — well inside the KV pool
/// (each request needs a handful of blocks) so admission never has to
/// queue a wave behind itself.
const WAVE: usize = 25;
/// Schema-valid generations required (SPEC §12 Phase 7 acceptance).
const TOTAL: usize = 100;

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
        let socket =
            std::env::temp_dir().join(format!("kiln-grammar-{tag}-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let child = Command::new(env!("CARGO_BIN_EXE_kiln-worker"))
            .arg("--model")
            .arg(model)
            .arg("--socket")
            .arg(&socket)
            .arg("--model-id")
            .arg(format!("grammar-test-{tag}"))
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

fn submission(
    id: &str,
    prompt: Vec<u32>,
    grammar: Option<grammar_spec::Grammar>,
    seed: u64,
    max_tokens: u32,
) -> SubmitRequest {
    SubmitRequest {
        request_id: id.to_owned(),
        input: Some(submit_request::Input::TokenIds(TokenIds { ids: prompt })),
        sampling: Some(SamplingParams {
            // Full-temperature sampling: the grammar mask, not the model's
            // inclinations, must carry schema validity.
            temperature: 1.0,
            top_p: 1.0,
            seed,
            ..SamplingParams::default()
        }),
        stopping: Some(StoppingParams {
            max_tokens,
            ..StoppingParams::default()
        }),
        grammar: grammar.map(|grammar| GrammarSpec {
            grammar: Some(grammar),
        }),
        priority: Priority::Interactive as i32,
        prefix_hint: 0,
        echo_prompt: false,
    }
}

/// Streams one submission to its terminal event, returning
/// `(finish_reason, error_code, error_detail, token_ids)`.
async fn run_to_finished(
    client: &mut WorkerClient<Channel>,
    request: SubmitRequest,
) -> (FinishReason, WorkerErrorCode, String, Vec<u32>) {
    let mut stream = client
        .submit(request)
        .await
        .expect("submit accepted")
        .into_inner();
    let mut tokens = Vec::new();
    loop {
        let event = tokio::time::timeout(EVENT_TIMEOUT, stream.message())
            .await
            .expect("stream read timed out")
            .expect("stream errored")
            .and_then(|event: TokenEvent| event.event);
        match event {
            Some(token_event::Event::Tokens(chunk)) => tokens.extend(chunk.token_ids),
            Some(token_event::Event::Finished(finished)) => {
                return (
                    finished.finish_reason(),
                    finished.error_code(),
                    finished.error_detail,
                    tokens,
                );
            }
            Some(_) => {}
            None => panic!("stream ended without a Finished event"),
        }
    }
}

/// The three acceptance schema shapes (nested objects, enums, arrays).
/// Every string/array/integer is bounded and `whitespace_flexible` is off
/// (compact JSON), so a complete generation has a hard token-length bound
/// — a Length finish is a real failure, not a sizing artifact.
fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "nested-object",
            serde_json::json!({
                "x-guidance": {"whitespace_flexible": false},
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 12},
                    "kind": {"type": "string", "enum": ["cat", "dog", "bird"]},
                    "stats": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer", "minimum": 0, "maximum": 120},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 8},
                                "maxItems": 3
                            }
                        },
                        "required": ["age", "tags"],
                        "additionalProperties": false
                    }
                },
                "required": ["name", "kind", "stats"],
                "additionalProperties": false
            }),
        ),
        (
            "enums-scalars",
            serde_json::json!({
                "x-guidance": {"whitespace_flexible": false},
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["red", "green", "blue", "amber"]},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10},
                    "active": {"type": "boolean"}
                },
                "required": ["color", "count", "active"],
                "additionalProperties": false
            }),
        ),
        (
            "object-array",
            serde_json::json!({
                "x-guidance": {"whitespace_flexible": false},
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "minimum": 0, "maximum": 999},
                                "label": {"type": "string", "enum": ["alpha", "beta", "gamma"]}
                            },
                            "required": ["id", "label"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": false
            }),
        ),
    ]
}

/// Hand-rolled validation per shape — explicit asserts instead of a
/// generic validator dependency.
fn validate(shape: &str, value: &serde_json::Value) {
    let object = value.as_object().unwrap_or_else(|| {
        panic!("{shape}: top level is not an object: {value}");
    });
    match shape {
        "nested-object" => {
            let name = object["name"].as_str().expect("name is a string");
            assert!(name.chars().count() <= 12, "name too long: {name:?}");
            let kind = object["kind"].as_str().expect("kind is a string");
            assert!(["cat", "dog", "bird"].contains(&kind), "bad kind {kind:?}");
            let stats = object["stats"].as_object().expect("stats is an object");
            let age = stats["age"].as_i64().expect("age is an integer");
            assert!((0..=120).contains(&age), "age out of range: {age}");
            let tags = stats["tags"].as_array().expect("tags is an array");
            assert!(tags.len() <= 3, "too many tags");
            for tag in tags {
                let tag = tag.as_str().expect("tag is a string");
                assert!(tag.chars().count() <= 8, "tag too long: {tag:?}");
            }
        }
        "enums-scalars" => {
            let color = object["color"].as_str().expect("color is a string");
            assert!(
                ["red", "green", "blue", "amber"].contains(&color),
                "bad color {color:?}"
            );
            let count = object["count"].as_i64().expect("count is an integer");
            assert!((1..=10).contains(&count), "count out of range: {count}");
            assert!(object["active"].is_boolean(), "active is not a boolean");
        }
        "object-array" => {
            let items = object["items"].as_array().expect("items is an array");
            assert!(
                (1..=4).contains(&items.len()),
                "items length out of range: {}",
                items.len()
            );
            for item in items {
                let item = item.as_object().expect("item is an object");
                let id = item["id"].as_i64().expect("id is an integer");
                assert!((0..=999).contains(&id), "id out of range: {id}");
                let label = item["label"].as_str().expect("label is a string");
                assert!(
                    ["alpha", "beta", "gamma"].contains(&label),
                    "bad label {label:?}"
                );
            }
        }
        other => panic!("unknown shape {other}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn grammar_constrained_decoding() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };
    let tokenizer =
        kiln_tokenize::Tokenizer::from_model_dir(&model).expect("test model tokenizer loads");
    let prompt = tokenizer
        .encode("Describe one pet as JSON.", true)
        .expect("prompt encodes");

    let worker = Worker::spawn(&model, "main");
    let mut client = worker.client_when_ready().await;

    // --- Capability: the rust worker advertises GRAMMAR (SPEC §5).
    let info = client
        .get_info(InfoRequest {})
        .await
        .expect("GetInfo")
        .into_inner();
    assert!(
        info.capabilities.contains(&(Capability::Grammar as i32)),
        "rust worker must advertise CAPABILITY_GRAMMAR, got {:?}",
        info.capabilities
    );

    // --- 100/100 schema-valid generations across the three shapes
    // (SPEC §12 Phase 7 acceptance), temperature 1.0, distinct seeds,
    // WAVE-sized concurrent batches (exercising the batched mask path).
    let schemas = schemas();
    let mut done = 0usize;
    let mut per_shape = vec![0usize; schemas.len()];
    while done < TOTAL {
        let wave: Vec<(usize, u64)> = (done..(done + WAVE).min(TOTAL))
            .map(|i| (i % schemas.len(), i as u64 + 1))
            .collect();
        let mut handles = Vec::new();
        for &(shape_idx, seed) in &wave {
            let (_, schema) = &schemas[shape_idx];
            // 512 comfortably clears the schemas' hard worst case (~280
            // tokens with every string char a \uXXXX escape), so Length
            // can only mean the grammar failed to terminate.
            let request = submission(
                &format!("grammar-{seed}"),
                prompt.clone(),
                Some(grammar_spec::Grammar::JsonSchema(schema.to_string())),
                seed,
                512,
            );
            let mut client = WorkerClient::new(worker.channel());
            handles.push(tokio::spawn(async move {
                run_to_finished(&mut client, request).await
            }));
        }
        for (&(shape_idx, seed), handle) in wave.iter().zip(handles) {
            let (shape, _) = schemas[shape_idx];
            let (reason, code, detail, tokens) = handle.await.expect("request task");
            assert_eq!(
                reason,
                FinishReason::Stop,
                "seed {seed} shape {shape}: expected Stop, got {reason:?} ({code:?}: {detail})"
            );
            let text = tokenizer.decode(&tokens, true).expect("decodes");
            let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|err| {
                panic!("seed {seed} shape {shape}: invalid JSON ({err}): {text}")
            });
            validate(shape, &value);
            per_shape[shape_idx] += 1;
            done += 1;
        }
    }
    assert_eq!(done, TOTAL);
    eprintln!("{TOTAL}/{TOTAL} schema-valid generations (per shape: {per_shape:?})");

    // --- Regex grammar: full-temperature output must match exactly.
    let (reason, code, detail, tokens) = run_to_finished(
        &mut client,
        submission(
            "grammar-regex",
            prompt.clone(),
            Some(grammar_spec::Grammar::Regex("[0-9]{3}-[0-9]{4}".to_owned())),
            7,
            32,
        ),
    )
    .await;
    assert_eq!(
        reason,
        FinishReason::Stop,
        "regex: expected Stop, got {reason:?} ({code:?}: {detail})"
    );
    let text = tokenizer.decode(&tokens, true).expect("decodes");
    let bytes = text.as_bytes();
    assert!(
        bytes.len() == 8
            && bytes[..3].iter().all(u8::is_ascii_digit)
            && bytes[3] == b'-'
            && bytes[4..].iter().all(u8::is_ascii_digit),
        "regex output does not match [0-9]{{3}}-[0-9]{{4}}: {text:?}"
    );

    // --- Proto error semantics: lark is out of the Phase 7 scope
    // (GRAMMAR_UNSUPPORTED); an uncompilable spec is GRAMMAR_COMPILE.
    let (reason, code, _, tokens) = run_to_finished(
        &mut client,
        submission(
            "grammar-lark",
            prompt.clone(),
            Some(grammar_spec::Grammar::Lark("start: \"hi\"".to_owned())),
            1,
            8,
        ),
    )
    .await;
    assert_eq!(reason, FinishReason::Error);
    assert_eq!(code, WorkerErrorCode::WorkerErrorGrammarUnsupported);
    assert!(tokens.is_empty(), "rejected request must not stream tokens");

    for (id, bad) in [
        (
            "grammar-bad-schema",
            grammar_spec::Grammar::JsonSchema(r#"{"type": "nonsense"}"#.to_owned()),
        ),
        (
            "grammar-bad-json",
            grammar_spec::Grammar::JsonSchema("{not json".to_owned()),
        ),
        (
            "grammar-bad-regex",
            grammar_spec::Grammar::Regex("(unclosed".to_owned()),
        ),
    ] {
        let (reason, code, detail, tokens) =
            run_to_finished(&mut client, submission(id, prompt.clone(), Some(bad), 1, 8)).await;
        assert_eq!(reason, FinishReason::Error, "{id}: {detail}");
        assert_eq!(
            code,
            WorkerErrorCode::WorkerErrorGrammarCompile,
            "{id}: {detail}"
        );
        assert!(!detail.is_empty(), "{id}: compile errors carry detail");
        assert!(tokens.is_empty(), "{id}: rejected request must not stream");
    }

    // --- Unconstrained requests through the same worker still greedy-run
    // to Length (the masking path must be inert without a grammar; the
    // golden suite pins the exact token ids).
    let mut plain = submission("grammar-none", prompt.clone(), None, 0, 12);
    if let Some(sampling) = plain.sampling.as_mut() {
        sampling.temperature = 0.0;
    }
    if let Some(stopping) = plain.stopping.as_mut() {
        stopping.ignore_eos = true;
    }
    let (reason, _, _, tokens) = run_to_finished(&mut client, plain).await;
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(tokens.len(), 12);
}
