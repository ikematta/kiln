//! Model registry: one entry per configured `[[model]]`, holding everything
//! the request path needs — resolved paths, the chat template, the worker
//! channel, and the supervisor-maintained status.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use kiln_proto::v1::WorkerInfo;
use kiln_tokenize::{ChatTemplate, Tokenizer};
use sha2::{Digest, Sha256};
use tokio::sync::{RwLock, watch};
use tonic::transport::Channel;

use crate::config::{KilnConfig, ModelConfig, WorkerKind};
use crate::uds::uds_channel;

/// Gateway-side view of a worker's lifecycle (SPEC §2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Process spawned (or about to be); waiting for the model to load.
    Starting,
    /// Health reports READY; requests are routed.
    Ready,
    /// Being unloaded (eviction or idle TTL): drain in progress.
    Draining,
    /// Deliberately not loaded; a request (or admin action) reloads it.
    Unloaded { reason: UnloadReason },
    /// Crashed; the supervisor is backing off before respawning.
    Restarting { attempt: u32 },
    /// Exceeded the restart budget; requires manual intervention.
    Failed,
    /// Gateway is shutting down.
    Stopped,
}

/// Why a model is not loaded (SPEC §2.2/§2.3 memory governance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnloadReason {
    /// LRU-evicted to make room for another model's load.
    Evicted,
    /// Idle past its configured `ttl_seconds`.
    IdleTtl,
    /// Its own load was rejected: over budget with no evictable model.
    OverBudget,
    /// Operator asked for it via `POST /admin/models/{id}/unload`.
    Admin,
}

impl UnloadReason {
    /// Bounded label for `kiln_worker_unloads_total{reason}`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Evicted => "evicted",
            Self::IdleTtl => "idle_ttl",
            Self::OverBudget => "over_budget",
            Self::Admin => "admin",
        }
    }
}

impl fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Ready => write!(f, "ready"),
            Self::Draining => write!(f, "draining"),
            Self::Unloaded {
                reason: UnloadReason::Evicted,
            } => write!(f, "unloaded (evicted)"),
            Self::Unloaded {
                reason: UnloadReason::IdleTtl,
            } => write!(f, "unloaded (idle ttl)"),
            Self::Unloaded {
                reason: UnloadReason::OverBudget,
            } => write!(f, "unloaded (over budget)"),
            Self::Unloaded {
                reason: UnloadReason::Admin,
            } => write!(f, "unloaded (admin)"),
            Self::Restarting { attempt } => write!(f, "restarting (attempt {attempt})"),
            Self::Failed => write!(f, "failed"),
            Self::Stopped => write!(f, "stopped"),
        }
    }
}

pub struct ModelEntry {
    pub id: String,
    /// Resolved local model directory.
    pub model_path: PathBuf,
    /// Resolved local draft-model directory; `Some` exactly when the
    /// config has `[model.speculative]` (SPEC §6.5/§10, Rust workers
    /// only — validated at registry build).
    pub draft_path: Option<PathBuf>,
    pub socket_path: PathBuf,
    pub config: ModelConfig,
    /// Which worker binary serves this model (Rust or Python; `auto` is
    /// resolved at registry build).
    pub worker_kind: WorkerKind,
    /// Gateway-side tokenizer, loaded for Rust-worker models only: the
    /// gateway encodes prompts (token_ids submit path, BOS contract) and
    /// detokenizes streamed ids. Python workers own their tokenizer.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// None when the model directory has no chat template; chat requests
    /// against such a model fail with a clear 400.
    pub template: Option<ChatTemplate>,
    /// Lazy UDS channel; survives worker restarts (same socket path).
    pub channel: Channel,
    pub status: watch::Receiver<WorkerStatus>,
    /// Cached `GetInfo` from the running worker, refreshed on each (re)start.
    pub info: RwLock<Option<WorkerInfo>>,
    /// Unix time the entry was registered; reported by `GET /v1/models`.
    pub created_unix: u64,
}

impl ModelEntry {
    pub fn status(&self) -> WorkerStatus {
        self.status.borrow().clone()
    }
}

pub struct Registry {
    by_id: HashMap<String, Arc<ModelEntry>>,
    ordered: Vec<Arc<ModelEntry>>,
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "model '{id}': path '{path}' is not a local model directory (config.json missing); \
         model downloads arrive with kiln-jobs (Phase 10) — fetch it first"
    )]
    NotLocal { id: String, path: String },
    #[error(
        "model '{id}': the rust worker cannot serve this model ({reason}); \
         set worker = \"python\""
    )]
    RustUnsupported { id: String, reason: String },
    #[error("model '{id}': failed to load tokenizer.json for gateway-side tokenization: {source}")]
    Tokenizer {
        id: String,
        #[source]
        source: kiln_tokenize::TokenizerError,
    },
    #[error("model '{id}': failed to build worker channel: {source}")]
    Channel {
        id: String,
        #[source]
        source: tonic::transport::Error,
    },
    #[error(
        "socket path '{0}' exceeds the macOS UDS path limit (104 bytes); \
         shorten server.runtime_dir"
    )]
    SocketPathTooLong(String),
    #[error(
        "model '{id}': [model.speculative] requires the rust worker, but this model is \
         served by the python worker ({reason}); speculative decoding is a rust-worker \
         capability (SPEC §6.5) — remove the speculative block or make the model rust-servable"
    )]
    SpeculativeNeedsRust { id: String, reason: String },
    #[error(
        "model '{id}': speculative draft '{path}' is not a local model directory \
         (config.json missing); model downloads arrive with kiln-jobs (Phase 10) — \
         fetch it first"
    )]
    DraftNotLocal { id: String, path: String },
}

impl Registry {
    /// Builds entries for every configured model. The returned watch senders
    /// are handed to the supervisor, which owns status transitions.
    pub fn from_config(
        config: &KilnConfig,
    ) -> Result<(Self, Vec<watch::Sender<WorkerStatus>>), RegistryError> {
        let runtime_dir = expand_tilde(&config.server.runtime_dir);
        let created_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut by_id = HashMap::new();
        let mut ordered = Vec::new();
        let mut senders = Vec::new();

        for model in &config.models {
            let model_path = expand_tilde(Path::new(&model.path));
            if !model_path.join("config.json").is_file() {
                return Err(RegistryError::NotLocal {
                    id: model.id.clone(),
                    path: model.path.clone(),
                });
            }

            let (worker_kind, tokenizer) = resolve_worker(model, &model_path)?;
            let draft_path = resolve_draft(model, worker_kind)?;

            let template = match ChatTemplate::from_model_dir(&model_path) {
                Ok(t) => Some(t),
                Err(err) => {
                    tracing::warn!(model = %model.id, error = %err,
                        "no usable chat template; chat completions will be rejected");
                    None
                }
            };

            let socket_path = socket_path_for(&runtime_dir, &model.id);
            // macOS sun_path is 104 bytes; a longer path fails at bind time
            // with a confusing error, so reject it up front.
            let socket_str = socket_path.to_string_lossy();
            if socket_str.len() > 100 {
                return Err(RegistryError::SocketPathTooLong(socket_str.into_owned()));
            }

            let channel =
                uds_channel(socket_path.clone()).map_err(|source| RegistryError::Channel {
                    id: model.id.clone(),
                    source,
                })?;

            let (status_tx, status_rx) = watch::channel(WorkerStatus::Starting);
            let entry = Arc::new(ModelEntry {
                id: model.id.clone(),
                model_path,
                draft_path,
                socket_path,
                config: model.clone(),
                worker_kind,
                tokenizer,
                template,
                channel,
                status: status_rx,
                info: RwLock::new(None),
                created_unix,
            });
            by_id.insert(model.id.clone(), Arc::clone(&entry));
            ordered.push(entry);
            senders.push(status_tx);
        }

        Ok((Self { by_id, ordered }, senders))
    }

    pub fn get(&self, id: &str) -> Option<&Arc<ModelEntry>> {
        self.by_id.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<ModelEntry>> {
        self.ordered.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

/// SPEC §10 `worker` resolution, plus the gateway-side tokenizer that a Rust
/// route requires (the gateway owns tokenization for Rust workers).
///
/// `auto` applies the Phase 6 routing matrix: Rust if `kiln_models` can serve
/// the checkpoint — implemented architecture (`SUPPORTED_ARCHITECTURES`),
/// known `rope_scaling`, and a quantization format the Rust worker honors
/// (uniform affine 4/8-bit at group size 32/64/128, or an unquantized
/// bf16/f16 checkpoint; SPEC §7.3) — else Python, with the reason logged.
/// The same parse rejects unsupported variants by name at load time, so the
/// decision is exactly "would the Rust worker load this".
///
/// Explicit `worker = "rust"` on an unservable model stays a startup error
/// (the operator asked for something impossible); explicit `"python"` skips
/// the check entirely. A tokenizer.json the gateway cannot load downgrades
/// `auto` to Python (the Python worker owns its tokenizer) but fails an
/// explicit `"rust"`.
fn resolve_worker(
    model: &ModelConfig,
    model_path: &Path,
) -> Result<(WorkerKind, Option<Arc<Tokenizer>>), RegistryError> {
    let load_tokenizer = || {
        Tokenizer::from_model_dir(model_path)
            .map(Arc::new)
            .map_err(|source| RegistryError::Tokenizer {
                id: model.id.clone(),
                source,
            })
    };
    match model.worker {
        WorkerKind::Python => Ok((WorkerKind::Python, None)),
        WorkerKind::Rust => {
            kiln_models::ArchConfig::from_model_dir(model_path).map_err(|err| {
                RegistryError::RustUnsupported {
                    id: model.id.clone(),
                    reason: err.to_string(),
                }
            })?;
            Ok((WorkerKind::Rust, Some(load_tokenizer()?)))
        }
        WorkerKind::Auto => match kiln_models::ArchConfig::from_model_dir(model_path) {
            Ok(config) => match load_tokenizer() {
                Ok(tokenizer) => {
                    tracing::info!(model = %model.id, model_type = %config.model_type(),
                        "worker=auto resolved to rust");
                    Ok((WorkerKind::Rust, Some(tokenizer)))
                }
                Err(err) => {
                    tracing::info!(model = %model.id, reason = %err,
                        "worker=auto resolved to python (gateway tokenizer unavailable)");
                    Ok((WorkerKind::Python, None))
                }
            },
            Err(err) => {
                tracing::info!(model = %model.id, reason = %err,
                    "worker=auto resolved to python (rust worker cannot serve this model)");
                Ok((WorkerKind::Python, None))
            }
        },
    }
}

/// Resolves `[model.speculative]` to a local draft directory (SPEC
/// §6.5/§10). Speculation is a rust-worker capability, so a speculative
/// block on a python-routed model is a startup error — dropping it
/// silently would be exactly the "requested speculation silently inert"
/// state ADR 0005 forbids the worker. Deeper validation (tokenizer
/// compatibility, the ADR 0005 envelope) is the worker's job at attach;
/// the registry only guarantees a spawnable `--draft-model` path.
fn resolve_draft(
    model: &ModelConfig,
    worker_kind: WorkerKind,
) -> Result<Option<PathBuf>, RegistryError> {
    let Some(spec) = &model.speculative else {
        return Ok(None);
    };
    if worker_kind != WorkerKind::Rust {
        let reason = if model.worker == WorkerKind::Python {
            "worker = \"python\" is configured".to_owned()
        } else {
            "worker = \"auto\" resolved to python".to_owned()
        };
        return Err(RegistryError::SpeculativeNeedsRust {
            id: model.id.clone(),
            reason,
        });
    }
    let draft_path = expand_tilde(Path::new(&spec.draft));
    if !draft_path.join("config.json").is_file() {
        return Err(RegistryError::DraftNotLocal {
            id: model.id.clone(),
            path: spec.draft.clone(),
        });
    }
    Ok(Some(draft_path))
}

/// `$KILN_RUNTIME_DIR/worker-<model_hash>.sock` (SPEC §3). The hash keeps the
/// name filesystem-safe regardless of what characters the model id uses.
fn socket_path_for(runtime_dir: &Path, model_id: &str) -> PathBuf {
    let digest = Sha256::digest(model_id.as_bytes());
    let mut hash = String::with_capacity(12);
    for byte in &digest[..6] {
        use std::fmt::Write as _;
        let _ = write!(hash, "{byte:02x}");
    }
    runtime_dir.join(format!("worker-{hash}.sock"))
}

/// Expands a leading `~/` using `$HOME`; other paths pass through verbatim.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if let Some(rest) = text.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_expansion() {
        let home = std::env::var("HOME").expect("HOME set in tests");
        assert_eq!(
            expand_tilde(Path::new("~/.kiln/run")),
            PathBuf::from(home).join(".kiln/run")
        );
        assert_eq!(
            expand_tilde(Path::new("/abs/path")),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_tilde(Path::new("rel/path")),
            PathBuf::from("rel/path")
        );
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kiln-registry-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        dir
    }

    /// A `[[model]]` entry pointing at a directory holding `config` as its
    /// config.json. No tokenizer.json is written, so a Rust resolution here
    /// exercises the tokenizer-unavailable path.
    fn model_in_dir(dir: &Path, worker: WorkerKind, config: serde_json::Value) -> ModelConfig {
        std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");
        ModelConfig {
            id: "test-model".into(),
            path: dir.to_string_lossy().into_owned(),
            worker,
            pinned: false,
            ttl_seconds: 0,
            speculative: None,
        }
    }

    /// Minimal servable llama config (matrix: supported arch, 4-bit uniform).
    fn supported_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5, "vocab_size": 128,
            "quantization": {"group_size": 64, "bits": 4}
        })
    }

    fn resolve(worker: WorkerKind, config: serde_json::Value) -> (WorkerKind, bool) {
        let dir = temp_dir("resolve");
        let model = model_in_dir(&dir, worker, config);
        let result = resolve_worker(&model, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        let (kind, tokenizer) = result.expect("resolution must not error");
        (kind, tokenizer.is_some())
    }

    #[test]
    fn auto_routes_unsupported_configs_to_python() {
        // SPEC §12 Phase 6 matrix: every rejection reason lands on python.
        let unsupported = [
            // Architecture not implemented by the rust worker.
            serde_json::json!({"model_type": "phi3"}),
            // Per-module quantization override (mixed-precision checkpoint).
            {
                let mut c = supported_config();
                c["quantization"]["model.embed_tokens"] =
                    serde_json::json!({"group_size": 64, "bits": 4});
                c
            },
            // Non-affine quantization mode.
            {
                let mut c = supported_config();
                c["quantization"]["mode"] = serde_json::json!("mxfp4");
                c
            },
            // Out-of-matrix bits / group size (SPEC §7.3: 4/8 @ 32/64/128).
            {
                let mut c = supported_config();
                c["quantization"] = serde_json::json!({"group_size": 64, "bits": 3});
                c
            },
            {
                let mut c = supported_config();
                c["quantization"] = serde_json::json!({"group_size": 16, "bits": 4});
                c
            },
            // rope_scaling variant the rust worker does not implement.
            {
                let mut c = supported_config();
                c["rope_scaling"] = serde_json::json!({"rope_type": "longrope", "factor": 4.0});
                c
            },
        ];
        for config in unsupported {
            let (kind, has_tokenizer) = resolve(WorkerKind::Auto, config.clone());
            assert_eq!(kind, WorkerKind::Python, "config: {config}");
            assert!(!has_tokenizer, "python route must not load a tokenizer");
        }
    }

    #[test]
    fn auto_prefers_rust_but_downgrades_without_tokenizer() {
        // Every supported architecture, servable quant — but the test dir has
        // no tokenizer.json, so auto downgrades to python instead of failing.
        // The from_json_str assertion pins WHY python was chosen: the matrix
        // said yes and only the tokenizer was missing.
        for (model_type, extra) in [
            ("llama", serde_json::json!({})),
            ("qwen2", serde_json::json!({})),
            (
                "qwen3",
                serde_json::json!({
                    "max_position_embeddings": 4096, "rope_theta": 1e6,
                    "head_dim": 16, "tie_word_embeddings": true
                }),
            ),
            ("gemma2", serde_json::json!({"head_dim": 16})),
            ("gemma3_text", serde_json::json!({})),
        ] {
            let mut config = supported_config();
            config["model_type"] = serde_json::json!(model_type);
            if let Some(obj) = extra.as_object() {
                for (k, v) in obj {
                    config[k] = v.clone();
                }
            }
            kiln_models::ArchConfig::from_json_str(&config.to_string())
                .unwrap_or_else(|err| panic!("matrix must accept {config}: {err}"));
            let (kind, _) = resolve(WorkerKind::Auto, config.clone());
            assert_eq!(kind, WorkerKind::Python, "config: {config}");
        }
        // BF16/F16 (no quantization block) is servable too — same downgrade.
        let mut dense = supported_config();
        dense
            .as_object_mut()
            .expect("object")
            .remove("quantization");
        kiln_models::ArchConfig::from_json_str(&dense.to_string()).expect("dense is servable");
        let (kind, _) = resolve(WorkerKind::Auto, dense);
        assert_eq!(kind, WorkerKind::Python);
    }

    #[test]
    fn explicit_rust_on_unservable_model_is_a_startup_error() {
        let dir = temp_dir("rust-unservable");
        let mut config = supported_config();
        config["quantization"]["lm_head"] = serde_json::json!(false);
        let model = model_in_dir(&dir, WorkerKind::Rust, config);
        let result = resolve_worker(&model, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        let err = result.expect_err("must fail loudly");
        assert!(
            matches!(err, RegistryError::RustUnsupported { .. }),
            "{err}"
        );
    }

    #[test]
    fn explicit_rust_accepts_every_supported_architecture() {
        // Regression guard: explicit rust validation must use the full
        // ArchConfig matrix, not the Phase 3 llama-only parse. The servable
        // config passes arch/quant validation and fails only at the missing
        // tokenizer.json — proving the matrix said yes.
        let mut config = supported_config();
        config["model_type"] = serde_json::json!("qwen2");
        let dir = temp_dir("rust-qwen2");
        let model = model_in_dir(&dir, WorkerKind::Rust, config);
        let result = resolve_worker(&model, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        let err = result.expect_err("no tokenizer.json in the test dir");
        assert!(matches!(err, RegistryError::Tokenizer { .. }), "{err}");
    }

    #[test]
    fn explicit_python_skips_validation_entirely() {
        let (kind, has_tokenizer) = resolve(
            WorkerKind::Python,
            serde_json::json!({"model_type": "phi3"}),
        );
        assert_eq!(kind, WorkerKind::Python);
        assert!(!has_tokenizer);
    }

    #[test]
    fn speculative_requires_the_rust_worker() {
        use crate::config::SpeculativeConfig;
        let dir = temp_dir("spec-python");
        // The draft points at a real local dir — the failure under test is
        // the worker kind, not draft locality.
        let mut model = model_in_dir(&dir, WorkerKind::Python, supported_config());
        model.speculative = Some(SpeculativeConfig {
            draft: dir.to_string_lossy().into_owned(),
            gamma: 4,
        });
        let err = resolve_draft(&model, WorkerKind::Python).expect_err("python cannot speculate");
        assert!(
            matches!(err, RegistryError::SpeculativeNeedsRust { .. }),
            "{err}"
        );
        assert!(err.to_string().contains("worker = \"python\""), "{err}");

        // auto that resolved to python is rejected just as loudly — silently
        // dropping the speculative block would hide the misconfiguration.
        model.worker = WorkerKind::Auto;
        let err = resolve_draft(&model, WorkerKind::Python)
            .expect_err("auto->python cannot speculate either");
        assert!(err.to_string().contains("resolved to python"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn speculative_draft_must_be_local() {
        use crate::config::SpeculativeConfig;
        let dir = temp_dir("spec-draft");
        let mut model = model_in_dir(&dir, WorkerKind::Rust, supported_config());
        model.speculative = Some(SpeculativeConfig {
            draft: "mlx-community/Qwen3-0.6B-4bit".into(),
            gamma: 4,
        });
        let err = resolve_draft(&model, WorkerKind::Rust).expect_err("an HF id is not local");
        assert!(matches!(err, RegistryError::DraftNotLocal { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn speculative_draft_resolves_to_the_local_dir() {
        use crate::config::SpeculativeConfig;
        let dir = temp_dir("spec-ok");
        let mut model = model_in_dir(&dir, WorkerKind::Rust, supported_config());
        model.speculative = Some(SpeculativeConfig {
            draft: dir.to_string_lossy().into_owned(),
            gamma: 4,
        });
        let path = resolve_draft(&model, WorkerKind::Rust)
            .expect("local draft resolves")
            .expect("speculative block yields a path");
        assert_eq!(path, dir);

        model.speculative = None;
        assert!(
            resolve_draft(&model, WorkerKind::Rust)
                .expect("no block is fine")
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn socket_name_is_stable_and_safe() {
        let a = socket_path_for(Path::new("/run"), "llama-3.2-1b-4bit");
        let b = socket_path_for(Path::new("/run"), "llama-3.2-1b-4bit");
        let c = socket_path_for(Path::new("/run"), "other/model");
        assert_eq!(a, b);
        assert_ne!(a, c);
        let name = a.file_name().and_then(|n| n.to_str()).expect("utf8");
        assert!(
            name.starts_with("worker-") && name.ends_with(".sock"),
            "{name}"
        );
        assert_eq!(name.len(), "worker-".len() + 12 + ".sock".len());
    }
}
