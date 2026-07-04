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
    /// Crashed; the supervisor is backing off before respawning.
    Restarting { attempt: u32 },
    /// Exceeded the restart budget; requires manual intervention.
    Failed,
    /// Gateway is shutting down.
    Stopped,
}

impl fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Ready => write!(f, "ready"),
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

            // `auto` still resolves to python until the Phase 6 routing
            // matrix (arch + quant-format detection) lands; explicit
            // worker="rust" is validated eagerly so a wrong arch fails at
            // startup, not at first request.
            let worker_kind = match model.worker {
                WorkerKind::Rust => {
                    kiln_models::LlamaConfig::from_model_dir(&model_path).map_err(|err| {
                        RegistryError::RustUnsupported {
                            id: model.id.clone(),
                            reason: err.to_string(),
                        }
                    })?;
                    WorkerKind::Rust
                }
                WorkerKind::Auto => {
                    tracing::info!(model = %model.id,
                        "worker=auto resolves to python (routing matrix lands in Phase 6)");
                    WorkerKind::Python
                }
                WorkerKind::Python => WorkerKind::Python,
            };
            let tokenizer = match worker_kind {
                WorkerKind::Rust => {
                    Some(Arc::new(Tokenizer::from_model_dir(&model_path).map_err(
                        |source| RegistryError::Tokenizer {
                            id: model.id.clone(),
                            source,
                        },
                    )?))
                }
                _ => None,
            };

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
