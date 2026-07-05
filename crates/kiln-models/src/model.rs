//! Architecture dispatch (SPEC §7.2): one enum over every implemented model
//! so the worker and test harnesses can load any supported checkpoint from
//! its `config.json` alone. Model-specific behavior stays in the per-arch
//! modules; this is routing only.

use std::path::Path;

use kiln_engine::{KvDims, PagedKv, StepBatch, StepModel};
use kiln_mlx::{Array, MlxError, Stream};

use crate::config::ArchConfig;
use crate::llama::LlamaModel;
use crate::nn::ModelError;
use crate::qwen2::Qwen2Model;
use crate::qwen3::Qwen3Model;

/// A loaded model of any supported architecture, dispatched on
/// `config.json`'s `model_type`.
#[derive(Debug)]
pub enum AnyModel {
    Llama(LlamaModel),
    Qwen2(Qwen2Model),
    Qwen3(Qwen3Model),
}

impl AnyModel {
    /// Loads the model at `dir`, selecting the implementation by
    /// `model_type`. Unsupported architectures/rope/quantization fail here
    /// with the named [`crate::ConfigError`] reasons.
    pub fn load(dir: impl AsRef<Path>, s: &Stream) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        match ArchConfig::from_model_dir(dir)? {
            ArchConfig::Llama(_) => Ok(Self::Llama(LlamaModel::load(dir, s)?)),
            ArchConfig::Qwen2(_) => Ok(Self::Qwen2(Qwen2Model::load(dir, s)?)),
            ArchConfig::Qwen3(_) => Ok(Self::Qwen3(Qwen3Model::load(dir, s)?)),
        }
    }

    pub fn model_type(&self) -> &str {
        match self {
            Self::Llama(m) => &m.config().model_type,
            Self::Qwen2(m) => &m.config().model_type,
            Self::Qwen3(m) => &m.config().model_type,
        }
    }

    /// EOS token ids from the model's `config.json` (worker stop set).
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Llama(m) => m.config().eos_token_ids(),
            Self::Qwen2(m) => m.config().eos_token_ids(),
            Self::Qwen3(m) => m.config().eos_token_ids(),
        }
    }

    /// KV geometry for the engine's paged pools.
    pub fn kv_dims(&self) -> KvDims {
        match self {
            Self::Llama(m) => m.kv_dims(),
            Self::Qwen2(m) => m.kv_dims(),
            Self::Qwen3(m) => m.kv_dims(),
        }
    }
}

impl StepModel for AnyModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        match self {
            Self::Llama(m) => m.forward_step(batch, kv, s),
            Self::Qwen2(m) => m.forward_step(batch, kv, s),
            Self::Qwen3(m) => m.forward_step(batch, kv, s),
        }
    }
}
