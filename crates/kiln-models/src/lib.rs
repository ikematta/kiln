#![deny(unsafe_code)]
//! `kiln-models`: model implementations (llama, qwen2/3, gemma2/3) and
//! `config.json` parsing per mlx-lm conventions (SPEC §7.2).
//!
//! Architecture dispatch is [`AnyModel`]/[`config::ArchConfig`]; the shared
//! decode paths live in the crate-private `nn` module and each architecture
//! module contributes config mapping + loading rules. Config parsing
//! compiles everywhere (the gateway's `worker = "auto"` routing predicate);
//! model code needs the `metal` feature (on by default, off for the Linux
//! compile-check).

pub mod config;
#[cfg(feature = "metal")]
pub mod draft;
#[cfg(feature = "metal")]
pub mod gemma2;
#[cfg(feature = "metal")]
pub mod gemma3;
#[cfg(feature = "metal")]
pub mod generate;
#[cfg(feature = "metal")]
pub mod kv_cache;
#[cfg(feature = "metal")]
pub mod llama;
#[cfg(feature = "metal")]
mod model;
#[cfg(feature = "metal")]
mod nn;
#[cfg(feature = "metal")]
pub mod qwen2;
#[cfg(feature = "metal")]
pub mod qwen3;
#[cfg(feature = "metal")]
pub mod weights;

pub use config::{
    ArchConfig, ConfigError, Gemma2Config, Gemma3Config, LlamaConfig, Quantization, Qwen2Config,
    Qwen3Config, RopeScaling, SUPPORTED_ARCHITECTURES,
};
#[cfg(feature = "metal")]
pub use draft::{DraftLoadError, DraftModel, DraftPoolSpec};
#[cfg(feature = "metal")]
pub use gemma2::Gemma2Model;
#[cfg(feature = "metal")]
pub use gemma3::Gemma3Model;
#[cfg(feature = "metal")]
pub use generate::{GenerateOutput, generate, generate_with};
#[cfg(feature = "metal")]
pub use kv_cache::KvCache;
#[cfg(feature = "metal")]
pub use llama::LlamaModel;
#[cfg(feature = "metal")]
pub use model::AnyModel;
#[cfg(feature = "metal")]
pub use nn::ModelError;
#[cfg(feature = "metal")]
pub use qwen2::Qwen2Model;
#[cfg(feature = "metal")]
pub use qwen3::Qwen3Model;
#[cfg(feature = "metal")]
pub use weights::{WeightStore, WeightsError};
