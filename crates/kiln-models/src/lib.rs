#![deny(unsafe_code)]
//! `kiln-models`: model implementations (llama, qwen2/3, gemma2/3) and
//! `config.json` parsing per mlx-lm conventions (SPEC §7.2).
//!
//! Phase 3 ships the Llama family (single-request, contiguous KV cache);
//! further architectures and the paged/batched integration arrive with
//! Phases 4–6. Config parsing compiles everywhere; model code needs the
//! `metal` feature (on by default, off for the Linux compile-check).

pub mod config;
#[cfg(feature = "metal")]
pub mod generate;
#[cfg(feature = "metal")]
pub mod kv_cache;
#[cfg(feature = "metal")]
pub mod llama;
#[cfg(feature = "metal")]
pub mod weights;

pub use config::{ConfigError, LlamaConfig, Quantization, RopeScaling};
#[cfg(feature = "metal")]
pub use generate::{GenerateOutput, generate};
#[cfg(feature = "metal")]
pub use kv_cache::KvCache;
#[cfg(feature = "metal")]
pub use llama::{LlamaModel, ModelError};
#[cfg(feature = "metal")]
pub use weights::{WeightStore, WeightsError};
