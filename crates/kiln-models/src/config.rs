//! `config.json` parsing per mlx-lm's conventions (SPEC §7.2) — that is the
//! ecosystem contract: field names, defaults, and `rope_scaling` handling
//! mirror `mlx_lm.models.llama.ModelArgs` / `rope_utils.initialize_rope`
//! exactly.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid config.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("unsupported model_type {:?} (rust worker supports: {})", .0, SUPPORTED_ARCHITECTURES.join(", "))]
    UnsupportedArchitecture(String),
    #[error("unsupported rope_scaling: {0}")]
    UnsupportedRope(String),
    #[error("unsupported quantization: {0}")]
    UnsupportedQuantization(String),
}

/// mlx-lm affine group quantization parameters (SPEC §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

/// `model_type` values the Rust worker implements (SPEC §7.2). Keyed to the
/// [`ArchConfig`] dispatch below — extend both together. (`gemma3_text` is
/// the text-only Gemma3 checkpoint type; multimodal `"gemma3"` is not
/// supported and routes to the Python worker.)
pub const SUPPORTED_ARCHITECTURES: &[&str] = &["llama", "qwen2", "qwen3", "gemma2", "gemma3_text"];

/// Resolved `rope_scaling` (the variants mlx-lm's `initialize_rope` supports
/// for the SPEC §7.2 architectures: default, linear, llama3, yarn).
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// No scaling — plain RoPE from `rope_theta`.
    Default,
    /// `rope_type: "linear"` — plain RoPE with `scale = 1/factor`.
    Linear { factor: f32 },
    /// `rope_type: "llama3"` — frequency-warped RoPE (Llama 3.1/3.2).
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: f32,
    },
    /// `rope_type: "yarn"` — NTK-by-parts RoPE (Qwen long-context configs).
    /// Fields stay `f64` because the correction range and mscale are computed
    /// host-side in double precision, exactly like the Python reference.
    Yarn {
        factor: f64,
        original_max_position_embeddings: f64,
        beta_fast: f64,
        beta_slow: f64,
        mscale: f64,
        mscale_all_dim: f64,
    },
}

/// Resolves a raw `rope_scaling` value with mlx-lm's key fallbacks and
/// defaults (`initialize_rope`): `type` or `rope_type` selects the variant;
/// per-variant defaults match the reference constructors.
fn resolve_rope_scaling(raw: Option<&serde_json::Value>) -> Result<RopeScaling, ConfigError> {
    let Some(raw) = raw else {
        return Ok(RopeScaling::Default);
    };
    // Model configs in the wild carry `"rope_scaling": null` (Qwen3 does).
    if raw.is_null() {
        return Ok(RopeScaling::Default);
    }
    let get_f64 = |key: &str| raw.get(key).and_then(serde_json::Value::as_f64);
    let rope_type = raw
        .get("type")
        .or_else(|| raw.get("rope_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    match rope_type {
        "default" => Ok(RopeScaling::Default),
        "linear" => {
            let factor = get_f64("factor")
                .ok_or_else(|| ConfigError::UnsupportedRope("linear without factor".into()))?;
            Ok(RopeScaling::Linear {
                factor: factor as f32,
            })
        }
        "llama3" => {
            let factor = get_f64("factor")
                .ok_or_else(|| ConfigError::UnsupportedRope("llama3 without factor".into()))?;
            Ok(RopeScaling::Llama3 {
                factor: factor as f32,
                low_freq_factor: get_f64("low_freq_factor").unwrap_or(1.0) as f32,
                high_freq_factor: get_f64("high_freq_factor").unwrap_or(4.0) as f32,
                original_max_position_embeddings: get_f64("original_max_position_embeddings")
                    .unwrap_or(8192.0) as f32,
            })
        }
        // mlx-lm routes all three spellings to the same YarnRoPE.
        "yarn" | "deepseek_yarn" | "telechat3-yarn" => {
            let factor = get_f64("factor")
                .ok_or_else(|| ConfigError::UnsupportedRope("yarn without factor".into()))?;
            Ok(RopeScaling::Yarn {
                factor,
                original_max_position_embeddings: get_f64("original_max_position_embeddings")
                    .unwrap_or(4096.0),
                beta_fast: get_f64("beta_fast").unwrap_or(32.0),
                beta_slow: get_f64("beta_slow").unwrap_or(1.0),
                mscale: get_f64("mscale").unwrap_or(1.0),
                mscale_all_dim: get_f64("mscale_all_dim").unwrap_or(0.0),
            })
        }
        other => Err(ConfigError::UnsupportedRope(format!(
            "rope_type {other:?} (supported: default, linear, llama3, yarn)"
        ))),
    }
}

/// EOS token ids per mlx-lm's `config.json` handling: a single id or a list.
fn eos_ids_from(value: Option<&serde_json::Value>) -> Vec<u32> {
    let as_u32 = |v: &serde_json::Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    match value {
        Some(serde_json::Value::Array(ids)) => ids.iter().filter_map(as_u32).collect(),
        Some(single) => as_u32(single).into_iter().collect(),
        None => Vec::new(),
    }
}

/// SPEC §7.3 uniform-quantization bounds shared by every architecture.
fn validate_quant_params(quantization: Option<Quantization>) -> Result<(), ConfigError> {
    if let Some(q) = quantization
        && (!matches!(q.bits, 4 | 8) || !matches!(q.group_size, 32 | 64 | 128))
    {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "bits={} group_size={} (supported: 4/8 bits, 32/64/128 groups)",
            q.bits, q.group_size
        )));
    }
    Ok(())
}

/// Parsed Llama-family `config.json` (fields and defaults per mlx-lm).
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    /// `config.json` allows a single id or a list (Llama 3.x ships a list);
    /// resolved via [`Self::eos_token_ids`].
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_true() -> bool {
    true
}

/// Reads `<dir>/config.json` to a string with a pathful error.
fn read_config_json(dir: &Path) -> Result<String, ConfigError> {
    let path = dir.join("config.json");
    std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
        path: path.display().to_string(),
        source,
    })
}

impl LlamaConfig {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document. Every unsupported
    /// variant fails HERE, at load time, with a named reason — never
    /// mid-forward as an opaque shape error.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let config: Self = serde_json::from_value(raw.clone())?;
        if config.model_type != "llama" {
            return Err(ConfigError::UnsupportedArchitecture(
                config.model_type.clone(),
            ));
        }
        config.rope_scaling()?;
        validate_quantization(&raw)?;
        validate_quant_params(config.quantization)?;
        Ok(config)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// EOS token ids the worker stops on — the same set mlx-lm derives from
    /// `config.json` (verified against the pinned reference: Llama-3.2's
    /// `[128001, 128008, 128009]`). Empty when the config declares none.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        eos_ids_from(self.eos_token_id.as_ref())
    }

    /// Resolves `rope_scaling` with mlx-lm's key fallbacks and defaults.
    pub fn rope_scaling(&self) -> Result<RopeScaling, ConfigError> {
        resolve_rope_scaling(self.rope_scaling.as_ref())
    }
}

/// Parsed Qwen2/2.5-family `config.json` — fields and defaults mirror
/// `mlx_lm.models.qwen2.ModelArgs` exactly (notably: no `head_dim` override,
/// attention bias implied by the checkpoint's `.bias` tensors).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_qwen2_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_qwen2_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_qwen2_max_position_embeddings() -> usize {
    32_768
}

fn default_qwen2_rope_theta() -> f32 {
    1_000_000.0
}

impl Qwen2Config {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document (load-time fail-loud;
    /// see [`LlamaConfig::from_json_str`]).
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let config: Self = serde_json::from_value(raw.clone())?;
        if config.model_type != "qwen2" {
            return Err(ConfigError::UnsupportedArchitecture(
                config.model_type.clone(),
            ));
        }
        config.rope_scaling()?;
        validate_quantization(&raw)?;
        validate_quant_params(config.quantization)?;
        Ok(config)
    }

    /// `mlx_lm.models.qwen2.Attention`: always `hidden_size // n_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        eos_ids_from(self.eos_token_id.as_ref())
    }

    pub fn rope_scaling(&self) -> Result<RopeScaling, ConfigError> {
        resolve_rope_scaling(self.rope_scaling.as_ref())
    }
}

/// Parsed Qwen3-family `config.json` — fields mirror
/// `mlx_lm.models.qwen3.ModelArgs` exactly (all required there, so no serde
/// defaults here; `head_dim` is its own field, qk-norm is implied by the
/// architecture).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl Qwen3Config {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document (load-time fail-loud;
    /// see [`LlamaConfig::from_json_str`]).
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let config: Self = serde_json::from_value(raw.clone())?;
        if config.model_type != "qwen3" {
            return Err(ConfigError::UnsupportedArchitecture(
                config.model_type.clone(),
            ));
        }
        config.rope_scaling()?;
        validate_quantization(&raw)?;
        validate_quant_params(config.quantization)?;
        Ok(config)
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        eos_ids_from(self.eos_token_id.as_ref())
    }

    pub fn rope_scaling(&self) -> Result<RopeScaling, ConfigError> {
        resolve_rope_scaling(self.rope_scaling.as_ref())
    }
}

/// Parsed Gemma2-family `config.json` — fields and defaults mirror
/// `mlx_lm.models.gemma2.ModelArgs` exactly. Note what the reference does
/// NOT implement at the pin and Kiln therefore must not either: no sliding
/// window (the config's `sliding_window` is ignored), no `rope_scaling`,
/// always-tied embeddings, attention + final logit softcapping always on.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma2Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_gemma2_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default = "default_gemma2_attn_softcap")]
    pub attn_logit_softcapping: f32,
    #[serde(default = "default_gemma2_final_softcap")]
    pub final_logit_softcapping: f32,
    #[serde(default = "default_gemma2_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f64,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_gemma2_rope_theta() -> f32 {
    10_000.0
}

fn default_gemma2_attn_softcap() -> f32 {
    50.0
}

fn default_gemma2_final_softcap() -> f32 {
    30.0
}

fn default_gemma2_query_pre_attn_scalar() -> f64 {
    144.0
}

impl Gemma2Config {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document (load-time fail-loud;
    /// see [`LlamaConfig::from_json_str`]).
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let config: Self = serde_json::from_value(raw.clone())?;
        if config.model_type != "gemma2" {
            return Err(ConfigError::UnsupportedArchitecture(
                config.model_type.clone(),
            ));
        }
        validate_quantization(&raw)?;
        validate_quant_params(config.quantization)?;
        Ok(config)
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        eos_ids_from(self.eos_token_id.as_ref())
    }
}

/// Parsed Gemma3 text-only `config.json` — fields and defaults mirror
/// `mlx_lm.models.gemma3_text.ModelArgs` exactly (that dataclass defaults
/// every field to the 1B checkpoint's values). Tying is NOT a config field:
/// the reference's `sanitize` ties iff the checkpoint has no
/// `lm_head.weight`, so Kiln decides at weight-load time.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3Config {
    pub model_type: String,
    #[serde(default = "default_gemma3_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_gemma3_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_gemma3_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_gemma3_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_gemma3_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_gemma3_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_gemma3_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_gemma3_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_gemma3_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_gemma3_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    #[serde(default = "default_gemma3_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f64,
    #[serde(default = "default_gemma3_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_gemma3_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    #[serde(default = "default_gemma3_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_gemma3_hidden_size() -> usize {
    1152
}
fn default_gemma3_num_hidden_layers() -> usize {
    26
}
fn default_gemma3_intermediate_size() -> usize {
    6912
}
fn default_gemma3_num_attention_heads() -> usize {
    4
}
fn default_gemma3_head_dim() -> usize {
    256
}
fn default_gemma3_rms_norm_eps() -> f32 {
    1.0e-6
}
fn default_gemma3_vocab_size() -> usize {
    262_144
}
fn default_gemma3_num_key_value_heads() -> usize {
    1
}
fn default_gemma3_rope_theta() -> f32 {
    1_000_000.0
}
fn default_gemma3_rope_local_base_freq() -> f32 {
    10_000.0
}
fn default_gemma3_query_pre_attn_scalar() -> f64 {
    256.0
}
fn default_gemma3_sliding_window() -> usize {
    512
}
fn default_gemma3_sliding_window_pattern() -> usize {
    6
}
fn default_gemma3_max_position_embeddings() -> usize {
    32_768
}

impl Gemma3Config {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document (load-time fail-loud;
    /// see [`LlamaConfig::from_json_str`]).
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let config: Self = serde_json::from_value(raw.clone())?;
        if config.model_type != "gemma3_text" {
            return Err(ConfigError::UnsupportedArchitecture(
                config.model_type.clone(),
            ));
        }
        config.rope_scaling()?;
        validate_quantization(&raw)?;
        validate_quant_params(config.quantization)?;
        Ok(config)
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        eos_ids_from(self.eos_token_id.as_ref())
    }

    /// Scaling for the GLOBAL layers' rope (`rope_theta`); local layers are
    /// always plain `rope_local_base_freq` (mlx-lm `gemma3_text.Attention`).
    pub fn rope_scaling(&self) -> Result<RopeScaling, ConfigError> {
        resolve_rope_scaling(self.rope_scaling.as_ref())
    }
}

/// A validated `config.json` for any architecture the Rust worker implements,
/// dispatched on `model_type` ([`SUPPORTED_ARCHITECTURES`]).
///
/// `from_json_str` is also the gateway's `worker = "auto"` routing predicate
/// (SPEC §10): `Ok` means the Rust worker can serve the model; every `Err`
/// names the reason (architecture, rope variant, quantization format) and the
/// model routes to the Python worker instead.
#[derive(Debug, Clone)]
pub enum ArchConfig {
    Llama(LlamaConfig),
    Qwen2(Qwen2Config),
    Qwen3(Qwen3Config),
    Gemma2(Gemma2Config),
    Gemma3(Gemma3Config),
}

impl ArchConfig {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json_str(&read_config_json(dir.as_ref())?)
    }

    /// Parses and validates a `config.json` document. Every unsupported
    /// variant fails HERE, at load time, with a named reason — never
    /// mid-forward as an opaque shape error.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let model_type = raw
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        match model_type.as_str() {
            "llama" => Ok(Self::Llama(LlamaConfig::from_json_str(text)?)),
            "qwen2" => Ok(Self::Qwen2(Qwen2Config::from_json_str(text)?)),
            "qwen3" => Ok(Self::Qwen3(Qwen3Config::from_json_str(text)?)),
            "gemma2" => Ok(Self::Gemma2(Gemma2Config::from_json_str(text)?)),
            "gemma3_text" => Ok(Self::Gemma3(Gemma3Config::from_json_str(text)?)),
            _ => Err(ConfigError::UnsupportedArchitecture(model_type)),
        }
    }

    pub fn model_type(&self) -> &str {
        match self {
            Self::Llama(c) => &c.model_type,
            Self::Qwen2(c) => &c.model_type,
            Self::Qwen3(c) => &c.model_type,
            Self::Gemma2(c) => &c.model_type,
            Self::Gemma3(c) => &c.model_type,
        }
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Llama(c) => c.eos_token_ids(),
            Self::Qwen2(c) => c.eos_token_ids(),
            Self::Qwen3(c) => c.eos_token_ids(),
            Self::Gemma2(c) => c.eos_token_ids(),
            Self::Gemma3(c) => c.eos_token_ids(),
        }
    }
}

/// Rejects `quantization` blocks this crate cannot honor. Checked against
/// the RAW json: mlx-lm mixed-precision checkpoints add per-module override
/// entries (`"model.embed_tokens": {"bits": 8, ...}` or `"lm_head": false`)
/// inside the block, and serde's permissive unknown-field handling would
/// silently drop them from [`Quantization`] — the model would then load with
/// the uniform parameters and fail far away (an opaque shape error on the
/// first forward pass), or worse, quietly misread a module. Route such
/// models to the Python worker instead (SPEC §7.3).
fn validate_quantization(raw: &serde_json::Value) -> Result<(), ConfigError> {
    let Some(quant) = raw
        .get("quantization")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(());
    };
    if let Some(mode) = quant.get("mode")
        && mode.as_str() != Some("affine")
    {
        return Err(ConfigError::UnsupportedQuantization(format!(
            "quantization mode {mode} (supported: \"affine\")"
        )));
    }
    let mut offending: Vec<&str> = quant
        .keys()
        .map(String::as_str)
        .filter(|key| !matches!(*key, "group_size" | "bits" | "mode"))
        .collect();
    if offending.is_empty() {
        return Ok(());
    }
    offending.sort_unstable();
    Err(ConfigError::UnsupportedQuantization(format!(
        "per-module quantization overrides {offending:?} — the rust worker supports only \
         uniform affine group_size/bits (SPEC §7.3); route this model to the python worker"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama32_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "rms_norm_eps": 1e-5,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "tie_word_embeddings": true,
            "quantization": {"group_size": 64, "bits": 4}
        })
    }

    #[test]
    fn parses_llama32_shape() {
        let config: LlamaConfig = serde_json::from_value(llama32_json()).unwrap();
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(
            config.quantization,
            Some(Quantization {
                group_size: 64,
                bits: 4
            })
        );
        assert_eq!(
            config.rope_scaling().unwrap(),
            RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192.0,
            }
        );
    }

    #[test]
    fn defaults_match_mlx_lm() {
        let config: LlamaConfig = serde_json::from_value(serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 100
        }))
        .unwrap();
        assert_eq!(config.num_kv_heads(), 4);
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.rope_theta, 10_000.0);
        assert!(config.tie_word_embeddings);
        assert!(!config.attention_bias);
        assert!(!config.mlp_bias);
        assert_eq!(config.rope_scaling().unwrap(), RopeScaling::Default);
        assert_eq!(config.quantization, None);
    }

    #[test]
    fn per_module_quantization_overrides_are_a_named_load_error() {
        // Mixed-precision checkpoint shape: uniform defaults plus per-module
        // entries in both forms mlx-lm emits (override dict / `false`).
        let mut json = llama32_json();
        json["quantization"] = serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "model.embed_tokens": {"group_size": 32, "bits": 8},
            "lm_head": false,
        });
        let err = LlamaConfig::from_json_str(&json.to_string())
            .expect_err("per-module overrides must not load");
        assert!(
            matches!(err, ConfigError::UnsupportedQuantization(_)),
            "wrong error variant: {err:?}"
        );
        // The message must say WHAT is unsupported (the override keys, by
        // name) and where such models go instead — not a generic failure.
        let message = err.to_string();
        for needle in [
            "unsupported quantization",
            "per-module",
            "model.embed_tokens",
            "lm_head",
            "python worker",
        ] {
            assert!(
                message.contains(needle),
                "error message {message:?} does not mention {needle:?}"
            );
        }
    }

    #[test]
    fn quantization_mode_affine_accepted_others_named() {
        let mut json = llama32_json();
        json["quantization"] = serde_json::json!({"group_size": 64, "bits": 4, "mode": "affine"});
        LlamaConfig::from_json_str(&json.to_string()).expect("affine mode is the supported mode");

        json["quantization"] = serde_json::json!({"group_size": 32, "bits": 4, "mode": "mxfp4"});
        let err = LlamaConfig::from_json_str(&json.to_string())
            .expect_err("non-affine modes must not load");
        assert!(
            matches!(err, ConfigError::UnsupportedQuantization(_)),
            "wrong error variant: {err:?}"
        );
        assert!(
            err.to_string().contains("mxfp4"),
            "error message {err} does not name the rejected mode"
        );
    }

    #[test]
    fn uniform_quantization_still_loads() {
        // The golden model's exact block (plus entry-point coverage for
        // from_json_str, which the fixture-driven tests reach via
        // from_model_dir).
        let config = LlamaConfig::from_json_str(&llama32_json().to_string()).expect("loads");
        assert_eq!(
            config.quantization,
            Some(Quantization {
                group_size: 64,
                bits: 4
            })
        );
    }

    #[test]
    fn eos_token_id_single_or_list() {
        let mut json = llama32_json();
        json["eos_token_id"] = serde_json::json!([128001, 128008, 128009]);
        let config: LlamaConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(config.eos_token_ids(), vec![128001, 128008, 128009]);

        json["eos_token_id"] = serde_json::json!(2);
        let config: LlamaConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(config.eos_token_ids(), vec![2]);

        json.as_object_mut().unwrap().remove("eos_token_id");
        let config: LlamaConfig = serde_json::from_value(json).unwrap();
        assert!(config.eos_token_ids().is_empty());
    }

    #[test]
    fn rope_type_key_fallback_and_rejects() {
        let mut json = llama32_json();
        json["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
        let config: LlamaConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            config.rope_scaling().unwrap(),
            RopeScaling::Linear { factor: 2.0 }
        );

        // Phase 6: yarn resolves (defaults per YarnRoPE's constructor);
        // genuinely unimplemented rope types still fail by name.
        json["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 2.0});
        let config: LlamaConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            config.rope_scaling().unwrap(),
            RopeScaling::Yarn {
                factor: 2.0,
                original_max_position_embeddings: 4096.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all_dim: 0.0,
            }
        );

        json["rope_scaling"] = serde_json::json!({"rope_type": "longrope", "factor": 2.0});
        let config: LlamaConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(
            config.rope_scaling(),
            Err(ConfigError::UnsupportedRope(_))
        ));
    }

    fn qwen3_06b_json() -> serde_json::Value {
        // The pinned qwen3-0.6b-4bit test model's config, trimmed to the
        // fields the parser reads (plus the null rope_scaling it ships).
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000,
            "rope_scaling": null,
            "tie_word_embeddings": true,
            "eos_token_id": 151645,
            "quantization": {"group_size": 64, "bits": 4}
        })
    }

    #[test]
    fn arch_config_dispatches_on_model_type() {
        let qwen3 = ArchConfig::from_json_str(&qwen3_06b_json().to_string()).expect("qwen3 loads");
        assert!(matches!(&qwen3, ArchConfig::Qwen3(c) if c.head_dim == 128));
        assert_eq!(qwen3.model_type(), "qwen3");
        assert_eq!(qwen3.eos_token_ids(), vec![151645]);

        let qwen2_json = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "eos_token_id": 151645,
            "quantization": {"group_size": 64, "bits": 4}
        });
        let qwen2 = ArchConfig::from_json_str(&qwen2_json.to_string()).expect("qwen2 loads");
        let ArchConfig::Qwen2(c) = &qwen2 else {
            panic!("wrong arch: {qwen2:?}");
        };
        // Defaults per mlx_lm.models.qwen2.ModelArgs.
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.max_position_embeddings, 32_768);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert!(c.tie_word_embeddings);

        let llama = ArchConfig::from_json_str(&llama32_json().to_string()).expect("llama loads");
        assert!(matches!(llama, ArchConfig::Llama(_)));

        let err = ArchConfig::from_json_str(
            &serde_json::json!({"model_type": "mamba", "hidden_size": 1}).to_string(),
        )
        .expect_err("unsupported arch is a named error");
        assert!(matches!(err, ConfigError::UnsupportedArchitecture(name) if name == "mamba"));
    }

    #[test]
    fn arch_config_rejects_what_the_loader_would_reject() {
        // Per-module quantization overrides fail at ArchConfig level too —
        // this is the `worker = "auto"` routing predicate (SPEC §10).
        let mut json = qwen3_06b_json();
        json["quantization"] = serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "lm_head": false,
        });
        let err = ArchConfig::from_json_str(&json.to_string()).expect_err("overrides rejected");
        assert!(matches!(err, ConfigError::UnsupportedQuantization(_)));

        let mut json = qwen3_06b_json();
        json["quantization"] = serde_json::json!({"group_size": 16, "bits": 4});
        let err = ArchConfig::from_json_str(&json.to_string()).expect_err("group 16 rejected");
        assert!(matches!(err, ConfigError::UnsupportedQuantization(_)));

        let mut json = qwen3_06b_json();
        json["rope_scaling"] = serde_json::json!({"rope_type": "longrope", "factor": 4.0});
        let err = ArchConfig::from_json_str(&json.to_string()).expect_err("longrope rejected");
        assert!(matches!(err, ConfigError::UnsupportedRope(_)));
    }

    #[test]
    fn yarn_parses_explicit_fields() {
        // The documented Qwen3 long-context recipe.
        let mut json = qwen3_06b_json();
        json["rope_scaling"] = serde_json::json!({
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        });
        let config: Qwen3Config = serde_json::from_value(json).unwrap();
        assert_eq!(
            config.rope_scaling().unwrap(),
            RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 32768.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all_dim: 0.0,
            }
        );
    }
}
