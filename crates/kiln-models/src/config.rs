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
    #[error("unsupported model_type {0:?} (rust worker v0 supports: llama)")]
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

/// Resolved `rope_scaling` (subset supported by the Llama path in v0).
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
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_true() -> bool {
    true
}

impl LlamaConfig {
    /// Loads and validates `<dir>/config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_json_str(&text)
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
        if let Some(q) = config.quantization
            && (!matches!(q.bits, 4 | 8) || !matches!(q.group_size, 32 | 64 | 128))
        {
            return Err(ConfigError::UnsupportedQuantization(format!(
                "bits={} group_size={} (supported: 4/8 bits, 32/64/128 groups)",
                q.bits, q.group_size
            )));
        }
        Ok(config)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Resolves `rope_scaling` with mlx-lm's key fallbacks and defaults
    /// (`type` or `rope_type`; low 1.0 / high 4.0 / old-context 8192).
    pub fn rope_scaling(&self) -> Result<RopeScaling, ConfigError> {
        let Some(raw) = &self.rope_scaling else {
            return Ok(RopeScaling::Default);
        };
        let get_f32 = |key: &str| raw.get(key).and_then(serde_json::Value::as_f64);
        let rope_type = raw
            .get("type")
            .or_else(|| raw.get("rope_type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        match rope_type {
            "default" => Ok(RopeScaling::Default),
            "linear" => {
                let factor = get_f32("factor")
                    .ok_or_else(|| ConfigError::UnsupportedRope("linear without factor".into()))?;
                Ok(RopeScaling::Linear {
                    factor: factor as f32,
                })
            }
            "llama3" => {
                let factor = get_f32("factor")
                    .ok_or_else(|| ConfigError::UnsupportedRope("llama3 without factor".into()))?;
                Ok(RopeScaling::Llama3 {
                    factor: factor as f32,
                    low_freq_factor: get_f32("low_freq_factor").unwrap_or(1.0) as f32,
                    high_freq_factor: get_f32("high_freq_factor").unwrap_or(4.0) as f32,
                    original_max_position_embeddings: get_f32("original_max_position_embeddings")
                        .unwrap_or(8192.0)
                        as f32,
                })
            }
            other => Err(ConfigError::UnsupportedRope(format!(
                "rope_type {other:?} (supported: default, linear, llama3)"
            ))),
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
    fn rope_type_key_fallback_and_rejects() {
        let mut json = llama32_json();
        json["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
        let config: LlamaConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            config.rope_scaling().unwrap(),
            RopeScaling::Linear { factor: 2.0 }
        );

        json["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 2.0});
        let config: LlamaConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(
            config.rope_scaling(),
            Err(ConfigError::UnsupportedRope(_))
        ));
    }
}
