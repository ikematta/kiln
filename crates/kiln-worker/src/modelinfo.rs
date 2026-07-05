//! Static model metadata for `WorkerInfo` — filesystem-only, computed before
//! the model loads.
//!
//! The `weights_fingerprint` and `chat_template_hash` schemes are ported
//! byte-for-byte from the Python worker's `modelinfo.py` so both workers
//! report identical values for the same model directory (the gateway
//! compares `chat_template_hash` against its own template).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum ModelInfoError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid config.json: {0}")]
    Config(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct StaticInfo {
    pub model_path: String,
    pub architecture: String,
    pub dtype: String,
    pub max_context_len: u32,
    /// Largest admissible PROMPT (0 = bounded only by `max_context_len`).
    /// Tighter than the context bound only where an architecture's prefill
    /// op stream is reference-shaped up to a prompt length (gemma2).
    pub max_prompt_len: u32,
    pub vocab_size: u32,
    pub weights_bytes: u64,
    pub weights_fingerprint: String,
    pub chat_template_hash: String,
}

pub fn read_static_info(model_dir: &Path) -> Result<StaticInfo, ModelInfoError> {
    let read = |path: PathBuf| -> Result<Vec<u8>, ModelInfoError> {
        std::fs::read(&path).map_err(|source| ModelInfoError::Io {
            path: path.display().to_string(),
            source,
        })
    };
    let config_bytes = read(model_dir.join("config.json"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;

    let architecture = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let dtype = dtype_string(&config);

    let mut weight_files: Vec<(String, u64)> = std::fs::read_dir(model_dir)
        .map_err(|source| ModelInfoError::Io {
            path: model_dir.display().to_string(),
            source,
        })?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            let size = entry.metadata().ok()?.len();
            name.ends_with(".safetensors").then_some((name, size))
        })
        .collect();
    weight_files.sort();
    let weights_bytes = weight_files.iter().map(|(_, size)| size).sum();

    // Identity scheme identical to modelinfo.py: config content plus each
    // weight file's name and size (not a content hash of the weights).
    let mut fp = Sha256::new();
    fp.update(format!("arch={architecture};dtype={dtype};").as_bytes());
    fp.update(Sha256::digest(&config_bytes));
    for (name, size) in &weight_files {
        fp.update(format!("{name}:{size};").as_bytes());
    }

    let template = chat_template_source(model_dir);
    let chat_template_hash = if template.is_empty() {
        String::new()
    } else {
        format!("{:x}", Sha256::digest(template.as_bytes()))
    };

    let as_u32 = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0)
    };
    // Parity envelopes (Phase 6 Task 2, mirrored from the kiln-models
    // per-arch accessors so WorkerInfo and admission agree before the
    // model finishes loading): gemma3's op stream is reference-shaped only
    // within the sliding window (`gemma3.rs` module docs — lifting it is
    // the recorded ring-gather follow-up); gemma2's manual softcapped
    // attention only for single-`prefill_step_size`-chunk prompts
    // (`gemma2.rs` module docs).
    let mut max_context_len = as_u32("max_position_embeddings");
    let mut max_prompt_len = 0;
    match architecture.as_str() {
        "gemma3_text" => {
            let window = match as_u32("sliding_window") {
                0 => 512, // mlx-lm ModelArgs default
                w => w,
            };
            max_context_len = if max_context_len == 0 {
                window
            } else {
                max_context_len.min(window)
            };
        }
        "gemma2" => max_prompt_len = 2048,
        _ => {}
    }
    Ok(StaticInfo {
        model_path: model_dir.display().to_string(),
        architecture,
        dtype,
        max_context_len,
        max_prompt_len,
        vocab_size: as_u32("vocab_size"),
        weights_bytes,
        weights_fingerprint: format!("{:x}", fp.finalize()),
        chat_template_hash,
    })
}

fn dtype_string(config: &serde_json::Value) -> String {
    if let Some(quant) = config.get("quantization").filter(|q| q.is_object())
        && let Some(bits) = quant.get("bits").and_then(serde_json::Value::as_u64)
    {
        let group = quant
            .get("group_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(64);
        return format!("q{bits}_g{group}");
    }
    match config
        .get("torch_dtype")
        .and_then(serde_json::Value::as_str)
    {
        Some("bfloat16") => "bf16".to_owned(),
        Some("float16") => "f16".to_owned(),
        Some("float32") => "f32".to_owned(),
        Some(other) => other.to_owned(),
        None => String::new(),
    }
}

/// Template source selection identical to modelinfo.py: chat_template.jinja,
/// else tokenizer_config.json's `chat_template` when it is a string.
fn chat_template_source(model_dir: &Path) -> String {
    if let Ok(text) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
        return text;
    }
    std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| {
            config
                .get("chat_template")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default()
}
