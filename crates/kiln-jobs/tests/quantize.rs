//! Real quantization through the jobs venv: BF16 SmolLM2 → 4-bit affine,
//! verified loadable via the same config parser the gateway and Rust worker
//! use (`kiln_models::ArchConfig`, SPEC §7.3 conventions).
//!
//! Gated like the other model-backed suites: skips unless the pinned
//! smollm2-135m-bf16 checkout is present under $KILN_TEST_MODELS and `uv`
//! is on PATH (the converter runs in the python/kiln_jobs_py venv). CI runs
//! this in a dedicated step after `uv sync`; dev machines run it as part of
//! `cargo test --workspace` with KILN_TEST_MODELS set.

use std::path::PathBuf;
use std::sync::Arc;

use kiln_jobs::events::StdoutSink;
use kiln_jobs::quantize::{ConvertSpec, run_convert};

fn smollm2_bf16() -> Option<PathBuf> {
    let root = std::env::var("KILN_TEST_MODELS").ok()?;
    let dir = PathBuf::from(root).join("smollm2-135m-bf16");
    dir.join("config.json").is_file().then_some(dir)
}

fn uv_available() -> bool {
    std::process::Command::new("uv")
        .arg("--version")
        .output()
        .is_ok()
}

fn jobs_venv() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/kiln_jobs_py")
}

#[tokio::test]
async fn bf16_to_4bit_convert_produces_a_parseable_quantized_checkpoint() {
    let Some(src) = smollm2_bf16() else {
        eprintln!(
            "skipping: pinned smollm2-135m-bf16 not found; set KILN_TEST_MODELS and run \
             ./scripts/fetch-test-model.sh"
        );
        return;
    };
    if !uv_available() {
        eprintln!("skipping: uv not on PATH (needed for the jobs venv)");
        return;
    }

    let out = std::env::temp_dir().join(format!("kiln-quantize-{}", uuid::Uuid::now_v7()));
    let spec = ConvertSpec {
        src: src.display().to_string(),
        out: out.clone(),
        bits: 4,
        group_size: 64,
        venv: jobs_venv(),
    };
    let result = run_convert(&spec, Arc::new(StdoutSink)).await;
    result.expect("mlx_lm convert succeeds");

    // The bar from SPEC §9.1 acceptance: the output is loadable by the
    // existing config-parsing infrastructure, as a quantized llama-family
    // checkpoint with exactly the requested parameters.
    let config = kiln_models::ArchConfig::from_model_dir(&out)
        .expect("quantized output parses via kiln-models");
    let kiln_models::ArchConfig::Llama(llama) = config else {
        panic!("smollm2 must parse as the llama architecture");
    };
    let quant = llama
        .quantization
        .as_ref()
        .expect("converted checkpoint declares quantization");
    assert_eq!((quant.bits, quant.group_size), (4, 64));

    // And the artifact set the loader expects is present.
    for required in [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        assert!(
            out.join(required).is_file(),
            "converted checkpoint is missing {required}"
        );
    }

    let _ = std::fs::remove_dir_all(&out);
}
