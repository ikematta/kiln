//! ADR 0002 B' deterministic-width calibration (design answer Q2): the
//! probe must return exactly the device's row-stability boundary — rows
//! bit-stable at W, divergent at W+1 — with no hardcoded hardware table.
//!
//! Verified against an INDEPENDENT implementation of the same physics:
//! this test re-derives the boundary for the model's distinct projection
//! shapes straight from the checkpoint tensors with raw
//! `ops::quantized_matmul` calls and asserts the calibrated W equals
//! `min(first divergent M) - 1`. That equality is the hardware-independent
//! definition of a correct answer (CI's M1-class GPUs have different
//! thresholds than the dev machine; both must satisfy it). The measured
//! value is printed for the PROGRESS record (9 on the Phase 6 dev machine).
//!
//! Skips when `KILN_TEST_MODELS` is unset or Metal is unavailable.

#![cfg(feature = "metal")]

use std::path::PathBuf;

use kiln_mlx::{Array, Stream, ops};
use kiln_models::{AnyModel, WeightStore};

const MODEL_NAME: &str = "qwen2.5-0.5b-4bit";

/// Independent row-stability boundary for one quantized projection:
/// smallest M in 2..=32 where row 0 of `x @ W^T` stops matching M=1
/// (33 when it never diverges).
fn independent_threshold(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    base: &Array,
    s: &Stream,
) -> usize {
    let k = scales.dim(1) * 64;
    let reps = ((k + base.dim(2) - 1) / base.dim(2)) as usize;
    let cats: Vec<&Array> = std::iter::repeat_n(base, reps).collect();
    let wide = ops::concatenate(&cats, 2, s).unwrap();
    let x1 = ops::contiguous(&ops::slice(&wide, &[0, 0, 0], &[1, 1, k], s).unwrap(), s).unwrap();
    let row0 = |x: &Array| -> Vec<u8> {
        let y = ops::quantized_matmul(x, weight, scales, biases, true, 64, 4, s).unwrap();
        let n = y.dim(2);
        let row = ops::contiguous(&ops::slice(&y, &[0, 0, 0], &[1, 1, n], s).unwrap(), s).unwrap();
        row.eval().unwrap();
        row.data_raw_bytes().unwrap()
    };
    let reference = row0(&x1);
    for m in 2..=32 {
        let rows: Vec<&Array> = std::iter::repeat_n(&x1, m).collect();
        let xm = ops::contiguous(&ops::concatenate(&rows, 1, s).unwrap(), s).unwrap();
        if row0(&xm) != reference {
            return m;
        }
    }
    33
}

#[test]
fn calibrated_width_is_the_row_stability_boundary() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(root) = std::env::var_os("KILN_TEST_MODELS") else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };
    let model_dir = PathBuf::from(root).join(MODEL_NAME);
    if !model_dir.join("config.json").is_file() {
        panic!("{MODEL_NAME} missing under KILN_TEST_MODELS — run ./scripts/fetch-test-model.sh");
    }
    kiln_mlx::init();
    let stream = Stream::gpu();
    let model = AnyModel::load(&model_dir, &stream).expect("model loads");

    let width = model
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");
    let width_again = model
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");
    assert_eq!(width, width_again, "calibration must be deterministic");
    assert!((1..=32).contains(&width), "width out of range: {width}");

    // Independent re-derivation over every distinct projection shape in
    // the checkpoint (layer 0 + the tied lm_head/embedding).
    let mut store = WeightStore::from_model_dir(&model_dir).expect("weights load");
    let ew = store.take("model.embed_tokens.weight").expect("embed");
    let es = store.take("model.embed_tokens.scales").expect("embed");
    let eb = store.take("model.embed_tokens.biases").expect("embed");
    let ids = Array::from_u32_slice(&[0], &[1, 1]).expect("ids");
    let w0 = ops::take(&ew, &ids, 0, &stream).unwrap();
    let s0 = ops::take(&es, &ids, 0, &stream).unwrap();
    let b0 = ops::take(&eb, &ids, 0, &stream).unwrap();
    let base = ops::dequantize(&w0, &s0, &b0, 64, 4, &stream).unwrap();
    base.eval().unwrap();

    let mut min_threshold = independent_threshold(&ew, &es, &eb, &base, &stream);
    for prefix in [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ] {
        let w = store.take(&format!("{prefix}.weight")).expect("tensor");
        let sc = store.take(&format!("{prefix}.scales")).expect("tensor");
        let bi = store.take(&format!("{prefix}.biases")).expect("tensor");
        min_threshold = min_threshold.min(independent_threshold(&w, &sc, &bi, &base, &stream));
    }
    let expected = (min_threshold - 1).clamp(1, 32);
    eprintln!(
        "calibrated deterministic width = {width} (independent boundary: first divergence \
         at M={min_threshold} -> expected {expected})"
    );
    assert_eq!(
        width, expected,
        "calibration does not match the independently measured row-stability boundary"
    );
}
