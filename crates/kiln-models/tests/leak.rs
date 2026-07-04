//! Leak gate (SPEC §7.1, CLAUDE.md): 1000 decode iterations, then the
//! kiln-mlx live-object counter must return to its pre-model baseline once
//! everything is dropped. Also reports MLX active-memory before/after as a
//! secondary signal (buffer cache is cleared first; the counter is the
//! authoritative gate).

#![cfg(feature = "metal")]

use std::path::PathBuf;

use kiln_engine::Sampler;
use kiln_mlx::{Stream, debug, memory};
use kiln_models::{LlamaModel, generate};
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
const ITERATIONS: usize = 1000;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

#[test]
fn thousand_iteration_decode_returns_to_baseline() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    kiln_mlx::init();
    let baseline_objects = debug::live_objects();
    let baseline_active = memory::active_memory().expect("memory query");

    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&dir).expect("tokenizer loads");
        let prompt = tokenizer
            .encode("Pottery is one of the oldest human inventions", true)
            .expect("encodes");

        let mut sampler = Sampler::greedy();
        let output = generate(
            &model,
            &prompt,
            ITERATIONS,
            |logprobs, s| sampler.sample(logprobs, s),
            &stream,
        )
        .expect("generates");
        assert_eq!(output.tokens.len(), ITERATIONS);
        eprintln!(
            "leak gate: {ITERATIONS} decode iterations at {:.1} tok/s, \
             live objects during run: {}",
            output.decode_tokens_per_sec(),
            debug::live_objects(),
        );
    }

    memory::clear_cache().expect("cache clears");
    let final_active = memory::active_memory().expect("memory query");
    eprintln!(
        "leak gate: mlx active memory {baseline_active}B -> {final_active}B, \
         live objects {} -> {}",
        baseline_objects,
        debug::live_objects(),
    );
    assert_eq!(
        debug::live_objects(),
        baseline_objects,
        "live mlx handles did not return to baseline after 1k iterations"
    );
}
