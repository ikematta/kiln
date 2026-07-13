//! Multi-turn conversation growth vs the prefix cache — Phase 5 gate
//! investigation (see the DECISION NEEDED entry in PROGRESS.md).
//!
//! The determinism rule shipped with Phase 5 serves a cache hit either as
//! full containment (exact rerun) or trimmed to canonical `prefill_chunk`
//! (2048) boundaries. A realistic conversation grows its prefix by a
//! sub-chunk increment every turn, so it is never contained and never
//! reaches a chunk boundary until ~2k tokens accumulate. This test
//! measures the actual per-turn prefill skip for that pattern — the
//! numbers behind the granularity decision — and asserts only
//! correctness: every warm turn must be bit-identical to a cache-cold
//! run of the same prompt, and every served hit must obey the rule
//! (chunk-aligned or containment).
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable. Single `#[test]` because the kiln-mlx live-object counter
//! is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug};
use kiln_models::LlamaModel;
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
const BLOCK: usize = 32;
const FINE: usize = kiln_engine::DEFAULT_PREFILL_FINE_CHUNK;
const TURN_GENERATION: usize = 64;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

struct Outcome {
    tokens: Rc<RefCell<Vec<u32>>>,
    hits: Rc<RefCell<Vec<(u32, bool)>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
}

fn request(prompt: &[u32], max_tokens: usize) -> (EngineRequest, Outcome) {
    let outcome = Outcome {
        tokens: Rc::new(RefCell::new(Vec::new())),
        hits: Rc::new(RefCell::new(Vec::new())),
        finish: Rc::new(RefCell::new(None)),
    };
    let (t, h, f) = (
        Rc::clone(&outcome.tokens),
        Rc::clone(&outcome.hits),
        Rc::clone(&outcome.finish),
    );
    let request = EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: std::collections::HashSet::new(),
        grammar: None,
        priority: Priority::Interactive,
        cancel: Arc::new(AtomicBool::new(false)),
        on_event: Box::new(move |event| {
            match event {
                SeqEvent::Token(token) => t.borrow_mut().push(token),
                SeqEvent::PrefixHit { tokens, from_ssd } => {
                    h.borrow_mut().push((tokens, from_ssd));
                }
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    };
    (request, outcome)
}

fn run(engine: &mut Engine<&LlamaModel>, prompt: &[u32], max_tokens: usize) -> Outcome {
    let (request, outcome) = request(prompt, max_tokens);
    engine.submit(request);
    for _ in 0..200_000 {
        if engine.is_idle() {
            let summary = outcome.finish.borrow().clone().expect("finished");
            assert_eq!(summary.reason, FinishKind::Length, "{summary:?}");
            return outcome;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain");
}

/// Encodes `text` repeated until at least `tokens` ids, truncated exactly.
fn filler(tokenizer: &Tokenizer, text: &str, tokens: usize) -> Vec<u32> {
    let mut ids = tokenizer
        .encode(&text.repeat(tokens / 4 + 2), false)
        .expect("encodes");
    assert!(ids.len() >= tokens, "filler too short for {tokens}");
    ids.truncate(tokens);
    ids
}

#[test]
fn multiturn_incremental_growth_prefill_skip() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let baseline = debug::live_objects();
    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");

        // A ~600-token opening prompt (system prompt + first user turn),
        // then 8 turns, each appending the assistant's reply plus a
        // 90-200-token user increment — realistic conversation growth.
        // The final turns push the accumulated prefix past one 2048
        // chunk, so the report shows both regimes of the current rule.
        let mut prompt = filler(
            &tokenizer,
            "You are a careful assistant for a pottery studio. Answer with \
             concrete steps and temperatures. ",
            600,
        );
        let increments: [usize; 8] = [180, 120, 160, 90, 200, 150, 140, 110];

        let config = EngineConfig::default(); // prefix cache on, chunk 2048
        let mut warm = Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
            .expect("engine builds");

        eprintln!(
            "turn | prompt | reused | skip%  | F=32 would | F=256 would | warm TTFT | cold TTFT"
        );
        let mut donor_settled = 0_usize;
        for (turn, &increment) in increments.iter().enumerate() {
            // Warm: sequential turn against the accumulated cache.
            let outcome = run(&mut warm, &prompt, TURN_GENERATION);
            let summary = outcome.finish.borrow().clone().expect("finished");
            let generated = outcome.tokens.borrow().clone();

            // Cold reference: fresh engine, same prompt. The determinism
            // contract must hold turn by turn (CLAUDE.md: prefix caching
            // must not change greedy outputs).
            let mut cold_engine =
                Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                    .expect("engine builds");
            let cold = run(&mut cold_engine, &prompt, TURN_GENERATION);
            let cold_summary = cold.finish.borrow().clone().expect("finished");
            assert_eq!(
                generated,
                cold.tokens.borrow().clone(),
                "turn {turn}: warm output diverged from cold (determinism contract)"
            );

            // The rule (Option B): a served hit resumes on a boundary of
            // the canonical schedule — fine-grid multiples in the final
            // super-chunk (2048 multiples are also fine multiples) — or is
            // full containment.
            let reused = summary.cached_prompt_tokens as usize;
            assert!(
                reused.is_multiple_of(FINE) || reused == prompt.len() - 1,
                "turn {turn}: reuse {reused} violates the resumable-boundary rule"
            );

            // What finer canonical grids would have served this turn:
            // the raw block-aligned overlap with the previous donor,
            // trimmed to the hypothetical grid.
            let overlap = donor_settled.min(prompt.len() - 1);
            let hyp = |grid: usize| overlap / grid * grid;
            eprintln!(
                "{turn:>4} | {:>6} | {reused:>6} | {:>5.1}% | {:>10} | {:>11} | {:>8.1}ms | {:>8.1}ms",
                prompt.len(),
                100.0 * reused as f64 / prompt.len() as f64,
                hyp(BLOCK),
                hyp(256),
                summary.prefill_seconds * 1e3,
                cold_summary.prefill_seconds * 1e3,
            );

            // Next turn's prompt: reply + user increment.
            donor_settled = prompt.len() - 1 + generated.len();
            prompt.extend(&generated);
            prompt.extend(filler(
                &tokenizer,
                "Now the glaze crawled near the rim and the kiln shelves show \
                 some flaking wash. What should change in the next firing? ",
                increment,
            ));
        }
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "leaked mlx objects across multi-turn scenario"
    );
}
