//! Batched-engine leak gate (SPEC §7.1, Phase 4 closeout): ≥1000 engine
//! iterations through the *full* production decode path — 16-deep batches,
//! paged KV, chunked prefill, preemption + resume, cancellation, and
//! finished-block recycling — then the kiln-mlx live-object counter must
//! return to its pre-model baseline. Complements `tests/leak.rs`, which
//! gates the Phase-3 single-request contiguous path.
//!
//! Own file (= own process): the live-object counter is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug, memory};
use kiln_models::LlamaModel;
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// Total engine iterations the gate must cover.
const ITERATIONS: usize = 1000;
/// Liveness bound per wave (a wave needs ~100 steps).
const MAX_WAVE_STEPS: usize = 3000;
/// The wave request that gets cancelled mid-stream.
const CANCEL_IDX: usize = 9;
/// Tokens `CANCEL_IDX` must have streamed before its flag is set.
const CANCEL_AFTER: usize = 6;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

struct Collected {
    tokens: Rc<RefCell<Vec<u32>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
    cancel: Arc<AtomicBool>,
}

fn submit(engine: &mut Engine<&LlamaModel>, prompt: &[u32], max_tokens: usize) -> Collected {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
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
        priority: Priority::Interactive,
        cancel: Arc::clone(&cancel),
        on_event: Box::new(move |event| {
            match event {
                SeqEvent::Token(token) => t.borrow_mut().push(token),
                SeqEvent::PrefixHit { .. } => {}
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    });
    Collected {
        tokens,
        finish,
        cancel,
    }
}

#[test]
fn thousand_iteration_batched_engine_returns_to_baseline() {
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

    let mut iterations = 0usize;
    let mut waves = 0usize;
    let mut cancelled = 0usize;
    let preemptions: u64;

    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&dir).expect("tokenizer loads");
        let text = "Paged attention splits the key-value cache into fixed-size blocks. ".repeat(24);
        let ids = tokenizer.encode(&text, true).expect("encodes");
        assert!(ids.len() >= 150, "need enough tokens to slice prompts");
        let short = &ids[..48]; // 1 chunk at chunk=48; 79 slots -> 3 blocks
        let long = &ids[..150]; // 4 prefill chunks; 181 slots -> 6 blocks

        // Pool 42 blocks: a 16-request wave (14 short + 2 long) peaks at
        // ~54 blocks, so the youngest requests are preempted and resumed
        // every wave; chunk 48 keeps the long prompts multi-chunk.
        let det_width = model
            .calibrate_deterministic_width(&stream)
            .expect("calibrates");
        let config = EngineConfig {
            num_blocks: 42,
            prefill_chunk: 48,
            deterministic_decode_width: det_width,
            ..EngineConfig::default()
        };
        let mut engine =
            Engine::new(&model, model.kv_dims(), config, Stream::gpu()).expect("engine builds");

        while iterations < ITERATIONS {
            waves += 1;
            let outs: Vec<Collected> = (0..16)
                .map(|k| {
                    let prompt = if k == 3 || k == 11 { long } else { short };
                    submit(&mut engine, prompt, 32)
                })
                .collect();
            let mut wave_steps = 0;
            while !engine.is_idle() {
                assert!(
                    wave_steps < MAX_WAVE_STEPS,
                    "wave {waves} livelocked after {wave_steps} steps"
                );
                engine.step().expect("engine step");
                wave_steps += 1;
                iterations += 1;
                let target = &outs[CANCEL_IDX];
                if !target.cancel.load(Ordering::Acquire)
                    && target.finish.borrow().is_none()
                    && target.tokens.borrow().len() >= CANCEL_AFTER
                {
                    target.cancel.store(true, Ordering::Release);
                }
            }
            for (k, out) in outs.iter().enumerate() {
                let summary = out.finish.borrow().clone().expect("request finished");
                if k == CANCEL_IDX {
                    assert_eq!(
                        summary.reason,
                        FinishKind::Cancelled,
                        "wave {waves} request {k}: {summary:?}"
                    );
                    cancelled += 1;
                } else {
                    assert_eq!(
                        summary.reason,
                        FinishKind::Length,
                        "wave {waves} request {k}: {summary:?}"
                    );
                }
            }
        }
        preemptions = engine.preemptions();
        assert!(
            preemptions >= 1,
            "the leak gate never exercised preemption (resize the pool)"
        );
        eprintln!(
            "leak gate (batched): {iterations} engine iterations over {waves} waves, \
             {preemptions} preemption(s), {cancelled} cancellation(s), \
             live objects during run: {}",
            debug::live_objects(),
        );
    }

    memory::clear_cache().expect("cache clears");
    let final_active = memory::active_memory().expect("memory query");
    eprintln!(
        "leak gate (batched): mlx active memory {baseline_active}B -> {final_active}B, \
         live objects {baseline_objects} -> {}, {preemptions} preemptions",
        debug::live_objects(),
    );
    assert_eq!(
        debug::live_objects(),
        baseline_objects,
        "live mlx handles did not return to baseline after {iterations} batched \
         iterations ({waves} waves)"
    );
}
