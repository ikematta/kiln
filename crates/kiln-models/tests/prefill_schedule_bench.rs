//! Miss-path cost of the fine-tail canonical schedule (Option B step 4,
//! PROGRESS 2026-07-04): cold prefill TTFT with the 64-token tail grid
//! vs the pre-Phase-5 single-tail-chunk schedule
//! (`prefill_fine_chunk >= prefill_chunk` restores it), same build, same
//! session. Pure measurement — prefix cache off, fresh engine per run,
//! medians over several runs; the only assertion is that runs finish.
//!
//! Release-only and `#[ignore]`d like the throughput gate.

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
use kiln_mlx::Stream;
use kiln_models::LlamaModel;
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
const RUNS: usize = 5;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

/// Cold TTFT (submit -> first sampled token) for one fresh engine.
fn cold_ttft(model: &LlamaModel, prompt: &[u32], fine: usize) -> f64 {
    let config = EngineConfig {
        num_blocks: 192,
        prefill_fine_chunk: fine,
        prefix_cache: false, // pure miss-path measurement
        ..EngineConfig::default()
    };
    let mut engine =
        Engine::new(model, model.kv_dims(), config, Stream::gpu()).expect("engine builds");
    let finish: Rc<RefCell<Option<FinishSummary>>> = Rc::new(RefCell::new(None));
    let f = Rc::clone(&finish);
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens: 4,
        sampling: SamplingOptions::default(),
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: std::collections::HashSet::new(),
        priority: Priority::Interactive,
        cancel: Arc::new(AtomicBool::new(false)),
        on_event: Box::new(move |event| {
            if let SeqEvent::Finished(summary) = event {
                *f.borrow_mut() = Some(summary);
            }
            true
        }),
    });
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let summary = finish.borrow().clone().expect("finished");
    assert_eq!(summary.reason, FinishKind::Length);
    summary.prefill_seconds
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

#[test]
#[ignore = "perf measurement: run explicitly in release during phase acceptance"]
fn fine_tail_schedule_miss_path_cost() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };
    if cfg!(debug_assertions) {
        panic!("schedule bench must run in release (--release)");
    }

    let stream = Stream::gpu();
    let model = LlamaModel::load(&dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("tokenizer loads");
    let mut ids = tokenizer
        .encode(
            &"Pottery is one of the oldest human inventions, and the kiln is \
              its oldest tool. "
                .repeat(160),
            true,
        )
        .expect("encodes");
    ids.truncate(2048);
    assert_eq!(ids.len(), 2048);

    eprintln!("prompt | tail fwds | fine=64 TTFT | old sched TTFT | delta");
    for &prompt_len in &[257_usize, 512, 1024, 2048] {
        let prompt = &ids[..prompt_len];
        // Warm-up (kernel caches, pool allocation) once per size.
        cold_ttft(&model, prompt, 64);
        let fine: Vec<f64> = (0..RUNS).map(|_| cold_ttft(&model, prompt, 64)).collect();
        let old: Vec<f64> = (0..RUNS)
            .map(|_| cold_ttft(&model, prompt, usize::MAX))
            .collect();
        let (fine_ms, old_ms) = (median(fine) * 1e3, median(old) * 1e3);
        let limit = prompt_len - 1;
        let tail = limit - limit / 2048 * 2048;
        eprintln!(
            "{prompt_len:>6} | {:>9} | {fine_ms:>9.1}ms | {old_ms:>11.1}ms | {:>+7.1}ms ({:+.1}%)",
            tail.div_ceil(64),
            fine_ms - old_ms,
            100.0 * (fine_ms - old_ms) / old_ms,
        );
    }

    // The tuning curve for the PM: coarser fine grids trade warm reuse
    // granularity (recompute up to fine-1 + increment tokens per turn)
    // for miss-path cost, at the worst-case prompt size.
    let prompt = &ids[..2048];
    let old_ms = median(
        (0..RUNS)
            .map(|_| cold_ttft(&model, prompt, usize::MAX))
            .collect(),
    ) * 1e3;
    eprintln!("fine grid @2048-token prompt | TTFT | delta vs old schedule");
    for &fine in &[64_usize, 128, 256] {
        cold_ttft(&model, prompt, fine);
        let ms = median((0..RUNS).map(|_| cold_ttft(&model, prompt, fine)).collect()) * 1e3;
        eprintln!(
            "  F={fine:<4} ({:>2} tail fwds) | {ms:>7.1}ms | {:>+6.1}ms ({:+.1}%)",
            2047_usize.div_ceil(fine),
            ms - old_ms,
            100.0 * (ms - old_ms) / old_ms,
        );
    }
}
