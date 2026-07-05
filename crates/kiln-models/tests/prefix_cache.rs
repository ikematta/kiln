//! Radix prefix cache + SSD tier on the real model (SPEC §12 Phase 5
//! acceptance):
//! - resubmitting a 2k-token prompt skips >= 95% of prefill and cuts TTFT
//!   by >= 5x, with bit-identical greedy output (CLAUDE.md: prefix caching
//!   must not change greedy outputs);
//! - a pipelined stop-token finish followed by an extended-prompt rerun
//!   serves no stale data (the settled-rows invariant, on real weights);
//! - restarting the engine over the same slab directory still hits, from
//!   SSD; a corrupted slab header is cleanly ignored (counter, no error).
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
    SamplingOptions, SeqEvent, SsdParams,
};
use kiln_mlx::{Stream, debug};
use kiln_models::LlamaModel;
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";

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

impl Outcome {
    fn summary(&self) -> FinishSummary {
        self.finish.borrow().clone().expect("request finished")
    }
}

fn request(prompt: &[u32], max_tokens: usize, stop: &[u32]) -> (EngineRequest, Outcome) {
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
        stop_tokens: stop.iter().copied().collect(),
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

fn drain(engine: &mut Engine<&LlamaModel>) {
    for _ in 0..200_000 {
        if engine.is_idle() {
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain");
}

fn run(
    engine: &mut Engine<&LlamaModel>,
    prompt: &[u32],
    max_tokens: usize,
    stop: &[u32],
) -> Outcome {
    let (request, outcome) = request(prompt, max_tokens, stop);
    engine.submit(request);
    drain(engine);
    outcome
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "kiln-prefix-cache-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("test dir");
    dir
}

#[test]
fn prefix_cache_and_ssd_tier() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let baseline = debug::live_objects();
    let ssd_dir = temp_dir("slabs");
    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");

        // A 2048-token prompt (SPEC §12 Phase 5 acceptance size).
        let mut long: Vec<u32> = tokenizer
            .encode(
                &"Paged attention splits the key-value cache into fixed-size \
                  blocks so requests can share memory. "
                    .repeat(120),
                true,
            )
            .expect("encodes");
        long.truncate(2048);
        assert_eq!(long.len(), 2048, "prompt must be exactly 2k tokens");

        let config = EngineConfig {
            num_blocks: 192,
            ..EngineConfig::default()
        };

        // --- 1) 2k resubmit: >= 95% prefill skip, TTFT >= 5x, bit-exact.
        {
            let mut engine = Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                .expect("engine builds");
            let cold = run(&mut engine, &long, 16, &[]);
            let cold_summary = cold.summary();
            assert_eq!(cold_summary.reason, FinishKind::Length);
            assert!(cold.hits.borrow().is_empty(), "first run cannot hit");

            let warm = run(&mut engine, &long, 16, &[]);
            let warm_summary = warm.summary();
            assert_eq!(
                warm.tokens.borrow().as_slice(),
                cold.tokens.borrow().as_slice(),
                "prefix caching changed greedy output (determinism contract)"
            );
            let hits = warm.hits.borrow().clone();
            assert_eq!(hits.len(), 1, "resubmit must hit the prefix cache");
            let (reused, from_ssd) = hits[0];
            assert!(!from_ssd, "pool-resident hit expected");
            assert_eq!(warm_summary.cached_prompt_tokens, reused);
            let skip = f64::from(reused) / long.len() as f64;
            let ratio = cold_summary.prefill_seconds / warm_summary.prefill_seconds.max(1e-9);
            eprintln!(
                "2k resubmit: reused {reused}/{} ({:.1}% skip), TTFT {:.1}ms -> {:.1}ms ({ratio:.1}x)",
                long.len(),
                skip * 100.0,
                cold_summary.prefill_seconds * 1e3,
                warm_summary.prefill_seconds * 1e3,
            );
            assert!(skip >= 0.95, "prefill skip {skip:.3} below 95%");
            assert!(ratio >= 5.0, "TTFT improvement {ratio:.2}x below 5x");
        }

        // --- 2) Pipelined stop, then (a) a full-containment rerun that
        // reuses the settled partial tail via copy-on-write, and (b) a
        // divergent extension, which under the determinism rule gets no
        // sub-chunk reuse at all (recomputing it in non-canonical shapes
        // could change KV bits) — both must be bit-identical to cold runs.
        {
            // Long enough that the settled range crosses a fine-grid
            // (prefill_fine_chunk) boundary, so the divergent-extension
            // case below exercises a real fine-aligned resume.
            let seed_prompt = tokenizer
                .encode(
                    &"The quick brown fox jumps over the lazy dog. ".repeat(16),
                    true,
                )
                .expect("encodes");
            assert!(seed_prompt.len() > kiln_engine::DEFAULT_PREFILL_FINE_CHUNK + 32);
            let mut probe_engine =
                Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                    .expect("engine builds");
            let probe = run(&mut probe_engine, &seed_prompt, 12, &[]);
            let generated = probe.tokens.borrow().clone();
            assert_eq!(generated.len(), 12);
            let stop = generated[5];
            drop(probe_engine);

            let mut warm_engine =
                Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                    .expect("engine builds");
            let stopped = run(&mut warm_engine, &seed_prompt, 24, &[stop]);
            let stopped_summary = stopped.summary();
            assert_eq!(stopped_summary.reason, FinishKind::Stop);
            assert!(
                warm_engine.pipelined_steps() > 0,
                "decode must run pipelined for this scenario to bite"
            );
            let settled = seed_prompt.len() - 1 + stopped_summary.completion_tokens as usize;

            // (a) Containment: rerunning the exact prompt reuses every
            // prefill position — full blocks plus the settled partial
            // tail (COW on the first append) — and recomputes nothing.
            let rerun = run(&mut warm_engine, &seed_prompt, 12, &[]);
            assert_eq!(
                rerun.tokens.borrow().as_slice(),
                generated.as_slice(),
                "containment rerun diverged after a pipelined stop"
            );
            let hits = rerun.hits.borrow().clone();
            assert_eq!(hits.len(), 1, "containment rerun must hit");
            assert_eq!(
                hits[0].0 as usize,
                seed_prompt.len() - 1,
                "containment must cover every prefill position"
            );
            assert!(hits[0].0 as usize <= settled, "match exceeded settled rows");

            // (b) Divergent extension: everything the stopped run computed
            // (including the stop token) plus fresh text. Served up to the
            // fine-grid boundary of the canonical schedule (Option B); the
            // remainder is recomputed in the cold schedule's exact shapes,
            // so bit-equality with a cold run must hold.
            let mut extended = seed_prompt.clone();
            extended.extend(&generated[..6]); // ..5 emitted + stop token
            extended.extend(
                tokenizer
                    .encode("Then the dog barked.", false)
                    .expect("encodes"),
            );
            let mut cold_engine =
                Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                    .expect("engine builds");
            let cold = run(&mut cold_engine, &extended, 12, &[]);
            let warm = run(&mut warm_engine, &extended, 12, &[]);
            assert_eq!(
                warm.tokens.borrow().as_slice(),
                cold.tokens.borrow().as_slice(),
                "stale data after a pipelined stop (settled-rows invariant)"
            );
            let ext_hits = warm.hits.borrow().clone();
            assert_eq!(ext_hits.len(), 1, "F-aligned overlap must be served");
            let fine = kiln_engine::DEFAULT_PREFILL_FINE_CHUNK;
            let served = ext_hits[0].0 as usize;
            assert!(
                served > 0 && served.is_multiple_of(fine),
                "divergent overlap must resume on the fine grid: {served}"
            );
            assert!(served <= settled, "match exceeded settled rows");
            eprintln!(
                "pipelined stop on real weights: settled {settled}, containment rerun \
                 reused {}, divergent extension resumed at {served}, both == cold",
                hits[0].0
            );
        }

        // --- 3) SSD tier: flush, "restart" (new engine, same directory),
        // hit from SSD with identical output.
        let ssd_params = SsdParams {
            dir: ssd_dir.clone(),
            max_bytes: 4 << 30,
            fingerprint: "prefix-cache-test-model".to_owned(),
        };
        let ssd_config = EngineConfig {
            num_blocks: 192,
            ssd: Some(ssd_params.clone()),
            ..EngineConfig::default()
        };
        let mut medium: Vec<u32> = long[..512].to_vec();
        // Distinct last token so this prompt's stream is its own.
        let cold_tokens;
        {
            let mut engine =
                Engine::new(&model, model.kv_dims(), ssd_config.clone(), Stream::gpu())
                    .expect("engine builds");
            assert!(engine.ssd_error().is_none(), "{:?}", engine.ssd_error());
            let cold = run(&mut engine, &medium, 12, &[]);
            cold_tokens = cold.tokens.borrow().clone();
            engine.flush_prefix_cache();
            let stats = engine.cache_stats();
            assert!(
                stats.ssd_writes_total >= (512 / kiln_engine::DEFAULT_BLOCK_SIZE) as u64,
                "flush must persist the donated prefix: {stats:?}"
            );
            assert_eq!(stats.ssd_writes_failed_total, 0);
        }
        {
            // Restart: fresh engine + pools; radix warm-loads lazily from
            // the slab index on the prefix walk.
            let mut engine =
                Engine::new(&model, model.kv_dims(), ssd_config.clone(), Stream::gpu())
                    .expect("engine builds");
            let warm = run(&mut engine, &medium, 12, &[]);
            assert_eq!(
                warm.tokens.borrow().as_slice(),
                cold_tokens.as_slice(),
                "SSD round-trip changed greedy output"
            );
            let hits = warm.hits.borrow().clone();
            assert_eq!(hits.len(), 1, "restart must hit from SSD");
            let (reused, from_ssd) = hits[0];
            assert!(from_ssd, "hit must come from the cold tier");
            assert!(
                reused as usize >= 512 / 32 * 32 - 32,
                "SSD hit too small: {reused}"
            );
            let stats = engine.cache_stats();
            assert!(stats.ssd_reads_total > 0);
            eprintln!(
                "SSD restart: reused {reused} tokens from disk, reads {}, output bit-exact",
                stats.ssd_reads_total
            );
        }

        // --- 4) Corrupt a slab header: cleanly ignored (counter bumps, no
        // error, request just prefills cold and re-persists).
        {
            let slab = std::fs::read_dir(&ssd_dir)
                .expect("slab dir")
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("slab-"))
                })
                .expect("a slab file exists");
            let mut bytes = std::fs::read(&slab).expect("read slab");
            bytes[0] ^= 0xff; // break the magic
            std::fs::write(&slab, &bytes).expect("corrupt slab");

            let mut engine =
                Engine::new(&model, model.kv_dims(), ssd_config.clone(), Stream::gpu())
                    .expect("engine builds");
            let stats = engine.cache_stats();
            assert!(
                stats.ssd_fingerprint_rejects_total >= 1,
                "corrupted slab must be counted: {stats:?}"
            );
            let after = run(&mut engine, &medium, 12, &[]);
            assert_eq!(
                after.tokens.borrow().as_slice(),
                cold_tokens.as_slice(),
                "corrupted slab affected generation"
            );
            let hits = after.hits.borrow().clone();
            let ssd_hit = hits.iter().any(|&(_, from_ssd)| from_ssd);
            assert!(!ssd_hit, "corrupted slab must not serve blocks: {hits:?}");
            eprintln!(
                "corrupt slab header: ignored (rejects {}), cold run bit-exact",
                stats.ssd_fingerprint_rejects_total
            );
        }

        // Keep `medium` alive for clarity of the scenario flow above.
        medium.clear();
    }
    std::fs::remove_dir_all(&ssd_dir).ok();
    assert_eq!(
        debug::live_objects(),
        baseline,
        "leaked mlx objects across prefix-cache scenarios"
    );
}
