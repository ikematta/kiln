//! Draft-model coexistence (SPEC §6.5): the pinned Qwen3-0.6B-4bit
//! checkpoint loads as a draft alongside a larger pinned target
//! (Llama-3.2-1B-4bit) in one process, sharing the Metal device/stream,
//! with its own weights and its own KV pool inside the same memory
//! accounting. Deliberately a cross-tokenizer pair, on purpose on both
//! sides of Phase 8: part 1 proved loading isolation; with the part-2
//! verify loop live, attaching this drafter at the ENGINE level (below
//! the worker's `check_draft_compat` gate, which rejects the pair — see
//! kiln-worker/tests/draft.rs) doubles as an adversarial-drafter
//! invariance case — its proposals are near-guaranteed garbage for the
//! target, verification must reject them, and greedy output must not
//! move by a bit.
//!
//! Same-device invariants asserted (device-independent tier — blocking
//! on CI):
//! - target greedy output is BIT-IDENTICAL before and after the draft's
//!   weights and materialized KV pool are resident, and with the drafter
//!   attached to the engine — two models in one Metal heap must not
//!   perturb each other's numerics (the kernel-class-divergence risk
//!   this build keeps meeting: ADR 0002/0004);
//! - the draft pool's bytes survive target generation untouched, and its
//!   accounting matches its geometry exactly (no cross-pool bleed);
//! - the combined measured footprint fits the SPEC §2.3 machine budget
//!   (0.80 x unified memory) — the coexistence claim, measured;
//! - the kiln-mlx live-object counter returns to baseline once both
//!   models drop (leak gate, CLAUDE.md).
//!
//! Single `#[test]` because the live-object counter is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    BlockManager, DEFAULT_GAMMA, DraftError, Drafter, Engine, EngineConfig, EngineRequest,
    FinishKind, PenaltyOptions, Priority, SamplingOptions, SeqEvent, WriteRun,
};
use kiln_mlx::{Dtype, Stream, debug, memory, ops};
use kiln_models::{AnyModel, DraftModel, DraftPoolSpec};

const TARGET_NAME: &str = "llama-3.2-1b-4bit";
const DRAFT_NAME: &str = "qwen3-0.6b-4bit";

/// Small draft pool for the test: the accounting math is geometry-exact
/// at any size, and 64 blocks keeps the materialized pool CI-friendly.
/// (The worker sizes the real pool to the target pool's token capacity.)
const DRAFT_BLOCKS: usize = 64;
const BLOCK_SIZE: usize = 32;

const MAX_TOKENS: usize = 32;

fn model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(name);
    dir.join("config.json").is_file().then_some(dir)
}

/// `.safetensors` bytes, the `StaticInfo.weights_bytes` convention.
fn fs_weights_bytes(dir: &PathBuf) -> u64 {
    std::fs::read_dir(dir)
        .expect("model dir readable")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
                .then(|| entry.metadata().ok().map(|meta| meta.len()))?
        })
        .sum()
}

fn machine_memory_bytes() -> u64 {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("sysctl runs");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("hw.memsize parses")
}

/// Greedy engine run of `prompt` on a fresh engine over `model`; returns
/// the generated tokens and the engine's allocated KV-pool bytes at
/// finish. `drafter` is attached before any submit when given.
fn run_target(
    model: &AnyModel,
    prompt: &[u32],
    drafter: Option<Box<dyn Drafter>>,
) -> (Vec<u32>, u64) {
    let config = EngineConfig {
        prefix_cache: false, // keep every run cold: identical shapes
        ..EngineConfig::default()
    };
    let mut engine =
        Engine::new(model, model.kv_dims(), config, Stream::gpu()).expect("engine builds");
    if let Some(drafter) = drafter {
        engine.set_drafter(drafter);
    }
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens: MAX_TOKENS,
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
                SeqEvent::PrefixHit { .. } => {}
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    });
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let summary = finish.borrow().clone().expect("request finished");
    assert_eq!(
        summary.reason,
        FinishKind::Length,
        "expected a full-length greedy run: {summary:?}"
    );
    let kv_bytes = engine.kv_allocated_bytes();
    (tokens.borrow().clone(), kv_bytes)
}

#[test]
fn draft_model_coexists_with_target() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let (Some(target_dir), Some(draft_dir)) = (model_dir(TARGET_NAME), model_dir(DRAFT_NAME))
    else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {TARGET_NAME}/{DRAFT_NAME} missing");
        return;
    };

    kiln_mlx::init();
    let baseline_objects = debug::live_objects();

    {
        let stream = Stream::gpu();
        let target = AnyModel::load(&target_dir, &stream).expect("target loads");
        // Crosses a block boundary and prefills a ragged tail piece.
        let prompt: Vec<u32> = (1..=40).collect();

        // --- Reference: target alone.
        let (tokens_alone, target_kv_bytes) = run_target(&target, &prompt, None);
        assert_eq!(tokens_alone.len(), MAX_TOKENS);

        // --- Draft loads alongside, on the same device/stream.
        let pool = DraftPoolSpec {
            block_size: BLOCK_SIZE,
            num_blocks: DRAFT_BLOCKS,
        };
        let mut draft = DraftModel::load(&draft_dir, pool, &stream).expect("draft loads");
        let draft_weights = Drafter::memory(&draft).weights_bytes;
        assert_eq!(
            draft_weights,
            fs_weights_bytes(&draft_dir),
            "draft weights accounting must follow the StaticInfo convention"
        );
        assert_eq!(
            Drafter::memory(&draft).kv_allocated_bytes,
            0,
            "draft pool must stay unmaterialized until first written"
        );

        // --- Materialize the draft pool and write a sentinel pattern.
        let dims = draft.kv_dims();
        draft
            .kv_mut()
            .ensure_pools(Dtype::Float16, &stream)
            .expect("draft pools materialize");
        let expected_pool_bytes = 2 // K and V
            * dims.layers as u64
            * DRAFT_BLOCKS as u64
            * dims.kv_heads as u64
            * BLOCK_SIZE as u64
            * dims.head_dim as u64
            * 2; // f16
        assert_eq!(
            Drafter::memory(&draft).kv_allocated_bytes,
            expected_pool_bytes,
            "draft pool bytes must match its geometry exactly"
        );
        assert_eq!(
            Drafter::memory(&draft).kv_used_bytes,
            0,
            "no draft sequence holds blocks yet"
        );
        // A scratch manager mints a valid block id for the sentinel write
        // (PagedKv holds no ownership state; ids are plain indices).
        let mut scratch = BlockManager::new(DRAFT_BLOCKS, BLOCK_SIZE).expect("scratch manager");
        let block = scratch.allocate().expect("block 0");
        let n = f64::from(dims.kv_heads * BLOCK_SIZE as i32 * dims.head_dim);
        let pattern = ops::arange(0.0, n, 1.0, Dtype::Float32, &stream).expect("pattern");
        let pattern = ops::astype(&pattern, Dtype::Float16, &stream).expect("f16 pattern");
        let pattern = ops::reshape(
            &pattern,
            &[1, dims.kv_heads, BLOCK_SIZE as i32, dims.head_dim],
            &stream,
        )
        .expect("pattern shape");
        let runs = [WriteRun {
            block,
            row_start: 0,
            src_start: 0,
            len: BLOCK_SIZE as i32,
        }];
        for layer in 0..dims.layers {
            draft
                .kv_mut()
                .write(layer, &runs, &pattern, &pattern, &stream)
                .expect("sentinel write");
        }
        let sentinel_before = draft
            .kv()
            .read_block_bytes(block, &stream)
            .expect("sentinel readback");

        // --- Target rerun with the draft resident: bit-identical output.
        let (tokens_beside_draft, _) = run_target(&target, &prompt, None);
        assert_eq!(
            tokens_beside_draft, tokens_alone,
            "target greedy output changed with the draft resident \
             (cross-model weight/KV contamination)"
        );

        // --- Draft pool untouched by target generation.
        let sentinel_after = draft
            .kv()
            .read_block_bytes(block, &stream)
            .expect("sentinel readback");
        assert_eq!(
            sentinel_before, sentinel_after,
            "target generation altered draft pool bytes"
        );

        // --- Drafter lifecycle on the real DraftModel (Phase 8 part 2:
        // propose really decodes on the draft's own pool now).
        let err = draft
            .propose(1, &[], DEFAULT_GAMMA, &stream)
            .expect_err("propose before begin");
        assert!(matches!(err, DraftError::UnknownSeq(1)), "{err}");
        draft.begin(1, &prompt, &stream).expect("begin");
        let proposal = draft
            .propose(1, &[], DEFAULT_GAMMA, &stream)
            .expect("propose");
        assert_eq!(
            proposal.len(),
            DEFAULT_GAMMA,
            "a healthy drafter proposes the full gamma"
        );
        let draft_vocab = 151_936; // qwen3 config.json vocab_size
        assert!(
            proposal.iter().all(|&t| t < draft_vocab),
            "proposed ids must come from the draft's logits width: {proposal:?}"
        );
        assert!(
            Drafter::memory(&draft).kv_used_bytes > 0,
            "a proposing sequence must hold draft pool blocks"
        );
        // Reconcile feed-through: accept the first proposed token plus a
        // deliberately different bonus, then propose again — the drafter
        // truncates its speculated tail (O(1) block release) and continues
        // from the corrected context.
        let bonus = if proposal[1] == 0 { 1 } else { 0 };
        let proposal2 = draft
            .propose(1, &[proposal[0], bonus], DEFAULT_GAMMA, &stream)
            .expect("propose after partial accept");
        assert_eq!(proposal2.len(), DEFAULT_GAMMA);
        draft.release(1);
        assert_eq!(
            Drafter::memory(&draft).kv_used_bytes,
            0,
            "release must return the sequence's draft blocks"
        );
        let err = draft
            .propose(1, &[], DEFAULT_GAMMA, &stream)
            .expect_err("propose after release");
        assert!(matches!(err, DraftError::UnknownSeq(1)), "{err}");

        // --- Measured coexistence within the SPEC §2.3 machine budget.
        let target_weights = fs_weights_bytes(&target_dir);
        let draft_kv_bytes = Drafter::memory(&draft).kv_allocated_bytes;
        let footprint = target_weights + draft_weights + target_kv_bytes + draft_kv_bytes;
        let budget = (machine_memory_bytes() as f64 * 0.80) as u64;
        let active = memory::active_memory().expect("memory query");
        eprintln!(
            "coexistence: target weights {target_weights}B + draft weights {draft_weights}B \
             + target kv {target_kv_bytes}B + draft kv {draft_kv_bytes}B \
             = {footprint}B; budget {budget}B; mlx active {active}B"
        );
        assert!(
            footprint <= budget,
            "combined footprint {footprint}B exceeds the 0.80 machine budget {budget}B"
        );

        // --- Attached to the engine (as the worker wires it): memory
        // report flows through, and target admission/generation still
        // holds bit-identically.
        let expected_memory = Drafter::memory(&draft);
        let config = EngineConfig {
            prefix_cache: false,
            ..EngineConfig::default()
        };
        let mut engine =
            Engine::new(&target, target.kv_dims(), config, Stream::gpu()).expect("engine builds");
        engine.set_drafter(Box::new(draft));
        assert_eq!(engine.drafter_memory(), Some(expected_memory));
        drop(engine);
        let boxed: Box<dyn Drafter> =
            Box::new(DraftModel::load(&draft_dir, pool, &stream).expect("draft reloads"));
        let (tokens_with_drafter, _) = run_target(&target, &prompt, Some(boxed));
        assert_eq!(
            tokens_with_drafter, tokens_alone,
            "attaching a drafter changed target greedy output"
        );
    }

    // --- Leak gate: both models, both pools, every engine dropped.
    memory::clear_cache().expect("cache clears");
    assert_eq!(
        debug::live_objects(),
        baseline_objects,
        "live mlx handles did not return to baseline after draft coexistence"
    );
}
