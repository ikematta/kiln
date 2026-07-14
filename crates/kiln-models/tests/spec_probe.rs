//! Characterization instrument for the Phase 8 part 2 finding (PROGRESS
//! 2026-07-13): on qwen2.5-0.5b-4bit, a gamma=4 verify segment (5 query
//! rows) leaves the pinned MLX's fused-SDPA dispatch envelope —
//! `(qL <= 8) && (qL * gqa_factor <= 32)`, and qwen2.5-0.5b's gqa_factor
//! is 7, so 5 x 7 = 35 > 32 — silently taking the UNFUSED composed-op
//! attention, whose reduction order differs at ulp level from the qL=1
//! vector kernel plain decode uses. A measured 1-fp16-ULP argmax race at
//! chat-basic generated index 33 (logits 16.765625 vs 16.75 for tokens
//! 9645/2585) flips under that class change; at qL <= 4 (gamma <= 3,
//! 4 x 7 = 28 <= 32) the verify forward is bit-identical to plain decode
//! on the measured lanes and every qwen2.5 fixture passes.
//!
//! These probes PRINT the evidence rather than asserting a bar (the bar
//! lives in spec_decode.rs and stays unweakened):
//! - `characterize_qwen25_divergence`: divergence indices per fixture
//!   under adversarial/oracle drafters at gamma 1/2/4, plus the
//!   self-draft/adversarial matrix for the models alphabetically after
//!   qwen2.5. Stable across fresh processes (4/4 identical runs), but
//!   NOT across differing in-process allocation histories — the same
//!   binary produced divergence in one suite layout and none in another,
//!   so no fixture run can certify these shapes at all.
//! - `measure_qwen25_divergence_gap`: replays the plain path to the
//!   divergence position (asserting fixture parity as a harness check),
//!   reads the top-2 gap, and reruns the same state through 2/3/4/5-row
//!   verify shapes — the bit boundary sits exactly at 5 rows.
//!
//! Keep in sync with the DECISION NEEDED entry in PROGRESS.md; delete
//! once the resolution lands and the evidence is recorded in an ADR.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    DraftError, Drafter, DrafterMemory, Engine, EngineConfig, EngineRequest, FinishKind,
    PenaltyOptions, Priority, SamplingOptions, SeqEvent,
};
use kiln_mlx::Stream;
use kiln_models::{AnyModel, DraftModel, DraftPoolSpec};
use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

const PINNED_DATE_STRING: &str = "26 Jul 2024";

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    prompt: String,
    chat_template: bool,
    max_tokens: usize,
    expected_token_ids: Vec<u32>,
    #[allow(dead_code)]
    mlx_lm_version: String,
    #[allow(dead_code)]
    weights_revision: String,
}

fn golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

struct AdversarialDrafter {
    seqs: HashSet<u64>,
    proposals: u64,
}

impl Drafter for AdversarialDrafter {
    fn memory(&self) -> DrafterMemory {
        DrafterMemory::default()
    }
    fn begin(&mut self, seq: u64, _p: &[u32], _s: &Stream) -> Result<(), DraftError> {
        self.seqs.insert(seq);
        Ok(())
    }
    fn propose(
        &mut self,
        seq: u64,
        _c: &[u32],
        gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        if !self.seqs.contains(&seq) {
            return Err(DraftError::UnknownSeq(seq));
        }
        self.proposals += 1;
        Ok([11u32, 23, 5, 42]
            .iter()
            .cycle()
            .skip((self.proposals as usize) % 4)
            .take(gamma)
            .copied()
            .collect())
    }
    fn release(&mut self, seq: u64) {
        self.seqs.remove(&seq);
    }
}

/// Proposes exactly the fixture's own continuation (perfect oracle).
struct OracleDrafter {
    expected: Vec<u32>,
    committed: usize,
}

impl Drafter for OracleDrafter {
    fn memory(&self) -> DrafterMemory {
        DrafterMemory::default()
    }
    fn begin(&mut self, _seq: u64, _p: &[u32], _s: &Stream) -> Result<(), DraftError> {
        self.committed = 0;
        Ok(())
    }
    fn propose(
        &mut self,
        _seq: u64,
        committed: &[u32],
        gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        self.committed += committed.len();
        let at = self.committed;
        Ok(self
            .expected
            .iter()
            .skip(at)
            .take(gamma.min(self.expected.len().saturating_sub(at)))
            .copied()
            .collect())
    }
    fn release(&mut self, _seq: u64) {}
}

fn run(
    model: &AnyModel,
    det_width: usize,
    gamma: usize,
    drafter: Option<Box<dyn Drafter>>,
    prompt: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let mut config = EngineConfig {
        num_blocks: 256,
        deterministic_decode_width: det_width,
        gamma,
        ..EngineConfig::default()
    };
    if model.monolithic_prefill_required() {
        config.prefill_fine_chunk = config.prefill_chunk;
    }
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
        max_tokens,
        sampling: SamplingOptions::default(),
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: HashSet::new(),
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
    assert_eq!(
        finish.borrow().clone().expect("finished").reason,
        FinishKind::Length
    );
    tokens.borrow().clone()
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn load_fixtures(model_name: &str, model_dir: &PathBuf) -> Vec<(String, Fixture, Vec<u32>)> {
    let tokenizer = Tokenizer::from_model_dir(model_dir).expect("tokenizer");
    let template = ChatTemplate::from_model_dir(model_dir).expect("template");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(golden_root().join(model_name))
        .expect("fixtures")
        .filter_map(|e| {
            let p = e.expect("entry").path();
            (p.extension().is_some_and(|x| x == "json")).then_some(p)
        })
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_owned();
            let fixture: Fixture =
                serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");
            let ids = if fixture.chat_template {
                let rendered = template
                    .render_with(
                        &[ChatMessage::text("user", fixture.prompt.clone())],
                        true,
                        &[("date_string", Value::from(PINNED_DATE_STRING))],
                    )
                    .expect("render");
                tokenizer.encode(&rendered, false).expect("encode")
            } else {
                tokenizer.encode(&fixture.prompt, true).expect("encode")
            };
            (name, fixture, ids)
        })
        .collect()
}

/// Plain-path replication + gap measurement at the divergence position.
/// Feeds the fixture's prompt (one 41-row piece — the canonical schedule
/// for a 42-token prompt) then decodes the expected tokens one at a time
/// through forward_step exactly like the engine's plain path, asserting
/// each argmax reproduces the fixture. At the divergence position it
/// reports the raw-logit and logprob gap between the fixture's token and
/// the speculation run's token, and whether a 5-row verify-shaped forward
/// from this SAME (plain-built) KV state flips row 0.
#[test]
fn measure_qwen25_divergence_gap() {
    use kiln_engine::{
        BlockManager, BlockTable, KvSpec, PagedKv, SeqStep, StepBatch, StepInput, StepModel,
        WriteRun,
    };
    use kiln_mlx::ops;

    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal");
        return;
    }
    let Some(root) = std::env::var_os("KILN_TEST_MODELS").map(PathBuf::from) else {
        eprintln!("skipping: KILN_TEST_MODELS unset");
        return;
    };
    kiln_mlx::init();
    let name = "qwen2.5-0.5b-4bit";
    let dir = root.join(name);
    let stream = Stream::gpu();
    let model = AnyModel::load(&dir, &stream).expect("model");
    let fixtures = load_fixtures(name, &dir);
    let (_, fixture, prompt) = fixtures
        .iter()
        .find(|(n, _, _)| n == "chat-basic")
        .expect("chat-basic fixture");
    let expected = &fixture.expected_token_ids;
    const DIVERGENCE: usize = 33;
    const SPEC_TOKEN: u32 = 2585; // what the gamma=4 runs sampled there
    let fixture_token = expected[DIVERGENCE];

    let dims = model.kv_dims();
    let mut mgr = BlockManager::new(64, 32).expect("mgr");
    let mut kv = PagedKv::new(KvSpec {
        layers: dims.layers,
        kv_heads: dims.kv_heads,
        head_dim: dims.head_dim,
        num_blocks: 64,
        block_size: 32,
    });
    let mut table = BlockTable::new();
    let forward = |table: &mut BlockTable,
                   kv: &mut PagedKv,
                   mgr: &mut BlockManager,
                   ids: Vec<u32>,
                   sample_rows: i32|
     -> kiln_mlx::Array {
        let len = ids.len();
        let offset = table.num_tokens();
        table.append_tokens(mgr, len).expect("append");
        let block_size = mgr.block_size();
        let mut writes = Vec::new();
        let mut pos = offset;
        while pos < offset + len {
            let run = (block_size - pos % block_size).min(offset + len - pos);
            writes.push(WriteRun {
                block: table.blocks()[pos / block_size],
                row_start: (pos % block_size) as i32,
                src_start: (pos - offset) as i32,
                len: run as i32,
            });
            pos += run;
        }
        let step = SeqStep {
            len: len as i32,
            offset: offset as i32,
            sample_rows,
            blocks: table.blocks().to_vec(),
            writes,
            paged_attn: None,
        };
        let batch = StepBatch {
            input: StepInput::Ids(ids),
            seqs: vec![step],
            pad_rows: 0,
        };
        let logits = model.forward_step(&batch, kv, &stream).expect("forward");
        let state = kv.state();
        let mut outputs: Vec<&kiln_mlx::Array> = state;
        match logits {
            Some(logits) => {
                outputs.push(&logits);
                kiln_mlx::eval(&outputs).expect("eval");
                logits
            }
            None => {
                kiln_mlx::eval(&outputs).expect("eval");
                ops::zeros(&[1, 1, 1], kiln_mlx::Dtype::Float32, &stream).expect("zeros")
            }
        }
    };

    // Prefill prompt[..n-1] (single 41-row canonical piece for 42 tokens).
    let n = prompt.len();
    drop(forward(
        &mut table,
        &mut kv,
        &mut mgr,
        prompt[..n - 1].to_vec(),
        0,
    ));
    // Plain decode: feed last prompt token then expected tokens.
    let mut feed = prompt[n - 1];
    for (i, &expected_token) in expected.iter().enumerate().take(DIVERGENCE + 1) {
        let logits = forward(&mut table, &mut kv, &mut mgr, vec![feed], 1);
        let vocab = logits.dim(2);
        let row = ops::reshape(&logits, &[1, vocab], &stream).expect("reshape");
        let logprobs = ops::subtract(
            &row,
            &ops::logsumexp(&row, true, &stream).expect("lse"),
            &stream,
        )
        .expect("sub");
        let sampled = ops::argmax(&logprobs, -1, false, &stream)
            .expect("argmax")
            .item_u32()
            .expect("item");
        if i == DIVERGENCE {
            let read = |token: u32, arr: &kiln_mlx::Array| -> f32 {
                let v = ops::slice(arr, &[0, token as i32], &[1, token as i32 + 1], &stream)
                    .expect("slice");
                let v = ops::astype(&v, kiln_mlx::Dtype::Float32, &stream).expect("cast");
                v.eval().expect("eval");
                v.item_f32().expect("item")
            };
            let (lp_fix, lp_spec) = (read(fixture_token, &logprobs), read(SPEC_TOKEN, &logprobs));
            let (l_fix, l_spec) = (read(fixture_token, &row), read(SPEC_TOKEN, &row));
            eprintln!(
                "plain path at position {DIVERGENCE}: argmax {sampled}; \
                 fixture token {fixture_token}: logit {l_fix} logprob {lp_fix}; \
                 spec token {SPEC_TOKEN}: logit {l_spec} logprob {lp_spec}; \
                 raw-logit delta {}",
                (l_fix - l_spec).abs()
            );
            assert_eq!(
                sampled, fixture_token,
                "plain replication must match fixture"
            );

            // Verify-shaped forward from the SAME plain-built KV: feed the
            // same token plus 4 junk rows; does row 0 flip?
            let mut vtable = table; // continue on the same table
            let junk = [11u32, 23, 5, 42];
            let mut ids = vec![feed];
            ids.extend_from_slice(&junk);
            // Rewind: the plain step above already appended this position.
            // Truncate its slot plus nothing else, then append 5.
            vtable
                .truncate(&mut mgr, vtable.num_tokens() - 1)
                .expect("rewind");
            let logits5 = forward(&mut vtable, &mut kv, &mut mgr, ids, 5);
            let vocab5 = logits5.dim(2);
            let row0 = ops::slice(&logits5, &[0, 0, 0], &[1, 1, vocab5], &stream).expect("slice");
            let row0 = ops::reshape(&row0, &[1, vocab5], &stream).expect("reshape");
            let argmax0 = ops::argmax(&row0, -1, false, &stream)
                .expect("argmax")
                .item_u32()
                .expect("item");
            let (l5_fix, l5_spec) = (read(fixture_token, &row0), read(SPEC_TOKEN, &row0));
            eprintln!(
                "verify-shaped (5-row) forward from the same KV: row-0 argmax {argmax0}; \
                 fixture token logit {l5_fix}; spec token logit {l5_spec}; \
                 row-0 bits {} the plain step",
                if l5_fix == l_fix && l5_spec == l_spec {
                    "MATCH (for these two lanes)"
                } else {
                    "DIFFER from"
                }
            );
            // Narrower verify shapes from the same state: where is the
            // row-0 bit boundary?
            for rows in [2usize, 3, 4] {
                vtable
                    .truncate(&mut mgr, vtable.num_tokens() - 5)
                    .expect("rewind");
                let mut ids = vec![feed];
                ids.extend_from_slice(&junk[..rows - 1]);
                let logits_n = forward(&mut vtable, &mut kv, &mut mgr, ids, rows as i32);
                let vocab_n = logits_n.dim(2);
                let row0 =
                    ops::slice(&logits_n, &[0, 0, 0], &[1, 1, vocab_n], &stream).expect("slice");
                let row0 = ops::reshape(&row0, &[1, vocab_n], &stream).expect("reshape");
                let argmax_n = ops::argmax(&row0, -1, false, &stream)
                    .expect("argmax")
                    .item_u32()
                    .expect("item");
                let (ln_fix, ln_spec) = (read(fixture_token, &row0), read(SPEC_TOKEN, &row0));
                eprintln!(
                    "{rows}-row verify shape: row-0 argmax {argmax_n}; fixture logit {ln_fix}; \
                     spec logit {ln_spec}; bits {} plain",
                    if ln_fix == l_fix && ln_spec == l_spec {
                        "MATCH"
                    } else {
                        "DIFFER from"
                    }
                );
                // Re-append junk so the loop's rewind arithmetic stays
                // uniform (each iteration rewinds 5 slots).
                let refill = 5 - rows;
                if refill > 0 {
                    vtable
                        .append_tokens(&mut mgr, refill)
                        .expect("refill slots");
                }
            }
            return;
        }
        assert_eq!(
            sampled, expected_token,
            "plain replication diverged from the fixture at {i} — probe harness bug"
        );
        feed = sampled;
    }
}

#[test]
fn characterize_qwen25_divergence() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal");
        return;
    }
    let Some(root) = std::env::var_os("KILN_TEST_MODELS").map(PathBuf::from) else {
        eprintln!("skipping: KILN_TEST_MODELS unset");
        return;
    };
    kiln_mlx::init();

    // ---- Focus model: qwen2.5-0.5b, all fixtures, shape matrix.
    {
        let name = "qwen2.5-0.5b-4bit";
        let dir = root.join(name);
        let stream = Stream::gpu();
        let model = AnyModel::load(&dir, &stream).expect("model");
        let w = model.calibrate_deterministic_width(&stream).expect("cal");
        eprintln!("== {name}, width {w}");
        for (fixture_name, fixture, prompt) in load_fixtures(name, &dir) {
            let expected = &fixture.expected_token_ids;
            // adversarial gamma=4: row-0-only commits, 5-row KV writes
            let adv = run(
                &model,
                w,
                4,
                Some(Box::new(AdversarialDrafter {
                    seqs: HashSet::new(),
                    proposals: 0,
                })),
                &prompt,
                fixture.max_tokens,
            );
            // oracle gamma=4: full-acceptance path
            let oracle = run(
                &model,
                w,
                4,
                Some(Box::new(OracleDrafter {
                    expected: expected.clone(),
                    committed: 0,
                })),
                &prompt,
                fixture.max_tokens,
            );
            // oracle gamma=1: 2-row verify
            let oracle_g1 = run(
                &model,
                w,
                1,
                Some(Box::new(OracleDrafter {
                    expected: expected.clone(),
                    committed: 0,
                })),
                &prompt,
                fixture.max_tokens,
            );
            // oracle gamma=2: 3-row verify
            let oracle_g2 = run(
                &model,
                w,
                2,
                Some(Box::new(OracleDrafter {
                    expected: expected.clone(),
                    committed: 0,
                })),
                &prompt,
                fixture.max_tokens,
            );
            eprintln!(
                "{name}/{fixture_name}: adversarial g4 div={:?}; oracle g4 div={:?}; \
                 oracle g2 div={:?}; oracle g1 div={:?}",
                first_divergence(&adv, expected),
                first_divergence(&oracle, expected),
                first_divergence(&oracle_g2, expected),
                first_divergence(&oracle_g1, expected),
            );
        }
    }

    // ---- Complete the matrix for the models the main run never reached.
    for name in ["qwen3-0.6b-4bit", "qwen3-0.6b-8bit", "smollm2-135m-bf16"] {
        let dir = root.join(name);
        if !dir.join("config.json").is_file() {
            eprintln!("{name}: missing, skipped");
            continue;
        }
        let stream = Stream::gpu();
        let model = AnyModel::load(&dir, &stream).expect("model");
        let w = model.calibrate_deterministic_width(&stream).expect("cal");
        eprintln!("== {name}, width {w}");
        for (fixture_name, fixture, prompt) in load_fixtures(name, &dir) {
            let expected = &fixture.expected_token_ids;
            let draft = DraftModel::load(
                &dir,
                DraftPoolSpec {
                    block_size: 32,
                    num_blocks: 256,
                },
                &stream,
            )
            .expect("self draft");
            let selfd = run(
                &model,
                w,
                4,
                Some(Box::new(draft)),
                &prompt,
                fixture.max_tokens,
            );
            let adv = run(
                &model,
                w,
                4,
                Some(Box::new(AdversarialDrafter {
                    seqs: HashSet::new(),
                    proposals: 0,
                })),
                &prompt,
                fixture.max_tokens,
            );
            eprintln!(
                "{name}/{fixture_name}: self-draft g4 div={:?}; adversarial g4 div={:?}",
                first_divergence(&selfd, expected),
                first_divergence(&adv, expected),
            );
        }
    }
}
