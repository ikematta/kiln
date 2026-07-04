//! The pipeline-discarded-row invariant (Phase 5 step zero; see the
//! module docs of `kiln_engine::radix`): a sequence that stops or is
//! cancelled at a *pipelined* apply has one speculative row in flight —
//! scheduled, never read back. Its block must never enter the prefix
//! cache; only settled rows (`processed + fed`) are donatable.
//!
//! This test drives the real engine with a mock model whose logits are
//! tied to a checksum of the KV it gathers: every sampled token equals
//! `last_token + 1` *plus the deviation of the gathered position-channel
//! from its closed form*. Any stale, unwritten, or misplaced row served
//! through the prefix cache shifts the checksum and derails the token
//! stream — so "no stale data" is asserted by bit-equality with a
//! cache-cold run, and the invariant itself by the exact match length.
//!
//! Runs on CPU when Metal is unavailable (the mock has no weights).
//! Single `#[test]` because the kiln-mlx live-object counter is
//! process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, KvDims, PagedKv,
    PenaltyOptions, Priority, SamplingOptions, SeqEvent, StepBatch, StepInput, StepModel,
};
use kiln_mlx::{Array, Dtype, MlxError, Stream, debug, ops};

const BLOCK: usize = 4;
const VOCAB: i32 = 512;

/// K channel 0: `token * 8 + position` (token-dependent).
/// K channel 1: `4 * position + 1` (closed form, checksummed at sampling).
/// V: K + 1000.
struct MockModel;

impl StepModel for MockModel {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        let n = batch.num_tokens() as i32;
        let ids = match &batch.input {
            StepInput::Ids(ids) => Array::from_u32_slice(ids, &[1, n])?,
            StepInput::Lazy(tokens) => tokens.clone(),
        };
        let ids = ops::astype(&ids, Dtype::Float32, s)?;

        // Writes first (mirrors the real model: all pool updates precede
        // any gather of the final pool version).
        let mut consumed = 0;
        for seq in &batch.seqs {
            let seg = ops::slice(&ids, &[0, consumed], &[1, consumed + seq.len], s)?;
            consumed += seq.len;
            let seg = ops::reshape(&seg, &[1, 1, seq.len, 1], s)?;
            let pos: Vec<f32> = (seq.offset..seq.offset + seq.len)
                .map(|p| p as f32)
                .collect();
            let pos = Array::from_f32_slice(&pos, &[1, 1, seq.len, 1])?;
            let ch0 = ops::add(&ops::multiply(&seg, &Array::from_f32(8.0), s)?, &pos, s)?;
            let ch1 = ops::add(
                &ops::multiply(&pos, &Array::from_f32(4.0), s)?,
                &Array::from_f32(1.0),
                s,
            )?;
            let k = ops::concatenate(&[&ch0, &ch1], 3, s)?;
            let v = ops::add(&k, &Array::from_f32(1000.0), s)?;
            kv.write(0, &seq.writes, &k, &v, s)?;
        }

        // Sampled rows: target = last_token + 1 + (checksum deviation).
        let mut rows: Vec<Array> = Vec::new();
        let mut consumed = 0;
        for seq in &batch.seqs {
            let last = ops::slice(
                &ids,
                &[0, consumed + seq.len - 1],
                &[1, consumed + seq.len],
                s,
            )?;
            consumed += seq.len;
            if !seq.sample {
                continue;
            }
            let total = seq.offset + seq.len;
            let (gk, _gv) = kv.gather(0, &seq.blocks, total, s)?;
            let ch1 = ops::slice(&gk, &[0, 0, 0, 1], &[1, 1, total, 2], s)?;
            let ch1 = ops::reshape(&ch1, &[total], s)?;
            let sums = ops::cumsum(&ch1, 0, false, true, s)?;
            let sum = ops::slice(&sums, &[total - 1], &[total], s)?;
            let expected: f32 = (0..total).map(|p| (4 * p + 1) as f32).sum();
            let deviation = ops::subtract(&sum, &Array::from_f32(expected), s)?;
            let target = ops::add(
                &ops::add(&ops::reshape(&last, &[1], s)?, &Array::from_f32(1.0), s)?,
                &deviation,
                s,
            )?;
            let grid = ops::arange(0.0, f64::from(VOCAB), 1.0, Dtype::Float32, s)?;
            let miss = ops::subtract(&grid, &target, s)?;
            let row = ops::negative(&ops::multiply(&miss, &miss, s)?, s)?;
            rows.push(ops::reshape(&row, &[1, 1, VOCAB], s)?);
        }
        if rows.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        Ok(Some(ops::concatenate(&refs, 1, s)?))
    }
}

fn stream() -> Stream {
    if kiln_mlx::memory::metal_is_available() {
        Stream::gpu()
    } else {
        Stream::cpu()
    }
}

fn engine() -> Engine<MockModel> {
    let config = EngineConfig {
        block_size: BLOCK,
        num_blocks: 32,
        // Chunk == block: non-containment matches trim to block-aligned
        // canonical boundaries, keeping sub-2048-token scenarios useful.
        prefill_chunk: BLOCK,
        ..EngineConfig::default()
    };
    let dims = KvDims {
        layers: 1,
        kv_heads: 1,
        head_dim: 2,
    };
    Engine::new(MockModel, dims, config, stream()).expect("engine builds")
}

struct Outcome {
    tokens: Rc<RefCell<Vec<u32>>>,
    hits: Rc<RefCell<Vec<(u32, bool)>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
    cancel: Arc<AtomicBool>,
}

fn request(prompt: &[u32], max_tokens: usize, stop: &[u32]) -> (EngineRequest, Outcome) {
    let outcome = Outcome {
        tokens: Rc::new(RefCell::new(Vec::new())),
        hits: Rc::new(RefCell::new(Vec::new())),
        finish: Rc::new(RefCell::new(None)),
        cancel: Arc::new(AtomicBool::new(false)),
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
        cancel: Arc::clone(&outcome.cancel),
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

fn drain(engine: &mut Engine<MockModel>) {
    for _ in 0..10_000 {
        if engine.is_idle() {
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain");
}

/// The mock's generation rule is `next = last + 1`, so the whole stream
/// is closed-form: prompt ++ (last+1, last+2, ...).
fn run(engine: &mut Engine<MockModel>, prompt: &[u32], max_tokens: usize, stop: &[u32]) -> Outcome {
    let (request, outcome) = request(prompt, max_tokens, stop);
    engine.submit(request);
    drain(engine);
    outcome
}

#[test]
fn pipelined_stop_never_donates_the_inflight_block() {
    let baseline = debug::live_objects();
    {
        // --- 1) A stops via a sampled stop token at a pipelined apply,
        // with prompt+generated landing exactly on a block boundary: the
        // discarded speculative row is the would-be last row of a full
        // block — the one case where the invariant changes what is
        // donatable.
        let prompt_a: Vec<u32> = (40..46).collect(); // p = 6, last = 45
        let stop_a = 45 + 6; // G = 6 => p + G = 12 = 3 full blocks
        let mut warm = engine();
        let a = run(&mut warm, &prompt_a, 20, &[stop_a]);
        let summary = a.finish.borrow().clone().expect("A finished");
        assert_eq!(summary.reason, FinishKind::Stop, "A must stop via token");
        assert_eq!(summary.completion_tokens, 6);
        assert_eq!(a.tokens.borrow().as_slice(), &[46, 47, 48, 49, 50]);
        assert!(
            warm.pipelined_steps() > 0,
            "the stop must land on the pipelined path (else this test \
             exercises nothing)"
        );

        // --- 2) B extends A's full stream (including the stop token).
        // Settled rows = p + G - 1 = 11 -> exactly 2 donatable blocks.
        // The block holding rows 8..11 carried the discarded in-flight
        // row and must NOT have been donated.
        let mut prompt_b: Vec<u32> = (40..52).collect(); // A's 12 tokens
        prompt_b.extend(52..56); // diverging continuation, 16 total
        let mut cold = engine();
        let b_cold = run(&mut cold, &prompt_b, 4, &[]);
        assert_eq!(b_cold.tokens.borrow().as_slice(), &[56, 57, 58, 59]);
        assert!(b_cold.hits.borrow().is_empty(), "cold engine cannot hit");

        let b_warm = run(&mut warm, &prompt_b, 4, &[]);
        let hits = b_warm.hits.borrow().clone();
        assert_eq!(
            hits,
            vec![(8, false)],
            "match must cover exactly the settled full blocks (2 x {BLOCK} \
             tokens) — the in-flight block's region is re-prefilled, not served"
        );
        // The checksum-tied logits make any stale/garbage row shift the
        // stream; warm must be bit-identical to cold.
        assert_eq!(
            b_warm.tokens.borrow().as_slice(),
            b_cold.tokens.borrow().as_slice(),
            "prefix-cache read after a pipelined stop served stale data"
        );
        let summary = b_warm.finish.borrow().clone().expect("B finished");
        assert_eq!(summary.cached_prompt_tokens, 8);
        eprintln!("pipelined stop: match trimmed to settled blocks; warm == cold");

        // --- 3) Same region again: a full re-run of A's stream now hits
        // what B re-donated (B's own prefill settled those rows), still
        // bit-exact.
        let b_again = run(&mut warm, &prompt_b, 4, &[]);
        assert_eq!(
            b_again.tokens.borrow().as_slice(),
            b_cold.tokens.borrow().as_slice()
        );
        let again_hits = b_again.hits.borrow().clone();
        assert_eq!(again_hits.len(), 1);
        assert!(
            again_hits[0].0 >= 8,
            "B's own finish should extend the cached prefix"
        );

        // --- 4) Cancel variant: flag a request mid-pipelined-decode; the
        // cancel is honored at an apply with a row in flight. Whatever was
        // donated must be block-aligned, within the settled bound, and
        // serve bit-exact data.
        let prompt_c: Vec<u32> = (140..146).collect();
        let (request_c, c) = request(&prompt_c, 30, &[]);
        let mut warm2 = engine();
        warm2.submit(request_c);
        while c.tokens.borrow().len() < 3 {
            warm2.step().expect("engine step");
        }
        assert!(
            warm2.pipelined_steps() > 0,
            "cancel must interrupt a pipeline"
        );
        c.cancel.store(true, Ordering::Release);
        drain(&mut warm2);
        let summary = c.finish.borrow().clone().expect("C finished");
        assert_eq!(summary.reason, FinishKind::Cancelled);
        let generated = summary.completion_tokens as usize;
        let settled = prompt_c.len() - 1 + generated;

        let mut prompt_d: Vec<u32> = (140..(146 + generated as u32)).collect();
        prompt_d.extend(200..204);
        let mut cold2 = engine();
        let d_cold = run(&mut cold2, &prompt_d, 4, &[]);
        let d_warm = run(&mut warm2, &prompt_d, 4, &[]);
        assert_eq!(
            d_warm.tokens.borrow().as_slice(),
            d_cold.tokens.borrow().as_slice(),
            "prefix-cache read after a pipelined cancel served stale data"
        );
        let hits = d_warm.hits.borrow().clone();
        assert_eq!(hits.len(), 1, "shared prefix must hit");
        let (matched, _) = hits[0];
        assert_eq!(matched as usize % BLOCK, 0, "matches are block-aligned");
        assert!(
            matched as usize <= settled / BLOCK * BLOCK,
            "match ({matched}) exceeded the settled bound ({} tokens settled)",
            settled
        );
        eprintln!(
            "pipelined cancel: {generated} generated, match {matched} <= aligned settled {}",
            settled / BLOCK * BLOCK
        );
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "leaked mlx objects (pool arrays or in-flight rows)"
    );
}
