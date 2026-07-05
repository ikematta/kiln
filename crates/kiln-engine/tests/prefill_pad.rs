//! The ADR 0002 pad-row contract (Phase 6 / B1): a ragged prefill tail
//! piece shorter than `PREFILL_PAD_MIN_ROWS` runs with kernel-class pad
//! rows — phantom content moving through the same pipes as real tokens,
//! the exact hazard class of the Phase 5 pipeline-discard bug. This test
//! pins the engine side of the contract:
//!
//! - `pad_rows` is set precisely per the rule (sub-32 pieces off the
//!   super-chunk grid; nothing else), and only on single-sequence prefill
//!   batches;
//! - the step's input ids and every K/V write run cover REAL rows only —
//!   pad rows are never given a slot in any block;
//! - pad rows are never sampled or emitted: token streams stay closed-form;
//! - blocks written by padded pieces serve prefix-cache hits bit-exactly
//!   (warm == cold), including a containment rerun that reuses the padded
//!   piece's own rows and an extension that re-pads on resume.
//!
//! Uses the pipeline_discard.rs checksum mock: sampled tokens equal
//! `last + 1` plus the deviation of the gathered position-channel from its
//! closed form, so any stale, misplaced, or phantom row derails the stream.
//! Runs on CPU when Metal is unavailable. Single `#[test]` because the
//! kiln-mlx live-object counter is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    DEFAULT_PREFILL_FINE_CHUNK, Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary,
    KvDims, PREFILL_PAD_MIN_ROWS, PagedKv, PenaltyOptions, Priority, SamplingOptions, SeqEvent,
    StepBatch, StepInput, StepModel,
};
use kiln_mlx::{Array, Dtype, MlxError, Stream, debug, ops};

const VOCAB: i32 = 1024;

/// What the model observed about one step.
#[derive(Debug, Clone)]
struct Obs {
    pad_rows: i32,
    /// `(len, offset, sample)` per sequence.
    seqs: Vec<(i32, i32, bool)>,
    /// `input.num_tokens()` — must never include pad rows.
    input_tokens: usize,
    /// Total rows across all write runs — must never include pad rows.
    write_rows: i32,
}

/// K channel 0: `token * 8 + position`; K channel 1: `4 * position + 1`
/// (closed form, checksummed at sampling); V = K + 1000.
struct RecordingMock {
    observations: Rc<RefCell<Vec<Obs>>>,
}

impl StepModel for RecordingMock {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        self.observations.borrow_mut().push(Obs {
            pad_rows: batch.pad_rows,
            seqs: batch
                .seqs
                .iter()
                .map(|seq| (seq.len, seq.offset, seq.sample))
                .collect(),
            input_tokens: batch.num_tokens(),
            write_rows: batch
                .seqs
                .iter()
                .flat_map(|seq| &seq.writes)
                .map(|run| run.len)
                .sum(),
        });

        let n = batch.num_tokens() as i32;
        let ids = match &batch.input {
            StepInput::Ids(ids) => Array::from_u32_slice(ids, &[1, n])?,
            StepInput::Lazy(tokens) => tokens.clone(),
        };
        let ids = ops::astype(&ids, Dtype::Float32, s)?;

        // Writes first; real rows only — the pad rows exist solely as
        // trunk filler in a real model and have no write slots here.
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

fn engine(observations: Rc<RefCell<Vec<Obs>>>) -> Engine<RecordingMock> {
    // Production prefill grid (2048 / 128) so the padding rule fires
    // exactly as shipped; small pool.
    let config = EngineConfig {
        num_blocks: 32,
        ..EngineConfig::default()
    };
    let dims = KvDims {
        layers: 1,
        kv_heads: 1,
        head_dim: 2,
    };
    Engine::new(RecordingMock { observations }, dims, config, stream()).expect("engine builds")
}

struct Outcome {
    tokens: Rc<RefCell<Vec<u32>>>,
    hits: Rc<RefCell<Vec<(u32, bool)>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
}

fn run(engine: &mut Engine<RecordingMock>, prompt: &[u32], max_tokens: usize) -> Outcome {
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
    });
    for _ in 0..10_000 {
        if engine.is_idle() {
            let summary = outcome.finish.borrow().clone().expect("request finished");
            assert_eq!(summary.reason, FinishKind::Length, "full run: {summary:?}");
            return outcome;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain");
}

/// `(pad_rows, per-seq (len, offset, sample))` of one padded step.
type PaddedStep = (i32, Vec<(i32, i32, bool)>);

/// Every observed step upholds the row-accounting contract; returns the
/// pad amounts of the padded steps in order.
fn audit(observations: &[Obs]) -> Vec<PaddedStep> {
    let mut padded = Vec::new();
    for obs in observations {
        let real: i32 = obs.seqs.iter().map(|&(len, _, _)| len).sum();
        assert_eq!(
            obs.input_tokens as i32, real,
            "step input must carry REAL tokens only: {obs:?}"
        );
        assert_eq!(
            obs.write_rows, real,
            "write runs must cover REAL rows only: {obs:?}"
        );
        if obs.pad_rows > 0 {
            assert_eq!(
                obs.seqs.len(),
                1,
                "pads only on single-sequence prefill pieces: {obs:?}"
            );
            assert!(
                !obs.seqs[0].2,
                "padded pieces never sample (the sampled step is separate): {obs:?}"
            );
            padded.push((obs.pad_rows, obs.seqs.clone()));
        }
    }
    padded
}

#[test]
fn pad_rows_never_reach_kv_sampling_or_the_prefix_cache() {
    let fine = DEFAULT_PREFILL_FINE_CHUNK;
    let baseline = debug::live_objects();
    {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let mut warm = engine(Rc::clone(&observations));

        // --- 1) Cold run. Prompt of fine + 22 tokens: prefill limit is
        // fine + 21, so the canonical pieces are [0, fine) (on the
        // super-chunk grid at offset 0 -> never padded) and
        // [fine, fine+21) — a 21-row ragged piece, padded by 11.
        let prompt_a: Vec<u32> = (40..40 + fine as u32 + 22).collect();
        let last_a = 40 + fine as u32 + 21;
        let a = run(&mut warm, &prompt_a, 8);
        assert_eq!(
            a.tokens.borrow().as_slice(),
            (last_a + 1..last_a + 9).collect::<Vec<u32>>().as_slice(),
            "cold stream must be closed-form (any phantom row shifts the checksum)"
        );
        let padded = audit(&observations.borrow());
        assert_eq!(
            padded,
            vec![(
                (PREFILL_PAD_MIN_ROWS - 21) as i32,
                vec![(21, fine as i32, false)]
            )],
            "exactly the ragged sub-32 piece is padded, by exactly 32 - len"
        );
        observations.borrow_mut().clear();

        // --- 2) Containment rerun: the donor's cached blocks include the
        // rows the padded piece wrote. The rerun is served from them and
        // must land on the identical stream — a pad row leaked into any
        // block would shift the gathered checksum.
        let a2 = run(&mut warm, &prompt_a, 8);
        assert_eq!(
            a2.tokens.borrow().as_slice(),
            a.tokens.borrow().as_slice(),
            "containment rerun served different data than the padded cold run wrote"
        );
        let hits = a2.hits.borrow().clone();
        assert_eq!(hits.len(), 1, "the identical prompt must hit the cache");
        assert!(
            hits[0].0 as usize >= fine,
            "containment hit should cover at least the first piece: {hits:?}"
        );
        // Whatever the rerun recomputed past the hit follows the same
        // canonical shapes, re-padded identically.
        let padded = audit(&observations.borrow());
        for (pad, seqs) in &padded {
            assert_eq!((*pad + seqs[0].0) as usize, PREFILL_PAD_MIN_ROWS);
        }
        observations.borrow_mut().clear();

        // --- 3) Extension resume: A's full stream plus a divergent tail,
        // sized so the recomputed resume piece is ragged again (31 rows,
        // pad 1). Warm must equal a cache-cold run of the same prompt:
        // donated blocks carry no pad garbage, and the resumed piece is
        // re-padded on the same canonical shape.
        let mut prompt_b = prompt_a.clone();
        prompt_b.extend(a.tokens.borrow().iter().copied());
        prompt_b.extend(600..600 + (2 * fine as u32 - prompt_b.len() as u32));
        prompt_b.truncate(fine + 32); // limit = fine + 31
        assert_eq!(prompt_b.len(), fine + 32);
        let mut cold = engine(Rc::new(RefCell::new(Vec::new())));
        let b_cold = run(&mut cold, &prompt_b, 6);
        let b_warm = run(&mut warm, &prompt_b, 6);
        assert_eq!(
            b_warm.tokens.borrow().as_slice(),
            b_cold.tokens.borrow().as_slice(),
            "warm extension diverged from cold — padded-piece rows or pad garbage \
             were served through the prefix cache"
        );
        let hits = b_warm.hits.borrow().clone();
        assert_eq!(hits.len(), 1, "the shared prefix must hit");
        assert_eq!(
            hits[0].0 as usize % fine,
            0,
            "resume lands on a fine boundary: {hits:?}"
        );
        let padded = audit(&observations.borrow());
        assert_eq!(
            padded.last(),
            Some(&(1, vec![(31, fine as i32, false)])),
            "the resumed 31-row piece must be re-padded by 1 (same canonical shape \
             warm and cold)"
        );
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "padded runs leaked mlx handles"
    );
}
