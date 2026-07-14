//! The ADR 0002 B' partition contract (design answer Q1): deterministic
//! rows (greedy or client-seeded) decode in sub-batches of at most
//! `deterministic_decode_width` rows; non-deterministic rows ride ONE
//! unrestricted full-width forward in the same step; and finished
//! non-deterministic sequences donate only their prefill-covered blocks
//! (their decode rows were computed at arbitrary trunk widths and must
//! never feed a deterministic request's warm run).
//!
//! Uses the pipeline_discard.rs checksum mock (any stale/misplaced row
//! derails the greedy stream) extended to record every forward's shape.
//! Runs on CPU when Metal is unavailable. Single `#[test]` because the
//! kiln-mlx live-object counter is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, KvDims, PagedKv,
    PenaltyOptions, Priority, SamplingOptions, SeqEvent, StepBatch, StepInput, StepModel,
};
use kiln_mlx::{Array, Dtype, MlxError, Stream, debug, ops};

const BLOCK: usize = 4;
const VOCAB: i32 = 1024;
const WIDTH: usize = 2;

/// One observed forward: row count, whether every row sampled (decode /
/// pipelined forwards; prefill pieces never sample), lazy input or ids.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Obs {
    rows: i32,
    all_sampled: bool,
    lazy: bool,
}

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
        let n = batch.num_tokens() as i32;
        self.observations.borrow_mut().push(Obs {
            rows: n,
            all_sampled: batch.seqs.iter().all(|seq| seq.sample_rows > 0),
            lazy: matches!(batch.input, StepInput::Lazy(_)),
        });
        let ids = match &batch.input {
            StepInput::Ids(ids) => Array::from_u32_slice(ids, &[1, n])?,
            StepInput::Lazy(tokens) => tokens.clone(),
        };
        let ids = ops::astype(&ids, Dtype::Float32, s)?;

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
            if seq.sample_rows == 0 {
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
            // Sharpen so even temperature-0.7 draws stay on the peak: the
            // nearest-neighbour logit sits ~1400 below after scaling.
            let row = ops::multiply(&row, &Array::from_f32(1400.0), s)?;
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
    let config = EngineConfig {
        block_size: BLOCK,
        num_blocks: 64,
        prefill_chunk: BLOCK,
        deterministic_decode_width: WIDTH,
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
    hits: Rc<RefCell<Vec<u32>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
}

fn request(
    prompt: &[u32],
    max_tokens: usize,
    sampling: SamplingOptions,
) -> (EngineRequest, Outcome) {
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
        sampling,
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
                SeqEvent::PrefixHit { tokens, .. } => h.borrow_mut().push(tokens),
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    };
    (request, outcome)
}

/// Non-deterministic but repeatable: temperature > 0 with a fixed
/// worker-style seed and `explicit_seed: false` (the classification is the
/// flag, not the seed value — mirroring the proto contract).
fn sampled_options(seed: u64) -> SamplingOptions {
    SamplingOptions {
        temperature: 0.7,
        seed,
        explicit_seed: false,
        ..SamplingOptions::default()
    }
}

fn drain(engine: &mut Engine<RecordingMock>) {
    for _ in 0..10_000 {
        if engine.is_idle() {
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain");
}

/// Decode/pipelined forwards only (every row samples).
fn decode_shapes(observations: &[Obs]) -> Vec<Obs> {
    observations
        .iter()
        .copied()
        .filter(|o| o.all_sampled)
        .collect()
}

/// One `#[test]` entry point (the kiln-mlx live-object counter is
/// process-global and two concurrent engines on separate test threads race
/// the Metal stream — same convention as pipeline_discard.rs).
#[test]
fn deterministic_partition_contract() {
    partition_shapes();
    donation_cap();
}

fn partition_shapes() {
    let baseline = debug::live_objects();
    {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let mut warm = engine(Rc::clone(&observations));

        // --- A) All-deterministic, width 5, W = 2: every decode forward
        // carries at most 2 rows, streams stay closed-form, and the
        // async pipeline still engages (partitioned Lazy forwards).
        let prompts: Vec<Vec<u32>> = (0..5u32)
            .map(|r| (40 + 50 * r..40 + 50 * r + 6).collect())
            .collect();
        let outs: Vec<Outcome> = prompts
            .iter()
            .map(|p| {
                let (req, out) = request(p, 8, SamplingOptions::default());
                warm.submit(req);
                out
            })
            .collect();
        drain(&mut warm);
        for (prompt, out) in prompts.iter().zip(&outs) {
            let last = *prompt.last().expect("nonempty");
            assert_eq!(
                out.tokens.borrow().as_slice(),
                (last + 1..last + 9).collect::<Vec<u32>>().as_slice(),
                "deterministic stream must be closed-form through partitioning"
            );
        }
        let decodes = decode_shapes(&observations.borrow());
        assert!(
            decodes.iter().all(|o| o.rows <= WIDTH as i32),
            "a deterministic decode forward exceeded W={WIDTH}: {decodes:?}"
        );
        assert!(
            decodes.iter().any(|o| o.lazy),
            "the async pipeline never engaged under partitioning"
        );
        assert!(warm.pipelined_steps() > 0, "pipeline counter never moved");
        observations.borrow_mut().clear();

        // --- B) All non-deterministic, width 5: rides full-width forwards
        // (5 rows once everyone is admitted) — no sub-batching tax.
        let nd_outs: Vec<Outcome> = prompts
            .iter()
            .enumerate()
            .map(|(r, p)| {
                let (req, out) = request(p, 8, sampled_options(100 + r as u64));
                warm.submit(req);
                out
            })
            .collect();
        drain(&mut warm);
        for out in &nd_outs {
            let summary = out.finish.borrow().clone().expect("finished");
            assert_eq!(summary.reason, FinishKind::Length);
        }
        let decodes = decode_shapes(&observations.borrow());
        let max_rows = decodes.iter().map(|o| o.rows).max().unwrap_or(0);
        assert_eq!(
            max_rows, 5,
            "non-deterministic rows must ride one full-width forward: {decodes:?}"
        );
        observations.borrow_mut().clear();

        // --- C) Mixed 3 det + 3 non-det: steady-state steps decompose as
        // det chunks [2,1] plus one full [3] non-det forward.
        let det_prompts: Vec<Vec<u32>> = (0..3u32)
            .map(|r| (400 + 40 * r..400 + 40 * r + 5).collect())
            .collect();
        let nd_prompts: Vec<Vec<u32>> = (0..3u32)
            .map(|r| (600 + 40 * r..600 + 40 * r + 5).collect())
            .collect();
        let det_outs: Vec<Outcome> = det_prompts
            .iter()
            .map(|p| {
                let (req, out) = request(p, 8, SamplingOptions::default());
                warm.submit(req);
                out
            })
            .collect();
        let _nd_outs: Vec<Outcome> = nd_prompts
            .iter()
            .enumerate()
            .map(|(r, p)| {
                let (req, out) = request(p, 8, sampled_options(200 + r as u64));
                warm.submit(req);
                out
            })
            .collect();
        drain(&mut warm);
        for (prompt, out) in det_prompts.iter().zip(&det_outs) {
            let last = *prompt.last().expect("nonempty");
            assert_eq!(
                out.tokens.borrow().as_slice(),
                (last + 1..last + 9).collect::<Vec<u32>>().as_slice(),
                "mixed-step deterministic stream must stay closed-form (bit-exact)"
            );
        }
        let decodes = decode_shapes(&observations.borrow());
        assert!(
            decodes
                .iter()
                .any(|o| o.rows == 3 && decodes.iter().any(|p| p.rows == 2)),
            "expected steady-state mixed steps with a full [3] non-det forward \
             alongside [2,1] det chunks: {decodes:?}"
        );
        assert!(
            !decodes.iter().any(|o| o.rows > 3),
            "no forward may carry more rows than the non-det share: {decodes:?}"
        );
    }
    assert_eq!(debug::live_objects(), baseline, "leaked mlx handles");
}

fn donation_cap() {
    let baseline = debug::live_objects();
    {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let mut warm = engine(Rc::clone(&observations));

        // Non-det donor A: 9-token prompt (prefill covers 8 rows = 2 full
        // blocks), 6 generated. Seeded-but-unflagged, so its stream is
        // repeatable while classified non-deterministic.
        let prompt_a: Vec<u32> = (40..49).collect();
        let (req_a, a) = request(&prompt_a, 6, sampled_options(7));
        warm.submit(req_a);
        drain(&mut warm);
        let a_tokens = a.tokens.borrow().clone();
        assert_eq!(a_tokens.len(), 6);

        // Deterministic B extends A's full stream. Its warm hit must stop
        // at A's prefill-covered bound (8 rows -> 2 blocks): A's decode
        // rows were computed at non-deterministic widths and must not
        // serve a deterministic request.
        let mut prompt_b = prompt_a.clone();
        prompt_b.extend(a_tokens.iter().copied());
        prompt_b.extend(300..304);
        let (req_b, b_warm) = request(&prompt_b, 4, SamplingOptions::default());
        warm.submit(req_b);
        drain(&mut warm);

        let mut cold = engine(Rc::new(RefCell::new(Vec::new())));
        let (req_bc, b_cold) = request(&prompt_b, 4, SamplingOptions::default());
        cold.submit(req_bc);
        drain(&mut cold);

        assert_eq!(
            b_warm.tokens.borrow().as_slice(),
            b_cold.tokens.borrow().as_slice(),
            "deterministic warm run diverged from cold — non-det decode rows leaked \
             through the prefix cache"
        );
        let prefill_covered = (prompt_a.len() - 1) / BLOCK * BLOCK;
        let hits = b_warm.hits.borrow().clone();
        assert_eq!(hits.len(), 1, "shared prefix must hit");
        assert!(
            hits[0] as usize <= prefill_covered,
            "non-det donor served past its prefill-covered bound \
             ({} > {prefill_covered})",
            hits[0]
        );

        // Contrast: a DETERMINISTIC donor's generated rows stay donatable.
        let prompt_c: Vec<u32> = (700..709).collect();
        let (req_c, c) = request(&prompt_c, 6, SamplingOptions::default());
        warm.submit(req_c);
        drain(&mut warm);
        let mut prompt_d = prompt_c.clone();
        prompt_d.extend(c.tokens.borrow().iter().copied());
        prompt_d.extend(900..904);
        let (req_d, d) = request(&prompt_d, 4, SamplingOptions::default());
        warm.submit(req_d);
        drain(&mut warm);
        let hits = d.hits.borrow().clone();
        assert_eq!(hits.len(), 1, "det extension must hit");
        assert!(
            hits[0] as usize > prefill_covered,
            "a deterministic donor's generated rows should extend the cached \
             prefix past the prefill bound: {hits:?}"
        );
    }
    assert_eq!(debug::live_objects(), baseline, "leaked mlx handles");
}
