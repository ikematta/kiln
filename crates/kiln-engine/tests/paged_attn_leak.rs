//! Targeted live-object repro for the PR #35 soak signature (PROGRESS
//! 2026-07-23): kernel-ON CI soaks end with llama-int mlx live objects at
//! floor+2, late-appearing and persistent, while local soaks stay bit-flat.
//! The ledger's candidate window: a cancel landing during a pipelined
//! decode step leaves one speculative row in flight — scheduled via
//! `async_eval`, never read — and that row (or the kernel's two
//! `PagedAttnInputs` arrays reachable from its graph) might outlive the
//! quiesce if reaping required a *subsequent* pipelined turn.
//!
//! This test drives the real engine with the paged-attention kernel ON and
//! a mock model that routes every decode-shaped sampled row through the
//! real `PagedKv::paged_sdpa` custom-kernel path, so the kernel node and
//! its inputs (block table + context length: exactly the +2-shaped pair)
//! are ancestors of every pipelined sampled-token array — the same graph
//! shape the real worker builds. It then enumerates every distinguishable
//! cancel-observation window (the engine reads the flag only at
//! `pipeline_ok`, at `settle_sampled`, and in the synchronous sweep, so
//! any wall-clock cancel timing collapses onto one of these):
//!
//!   R1  flag flips while a pipelined row is parked; observed by
//!       `pipeline_ok` before the next build (no next row built).
//!   R2  flag flips between build and apply of the SAME pipelined turn
//!       (injected from an earlier row's `on_event`, which runs inside
//!       `apply_inflight`): the next row for the cancelled sequence was
//!       already built and `async_eval`-scheduled, and is reaped unread
//!       by the retain in `pipelined_turn`.
//!   R3  the client stream drops instead of a Cancel RPC (`on_event`
//!       returns `false`) — same windows, event-refusal flavor.
//!   R4  a sequence finishes by LENGTH at a pipelined apply (its final
//!       token was in flight) while its neighbor is cancelled in the same
//!       window — the reap path for a non-cancel finish.
//!
//! After every scenario the engine is drained to idle exactly like the
//! worker's serve loop (step while busy or flush-pending, then idle
//! ticks) and the kiln-mlx live-object counter must be back at the
//! post-warmup baseline — the soak's quiesced-checkpoint gate, minus the
//! heartbeat sampling skew.
//!
//! Single `#[test]` because the live-object counter is process-global.
//! Requires Metal (custom-kernel handles need a GPU architecture string);
//! skips cleanly on CPU-only hosts.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, KvDims, PagedKv,
    PenaltyOptions, Priority, SamplingOptions, SeqEvent, SsdParams, StepBatch, StepInput,
    StepModel,
};
use kiln_mlx::{Array, Dtype, MlxError, Stream, debug, ops};

const BLOCK: usize = 4;
const VOCAB: i32 = 512;
const HEAD_DIM: i32 = 32; // divisible by the kernels' BD = 32

/// Generation rule: `next = last + 1`, closed-form and greedy, with the
/// paged-attention kernel output folded in at weight zero — the custom
/// kernel node (and its `PagedAttnInputs` pair) is a real ancestor of
/// every decode-shaped sampled token, exactly as in the production model,
/// without perturbing the token stream. Writes every layer's pools so the
/// SSD tier's block captures (`read_block_bytes`) exercise the full
/// per-layer loop.
struct MockModel {
    layers: usize,
}

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

        // Writes first (mirrors the real model).
        let mut consumed = 0;
        for seq in &batch.seqs {
            let seg = ops::slice(&ids, &[0, consumed], &[1, consumed + seq.len], s)?;
            consumed += seq.len;
            let seg = ops::reshape(&seg, &[1, 1, seq.len, 1], s)?;
            let wide = ops::zeros(&[1, 1, seq.len, HEAD_DIM], Dtype::Float32, s)?;
            let k = ops::add(&seg, &wide, s)?;
            let v = ops::add(&k, &Array::from_f32(1000.0), s)?;
            for layer in 0..self.layers {
                kv.write(layer, &seq.writes, &k, &v, s)?;
            }
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
            // Zero-weight kernel contribution for decode-shaped segments
            // (the engine prepares `paged_attn` exactly there).
            let deviation = match &seq.paged_attn {
                Some(paged) => {
                    debug_assert_eq!(seq.len, 1, "kernel inputs on a non-decode segment");
                    let q = ops::reshape(&last, &[1, 1, 1, 1], s)?;
                    let q = ops::add(&q, &ops::zeros(&[1, 1, 1, HEAD_DIM], Dtype::Float32, s)?, s)?;
                    let o = kv.paged_sdpa(0, &q, paged, &Array::from_f32(1.0), s)?;
                    let flat = ops::reshape(&o, &[HEAD_DIM], s)?;
                    let sums = ops::cumsum(&flat, 0, false, true, s)?;
                    let osum = ops::slice(&sums, &[HEAD_DIM - 1], &[HEAD_DIM], s)?;
                    ops::multiply(&osum, &Array::from_f32(0.0), s)?
                }
                None => Array::from_f32(0.0),
            };
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

fn engine(ssd: Option<SsdParams>, layers: usize) -> Engine<MockModel> {
    let config = EngineConfig {
        block_size: BLOCK,
        num_blocks: 64,
        prefill_chunk: BLOCK,
        paged_attention_kernel: true,
        ssd,
        ..EngineConfig::default()
    };
    let dims = KvDims {
        layers,
        kv_heads: 1,
        head_dim: HEAD_DIM,
    };
    Engine::new(MockModel { layers }, dims, config, Stream::gpu()).expect("engine builds")
}

/// Per-token hook: called with the 1-based count of tokens received;
/// returning `false` refuses the event (the engine treats it as a client
/// drop and finishes the request Cancelled).
type TokenHook = Box<dyn FnMut(usize) -> bool>;

struct Req {
    tokens: Rc<RefCell<Vec<u32>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
    cancel: Arc<AtomicBool>,
}

fn submit(
    engine: &mut Engine<MockModel>,
    prompt: &[u32],
    max_tokens: usize,
    hook: TokenHook,
) -> Req {
    submit_with_cancel(
        engine,
        prompt,
        max_tokens,
        Arc::new(AtomicBool::new(false)),
        hook,
    )
}

/// `submit` with a caller-owned cancel flag, so another sequence's
/// `on_event` hook can flip THIS request's real flag from inside an apply
/// (the mid-step injection the R2/R4 windows need).
fn submit_with_cancel(
    engine: &mut Engine<MockModel>,
    prompt: &[u32],
    max_tokens: usize,
    cancel: Arc<AtomicBool>,
    hook: TokenHook,
) -> Req {
    let req = Req {
        tokens: Rc::new(RefCell::new(Vec::new())),
        finish: Rc::new(RefCell::new(None)),
        cancel,
    };
    let (t, f) = (Rc::clone(&req.tokens), Rc::clone(&req.finish));
    let mut hook = hook;
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy => pipeline-eligible
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: Default::default(),
        grammar: None,
        priority: Priority::Interactive,
        cancel: Arc::clone(&req.cancel),
        on_event: Box::new(move |event| match event {
            SeqEvent::Token(token) => {
                t.borrow_mut().push(token);
                let count = t.borrow().len();
                hook(count)
            }
            SeqEvent::PrefixHit { .. } => true,
            SeqEvent::Finished(summary) => {
                *f.borrow_mut() = Some(summary);
                true
            }
        }),
    });
    req
}

fn keep() -> TokenHook {
    Box::new(|_| true)
}

/// The worker serve-loop quiesce: step while busy or flush-pending, then a
/// few idle ticks (idle steps still run cache maintenance).
fn quiesce(engine: &mut Engine<MockModel>) {
    for _ in 0..10_000 {
        if engine.is_idle() && !engine.has_pending_cache_io() {
            for _ in 0..5 {
                engine.step().expect("idle tick");
            }
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to quiesce");
}

fn step_until(engine: &mut Engine<MockModel>, mut done: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if done() {
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("condition never reached");
}

fn assert_floor(baseline: i64, label: &str) {
    let live = debug::live_objects();
    assert_eq!(
        live, baseline,
        "{label}: mlx live objects {live} != baseline {baseline} after quiesce \
         (the PR #35 soak +2 signature, reproduced)"
    );
}

fn finished(req: &Req) -> FinishKind {
    req.finish
        .borrow()
        .as_ref()
        .expect("request finished")
        .reason
}

#[test]
fn pipelined_cancel_quiesce_returns_live_objects_to_baseline() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: paged-attention kernels need Metal");
        return;
    }

    // ---- Warmup: materialize pools, JIT the kernel, settle lazy init.
    let mut eng = engine(None, 1);
    let warm = submit(&mut eng, &(40..46).collect::<Vec<_>>(), 8, keep());
    quiesce(&mut eng);
    assert_eq!(finished(&warm), FinishKind::Length);
    assert!(
        eng.pipelined_steps() > 0,
        "warmup must engage the pipeline (else nothing here tests it)"
    );
    let baseline = debug::live_objects();

    // ---- R1: cancel flags flip while a pipelined row is parked; the next
    // `pipeline_ok` sees them, no next row is built, the parked rows are
    // read back and settled as Cancelled. Swept across depths so the flag
    // observation lands at different pipeline states.
    for depth in 1..=8usize {
        let before = eng.pipelined_steps();
        let a = submit(&mut eng, &(40..47).collect::<Vec<_>>(), 30, keep());
        let b = submit(&mut eng, &(140..145).collect::<Vec<_>>(), 30, keep());
        step_until(&mut eng, || {
            a.tokens.borrow().len() >= depth && b.tokens.borrow().len() >= depth
        });
        assert!(
            eng.pipelined_steps() > before,
            "R1 depth {depth}: no pipeline"
        );
        a.cancel.store(true, Ordering::Release);
        b.cancel.store(true, Ordering::Release);
        quiesce(&mut eng);
        assert_eq!(finished(&a), FinishKind::Cancelled);
        assert_eq!(finished(&b), FinishKind::Cancelled);
        assert_floor(baseline, &format!("R1 depth {depth}"));
    }

    // ---- R2: A's on_event (inside apply_inflight) sets B's cancel flag —
    // the flag flips BETWEEN build and apply of the same pipelined turn,
    // so B's next row was already built and async_eval-scheduled when B
    // settles Cancelled; the retain in pipelined_turn reaps it unread.
    // A is then cancelled while its own row is parked (R1 flavor), so the
    // scenario ends with no further pipelined turn — the ledger's window.
    for depth in 2..=8usize {
        let before = eng.pipelined_steps();
        let cancel_b = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&cancel_b);
        let a = submit(
            &mut eng,
            &(40..47).collect::<Vec<_>>(),
            30,
            Box::new(move |count| {
                if count == depth {
                    // Flips B's REAL flag from inside apply_inflight: B's
                    // settle in this same apply (A's row settles first —
                    // arrival order) honors it with B's next row already
                    // built and scheduled.
                    cb.store(true, Ordering::Release);
                }
                true
            }),
        );
        let b = submit_with_cancel(
            &mut eng,
            &(140..145).collect::<Vec<_>>(),
            30,
            Arc::clone(&cancel_b),
            keep(),
        );
        step_until(&mut eng, || b.finish.borrow().is_some());
        assert!(
            eng.pipelined_steps() > before,
            "R2 depth {depth}: no pipeline"
        );
        // B is gone mid-window; park-cancel A so no further pipelined turn
        // ever runs (the ledger's orphan-reap window).
        a.cancel.store(true, Ordering::Release);
        quiesce(&mut eng);
        assert_eq!(finished(&a), FinishKind::Cancelled);
        assert_eq!(finished(&b), FinishKind::Cancelled);
        assert_floor(baseline, &format!("R2 depth {depth}"));
    }

    // ---- R3: event-refusal (client stream drop): both sequences refuse
    // their token at the same apply — every row of the in-flight step
    // finishes Cancelled in one window and the freshly built next step is
    // fully reaped (retain leaves it empty; no further pipelined turn).
    for depth in 1..=8usize {
        let before = eng.pipelined_steps();
        let a = submit(
            &mut eng,
            &(40..47).collect::<Vec<_>>(),
            30,
            Box::new(move |count| count != depth),
        );
        let b = submit(
            &mut eng,
            &(140..145).collect::<Vec<_>>(),
            30,
            Box::new(move |count| count != depth),
        );
        quiesce(&mut eng);
        assert!(
            eng.pipelined_steps() > before,
            "R3 depth {depth}: no pipeline"
        );
        assert_eq!(finished(&a), FinishKind::Cancelled);
        assert_eq!(finished(&b), FinishKind::Cancelled);
        assert_floor(baseline, &format!("R3 depth {depth}"));
    }

    // ---- R4: A finishes by LENGTH at a pipelined apply (its final token
    // was the in-flight row; build_pipelined already skipped it) while B
    // is cancelled inside the same window via A's last on_event.
    for len_at in 3..=6usize {
        let before = eng.pipelined_steps();
        let cancel_b = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&cancel_b);
        let a = submit(
            &mut eng,
            &(40..47).collect::<Vec<_>>(),
            len_at,
            Box::new(move |count| {
                if count == len_at {
                    cb.store(true, Ordering::Release);
                }
                true
            }),
        );
        let b = submit_with_cancel(
            &mut eng,
            &(140..145).collect::<Vec<_>>(),
            30,
            Arc::clone(&cancel_b),
            keep(),
        );
        quiesce(&mut eng);
        assert!(
            eng.pipelined_steps() > before,
            "R4 len {len_at}: no pipeline"
        );
        assert_eq!(finished(&a), FinishKind::Length);
        assert_eq!(finished(&b), FinishKind::Cancelled);
        assert_floor(baseline, &format!("R4 len {len_at}"));
    }

    drop(eng);

    // ---- R5: the same windows with the SSD tier live (hypothesis (b):
    // flush-path interaction) — donations from cancelled sequences enter
    // the write-behind queue and the quiesce must still return the counter
    // to this engine's own post-warmup floor.
    let slab_dir =
        std::env::temp_dir().join(format!("kiln-paged-attn-leak-{}", std::process::id()));
    std::fs::create_dir_all(&slab_dir).expect("slab dir");
    let mut eng = engine(
        Some(SsdParams {
            dir: slab_dir.clone(),
            max_bytes: 64 << 20,
            fingerprint: "paged-attn-leak-test".to_owned(),
        }),
        4,
    );
    let warm = submit(&mut eng, &(40..46).collect::<Vec<_>>(), 8, keep());
    quiesce(&mut eng);
    assert_eq!(finished(&warm), FinishKind::Length);
    let ssd_baseline = debug::live_objects();

    for depth in 1..=6usize {
        let a = submit(&mut eng, &(40..47).collect::<Vec<_>>(), 30, keep());
        let b = submit(&mut eng, &(150..158).collect::<Vec<_>>(), 30, keep());
        step_until(&mut eng, || {
            a.tokens.borrow().len() >= depth && b.tokens.borrow().len() >= depth
        });
        a.cancel.store(true, Ordering::Release);
        b.cancel.store(true, Ordering::Release);
        quiesce(&mut eng);
        assert_eq!(finished(&a), FinishKind::Cancelled);
        assert_eq!(finished(&b), FinishKind::Cancelled);
        assert_floor(ssd_baseline, &format!("R5 depth {depth}"));
    }

    // ---- R6: the observable-transient bound. The soak's heartbeat reads
    // the live-object counter from another thread; the one engine state a
    // "quiesced" checkpoint can catch with NO client request open is the
    // idle-with-pending-flushes loop (the serve loop keeps ticking bounded
    // cache-io steps). During a PURE idle-flush tick the only wrapper
    // constructor on the engine thread is `read_block_bytes`, whose
    // shadowed slice + contiguous pair holds exactly 2 objects mid-copy —
    // the P9-characterized transient and the exact magnitude of the PR #35
    // CI signature. An async sampler races the drain of a genuine
    // post-cancel-burst backlog and must never read more than floor + 2 in
    // that phase, and exactly floor once the queue is empty.
    let phase = Arc::new(AtomicU8::new(0)); // 0 traffic/settle, 2 idle-flush, 3 drained, 4 stop
    let flush_max = Arc::new(AtomicI64::new(i64::MIN));
    let flush_plus2 = Arc::new(AtomicU64::new(0));
    let flush_reads = Arc::new(AtomicU64::new(0));
    let drained_bad = Arc::new(AtomicI64::new(0));
    let sampler = {
        let phase = Arc::clone(&phase);
        let flush_max = Arc::clone(&flush_max);
        let flush_plus2 = Arc::clone(&flush_plus2);
        let flush_reads = Arc::clone(&flush_reads);
        let drained_bad = Arc::clone(&drained_bad);
        let floor = ssd_baseline;
        std::thread::spawn(move || {
            loop {
                let p = phase.load(Ordering::Acquire);
                let live = debug::live_objects();
                if phase.load(Ordering::Acquire) != p {
                    // Phase flipped mid-read: attribution would be racy.
                    continue;
                }
                match p {
                    2 => {
                        flush_reads.fetch_add(1, Ordering::Relaxed);
                        flush_max.fetch_max(live, Ordering::Relaxed);
                        if live == floor + 2 {
                            flush_plus2.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    3 => {
                        if live != floor {
                            drained_bad.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    4 => break,
                    _ => {}
                }
            }
        })
    };
    for iter in 0u32..4 {
        // Distinct token bases per sequence per iteration: every donated
        // full block is novel content, so the whole burst enqueues for
        // flush (known content is marked on-SSD at donation and skipped).
        let base = 40 + iter * 90;
        let reqs: Vec<Req> = (0..3)
            .map(|j| {
                let start = base + j * 30;
                submit(
                    &mut eng,
                    &(start..start + 7).collect::<Vec<_>>(),
                    24,
                    keep(),
                )
            })
            .collect();
        step_until(&mut eng, || {
            reqs.iter().all(|r| r.tokens.borrow().len() >= 20)
        });
        // The CI quiesce shape: a synchronized abort burst, settled on the
        // engine thread (donations enqueue their novel blocks here).
        for r in &reqs {
            r.cancel.store(true, Ordering::Release);
        }
        while !eng.is_idle() {
            eng.step().expect("engine step");
        }
        assert!(
            eng.has_pending_cache_io(),
            "R6 iter {iter}: the burst enqueued no novel flushes — the \
             scenario demonstrates nothing"
        );
        // Pure idle-flush ticks: the exact engine state a CI checkpoint
        // samples when the post-burst flush tail outlives the settle.
        phase.store(2, Ordering::Release);
        let flush_started = Instant::now();
        let mut ticks = 0u32;
        while eng.has_pending_cache_io() {
            eng.step().expect("flush tick");
            ticks += 1;
        }
        eng.step().expect("post-flush tick");
        phase.store(3, Ordering::Release);
        let flush_ms = flush_started.elapsed().as_secs_f64() * 1e3;
        for _ in 0..20 {
            eng.step().expect("idle tick");
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_floor(ssd_baseline, &format!("R6 iter {iter}"));
        eprintln!(
            "R6 iter {iter}: flush backlog drained in {ticks} ticks / {flush_ms:.1} ms \
             (<= 2 block captures per tick)"
        );
        phase.store(0, Ordering::Release);
    }
    phase.store(4, Ordering::Release);
    sampler.join().expect("sampler joins");
    let reads = flush_reads.load(Ordering::Relaxed);
    let max = flush_max.load(Ordering::Relaxed);
    let plus2 = flush_plus2.load(Ordering::Relaxed);
    let bad = drained_bad.load(Ordering::Relaxed);
    assert_eq!(
        bad, 0,
        "async sampler read non-floor live objects AFTER the flush queue \
         drained — that would be a real parked leak"
    );
    if reads > 0 {
        assert!(
            max <= ssd_baseline + 2,
            "idle-flush ticks exposed +{} live objects over floor — more \
             than read_block_bytes' slice+contiguous pair; a third \
             mechanism exists",
            max - ssd_baseline
        );
        eprintln!(
            "R6: {reads} async samples during idle-flush ticks; max +{} over \
             floor; floor+2 read {plus2} times (the CI checkpoint signature, \
             sampled live, with zero leak)",
            (max - ssd_baseline).max(0)
        );
    }

    drop(eng);
    let _ = std::fs::remove_dir_all(&slab_dir);
    eprintln!("all cancel windows quiesced at baseline (kernel path in-graph)");
}
