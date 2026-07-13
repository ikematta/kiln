//! Engine step-overhead bench (SPEC §6.2, Phase 4 acceptance): steady-state
//! decode step overhead **outside the MLX forward call** must be < 200µs at
//! batch 16.
//!
//! Strategy: drive the real engine with a null [`StepModel`] whose
//! `forward_step` returns a clone of one precomputed constant logits array,
//! so a measured `Engine::step()` is everything the engine does *around*
//! the forward — cancel sweep, admission, segment planning, block-table
//! appends, write-run derivation, the per-request sampling graphs
//! (slice/logsumexp/argmax), the step-boundary eval, token readback, event
//! emission, and preemption bookkeeping. KV writes are excluded: in
//! production the model issues them inside the forward call.
//!
//! A null forward makes the step-boundary `eval` stand alone, so the
//! headline number carries a fixed Metal round-trip the real engine pays
//! *inside* the step's single eval alongside the forward. Two attribution
//! benches split that out: `sampling_graph_build_batch16` (host-side graph
//! construction — genuine overhead) and `sampling_eval_floor_batch16` (the
//! standalone eval+readback round-trip — a bench artifact). The §6.2
//! non-GPU overhead is `step_overhead − sampling_eval_floor`.
//!
//! Run: `cargo bench -p kiln-engine` (skips without a Metal device).

#[cfg(feature = "metal")]
mod imp {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    use criterion::Criterion;
    use kiln_engine::{
        Engine, EngineConfig, EngineRequest, KvDims, PagedKv, PenaltyOptions, Priority,
        SamplingOptions, StepBatch, StepModel,
    };
    use kiln_mlx::{Array, MlxError, Stream, eval, memory, ops};

    const BATCH: usize = 16;
    const VOCAB: i32 = 256;

    /// Zero-cost forward: hands back the same evaluated logits every step.
    struct NullModel {
        logits: Array,
    }

    impl NullModel {
        fn new() -> Result<Self, MlxError> {
            // [1, BATCH, VOCAB], argmax at index 3 for every row: the
            // engine samples a constant non-stop token forever.
            let mut row = vec![0.0f32; VOCAB as usize];
            row[3] = 1.0;
            let data: Vec<f32> = row
                .iter()
                .cycle()
                .take(BATCH * VOCAB as usize)
                .copied()
                .collect();
            let logits = Array::from_f32_slice(&data, &[1, BATCH as i32, VOCAB])?;
            logits.eval()?;
            Ok(Self { logits })
        }
    }

    impl StepModel for NullModel {
        fn forward_step(
            &self,
            batch: &StepBatch,
            _kv: &mut PagedKv,
            _s: &Stream,
        ) -> Result<Option<Array>, MlxError> {
            let sampled = batch.seqs.iter().filter(|seq| seq.sample).count();
            if sampled == 0 {
                return Ok(None);
            }
            debug_assert_eq!(sampled, BATCH, "bench holds a full batch of {BATCH}");
            Ok(Some(self.logits.clone()))
        }
    }

    /// A fresh engine with `BATCH` sequences already decoding. 1-token
    /// prompts skip prefill entirely, so every measured step is a pure
    /// 16-wide decode step; `max_tokens: usize::MAX` keeps them alive for
    /// the whole sample. The pool covers ~130k steps — far beyond what one
    /// criterion sample draws — and the post-loop preemption assert
    /// guarantees the measurement never silently degraded into thrash.
    fn build_engine() -> Engine<NullModel> {
        let model = NullModel::new().expect("null model builds");
        let dims = KvDims {
            layers: 1,
            kv_heads: 1,
            head_dim: 8,
        };
        let config = EngineConfig {
            num_blocks: 65_536,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(model, dims, config, Stream::gpu()).expect("engine builds");
        for _ in 0..BATCH {
            engine.submit(EngineRequest {
                prompt: vec![1],
                max_tokens: usize::MAX,
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
                on_event: Box::new(|_| true),
            });
        }
        engine
    }

    /// The engine's per-row sampling chain (mirror of `run_iteration`'s
    /// greedy path: slice → reshape → logsumexp → subtract → argmax), used
    /// by the attribution benches below.
    fn build_sampling_chains(logits: &Array, s: &Stream) -> Vec<Array> {
        let vocab = logits.dim(2);
        (0..BATCH as i32)
            .map(|row| {
                let last =
                    ops::slice(logits, &[0, row, 0], &[1, row + 1, vocab], s).expect("slice");
                let last = ops::reshape(&last, &[1, vocab], s).expect("reshape");
                let logprobs = ops::subtract(
                    &last,
                    &ops::logsumexp(&last, true, s).expect("logsumexp"),
                    s,
                )
                .expect("subtract");
                ops::argmax(&logprobs, -1, false, s).expect("argmax")
            })
            .collect()
    }

    pub fn run() {
        if !memory::metal_is_available() {
            eprintln!("skipping step_overhead bench: no Metal device");
            return;
        }
        kiln_mlx::init();
        let mut c = Criterion::default().configure_from_args();
        let mut group = c.benchmark_group("engine");
        group
            .sample_size(20)
            .warm_up_time(Duration::from_secs(2))
            .measurement_time(Duration::from_secs(4));

        // The headline number: a full engine step around a null forward.
        // Includes the standalone eval round-trip measured separately below.
        group.bench_function("step_overhead_batch16", |b| {
            b.iter_custom(|iters| {
                let mut engine = build_engine();
                // Warm one step: admission + first-token bookkeeping.
                engine.step().expect("warm step");
                let start = Instant::now();
                for _ in 0..iters {
                    engine.step().expect("engine step");
                }
                let elapsed = start.elapsed();
                assert_eq!(engine.num_running(), BATCH, "a bench request died");
                assert_eq!(engine.preemptions(), 0, "bench pool thrashed");
                elapsed
            });
        });

        // Attribution 1 (CPU, counts toward the §6.2 overhead target):
        // host-side construction of the 16 per-row sampling graphs.
        let model = NullModel::new().expect("null model builds");
        let stream = Stream::gpu();
        group.bench_function("sampling_graph_build_batch16", |b| {
            b.iter(|| build_sampling_chains(&model.logits, &stream));
        });

        // Attribution 2 (GPU round-trip, absorbed by the forward's
        // step-boundary eval in production — a null-forward artifact):
        // evaluating those graphs standalone and reading the tokens back.
        group.bench_function("sampling_eval_floor_batch16", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let tokens = build_sampling_chains(&model.logits, &stream);
                    let refs: Vec<&Array> = tokens.iter().collect();
                    let start = Instant::now();
                    eval(&refs).expect("eval");
                    for token in &tokens {
                        token.item_u32().expect("readback");
                    }
                    total += start.elapsed();
                }
                total
            });
        });

        group.finish();
        c.final_summary();
    }
}

fn main() {
    #[cfg(feature = "metal")]
    imp::run();
}
