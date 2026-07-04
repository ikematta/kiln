//! The continuous batching loop (SPEC §6.2): a single engine loop owns the
//! Metal stream, and every iteration admits waiting requests, builds one
//! step (all decoding sequences + at most one prefill chunk), runs the
//! forward pass(es), samples, streams tokens, and releases the blocks of
//! finished requests.
//!
//! Determinism (CLAUDE.md): batching must not change greedy outputs, so a
//! step is built as **two** forward calls sharing the KV pools:
//! - the prefill chunk runs alone, with exactly the op shapes and chunk
//!   boundaries of the Phase-3 single-request path (chunks of
//!   `prefill_chunk` over `prompt[..n-1]`, mirroring mlx-lm's
//!   `generate_step`) — a request's prefill numerics can never depend on
//!   what else is in the batch;
//! - the decoding sequences' tokens run concatenated (SPEC §6.2 step 2);
//!   position-wise trunk ops are row-independent at small batch widths, and
//!   `tests/batching.rs` in kiln-models asserts batched greedy outputs are
//!   bit-identical to single-stream runs.
//!
//! Both calls are evaluated together at the step boundary.
//!
//! Preemption (SPEC §6.1): when the block pool cannot cover a planned
//! segment, the planner frees a victim — lowest priority first, then most
//! recently admitted — which returns to WAITING with its generated tokens
//! retained. A resumed request re-prefills its prompt with the original
//! chunk boundaries and then **replays** its generated tokens as
//! single-token, non-sampled steps. Replaying them as one prefill-style
//! chunk would move the trunk matmuls to a different M and the attention
//! to a multi-token query length — kernel dispatches this codebase has not
//! validated for bit-parity (see the mlx#3120 note in PROGRESS.md) —
//! while single-token steps are exactly the shapes the batching tests
//! already pin against solo runs. Replay emits nothing: those tokens were
//! streamed before the preemption; `tests/preemption.rs` in kiln-models
//! asserts a preempted request resumes onto the identical token stream.
//!
//! Decode pipelining (SPEC §6.2, the async_eval pipeline of
//! `generate.rs` lifted to the batch): a steady-state pure-decode step
//! defers its token readback — the next step's forward is built feeding
//! the previous step's *still-lazy* sampled arrays and scheduled with
//! `async_eval` before the previous tokens are read, so host-side graph
//! construction overlaps GPU execution. The pipeline engages only when
//! every running sequence is sampling with penalties off, nothing is
//! waiting, no cancel flag is up, and the next single-token appends
//! provably fit the pool — prefill, replay, admission, preemption, and
//! capacity decisions always run on the synchronous path with fully
//! applied state, so scheduling semantics (victim choice, seniority,
//! admission projection) are identical to the unpipelined engine. A
//! sequence that stops mid-pipeline has one speculative row in flight;
//! its readback is discarded and its KV write lands in blocks the
//! sequence owned at build time — releasing them is safe because a
//! future owner rewrites every row below its own length and gathers are
//! trimmed to that length (see `PagedKv::gather`), but NOTE for Phase 5:
//! radix sharing must re-review this invariant before blocks with a
//! stale speculative tail row can enter the prefix cache.

use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kiln_mlx::{Array, MlxError, Stream, async_eval, eval, memory, ops};
use thiserror::Error;

use crate::block::{BlockError, BlockManager, BlockTable};
use crate::paged::{KvSpec, PagedKv, WriteRun};
use crate::sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
use crate::step::{SeqStep, StepBatch, StepInput, StepModel};

/// SPEC §10 `[defaults]` block_size.
pub const DEFAULT_BLOCK_SIZE: usize = 32;
/// SPEC §6.2 `max_batch_tokens` default.
pub const DEFAULT_MAX_BATCH_TOKENS: usize = 8192;
/// SPEC §6.2 `prefill_chunk` default (also mlx-lm's `prefill_step_size` —
/// chunk boundaries are part of the golden-parity contract).
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;
/// Pool capacity until memory-budget admission lands (SPEC §6.4, Phase 4
/// part 3+): 512 blocks × 32 tokens = 16k tokens of KV.
pub const DEFAULT_NUM_BLOCKS: usize = 512;

/// Cache maintenance cadence in decode iterations (mirrors the Phase-3
/// loop's `clear_cache` every 256 steps).
const MAINTENANCE_INTERVAL: u64 = 256;

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_batch_tokens: usize,
    pub prefill_chunk: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            num_blocks: DEFAULT_NUM_BLOCKS,
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
        }
    }
}

/// KV geometry of the model being served.
#[derive(Debug, Clone, Copy)]
pub struct KvDims {
    pub layers: usize,
    pub kv_heads: i32,
    pub head_dim: i32,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Block(#[from] BlockError),
    #[error("invalid engine config: {0}")]
    Config(String),
}

/// Request priority (proto `Priority`). Variant order is the preemption
/// order: `Batch` sorts below `Interactive`, so it is preempted first
/// under memory pressure (SPEC §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Proto `PRIORITY_BATCH`: preempted first.
    Batch,
    /// Proto `PRIORITY_INTERACTIVE` (and `UNSPECIFIED`): the default.
    Interactive,
}

/// Why a sequence finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishKind {
    /// A stop token id was sampled (counted in `completion_tokens`, not
    /// emitted as a token event).
    Stop,
    /// `max_tokens` generated.
    Length,
    /// Cancel flag set, or the event sink reported the receiver gone.
    Cancelled,
    /// The request failed; see `error`/`error_cause`.
    Error,
}

/// What class of failure produced `FinishKind::Error` — drives the proto
/// `WorkerErrorCode` in the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCause {
    /// The request can never fit the KV pool (admission refusal; proto
    /// `WORKER_ERROR_OOM_REJECTED`).
    Capacity,
    /// Engine/MLX fault (proto `WORKER_ERROR_INTERNAL`).
    Internal,
}

/// Terminal report for one request.
#[derive(Debug, Clone)]
pub struct FinishSummary {
    pub reason: FinishKind,
    /// Generated tokens, including a matched stop token.
    pub completion_tokens: u32,
    pub matched_stop_token: Option<u32>,
    pub error: Option<String>,
    /// Set iff `reason == Error`.
    pub error_cause: Option<ErrorCause>,
    /// Times this request was preempted (and later resumed) under memory
    /// pressure (SPEC §6.1).
    pub preemptions: u32,
    /// Submit → first sampled token readable.
    pub prefill_seconds: f64,
    /// First sampled token → finish.
    pub decode_seconds: f64,
}

/// Events delivered to a request's sink, on the engine thread. The sink
/// returns `false` when the consumer is gone (the engine then cancels the
/// request); the return value of a `Finished` delivery is ignored.
pub enum SeqEvent {
    Token(u32),
    Finished(FinishSummary),
}

pub type EventSink = Box<dyn FnMut(SeqEvent) -> bool>;

/// One generation request, engine-facing (the worker maps proto
/// `SubmitRequest` fields onto this; the caller correlates events through
/// its `on_event` sink and cancels through the shared flag, so no id
/// crosses this boundary).
pub struct EngineRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingOptions,
    pub penalties: PenaltyOptions,
    /// Recent-token window for the penalties (ignored when disabled).
    pub penalty_window: usize,
    pub stop_tokens: HashSet<u32>,
    /// Preemption class (SPEC §6.1).
    pub priority: Priority,
    pub cancel: Arc<AtomicBool>,
    pub on_event: EventSink,
}

/// Internal per-request state.
///
/// Lifecycle (SPEC §6.1): WAITING (in `Engine::waiting`) → PREFILLING
/// (`processed < prompt.len()-1`) → DECODING (feeding `history[fed]`,
/// sampling when it is the newest history entry, replaying it silently
/// when a preemption discarded its KV) → FINISHED. PREEMPTED = blocks
/// released, `processed`/`fed` rewound to 0, back into `waiting` with
/// `history` (the generated tokens) intact.
struct Seq {
    prompt: Vec<u32>,
    max_tokens: usize,
    stop_tokens: HashSet<u32>,
    sampler: Sampler,
    penalties: PenaltyOptions,
    penalty_window: usize,
    priority: Priority,
    /// Submit-order sequence number; stable across preemption, so a
    /// resumed request keeps its seniority ("most-recently-admitted"
    /// victims are picked by the highest `arrival`).
    arrival: u64,
    cancel: Arc<AtomicBool>,
    on_event: EventSink,
    table: BlockTable,
    /// Prompt tokens whose KV has been written (prefill progress over
    /// `prompt[..n-1]`).
    processed: usize,
    /// mlx-lm logits-processor history: last prompt token + generated.
    history: Vec<u32>,
    /// `history` tokens fed to the model (KV written). Steady state keeps
    /// `fed == history.len() - 1` — feed the newest entry, sample, push;
    /// after a preemption `fed` rewinds to 0 and the gap is replayed
    /// without sampling or emitting.
    fed: usize,
    generated: u32,
    preemptions: u32,
    /// Preempted during the current iteration; moved back to `waiting`
    /// (and reset) once the step completes.
    preempted: bool,
    submitted_at: Instant,
    first_token_at: Option<Instant>,
    finished: bool,
}

impl Seq {
    /// Prefill is complete: the sequence feeds decode/replay tokens.
    fn decoding(&self) -> bool {
        self.processed + 1 >= self.prompt.len()
    }

    /// SPEC §6.1 preemption order, as a key where the *minimum* is the
    /// first victim: lowest priority first, then most recently admitted.
    fn deservingness(&self) -> (Priority, Reverse<u64>) {
        (self.priority, Reverse(self.arrival))
    }

    /// Pool blocks this request needs to reach its next sampled token:
    /// full prefill plus (after a preemption) the replay of everything
    /// generated so far. The §6.4 admission projection — a request is
    /// admitted only once this fits in free blocks, so admission cannot
    /// thrash straight back into preemption.
    fn admission_blocks(&self, block_size: usize) -> usize {
        (self.prompt.len() - 1 + self.history.len().max(1)).div_ceil(block_size)
    }
}

/// Emits the terminal event and releases the sequence's blocks.
fn finish(
    seq: &mut Seq,
    mgr: &mut BlockManager,
    reason: FinishKind,
    matched_stop_token: Option<u32>,
    error: Option<(ErrorCause, String)>,
) {
    if seq.finished {
        return;
    }
    seq.finished = true;
    let now = Instant::now();
    let first = seq.first_token_at.unwrap_or(now);
    let (error_cause, error) = match error {
        Some((cause, detail)) => (Some(cause), Some(detail)),
        None => (None, None),
    };
    let summary = FinishSummary {
        reason,
        completion_tokens: seq.generated,
        matched_stop_token,
        error,
        error_cause,
        preemptions: seq.preemptions,
        prefill_seconds: first.duration_since(seq.submitted_at).as_secs_f64(),
        decode_seconds: now.duration_since(first).as_secs_f64(),
    };
    let _ = (seq.on_event)(SeqEvent::Finished(summary));
    let released = std::mem::take(&mut seq.table).release(mgr);
    debug_assert!(released.is_ok(), "finish released a foreign block");
}

/// One planned decode/replay segment (index into `running` + its step).
struct DecodePlan {
    seq: usize,
    sample: bool,
    step: SeqStep,
}

/// One scheduled-but-unread decode step: forward and sampling graphs are
/// queued on the stream via `async_eval`; the sampled token values are
/// not host-visible yet, and all per-sequence bookkeeping (history push,
/// emission, stop checks) is deferred to the apply at the start of the
/// next `step()` call.
struct InFlight {
    /// `(arrival, sampled token [1] u32)` in plan order. Arrival ids are
    /// unique and survive `running` mutations between build and apply.
    rows: Vec<(u64, Array)>,
}

/// Last-position logits row -> sampled token `[1]` u32 (the SPEC §6.2
/// step-3 tail: penalties over the recent window, logprob-normalize,
/// sample). Shared by the synchronous and pipelined paths; the pipelined
/// path only ever calls it with penalties disabled, so both build the
/// identical op graph.
fn sample_from_row(
    seq: &mut Seq,
    logits: &Array,
    row: i32,
    s: &Stream,
) -> Result<Array, EngineError> {
    let vocab = logits.dim(2);
    let last = ops::slice(logits, &[0, row, 0], &[1, row + 1, vocab], s)?;
    let mut last = ops::reshape(&last, &[1, vocab], s)?;
    if !seq.penalties.is_disabled() {
        let window = seq.history.len().saturating_sub(seq.penalty_window);
        last = apply_penalties(&last, &seq.history[window..], seq.penalties, s)?;
    }
    let logprobs = ops::subtract(&last, &ops::logsumexp(&last, true, s)?, s)?;
    Ok(seq.sampler.sample(&logprobs, s)?)
}

/// Applies one read-back sampled token to its sequence: cursor advance,
/// then the cancel/stop/emission/length checks (SPEC §6.2 steps 3-4).
/// Identical on the synchronous and pipelined paths — only *when* it
/// runs differs (immediately vs at the next `step()` call).
fn settle_sampled(seq: &mut Seq, mgr: &mut BlockManager, token: u32) {
    if seq.finished {
        return;
    }
    if seq.first_token_at.is_none() {
        seq.first_token_at = Some(Instant::now());
    }
    seq.generated += 1;
    seq.history.push(token);
    seq.fed += 1;
    if seq.cancel.load(Ordering::Acquire) {
        finish(seq, mgr, FinishKind::Cancelled, None, None);
    } else if seq.stop_tokens.contains(&token) {
        // Counted in usage but not emitted: stop text is excluded from
        // the stream by contract.
        finish(seq, mgr, FinishKind::Stop, Some(token), None);
    } else if !(seq.on_event)(SeqEvent::Token(token)) {
        finish(seq, mgr, FinishKind::Cancelled, None, None);
    } else if seq.generated as usize >= seq.max_tokens {
        finish(seq, mgr, FinishKind::Length, None, None);
    }
}

/// The continuous-batching engine. Single-threaded by construction: it
/// owns the `Stream` (`!Send`), so the whole engine is confined to the
/// thread that created it (SPEC §6.2).
pub struct Engine<M> {
    model: M,
    stream: Stream,
    config: EngineConfig,
    mgr: BlockManager,
    kv: PagedKv,
    /// Sorted by `arrival` ascending: fresh submits append with increasing
    /// numbers and preempted requests re-insert in order.
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    /// The scheduled-but-unread decode step, when pipelining (never
    /// `Some` while the synchronous path plans/preempts/admits).
    inflight: Option<InFlight>,
    next_arrival: u64,
    preemptions_total: u64,
    iterations: u64,
}

impl<M: StepModel> Engine<M> {
    pub fn new(
        model: M,
        dims: KvDims,
        config: EngineConfig,
        stream: Stream,
    ) -> Result<Self, EngineError> {
        if dims.layers == 0 || dims.kv_heads <= 0 || dims.head_dim <= 0 {
            return Err(EngineError::Config(format!("invalid kv dims: {dims:?}")));
        }
        if config.prefill_chunk == 0 || config.max_batch_tokens == 0 {
            return Err(EngineError::Config(
                "prefill_chunk and max_batch_tokens must be >= 1".to_owned(),
            ));
        }
        config
            .num_blocks
            .checked_mul(config.block_size)
            .filter(|&tokens| tokens <= i32::MAX as usize)
            .ok_or_else(|| {
                EngineError::Config(format!(
                    "pool of {} blocks x {} tokens overflows i32 addressing",
                    config.num_blocks, config.block_size
                ))
            })?;
        let mgr = BlockManager::new(config.num_blocks, config.block_size)?;
        let kv = PagedKv::new(KvSpec {
            layers: dims.layers,
            kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            num_blocks: config.num_blocks,
            block_size: config.block_size,
        });
        Ok(Self {
            model,
            stream,
            config,
            mgr,
            kv,
            waiting: VecDeque::new(),
            running: Vec::new(),
            inflight: None,
            next_arrival: 0,
            preemptions_total: 0,
            iterations: 0,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Requests currently admitted or queued.
    pub fn num_active(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Requests queued (or preempted back) to WAITING.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Requests currently prefilling or decoding.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Total preemptions since engine start (proto
    /// `WorkerStats.requests_preempted` counts events, not requests).
    pub fn preemptions(&self) -> u64 {
        self.preemptions_total
    }

    pub fn is_idle(&self) -> bool {
        self.num_active() == 0
    }

    /// Bytes backing the KV pools (0 until first use).
    pub fn kv_allocated_bytes(&self) -> u64 {
        self.kv.allocated_bytes()
    }

    /// Bytes of pool blocks currently owned by live requests.
    pub fn kv_used_bytes(&self) -> u64 {
        let live = (self.mgr.capacity() - self.mgr.num_free()) as u64;
        live * self.kv.bytes_per_block()
    }

    /// Queues a request. Invalid or unservable requests get an immediate
    /// `Finished{Error}` event instead of an `Err` (mirroring the proto's
    /// in-band error contract).
    pub fn submit(&mut self, request: EngineRequest) {
        let EngineRequest {
            prompt,
            max_tokens,
            sampling,
            penalties,
            penalty_window,
            stop_tokens,
            priority,
            cancel,
            on_event,
        } = request;
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        let mut seq = Seq {
            prompt,
            max_tokens,
            stop_tokens,
            sampler: Sampler::new(sampling),
            penalties,
            penalty_window,
            priority,
            arrival,
            cancel,
            on_event,
            table: BlockTable::new(),
            processed: 0,
            history: Vec::new(),
            fed: 0,
            generated: 0,
            preemptions: 0,
            preempted: false,
            submitted_at: Instant::now(),
            first_token_at: None,
            finished: false,
        };
        let error = if seq.prompt.is_empty() {
            Some((ErrorCause::Internal, "empty prompt".to_owned()))
        } else if seq.max_tokens == 0 {
            Some((ErrorCause::Internal, "max_tokens must be >= 1".to_owned()))
        } else {
            let needed = seq.prompt.len() + seq.max_tokens;
            let capacity = self.config.num_blocks * self.config.block_size;
            (needed > capacity).then(|| {
                (
                    ErrorCause::Capacity,
                    format!(
                        "prompt ({}) + max_tokens ({}) exceeds the KV pool ({capacity} tokens)",
                        seq.prompt.len(),
                        seq.max_tokens
                    ),
                )
            })
        };
        if let Some(error) = error {
            finish(
                &mut seq,
                &mut self.mgr,
                FinishKind::Error,
                None,
                Some(error),
            );
            return;
        }
        // A 1-token prompt has no prefill: it decodes immediately.
        if seq.prompt.len() == 1 {
            seq.history.push(seq.prompt[0]);
        }
        self.waiting.push_back(seq);
    }

    /// One SPEC §6.2 iteration. On `Err` every in-flight request has been
    /// failed and the pools reset; the engine stays usable.
    pub fn step(&mut self) -> Result<(), EngineError> {
        if let Some(prev) = self.inflight.take() {
            match self.pipelined_turn(prev) {
                // A new step is in flight; the turn is complete.
                Ok(true) => return Ok(()),
                // Pipeline drained (readback applied); fall through to
                // the synchronous path on fully-applied state.
                Ok(false) => {}
                Err(err) => {
                    self.fail_all(&err.to_string());
                    return Err(err);
                }
            }
        }
        self.sweep_cancelled();
        self.admit();
        if self.running.is_empty() {
            return Ok(());
        }
        match self.run_iteration() {
            Ok(()) => Ok(()),
            Err(err) => {
                self.fail_all(&err.to_string());
                Err(err)
            }
        }
    }

    /// One pipelined turn: schedule the next pure-decode step from the
    /// still-unread sampled tokens (the pipeline win — graph build
    /// overlaps the GPU executing `prev`), then read `prev` back and
    /// settle it. Returns whether a new step is now in flight; `false`
    /// hands control back to the synchronous path, where admission,
    /// prefill, replay, and preemption live.
    fn pipelined_turn(&mut self, prev: InFlight) -> Result<bool, EngineError> {
        let next = if self.pipeline_ok() {
            self.build_pipelined(&prev)?
        } else {
            None
        };
        self.apply_inflight(prev)?;
        let Some(mut next) = next else {
            return Ok(false);
        };
        // Sequences that stopped or were cancelled at the apply have one
        // speculative row in flight: drop it unread. Its compute is dead
        // graph and its KV write went to blocks the sequence owned at
        // build time (safe to release, see the module docs).
        next.rows
            .retain(|(arrival, _)| self.running.iter().any(|seq| seq.arrival == *arrival));
        self.inflight = (!next.rows.is_empty()).then_some(next);
        Ok(true)
    }

    /// The async_eval pipeline may only span steady-state decode: every
    /// running sequence sampling, penalties off (their windows need the
    /// previous token host-side), nothing waiting (admission and prefill
    /// want fresh pool state), no cancel pending (the sweep runs on the
    /// synchronous path). Capacity is checked exactly in
    /// `build_pipelined`.
    fn pipeline_ok(&self) -> bool {
        self.waiting.is_empty()
            && self.running.iter().all(|seq| {
                seq.decoding() && seq.penalties.is_disabled() && !seq.cancel.load(Ordering::Acquire)
            })
    }

    /// Builds and schedules the next decode step feeding `prev`'s
    /// still-lazy sampled tokens. Returns `None` (no table slot touched)
    /// when the step should run synchronously instead: a sequence
    /// vanished, everyone is finishing, or the appends would not fit —
    /// preemption stays a synchronous-path decision.
    fn build_pipelined(&mut self, prev: &InFlight) -> Result<Option<InFlight>, EngineError> {
        // Pass 1 (read-only): which rows continue, and do their appends
        // provably fit? Never append a slot the step might not execute —
        // an unexecuted slot would desync `table` from the KV it claims.
        let block_size = self.mgr.block_size();
        let mut included: Vec<(usize, &Array)> = Vec::with_capacity(prev.rows.len());
        let mut new_blocks = 0;
        for (arrival, pending) in &prev.rows {
            let Some(i) = self.running.iter().position(|seq| seq.arrival == *arrival) else {
                return Ok(None);
            };
            let seq = &self.running[i];
            if seq.generated as usize + 1 >= seq.max_tokens {
                // Its final token is in `prev`; it finishes at the apply
                // (mirrors generate.rs's `i + 1 < max_tokens` guard).
                continue;
            }
            if seq.table.num_tokens() == seq.table.blocks().len() * block_size {
                new_blocks += 1;
            }
            included.push((i, pending));
        }
        if included.is_empty() || self.mgr.num_free() < new_blocks {
            return Ok(None);
        }

        // Pass 2: append the slots and build the step. Inputs are the
        // pending `[1]` arrays reshaped and concatenated to `[1, n]`.
        self.iterations += 1;
        let mut seqs = Vec::with_capacity(included.len());
        let mut inputs = Vec::with_capacity(included.len());
        for &(i, pending) in &included {
            let step = build_seq_step(
                &mut self.running[i],
                &mut self.mgr,
                &mut self.kv,
                1,
                true,
                &self.stream,
            )?
            .ok_or_else(|| MlxError {
                message: "pipelined append exceeded the precounted pool".to_owned(),
            })?;
            seqs.push(step);
            inputs.push(ops::reshape(pending, &[1, 1], &self.stream)?);
        }
        let refs: Vec<&Array> = inputs.iter().collect();
        let input = ops::concatenate(&refs, 1, &self.stream)?;
        let batch = StepBatch {
            input: StepInput::Lazy(input),
            seqs,
        };
        let logits = self
            .model
            .forward_step(&batch, &mut self.kv, &self.stream)?
            .ok_or_else(|| MlxError {
                message: "pipelined decode returned no logits".to_owned(),
            })?;
        let mut rows = Vec::with_capacity(included.len());
        for (row, &(i, _)) in included.iter().enumerate() {
            let seq = &mut self.running[i];
            debug_assert!(
                seq.penalties.is_disabled(),
                "pipeline_ok admitted penalties"
            );
            let token = sample_from_row(seq, &logits, row as i32, &self.stream)?;
            rows.push((seq.arrival, token));
        }
        {
            let outputs: Vec<&Array> = rows.iter().map(|(_, token)| token).collect();
            // Sampled tokens only: every row samples, so each sequence's
            // KV write is an ancestor of its token; the pool handles stay
            // solely on the write chain and keep their donation.
            async_eval(&outputs)?;
        }
        Ok(Some(InFlight { rows }))
    }

    /// Reads back a scheduled step and settles it: emission, stop/cancel
    /// checks, finishes, and the maintenance cadence.
    fn apply_inflight(&mut self, prev: InFlight) -> Result<(), EngineError> {
        for (arrival, token) in &prev.rows {
            let Some(i) = self.running.iter().position(|seq| seq.arrival == *arrival) else {
                debug_assert!(false, "in-flight row for a departed sequence");
                continue;
            };
            let token = token.item_u32()?;
            settle_sampled(&mut self.running[i], &mut self.mgr, token);
        }
        self.running.retain(|seq| !seq.finished);
        if self.iterations.is_multiple_of(MAINTENANCE_INTERVAL) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    /// Cancels flagged requests between steps (proto `Cancel`: the flag is
    /// honored within 2 engine steps; this sweep plus the post-sample
    /// check keep it within 1).
    fn sweep_cancelled(&mut self) {
        let mgr = &mut self.mgr;
        for seq in &mut self.waiting {
            if seq.cancel.load(Ordering::Acquire) {
                finish(seq, mgr, FinishKind::Cancelled, None, None);
            }
        }
        self.waiting.retain(|seq| !seq.finished);
        for seq in &mut self.running {
            if seq.cancel.load(Ordering::Acquire) {
                finish(seq, mgr, FinishKind::Cancelled, None, None);
            }
        }
        self.running.retain(|seq| !seq.finished);
    }

    /// SPEC §6.2 step 1: admit while the token budget allows. One request
    /// prefills at a time (each step carries at most one prefill chunk),
    /// so admission pauses while a prefill is in flight; the head of the
    /// queue additionally waits until its admission projection fits in
    /// free blocks (SPEC §6.4).
    fn admit(&mut self) {
        loop {
            let mid_prefill = self.running.iter().any(|seq| !seq.decoding());
            let budget_left = self.running.len() < self.config.max_batch_tokens;
            if mid_prefill || !budget_left {
                return;
            }
            let Some(head) = self.waiting.front() else {
                return;
            };
            if self.mgr.num_free() < head.admission_blocks(self.mgr.block_size()) {
                return;
            }
            match self.waiting.pop_front() {
                Some(seq) => self.running.push(seq),
                None => return,
            }
        }
    }

    fn run_iteration(&mut self) -> Result<(), EngineError> {
        self.iterations += 1;

        // --- Plan (SPEC §6.2 step 2, §6.1 preemption): decode/replay
        // segments in `running` order, then at most one prefill chunk.
        // Planning is pure bookkeeping (block tables + write runs), so a
        // preemption mid-plan drops the victim's segment before any
        // compute references its blocks.
        let mut decode_plans: Vec<DecodePlan> = Vec::new();
        for i in 0..self.running.len() {
            let seq = &self.running[i];
            if seq.finished || seq.preempted || !seq.decoding() {
                continue;
            }
            // Feed history[fed]; sample only at the newest entry — older
            // entries are post-preemption replay, already streamed.
            let sample = seq.fed + 1 == seq.history.len();
            if let Some(step) = self.plan_step(i, 1, sample, &mut decode_plans)? {
                decode_plans.push(DecodePlan {
                    seq: i,
                    sample,
                    step,
                });
            }
        }
        let mut prefill: Option<(usize, StepBatch)> = None;
        if let Some(i) = self
            .running
            .iter()
            .position(|seq| !seq.finished && !seq.preempted && !seq.decoding())
        {
            let budget = self
                .config
                .max_batch_tokens
                .saturating_sub(decode_plans.len());
            let seq = &self.running[i];
            // Phase-3/mlx-lm chunk rule: prefill covers prompt[..n-1]; the
            // last prompt token is fed by the first sampled (decode) step.
            let remaining = seq.prompt.len() - 1 - seq.processed;
            let chunk = remaining.min(self.config.prefill_chunk).min(budget);
            if chunk > 0
                && let Some(step) = self.plan_step(i, chunk, false, &mut decode_plans)?
            {
                let seq = &self.running[i];
                let tokens = seq.prompt[seq.processed..seq.processed + chunk].to_vec();
                prefill = Some((
                    i,
                    StepBatch {
                        input: StepInput::Ids(tokens),
                        seqs: vec![step],
                    },
                ));
            }
        }

        let mut decode_tokens: Vec<u32> = Vec::with_capacity(decode_plans.len());
        let mut decode_seqs: Vec<SeqStep> = Vec::with_capacity(decode_plans.len());
        let mut sampled_ids: Vec<usize> = Vec::new();
        let mut replay_ids: Vec<usize> = Vec::new();
        for plan in decode_plans {
            let seq = &self.running[plan.seq];
            decode_tokens.push(seq.history[seq.fed]);
            decode_seqs.push(plan.step);
            if plan.sample {
                sampled_ids.push(plan.seq);
            } else {
                replay_ids.push(plan.seq);
            }
        }
        let decode_batch = StepBatch {
            input: StepInput::Ids(decode_tokens),
            seqs: decode_seqs,
        };

        // --- Forward (SPEC §6.2 step 3). Prefill first so its KV writes
        // sit earliest on the pools' functional-update chain.
        if let Some((_, batch)) = &prefill {
            let logits = self.model.forward_step(batch, &mut self.kv, &self.stream)?;
            debug_assert!(logits.is_none(), "prefill chunks never sample");
        }
        let mut sampled: Vec<(usize, Array)> = Vec::with_capacity(sampled_ids.len());
        if !decode_batch.seqs.is_empty() {
            let logits = self
                .model
                .forward_step(&decode_batch, &mut self.kv, &self.stream)?;
            if sampled_ids.is_empty() {
                debug_assert!(logits.is_none(), "replay-only steps never sample");
            } else {
                let logits = logits.ok_or_else(|| MlxError {
                    message: "model returned no logits for a decode step".to_owned(),
                })?;
                for (row, &i) in sampled_ids.iter().enumerate() {
                    let token =
                        sample_from_row(&mut self.running[i], &logits, row as i32, &self.stream)?;
                    sampled.push((i, token));
                }
            }
        }

        // --- Pipeline entry (module docs): a steady-state pure-decode
        // step defers its readback with async_eval; the next step's
        // graph is built from these still-lazy tokens at the start of
        // the next step() call, overlapping this step's GPU execution.
        // Every row samples, so each sequence's KV write is an ancestor
        // of its token — no pool-state eval needed.
        if prefill.is_none()
            && replay_ids.is_empty()
            && sampled.len() == self.running.len()
            && self.pipeline_ok()
        {
            {
                let outputs: Vec<&Array> = sampled.iter().map(|(_, token)| token).collect();
                async_eval(&outputs)?;
            }
            self.inflight = Some(InFlight {
                rows: sampled
                    .into_iter()
                    .map(|(i, token)| (self.running[i].arrival, token))
                    .collect(),
            });
            return Ok(());
        }

        // --- Evaluate the step: sampled tokens + pool state together
        // (prefill/replay-only steps still materialize their KV writes
        // here).
        {
            let mut outputs: Vec<&Array> = sampled.iter().map(|(_, token)| token).collect();
            let state = self.kv.state();
            outputs.extend(state);
            eval(&outputs)?;
        }

        // --- Emit, check stops, release finished (SPEC §6.2 steps 3-4).
        for (i, token) in &sampled {
            let token = token.item_u32()?;
            settle_sampled(&mut self.running[*i], &mut self.mgr, token);
        }
        // Replay segments advance silently: their tokens were emitted
        // before the preemption.
        for &i in &replay_ids {
            self.running[i].fed += 1;
        }

        // --- Advance prefill progress; a fully prefilled sequence starts
        // (or, after a preemption, resumes) feeding its history.
        if let Some((i, batch)) = &prefill {
            let seq = &mut self.running[*i];
            seq.processed += batch.num_tokens();
            if seq.processed + 1 == seq.prompt.len() && seq.history.is_empty() {
                seq.history.push(seq.prompt[seq.processed]);
            }
        }

        // --- Preempted requests return to WAITING (SPEC §6.1), reset for
        // re-prefill; insertion keeps the queue sorted by arrival so a
        // resumed request keeps its seniority.
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].preempted {
                let mut seq = self.running.remove(i);
                seq.preempted = false;
                let at = self
                    .waiting
                    .iter()
                    .position(|w| w.arrival > seq.arrival)
                    .unwrap_or(self.waiting.len());
                self.waiting.insert(at, seq);
            } else {
                i += 1;
            }
        }
        self.running.retain(|seq| !seq.finished);

        // --- SPEC §6.2 step 5: cache maintenance (Phase-3 cadence: after
        // every prefill chunk, else every 256 iterations).
        if prefill.is_some() || self.iterations.is_multiple_of(MAINTENANCE_INTERVAL) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    /// Plans one segment (`len` KV slots) for `running[i]`, preempting
    /// less-deserving requests when the pool is exhausted (SPEC §6.1).
    /// Returns `None` when the requester itself yielded (self-preemption)
    /// or was failed in-band; preempted victims' segments are dropped from
    /// `plans`.
    fn plan_step(
        &mut self,
        i: usize,
        len: usize,
        sample: bool,
        plans: &mut Vec<DecodePlan>,
    ) -> Result<Option<SeqStep>, EngineError> {
        loop {
            if let Some(step) = build_seq_step(
                &mut self.running[i],
                &mut self.mgr,
                &mut self.kv,
                len,
                sample,
                &self.stream,
            )? {
                return Ok(Some(step));
            }
            // Out of blocks: preempt the least-deserving block-holder
            // (the requester competes with its own key, so a newcomer
            // can never displace a more deserving request).
            let victim = self
                .running
                .iter()
                .enumerate()
                .filter(|&(j, seq)| {
                    !seq.finished && !seq.preempted && (j == i || !seq.table.blocks().is_empty())
                })
                .min_by_key(|(_, seq)| seq.deservingness())
                .map(|(j, _)| j);
            match victim {
                Some(j) if j != i => {
                    self.preempt(j);
                    plans.retain(|plan| plan.seq != j);
                }
                _ => {
                    // The requester is the least deserving: it yields back
                    // to WAITING. If even an empty pool cannot cover the
                    // segment the request is unservable — unreachable, as
                    // submit() prechecks capacity — and fails in-band
                    // rather than retrying forever.
                    if self.running[i].table.blocks().is_empty()
                        && self.mgr.num_free() == self.mgr.capacity()
                    {
                        let error = format!(
                            "segment of {len} token(s) exceeds the KV pool ({} blocks x {})",
                            self.config.num_blocks, self.config.block_size
                        );
                        finish(
                            &mut self.running[i],
                            &mut self.mgr,
                            FinishKind::Error,
                            None,
                            Some((ErrorCause::Capacity, error)),
                        );
                    } else {
                        self.preempt(i);
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// Releases `running[j]`'s blocks and rewinds it for re-prefill; the
    /// sequence moves back to `waiting` at the end of the iteration. Its
    /// history (generated tokens) survives — the resume replays it.
    fn preempt(&mut self, j: usize) {
        let seq = &mut self.running[j];
        let released = std::mem::take(&mut seq.table).release(&mut self.mgr);
        debug_assert!(released.is_ok(), "preemption released a foreign block");
        seq.processed = 0;
        seq.fed = 0;
        seq.preemptions += 1;
        seq.preempted = true;
        self.preemptions_total += 1;
    }

    /// Fails every in-flight request and resets the pools — fault recovery
    /// for step-level MLX errors, which cannot be attributed to a single
    /// request.
    fn fail_all(&mut self, error: &str) {
        // Any scheduled-but-unread step dies with the pools it wrote to.
        self.inflight = None;
        let mgr = &mut self.mgr;
        for seq in self.running.iter_mut().chain(self.waiting.iter_mut()) {
            finish(
                seq,
                mgr,
                FinishKind::Error,
                None,
                Some((ErrorCause::Internal, format!("engine step failed: {error}"))),
            );
        }
        self.running.clear();
        self.waiting.clear();
        // Rebuild ownership state from scratch in case the fault interrupted
        // an append mid-flight; drop pool storage with possibly-poisoned
        // pending graphs.
        if let Ok(mgr) = BlockManager::new(self.config.num_blocks, self.config.block_size) {
            self.mgr = mgr;
        }
        self.kv.reset();
        let _ = memory::clear_cache();
    }
}

/// Appends `len` token slots to the sequence's table and derives the write
/// runs; `Ok(None)` means the pool is exhausted (the planner resolves that
/// via preemption, SPEC §6.1).
fn build_seq_step(
    seq: &mut Seq,
    mgr: &mut BlockManager,
    kv: &mut PagedKv,
    len: usize,
    sample: bool,
    s: &Stream,
) -> Result<Option<SeqStep>, EngineError> {
    let offset = seq.table.num_tokens();
    let plan = match seq.table.append_tokens(mgr, len) {
        Ok(plan) => plan,
        Err(BlockError::OutOfBlocks { .. }) => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    // No prefix sharing yet (radix cache is Phase 5), but honor the plan:
    // a detached tail must be copied before this step writes to it. (If a
    // later preemption drops this segment the copy is dead work into a
    // freed block — harmless, and unreachable until blocks are shared.)
    if let Some(cow) = plan.cow {
        kv.copy_block(cow, s)?;
    }
    let block_size = mgr.block_size();
    let mut writes = Vec::with_capacity(len / block_size + 2);
    let mut pos = offset;
    while pos < offset + len {
        let run = (block_size - pos % block_size).min(offset + len - pos);
        writes.push(WriteRun {
            block: seq.table.blocks()[pos / block_size],
            row_start: (pos % block_size) as i32,
            src_start: (pos - offset) as i32,
            len: run as i32,
        });
        pos += run;
    }
    Ok(Some(SeqStep {
        len: len as i32,
        offset: offset as i32,
        sample,
        blocks: seq.table.blocks().to_vec(),
        writes,
    }))
}
