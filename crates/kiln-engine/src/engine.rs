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
//! Out of scope until Phase 4 parts 3/4: preemption (pool exhaustion
//! mid-request fails the request instead), Drain, the async_eval decode
//! pipeline, and the step-overhead criterion bench. Cancellation via the
//! request's flag is honored between steps to preserve the worker's
//! existing `Cancel` behavior.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kiln_mlx::{Array, MlxError, Stream, eval, memory, ops};
use thiserror::Error;

use crate::block::{BlockError, BlockManager, BlockTable};
use crate::paged::{KvSpec, PagedKv, WriteRun};
use crate::sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
use crate::step::{SeqStep, StepBatch, StepModel};

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
    /// The request failed; see `error`.
    Error,
}

/// Terminal report for one request.
#[derive(Debug, Clone)]
pub struct FinishSummary {
    pub reason: FinishKind,
    /// Generated tokens, including a matched stop token.
    pub completion_tokens: u32,
    pub matched_stop_token: Option<u32>,
    pub error: Option<String>,
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
/// its `on_event` sink, so no id crosses this boundary until Cancel-by-id
/// lands in part 3).
pub struct EngineRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingOptions,
    pub penalties: PenaltyOptions,
    /// Recent-token window for the penalties (ignored when disabled).
    pub penalty_window: usize,
    pub stop_tokens: HashSet<u32>,
    pub cancel: Arc<AtomicBool>,
    pub on_event: EventSink,
}

/// Internal per-request state.
struct Seq {
    prompt: Vec<u32>,
    max_tokens: usize,
    stop_tokens: HashSet<u32>,
    sampler: Sampler,
    penalties: PenaltyOptions,
    penalty_window: usize,
    cancel: Arc<AtomicBool>,
    on_event: EventSink,
    table: BlockTable,
    /// Prompt tokens whose KV has been written (prefill progress).
    processed: usize,
    /// `Some(tok)` once the sequence is decoding: the next input token
    /// (initially the last prompt token, then each sampled token).
    next_input: Option<u32>,
    /// mlx-lm logits-processor history: last prompt token + generated.
    history: Vec<u32>,
    generated: u32,
    submitted_at: Instant,
    first_token_at: Option<Instant>,
    finished: bool,
}

impl Seq {
    fn decoding(&self) -> bool {
        self.next_input.is_some()
    }
}

/// Emits the terminal event and releases the sequence's blocks.
fn finish(
    seq: &mut Seq,
    mgr: &mut BlockManager,
    reason: FinishKind,
    matched_stop_token: Option<u32>,
    error: Option<String>,
) {
    if seq.finished {
        return;
    }
    seq.finished = true;
    let now = Instant::now();
    let first = seq.first_token_at.unwrap_or(now);
    let summary = FinishSummary {
        reason,
        completion_tokens: seq.generated,
        matched_stop_token,
        error,
        prefill_seconds: first.duration_since(seq.submitted_at).as_secs_f64(),
        decode_seconds: now.duration_since(first).as_secs_f64(),
    };
    let _ = (seq.on_event)(SeqEvent::Finished(summary));
    let released = std::mem::take(&mut seq.table).release(mgr);
    debug_assert!(released.is_ok(), "finish released a foreign block");
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
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
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
            cancel,
            on_event,
        } = request;
        let mut seq = Seq {
            prompt,
            max_tokens,
            stop_tokens,
            sampler: Sampler::new(sampling),
            penalties,
            penalty_window,
            cancel,
            on_event,
            table: BlockTable::new(),
            processed: 0,
            next_input: None,
            history: Vec::new(),
            generated: 0,
            submitted_at: Instant::now(),
            first_token_at: None,
            finished: false,
        };
        let error = if seq.prompt.is_empty() {
            Some("empty prompt".to_owned())
        } else if seq.max_tokens == 0 {
            Some("max_tokens must be >= 1".to_owned())
        } else {
            let needed = seq.prompt.len() + seq.max_tokens;
            let capacity = self.config.num_blocks * self.config.block_size;
            (needed > capacity).then(|| {
                format!(
                    "prompt ({}) + max_tokens ({}) exceeds the KV pool ({capacity} tokens)",
                    seq.prompt.len(),
                    seq.max_tokens
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
            seq.next_input = Some(seq.prompt[0]);
            seq.history.push(seq.prompt[0]);
        }
        self.waiting.push_back(seq);
    }

    /// One SPEC §6.2 iteration. On `Err` every in-flight request has been
    /// failed and the pools reset; the engine stays usable.
    pub fn step(&mut self) -> Result<(), EngineError> {
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

    /// Cancels flagged requests between steps (Phase-3 behavior; full
    /// `Cancel` semantics land in part 3).
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
    /// so admission pauses while a prefill is in flight.
    fn admit(&mut self) {
        loop {
            let mid_prefill = self.running.iter().any(|seq| !seq.decoding());
            let budget_left = self.running.len() < self.config.max_batch_tokens;
            if mid_prefill || !budget_left {
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

        // --- Build (SPEC §6.2 step 2). Decode group first in `running`
        // order; then at most one prefill chunk.
        let mut decode_ids: Vec<usize> = Vec::new();
        let mut prefill_id: Option<usize> = None;
        for (i, seq) in self.running.iter().enumerate() {
            if seq.decoding() {
                decode_ids.push(i);
            } else if prefill_id.is_none() {
                prefill_id = Some(i);
            }
        }

        let mut decode_batch = StepBatch {
            tokens: Vec::with_capacity(decode_ids.len()),
            seqs: Vec::with_capacity(decode_ids.len()),
        };
        let mut sampled_ids: Vec<usize> = Vec::with_capacity(decode_ids.len());
        for &i in &decode_ids {
            let seq = &mut self.running[i];
            // `decoding()` guarantees next_input is set.
            let Some(token) = seq.next_input else {
                continue;
            };
            match build_seq_step(seq, &mut self.mgr, &mut self.kv, 1, true, &self.stream)? {
                Some(step) => {
                    decode_batch.tokens.push(token);
                    decode_batch.seqs.push(step);
                    sampled_ids.push(i);
                }
                None => finish(
                    seq,
                    &mut self.mgr,
                    FinishKind::Error,
                    None,
                    Some("KV pool exhausted (preemption lands in Phase 4 part 3)".to_owned()),
                ),
            }
        }

        let mut prefill_batch: Option<StepBatch> = None;
        let mut prefill_seq: Option<usize> = None;
        if let Some(i) = prefill_id {
            let budget = self
                .config
                .max_batch_tokens
                .saturating_sub(decode_batch.tokens.len());
            let seq = &mut self.running[i];
            // Phase-3/mlx-lm chunk rule: prefill covers prompt[..n-1]; the
            // last prompt token is fed by the first sampled (decode) step.
            let remaining = seq.prompt.len() - 1 - seq.processed;
            let chunk = remaining.min(self.config.prefill_chunk).min(budget);
            if chunk > 0 {
                match build_seq_step(seq, &mut self.mgr, &mut self.kv, chunk, false, &self.stream)?
                {
                    Some(step) => {
                        let tokens = seq.prompt[seq.processed..seq.processed + chunk].to_vec();
                        prefill_batch = Some(StepBatch {
                            tokens,
                            seqs: vec![step],
                        });
                        prefill_seq = Some(i);
                    }
                    None => finish(
                        seq,
                        &mut self.mgr,
                        FinishKind::Error,
                        None,
                        Some("KV pool exhausted (preemption lands in Phase 4 part 3)".to_owned()),
                    ),
                }
            }
        }

        // --- Forward (SPEC §6.2 step 3). Prefill first so its KV writes
        // sit earliest on the pools' functional-update chain.
        if let Some(batch) = &prefill_batch {
            let logits = self.model.forward_step(batch, &mut self.kv, &self.stream)?;
            debug_assert!(logits.is_none(), "prefill chunks never sample");
        }
        let mut sampled: Vec<(usize, Array)> = Vec::with_capacity(sampled_ids.len());
        if !decode_batch.seqs.is_empty() {
            let logits = self
                .model
                .forward_step(&decode_batch, &mut self.kv, &self.stream)?
                .ok_or_else(|| MlxError {
                    message: "model returned no logits for a decode step".to_owned(),
                })?;
            let vocab = logits.dim(2);
            for (row, &i) in sampled_ids.iter().enumerate() {
                let seq = &mut self.running[i];
                let row = row as i32;
                let last = ops::slice(&logits, &[0, row, 0], &[1, row + 1, vocab], &self.stream)?;
                let mut last = ops::reshape(&last, &[1, vocab], &self.stream)?;
                if !seq.penalties.is_disabled() {
                    let window = seq.history.len().saturating_sub(seq.penalty_window);
                    last = apply_penalties(
                        &last,
                        &seq.history[window..],
                        seq.penalties,
                        &self.stream,
                    )?;
                }
                let logprobs = ops::subtract(
                    &last,
                    &ops::logsumexp(&last, true, &self.stream)?,
                    &self.stream,
                )?;
                let token = seq.sampler.sample(&logprobs, &self.stream)?;
                sampled.push((i, token));
            }
        }

        // --- Evaluate the step: sampled tokens + pool state together
        // (prefill-only steps still materialize their KV writes here).
        {
            let mut outputs: Vec<&Array> = sampled.iter().map(|(_, token)| token).collect();
            let state = self.kv.state();
            outputs.extend(state);
            eval(&outputs)?;
        }

        // --- Emit, check stops, release finished (SPEC §6.2 steps 3-4).
        for (i, token) in &sampled {
            let seq = &mut self.running[*i];
            let token = token.item_u32()?;
            if seq.first_token_at.is_none() {
                seq.first_token_at = Some(Instant::now());
            }
            seq.generated += 1;
            seq.history.push(token);
            seq.next_input = None;
            if seq.cancel.load(Ordering::Acquire) {
                finish(seq, &mut self.mgr, FinishKind::Cancelled, None, None);
            } else if seq.stop_tokens.contains(&token) {
                // Counted in usage but not emitted: stop text is excluded
                // from the stream by contract.
                finish(seq, &mut self.mgr, FinishKind::Stop, Some(token), None);
            } else if !(seq.on_event)(SeqEvent::Token(token)) {
                finish(seq, &mut self.mgr, FinishKind::Cancelled, None, None);
            } else if seq.generated as usize >= seq.max_tokens {
                finish(seq, &mut self.mgr, FinishKind::Length, None, None);
            } else {
                seq.next_input = Some(token);
            }
        }

        // --- Advance prefill progress; hand fully prefilled sequences
        // their first sampled step.
        if let (Some(i), Some(batch)) = (prefill_seq, &prefill_batch) {
            let seq = &mut self.running[i];
            seq.processed += batch.tokens.len();
            if seq.processed == seq.prompt.len() - 1 {
                let last = seq.prompt[seq.processed];
                seq.next_input = Some(last);
                seq.history.push(last);
            }
        }

        self.running.retain(|seq| !seq.finished);

        // --- SPEC §6.2 step 5: cache maintenance (Phase-3 cadence: after
        // every prefill chunk, else every 256 iterations).
        if prefill_batch.is_some() || self.iterations.is_multiple_of(MAINTENANCE_INTERVAL) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    /// Fails every in-flight request and resets the pools — fault recovery
    /// for step-level MLX errors, which cannot be attributed to a single
    /// request.
    fn fail_all(&mut self, error: &str) {
        let mgr = &mut self.mgr;
        for seq in self.running.iter_mut().chain(self.waiting.iter_mut()) {
            finish(
                seq,
                mgr,
                FinishKind::Error,
                None,
                Some(format!("engine step failed: {error}")),
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
/// runs; `Ok(None)` means the pool is exhausted (caller fails the request
/// until preemption lands).
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
    // a detached tail must be copied before this step writes to it.
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
