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
//! Speculative decoding (SPEC §6.5, scheduler-native): an eligible
//! request — greedy, penalties off, no grammar, batch within
//! `spec_max_batch` — swaps its 1-token decode segment for a
//! gamma+1-token verify segment: the drafter's proposed tokens ride the
//! same forward as the request's fed token, every position is sampled,
//! and the longest draft-agreeing prefix plus the target's own next
//! token commits. Rejected positions roll back by releasing their
//! blocks — bookkeeping only. Greedy invariance rests on the verify
//! forward staying in the M=1 kernel classes (ADR 0002): gamma+1 is
//! clamped under the calibrated deterministic width for the trunk
//! matmuls, each verify segment runs as its own forward group, and the
//! SDPA query length gamma+1 must stay inside the pinned MLX's
//! short-query (vector) kernel class. Row 0 then reproduces the plain
//! decode's bits, and each later row's input token equals the token the
//! plain path would have fed — so committed tokens are the plain path's
//! tokens by induction. That argument is held to measurement, not
//! trusted: the spec_decode suite reruns every golden fixture with
//! speculation on (self-draft and adversarial) and requires
//! token-identical output.
//!
//! The SDPA leg is held by the ADR 0005 envelope, enforced at drafter
//! attachment (worker) plus per-round here: the pinned MLX keeps the
//! fused vector kernel only while `query_len <= 8` and
//! `query_len * gqa_factor <= 32` (so `gamma` is clamped per model via
//! `AnyModel::speculative_gamma_bound`), and only while the key length
//! stays inside the 1-pass region ([`VERIFY_MAX_KEY_LEN`] — the 2-pass
//! variant's partition geometry varies with query count, key length,
//! and device class, so it is out of the certified envelope). Within
//! the 1-pass vector kernel, per-row bits are invariant to query count
//! and key length by kernel construction (fixed stride-32 key
//! assignment, index-ordered online softmax, fixed reduction tree,
//! identical used-key sets under the bottom-right causal mask) — the
//! source-verified argument recorded in ADR 0005, discovered when a
//! gamma=4 verify on gqa-factor-7 qwen2.5-0.5b silently left the fused
//! kernel and flipped a measured 1-fp16-ULP argmax race.
//!
//! Auto-disable heuristics (SPEC §6.5 / SPEC §12 Phase 8, part 3) sit
//! UNDER those hard cutoffs and only ever shrink a round — they are
//! throughput policy, never a correctness lever (turning speculation off
//! is output-neutral by the invariance above):
//! - **batch width** ([`crate::drafter::spec_gamma_at_width`]): full
//!   `gamma` single-stream, standing down linearly as the admitted batch
//!   approaches `spec_max_batch` (batching alone is saturating the GPU
//!   there, so speculation's extra verify rows and draft forwards pay
//!   progressively less), and off strictly above it — SPEC §6.5's
//!   auto-disable, with a ramp instead of a cliff.
//! - **acceptance rate** (`EngineConfig::spec_min_acceptance`): a
//!   request whose verified acceptance sits below the floor after
//!   [`crate::drafter::SPEC_ACCEPTANCE_WARMUP_PROPOSED`] judged tokens
//!   stands down permanently — its draft-side state is released and the
//!   request may re-enter the async_eval pipeline below, so a mis-paired
//!   draft costs a bounded warmup, not the whole generation.
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
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kiln_mlx::{Array, Dtype, MlxError, Stream, async_eval, eval, memory, ops};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::block::{BlockError, BlockId, BlockManager, BlockTable};
use crate::drafter::{
    DEFAULT_GAMMA, DEFAULT_SPEC_MAX_BATCH, DEFAULT_SPEC_MIN_ACCEPTANCE, Drafter, DrafterMemory,
    SPEC_ACCEPTANCE_WARMUP_PROPOSED, spec_gamma_at_width,
};
use crate::grammar::Grammar;
use crate::paged::{KvSpec, PagedKv, WriteRun};
use crate::paged_attn::PagedAttnInputs;
use crate::radix::{ChainHash, ROOT, RadixCache};
use crate::sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties, neg_inf_like};
use crate::ssd::{SlabGeometry, SsdStore};
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

/// Fine grid of the canonical prefill schedule's final super-chunk
/// (PROGRESS 2026-07-04 Option B): default for
/// `EngineConfig::prefill_fine_chunk`. 128 per the step-4 tuning curve:
/// +2.9% miss-path TTFT at the worst-case prompt size (vs +13.9% at 64)
/// while warm-turn recompute grows by at most 64 extra tokens.
pub const DEFAULT_PREFILL_FINE_CHUNK: usize = 128;

/// Canonical prefill schedule (SPEC §6.2 + Phase 5 Option B, PROGRESS
/// 2026-07-04): positions `0..limit` are prefilled in bulk chunks of
/// `prefill_chunk` split at absolute multiples of `prefill_chunk`, and
/// the final partial super-chunk is split at absolute multiples of
/// `fine` (plus the final sub-`fine` remainder). Returns the length of
/// the chunk starting at `at` (`at < limit`).
///
/// Every (offset, length) pair this yields for a given `limit` is a
/// function of the absolute grid alone — independent of where prefill
/// started — so a warm prefix-cache resume at any boundary recomputes
/// its segments in exactly the shapes a cold run of the same prompt
/// uses. That is the bit-exactness requirement for cache hits: KV bits
/// are chunk-shape dependent (kernel dispatch varies with query
/// length), so no position may ever be computed in a shape the
/// cache-off run would not use. `fine >= prefill_chunk` degenerates to
/// the pre-Phase-5 single-tail-chunk schedule (used to bench the miss
/// path cost of the fine grid).
pub fn canonical_prefill_len(at: usize, limit: usize, prefill_chunk: usize, fine: usize) -> usize {
    debug_assert!(at < limit, "no prefill remains at {at} of {limit}");
    let fine = fine.clamp(1, prefill_chunk);
    let tail_start = limit / prefill_chunk * prefill_chunk;
    if at < tail_start {
        // Bulk: run to the next absolute super-chunk boundary.
        prefill_chunk - at % prefill_chunk
    } else {
        // Final partial super-chunk: run to the next absolute fine
        // boundary, or to the end.
        (fine - at % fine).min(limit - at)
    }
}

/// Minimum trunk row count for ragged prefill tail pieces (ADR 0002): a
/// piece shorter than this that does not start on a `prefill_chunk`
/// boundary is computed with pad rows so every matmul/SDPA stays in the
/// same kernel class the mlx-lm reference used for those positions
/// (MLX's `get_qmv_batch_limit` tops out at 32 across GPU classes; the
/// SDPA vector-kernel bound is lower). Pieces starting ON a
/// `prefill_chunk` boundary are exactly the piece the reference also
/// computes at that size and are never padded. Pad rows are compute
/// filler only — never written to KV, sampled, or emitted (see
/// `StepBatch::pad_rows` and the `prefill_pad` tests).
pub const PREFILL_PAD_MIN_ROWS: usize = 32;

/// Default `EngineConfig::deterministic_decode_width` (ADR 0002 / B'):
/// deterministic rows (greedy or client-seeded) decode in sub-batches of
/// at most this many rows so every trunk matmul stays in the same kernel
/// class as M=1 — bit-identical to single-stream at any admitted batch
/// width. 4 is safe under the smallest dispatch threshold anywhere in
/// MLX's device table (6); workers RAISE it at load via
/// `AnyModel::calibrate_deterministic_width` (9 on the Phase 6 dev
/// machine), so this conservative default only governs engines built
/// without calibration.
pub const DEFAULT_DETERMINISTIC_DECODE_WIDTH: usize = 4;

/// Largest key length a speculative verify forward may reach (ADR 0005):
/// the pinned MLX routes fused-vector SDPA to its 2-pass variant at key
/// length >= 1024 on 'd'/'s'-class GPUs (>= 4096 with GQA elsewhere), and
/// the 2-pass partition geometry varies with query count and key length —
/// outside the certified kernel class. 1023 keeps every verify in the
/// 1-pass region on EVERY device class; requests simply stop speculating
/// (plain decode continues) once their context outgrows it. Refinable to
/// the per-device boundary via `mlx_device_info` (architecture string) —
/// recorded in ADR 0005, not implemented.
pub const VERIFY_MAX_KEY_LEN: usize = 1023;

/// Cache maintenance cadence in decode iterations (mirrors the Phase-3
/// loop's `clear_cache` every 256 steps).
const MAINTENANCE_INTERVAL: u64 = 256;

/// SSD captures per synchronous step — bounds the host-copy stall the
/// write-behind flush adds to any one iteration.
const FLUSH_BATCH: usize = 2;

/// SSD cold tier settings (SPEC §6.4).
#[derive(Debug, Clone)]
pub struct SsdParams {
    /// Slab directory (the worker derives it from `$KILN_CACHE_DIR` and
    /// its model fingerprint).
    pub dir: PathBuf,
    /// `ssd_cache_max_gb`, in bytes.
    pub max_bytes: u64,
    /// Model identity material folded into the slab fingerprint and every
    /// chain hash — the worker passes its `weights_fingerprint`, which
    /// already binds weights, architecture, and config dtype; the slab
    /// header additionally pins the KV geometry and element dtype.
    pub fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_batch_tokens: usize,
    pub prefill_chunk: usize,
    /// Fine grid of the final partial super-chunk (see
    /// [`canonical_prefill_len`], default 128); values `>= prefill_chunk`
    /// restore the single-tail-chunk schedule.
    pub prefill_fine_chunk: usize,
    /// Radix prefix cache (SPEC §6.3); on by default.
    pub prefix_cache: bool,
    /// SSD cold tier (SPEC §6.4); requires `prefix_cache`.
    pub ssd: Option<SsdParams>,
    /// Max rows per forward for DETERMINISTIC decode rows (ADR 0002 B');
    /// see [`DEFAULT_DETERMINISTIC_DECODE_WIDTH`]. Non-deterministic rows
    /// always ride one unrestricted full-width forward.
    pub deterministic_decode_width: usize,
    /// Custom block-table-aware paged-attention kernel for decode steps
    /// (SPEC §7.4 Phase 7). OFF by default: the gather path is the
    /// correctness reference; this flag stays opt-in until the kernel's
    /// parity and throughput bars are proven on the serving device.
    pub paged_attention_kernel: bool,
    /// Tokens proposed per speculation round (SPEC §6.5 `gamma`); only
    /// consulted when a drafter is attached. The effective per-round value
    /// is additionally clamped so a verify segment's `gamma + 1` rows stay
    /// within `deterministic_decode_width` (every speculating request is
    /// greedy, so its verify forward must keep the M=1 kernel classes —
    /// ADR 0002) and within the request's remaining token budget. 0
    /// disables speculation.
    pub gamma: usize,
    /// Speculation auto-disables when more than this many requests are
    /// admitted (SPEC §6.5 `spec_max_batch`): batching already saturates
    /// the GPU there. Below the threshold the per-round gamma already
    /// stands down linearly with the admitted width
    /// ([`crate::drafter::spec_gamma_at_width`]).
    pub spec_max_batch: usize,
    /// Auto-disable by acceptance (SPEC §12 Phase 8): the minimum
    /// verified acceptance rate (accepted / proposed) a request must
    /// hold, once
    /// [`crate::drafter::SPEC_ACCEPTANCE_WARMUP_PROPOSED`] proposed
    /// tokens have been judged, to keep speculating. Below it the
    /// request stands down for the rest of its life — plain decode
    /// continues, its draft-side state is released, and it may pipeline
    /// again. 0.0 disables the heuristic (tests that need sustained
    /// adversarial pressure use this). Advisory only: the ADR 0005
    /// envelope and [`VERIFY_MAX_KEY_LEN`] are hard cutoffs that bind
    /// whether or not any heuristic fires.
    pub spec_min_acceptance: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            num_blocks: DEFAULT_NUM_BLOCKS,
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            prefill_fine_chunk: DEFAULT_PREFILL_FINE_CHUNK,
            prefix_cache: true,
            ssd: None,
            deterministic_decode_width: DEFAULT_DETERMINISTIC_DECODE_WIDTH,
            paged_attention_kernel: false,
            gamma: DEFAULT_GAMMA,
            spec_max_batch: DEFAULT_SPEC_MAX_BATCH,
            spec_min_acceptance: DEFAULT_SPEC_MIN_ACCEPTANCE,
        }
    }
}

/// Cache observability snapshot (drives `WorkerStats`, SPEC §5).
#[derive(Debug, Default, Clone, Copy)]
pub struct PrefixCacheStats {
    pub enabled: bool,
    pub hits_total: u64,
    pub tokens_reused_total: u64,
    pub tokens_reused_ssd_total: u64,
    pub resident_blocks: u64,
    pub ssd_blocks_stored: u64,
    pub ssd_bytes_stored: u64,
    pub ssd_reads_total: u64,
    pub ssd_writes_total: u64,
    pub ssd_writes_failed_total: u64,
    pub ssd_fingerprint_rejects_total: u64,
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
    /// Prompt tokens served from the prefix cache instead of prefilled
    /// (proto `Finished.cached_prompt_tokens`).
    pub cached_prompt_tokens: u32,
    /// Draft tokens proposed for this request across its verify rounds
    /// (proto `Timings.spec_tokens_proposed`; 0 when speculation off).
    pub spec_tokens_proposed: u32,
    /// Proposed tokens the target accepted (longest agreeing prefix sums;
    /// bonus tokens are not counted — they are ordinary target samples).
    pub spec_tokens_accepted: u32,
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
    /// Proto `PrefixCacheHit` (SPEC §5, observability): `tokens` prompt
    /// tokens were reused from the radix cache at admission — skipped in
    /// prefill — `from_ssd` when any block was pulled from the cold tier.
    /// May repeat if a preemption resume re-matches.
    PrefixHit {
        tokens: u32,
        from_ssd: bool,
    },
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
    /// Structured-output constraint (SPEC §12 Phase 7): the compiled
    /// llguidance grammar whose allowed-token mask is applied to this
    /// request's logits before sampling. `None` = unconstrained — the
    /// decode step then adds zero ops (golden parity depends on this).
    pub grammar: Option<Grammar>,
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
    /// Structured-output constraint; advances by exactly the sampled
    /// tokens (committed in `settle_sampled`), so its state survives
    /// preemption untouched — replay never samples, so it never
    /// re-commits.
    grammar: Option<Grammar>,
    /// Mask computed at plan time for this iteration's sample, consumed
    /// by `sample_from_row`. Always recomputed at the next plan, so a
    /// mask left behind by a preempted (plan-dropped) step goes stale
    /// harmlessly: the grammar state it reflects has not changed.
    pending_mask: Option<Array>,
    priority: Priority,
    /// Submit-order sequence number; stable across preemption, so a
    /// resumed request keeps its seniority ("most-recently-admitted"
    /// victims are picked by the highest `arrival`).
    arrival: u64,
    /// Greedy or client-seeded (ADR 0002 B'): decode/replay rows ride
    /// sub-batched forwards bit-identical to single-stream, and finished
    /// blocks donate in full. Non-deterministic sequences decode at full
    /// width and donate only prefill-covered blocks.
    deterministic: bool,
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
    /// The prefix cache was consulted for the current (re-)admission;
    /// cleared by preemption so a resume re-matches.
    cache_checked: bool,
    /// Prompt tokens served from the prefix cache (max across resumes).
    cached_tokens: u32,
    /// The drafter holds state for this sequence (`begin` was called);
    /// finish/teardown must `release` it.
    draft_begun: bool,
    /// `history[1..]` tokens already fed through `Drafter::propose`'s
    /// `committed` argument (history[0] rides in the prompt). Never
    /// rewound: history only grows, even across preemption.
    draft_committed: usize,
    /// The drafter errored for this sequence — speculation permanently off
    /// for it (a draft fault must never fail a generation, SPEC §6.5).
    draft_dead: bool,
    /// Stood down by the acceptance heuristic
    /// (`EngineConfig::spec_min_acceptance`): the measured acceptance
    /// says the draft is costing more than it saves for this request, so
    /// speculation is off for the rest of its life. Survives preemption
    /// (the content that drafted badly is still the content).
    draft_standdown: bool,
    /// Per-request speculation counters (proto `Timings`).
    spec_proposed: u32,
    spec_accepted: u32,
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

    /// Whether this request may ride a speculative verify round at all
    /// (SPEC §6.5), independent of per-step conditions. Greedy only:
    /// argmax consumes no PRNG draws, so committing a variable number of
    /// tokens per round cannot desync a seeded key chain, and the ADR 0002
    /// deterministic-width clamp keeps the verify forward bit-identical to
    /// single-stream. Penalties and grammars need host-visible commits
    /// between positions, which a batched verify does not have.
    fn spec_eligible(&self) -> bool {
        !self.draft_dead
            && !self.draft_standdown
            && self.sampler.is_greedy()
            && self.penalties.is_disabled()
            && self.grammar.is_none()
    }

    /// Pool blocks this request needs to reach its next sampled token:
    /// the prefill still owed (a prefix-cache match already covers
    /// `processed` tokens with blocks the table holds) plus (after a
    /// preemption) the replay of everything generated so far. The §6.4
    /// admission projection — a request is admitted only once this fits
    /// in free blocks, so admission cannot thrash straight back into
    /// preemption.
    fn admission_blocks(&self, block_size: usize) -> usize {
        (self.prompt.len() - 1 - self.processed + self.history.len().max(1)).div_ceil(block_size)
    }
}

/// Prefix-cache state owned by the engine: the radix tree, the optional
/// SSD store, and the write-behind flush queue between them.
struct CacheState {
    radix: RadixCache,
    store: Option<SsdStore>,
    /// Donated nodes awaiting capture + flush (`(node, generation)`).
    flush_queue: VecDeque<(usize, u64)>,
    hits_total: u64,
    tokens_reused_total: u64,
    tokens_reused_ssd_total: u64,
}

impl CacheState {
    /// Evicts the LRU sole-owned leaf, freeing its pool block (SPEC §6.3
    /// eviction). A node without a confirmed SSD copy is pruned — the
    /// flush-else-lose policy keeps the engine from ever blocking on IO.
    fn evict_one(&mut self, mgr: &mut BlockManager) -> bool {
        let Some(node) = self.radix.evict_candidate(mgr) else {
            return false;
        };
        self.radix.evict(mgr, node).is_some()
    }

    /// Moves a finished sequence's settled full blocks into the radix
    /// tree and releases the rest.
    ///
    /// This is the pipeline-discarded-row invariant (see the module docs
    /// of `crate::radix`): exactly `processed + fed` rows are settled —
    /// forced by a token readback or a step-boundary eval — and a
    /// discarded speculative row sits at slot `processed + fed`, so
    /// donation stops at the last block fully inside the settled range.
    /// Whatever holds the in-flight row goes back to the free list, where
    /// only writers can reacquire it.
    fn donate(&mut self, mgr: &mut BlockManager, seq: &Seq, table: BlockTable) {
        let block_size = self.radix.block_size();
        // ADR 0002 B': non-deterministic sequences' decode rows were
        // computed at arbitrary trunk widths (ulp-off from M=1), so only
        // their prefill-written rows — canonical at any load — are
        // donatable; a deterministic request reusing more would silently
        // lose its bit guarantee. Deterministic sequences' decode rows
        // ARE single-stream bits and donate in full.
        let settled = if seq.deterministic {
            seq.processed + seq.fed
        } else {
            seq.processed
        };
        let blocks = table.into_blocks();
        let full = (settled / block_size).min(blocks.len());
        let token_at = |pos: usize| {
            if pos < seq.processed {
                seq.prompt[pos]
            } else {
                seq.history[pos - seq.processed]
            }
        };
        let mut cur = ROOT;
        let mut chunk = Vec::with_capacity(block_size);
        for (i, &block) in blocks.iter().enumerate() {
            if i > full || (i == full && settled.is_multiple_of(block_size)) {
                let released = mgr.release(block);
                debug_assert!(released.is_ok(), "finish released a foreign block");
                continue;
            }
            chunk.clear();
            for pos in i * block_size..((i + 1) * block_size).min(settled) {
                chunk.push(token_at(pos));
            }
            if i == full {
                // Settled sub-block tail: a pool-only partial leaf, so a
                // full-containment rerun recomputes nothing (see
                // `match_head_prefix`). Never flushed to SSD.
                self.donate_partial(mgr, cur, block, &chunk);
                continue;
            }
            cur = match self.radix.child(cur, &chunk) {
                Some(node) => {
                    if self.radix.node_block(node).is_some() {
                        // Shared prefix already resident (typically the
                        // donor's own cache hit): drop the duplicate ref.
                        let released = mgr.release(block);
                        debug_assert!(released.is_ok(), "finish released a foreign block");
                    } else {
                        // Known (SSD-backed) node re-warmed for free.
                        self.radix.set_resident(node, block);
                    }
                    self.radix.touch(node);
                    node
                }
                None => {
                    let node = self.radix.insert_child(cur, &chunk);
                    self.radix.set_resident(node, block);
                    if let Some(store) = &self.store
                        && store.contains(&self.radix.node_hash(node))
                    {
                        // A previous run already persisted this block.
                        self.radix.set_on_ssd(node);
                    }
                    // Partial siblings fully covered by this block are
                    // redundant now (their rows are a bit-identical
                    // prefix of it).
                    let covered: Vec<usize> = self
                        .radix
                        .partial_children(cur)
                        .iter()
                        .copied()
                        .filter(|&partial| chunk.starts_with(self.radix.node_tokens(partial)))
                        .collect();
                    for partial in covered {
                        self.radix.prune_subtree(mgr, partial);
                    }
                    node
                }
            };
            if self.store.is_some()
                && !self.radix.node_on_ssd(cur)
                && !self.radix.flush_pending(cur)
            {
                self.flush_queue
                    .push_back((cur, self.radix.node_generation(cur)));
            }
        }
    }

    /// Places one settled partial tail under `parent`: deduplicated
    /// against full children and other partials, upgrading a shorter
    /// partial in place when this donor settled further.
    fn donate_partial(
        &mut self,
        mgr: &mut BlockManager,
        parent: usize,
        block: BlockId,
        tokens: &[u32],
    ) {
        let release = |mgr: &mut BlockManager, block| {
            let released = mgr.release(block);
            debug_assert!(released.is_ok(), "finish released a foreign block");
        };
        // A resident full child already covers these rows bit-identically.
        let covered = self.radix.full_children(parent).any(|node| {
            self.radix.node_block(node).is_some()
                && self.radix.node_tokens(node).starts_with(tokens)
        });
        if covered {
            release(mgr, block);
            return;
        }
        let partials: Vec<usize> = self.radix.partial_children(parent).to_vec();
        for node in partials {
            if self.radix.node_tokens(node).starts_with(tokens) {
                // An equal-or-longer settled tail is already cached.
                release(mgr, block);
                return;
            }
            if tokens.starts_with(self.radix.node_tokens(node)) {
                // This donor settled further along the same tail.
                if let Some(old) = self.radix.upgrade_partial(node, tokens, block) {
                    release(mgr, old);
                }
                return;
            }
        }
        let node = self.radix.insert_partial_child(parent, tokens);
        self.radix.set_resident(node, block);
    }
}

/// Emits the terminal event and disposes of the sequence's blocks —
/// donated into the prefix cache for clean finishes, released otherwise.
fn finish(
    seq: &mut Seq,
    mgr: &mut BlockManager,
    cache: Option<&mut CacheState>,
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
        cached_prompt_tokens: seq.cached_tokens,
        spec_tokens_proposed: seq.spec_proposed,
        spec_tokens_accepted: seq.spec_accepted,
        prefill_seconds: first.duration_since(seq.submitted_at).as_secs_f64(),
        decode_seconds: now.duration_since(first).as_secs_f64(),
    };
    let _ = (seq.on_event)(SeqEvent::Finished(summary));
    let table = std::mem::take(&mut seq.table);
    match cache {
        // Error finishes never donate: an engine fault mid-step leaves no
        // settled-rows guarantee to stand on.
        Some(state) if reason != FinishKind::Error => state.donate(mgr, seq, table),
        _ => {
            let released = table.release(mgr);
            debug_assert!(released.is_ok(), "finish released a foreign block");
        }
    }
}

/// One planned decode/replay segment (index into `running` + its step).
struct DecodePlan {
    seq: usize,
    sample: bool,
    step: SeqStep,
    /// `Some(proposal)` marks a speculative verify segment (SPEC §6.5):
    /// the step feeds `history[fed]` followed by the proposed tokens and
    /// samples every position. Always runs as its own forward group.
    draft: Option<Vec<u32>>,
}

/// Speculative-decoding observability (SPEC §6.5; proto `WorkerStats`
/// `spec_tokens_*_total`). Rollback cost is recorded, not assumed: the
/// nanos counter times exactly the block-release/table-truncate work a
/// rejection triggers, so tests can assert it stays flat as sequences
/// grow (the O(1) claim of the paged design).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpecStats {
    /// Verify rounds run (each = one draft proposal scored by the target).
    pub rounds_total: u64,
    /// Draft tokens proposed across all rounds.
    pub proposed_total: u64,
    /// Proposed tokens accepted (longest agreeing prefix sums).
    pub accepted_total: u64,
    /// Rounds that rejected at least one proposed token.
    pub rollback_rounds_total: u64,
    /// Rejected (rolled-back) draft positions.
    pub rollback_tokens_total: u64,
    /// Wall-clock nanoseconds spent inside rollback bookkeeping.
    pub rollback_nanos_total: u64,
    /// Drafter faults (speculation disabled for the affected request).
    pub draft_errors_total: u64,
    /// Requests stood down by the acceptance heuristic
    /// (`EngineConfig::spec_min_acceptance`) — sustained low verified
    /// acceptance, speculation off for the rest of the request.
    pub standdowns_total: u64,
}

/// Splits decode entries into the ADR 0002 B' forward groups:
/// deterministic entries in chunks of at most `width` rows (each such
/// forward reproduces M=1 bits — qmv row M-invariance below the
/// calibrated dispatch threshold), then ONE unrestricted group carrying
/// every non-deterministic entry. Order within each class is preserved.
fn partition_decode_plans<T>(
    plans: Vec<T>,
    width: usize,
    mut deterministic: impl FnMut(&T) -> bool,
) -> Vec<Vec<T>> {
    let (det, nondet): (Vec<T>, Vec<T>) = plans.into_iter().partition(|p| deterministic(p));
    let mut groups: Vec<Vec<T>> = Vec::new();
    let mut det = det.into_iter();
    loop {
        let chunk: Vec<T> = det.by_ref().take(width).collect();
        if chunk.is_empty() {
            break;
        }
        groups.push(chunk);
    }
    if !nondet.is_empty() {
        groups.push(nondet);
    }
    groups
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
/// step-3 tail: penalties over the recent window, grammar mask,
/// logprob-normalize, sample). Shared by the synchronous and pipelined
/// paths; the pipelined path only ever calls it with penalties disabled
/// and no grammar, so both build the identical op graph.
///
/// The grammar mask (planned in `run_iteration`, where a grammar fault
/// can finish the sequence in-band) comes last so no other processor can
/// resurrect a banned token; disallowed logits go to `-inf` before the
/// normalization, renormalizing the distribution over the allowed set.
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
    if let Some(mask) = seq.pending_mask.take() {
        // [1, n_vocab] bool, n_vocab = the worker's configured vocab
        // (GrammarEnv::load); both sides are sized from config.json's
        // vocab_size, so the shapes agree by construction.
        debug_assert_eq!(mask.dim(1), vocab, "grammar mask width != logits vocab");
        last = ops::where_cond(&mask, &last, &neg_inf_like(&last, s)?, s)?;
    }
    let logprobs = ops::subtract(&last, &ops::logsumexp(&last, true, s)?, s)?;
    Ok(seq.sampler.sample(&logprobs, s)?)
}

/// Applies one read-back sampled token to its sequence: cursor advance,
/// then the cancel/stop/emission/length checks (SPEC §6.2 steps 3-4).
/// Identical on the synchronous and pipelined paths — only *when* it
/// runs differs (immediately vs at the next `step()` call).
fn settle_sampled(
    seq: &mut Seq,
    mgr: &mut BlockManager,
    cache: Option<&mut CacheState>,
    token: u32,
) {
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
        finish(seq, mgr, cache, FinishKind::Cancelled, None, None);
        return;
    }
    if seq.stop_tokens.contains(&token) {
        // Counted in usage but not emitted: stop text is excluded from
        // the stream by contract. The grammar (if any) never sees the
        // stop token — its state only matters for future masks.
        finish(seq, mgr, cache, FinishKind::Stop, Some(token), None);
        return;
    }
    if !(seq.on_event)(SeqEvent::Token(token)) {
        finish(seq, mgr, cache, FinishKind::Cancelled, None, None);
        return;
    }
    // Advance the grammar by the emitted token; completion is a Stop
    // (the constrained text is done even if the model never sampled EOS,
    // e.g. under ignore_eos). The token was sampled under this grammar's
    // mask, so a rejection here is an engine-internal fault.
    let mut grammar_done = false;
    if let Some(grammar) = seq.grammar.as_mut() {
        match grammar.commit(token) {
            Ok(done) => grammar_done = done,
            Err(err) => {
                finish(
                    seq,
                    mgr,
                    cache,
                    FinishKind::Error,
                    None,
                    Some((ErrorCause::Internal, err.to_string())),
                );
                return;
            }
        }
    }
    if grammar_done {
        finish(seq, mgr, cache, FinishKind::Stop, None, None);
    } else if seq.generated as usize >= seq.max_tokens {
        finish(seq, mgr, cache, FinishKind::Length, None, None);
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
    /// Radix prefix cache + SSD tier (SPEC §6.3/§6.4), when enabled.
    cache: Option<CacheState>,
    /// Speculative-decoding proposer (SPEC §6.5), when configured. Owned
    /// here because spec decode is scheduler-native — the draft/verify
    /// loop runs inside the batch step — and its memory report feeds the
    /// worker heartbeat: the drafter's weights and KV pool are part of
    /// this worker's budget envelope (SPEC §2.3).
    drafter: Option<Box<dyn Drafter>>,
    /// Speculation counters (SPEC §6.5 acceptance-rate metrics).
    spec: SpecStats,
    /// The target's logits width, learned from the first sampled forward.
    /// Proposed draft ids are validated against it before they are ever
    /// embedded — a defensive bound; the worker's tokenizer-compatibility
    /// check at drafter attachment is the real gate. Speculation waits
    /// until this is known (the second decode step at the earliest).
    logits_vocab: Option<i32>,
    /// Why the SSD tier is off despite being configured, if it failed to
    /// open (silent-skip policy; the worker logs it once).
    ssd_error: Option<String>,
    next_arrival: u64,
    preemptions_total: u64,
    pipelined_total: u64,
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
        let mut kv = PagedKv::new(KvSpec {
            layers: dims.layers,
            kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            num_blocks: config.num_blocks,
            block_size: config.block_size,
        });
        if config.paged_attention_kernel {
            kv.enable_attention_kernel()?;
        }
        let mut ssd_error = None;
        let cache = config.prefix_cache.then(|| {
            let material = config
                .ssd
                .as_ref()
                .map(|params| params.fingerprint.as_str())
                .unwrap_or("");
            let mut hasher = Sha256::new();
            hasher.update(b"kiln-prefix-cache-v1;");
            hasher.update(material.as_bytes());
            let seed: ChainHash = hasher.finalize().into();
            let store = config.ssd.as_ref().and_then(|params| {
                let geometry = SlabGeometry {
                    layers: dims.layers as u32,
                    kv_heads: dims.kv_heads as u32,
                    head_dim: dims.head_dim as u32,
                    block_size: config.block_size as u32,
                };
                match SsdStore::open(&params.dir, seed, geometry, params.max_bytes) {
                    Ok(store) => Some(store),
                    Err(err) => {
                        // SPEC §6.4 failure policy: the tier silently
                        // degrades; requests are never affected.
                        ssd_error = Some(format!(
                            "ssd tier disabled: {err} ({})",
                            params.dir.display()
                        ));
                        None
                    }
                }
            });
            CacheState {
                radix: RadixCache::new(config.block_size, seed),
                store,
                flush_queue: VecDeque::new(),
                hits_total: 0,
                tokens_reused_total: 0,
                tokens_reused_ssd_total: 0,
            }
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
            cache,
            drafter: None,
            spec: SpecStats::default(),
            logits_vocab: None,
            ssd_error,
            next_arrival: 0,
            preemptions_total: 0,
            pipelined_total: 0,
            iterations: 0,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Attaches the speculative-decoding proposer (SPEC §6.5): eligible
    /// greedy requests now decode through draft/verify rounds inside the
    /// batch step. Preconditions (enforced by the worker, not here):
    /// - the drafter proposes token ids in the target's tokenizer —
    ///   attach only after `kiln_models::check_draft_compat` (or
    ///   equivalent) has passed; a mismatched pair must be rejected
    ///   loudly at load, never attached;
    /// - `config.gamma` has been clamped to the target's ADR 0005
    ///   envelope (`AnyModel::speculative_gamma_bound`); a target with no
    ///   envelope (`None`) must be rejected loudly, never attached with
    ///   speculation silently inert.
    pub fn set_drafter(&mut self, drafter: Box<dyn Drafter>) {
        self.drafter = Some(drafter);
    }

    /// Speculation counters (SPEC §6.5 acceptance-rate metrics); zeros
    /// when no drafter is attached.
    pub fn spec_stats(&self) -> SpecStats {
        self.spec
    }

    /// The attached drafter's memory footprint (SPEC §2.3 accounting),
    /// `None` when speculation is not configured.
    pub fn drafter_memory(&self) -> Option<DrafterMemory> {
        self.drafter.as_ref().map(|drafter| drafter.memory())
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

    /// Engine iterations since start (proto `WorkerStats.engine_steps_total`).
    pub fn steps(&self) -> u64 {
        self.iterations
    }

    /// Iterations that ran on the async_eval pipelined decode path.
    pub fn pipelined_steps(&self) -> u64 {
        self.pipelined_total
    }

    /// Why the configured SSD tier is inactive, if it failed to open.
    pub fn ssd_error(&self) -> Option<&str> {
        self.ssd_error.as_deref()
    }

    /// Prefix-cache observability snapshot (SPEC §5 `WorkerStats`).
    pub fn cache_stats(&self) -> PrefixCacheStats {
        let Some(state) = &self.cache else {
            return PrefixCacheStats::default();
        };
        let mut stats = PrefixCacheStats {
            enabled: true,
            hits_total: state.hits_total,
            tokens_reused_total: state.tokens_reused_total,
            tokens_reused_ssd_total: state.tokens_reused_ssd_total,
            resident_blocks: state.radix.resident_blocks() as u64,
            ..PrefixCacheStats::default()
        };
        if let Some(store) = &state.store {
            let counters = store.counters();
            stats.ssd_blocks_stored = store.blocks_stored();
            stats.ssd_bytes_stored = store.bytes_stored();
            stats.ssd_reads_total = counters.reads_total;
            stats.ssd_writes_total = counters.writes_total;
            stats.ssd_writes_failed_total = counters.writes_failed_total;
            stats.ssd_fingerprint_rejects_total = counters.fingerprint_rejects_total;
        }
        stats
    }

    /// Whether donated blocks still await capture/flush to the SSD tier —
    /// the worker keeps ticking an idle engine while this is true.
    pub fn has_pending_cache_io(&self) -> bool {
        self.cache
            .as_ref()
            .is_some_and(|state| state.store.is_some() && !state.flush_queue.is_empty())
    }

    /// Drains the whole flush queue synchronously and waits for the writer
    /// (worker drain/shutdown and tests; the steady-state path flushes
    /// write-behind in small batches instead).
    pub fn flush_prefix_cache(&mut self) {
        let Self {
            cache, kv, stream, ..
        } = self;
        let Some(state) = cache else { return };
        while !state.flush_queue.is_empty() {
            if flush_entries(state, kv, stream, usize::MAX) == 0 {
                break;
            }
        }
        if let Some(store) = &mut state.store {
            for ack in store.sync() {
                apply_ack(&mut state.radix, ack);
            }
        }
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

    /// Pool-block gauges `(in use, free)` — proto `WorkerStats`
    /// `kv_blocks_allocated` / `kv_blocks_free`.
    pub fn kv_blocks(&self) -> (u64, u64) {
        let free = self.mgr.num_free() as u64;
        (self.mgr.capacity() as u64 - free, free)
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
            grammar,
            priority,
            cancel,
            mut on_event,
        } = request;
        // Key creation is a host-side mlx allocation; a failure (never seen
        // in practice) is the proto's in-band error. No engine resources
        // exist for the request yet, so the summary is emitted directly.
        let sampler = match Sampler::new(sampling) {
            Ok(sampler) => sampler,
            Err(err) => {
                let _ = on_event(SeqEvent::Finished(FinishSummary {
                    reason: FinishKind::Error,
                    completion_tokens: 0,
                    matched_stop_token: None,
                    error: Some(format!("sampler key init failed: {err}")),
                    error_cause: Some(ErrorCause::Internal),
                    preemptions: 0,
                    cached_prompt_tokens: 0,
                    spec_tokens_proposed: 0,
                    spec_tokens_accepted: 0,
                    prefill_seconds: 0.0,
                    decode_seconds: 0.0,
                }));
                return;
            }
        };
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        let mut seq = Seq {
            prompt,
            max_tokens,
            stop_tokens,
            deterministic: sampling.deterministic(),
            sampler,
            penalties,
            penalty_window,
            grammar,
            pending_mask: None,
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
            cache_checked: false,
            cached_tokens: 0,
            draft_begun: false,
            draft_committed: 0,
            draft_dead: false,
            draft_standdown: false,
            spec_proposed: 0,
            spec_accepted: 0,
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
                None,
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
                // A new step is in flight; the turn is complete. Flush
                // captures would eval the pools and stall the overlap, so
                // only the (cheap) write acks are applied.
                Ok(true) => {
                    self.cache_io_tick(0);
                    return Ok(());
                }
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
            // Idle (or everything waiting on capacity): make write-behind
            // progress so persistence never depends on new traffic.
            self.cache_io_tick(FLUSH_BATCH);
            return Ok(());
        }
        let result = match self.run_iteration() {
            Ok(()) => Ok(()),
            Err(err) => {
                self.fail_all(&err.to_string());
                Err(err)
            }
        };
        if result.is_ok() && self.inflight.is_none() {
            // SPEC §6.2 step 5 (cache maintenance): a bounded number of
            // block captures per synchronous step, plus writer acks.
            self.cache_io_tick(FLUSH_BATCH);
        }
        result
    }

    /// Applies SSD writer acks and captures up to `captures` queued blocks
    /// for the write-behind flush. Captures force pool evaluation, so the
    /// pipelined turn passes 0.
    fn cache_io_tick(&mut self, captures: usize) {
        let Self {
            cache, kv, stream, ..
        } = self;
        let Some(state) = cache else { return };
        let Some(store) = &mut state.store else {
            return;
        };
        for ack in store.drain_acks() {
            apply_ack(&mut state.radix, ack);
        }
        if captures > 0 {
            flush_entries(state, kv, stream, captures);
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
        if !next.rows.is_empty() {
            self.pipelined_total += 1;
            self.inflight = Some(next);
        }
        Ok(true)
    }

    /// The async_eval pipeline may only span steady-state decode: every
    /// running sequence sampling, penalties off (their windows need the
    /// previous token host-side) and no grammar (its mask for step N+1
    /// needs step N's token committed host-side), nothing waiting
    /// (admission and prefill want fresh pool state), no cancel pending
    /// (the sweep runs on the synchronous path), and no request that
    /// speculation should be serving — a verify round needs its tokens
    /// host-side every step, so it lives on the synchronous path and the
    /// pipeline must yield to it. Capacity is checked exactly in
    /// `build_pipelined`.
    fn pipeline_ok(&self) -> bool {
        !self.spec_would_engage()
            && self.waiting.is_empty()
            && self.running.iter().all(|seq| {
                seq.decoding()
                    && seq.penalties.is_disabled()
                    && seq.grammar.is_none()
                    && !seq.cancel.load(Ordering::Acquire)
            })
    }

    /// Whether the next synchronous step would (or could, once the logits
    /// width is known) run a verify round for some running request — the
    /// pipeline yields to speculation (SPEC §6.5 is scheduler-native, not
    /// an alternate engine).
    fn spec_would_engage(&self) -> bool {
        self.drafter.is_some()
            && self.config.deterministic_decode_width > 1
            // The width ramp (SPEC §6.5 stand-down) in lockstep with
            // `collect_proposal`: 0 = no round at this admitted width.
            && spec_gamma_at_width(
                self.config.gamma,
                self.config.spec_max_batch,
                self.running.len(),
            ) > 0
            && self.running.iter().any(|seq| {
                seq.spec_eligible()
                    && seq.max_tokens.saturating_sub(seq.generated as usize) >= 2
                    // A gamma=1 verify (2 rows) must still fit the ADR 0005
                    // key-length envelope; past it the request decodes
                    // plainly and may pipeline again.
                    && seq.processed + seq.fed + 2 <= VERIFY_MAX_KEY_LEN
            })
    }

    /// Runs one drafter round for `running[i]` (SPEC §6.5): lazily `begin`s
    /// the sequence, feeds the tokens committed since the last round, and
    /// returns the validated proposal. `None` = no speculation this step —
    /// per-step conditions failed, the drafter declined, or it faulted (a
    /// draft fault disables speculation for the request and is otherwise
    /// invisible: the step decodes normally).
    fn collect_proposal(&mut self, i: usize) -> Option<Vec<u32>> {
        let Some(vocab) = self.logits_vocab else {
            return None; // first sampled forward not seen yet
        };
        // SPEC §6.5 batch-width stand-down: full gamma single-stream,
        // ramping down as the admitted batch approaches spec_max_batch,
        // 0 (off) strictly above it.
        let width_gamma = spec_gamma_at_width(
            self.config.gamma,
            self.config.spec_max_batch,
            self.running.len(),
        );
        if width_gamma == 0 {
            return None;
        }
        {
            let seq = &self.running[i];
            if !seq.spec_eligible() {
                return None;
            }
        }
        // Hard cutoffs UNDER which the heuristic ramp operates: the verify
        // segment's gamma+1 rows stay within the ADR 0002 deterministic
        // width (speculating requests are greedy — their verify forward
        // must reproduce M=1 bits), within the request's remaining budget
        // (proposing past max_tokens is waste), and within the ADR 0005
        // 1-pass key-length envelope (the verify's key length is
        // offset + gamma + 1; past [`VERIFY_MAX_KEY_LEN`] the request
        // stops speculating and decodes plainly). The per-model
        // fused-vector clamp (gamma+1 <= min(8, 32/gqa)) is baked into
        // `config.gamma` by the worker at drafter attachment. These bind
        // regardless of what any heuristic decides — heuristics shrink
        // rounds, never widen the envelope.
        let seq = &self.running[i];
        let remaining = seq.max_tokens.saturating_sub(seq.generated as usize);
        let offset = seq.processed + seq.fed;
        let gamma = width_gamma
            .min(self.config.deterministic_decode_width.saturating_sub(1))
            .min(remaining.saturating_sub(1))
            .min(VERIFY_MAX_KEY_LEN.saturating_sub(offset + 1));
        if gamma == 0 {
            return None;
        }
        let Self {
            drafter,
            running,
            stream,
            spec,
            ..
        } = self;
        let drafter = drafter.as_mut()?;
        let seq = &mut running[i];
        let mut round = || -> Result<Vec<u32>, crate::drafter::DraftError> {
            if !seq.draft_begun {
                drafter.begin(seq.arrival, &seq.prompt, stream)?;
                seq.draft_begun = true;
                seq.draft_committed = 0;
            }
            // history[0] is the last prompt token (already in the prompt);
            // everything after it that the drafter has not seen is newly
            // committed. history never shrinks, so this is monotonic even
            // across preemption.
            let committed = &seq.history[1 + seq.draft_committed..];
            let proposal = drafter.propose(seq.arrival, committed, gamma, stream)?;
            seq.draft_committed = seq.history.len() - 1;
            Ok(proposal)
        };
        match round() {
            Ok(mut proposal) => {
                // Defensive bounds: ids past the target's logits width can
                // never verify and must never reach the embedding gather;
                // over-long proposals lose their tail.
                if let Some(bad) = proposal.iter().position(|&t| (t as i64) >= vocab as i64) {
                    proposal.truncate(bad);
                }
                proposal.truncate(gamma);
                (!proposal.is_empty()).then_some(proposal)
            }
            Err(_) => {
                spec.draft_errors_total += 1;
                seq.draft_dead = true;
                if seq.draft_begun {
                    drafter.release(seq.arrival);
                    seq.draft_begun = false;
                }
                None
            }
        }
    }

    /// The acceptance auto-disable (SPEC §12 Phase 8, part 3), judged at
    /// round settle where acceptance is a token-value fact: once
    /// [`SPEC_ACCEPTANCE_WARMUP_PROPOSED`] proposed tokens have been
    /// verified for `running[i]`, a verified acceptance rate below
    /// `config.spec_min_acceptance` stands the request down for good —
    /// the draft is costing more than it saves. Its draft-side state is
    /// released immediately (KV blocks back to the draft pool), and with
    /// speculation no longer asking for the synchronous path the request
    /// may re-enter the async_eval pipeline.
    fn maybe_stand_down(&mut self, i: usize) {
        let Self {
            drafter,
            running,
            spec,
            config,
            ..
        } = self;
        let seq = &mut running[i];
        if config.spec_min_acceptance <= 0.0
            || seq.finished
            || seq.draft_standdown
            || seq.draft_dead
            || seq.spec_proposed < SPEC_ACCEPTANCE_WARMUP_PROPOSED
            || f64::from(seq.spec_accepted)
                >= config.spec_min_acceptance * f64::from(seq.spec_proposed)
        {
            return;
        }
        seq.draft_standdown = true;
        spec.standdowns_total += 1;
        if seq.draft_begun
            && let Some(drafter) = drafter.as_mut()
        {
            drafter.release(seq.arrival);
            seq.draft_begun = false;
        }
    }

    /// Releases drafter state for finished sequences (safe on unknown
    /// sequences — `Drafter::release` is a no-op there). Runs before every
    /// `retain(!finished)` sweep so no draft-side state outlives its
    /// sequence.
    fn release_finished_drafts(&mut self) {
        let Some(drafter) = &mut self.drafter else {
            return;
        };
        for seq in self.waiting.iter().chain(self.running.iter()) {
            if seq.finished && seq.draft_begun {
                drafter.release(seq.arrival);
            }
        }
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

        // Pass 2: append the slots and build the step, partitioned into
        // the same ADR 0002 B' forward groups as the synchronous path
        // (deterministic rows in <= width chunks, non-deterministic rows
        // full-width). Inputs are the pending `[1]` arrays reshaped and
        // concatenated to `[1, n]` per group.
        self.iterations += 1;
        let total = included.len();
        let groups = partition_decode_plans(
            included,
            self.config.deterministic_decode_width.max(1),
            |&(i, _)| self.running[i].deterministic,
        );
        let mut rows = Vec::with_capacity(total);
        for group in groups {
            let mut seqs = Vec::with_capacity(group.len());
            let mut inputs = Vec::with_capacity(group.len());
            for &(i, pending) in &group {
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
                pad_rows: 0,
            };
            let logits = self
                .model
                .forward_step(&batch, &mut self.kv, &self.stream)?
                .ok_or_else(|| MlxError {
                    message: "pipelined decode returned no logits".to_owned(),
                })?;
            for (row, &(i, _)) in group.iter().enumerate() {
                let seq = &mut self.running[i];
                debug_assert!(
                    seq.penalties.is_disabled() && seq.grammar.is_none(),
                    "pipeline_ok admitted penalties or a grammar"
                );
                let token = sample_from_row(seq, &logits, row as i32, &self.stream)?;
                rows.push((seq.arrival, token));
            }
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
            settle_sampled(
                &mut self.running[i],
                &mut self.mgr,
                self.cache.as_mut(),
                token,
            );
        }
        self.release_finished_drafts();
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
        let Self {
            waiting,
            running,
            mgr,
            cache,
            ..
        } = self;
        for seq in waiting.iter_mut().chain(running.iter_mut()) {
            if seq.cancel.load(Ordering::Acquire) {
                finish(seq, mgr, cache.as_mut(), FinishKind::Cancelled, None, None);
            }
        }
        self.release_finished_drafts();
        self.waiting.retain(|seq| !seq.finished);
        self.running.retain(|seq| !seq.finished);
    }

    /// SPEC §6.2 step 1: admit while the token budget allows. One request
    /// prefills at a time (each step carries at most one prefill chunk),
    /// so admission pauses while a prefill is in flight. The head of the
    /// queue gets its prefix-cache match here (SPEC §6.3 "on admit"), then
    /// waits until its remaining admission projection fits in free blocks
    /// (SPEC §6.4) — evicting cache-only blocks to make room before giving
    /// up.
    fn admit(&mut self) {
        loop {
            let mid_prefill = self.running.iter().any(|seq| !seq.decoding());
            let budget_left = self.running.len() < self.config.max_batch_tokens;
            if mid_prefill || !budget_left {
                return;
            }
            self.match_head_prefix();
            let Some(head) = self.waiting.front() else {
                return;
            };
            let needed = head.admission_blocks(self.mgr.block_size());
            while self.mgr.num_free() < needed {
                let evicted = self
                    .cache
                    .as_mut()
                    .is_some_and(|state| state.evict_one(&mut self.mgr));
                if !evicted {
                    return;
                }
            }
            match self.waiting.pop_front() {
                Some(seq) => self.running.push(seq),
                None => return,
            }
        }
    }

    /// Longest-prefix match for the head of the waiting queue (SPEC §6.3):
    /// reuse cached blocks (refcount++), skip those tokens in prefill.
    /// Non-resident nodes are pulled from the SSD tier when possible; the
    /// walk also discovers persisted blocks hash-first, which is how a
    /// restarted worker warms up lazily (SPEC §6.4). Idempotent per
    /// (re-)admission via `cache_checked`.
    ///
    /// Determinism rule (CLAUDE.md: prefix caching must not change greedy
    /// outputs — KV bits are chunk-shape dependent, so no position may be
    /// recomputed in a shape the cache-off run would not use):
    /// - **containment**: when the cache covers every prefill position
    ///   (`0..prompt.len()-2`, the resubmit/rerun case), serve all of it —
    ///   including a partial tail block, whose first write resolves via
    ///   copy-on-write — and recompute nothing;
    /// - otherwise **trim to canonical `prefill_chunk` boundaries**, so
    ///   every recomputed chunk has exactly the cold run's shape (causal
    ///   attention makes per-row bits independent of later tokens within
    ///   a chunk).
    fn match_head_prefix(&mut self) {
        let Self {
            waiting,
            mgr,
            kv,
            cache,
            stream,
            config,
            ..
        } = self;
        let Some(state) = cache else { return };
        let Some(seq) = waiting.front_mut() else {
            return;
        };
        if seq.cache_checked {
            return;
        }
        seq.cache_checked = true;
        if seq.processed != 0 || !seq.table.blocks().is_empty() {
            return;
        }
        let block_size = mgr.block_size();
        // The last prompt token is always computed (its forward yields the
        // first sampled logits).
        let limit = seq.prompt.len() - 1;
        let mut from_ssd = false;

        // -- 1) Full-block walk: resolve (or SSD-load) nodes; no table
        // mutation yet, since the serveable length is decided below.
        let mut cur = ROOT;
        let mut walked: Vec<BlockId> = Vec::new();
        while (walked.len() + 1) * block_size <= limit {
            let at = walked.len() * block_size;
            let chunk = &seq.prompt[at..at + block_size];
            let Some(node) = resolve_chunk(state, mgr, kv, stream, cur, chunk, &mut from_ssd)
            else {
                break;
            };
            let Some(block) = state.radix.node_block(node) else {
                break;
            };
            state.radix.touch(node);
            walked.push(block);
            cur = node;
        }
        let d_full = walked.len() * block_size;

        // -- 2) Containment probe: can a cached block cover the final
        // `r` sub-block positions? (A full node's first rows, or a donated
        // partial tail.)
        let r = limit - d_full;
        let mut tail: Option<BlockId> = None;
        if r > 0 && r < block_size {
            let want = &seq.prompt[d_full..limit];
            let mut candidate = state
                .radix
                .full_children(cur)
                .find(|&node| state.radix.node_tokens(node).starts_with(want));
            if candidate.is_none() {
                candidate = state
                    .radix
                    .partial_children(cur)
                    .iter()
                    .copied()
                    .find(|&node| state.radix.node_tokens(node).starts_with(want));
            }
            if candidate.is_none()
                && seq.prompt.len() >= d_full + block_size
                && let Some(store) = state.store.as_ref()
            {
                // Hash-first discovery needs a full chunk's tokens, which
                // the request has only when exactly one position is left
                // to compute.
                let chunk = &seq.prompt[d_full..d_full + block_size];
                let hash = state.radix.chain_hash_of(cur, chunk);
                if store.contains(&hash) {
                    let node = state.radix.insert_child(cur, chunk);
                    state.radix.set_on_ssd(node);
                    candidate = Some(node);
                }
            }
            if let Some(node) = candidate {
                if state.radix.node_block(node).is_none()
                    && !state.radix.node_is_partial(node)
                    && state.radix.node_on_ssd(node)
                {
                    let tokens = state.radix.node_tokens(node).to_vec();
                    if load_from_ssd(state, mgr, kv, stream, node, &tokens) {
                        from_ssd = true;
                    }
                }
                if let Some(block) = state.radix.node_block(node) {
                    state.radix.touch(node);
                    tail = Some(block);
                }
            }
        }

        // -- 3) Serve: containment in full, otherwise trimmed to the
        // nearest *resumable* boundary of the canonical schedule (see
        // `canonical_prefill_len`): absolute `prefill_chunk` multiples in
        // the bulk region, absolute `prefill_fine_chunk` multiples inside
        // the final partial super-chunk. Resuming there recomputes the
        // remainder in exactly the cold schedule's shapes, which is what
        // keeps warm outputs bit-identical to cache-off runs.
        let (serve_full, serve_tail_rows) = if r == 0 || tail.is_some() {
            (walked.len(), r)
        } else {
            let fine = config.prefill_fine_chunk.clamp(1, config.prefill_chunk);
            let tail_start = limit / config.prefill_chunk * config.prefill_chunk;
            let m = if d_full >= tail_start {
                tail_start.max(d_full / fine * fine)
            } else {
                d_full / config.prefill_chunk * config.prefill_chunk
            };
            tail = (!m.is_multiple_of(block_size)).then(|| walked[m / block_size]);
            (m / block_size, m % block_size)
        };
        let mut served = 0;
        for &block in &walked[..serve_full] {
            if mgr.retain(block).is_err() {
                break;
            }
            if seq.table.push_full_block(block, block_size).is_err() {
                let released = mgr.release(block);
                debug_assert!(released.is_ok(), "match released a foreign block");
                break;
            }
            served += block_size;
        }
        if served == serve_full * block_size
            && serve_tail_rows > 0
            && let Some(block) = tail
            && mgr.retain(block).is_ok()
        {
            if seq
                .table
                .push_partial_block(block, serve_tail_rows, block_size)
                .is_ok()
            {
                served += serve_tail_rows;
            } else {
                let released = mgr.release(block);
                debug_assert!(released.is_ok(), "match released a foreign block");
            }
        }
        if served == 0 {
            return;
        }
        seq.processed = served;
        if seq.processed + 1 == seq.prompt.len() && seq.history.is_empty() {
            // Fully-covered prompt: the sequence decodes immediately,
            // feeding the last prompt token (mirrors prefill completion).
            seq.history.push(seq.prompt[seq.processed]);
        }
        seq.cached_tokens = seq.cached_tokens.max(served as u32);
        state.hits_total += 1;
        state.tokens_reused_total += served as u64;
        if from_ssd {
            state.tokens_reused_ssd_total += served as u64;
        }
        let _ = (seq.on_event)(SeqEvent::PrefixHit {
            tokens: served as u32,
            from_ssd,
        });
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
            // Grammar mask for this sample (SPEC §6.2 step 3): computed
            // host-side while the matcher state is authoritative, stored
            // on the sequence for `sample_from_row`. A grammar fault
            // fails only this request, in-band.
            if sample {
                let Self {
                    running,
                    mgr,
                    cache,
                    ..
                } = self;
                let seq = &mut running[i];
                if let Some(grammar) = seq.grammar.as_mut() {
                    match grammar.mask_array() {
                        Ok(mask) => seq.pending_mask = Some(mask),
                        Err(err) => {
                            finish(
                                seq,
                                mgr,
                                cache.as_mut(),
                                FinishKind::Error,
                                None,
                                Some((ErrorCause::Internal, err.to_string())),
                            );
                            continue;
                        }
                    }
                }
            }
            // Speculation (SPEC §6.5): an eligible sampling sequence asks
            // the drafter for a proposal and plans a gamma+1-slot verify
            // segment instead of a 1-slot decode. Speculation never causes
            // eviction or preemption — if the pool cannot cover the wider
            // segment the request falls back to the plain planner below.
            if sample && let Some(proposal) = self.collect_proposal(i) {
                let step = build_seq_step(
                    &mut self.running[i],
                    &mut self.mgr,
                    &mut self.kv,
                    1 + proposal.len(),
                    true,
                    &self.stream,
                )?;
                if let Some(step) = step {
                    decode_plans.push(DecodePlan {
                        seq: i,
                        sample: true,
                        step,
                        draft: Some(proposal),
                    });
                    continue;
                }
            }
            if let Some(step) = self.plan_step(i, 1, sample, &mut decode_plans)? {
                decode_plans.push(DecodePlan {
                    seq: i,
                    sample,
                    step,
                    draft: None,
                });
            }
        }
        let mut prefill: Option<(usize, StepBatch)> = None;
        if let Some(i) = self
            .running
            .iter()
            .position(|seq| !seq.finished && !seq.preempted && !seq.decoding())
        {
            // Speculative segments contribute their gamma+1 positions to
            // the step budget (SPEC §6.5).
            let budget = self
                .config
                .max_batch_tokens
                .saturating_sub(decode_plans.iter().map(|plan| plan.step.len as usize).sum());
            let seq = &self.running[i];
            // Phase-3/mlx-lm chunk rule: prefill covers prompt[..n-1]; the
            // last prompt token is fed by the first sampled (decode) step.
            let at = seq.processed;
            let chunk = canonical_prefill_len(
                at,
                seq.prompt.len() - 1,
                self.config.prefill_chunk,
                self.config.prefill_fine_chunk,
            )
            .min(budget);
            if chunk > 0
                && let Some(step) = self.plan_step(i, chunk, false, &mut decode_plans)?
            {
                let seq = &self.running[i];
                let tokens = seq.prompt[at..at + chunk].to_vec();
                // ADR 0002 kernel-class padding: sub-32-row ragged pieces
                // off the super-chunk grid run with pad rows. Capped at
                // `at` so the padded SDPA query never exceeds the KV
                // coverage (`at` is the piece's RoPE offset).
                let pad_rows = if chunk < PREFILL_PAD_MIN_ROWS
                    && !at.is_multiple_of(self.config.prefill_chunk)
                {
                    (PREFILL_PAD_MIN_ROWS - chunk).min(at) as i32
                } else {
                    0
                };
                prefill = Some((
                    i,
                    StepBatch {
                        input: StepInput::Ids(tokens),
                        seqs: vec![step],
                        pad_rows,
                    },
                ));
            }
        }

        // --- Split off verify segments (SPEC §6.5): each runs as its own
        // forward — a speculating request is greedy, and its gamma+1 rows
        // (clamped under `deterministic_decode_width` at proposal time)
        // must stay in the M=1 kernel classes on their own, not share a
        // trunk with other rows. The split happens after prefill planning
        // so a preemption there dropped victims from the one plans vec.
        let (verify_plans, decode_plans): (Vec<DecodePlan>, Vec<DecodePlan>) = decode_plans
            .into_iter()
            .partition(|plan| plan.draft.is_some());

        // --- Partition (ADR 0002 B'): deterministic rows (greedy or
        // client-seeded) decode in sub-batches of at most
        // `deterministic_decode_width` rows, keeping every trunk matmul
        // in the M=1 kernel class — bit-identical to single-stream at any
        // admitted width. Non-deterministic rows ride ONE unrestricted
        // full-width forward (no weight re-reads). Grouping is
        // value-neutral across sequences: each gathers only its own
        // history within the step.
        let groups = partition_decode_plans(
            decode_plans,
            self.config.deterministic_decode_width.max(1),
            |plan| self.running[plan.seq].deterministic,
        );

        // --- Forward (SPEC §6.2 step 3). Prefill first so its KV writes
        // sit earliest on the pools' functional-update chain.
        if let Some((_, batch)) = &prefill {
            let logits = self.model.forward_step(batch, &mut self.kv, &self.stream)?;
            debug_assert!(logits.is_none(), "prefill chunks never sample");
        }
        let mut sampled: Vec<(usize, Array)> = Vec::new();
        let mut replay_ids: Vec<usize> = Vec::new();
        for group in groups {
            let mut tokens: Vec<u32> = Vec::with_capacity(group.len());
            let mut seqs: Vec<SeqStep> = Vec::with_capacity(group.len());
            let mut sampled_ids: Vec<usize> = Vec::new();
            for plan in group {
                let seq = &self.running[plan.seq];
                tokens.push(seq.history[seq.fed]);
                seqs.push(plan.step);
                if plan.sample {
                    sampled_ids.push(plan.seq);
                } else {
                    replay_ids.push(plan.seq);
                }
            }
            let batch = StepBatch {
                input: StepInput::Ids(tokens),
                seqs,
                pad_rows: 0,
            };
            let logits = self
                .model
                .forward_step(&batch, &mut self.kv, &self.stream)?;
            if sampled_ids.is_empty() {
                debug_assert!(logits.is_none(), "replay-only groups never sample");
                continue;
            }
            let logits = logits.ok_or_else(|| MlxError {
                message: "model returned no logits for a decode step".to_owned(),
            })?;
            // Speculation waits for the logits width (proposal-id bounds).
            self.logits_vocab.get_or_insert(logits.dim(2));
            for (row, &i) in sampled_ids.iter().enumerate() {
                let token =
                    sample_from_row(&mut self.running[i], &logits, row as i32, &self.stream)?;
                sampled.push((i, token));
            }
        }

        // --- Verify forwards (SPEC §6.5): one target forward per
        // speculating sequence scores the fed token AND every proposed
        // token (gamma+1 rows, all sampled). Rows are sampled lazily like
        // everything else; acceptance is decided host-side after the step
        // eval. The segment's K/V writes cover the proposed positions too
        // — rejected rows are rolled back after settling.
        let mut verifies: Vec<(usize, Vec<u32>, Vec<Array>)> = Vec::new();
        for plan in verify_plans {
            let i = plan.seq;
            let draft = plan.draft.unwrap_or_default();
            let mut tokens: Vec<u32> = Vec::with_capacity(1 + draft.len());
            {
                let seq = &self.running[i];
                tokens.push(seq.history[seq.fed]);
            }
            tokens.extend_from_slice(&draft);
            let batch = StepBatch {
                input: StepInput::Ids(tokens),
                seqs: vec![plan.step],
                pad_rows: 0,
            };
            let logits = self
                .model
                .forward_step(&batch, &mut self.kv, &self.stream)?
                .ok_or_else(|| MlxError {
                    message: "model returned no logits for a verify step".to_owned(),
                })?;
            let mut rows: Vec<Array> = Vec::with_capacity(1 + draft.len());
            for row in 0..=draft.len() {
                rows.push(sample_from_row(
                    &mut self.running[i],
                    &logits,
                    row as i32,
                    &self.stream,
                )?);
            }
            verifies.push((i, draft, rows));
        }

        // --- Pipeline entry (module docs): a steady-state pure-decode
        // step defers its readback with async_eval; the next step's
        // graph is built from these still-lazy tokens at the start of
        // the next step() call, overlapping this step's GPU execution.
        // Every row samples, so each sequence's KV write is an ancestor
        // of its token — no pool-state eval needed.
        if prefill.is_none()
            && verifies.is_empty()
            && replay_ids.is_empty()
            && sampled.len() == self.running.len()
            && self.pipeline_ok()
        {
            {
                let outputs: Vec<&Array> = sampled.iter().map(|(_, token)| token).collect();
                async_eval(&outputs)?;
            }
            self.pipelined_total += 1;
            self.inflight = Some(InFlight {
                rows: sampled
                    .into_iter()
                    .map(|(i, token)| (self.running[i].arrival, token))
                    .collect(),
            });
            return Ok(());
        }

        // --- Evaluate the step: sampled tokens (plain and verify rows) +
        // pool state together (prefill/replay-only steps still materialize
        // their KV writes here).
        {
            let mut outputs: Vec<&Array> = sampled.iter().map(|(_, token)| token).collect();
            for (_, _, rows) in &verifies {
                outputs.extend(rows.iter());
            }
            let state = self.kv.state();
            outputs.extend(state);
            eval(&outputs)?;
        }

        // --- Emit, check stops, release finished (SPEC §6.2 steps 3-4).
        for (i, token) in &sampled {
            let token = token.item_u32()?;
            settle_sampled(
                &mut self.running[*i],
                &mut self.mgr,
                self.cache.as_mut(),
                token,
            );
        }
        // Replay segments advance silently: their tokens were emitted
        // before the preemption.
        for &i in &replay_ids {
            self.running[i].fed += 1;
        }

        // --- Settle verify rounds (SPEC §6.5): commit the longest
        // agreeing prefix plus the target's own token (bonus on full
        // agreement, correction on the first mismatch), then roll back the
        // rejected positions by releasing their blocks — bookkeeping only,
        // timed into the spec counters so tests can hold the O(1) claim to
        // measurement. Row j's logits are valid only when rows 0..j all
        // agreed (its input was draft[j-1]), which is exactly where the
        // walk stops.
        for (i, draft, rows) in verifies {
            self.spec.rounds_total += 1;
            self.spec.proposed_total += draft.len() as u64;
            self.running[i].spec_proposed += draft.len() as u32;
            let mut accepted = 0usize;
            let mut committed = 0usize;
            for (j, row) in rows.iter().enumerate() {
                if self.running[i].finished {
                    break;
                }
                let token = row.item_u32()?;
                // Acceptance is a token-value fact, so count it BEFORE
                // settling: settle_sampled may finish the request
                // (stop/length) and emit its summary, which must already
                // carry this round's acceptances.
                let agrees = j < draft.len() && token == draft[j];
                if agrees {
                    accepted += 1;
                    self.running[i].spec_accepted += 1;
                }
                settle_sampled(
                    &mut self.running[i],
                    &mut self.mgr,
                    self.cache.as_mut(),
                    token,
                );
                committed += 1;
                if !agrees {
                    break;
                }
            }
            self.spec.accepted_total += accepted as u64;
            let stale = (1 + draft.len()).saturating_sub(committed);
            if stale > 0 {
                self.spec.rollback_rounds_total += 1;
                self.spec.rollback_tokens_total += stale as u64;
                let seq = &mut self.running[i];
                if !seq.finished {
                    // A finished sequence's table was already disposed by
                    // `finish` (donation stops at the settled rows, extra
                    // blocks are released there).
                    let rollback_started = Instant::now();
                    seq.table.truncate(&mut self.mgr, seq.processed + seq.fed)?;
                    self.spec.rollback_nanos_total += rollback_started.elapsed().as_nanos() as u64;
                }
            }
            self.maybe_stand_down(i);
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
        self.release_finished_drafts();
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
            // Out of blocks: reclaim from the prefix cache first — its
            // sole-owned LRU leaves are free memory (SPEC §6.3 eviction) —
            // and only then preempt anyone.
            if self
                .cache
                .as_mut()
                .is_some_and(|state| state.evict_one(&mut self.mgr))
            {
                continue;
            }
            // Preempt the least-deserving block-holder (the requester
            // competes with its own key, so a newcomer can never displace
            // a more deserving request).
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
                            self.cache.as_mut(),
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
        // The resume re-consults the prefix cache (its old blocks are
        // gone; a sibling's donation may cover the prompt by then).
        seq.cache_checked = false;
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
                None, // error finishes never donate
                FinishKind::Error,
                None,
                Some((ErrorCause::Internal, format!("engine step failed: {error}"))),
            );
        }
        self.release_finished_drafts();
        self.running.clear();
        self.waiting.clear();
        // The prefix cache dies with the pools: its nodes name blocks of
        // the manager being rebuilt below and KV whose pending graphs may
        // be poisoned. SSD copies survive (their bytes were captured from
        // settled state) and are re-discovered hash-first on later walks.
        if let Some(state) = &mut self.cache {
            state.radix.reset();
            state.flush_queue.clear();
        }
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

/// Resolves the child of `cur` keyed by `chunk` for a match walk:
/// existing node, or hash-first discovery from the SSD index. Ensures the
/// node is resident (loading from SSD when needed); `None` means the walk
/// stops here (silent-skip policy).
fn resolve_chunk(
    state: &mut CacheState,
    mgr: &mut BlockManager,
    kv: &mut PagedKv,
    stream: &Stream,
    cur: usize,
    chunk: &[u32],
    from_ssd: &mut bool,
) -> Option<usize> {
    let node = match state.radix.child(cur, chunk) {
        Some(node) => node,
        None => {
            let store = state.store.as_ref()?;
            let hash = state.radix.chain_hash_of(cur, chunk);
            if !store.contains(&hash) {
                return None;
            }
            let node = state.radix.insert_child(cur, chunk);
            state.radix.set_on_ssd(node);
            node
        }
    };
    if state.radix.node_block(node).is_none() {
        if !state.radix.node_on_ssd(node) || state.store.is_none() {
            // Unreachable bookkeeping (non-resident, nothing on SSD):
            // drop the dead subtree and stop.
            state.radix.prune_subtree(mgr, node);
            return None;
        }
        if !load_from_ssd(state, mgr, kv, stream, node, chunk) {
            return None;
        }
        *from_ssd = true;
    }
    Some(node)
}

/// Pulls one persisted block into the pool for a non-resident radix node
/// (SPEC §6.4 read path). Returns false — after cleaning up — when the
/// slot is missing, fails verification, or no pool block can be freed;
/// failures are silent by policy.
fn load_from_ssd(
    state: &mut CacheState,
    mgr: &mut BlockManager,
    kv: &mut PagedKv,
    stream: &Stream,
    node: usize,
    chunk: &[u32],
) -> bool {
    let hash = state.radix.node_hash(node);
    let pool_tag = kv.dtype().map(dtype_tag);
    let Some(store) = &mut state.store else {
        return false;
    };
    let Some((payload, tag)) = store.read(&hash, chunk, pool_tag) else {
        // Verification failed (torn slot, dtype change, evicted file):
        // the store dropped its index entry; drop the dead subtree too.
        state.radix.prune_subtree(mgr, node);
        return false;
    };
    let Some(dtype) = dtype_from_tag(tag) else {
        state.radix.prune_subtree(mgr, node);
        return false;
    };
    let block = match mgr.allocate() {
        Ok(block) => Some(block),
        Err(_) => state.evict_one(mgr).then(|| mgr.allocate().ok()).flatten(),
    };
    let Some(block) = block else {
        return false; // pool genuinely full of live requests; miss
    };
    if kv
        .write_block_bytes(block, &payload, dtype, stream)
        .is_err()
    {
        let released = mgr.release(block);
        debug_assert!(released.is_ok(), "load released a foreign block");
        return false;
    }
    state.radix.set_resident(node, block);
    true
}

/// Captures up to `max` queued blocks and hands them to the SSD writer
/// (write-behind flush). Skips stale, already-flushed, or non-resident
/// entries; capture failures drop the entry (silent-skip policy).
fn flush_entries(state: &mut CacheState, kv: &PagedKv, stream: &Stream, max: usize) -> usize {
    let Some(store) = &mut state.store else {
        state.flush_queue.clear();
        return 0;
    };
    let Some(dtype) = kv.dtype() else {
        // Nothing has been written yet, so nothing donatable exists;
        // queued entries (there should be none) can only be stale.
        state.flush_queue.clear();
        return 0;
    };
    let tag = dtype_tag(dtype);
    let size = dtype.size() as u32;
    let mut flushed = 0;
    while flushed < max {
        let Some((node, generation)) = state.flush_queue.pop_front() else {
            break;
        };
        if !state.radix.is_live(node, generation)
            || state.radix.node_on_ssd(node)
            || state.radix.flush_pending(node)
        {
            continue;
        }
        let Some(block) = state.radix.node_block(node) else {
            continue;
        };
        let Ok(payload) = kv.read_block_bytes(block, stream) else {
            continue;
        };
        store.enqueue_write(
            crate::ssd::FlushTicket {
                node,
                generation,
                hash: state.radix.node_hash(node),
            },
            state.radix.node_tokens(node),
            payload,
            tag,
            size,
        );
        state.radix.set_flush_pending(node);
        flushed += 1;
    }
    flushed
}

/// Applies one SSD writer ack to the radix bookkeeping.
fn apply_ack(radix: &mut RadixCache, ack: crate::ssd::FlushAck) {
    if !radix.is_live(ack.node, ack.generation) || radix.node_hash(ack.node) != ack.hash {
        return;
    }
    if ack.ok {
        radix.set_on_ssd(ack.node);
    } else {
        // Allow a later donation to retry the flush.
        radix.clear_flush_pending(ack.node);
    }
}

/// Stable on-disk dtype tags for slab headers (never renumber).
fn dtype_tag(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Bool => 1,
        Dtype::Uint8 => 2,
        Dtype::Uint16 => 3,
        Dtype::Uint32 => 4,
        Dtype::Int32 => 5,
        Dtype::Float16 => 6,
        Dtype::Bfloat16 => 7,
        Dtype::Float32 => 8,
    }
}

fn dtype_from_tag(tag: u32) -> Option<Dtype> {
    Some(match tag {
        1 => Dtype::Bool,
        2 => Dtype::Uint8,
        3 => Dtype::Uint16,
        4 => Dtype::Uint32,
        5 => Dtype::Int32,
        6 => Dtype::Float16,
        7 => Dtype::Bfloat16,
        8 => Dtype::Float32,
        _ => return None,
    })
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
    // Kernel-path inputs (SPEC §7.4): decode-shaped segments only. Built
    // here — once per sequence per step — so every layer's attention call
    // shares the same block-table/length arrays. A `len == 1` ragged
    // PREFILL piece may also land here; if the planner later pads it
    // (ADR 0002), the model's `pad == 0` gate skips the kernel route and
    // these inputs go unused, which is correct (padded pieces must run the
    // reference's padded-SDPA class).
    let paged_attn = if kv.attention_kernel_enabled() && len == 1 {
        Some(PagedAttnInputs::build(
            seq.table.blocks().iter().map(|b| b.index() as u32),
            offset as i32 + 1,
        )?)
    } else {
        None
    };
    Ok(Some(SeqStep {
        len: len as i32,
        offset: offset as i32,
        // Sampling segments return logits for every fed position: exactly
        // 1 for plain decode (len 1), the whole segment for a speculative
        // verify (SPEC §6.5). Replay and prefill sample nothing.
        sample_rows: if sample { len as i32 } else { 0 },
        blocks: seq.table.blocks().to_vec(),
        writes,
        paged_attn,
    }))
}

#[cfg(test)]
mod tests {
    use super::canonical_prefill_len;

    /// Walks the schedule from `start`, returning (offset, len) segments.
    fn walk(start: usize, limit: usize, chunk: usize, fine: usize) -> Vec<(usize, usize)> {
        let mut at = start;
        let mut segments = Vec::new();
        while at < limit {
            let len = canonical_prefill_len(at, limit, chunk, fine);
            assert!(len >= 1 && at + len <= limit, "bad segment ({at}, {len})");
            segments.push((at, len));
            at += len;
        }
        segments
    }

    #[test]
    fn schedule_shapes_and_boundaries() {
        // Sub-super-chunk prompt: pure fine grid plus remainder.
        assert_eq!(
            walk(0, 248, 2048, 64),
            vec![(0, 64), (64, 64), (128, 64), (192, 56)]
        );
        // Bulk super-chunk, then the fine tail.
        let schedule = walk(0, 4095, 2048, 64);
        assert_eq!(schedule[0], (0, 2048));
        assert_eq!(schedule[1], (2048, 64));
        assert_eq!(*schedule.last().unwrap(), (4032, 63));
        // fine >= chunk degenerates to the pre-Phase-5 schedule.
        assert_eq!(walk(0, 248, 2048, 2048), vec![(0, 248)]);
        assert_eq!(walk(0, 4095, 2048, 2048), vec![(0, 2048), (2048, 2047)]);
        // Exact super-chunk multiple: no partial tail exists.
        assert_eq!(walk(0, 4096, 2048, 64), vec![(0, 2048), (2048, 2048)]);
    }

    /// The bit-exactness property: resuming at any boundary of the cold
    /// schedule yields exactly the cold schedule's suffix — no (offset,
    /// length) shape a cold run would not produce.
    #[test]
    fn schedule_is_resume_invariant() {
        for &(chunk, fine) in &[(16_usize, 4_usize), (2048, 64), (8, 8), (48, 48)] {
            for limit in 1..=(3 * chunk + 5) {
                let cold = walk(0, limit, chunk, fine);
                for i in 0..cold.len() {
                    let (offset, _) = cold[i];
                    assert_eq!(
                        walk(offset, limit, chunk, fine),
                        cold[i..],
                        "resume at {offset} diverges (limit {limit}, chunk {chunk}, fine {fine})"
                    );
                }
            }
        }
    }
}
