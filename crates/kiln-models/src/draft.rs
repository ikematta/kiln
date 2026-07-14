//! The `DraftModel` speculation mode (SPEC §6.5): a second, smaller model
//! loaded in the same worker process as the target — its own weight set
//! and its own paged KV pool, sharing nothing but the Metal device/stream
//! (every op still issues from the single engine thread).
//!
//! Isolation is structural: the draft's weights are a separate
//! [`AnyModel`] over separate arrays, and its KV lives in a separate
//! [`PagedKv`] addressed by a separate [`BlockManager`] — no pool, block
//! id, or array is shared with the target, so draft activity cannot
//! perturb target numerics (asserted bit-level by `tests/draft.rs`).
//! Accounting is shared: the draft's weights and pool bytes report
//! through [`Drafter::memory`] into the same worker heartbeat totals the
//! target uses (SPEC §2.3), so gateway budget math sees one combined
//! footprint per worker.
//!
//! Proposal generation ([`Drafter::propose`]) is plain greedy decoding on
//! the draft's own paged pool: reconcile with the committed context
//! (truncating any speculated tail — O(1) block release, the same
//! rollback primitive the target side uses), catch up on unfed context in
//! prefill-style chunks, then autoregressively propose `gamma` tokens.
//! The proposed-token feeds chain lazily (each argmax feeds the next
//! forward unevaluated) and the whole round evaluates once. Draft
//! numerics are deliberately unconstrained — no deterministic-width
//! calibration, no canonical chunk schedule — because every proposal is
//! verified by the target before anything is committed (SPEC §6.5); a
//! wrong draft token costs throughput, never correctness.
//!
//! Compatibility: a draft is only usable when its token ids MEAN the same
//! text as the target's — see [`check_draft_compat`], which the worker
//! runs before attaching a drafter. A mismatched pair (e.g. a qwen3 draft
//! under a llama target) is rejected loudly at load; nothing guards the
//! verify loop itself against a mismatched attach beyond the engine's
//! logits-width bound.

use std::collections::HashMap;
use std::path::Path;

use kiln_engine::{
    BlockError, BlockManager, BlockTable, DEFAULT_BLOCK_SIZE, DEFAULT_NUM_BLOCKS, DraftError,
    Drafter, DrafterMemory, KvDims, KvSpec, PagedKv, SeqStep, StepBatch, StepInput, StepModel,
    WriteRun,
};
use kiln_mlx::{MlxError, Stream, eval, ops};

use crate::model::AnyModel;
use crate::nn::ModelError;

/// Chunk size for draft-side context catch-up. Shape freedom is deliberate
/// (draft numerics never bind correctness); 2048 mirrors the target's
/// prefill chunk so draft prefill memory stays step-bounded.
const DRAFT_PREFILL_CHUNK: usize = 2048;

/// Geometry of the draft's own KV pool. Defaults mirror the engine's
/// target-pool defaults so the draft can shadow the same token capacity
/// the target pool admits — a sequence the target accepted never needs a
/// draft-side capacity decision (auto-disable heuristics are a later
/// Phase 8 part, not a loading concern).
#[derive(Debug, Clone, Copy)]
pub struct DraftPoolSpec {
    /// Tokens per block (SPEC §6.3: power of two).
    pub block_size: usize,
    /// Blocks in the draft pool.
    pub num_blocks: usize,
}

impl Default for DraftPoolSpec {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            num_blocks: DEFAULT_NUM_BLOCKS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DraftLoadError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("invalid draft KV pool spec: {0}")]
    Pool(String),
    #[error("failed to scan draft weights at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// The draft/target pair cannot speculate together (SPEC §6.5): the
    /// draft's token ids would not mean the same text to the target. The
    /// worker treats this as a load failure (UNHEALTHY), never a silent
    /// downgrade.
    #[error("draft/target tokenizers are incompatible: {0}")]
    Incompatible(String),
}

/// Per-sequence draft state.
#[derive(Debug, Default)]
struct DraftSeq {
    /// The committed target-side context: prompt ++ committed tokens, as
    /// fed through `begin`/`propose`.
    context: Vec<u32>,
    /// Tokens whose K/V rows live in the draft pool, in position order.
    /// Always `context[..k]` for some `k`, possibly extended by tokens
    /// this drafter itself proposed last round (its speculation) — the
    /// part reconciliation may roll back.
    fed: Vec<u32>,
    table: BlockTable,
}

/// A loaded draft model plus its own KV pool bookkeeping.
#[derive(Debug)]
pub struct DraftModel {
    model: AnyModel,
    mgr: BlockManager,
    kv: PagedKv,
    /// Live draft-side sequence state, keyed by the engine's arrival
    /// numbers.
    seqs: HashMap<u64, DraftSeq>,
    /// Sum of the checkpoint's `.safetensors` file sizes — the same
    /// convention the worker's `StaticInfo.weights_bytes` uses for the
    /// target, so the two add up coherently in `MemoryReport`.
    weights_bytes: u64,
}

impl DraftModel {
    /// Loads the draft checkpoint at `dir` (any supported architecture)
    /// and sizes — but does not materialize — its private KV pool. Pool
    /// bytes stay 0 until first written, exactly like the target's.
    pub fn load(
        dir: impl AsRef<Path>,
        pool: DraftPoolSpec,
        s: &Stream,
    ) -> Result<Self, DraftLoadError> {
        let dir = dir.as_ref();
        pool.num_blocks
            .checked_mul(pool.block_size)
            .filter(|&tokens| tokens <= i32::MAX as usize)
            .ok_or_else(|| {
                DraftLoadError::Pool(format!(
                    "pool of {} blocks x {} tokens overflows i32 addressing",
                    pool.num_blocks, pool.block_size
                ))
            })?;
        let model = AnyModel::load(dir, s)?;
        let dims = model.kv_dims();
        let mgr = BlockManager::new(pool.num_blocks, pool.block_size)
            .map_err(|err| DraftLoadError::Pool(err.to_string()))?;
        let kv = PagedKv::new(KvSpec {
            layers: dims.layers,
            kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            num_blocks: pool.num_blocks,
            block_size: pool.block_size,
        });
        let weights_bytes = weights_bytes(dir)?;
        Ok(Self {
            model,
            mgr,
            kv,
            seqs: HashMap::new(),
            weights_bytes,
        })
    }

    pub fn model(&self) -> &AnyModel {
        &self.model
    }

    pub fn kv_dims(&self) -> KvDims {
        self.model.kv_dims()
    }

    /// The draft's own KV pool (test-facing: the coexistence suite writes
    /// sentinels through it).
    pub fn kv(&self) -> &PagedKv {
        &self.kv
    }

    pub fn kv_mut(&mut self) -> &mut PagedKv {
        &mut self.kv
    }
}

/// One draft forward over a fresh `len`-slot segment of `table`: appends
/// the slots, derives the write runs, and runs the model against the
/// draft pool. `sample_rows` as in [`SeqStep`].
fn draft_forward(
    model: &AnyModel,
    kv: &mut PagedKv,
    mgr: &mut BlockManager,
    table: &mut BlockTable,
    input: StepInput,
    sample_rows: i32,
    s: &Stream,
) -> Result<Option<kiln_mlx::Array>, DraftError> {
    let len = input.num_tokens();
    let offset = table.num_tokens();
    let plan = table.append_tokens(mgr, len).map_err(|err| match err {
        BlockError::OutOfBlocks { .. } => DraftError::Capacity(err.to_string()),
        other => DraftError::Capacity(other.to_string()),
    })?;
    // Draft blocks are never shared (no prefix cache on the draft pool),
    // so a copy-on-write can never arise.
    debug_assert!(plan.cow.is_none(), "draft table hit a shared block");
    let block_size = mgr.block_size();
    let mut writes = Vec::with_capacity(len / block_size + 2);
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
        input,
        seqs: vec![step],
        pad_rows: 0,
    };
    model.forward_step(&batch, kv, s).map_err(DraftError::Mlx)
}

/// Weight footprint by the `StaticInfo.weights_bytes` convention: the
/// checkpoint's `.safetensors` file sizes summed.
fn weights_bytes(dir: &Path) -> Result<u64, DraftLoadError> {
    let entries = std::fs::read_dir(dir).map_err(|source| DraftLoadError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    let mut bytes = 0;
    for entry in entries {
        let entry = entry.map_err(|source| DraftLoadError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".safetensors") {
            let meta = entry.metadata().map_err(|source| DraftLoadError::Io {
                path: entry.path().display().to_string(),
                source,
            })?;
            bytes += meta.len();
        }
    }
    Ok(bytes)
}

impl Drafter for DraftModel {
    fn memory(&self) -> DrafterMemory {
        let live = (self.mgr.capacity() - self.mgr.num_free()) as u64;
        DrafterMemory {
            weights_bytes: self.weights_bytes,
            kv_allocated_bytes: self.kv.allocated_bytes(),
            kv_used_bytes: live * self.kv.bytes_per_block(),
        }
    }

    fn begin(&mut self, seq: u64, prompt: &[u32], _s: &Stream) -> Result<(), DraftError> {
        // Re-begin of a known sequence is the preemption-resume reset
        // (trait contract): all prior draft state — including its pool
        // blocks — is discarded.
        if let Some(old) = self.seqs.remove(&seq) {
            let released = old.table.release(&mut self.mgr);
            debug_assert!(released.is_ok(), "draft begin released a foreign block");
        }
        self.seqs.insert(
            seq,
            DraftSeq {
                context: prompt.to_vec(),
                ..DraftSeq::default()
            },
        );
        Ok(())
    }

    fn propose(
        &mut self,
        seq: u64,
        committed: &[u32],
        gamma: usize,
        s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        let state = self.seqs.get_mut(&seq).ok_or(DraftError::UnknownSeq(seq))?;
        let mgr = &mut self.mgr;

        // --- Reconcile (trait contract): the committed tokens extend the
        // context; whatever this drafter speculated beyond them rolls back
        // by block release — the same O(1) primitive as the target side.
        // fed[..old context] always agrees with the context by invariant,
        // so only the tail past the previous context needs comparing.
        let base = state.context.len().min(state.fed.len());
        state.context.extend_from_slice(committed);
        let mut keep = base;
        while keep < state.fed.len()
            && keep < state.context.len()
            && state.fed[keep] == state.context[keep]
        {
            keep += 1;
        }
        if keep < state.fed.len() {
            state
                .table
                .truncate(mgr, keep)
                .map_err(|err| DraftError::Capacity(err.to_string()))?;
            state.fed.truncate(keep);
        }
        // Defensive resync after a mid-round fault in some earlier call
        // (table slots appended whose feed never completed).
        if state.table.num_tokens() != state.fed.len() {
            state
                .table
                .truncate(mgr, state.fed.len())
                .map_err(|err| DraftError::Capacity(err.to_string()))?;
        }

        let len = state.context.len();
        if gamma == 0 || len == 0 || state.fed.len() >= len {
            // Nothing to propose from (or nothing asked): a legal "no
            // speculation this round".
            return Ok(Vec::new());
        }

        // --- Catch up: feed unfed context short of the last token (whose
        // forward opens the proposal chain).
        let last = len - 1;
        let mut at = state.fed.len();
        while at < last {
            let chunk = (last - at).min(DRAFT_PREFILL_CHUNK);
            let ids = state.context[at..at + chunk].to_vec();
            let logits = draft_forward(
                &self.model,
                &mut self.kv,
                mgr,
                &mut state.table,
                StepInput::Ids(ids),
                0,
                s,
            )?;
            debug_assert!(logits.is_none(), "draft catch-up chunks never sample");
            state.fed.extend_from_slice(&state.context[at..at + chunk]);
            at += chunk;
        }

        // --- Propose gamma tokens greedily. Each argmax feeds the next
        // forward still-lazy; the whole round evaluates once. argmax runs
        // on raw logits (logprob normalization is monotone — same winner,
        // fewer ops; draft bits bind nothing).
        let mut proposal_arrays: Vec<kiln_mlx::Array> = Vec::with_capacity(gamma);
        let mut input = Some(StepInput::Ids(vec![state.context[last]]));
        state.fed.push(state.context[last]);
        for k in 0..gamma {
            let Some(step_input) = input.take() else {
                break; // unreachable: refilled below on every non-final turn
            };
            let logits = draft_forward(
                &self.model,
                &mut self.kv,
                mgr,
                &mut state.table,
                step_input,
                1,
                s,
            )?
            .ok_or_else(|| {
                DraftError::Mlx(MlxError {
                    message: "draft decode returned no logits".to_owned(),
                })
            })?;
            let vocab = logits.dim(2);
            let row = ops::reshape(&logits, &[1, vocab], s).map_err(DraftError::Mlx)?;
            let token = ops::argmax(&row, -1, false, s).map_err(DraftError::Mlx)?;
            if k + 1 < gamma {
                input = Some(StepInput::Lazy(
                    ops::reshape(&token, &[1, 1], s).map_err(DraftError::Mlx)?,
                ));
            }
            proposal_arrays.push(token);
        }

        // --- One evaluation for the round: proposed ids + the pool chain.
        {
            let mut outputs: Vec<&kiln_mlx::Array> = proposal_arrays.iter().collect();
            outputs.extend(self.kv.state());
            eval(&outputs).map_err(DraftError::Mlx)?;
        }
        let proposal: Vec<u32> = proposal_arrays
            .iter()
            .map(|token| token.item_u32())
            .collect::<Result<_, _>>()
            .map_err(DraftError::Mlx)?;
        // All but the final proposed token were fed back into the draft KV.
        state.fed.extend_from_slice(&proposal[..gamma - 1]);
        Ok(proposal)
    }

    fn release(&mut self, seq: u64) {
        if let Some(state) = self.seqs.remove(&seq) {
            let released = state.table.release(&mut self.mgr);
            debug_assert!(released.is_ok(), "draft release freed a foreign block");
        }
    }
}

/// The drafter-attachment compatibility gate (SPEC §6.5, Phase 8 part 2):
/// a draft may only propose for a target whose token ids mean the same
/// text. Verified structurally from the checkpoints, without loading
/// weights:
///
/// - `tokenizer.json` must exist on both sides and agree on the id→token
///   mapping — the `model.vocab` table and the `added_tokens` list. (BPE
///   merges, normalizers, and templates may differ: they affect encoding
///   choices, not what an id means.)
/// - the draft's `config.json` `vocab_size` (its logits width) must not
///   exceed the target's, so every id the draft can emit is embeddable by
///   the target.
///
/// Failures are loud by contract: the worker maps `Err` to UNHEALTHY at
/// load. Callers must run this before `DraftModel::load`/attach; a pair
/// that fails here must never reach the verify loop.
pub fn check_draft_compat(target_dir: &Path, draft_dir: &Path) -> Result<(), DraftLoadError> {
    let read_json = |dir: &Path, file: &str| -> Result<serde_json::Value, DraftLoadError> {
        let path = dir.join(file);
        let bytes = std::fs::read(&path).map_err(|source| DraftLoadError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|err| {
            DraftLoadError::Incompatible(format!("{} does not parse: {err}", path.display()))
        })
    };

    let vocab_size = |dir: &Path| -> Result<u64, DraftLoadError> {
        read_json(dir, "config.json")?
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                DraftLoadError::Incompatible(format!(
                    "{} has no vocab_size to compare",
                    dir.join("config.json").display()
                ))
            })
    };
    let target_vocab = vocab_size(target_dir)?;
    let draft_vocab = vocab_size(draft_dir)?;
    if draft_vocab > target_vocab {
        return Err(DraftLoadError::Incompatible(format!(
            "draft logits width {draft_vocab} exceeds the target's {target_vocab} — the draft \
             can propose ids the target cannot even embed"
        )));
    }

    let target_tok = read_json(target_dir, "tokenizer.json")?;
    let draft_tok = read_json(draft_dir, "tokenizer.json")?;
    let section = |tok: &serde_json::Value, pointer: &str| tok.pointer(pointer).cloned();
    let (t_vocab, d_vocab) = (
        section(&target_tok, "/model/vocab"),
        section(&draft_tok, "/model/vocab"),
    );
    if t_vocab.is_none() || d_vocab.is_none() {
        return Err(DraftLoadError::Incompatible(
            "tokenizer.json lacks a model.vocab table on one side; id equivalence cannot be \
             verified"
                .to_owned(),
        ));
    }
    if t_vocab != d_vocab {
        // Name one witness so the log is actionable without a JSON diff.
        let detail = vocab_mismatch_witness(t_vocab.as_ref(), d_vocab.as_ref());
        return Err(DraftLoadError::Incompatible(format!(
            "tokenizer model.vocab tables differ ({detail}) — draft token ids would not mean \
             the same text to the target"
        )));
    }
    if section(&target_tok, "/added_tokens") != section(&draft_tok, "/added_tokens") {
        return Err(DraftLoadError::Incompatible(
            "tokenizer added_tokens differ — special-token ids would not mean the same text \
             to the target"
                .to_owned(),
        ));
    }
    Ok(())
}

/// One human-readable witness of a vocab-table mismatch.
fn vocab_mismatch_witness(
    target: Option<&serde_json::Value>,
    draft: Option<&serde_json::Value>,
) -> String {
    let (Some(serde_json::Value::Object(target)), Some(serde_json::Value::Object(draft))) =
        (target, draft)
    else {
        return "non-object vocab tables".to_owned();
    };
    if target.len() != draft.len() {
        return format!("{} vs {} entries", target.len(), draft.len());
    }
    for (token, id) in target {
        if draft.get(token) != Some(id) {
            return format!("e.g. token {token:?}: id {id} vs {:?}", draft.get(token));
        }
    }
    "tables agree entry-wise but differ structurally".to_owned()
}
