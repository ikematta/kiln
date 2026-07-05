//! SSD cold tier for KV blocks (SPEC §6.4): fixed-layout binary slab
//! files, one per group of [`SLOTS_PER_FILE`] blocks, under a per-model
//! directory. Persistence across restarts is the feature: the index is
//! rebuilt on startup by scanning slab headers, and the radix tree
//! re-discovers entries hash-first on later prefix walks.
//!
//! MLX-free by design (raw bytes in, raw bytes out); the capture/upload
//! of pool blocks lives in `paged.rs`/`engine.rs`. Writes are
//! asynchronous on a dedicated writer thread — the engine enqueues a slot
//! and learns of completion through acks it drains between steps; a block
//! is only marked SSD-backed (and its index entry inserted) once its
//! write acked, so a read can never observe a half-written slot from this
//! process. (SPEC §6.4 says "a dedicated tokio blocking pool"; kiln-engine
//! has no tokio — the engine loop is a plain OS thread per SPEC §6.2 — so
//! a dedicated writer thread implements the same contract. Recorded as a
//! deviation in PROGRESS.md.)
//!
//! Failure policy (SPEC §6.4): a fingerprint/geometry mismatch, torn slot,
//! or IO error is a silent skip plus a counter increment — never an error
//! surfaced to a request.
//!
//! ## Layout (little-endian throughout)
//!
//! File `slab-<id>.kiln`: a 128-byte header, then `slots` fixed-stride
//! slot entries.
//!
//! ```text
//! header:  magic "KILNSLB1" | version u32 | layers u32 | kv_heads u32
//!          | head_dim u32 | block_size u32 | dtype_tag u32 | dtype_size u32
//!          | slots u32 | fingerprint [32] | zero padding to 128
//! slot:    used u8 | pad [7] | chain_hash [32] | payload_sha256 [32]
//!          | token_ids [block_size * u32] | payload
//! payload: per layer: K rows then V rows, each kv_heads * block_size *
//!          head_dim * dtype_size bytes
//! ```
//!
//! The chain hash (see [`crate::radix`]) keys the slot; the stored token
//! ids double as an equality check on load, and the payload digest makes
//! torn writes detectable. Slabs from another model, dtype, or geometry
//! are rejected by the header fingerprint before any slot is trusted.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

use sha2::{Digest, Sha256};

use crate::radix::ChainHash;

const MAGIC: &[u8; 8] = b"KILNSLB1";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 128;
/// Blocks per slab file (SPEC §6.4: one file per block group of 64).
pub const SLOTS_PER_FILE: u32 = 64;

/// Fixed per-pool geometry a slab must match to be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabGeometry {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
}

impl SlabGeometry {
    /// Payload bytes for one block at `dtype_size` bytes per element.
    pub fn payload_bytes(&self, dtype_size: u32) -> u64 {
        u64::from(self.layers)
            * 2
            * u64::from(self.kv_heads)
            * u64::from(self.block_size)
            * u64::from(self.head_dim)
            * u64::from(dtype_size)
    }

    fn slot_header_bytes(&self) -> u64 {
        8 + 32 + 32 + u64::from(self.block_size) * 4
    }

    fn slot_stride(&self, dtype_size: u32) -> u64 {
        self.slot_header_bytes() + self.payload_bytes(dtype_size)
    }

    fn slot_offset(&self, dtype_size: u32, slot: u32) -> u64 {
        HEADER_BYTES as u64 + u64::from(slot) * self.slot_stride(dtype_size)
    }
}

/// Counters re-exported through `WorkerStats` (SPEC §5).
#[derive(Debug, Default, Clone, Copy)]
pub struct SsdCounters {
    pub reads_total: u64,
    pub writes_total: u64,
    pub writes_failed_total: u64,
    pub fingerprint_rejects_total: u64,
}

/// Where a committed block lives.
#[derive(Debug, Clone, Copy)]
struct SlotRef {
    file: u32,
    slot: u32,
    dtype_tag: u32,
    dtype_size: u32,
}

#[derive(Debug, Default)]
struct FileMeta {
    slots_used: u32,
    /// Committed bytes (header + acked slots; the scan uses file length).
    bytes: u64,
    /// LRU clock for the byte-cap eviction (file granularity).
    last_touch: u64,
    /// Enqueued-but-unacked slots and their bytes.
    pending_slots: u32,
    pending_bytes: u64,
}

/// Identity of one flush request: the radix node asking (with its
/// generation guard) and the chain hash being persisted; echoed back in
/// the [`FlushAck`].
#[derive(Debug, Clone, Copy)]
pub struct FlushTicket {
    pub node: usize,
    pub generation: u64,
    pub hash: ChainHash,
}

/// Completion notice for one enqueued write, correlated back to the radix
/// node that requested the flush.
#[derive(Debug, Clone, Copy)]
pub struct FlushAck {
    pub node: usize,
    pub generation: u64,
    pub hash: ChainHash,
    pub ok: bool,
}

enum Job {
    /// Create `file` and write its 128-byte header.
    Create { file: u32, header: Vec<u8> },
    /// Write one full slot image at `offset` of `file`.
    Write {
        file: u32,
        offset: u64,
        bytes: Vec<u8>,
        ack: FlushAck,
    },
    /// Unlink a slab (byte-cap eviction).
    Delete { file: u32 },
    /// Reply when everything enqueued so far has been written.
    Sync(Sender<()>),
}

/// Engine-thread facade over the slab directory + writer thread.
pub struct SsdStore {
    dir: PathBuf,
    fingerprint: ChainHash,
    geometry: SlabGeometry,
    max_bytes: u64,
    index: HashMap<ChainHash, SlotRef>,
    files: HashMap<u32, FileMeta>,
    /// Bytes on disk plus bytes reserved by in-flight writes.
    bytes: u64,
    next_file: u32,
    /// Slot cursor in the file currently being appended to.
    open_file: Option<(u32, u32)>,
    clock: u64,
    counters: SsdCounters,
    jobs: Sender<Job>,
    acks_rx: Receiver<FlushAck>,
    /// Ack metadata keyed by hash, applied when the ack arrives.
    inflight: HashMap<ChainHash, SlotRef>,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl SsdStore {
    /// Opens (creating if needed) the slab directory and scans it into the
    /// index. `fingerprint` must bind weights, architecture, dtype and
    /// block geometry (SPEC §6.4) — the caller derives it from the model
    /// fingerprint; files that fail it are skipped with a counter bump.
    pub fn open(
        dir: &Path,
        fingerprint: ChainHash,
        geometry: SlabGeometry,
        max_bytes: u64,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (jobs, jobs_rx) = channel();
        let (acks_tx, acks_rx) = channel();
        let writer_dir = dir.to_path_buf();
        let writer = std::thread::Builder::new()
            .name("kiln-ssd-writer".to_owned())
            .spawn(move || writer_main(writer_dir, jobs_rx, acks_tx))?;
        let mut store = Self {
            dir: dir.to_path_buf(),
            fingerprint,
            geometry,
            max_bytes,
            index: HashMap::new(),
            files: HashMap::new(),
            bytes: 0,
            next_file: 0,
            open_file: None,
            clock: 0,
            counters: SsdCounters::default(),
            jobs,
            acks_rx,
            inflight: HashMap::new(),
            writer: Some(writer),
        };
        store.scan()?;
        Ok(store)
    }

    pub fn counters(&self) -> SsdCounters {
        self.counters
    }

    /// Committed (readable) blocks.
    pub fn blocks_stored(&self) -> u64 {
        self.index.len() as u64
    }

    /// Bytes on disk (including reservations for in-flight writes).
    pub fn bytes_stored(&self) -> u64 {
        self.bytes
    }

    /// Whether `hash` has a committed slot.
    pub fn contains(&self, hash: &ChainHash) -> bool {
        self.index.contains_key(hash)
    }

    /// Reads and verifies the payload for `hash`, returning it with its
    /// dtype tag. `expected_tokens` is the block's token chunk from the
    /// request being matched; `pool_dtype_tag` is the engine's pool dtype
    /// when already fixed (`None` accepts the slab's — the first request
    /// after a warm restart). Any mismatch or IO failure removes the index
    /// entry and returns `None` (silent skip + counter).
    pub fn read(
        &mut self,
        hash: &ChainHash,
        expected_tokens: &[u32],
        pool_dtype_tag: Option<u32>,
    ) -> Option<(Vec<u8>, u32)> {
        let slot = *self.index.get(hash)?;
        match self.read_verified(&slot, hash, expected_tokens, pool_dtype_tag) {
            Some(payload) => {
                self.counters.reads_total += 1;
                self.clock += 1;
                if let Some(meta) = self.files.get_mut(&slot.file) {
                    meta.last_touch = self.clock;
                }
                Some((payload, slot.dtype_tag))
            }
            None => {
                self.counters.fingerprint_rejects_total += 1;
                self.index.remove(hash);
                None
            }
        }
    }

    fn read_verified(
        &self,
        slot: &SlotRef,
        hash: &ChainHash,
        expected_tokens: &[u32],
        pool_dtype_tag: Option<u32>,
    ) -> Option<Vec<u8>> {
        if pool_dtype_tag.is_some_and(|tag| slot.dtype_tag != tag) {
            return None;
        }
        let g = &self.geometry;
        let file = File::open(self.dir.join(file_name(slot.file))).ok()?;
        let offset = g.slot_offset(slot.dtype_size, slot.slot);
        let mut header = vec![0_u8; g.slot_header_bytes() as usize];
        file.read_exact_at(&mut header, offset).ok()?;
        if header[0] != 1 || header[8..40] != hash[..] {
            return None;
        }
        let tokens_off = 72;
        for (i, token) in expected_tokens.iter().enumerate() {
            let at = tokens_off + i * 4;
            if header[at..at + 4] != token.to_le_bytes() {
                return None;
            }
        }
        let mut payload = vec![0_u8; g.payload_bytes(slot.dtype_size) as usize];
        file.read_exact_at(&mut payload, offset + g.slot_header_bytes())
            .ok()?;
        let digest: ChainHash = Sha256::digest(&payload).into();
        (digest[..] == header[40..72]).then_some(payload)
    }

    /// Enqueues one block for the writer thread. The caller learns of
    /// completion via [`Self::drain_acks`]; until then the block is not
    /// readable. Enforces the byte cap by unlinking LRU slabs first.
    pub fn enqueue_write(
        &mut self,
        ticket: FlushTicket,
        tokens: &[u32],
        payload: Vec<u8>,
        dtype_tag: u32,
        dtype_size: u32,
    ) {
        let hash = ticket.hash;
        let g = self.geometry;
        debug_assert_eq!(payload.len() as u64, g.payload_bytes(dtype_size));
        let stride = g.slot_stride(dtype_size);

        // Byte cap (SPEC §6.4 `ssd_cache_max_gb`): drop least-recently-hit
        // slabs — never the one this write will land in, and never one
        // with writes still in flight. A full slab is fair game even if it
        // was the most recent append target.
        let starts_new_file = !matches!(self.open_file, Some((_, used)) if used < SLOTS_PER_FILE);
        let incoming = stride
            + if starts_new_file {
                HEADER_BYTES as u64
            } else {
                0
            };
        let protected = if starts_new_file {
            None
        } else {
            self.open_file.map(|(file, _)| file)
        };
        while self.bytes + incoming > self.max_bytes {
            let victim = self
                .files
                .iter()
                .filter(|(id, meta)| protected != Some(**id) && meta.pending_slots == 0)
                .min_by_key(|(_, meta)| meta.last_touch)
                .map(|(id, _)| *id);
            let Some(victim) = victim else { break };
            self.drop_file(victim);
            if self.open_file.map(|(file, _)| file) == Some(victim) {
                self.open_file = None;
            }
        }

        // Slot placement: continue the open slab or start a new one.
        let (file, slot) = match self.open_file {
            Some((file, used)) if used < SLOTS_PER_FILE => (file, used),
            _ => {
                let file = self.next_file;
                self.next_file += 1;
                let _ = self.jobs.send(Job::Create {
                    file,
                    header: file_header(&self.fingerprint, &g, dtype_tag, dtype_size),
                });
                self.bytes += HEADER_BYTES as u64;
                self.files.insert(
                    file,
                    FileMeta {
                        bytes: HEADER_BYTES as u64,
                        last_touch: self.clock,
                        ..FileMeta::default()
                    },
                );
                (file, 0)
            }
        };
        self.open_file = Some((file, slot + 1));

        let mut bytes = Vec::with_capacity(stride as usize);
        bytes.push(1_u8);
        bytes.extend_from_slice(&[0_u8; 7]);
        bytes.extend_from_slice(&hash);
        let digest: ChainHash = Sha256::digest(&payload).into();
        bytes.extend_from_slice(&digest);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        debug_assert_eq!(bytes.len() as u64, g.slot_header_bytes());
        bytes.extend_from_slice(&payload);

        self.bytes += stride;
        if let Some(meta) = self.files.get_mut(&file) {
            meta.pending_slots += 1;
            meta.pending_bytes += stride;
        }
        self.inflight.insert(
            hash,
            SlotRef {
                file,
                slot,
                dtype_tag,
                dtype_size,
            },
        );
        let _ = self.jobs.send(Job::Write {
            file,
            offset: g.slot_offset(dtype_size, slot),
            bytes,
            ack: FlushAck {
                node: ticket.node,
                generation: ticket.generation,
                hash,
                ok: true,
            },
        });
    }

    /// Applies completed writes: commits their index entries and returns
    /// the acks so the engine can mark radix nodes SSD-backed.
    pub fn drain_acks(&mut self) -> Vec<FlushAck> {
        let mut acks = Vec::new();
        while let Ok(ack) = self.acks_rx.try_recv() {
            self.commit(&ack);
            acks.push(ack);
        }
        acks
    }

    fn commit(&mut self, ack: &FlushAck) {
        let Some(slot) = self.inflight.remove(&ack.hash) else {
            // The slab was evicted (or never tracked) while the write was
            // queued; the delete already reclaimed the reservation.
            return;
        };
        let stride = self.geometry.slot_stride(slot.dtype_size);
        let Some(meta) = self.files.get_mut(&slot.file) else {
            return;
        };
        meta.pending_slots = meta.pending_slots.saturating_sub(1);
        meta.pending_bytes = meta.pending_bytes.saturating_sub(stride);
        if ack.ok {
            self.counters.writes_total += 1;
            self.clock += 1;
            meta.slots_used += 1;
            meta.bytes += stride;
            meta.last_touch = self.clock;
            self.index.insert(ack.hash, slot);
        } else {
            self.counters.writes_failed_total += 1;
            self.bytes = self.bytes.saturating_sub(stride);
        }
    }

    fn drop_file(&mut self, file: u32) {
        if let Some(meta) = self.files.remove(&file) {
            self.bytes = self.bytes.saturating_sub(meta.bytes + meta.pending_bytes);
        }
        self.index.retain(|_, slot| slot.file != file);
        self.inflight.retain(|_, slot| slot.file != file);
        let _ = self.jobs.send(Job::Delete { file });
    }

    /// Blocks until the writer thread has drained everything enqueued so
    /// far, then commits the acks. Shutdown/tests only — never called from
    /// the steady-state step path.
    pub fn sync(&mut self) -> Vec<FlushAck> {
        let (tx, rx) = channel();
        if self.jobs.send(Job::Sync(tx)).is_ok() {
            let _ = rx.recv();
        }
        let acks = self.drain_acks();
        self.write_manifest();
        acks
    }

    /// Advisory `manifest.json` (SPEC §6.4: the authoritative index is
    /// always rebuilt from slab headers; this is for humans/debugging).
    fn write_manifest(&self) {
        let mut files: Vec<u32> = self.files.keys().copied().collect();
        files.sort_unstable();
        let entries: Vec<String> = files
            .iter()
            .map(|id| {
                format!(
                    "{{\"file\":\"{}\",\"slots\":{}}}",
                    file_name(*id),
                    self.files[id].slots_used
                )
            })
            .collect();
        let manifest = format!(
            "{{\"version\":{VERSION},\"blocks\":{},\"bytes\":{},\"files\":[{}]}}\n",
            self.index.len(),
            self.bytes,
            entries.join(",")
        );
        let _ = std::fs::write(self.dir.join("manifest.json"), manifest);
    }

    /// Startup index scan (SPEC §6.4): trust nothing but slab headers.
    fn scan(&mut self) -> std::io::Result<()> {
        let mut ids: Vec<u32> = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(id) = parse_file_name(name) {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        for id in ids {
            self.next_file = self.next_file.max(id + 1);
            if !self.scan_file(id) {
                self.counters.fingerprint_rejects_total += 1;
            }
        }
        Ok(())
    }

    /// Returns false when the file was rejected wholesale.
    fn scan_file(&mut self, id: u32) -> bool {
        let path = self.dir.join(file_name(id));
        let Ok(mut file) = File::open(&path) else {
            return false;
        };
        let mut header = [0_u8; HEADER_BYTES];
        if file.read_exact(&mut header).is_err() {
            return false;
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
        };
        let g = self.geometry;
        if &header[0..8] != MAGIC
            || u32_at(8) != VERSION
            || u32_at(12) != g.layers
            || u32_at(16) != g.kv_heads
            || u32_at(20) != g.head_dim
            || u32_at(24) != g.block_size
            || u32_at(32) != SLOTS_PER_FILE
            || header[36..68] != self.fingerprint[..]
        {
            return false;
        }
        let dtype_tag = u32_at(28);
        // dtype_size is implied by the header's geometry contract; stored
        // right after the fingerprint block for forward compatibility.
        let dtype_size = u32_at(68);
        if dtype_size == 0 || dtype_size > 8 {
            return false;
        }
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut slots_used = 0;
        for slot in 0..SLOTS_PER_FILE {
            let offset = g.slot_offset(dtype_size, slot);
            let end = offset + g.slot_stride(dtype_size);
            if end > file_len {
                break;
            }
            let mut slot_header = vec![0_u8; g.slot_header_bytes() as usize];
            if file.read_exact_at(&mut slot_header, offset).is_err() {
                break;
            }
            if slot_header[0] != 1 {
                continue;
            }
            let mut hash = [0_u8; 32];
            hash.copy_from_slice(&slot_header[8..40]);
            slots_used += 1;
            // Later files win on duplicates (ids ascend with time).
            self.index.insert(
                hash,
                SlotRef {
                    file: id,
                    slot,
                    dtype_tag,
                    dtype_size,
                },
            );
        }
        self.clock += 1;
        self.files.insert(
            id,
            FileMeta {
                slots_used,
                bytes: file_len,
                last_touch: self.clock,
                ..FileMeta::default()
            },
        );
        self.bytes += file_len;
        true
    }
}

impl Drop for SsdStore {
    fn drop(&mut self) {
        // Closing the job channel ends the writer loop; join so queued
        // writes land before the process moves on (worker shutdown).
        let (tx, _rx) = channel();
        drop(std::mem::replace(&mut self.jobs, tx));
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

fn file_name(id: u32) -> String {
    format!("slab-{id:08}.kiln")
}

fn parse_file_name(name: &str) -> Option<u32> {
    let id = name.strip_prefix("slab-")?.strip_suffix(".kiln")?;
    (id.len() == 8).then(|| id.parse().ok())?
}

fn file_header(
    fingerprint: &ChainHash,
    g: &SlabGeometry,
    dtype_tag: u32,
    dtype_size: u32,
) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(MAGIC);
    for value in [
        VERSION,
        g.layers,
        g.kv_heads,
        g.head_dim,
        g.block_size,
        dtype_tag,
        SLOTS_PER_FILE,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    header.extend_from_slice(fingerprint);
    header.extend_from_slice(&dtype_size.to_le_bytes());
    header.resize(HEADER_BYTES, 0);
    header
}

/// The writer thread: owns the file handles, performs creates/writes/
/// deletes in order, acks every write. IO failures are acked `ok=false`
/// (silent-skip policy); the loop itself never panics.
fn writer_main(dir: PathBuf, jobs: Receiver<Job>, acks: Sender<FlushAck>) {
    let mut handles: HashMap<u32, File> = HashMap::new();
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Create { file, header } => {
                let path = dir.join(file_name(file));
                match File::create(&path) {
                    Ok(handle) => {
                        let _ = handle.write_all_at(&header, 0);
                        handles.insert(file, handle);
                    }
                    Err(_) => {
                        // Writes to this file will fail their acks below.
                    }
                }
            }
            Job::Write {
                file,
                offset,
                bytes,
                mut ack,
            } => {
                ack.ok = handles
                    .get(&file)
                    .is_some_and(|handle| handle.write_all_at(&bytes, offset).is_ok());
                let _ = acks.send(ack);
            }
            Job::Delete { file } => {
                handles.remove(&file);
                let _ = std::fs::remove_file(dir.join(file_name(file)));
            }
            Job::Sync(done) => {
                let _ = done.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kiln-ssd-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        dir
    }

    fn ticket(node: usize, generation: u64, hash: ChainHash) -> FlushTicket {
        FlushTicket {
            node,
            generation,
            hash,
        }
    }

    fn geometry() -> SlabGeometry {
        SlabGeometry {
            layers: 2,
            kv_heads: 2,
            head_dim: 4,
            block_size: 4,
        }
    }

    fn payload(seed: u8, g: &SlabGeometry) -> Vec<u8> {
        (0..g.payload_bytes(2)).map(|i| seed ^ (i as u8)).collect()
    }

    #[test]
    fn round_trip_and_restart_scan() {
        let dir = temp_dir("roundtrip");
        let g = geometry();
        let tokens = [1_u32, 2, 3, 4];
        let hash = [9_u8; 32];
        let body = payload(0x5a, &g);
        {
            let mut store = SsdStore::open(&dir, [1; 32], g, 1 << 20).expect("open");
            store.enqueue_write(ticket(5, 42, hash), &tokens, body.clone(), 3, 2);
            let acks = store.sync();
            assert_eq!(acks.len(), 1);
            assert!(acks[0].ok);
            assert_eq!((acks[0].node, acks[0].generation), (5, 42));
            assert!(store.contains(&hash));
            assert_eq!(store.read(&hash, &tokens, Some(3)), Some((body.clone(), 3)));
            // Wrong pool dtype or wrong tokens: silent miss + counter.
            assert_eq!(store.read(&hash, &tokens, Some(4)), None);
            assert_eq!(store.counters().fingerprint_rejects_total, 1);
        }
        // Restart: rebuilt from headers alone; a fresh pool (no dtype yet)
        // accepts the slab's own tag.
        let mut store = SsdStore::open(&dir, [1; 32], g, 1 << 20).expect("reopen");
        assert_eq!(store.blocks_stored(), 1);
        assert_eq!(store.read(&hash, &tokens, None), Some((body, 3)));
        // The token verification catches a mismatched request chunk.
        assert_eq!(store.read(&hash, &[1, 2, 3, 5], Some(3)), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_header_and_wrong_fingerprint_are_skipped() {
        let dir = temp_dir("corrupt");
        let g = geometry();
        let hash = [7_u8; 32];
        {
            let mut store = SsdStore::open(&dir, [1; 32], g, 1 << 20).expect("open");
            store.enqueue_write(ticket(0, 1, hash), &[1, 2, 3, 4], payload(1, &g), 3, 2);
            store.sync();
        }
        // Different fingerprint: whole file ignored, counter bumped.
        let store = SsdStore::open(&dir, [2; 32], g, 1 << 20).expect("open other model");
        assert_eq!(store.blocks_stored(), 0);
        assert_eq!(store.counters().fingerprint_rejects_total, 1);
        drop(store);
        // Flip a magic byte: same silent skip for the right model.
        let path = dir.join(file_name(0));
        let mut bytes = std::fs::read(&path).expect("read slab");
        bytes[0] ^= 0xff;
        std::fs::write(&path, &bytes).expect("corrupt slab");
        let store = SsdStore::open(&dir, [1; 32], g, 1 << 20).expect("reopen");
        assert_eq!(store.blocks_stored(), 0);
        assert_eq!(store.counters().fingerprint_rejects_total, 1);
        drop(store);
        // Torn payload: header scans fine, read fails the digest.
        bytes[0] ^= 0xff; // restore magic
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, &bytes).expect("tear payload");
        let mut store = SsdStore::open(&dir, [1; 32], g, 1 << 20).expect("reopen");
        assert_eq!(store.blocks_stored(), 1);
        assert_eq!(store.read(&hash, &[1, 2, 3, 4], Some(3)), None);
        assert_eq!(store.counters().fingerprint_rejects_total, 1);
        assert!(!store.contains(&hash), "failed read drops the entry");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn byte_cap_drops_lru_slabs() {
        let dir = temp_dir("cap");
        let g = geometry();
        let stride = g.slot_stride(2);
        // Room for one slab of SLOTS_PER_FILE plus change; writing a second
        // slab's worth must drop the first file.
        let cap = HEADER_BYTES as u64 + stride * u64::from(SLOTS_PER_FILE) + stride;
        let mut store = SsdStore::open(&dir, [1; 32], g, cap).expect("open");
        let hash_of = |i: u32| {
            let mut hash = [0_u8; 32];
            hash[..4].copy_from_slice(&i.to_le_bytes());
            hash
        };
        for i in 0..SLOTS_PER_FILE {
            store.enqueue_write(
                ticket(i as usize, 1, hash_of(i)),
                &[i, i, i, i],
                payload(i as u8, &g),
                3,
                2,
            );
            store.sync();
        }
        assert_eq!(store.blocks_stored(), u64::from(SLOTS_PER_FILE));
        assert!(store.bytes_stored() <= cap);
        // One more block starts slab 1; the cap evicts slab 0 wholesale.
        store.enqueue_write(
            ticket(99, 1, hash_of(999)),
            &[9, 9, 9, 9],
            payload(0xaa, &g),
            3,
            2,
        );
        store.sync();
        assert!(store.contains(&hash_of(999)));
        assert!(!store.contains(&hash_of(0)), "old slab evicted");
        assert!(store.bytes_stored() <= cap);
        assert!(!dir.join(file_name(0)).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
