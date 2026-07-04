//! Read-only file mapping for weight loading (SPEC §3: safetensors + mmap).
//!
//! Lives here, not in `kiln-models`, because `memmap2::Mmap::map` is an
//! `unsafe` call and all `unsafe` in the workspace is confined to this crate
//! (CLAUDE.md). Consumers get a safe `&[u8]` view.
//!
//! Soundness (memmap2's documented contract): the returned `&[u8]` is only
//! valid while the underlying file's contents are unchanged. Modification by
//! ANY process during the mapping's lifetime is undefined behavior;
//! truncation is the milder case (SIGBUS on access). Kiln upholds this
//! operationally: weight files are written once by the fetch/convert tooling
//! and nothing in Kiln writes into a model directory while a worker is
//! loading — and the exposure window is the load phase only, because
//! `WeightStore` copies every tensor into owned MLX arrays and drops the
//! mapping before serving begins. Mutating weight files under a live loader
//! is out of contract (mlx-lm's own mmap load has the same failure mode).

use std::fs::File;
use std::path::Path;

/// A read-only memory-mapped file.
pub struct MappedFile {
    map: memmap2::Mmap,
}

impl MappedFile {
    /// Maps `path` read-only.
    #[allow(unsafe_code)]
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: shared read-only mapping (PROT_READ); the memmap2 contract
        // additionally requires that no process modifies the file while the
        // mapping is alive, or reads through the `&[u8]` are UB. Upheld
        // operationally, not locally provable: the file is opened read-only
        // and never written through the map, weight files are immutable
        // during a run (written once by fetch/convert tooling), and callers
        // hold the mapping only for the load-and-copy phase (module docs).
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { map })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

impl std::ops::Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.bytes()
    }
}
