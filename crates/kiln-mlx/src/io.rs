//! Read-only file mapping for weight loading (SPEC §3: safetensors + mmap).
//!
//! Lives here, not in `kiln-models`, because `memmap2::Mmap::map` is an
//! `unsafe` call and all `unsafe` in the workspace is confined to this crate
//! (CLAUDE.md). Consumers get a safe `&[u8]` view.
//!
//! Soundness caveat (inherent to mmap): the mapping is only as immutable as
//! the underlying file. Kiln treats model directories as immutable while a
//! worker is loading/serving; a concurrently truncated weights file can fault
//! the process — the same failure mode mlx-lm has.

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
        // SAFETY: read-only private mapping; see module caveat on external
        // file mutation.
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
