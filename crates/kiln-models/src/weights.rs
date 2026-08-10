//! safetensors weight loading (SPEC §3: `safetensors` crate, mmap).
//!
//! Files are mmapped via `kiln_mlx::io::MappedFile` (the mmap is the one
//! unsafe operation, confined to kiln-mlx) and parsed zero-copy; each tensor
//! is then copied once into an owned MLX array in its stored dtype.
//!
//! Sharded checkpoints are handled through `model.safetensors.index.json`
//! (`weight_map`); single-file checkpoints load `model.safetensors` directly.

use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path};

use kiln_mlx::{Array, Dtype};

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid safetensors file {path}: {message}")]
    Parse { path: String, message: String },
    #[error("invalid weight index {path}: {message}")]
    Index { path: String, message: String },
    #[error("weight file {file} resolves outside the model directory {dir}")]
    Escape { dir: String, file: String },
    #[error("tensor {name} has unsupported dtype {dtype}")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("missing tensor {0}")]
    Missing(String),
    #[error(transparent)]
    Mlx(#[from] kiln_mlx::MlxError),
}

/// All tensors of a checkpoint, keyed by safetensors name. Consumers `take`
/// tensors out; whatever remains at the end is reported by [`Self::remaining`]
/// so unexpected/unused weights are caught at load time.
#[derive(Debug, Default)]
pub struct WeightStore {
    tensors: HashMap<String, Array>,
}

impl WeightStore {
    /// Loads every tensor under `dir` per the mlx-lm checkpoint layout.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, WeightsError> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let files: BTreeSet<String> = if index_path.is_file() {
            let text = std::fs::read_to_string(&index_path).map_err(|source| WeightsError::Io {
                path: index_path.display().to_string(),
                source,
            })?;
            let index: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| WeightsError::Index {
                    path: index_path.display().to_string(),
                    message: e.to_string(),
                })?;
            let map = index
                .get("weight_map")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| WeightsError::Index {
                    path: index_path.display().to_string(),
                    message: "no weight_map object".to_owned(),
                })?;
            let index_error = |message: String| WeightsError::Index {
                path: index_path.display().to_string(),
                message,
            };
            let mut files = BTreeSet::new();
            for (tensor, value) in map {
                // A malformed entry is named and fatal rather than skipped:
                // dropping it silently just resurfaces later as a confusing
                // `Missing(tensor)` from whichever loader wanted it.
                let file = value.as_str().ok_or_else(|| {
                    index_error(format!("weight_map entry for {tensor} is not a string"))
                })?;
                if !is_shard_file_name(file) {
                    return Err(index_error(format!(
                        "weight_map entry for {tensor} is not a plain filename: {file}"
                    )));
                }
                files.insert(file.to_owned());
            }
            files
        } else {
            BTreeSet::from(["model.safetensors".to_owned()])
        };

        // Defense in depth for the name check above, which cannot see a
        // symlink: every shard's resolved path must still land under the
        // resolved model directory.
        let root = dir.canonicalize().map_err(|source| WeightsError::Io {
            path: dir.display().to_string(),
            source,
        })?;

        let mut tensors = HashMap::new();
        for file in files {
            let path = dir.join(&file);
            let resolved = path.canonicalize().map_err(|source| WeightsError::Io {
                path: path.display().to_string(),
                source,
            })?;
            if !resolved.starts_with(&root) {
                return Err(WeightsError::Escape {
                    dir: root.display().to_string(),
                    file,
                });
            }
            let mapped =
                kiln_mlx::io::MappedFile::open(&resolved).map_err(|source| WeightsError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
            let parsed = safetensors::SafeTensors::deserialize(mapped.bytes()).map_err(|e| {
                WeightsError::Parse {
                    path: path.display().to_string(),
                    message: e.to_string(),
                }
            })?;
            for (name, view) in parsed.tensors() {
                let dtype = match view.dtype() {
                    safetensors::Dtype::F16 => Dtype::Float16,
                    safetensors::Dtype::BF16 => Dtype::Bfloat16,
                    safetensors::Dtype::F32 => Dtype::Float32,
                    safetensors::Dtype::U32 => Dtype::Uint32,
                    safetensors::Dtype::I32 => Dtype::Int32,
                    other => {
                        return Err(WeightsError::UnsupportedDtype {
                            name,
                            dtype: format!("{other:?}"),
                        });
                    }
                };
                let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
                let array = Array::from_raw_bytes(view.data(), &shape, dtype)?;
                tensors.insert(name, array);
            }
        }
        Ok(Self { tensors })
    }

    /// Removes and returns the named tensor.
    pub fn take(&mut self, name: &str) -> Result<Array, WeightsError> {
        self.tensors
            .remove(name)
            .ok_or_else(|| WeightsError::Missing(name.to_owned()))
    }

    /// Removes and returns the named tensor if present.
    pub fn take_optional(&mut self, name: &str) -> Option<Array> {
        self.tensors.remove(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Names of tensors nobody consumed (sorted, for stable diagnostics).
    pub fn remaining(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

/// True when `file` names a shard sitting directly in the checkpoint
/// directory.
///
/// `model.safetensors.index.json` is authored by whoever published the
/// checkpoint, so its `weight_map` values are untrusted input. `Path::join`
/// adopts an absolute component wholesale and honours `..`, so an
/// unvalidated value can name any file on the host — and its tensors would
/// then be bound into the attacker's model and readable back through the
/// inference API. Real mlx-lm indices only ever hold plain filenames
/// (`model-00001-of-00002.safetensors`), so require exactly that.
fn is_shard_file_name(file: &str) -> bool {
    let mut components = Path::new(file).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_shard_names_accepted() {
        assert!(is_shard_file_name("model.safetensors"));
        assert!(is_shard_file_name("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn escaping_shard_names_rejected() {
        for hostile in [
            "../../other-model/model.safetensors",
            "/Users/op/private-models/proprietary/model.safetensors",
            "sub/model.safetensors",
            "./model.safetensors",
            "..",
            "",
        ] {
            assert!(
                !is_shard_file_name(hostile),
                "weight_map value {hostile:?} must be rejected"
            );
        }
    }
}
