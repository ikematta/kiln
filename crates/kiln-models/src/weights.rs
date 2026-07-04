//! safetensors weight loading (SPEC §3: `safetensors` crate, mmap).
//!
//! Files are mmapped via `kiln_mlx::io::MappedFile` (the mmap is the one
//! unsafe operation, confined to kiln-mlx) and parsed zero-copy; each tensor
//! is then copied once into an owned MLX array in its stored dtype.
//!
//! Sharded checkpoints are handled through `model.safetensors.index.json`
//! (`weight_map`); single-file checkpoints load `model.safetensors` directly.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

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
            index
                .get("weight_map")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| WeightsError::Index {
                    path: index_path.display().to_string(),
                    message: "no weight_map object".to_owned(),
                })?
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else {
            BTreeSet::from(["model.safetensors".to_owned()])
        };

        let mut tensors = HashMap::new();
        for file in files {
            let path = dir.join(&file);
            let mapped =
                kiln_mlx::io::MappedFile::open(&path).map_err(|source| WeightsError::Io {
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
