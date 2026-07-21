//! `kiln.toml` parsing (SPEC §10) via figment: TOML file merged with
//! `KILN_`-prefixed environment overrides (`__` separates nesting, e.g.
//! `KILN_SERVER__PORT=9090`).
//!
//! Directory paths are kept verbatim (`~` is not expanded here); expansion
//! happens where the paths are used, from Phase 2 onward.

use std::path::{Path, PathBuf};

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use serde::Deserialize;

/// Env-addressable top-level sections. `[[model]]` entries come from the file
/// only, and unrelated `KILN_*` variables (e.g. `KILN_TEST_MODELS`, used by
/// the test suite) must not leak into config keys.
const ENV_SECTIONS: &[&str] = &["SERVER__", "MEMORY__", "DEFAULTS__", "AUTH__"];

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to load configuration: {0}")]
    Load(#[source] Box<figment::Error>),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

impl From<figment::Error> for ConfigError {
    fn from(err: figment::Error) -> Self {
        Self::Load(Box::new(err))
    }
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct KilnConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub defaults: EngineDefaults,
    #[serde(default, rename = "model")]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "defaults::host")]
    pub host: String,
    #[serde(default = "defaults::port")]
    pub port: u16,
    #[serde(default = "defaults::runtime_dir")]
    pub runtime_dir: PathBuf,
    #[serde(default = "defaults::cache_dir")]
    pub cache_dir: PathBuf,
    #[serde(default = "defaults::model_dir")]
    pub model_dir: PathBuf,
    /// Command prefix the supervisor uses to launch the Python worker; the
    /// gateway appends `--model <path> --socket <path> --model-id <id>`.
    /// The default assumes the gateway runs from a Kiln checkout (Phase 2
    /// tracer bullet); packaged installs override it (SPEC §12 Phase 10).
    #[serde(default = "defaults::python_worker_argv")]
    pub python_worker_argv: Vec<String>,
    /// Command prefix for the Rust worker (same appended args). Default: a
    /// `kiln-worker` binary next to the running gateway binary — true for
    /// cargo target dirs and packaged installs alike.
    #[serde(default = "defaults::rust_worker_argv")]
    pub rust_worker_argv: Vec<String>,
    /// Command prefix for the job runner; the gateway appends
    /// `serve --socket <path> --db <path> --dest-root <model_dir>` when it
    /// spawns kiln-jobs on demand for `/admin/jobs/*` (SPEC §2.1, §9.1).
    /// Default: a `kiln-jobs` binary next to the running gateway binary.
    #[serde(default = "defaults::jobs_argv")]
    pub jobs_argv: Vec<String>,
    /// SQLite job store handed to the spawned kiln-jobs server.
    #[serde(default = "defaults::jobs_db")]
    pub jobs_db: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MemoryConfig {
    /// Fraction of unified memory the gateway may hand out across workers.
    #[serde(default = "defaults::budget_fraction")]
    pub budget_fraction: f64,
    /// Explicit machine budget in bytes; overrides `budget_fraction`
    /// (operator caps and eviction tests need an exact number).
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EngineDefaults {
    #[serde(default = "defaults::max_batch_tokens")]
    pub max_batch_tokens: u32,
    #[serde(default = "defaults::prefill_chunk")]
    pub prefill_chunk: u32,
    /// Tokens per KV block; must be a power of two (SPEC §6.3).
    #[serde(default = "defaults::block_size")]
    pub block_size: u32,
    #[serde(default = "defaults::ssd_cache_max_gb")]
    pub ssd_cache_max_gb: u64,
    #[serde(default = "defaults::ssd_tier")]
    pub ssd_tier: bool,
    /// Custom block-table-aware paged-attention kernel for decode steps
    /// (SPEC §7.4 Phase 7). Opt-in: the gather path is the default until
    /// the kernel's parity/throughput bars are proven on the serving
    /// device. Rust workers only.
    #[serde(default = "defaults::paged_attention_kernel")]
    pub paged_attention_kernel: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerKind {
    Rust,
    Python,
    /// Rust if the architecture is supported and the quant format is known,
    /// else Python (SPEC §10).
    #[default]
    Auto,
}

impl WorkerKind {
    /// The kiln.toml spelling (the serde `lowercase` names), used when the
    /// add-model flow writes a `[[model]]` block back to disk.
    pub fn as_config_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    /// Hugging Face repo id or local path.
    pub path: String,
    #[serde(default)]
    pub worker: WorkerKind,
    /// Never evicted under memory pressure.
    #[serde(default)]
    pub pinned: bool,
    /// Idle auto-unload; 0 = never.
    #[serde(default)]
    pub ttl_seconds: u64,
    pub speculative: Option<SpeculativeConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft model: HF repo id or local path.
    pub draft: String,
    /// Tokens proposed per speculation round.
    #[serde(default = "defaults::gamma")]
    pub gamma: u32,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct AuthConfig {
    /// argon2 hash of the admin bearer token; never the raw token.
    pub admin_token_hash: Option<String>,
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ApiKeyConfig {
    pub name: String,
    /// argon2 hash of the API key.
    pub key_hash: String,
    /// Requests/minute; absent = unlimited.
    pub rpm: Option<u32>,
    /// Tokens/minute; absent = unlimited.
    pub tpm: Option<u32>,
}

impl KilnConfig {
    /// Loads configuration from `path` (which must exist), then applies
    /// `KILN_`-prefixed environment overrides, then validates.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config: Self = Figment::new()
            .merge(Toml::file_exact(path.as_ref()))
            .merge(Self::env_provider())
            .extract()?;
        config.validate()?;
        Ok(config)
    }

    /// Like [`KilnConfig::load`], but with no config file: built-in
    /// defaults plus `KILN_` environment overrides. Used by the `kiln`
    /// CLI when no kiln.toml is found (e.g. `kiln models` against the
    /// default localhost address).
    pub fn load_env_only() -> Result<Self, ConfigError> {
        let config: Self = Figment::new().merge(Self::env_provider()).extract()?;
        config.validate()?;
        Ok(config)
    }

    /// Parses and validates TOML text with NO environment overrides: the
    /// re-verification gate the add-model config write runs before it
    /// replaces kiln.toml on disk.
    pub fn parse_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = Figment::new().merge(Toml::string(text)).extract()?;
        config.validate()?;
        Ok(config)
    }

    fn env_provider() -> Env {
        Env::prefixed("KILN_")
            .filter(|key| {
                let key = key.as_str();
                ENV_SECTIONS.iter().any(|section| {
                    key.len() > section.len() && key[..section.len()].eq_ignore_ascii_case(section)
                })
            })
            .split("__")
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |msg: String| Err(ConfigError::Invalid(msg));

        if self.server.port == 0 {
            return invalid("server.port must be non-zero".into());
        }
        if self.server.python_worker_argv.is_empty() || self.server.python_worker_argv[0].is_empty()
        {
            return invalid("server.python_worker_argv must name an executable".into());
        }
        if self.server.rust_worker_argv.is_empty() || self.server.rust_worker_argv[0].is_empty() {
            return invalid("server.rust_worker_argv must name an executable".into());
        }
        if self.server.jobs_argv.is_empty() || self.server.jobs_argv[0].is_empty() {
            return invalid("server.jobs_argv must name an executable".into());
        }
        let fraction = self.memory.budget_fraction;
        if !(fraction > 0.0 && fraction <= 1.0) {
            return invalid(format!(
                "memory.budget_fraction must be in (0, 1], got {fraction}"
            ));
        }
        if self.memory.budget_bytes == Some(0) {
            return invalid("memory.budget_bytes must be non-zero when set".into());
        }
        let block_size = self.defaults.block_size;
        if !block_size.is_power_of_two() {
            return invalid(format!(
                "defaults.block_size must be a power of two, got {block_size}"
            ));
        }
        if self.defaults.prefill_chunk == 0 {
            return invalid("defaults.prefill_chunk must be non-zero".into());
        }
        if self.defaults.max_batch_tokens < self.defaults.block_size {
            return invalid(format!(
                "defaults.max_batch_tokens ({}) must be >= defaults.block_size ({block_size})",
                self.defaults.max_batch_tokens
            ));
        }

        let mut seen = std::collections::HashSet::new();
        for model in &self.models {
            if model.id.is_empty() {
                return invalid("model.id must be non-empty".into());
            }
            if model.path.is_empty() {
                return invalid(format!("model '{}': path must be non-empty", model.id));
            }
            if !seen.insert(model.id.as_str()) {
                return invalid(format!("duplicate model id '{}'", model.id));
            }
            if let Some(spec) = &model.speculative {
                if spec.draft.is_empty() {
                    return invalid(format!(
                        "model '{}': speculative.draft must be non-empty",
                        model.id
                    ));
                }
                if spec.gamma == 0 {
                    return invalid(format!(
                        "model '{}': speculative.gamma must be >= 1",
                        model.id
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: defaults::host(),
            port: defaults::port(),
            runtime_dir: defaults::runtime_dir(),
            cache_dir: defaults::cache_dir(),
            model_dir: defaults::model_dir(),
            python_worker_argv: defaults::python_worker_argv(),
            rust_worker_argv: defaults::rust_worker_argv(),
            jobs_argv: defaults::jobs_argv(),
            jobs_db: defaults::jobs_db(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            budget_fraction: defaults::budget_fraction(),
            budget_bytes: None,
        }
    }
}

impl Default for EngineDefaults {
    fn default() -> Self {
        Self {
            max_batch_tokens: defaults::max_batch_tokens(),
            prefill_chunk: defaults::prefill_chunk(),
            block_size: defaults::block_size(),
            ssd_cache_max_gb: defaults::ssd_cache_max_gb(),
            ssd_tier: defaults::ssd_tier(),
            paged_attention_kernel: defaults::paged_attention_kernel(),
        }
    }
}

/// SPEC §10 default values, shared by `serde(default = ...)` and `Default`.
mod defaults {
    use std::path::PathBuf;

    pub(super) fn host() -> String {
        "127.0.0.1".to_string()
    }
    pub(super) fn port() -> u16 {
        8080
    }
    pub(super) fn runtime_dir() -> PathBuf {
        PathBuf::from("~/.kiln/run")
    }
    pub(super) fn cache_dir() -> PathBuf {
        PathBuf::from("~/.kiln/cache")
    }
    pub(super) fn model_dir() -> PathBuf {
        PathBuf::from("~/.kiln/models")
    }
    pub(super) fn python_worker_argv() -> Vec<String> {
        [
            "uv",
            "run",
            "--project",
            "python/kiln_worker_py",
            "python",
            "-m",
            "kiln_worker_py",
        ]
        .map(String::from)
        .to_vec()
    }
    pub(super) fn rust_worker_argv() -> Vec<String> {
        // Sibling of the gateway binary; falls back to $PATH lookup.
        vec![sibling_binary("kiln-worker")]
    }
    pub(super) fn jobs_argv() -> Vec<String> {
        vec![sibling_binary("kiln-jobs")]
    }
    pub(super) fn jobs_db() -> PathBuf {
        PathBuf::from("~/.kiln/jobs.sqlite")
    }
    fn sibling_binary(name: &str) -> String {
        std::env::current_exe()
            .ok()
            .and_then(|exe| Some(exe.parent()?.join(name)))
            .filter(|path| path.is_file())
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.to_string())
    }
    pub(super) fn budget_fraction() -> f64 {
        0.80
    }
    pub(super) fn max_batch_tokens() -> u32 {
        8192
    }
    pub(super) fn prefill_chunk() -> u32 {
        2048
    }
    pub(super) fn block_size() -> u32 {
        32
    }
    pub(super) fn ssd_cache_max_gb() -> u64 {
        64
    }
    pub(super) fn ssd_tier() -> bool {
        true
    }
    pub(super) fn paged_attention_kernel() -> bool {
        false
    }
    pub(super) fn gamma() -> u32 {
        4
    }
}

#[cfg(test)]
// `figment::Jail::expect_with` fixes the closure return type to
// `Result<(), figment::Error>`, whose Err variant clippy considers large.
#[allow(clippy::result_large_err)]
mod tests {
    use super::*;

    fn example_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kiln.toml.example")
    }

    #[test]
    fn parses_the_committed_example_file() {
        // Jail isolates the process env so stray KILN_* vars can't interfere.
        figment::Jail::expect_with(|_| {
            let config = KilnConfig::load(example_path()).expect("example must parse");

            assert_eq!(config.server.host, "127.0.0.1");
            assert_eq!(config.server.port, 8080);
            assert_eq!(config.server.model_dir, PathBuf::from("~/.kiln/models"));
            assert_eq!(config.memory.budget_fraction, 0.80);
            assert_eq!(config.defaults.max_batch_tokens, 8192);
            assert!(config.defaults.ssd_tier);

            assert_eq!(config.models.len(), 1);
            let model = &config.models[0];
            assert_eq!(model.id, "qwen3-14b-4bit");
            assert_eq!(model.worker, WorkerKind::Auto);
            assert!(model.pinned);
            assert_eq!(model.ttl_seconds, 0);
            let spec = model.speculative.as_ref().expect("speculative block");
            assert_eq!(spec.draft, "mlx-community/Qwen3-0.6B-4bit");
            assert_eq!(spec.gamma, 4);

            assert_eq!(config.auth.api_keys.len(), 1);
            assert_eq!(config.auth.api_keys[0].rpm, Some(600));
            assert_eq!(config.auth.api_keys[0].tpm, Some(500_000));
            Ok(())
        });
    }

    #[test]
    fn empty_file_yields_spec_defaults() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "")?;
            let config = KilnConfig::load("kiln.toml").expect("empty config is valid");
            assert_eq!(config, KilnConfig::default());
            assert_eq!(config.server.port, 8080);
            assert_eq!(config.memory.budget_fraction, 0.80);
            assert_eq!(config.defaults.block_size, 32);
            assert!(config.models.is_empty());
            Ok(())
        });
    }

    #[test]
    fn python_worker_argv_defaults_and_rejects_empty() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "")?;
            let config = KilnConfig::load("kiln.toml").expect("defaults valid");
            assert_eq!(config.server.python_worker_argv[0], "uv");

            jail.create_file("bad.toml", "[server]\npython_worker_argv = []\n")?;
            let err = KilnConfig::load("bad.toml").unwrap_err();
            assert!(matches!(err, ConfigError::Invalid(_)));
            Ok(())
        });
    }

    #[test]
    fn load_env_only_gives_defaults_plus_env() {
        figment::Jail::expect_with(|jail| {
            let config = KilnConfig::load_env_only().expect("defaults are valid");
            assert_eq!(config, KilnConfig::default());

            jail.set_env("KILN_SERVER__PORT", "9191");
            let config = KilnConfig::load_env_only().expect("env override is valid");
            assert_eq!(config.server.port, 9191);
            Ok(())
        });
    }

    #[test]
    fn env_overrides_file_values() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "[server]\nport = 8080\n")?;
            jail.set_env("KILN_SERVER__PORT", "9090");
            jail.set_env("KILN_MEMORY__BUDGET_FRACTION", "0.5");
            jail.set_env("KILN_DEFAULTS__SSD_TIER", "false");
            let config = KilnConfig::load("kiln.toml").expect("env overrides apply");
            assert_eq!(config.server.port, 9090);
            assert_eq!(config.memory.budget_fraction, 0.5);
            assert!(!config.defaults.ssd_tier);
            Ok(())
        });
    }

    #[test]
    fn unrelated_kiln_env_vars_are_ignored() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "")?;
            // CLAUDE.md's test-model env var must not become a config key.
            jail.set_env("KILN_TEST_MODELS", "/tmp/models");
            let config = KilnConfig::load("kiln.toml").expect("unrelated env ignored");
            assert_eq!(config, KilnConfig::default());
            Ok(())
        });
    }

    #[test]
    fn missing_file_is_an_error() {
        figment::Jail::expect_with(|_| {
            let err = KilnConfig::load("does-not-exist.toml").unwrap_err();
            assert!(matches!(err, ConfigError::Load(_)));
            Ok(())
        });
    }

    #[test]
    fn out_of_range_budget_fraction_is_rejected() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "[memory]\nbudget_fraction = 1.5\n")?;
            let err = KilnConfig::load("kiln.toml").unwrap_err();
            assert!(matches!(err, ConfigError::Invalid(_)));
            Ok(())
        });
    }

    #[test]
    fn non_power_of_two_block_size_is_rejected() {
        figment::Jail::expect_with(|jail| {
            jail.create_file("kiln.toml", "[defaults]\nblock_size = 48\n")?;
            let err = KilnConfig::load("kiln.toml").unwrap_err();
            assert!(matches!(err, ConfigError::Invalid(_)));
            Ok(())
        });
    }

    #[test]
    fn duplicate_model_ids_are_rejected() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "kiln.toml",
                r#"
                [[model]]
                id = "m"
                path = "org/a"
                [[model]]
                id = "m"
                path = "org/b"
                "#,
            )?;
            let err = KilnConfig::load("kiln.toml").unwrap_err();
            assert!(matches!(err, ConfigError::Invalid(_)));
            Ok(())
        });
    }
}
