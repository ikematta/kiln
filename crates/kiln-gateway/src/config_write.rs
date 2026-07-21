//! kiln.toml persistence for `POST /admin/models`: appends one
//! `[[model]]` block in place via toml_edit, preserving comments,
//! formatting, and every existing entry byte-for-byte.
//!
//! Safety contract (the config file is operator-owned):
//! - The file is read fresh from disk at write time — never from the
//!   config the gateway booted with — so hand edits made while the
//!   service runs are preserved, not clobbered.
//! - Any parse failure aborts loudly before anything touches disk.
//! - A `[[model]]` id already present in the FILE is a conflict (the file
//!   was hand-edited past the gateway's view), never an overwrite.
//! - The edited text must round-trip through the real config parser
//!   ([`KilnConfig::parse_str`]) before it replaces the file; the write
//!   itself is atomic (temp file + rename, original permissions kept).

use std::path::Path;

use toml_edit::{ArrayOfTables, DocumentMut, Item, value};

use crate::config::{KilnConfig, ModelConfig};

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "refusing to edit {path}: it does not parse as TOML ({detail}); \
         fix the file by hand — nothing was written"
    )]
    Unparseable { path: String, detail: String },
    #[error(
        "kiln.toml already contains a [[model]] block with id '{0}' (the file was \
         edited since the gateway started); restart the gateway to pick it up, or \
         choose a different id"
    )]
    DuplicateInFile(String),
    #[error(
        "refusing to write {path}: the edited config failed re-validation ({detail}); \
         nothing was written"
    )]
    Reverify { path: String, detail: String },
    #[error("failed to write {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Appends `model` as a `[[model]]` block to the config file on disk.
/// `speculative` is not part of the add-model surface and is ignored
/// (always absent on runtime-added models).
pub fn append_model(config_path: &Path, model: &ModelConfig) -> Result<(), PersistError> {
    let path_str = config_path.display().to_string();
    // Fresh read: the on-disk file — possibly hand-edited since boot — is
    // the document being extended, not the parsed config in memory.
    let text = std::fs::read_to_string(config_path).map_err(|source| PersistError::Read {
        path: path_str.clone(),
        source,
    })?;
    let mut doc: DocumentMut =
        text.parse()
            .map_err(|err: toml_edit::TomlError| PersistError::Unparseable {
                path: path_str.clone(),
                detail: err.to_string(),
            })?;

    if let Some(models) = doc.get("model").and_then(Item::as_array_of_tables)
        && models
            .iter()
            .any(|table| table.get("id").and_then(Item::as_str) == Some(model.id.as_str()))
    {
        return Err(PersistError::DuplicateInFile(model.id.clone()));
    }

    let mut block = toml_edit::Table::new();
    block["id"] = value(model.id.as_str());
    block["path"] = value(model.path.as_str());
    block["worker"] = value(model.worker.as_config_str());
    block["pinned"] = value(model.pinned);
    block["ttl_seconds"] =
        value(
            i64::try_from(model.ttl_seconds).map_err(|_| PersistError::Reverify {
                path: path_str.clone(),
                detail: format!(
                    "ttl_seconds {} exceeds the TOML integer range",
                    model.ttl_seconds
                ),
            })?,
        );

    let models = doc
        .entry("model")
        .or_insert(Item::ArrayOfTables(ArrayOfTables::new()));
    let Some(models) = models.as_array_of_tables_mut() else {
        return Err(PersistError::Unparseable {
            path: path_str,
            detail: "the `model` key is not an array of [[model]] tables".into(),
        });
    };
    models.push(block);

    let edited = doc.to_string();
    // The gate against corrupting the operator's file: the result must
    // still parse and validate as a KilnConfig before it replaces anything.
    if let Err(err) = KilnConfig::parse_str(&edited) {
        return Err(PersistError::Reverify {
            path: path_str,
            detail: err.to_string(),
        });
    }

    atomic_replace(config_path, &edited).map_err(|source| PersistError::Write {
        path: config_path.display().to_string(),
        source,
    })
}

/// Temp-file + rename so a crash mid-write cannot leave a truncated
/// config; the original file's permissions are carried over (kiln.toml
/// holds credential hashes and may be 0600).
fn atomic_replace(path: &Path, contents: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("toml.kiln-tmp");
    std::fs::write(&tmp, contents)?;
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkerKind;
    use std::path::PathBuf;

    fn temp_config(tag: &str, contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kiln-config-write-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        let path = dir.join("kiln.toml");
        std::fs::write(&path, contents).expect("write config");
        path
    }

    fn model(id: &str) -> ModelConfig {
        ModelConfig {
            id: id.into(),
            path: "/models/somewhere".into(),
            worker: WorkerKind::Auto,
            pinned: true,
            ttl_seconds: 300,
            speculative: None,
        }
    }

    const EXISTING: &str = r#"# Operator notes live here and MUST survive edits.
[server]
port = 9090   # non-default, hand-picked

# The resident model.
[[model]]
id = "resident"
path = "/models/resident"   # local path
worker = "rust"

[[auth.api_keys]]
name = "ops"
key_hash = "x"
"#;

    #[test]
    fn append_preserves_every_existing_line_and_adds_one_block() {
        let path = temp_config("append", EXISTING);
        append_model(&path, &model("added")).expect("append succeeds");
        let after = std::fs::read_to_string(&path).expect("read back");

        // Every original line survives, in order (comments included).
        let mut after_lines = after.lines();
        for line in EXISTING.lines() {
            assert!(
                after_lines.any(|l| l == line),
                "original line lost or reordered: {line:?}\nfull file:\n{after}"
            );
        }
        // The result parses, and only the one model was added.
        let config = KilnConfig::parse_str(&after).expect("edited file is valid");
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.models.len(), 2);
        let added = &config.models[1];
        assert_eq!(added.id, "added");
        assert_eq!(added.path, "/models/somewhere");
        assert_eq!(added.worker, WorkerKind::Auto);
        assert!(added.pinned);
        assert_eq!(added.ttl_seconds, 300);
        assert_eq!(config.auth.api_keys.len(), 1);
        let _ = std::fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn append_works_on_a_config_with_no_model_blocks() {
        let path = temp_config("first", "[server]\nport = 8081\n");
        append_model(&path, &model("first")).expect("append succeeds");
        let config =
            KilnConfig::parse_str(&std::fs::read_to_string(&path).expect("read")).expect("valid");
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "first");
        let _ = std::fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn duplicate_id_in_file_is_a_conflict_and_nothing_is_written() {
        let path = temp_config("dup", EXISTING);
        let err = append_model(&path, &model("resident")).expect_err("duplicate rejected");
        assert!(matches!(err, PersistError::DuplicateInFile(_)), "{err}");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            EXISTING,
            "file must be untouched"
        );
        let _ = std::fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn unparseable_file_fails_loudly_and_is_untouched() {
        let broken = "[server\nport = ???";
        let path = temp_config("broken", broken);
        let err = append_model(&path, &model("m")).expect_err("parse failure");
        assert!(matches!(err, PersistError::Unparseable { .. }), "{err}");
        assert!(err.to_string().contains("nothing was written"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), broken);
        let _ = std::fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn reverify_gate_refuses_a_config_the_parser_would_reject() {
        // Valid TOML, invalid KilnConfig (port 0): the append itself is
        // fine but re-validation must refuse to write the result.
        let path = temp_config("reverify", "[server]\nport = 0\n");
        let err = append_model(&path, &model("m")).expect_err("re-validation");
        assert!(matches!(err, PersistError::Reverify { .. }), "{err}");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "[server]\nport = 0\n"
        );
        let _ = std::fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn missing_file_is_a_read_error() {
        let err =
            append_model(Path::new("/nonexistent/kiln.toml"), &model("m")).expect_err("no file");
        assert!(matches!(err, PersistError::Read { .. }), "{err}");
    }
}
