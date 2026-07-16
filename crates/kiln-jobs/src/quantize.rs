//! Quantization wrapper (SPEC §9.1): shells out to `python -m mlx_lm convert`
//! in the jobs venv (`python/kiln_jobs_py`). v1 never reimplements
//! quantization in Rust — mlx-lm's converter is the ecosystem reference the
//! golden harness is calibrated against — and never monkey-patches it.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};

use crate::events::{Event, Sink};

/// Bits-per-weight values mlx-lm affine quantization supports.
pub const SUPPORTED_BITS: &[u32] = &[2, 3, 4, 6, 8];
/// Group sizes mlx-lm affine quantization supports.
pub const SUPPORTED_GROUP_SIZES: &[u32] = &[32, 64, 128];
pub const DEFAULT_BITS: u32 = 4;
pub const DEFAULT_GROUP_SIZE: u32 = 64;
/// The jobs venv, relative to the working directory (a Kiln checkout);
/// packaged installs override with `--venv`.
pub const DEFAULT_VENV: &str = "python/kiln_jobs_py";

#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("unsupported bits {0}; mlx-lm affine quantization supports {SUPPORTED_BITS:?}")]
    Bits(u32),
    #[error(
        "unsupported group size {0}; mlx-lm affine quantization supports {SUPPORTED_GROUP_SIZES:?}"
    )]
    GroupSize(u32),
    #[error("output path {0} already exists; refusing to overwrite")]
    OutputExists(PathBuf),
    #[error("failed to run the converter ({argv}): {source}")]
    Spawn {
        argv: String,
        #[source]
        source: std::io::Error,
    },
    #[error("converter exited with {status}: {tail}")]
    Converter { status: String, tail: String },
    #[error("converter succeeded but wrote no config.json under {0}")]
    NoConfig(PathBuf),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct ConvertSpec {
    /// Local model directory (or HF id, passed through to mlx_lm).
    pub src: String,
    pub out: PathBuf,
    pub bits: u32,
    pub group_size: u32,
    /// uv project directory of the jobs venv.
    pub venv: PathBuf,
}

pub fn validate(bits: u32, group_size: u32) -> Result<(), QuantizeError> {
    if !SUPPORTED_BITS.contains(&bits) {
        return Err(QuantizeError::Bits(bits));
    }
    if !SUPPORTED_GROUP_SIZES.contains(&group_size) {
        return Err(QuantizeError::GroupSize(group_size));
    }
    Ok(())
}

/// Runs the converter, streaming its output as `Log` events. On success the
/// produced directory contains a parseable model (config.json presence is
/// checked here; the caller decides how much further to validate).
pub async fn run_convert(spec: &ConvertSpec, sink: Arc<dyn Sink>) -> Result<(), QuantizeError> {
    validate(spec.bits, spec.group_size)?;
    if spec.out.exists() {
        return Err(QuantizeError::OutputExists(spec.out.clone()));
    }
    if let Some(parent) = spec.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let argv: Vec<String> = [
        "uv",
        "run",
        "--project",
        &spec.venv.display().to_string(),
        "python",
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        &spec.src,
        "--mlx-path",
        &spec.out.display().to_string(),
        "-q",
        "--q-bits",
        &spec.bits.to_string(),
        "--q-group-size",
        &spec.group_size.to_string(),
    ]
    .map(String::from)
    .to_vec();
    sink.emit(&Event::Log {
        line: format!("running: {}", argv.join(" ")),
    });

    let mut child = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // If kiln-jobs dies, don't leave a converter chewing on the GPU.
        .kill_on_drop(true)
        .spawn()
        .map_err(|source| QuantizeError::Spawn {
            argv: argv.join(" "),
            source,
        })?;

    // Stream both pipes live as Log events, keeping a tail for the error
    // message. The pipes must be drained concurrently with wait() or a
    // chatty converter would deadlock on a full pipe buffer.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_task = tokio::spawn(pump_lines(stdout, Arc::clone(&sink)));
    let stderr_task = tokio::spawn(pump_lines(stderr, Arc::clone(&sink)));

    let status = child.wait().await?;
    let mut tail: Vec<String> = Vec::new();
    for task in [stdout_task, stderr_task] {
        for line in task.await.unwrap_or_default() {
            tail.push(line);
            if tail.len() > 20 {
                tail.remove(0);
            }
        }
    }

    if !status.success() {
        return Err(QuantizeError::Converter {
            status: status.to_string(),
            tail: tail.join(" | "),
        });
    }
    if !spec.out.join("config.json").is_file() {
        return Err(QuantizeError::NoConfig(spec.out.clone()));
    }
    sink.emit(&Event::Done {
        dest: spec.out.display().to_string(),
    });
    Ok(())
}

async fn pump_lines(
    pipe: Option<impl tokio::io::AsyncRead + Unpin>,
    sink: Arc<dyn Sink>,
) -> Vec<String> {
    let Some(pipe) = pipe else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    let mut reader = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        if !line.trim().is_empty() {
            sink.emit(&Event::Log { line: line.clone() });
            lines.push(line);
        }
    }
    lines
}

/// Default output directory: `<dest_root>/<basename(src)>-<bits>bit-g<gs>`.
pub fn default_out(dest_root: &Path, src: &str, bits: u32, group_size: u32) -> PathBuf {
    let base = Path::new(src)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| src.replace('/', "--"));
    dest_root.join(format!("{base}-{bits}bit-g{group_size}"))
}
