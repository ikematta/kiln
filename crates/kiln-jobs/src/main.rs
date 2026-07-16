#![deny(unsafe_code)]
//! `kiln-jobs`: download/quantize job runner (SPEC §9.1).
//!
//! Usage:
//!   kiln-jobs download <hf_repo> [--revision <rev>] [--dest <dir>]
//!                      [--dest-root <dir>] [--db <path>]
//!   kiln-jobs quantize <path> [--bits <n>] [--group-size <n>] [--out <dir>]
//!                      [--dest-root <dir>] [--venv <dir>] [--db <path>]
//!   kiln-jobs serve --socket <path> [--db <path>] [--dest-root <dir>]
//!                      [--venv <dir>]
//!
//! Progress goes to stdout as JSON lines; logs go to stderr. Every job is
//! recorded in the SQLite store (default `~/.kiln/jobs.sqlite`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use kiln_jobs::runner::RunnerConfig;
use kiln_jobs::store::{JobKind, JobState, JobStore};
use kiln_jobs::{expand_tilde, quantize, runner, serve};

const DEFAULT_DB: &str = "~/.kiln/jobs.sqlite";
const DEFAULT_DEST_ROOT: &str = "~/.kiln/models";

const USAGE: &str = "usage:
  kiln-jobs download <hf_repo> [--revision <rev>] [--dest <dir>] [--dest-root <dir>] [--db <path>]
  kiln-jobs quantize <path> [--bits <n>] [--group-size <n>] [--out <dir>] [--dest-root <dir>] [--venv <dir>] [--db <path>]
  kiln-jobs serve --socket <path> [--db <path>] [--dest-root <dir>] [--venv <dir>]";

/// Flags shared by every subcommand, parsed by simple key/value scanning.
#[derive(Debug, Default)]
struct Flags {
    positional: Vec<String>,
    revision: Option<String>,
    dest: Option<PathBuf>,
    dest_root: Option<PathBuf>,
    db: Option<PathBuf>,
    bits: Option<u32>,
    group_size: Option<u32>,
    out: Option<PathBuf>,
    venv: Option<PathBuf>,
    socket: Option<PathBuf>,
}

fn parse_flags(args: impl Iterator<Item = String>) -> Result<Flags, String> {
    let mut flags = Flags::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--revision" => flags.revision = Some(value("--revision")?),
            "--dest" => flags.dest = Some(PathBuf::from(value("--dest")?)),
            "--dest-root" => flags.dest_root = Some(PathBuf::from(value("--dest-root")?)),
            "--db" => flags.db = Some(PathBuf::from(value("--db")?)),
            "--out" => flags.out = Some(PathBuf::from(value("--out")?)),
            "--venv" => flags.venv = Some(PathBuf::from(value("--venv")?)),
            "--socket" => flags.socket = Some(PathBuf::from(value("--socket")?)),
            "--bits" => {
                flags.bits = Some(
                    value("--bits")?
                        .parse()
                        .map_err(|_| "--bits must be an integer".to_string())?,
                )
            }
            "--group-size" => {
                flags.group_size = Some(
                    value("--group-size")?
                        .parse()
                        .map_err(|_| "--group-size must be an integer".to_string())?,
                )
            }
            other if other.starts_with("--") => return Err(format!("unknown flag '{other}'")),
            other => flags.positional.push(other.to_string()),
        }
    }
    Ok(flags)
}

impl Flags {
    fn runner_config(&self) -> RunnerConfig {
        RunnerConfig {
            dest_root: expand_tilde(
                self.dest_root
                    .as_deref()
                    .unwrap_or(Path::new(DEFAULT_DEST_ROOT)),
            ),
            venv: self
                .venv
                .clone()
                .unwrap_or_else(|| PathBuf::from(quantize::DEFAULT_VENV)),
        }
    }

    fn open_store(&self) -> Result<Arc<JobStore>, String> {
        let path = expand_tilde(self.db.as_deref().unwrap_or(Path::new(DEFAULT_DB)));
        JobStore::open(&path)
            .map(Arc::new)
            .map_err(|err| format!("failed to open job store at {}: {err}", path.display()))
    }
}

fn main() -> ExitCode {
    init_tracing();
    let mut args = std::env::args().skip(1);
    let command = args.next();
    let flags = match parse_flags(args) {
        Ok(flags) => flags,
        Err(message) => return usage_error(&message),
    };
    match command.as_deref() {
        Some("download") => run(flags, JobKind::Download),
        Some("quantize") => run(flags, JobKind::Quantize),
        Some("serve") => run_serve(flags),
        Some(other) => usage_error(&format!("unknown command '{other}'")),
        None => usage_error("a command is required"),
    }
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!("kiln-jobs: {message}\n{USAGE}");
    ExitCode::from(2)
}

/// One-shot CLI job: record it, run it, exit by terminal state.
fn run(flags: Flags, kind: JobKind) -> ExitCode {
    let config = flags.runner_config();
    let job_spec = match kind {
        JobKind::Download => {
            let [repo] = flags.positional.as_slice() else {
                return usage_error("download takes exactly one <hf_repo>");
            };
            runner::download_job(repo, flags.revision.as_deref(), flags.dest.clone(), &config)
                .map(|job| runner::to_spec_json(&job))
        }
        JobKind::Quantize => {
            let [src] = flags.positional.as_slice() else {
                return usage_error("quantize takes exactly one <path>");
            };
            runner::quantize_job(
                src,
                flags.bits,
                flags.group_size,
                flags.out.clone(),
                &config,
            )
            .map(|job| runner::to_spec_json(&job))
        }
    };
    let spec_json = match job_spec {
        Ok(spec) => spec,
        Err(err) => return usage_error(&err.to_string()),
    };
    let store = match flags.open_store() {
        Ok(store) => store,
        Err(message) => {
            eprintln!("kiln-jobs: {message}");
            return ExitCode::FAILURE;
        }
    };
    let record = match store.insert(kind, &spec_json) {
        Ok(record) => record,
        Err(err) => {
            eprintln!("kiln-jobs: failed to record job: {err}");
            return ExitCode::FAILURE;
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("kiln-jobs: failed to start tokio runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    let state = runtime.block_on(runner::execute(&store, &record, &config, true));
    match state {
        JobState::Succeeded => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

fn run_serve(flags: Flags) -> ExitCode {
    let Some(socket) = flags.socket.clone() else {
        return usage_error("serve requires --socket <path>");
    };
    let config = flags.runner_config();
    let store = match flags.open_store() {
        Ok(store) => store,
        Err(message) => {
            eprintln!("kiln-jobs: {message}");
            return ExitCode::FAILURE;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("kiln-jobs: failed to start tokio runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(serve::serve(&socket, store, config)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("kiln-jobs: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Structured logs to STDERR — stdout is reserved for progress JSON lines
/// (SPEC §9.1).
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .with_current_span(false)
        .with_writer(std::io::stderr)
        .init();
}
