#![deny(unsafe_code)]
//! `kiln-worker`: the native Rust model worker (SPEC §2.1) — a tonic gRPC
//! server over a Unix domain socket, wiring kiln-models' Llama
//! implementation and kiln-engine's continuous batching loop (paged KV,
//! SPEC §6.2) behind the frozen `worker.proto`.
//!
//! Spawned by the gateway supervisor with the same argv contract as the
//! Python worker: `--model <dir> --socket <path> --model-id <id>`.

#[cfg(feature = "metal")]
mod engine;
#[cfg(feature = "metal")]
mod modelinfo;
#[cfg(feature = "metal")]
mod service;

#[cfg(feature = "metal")]
fn main() -> std::process::ExitCode {
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    const USAGE: &str = "usage: kiln-worker --model <dir> --socket <path> [--model-id <id>] \
                         [--no-prefix-cache] [--ssd-dir <dir>] [--ssd-max-gb <n>] \
                         [--paged-attention-kernel] [--draft-model <dir>]";

    let mut model: Option<PathBuf> = None;
    let mut socket: Option<PathBuf> = None;
    let mut model_id: Option<String> = None;
    let mut opts = engine::EngineOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        let result = match arg.as_str() {
            "--model" => value("--model").map(|v| model = Some(PathBuf::from(v))),
            "--socket" => value("--socket").map(|v| socket = Some(PathBuf::from(v))),
            "--model-id" => value("--model-id").map(|v| model_id = Some(v)),
            // SPEC §10 [defaults]: prefix cache + SSD tier config flags.
            "--no-prefix-cache" => {
                opts.prefix_cache = false;
                Ok(())
            }
            "--ssd-dir" => value("--ssd-dir").map(|v| opts.ssd_dir = Some(PathBuf::from(v))),
            "--ssd-max-gb" => value("--ssd-max-gb").and_then(|v| {
                v.parse::<u64>()
                    .map(|gb| opts.ssd_max_bytes = gb << 30)
                    .map_err(|_| format!("--ssd-max-gb needs an integer, got {v:?}"))
            }),
            // SPEC §7.4: block-table-aware attention kernel (Phase 7),
            // opt-in — the gather path stays the default.
            "--paged-attention-kernel" => {
                opts.paged_attention_kernel = true;
                Ok(())
            }
            // SPEC §6.5: draft model for speculative decoding (Phase 8),
            // loaded alongside the target in this process.
            "--draft-model" => {
                value("--draft-model").map(|v| opts.draft_model = Some(PathBuf::from(v)))
            }
            other => Err(format!("unknown argument {other:?}")),
        };
        if let Err(err) = result {
            eprintln!("kiln-worker: {err}");
            eprintln!("{USAGE}");
            return std::process::ExitCode::FAILURE;
        }
    }
    let (Some(model), Some(socket)) = (model, socket) else {
        eprintln!("{USAGE}");
        return std::process::ExitCode::FAILURE;
    };
    let model_id = model_id.unwrap_or_else(|| {
        model
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_owned())
    });

    match run(model, socket, model_id, opts) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("kiln-worker: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn run(
    model: std::path::PathBuf,
    socket: std::path::PathBuf,
    model_id: String,
    opts: engine::EngineOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use kiln_proto::v1::worker_server::WorkerServer;

    // Startup order per CLAUDE.md: error handler before any MLX work (the
    // engine thread calls init() again — idempotent), fd headroom for
    // sockets and the future mmap'd slab tier.
    kiln_mlx::init();
    match kiln_mlx::os::raise_nofile_limit(4096) {
        Ok(limit) => tracing::debug!(limit, "RLIMIT_NOFILE"),
        Err(err) => tracing::warn!(error = %err, "failed to raise RLIMIT_NOFILE"),
    }

    // Static info is filesystem-only, so GetInfo/Health answer correctly
    // while (and even if) the model load fails on the engine thread.
    let info = modelinfo::read_static_info(&model)?;
    tracing::info!(model = %model_id, path = %model.display(),
        dtype = %info.dtype, "starting worker");

    let shared = Arc::new(engine::Shared::new(model_id, info, &opts));
    let submissions = engine::spawn(model.clone(), Arc::clone(&shared), opts)?;
    let service = service::WorkerService::new(shared, submissions);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        // A stale socket from a crashed predecessor would fail the bind.
        if socket.exists() {
            std::fs::remove_file(&socket)?;
        }
        let listener = tokio::net::UnixListener::bind(&socket)?;
        tracing::info!(socket = %socket.display(), "listening");
        let incoming = async_stream::stream! {
            loop {
                yield listener.accept().await.map(|(stream, _addr)| stream);
            }
        };
        tonic::transport::Server::builder()
            .add_service(WorkerServer::new(service))
            .serve_with_incoming(incoming)
            .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("kiln-worker was built without the `metal` feature; it needs MLX to serve.");
    std::process::exit(1);
}
