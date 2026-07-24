#![deny(unsafe_code)]
//! `kiln-gateway`: HTTP front end, OpenAI adapter, router, and worker
//! supervisor (SPEC §8).
//!
//! Usage:
//!   kiln-gateway [--config kiln.toml]   serve
//!   kiln-gateway hash-key [key]         print the argon2 hash for a key
//!                                       (reads stdin when no arg is given)

use std::process::ExitCode;
use std::sync::Arc;

use kiln_gateway::app::{self, AppState};
use kiln_gateway::auth::Auth;
use kiln_gateway::config::KilnConfig;
use kiln_gateway::metrics::Metrics;
use kiln_gateway::registry::expand_tilde;
use kiln_gateway::supervisor::Supervisor;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("hash-key") => hash_key(args.next()),
        Some("--config") => match args.next() {
            Some(path) => serve(path),
            None => usage_error("--config requires a path"),
        },
        None => serve("kiln.toml".to_string()),
        Some(other) => usage_error(&format!("unknown argument '{other}'")),
    }
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!(
        "kiln-gateway: {message}\nusage: kiln-gateway [--config kiln.toml] | kiln-gateway hash-key [key]"
    );
    ExitCode::from(2)
}

fn serve(config_path: String) -> ExitCode {
    init_tracing();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("kiln-gateway: failed to start tokio runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(&config_path)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!(error = %err, "gateway exiting with error");
            eprintln!("kiln-gateway: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = KilnConfig::load(config_path)?;

    let runtime_dir = expand_tilde(&config.server.runtime_dir);
    tokio::fs::create_dir_all(&runtime_dir).await?;

    // Auth config errors (malformed admin_token_hash) abort before any
    // worker process is spawned.
    let auth = Auth::from_config(&config.auth)?;
    let metrics = Arc::new(Metrics::new()?);
    let (registry, lifecycle, supervisor) = Supervisor::start(&config, Arc::clone(&metrics))?;
    if registry.is_empty() {
        tracing::warn!("no [[model]] entries configured; chat endpoints will 404");
    }
    let jobs = kiln_gateway::admin::JobsProxy::from_config(&config)?;
    let registrar = kiln_gateway::admin_register::Registrar::new(
        supervisor.spawner(),
        &config,
        std::path::PathBuf::from(config_path),
    )?;
    // Signals never-ending responses (the stats SSE stream) to finish so
    // axum's graceful connection drain can complete.
    let (http_shutdown_tx, http_shutdown_rx) = tokio::sync::watch::channel(false);
    let state = Arc::new(AppState {
        registry,
        lifecycle,
        metrics,
        auth,
        rate: kiln_gateway::ratelimit::RateLimiter::from_config(&config.auth),
        jobs,
        registrar,
        shutdown: http_shutdown_rx,
    });

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, config = %config_path, "kiln-gateway listening");

    axum::serve(listener, app::router(state))
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            let _ = http_shutdown_tx.send(true);
        })
        .await?;

    tracing::info!("http server stopped; shutting down workers");
    supervisor.shutdown().await;
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let ctrl_c = tokio::signal::ctrl_c();
    match signal(SignalKind::terminate()) {
        Ok(mut sigterm) => {
            tokio::select! {
                _ = ctrl_c => {}
                _ = sigterm.recv() => {}
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "SIGTERM handler unavailable; ctrl-c only");
            let _ = ctrl_c.await;
        }
    }
}

/// Structured JSON logs to stdout (SPEC §1.7); filter via RUST_LOG
/// (default `info`).
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .with_current_span(false)
        .init();
}

/// `kiln-gateway hash-key [key]`: argon2 PHC string for kiln.toml
/// (`auth.api_keys[].key_hash`, `auth.admin_token_hash`).
fn hash_key(key: Option<String>) -> ExitCode {
    use argon2::password_hash::{SaltString, rand_core::OsRng};
    use argon2::{Argon2, PasswordHasher};

    let key = match key {
        Some(key) => key,
        None => {
            let mut line = String::new();
            if let Err(err) = std::io::stdin().read_line(&mut line) {
                eprintln!("kiln-gateway: failed to read key from stdin: {err}");
                return ExitCode::FAILURE;
            }
            line.trim_end_matches(['\r', '\n']).to_string()
        }
    };
    if key.is_empty() {
        eprintln!("kiln-gateway: refusing to hash an empty key");
        return ExitCode::from(2);
    }
    match Argon2::default().hash_password(key.as_bytes(), &SaltString::generate(&mut OsRng)) {
        Ok(hash) => {
            println!("{hash}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("kiln-gateway: hashing failed: {err}");
            ExitCode::FAILURE
        }
    }
}
