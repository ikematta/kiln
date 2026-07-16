#![deny(unsafe_code)]
//! `kiln`: the operator CLI (SPEC §12 Phase 10; §10's CLI surface).
//!
//! Thin wrappers only — no serving logic lives here:
//!   kiln serve  → exec the sibling `kiln-gateway` with the resolved config
//!   kiln models → render `GET /admin/models` (the admin API's models table
//!                 plus the machine memory ledger) as text or `--json`
//!   kiln bench  → exec `scripts/bench.sh` (throughput/TTFT harness)
//!
//! Config resolution (serve and models): `--config <path>`, else
//! `$KILN_CONFIG`, else `./kiln.toml`, else `<prefix>/etc/kiln/kiln.toml`
//! relative to the installed binary (`<prefix>/bin/kiln`). `kiln models`
//! falls back to built-in defaults + `KILN_` env overrides when none exists.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kiln_gateway::config::KilnConfig;

const USAGE: &str = "usage:
  kiln serve [--config <path>]             run the gateway
  kiln models [--config <path>] [--json]   list models via the admin API
                                           (token from $KILN_ADMIN_TOKEN)
  kiln bench [args...]                     run scripts/bench.sh (see --help)
  kiln --version";

#[derive(Debug, PartialEq)]
enum Command {
    Serve { config: Option<PathBuf> },
    Models { config: Option<PathBuf>, json: bool },
    Bench { args: Vec<String> },
    Version,
    Help,
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let mut config = None;
    let mut json = false;
    let Some((subcommand, rest)) = args.split_first() else {
        return Ok(Command::Help);
    };
    match subcommand.as_str() {
        // Bench arguments belong to bench.sh; pass them through untouched.
        "bench" => {
            return Ok(Command::Bench {
                args: rest.to_vec(),
            });
        }
        "--version" | "-V" => return Ok(Command::Version),
        "--help" | "-h" | "help" => return Ok(Command::Help),
        "serve" | "models" => {}
        other => return Err(format!("unknown command '{other}'")),
    }
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => return Err("--config requires a path".into()),
            },
            "--json" if subcommand == "models" => json = true,
            other => return Err(format!("unknown argument '{other}' for '{subcommand}'")),
        }
    }
    Ok(match subcommand.as_str() {
        "serve" => Command::Serve { config },
        _ => Command::Models { config, json },
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(Command::Serve { config }) => serve(config),
        Ok(Command::Models { config, json }) => models(config, json),
        Ok(Command::Bench { args }) => bench(&args),
        Ok(Command::Version) => {
            println!("kiln {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("kiln: {message}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// `--config` > `$KILN_CONFIG` > `./kiln.toml` > `<prefix>/etc/kiln/kiln.toml`.
fn resolve_config(explicit: Option<PathBuf>) -> Option<PathBuf> {
    if explicit.is_some() {
        return explicit;
    }
    if let Ok(env) = std::env::var("KILN_CONFIG")
        && !env.is_empty()
    {
        return Some(PathBuf::from(env));
    }
    let cwd = PathBuf::from("kiln.toml");
    if cwd.is_file() {
        return Some(cwd);
    }
    // Installed layout: <prefix>/bin/kiln → <prefix>/etc/kiln/kiln.toml
    // (the path the Homebrew formula writes).
    exe_relative("etc/kiln/kiln.toml")
}

/// `<exe dir>/../<rel>`, from the invoked path and then from the
/// symlink-resolved one. Homebrew splits the two: `<prefix>/bin/kiln` is a
/// symlink into the keg, config lives at the *prefix* (`<prefix>/etc/...`,
/// raw path) while libexec is unlinked and exists only in the *keg*
/// (resolved path).
fn exe_relative(rel: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    for base in [Some(exe.clone()), std::fs::canonicalize(&exe).ok()] {
        if let Some(path) = base.and_then(|exe| Some(exe.parent()?.parent()?.join(rel)))
            && path.is_file()
        {
            return Some(path);
        }
    }
    None
}

/// The named binary next to this executable, else bare (a `$PATH` lookup) —
/// the same sibling convention the gateway uses for its worker binaries.
fn sibling_binary(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join(name)))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(name))
}

/// Replaces this process with `argv` (unix exec: signals, exit code, and
/// terminal ownership all belong to the wrapped binary).
fn exec(binary: &Path, args: &[String]) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(binary).args(args).exec();
    eprintln!("kiln: failed to exec {}: {err}", binary.display());
    ExitCode::FAILURE
}

fn serve(config: Option<PathBuf>) -> ExitCode {
    let Some(config) = resolve_config(config) else {
        eprintln!(
            "kiln: no configuration found: pass --config <path>, set $KILN_CONFIG, \
             or create ./kiln.toml (installed default: <prefix>/etc/kiln/kiln.toml)"
        );
        return ExitCode::from(2);
    };
    let gateway = sibling_binary("kiln-gateway");
    exec(
        &gateway,
        &["--config".to_string(), config.display().to_string()],
    )
}

fn bench(args: &[String]) -> ExitCode {
    let Some(script) = find_bench_script() else {
        eprintln!(
            "kiln: scripts/bench.sh not found: run from a Kiln checkout, set \
             $KILN_BENCH_SCRIPT, or reinstall (expected <prefix>/libexec/scripts/bench.sh)"
        );
        return ExitCode::from(2);
    };
    exec(&script, args)
}

/// `$KILN_BENCH_SCRIPT` > `./scripts/bench.sh` (checkout) >
/// `<prefix>/libexec/scripts/bench.sh` (installed).
fn find_bench_script() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("KILN_BENCH_SCRIPT")
        && !env.is_empty()
    {
        return Some(PathBuf::from(env));
    }
    let checkout = PathBuf::from("scripts/bench.sh");
    if checkout.is_file() {
        return Some(checkout);
    }
    exe_relative("libexec/scripts/bench.sh")
}

fn models(config: Option<PathBuf>, json: bool) -> ExitCode {
    let config = match resolve_config(config) {
        Some(path) => KilnConfig::load(&path),
        None => KilnConfig::load_env_only(),
    };
    let config = match config {
        Ok(config) => config,
        Err(err) => {
            eprintln!("kiln: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Ok(token) = std::env::var("KILN_ADMIN_TOKEN") else {
        eprintln!(
            "kiln: KILN_ADMIN_TOKEN is not set: export the admin bearer token \
             (the raw token whose argon2 hash is auth.admin_token_hash; \
             hash one with `kiln-gateway hash-key`)"
        );
        return ExitCode::from(2);
    };
    let url = format!(
        "http://{}:{}/admin/models",
        config.server.host, config.server.port
    );

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("kiln: failed to start async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(fetch_models(&url, &token)) {
        Ok(body) => {
            if json {
                println!("{body}");
            } else {
                match body.parse::<serde_json::Value>() {
                    Ok(value) => print!("{}", render_models(&value)),
                    Err(err) => {
                        eprintln!("kiln: {url} returned unparseable JSON: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("kiln: {message}");
            ExitCode::FAILURE
        }
    }
}

/// GET the models table; API error bodies are surfaced verbatim (the same
/// convention the admin UI follows).
async fn fetch_models(url: &str, token: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let response = client
        .get(url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(|err| format!("GET {url} failed: {err} (is the gateway running?)"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|err| format!("failed reading response from {url}: {err}"))?;
    if !status.is_success() {
        return Err(format!("{url} returned {status}: {body}"));
    }
    Ok(body)
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)];
    for (unit, size) in UNITS {
        if bytes >= size {
            return format!("{:.1} {unit}", bytes as f64 / size as f64);
        }
    }
    format!("{bytes} B")
}

fn fmt_ttl(seconds: Option<u64>) -> String {
    match seconds {
        Some(0) | None => "-".to_string(),
        Some(s) => format!("{s}s"),
    }
}

/// Plain-text table over the `/admin/models` payload
/// (`{models: [...], memory: {...}}`).
fn render_models(value: &serde_json::Value) -> String {
    let mut rows: Vec<[String; 6]> = vec![[
        "ID".into(),
        "WORKER".into(),
        "STATUS".into(),
        "PINNED".into(),
        "TTL".into(),
        "MEMORY".into(),
    ]];
    let models = value["models"].as_array().cloned().unwrap_or_default();
    for model in &models {
        rows.push([
            model["id"].as_str().unwrap_or("?").to_string(),
            model["worker"].as_str().unwrap_or("?").to_string(),
            model["status"].as_str().unwrap_or("?").to_string(),
            if model["pinned"].as_bool().unwrap_or(false) {
                "yes".into()
            } else {
                "no".into()
            },
            fmt_ttl(model["ttl_seconds"].as_u64()),
            model["usage_bytes"]
                .as_u64()
                .map_or_else(|| "-".to_string(), fmt_bytes),
        ]);
    }

    let mut widths = [0usize; 6];
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let mut out = String::new();
    if models.is_empty() {
        out.push_str("no models configured\n");
    } else {
        for row in &rows {
            let mut line = String::new();
            for (width, cell) in widths.iter().zip(row) {
                line.push_str(&format!("{cell:<width$}  "));
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }
    let memory = &value["memory"];
    if let (Some(used), Some(budget)) = (
        memory["used_bytes"].as_u64(),
        memory["budget_bytes"].as_u64(),
    ) {
        out.push_str(&format!(
            "memory: {} used / {} budget ({} reserved, {} machine)\n",
            fmt_bytes(used),
            fmt_bytes(budget),
            fmt_bytes(memory["reserved_bytes"].as_u64().unwrap_or(0)),
            fmt_bytes(memory["total_bytes"].as_u64().unwrap_or(0)),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_subcommands() {
        assert_eq!(parse_args(&[]).unwrap(), Command::Help);
        assert_eq!(
            parse_args(&strings(&["serve", "--config", "/etc/k.toml"])).unwrap(),
            Command::Serve {
                config: Some(PathBuf::from("/etc/k.toml"))
            }
        );
        assert_eq!(
            parse_args(&strings(&["models", "--json"])).unwrap(),
            Command::Models {
                config: None,
                json: true
            }
        );
        // Bench arguments pass through verbatim, including flags kiln
        // itself does not know.
        assert_eq!(
            parse_args(&strings(&["bench", "--model", "x", "--engine"])).unwrap(),
            Command::Bench {
                args: strings(&["--model", "x", "--engine"])
            }
        );
        assert!(parse_args(&strings(&["frobnicate"])).is_err());
        assert!(parse_args(&strings(&["serve", "--json"])).is_err());
        assert!(parse_args(&strings(&["models", "--config"])).is_err());
    }

    #[test]
    fn renders_models_table_and_memory_line() {
        let value: serde_json::Value = serde_json::json!({
            "models": [
                {"id": "llama-1b", "worker": "rust", "status": "ready",
                 "pinned": true, "ttl_seconds": 0, "usage_bytes": 1288490189u64},
                {"id": "qwen-0.6b", "worker": "python", "status": "unloaded (admin)",
                 "pinned": false, "ttl_seconds": 300, "usage_bytes": null},
            ],
            "memory": {"budget_bytes": 27487790694u64, "used_bytes": 1288490189u64,
                        "reserved_bytes": 0, "total_bytes": 34359738368u64},
        });
        let rendered = render_models(&value);
        assert!(rendered.contains("ID"), "{rendered}");
        assert!(rendered.contains("llama-1b"), "{rendered}");
        assert!(rendered.contains("unloaded (admin)"), "{rendered}");
        assert!(rendered.contains("1.2 GiB"), "{rendered}");
        assert!(rendered.contains("300s"), "{rendered}");
        assert!(
            rendered.contains("memory: 1.2 GiB used / 25.6 GiB budget"),
            "{rendered}"
        );
    }

    #[test]
    fn renders_empty_registry() {
        let value = serde_json::json!({"models": [], "memory": {}});
        assert!(render_models(&value).contains("no models configured"));
    }

    #[test]
    fn formats_bytes_and_ttl() {
        assert_eq!(fmt_bytes(0), "0 B");
        assert_eq!(fmt_bytes(2048), "2.0 KiB");
        assert_eq!(fmt_bytes(5 << 20), "5.0 MiB");
        assert_eq!(fmt_bytes(3 << 30), "3.0 GiB");
        assert_eq!(fmt_ttl(Some(0)), "-");
        assert_eq!(fmt_ttl(None), "-");
        assert_eq!(fmt_ttl(Some(90)), "90s");
    }
}
