//! MCP client (SPEC §8.4): connects to configured external MCP servers,
//! discovers their tools, merges them into chat/messages requests, and
//! executes matching tool calls gateway-side.
//!
//! # Shape
//!
//! One `[[mcp_server]]` block per server ([`crate::config::McpServerConfig`]),
//! two transports per the MCP spec: **stdio** (the server is a child process
//! of the gateway speaking newline-delimited JSON-RPC 2.0 on stdin/stdout)
//! and **http** (the streamable-HTTP transport: JSON-RPC POSTed to one
//! endpoint; responses arrive as `application/json` or as a short-lived
//! `text/event-stream`). Both are built on crates the gateway already
//! carries (tokio process/io, reqwest, serde_json) — no new dependencies.
//!
//! Per server, a supervision task mirrors the worker supervisor's loop:
//! connect (`initialize` handshake → `notifications/initialized` →
//! paginated `tools/list`), serve until the connection dies, then reconnect
//! on the worker supervisor's exact backoff curve
//! ([`crate::supervisor::backoff`]: 500ms doubling, 10s cap). Unlike a
//! crashing worker — our own binary, where a crash loop means a bug and
//! stops after 3 attempts — an unreachable external server is a normal
//! transient condition and reconnection costs nothing, so MCP retries
//! indefinitely; the state is visible in `GET /admin/mcp` and
//! `kiln_mcp_up`. A server that is down at startup (or forever) costs the
//! gateway nothing but log lines: requests simply see no tools from it.
//!
//! # Tool merge (SPEC §8.2 pipeline)
//!
//! Discovered tools are converted to the same OpenAI function shape both
//! API adapters already normalize client tools into, and appended to the
//! request's tool list before template rendering — from the model's
//! perspective an MCP tool and a client-supplied tool are indistinguishable.
//! Collisions resolve deterministically:
//!
//! - a client-supplied tool **shadows** a same-named MCP tool for that
//!   request (the client's definition renders; the call returns to the
//!   client — the gateway never executes a name the client redefined);
//! - across servers, config order wins: the first server exposing a name
//!   owns it, later duplicates are inert (marked in `GET /admin/mcp`).
//!
//! The merge is skipped when the model has no known tool-call format (it
//! cannot emit calls; requests without client tools keep working instead of
//! turning into 400s), when the request forces `tool_choice: "none"`, and
//! when a structured-output grammar is active (a JSON-constrained decode
//! cannot emit tool-call markers).
//!
//! # Execution loop
//!
//! When a completed turn's calls are **all** MCP-sourced, the gateway
//! executes them in order against their servers, appends the assistant
//! turn and one `tool` message per result to the conversation, re-renders,
//! and resubmits — the exact message flow a client performs manually in
//! the Phase 7 round trip, driven server-side. The loop ends when a turn
//! completes without tool calls (the final answer), or after
//! [`MAX_ROUNDS`] rounds (a model stuck calling tools forever), in which
//! case the turn is returned to the client as a normal `tool_calls`
//! response. A turn containing **any** client-supplied call is returned to
//! the client whole (mixed turns cannot be half-executed: the wire has no
//! way to attach results to some calls and hand over others).
//!
//! Tool results feed back as text: MCP `content` items are flattened
//! (text items verbatim, other kinds as their JSON), and `isError` results
//! pass through as-is — the model sees the error and can recover, which is
//! MCP's own convention for tool failures.
//!
//! # Timeouts (the §8.3 interaction — decided, not accidental)
//!
//! Two bounds govern an MCP round trip, deliberately mirroring the
//! reservation-style honesty of the tpm and TTFT-anchor rulings:
//!
//! - **`server.total_timeout_secs` counts MCP execution.** The total
//!   budget is a client-experience bound anchored at request arrival
//!   (crate::timeout module docs), and a tool round trip is wall-clock
//!   time the client spends waiting — excluding it would let one request
//!   hold a slot far past the operator's cap. The request's [`Deadlines`]
//!   survive across rounds, and every MCP call runs under
//!   `min(per-tool deadline, total deadline)`. Total expiry mid-call
//!   aborts the request through the same 504 `total_timeout` path as B1
//!   enforcement (there is no worker round in flight to cancel; the MCP
//!   call gets a best-effort `notifications/cancelled`).
//! - **`tool_timeout_secs` bounds each call independently** (default 30s,
//!   per server). Expiry is NOT request death: the call resolves to an
//!   error tool-result fed back to the model, which can answer without
//!   the tool. This is what keeps a hung MCP server from stalling
//!   requests **in the default configuration**, where
//!   `total_timeout_secs` is unset: without it, a hung server would
//!   wedge a request forever with nothing armed to stop it. With both
//!   configured, the worst case is bounded by `total_timeout` and each
//!   individual hang by `tool_timeout`.
//!
//! TTFT semantics are unchanged: arrival → first generated token of the
//! first round, exactly as B1 defined it. tpm reserves per round
//! (prompt + max_tokens held before each Submit, settled to actuals when
//! the round finishes); a mid-loop denial fails the request with the
//! normal 429 — rate pressure is not a reason to silently stop looping.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Notify, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::config::{McpServerConfig, McpTransportKind};
use crate::metrics::Metrics;
use crate::supervisor::backoff;

/// MCP protocol revision this client speaks (sent in `initialize`; the
/// server's accepted revision is echoed on http requests).
const PROTOCOL_VERSION: &str = "2025-06-18";
/// Bound on each handshake step (`initialize`, `tools/list`): a hung server
/// must cycle the supervision loop into Retrying, not wedge it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Executed-tool rounds per request before the turn is handed back to the
/// client as a normal `tool_calls` response — bounds a model stuck calling
/// tools forever. Deliberately a constant: it is a loop-safety net, not a
/// tuning knob (module docs).
pub(crate) const MAX_ROUNDS: usize = 8;

// ---------------------------------------------------------------------------
// Public state types
// ---------------------------------------------------------------------------

/// One discovered MCP tool, as the server declared it.
#[derive(Debug, Clone)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

impl McpTool {
    /// The OpenAI function shape both adapters normalize client tools into
    /// — what renders into the chat template and hints the parsers.
    pub fn openai_def(&self) -> Value {
        let mut function = serde_json::Map::new();
        function.insert("name".into(), self.name.clone().into());
        if let Some(description) = &self.description {
            function.insert("description".into(), description.clone().into());
        }
        function.insert("parameters".into(), self.input_schema.clone());
        json!({"type": "function", "function": function})
    }
}

/// Connection state, updated by the supervision task; snapshot via
/// [`McpServer::state`].
#[derive(Debug, Clone)]
pub enum McpState {
    /// First attempt in flight (startup).
    Connecting,
    Connected {
        tools: Vec<McpTool>,
        protocol_version: String,
    },
    Retrying {
        attempt: u32,
        last_error: String,
    },
}

impl McpState {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Connecting => "connecting",
            Self::Connected { .. } => "connected",
            Self::Retrying { .. } => "retrying",
        }
    }
}

pub struct McpServer {
    pub config: McpServerConfig,
    state: Mutex<McpState>,
    /// Live connection while Connected; calls race disconnection benignly
    /// (a dead handle resolves to an Unavailable outcome).
    conn: Mutex<Option<Arc<Conn>>>,
}

impl McpServer {
    fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(McpState::Connecting),
            conn: Mutex::new(None),
        }
    }

    pub fn state(&self) -> McpState {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn set_state(&self, state: McpState) {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = state;
    }

    fn set_conn(&self, conn: Option<Arc<Conn>>) {
        *self
            .conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = conn;
    }

    fn current_conn(&self) -> Option<Arc<Conn>> {
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Tool timeout for calls against this server.
    fn tool_timeout(&self) -> Duration {
        Duration::from_secs(self.config.tool_timeout_secs)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub struct McpRegistry {
    /// Config order — the collision-resolution order (module docs).
    servers: Vec<Arc<McpServer>>,
    shutdown: watch::Sender<bool>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl McpRegistry {
    /// Builds the registry and spawns one supervision task per configured
    /// server. With no servers this is inert: snapshots are empty and the
    /// request paths take their historical code paths untouched.
    pub fn start(configs: &[McpServerConfig], metrics: Arc<Metrics>) -> Arc<Self> {
        let (shutdown, _) = watch::channel(false);
        let servers: Vec<Arc<McpServer>> = configs
            .iter()
            .map(|config| Arc::new(McpServer::new(config.clone())))
            .collect();
        let registry = Arc::new(Self {
            servers,
            shutdown,
            tasks: Mutex::new(Vec::new()),
        });
        let mut tasks = Vec::new();
        for server in &registry.servers {
            let server = Arc::clone(server);
            let metrics = Arc::clone(&metrics);
            let shutdown = registry.shutdown.subscribe();
            tasks.push(tokio::spawn(supervise(server, metrics, shutdown)));
        }
        *registry
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = tasks;
        registry
    }

    /// Inert registry for handler unit tests.
    pub fn disabled() -> Arc<Self> {
        let (shutdown, _) = watch::channel(false);
        Arc::new(Self {
            servers: Vec::new(),
            shutdown,
            tasks: Mutex::new(Vec::new()),
        })
    }

    pub fn servers(&self) -> &[Arc<McpServer>] {
        &self.servers
    }

    /// Consistent per-request view of every connected server's tools,
    /// config-order first-wins on duplicate names. Empty when nothing is
    /// connected — the gate the request paths branch on.
    pub fn snapshot(&self) -> McpToolSet {
        let mut entries: Vec<(String, McpTool, Arc<McpServer>)> = Vec::new();
        for server in &self.servers {
            let McpState::Connected { tools, .. } = server.state() else {
                continue;
            };
            for tool in tools {
                if entries.iter().any(|(name, ..)| *name == tool.name) {
                    continue; // earlier server owns the name (module docs)
                }
                entries.push((tool.name.clone(), tool, Arc::clone(server)));
            }
        }
        McpToolSet { entries }
    }

    /// Stops every supervision task and tears down live connections
    /// (killing stdio children).
    pub async fn shutdown(&self) {
        let _ = self.shutdown.send(true);
        let tasks = std::mem::take(
            &mut *self
                .tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for task in tasks {
            let _ = task.await;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-request tool set: merge + dispatch
// ---------------------------------------------------------------------------

/// How one executed MCP call resolved — the `outcome` label on
/// `kiln_mcp_tool_calls_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallOutcome {
    Ok,
    /// The server answered with `isError: true` (a tool-level failure).
    ToolError,
    /// Transport or JSON-RPC failure.
    Error,
    /// The per-tool timeout expired.
    Timeout,
    /// The server is not connected (or vanished mid-request).
    Unavailable,
}

impl CallOutcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::ToolError => "tool_error",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Unavailable => "unavailable",
        }
    }
}

/// The MCP tools in scope for one request. Captured once at request start
/// so discovery churn mid-request cannot re-route a call.
pub struct McpToolSet {
    entries: Vec<(String, McpTool, Arc<McpServer>)>,
}

impl McpToolSet {
    /// The empty set: the gate that keeps a request on the historical
    /// single-round code path.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|(n, ..)| n == name)
    }

    /// Merges this set into a request's validated tool list (OpenAI
    /// shapes): client names shadow — the shadowed MCP tool leaves the set
    /// entirely, so the gateway will not execute a name the client
    /// redefined — and the surviving defs are appended for rendering.
    pub fn merge_into(&mut self, tools: &mut Vec<Value>, model: &str) {
        self.entries.retain(|(name, _, server)| {
            let shadowed = tools.iter().any(|tool| {
                tool.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    == Some(name)
            });
            if shadowed {
                tracing::info!(target: "kiln::mcp", model, server = %server.config.name,
                    tool = %name,
                    "client-supplied tool shadows the MCP tool of the same name for this request");
            }
            !shadowed
        });
        tools.extend(self.entries.iter().map(|(_, tool, _)| tool.openai_def()));
    }

    /// Executes one model-emitted call. `total_deadline` is the request's
    /// surviving total-timeout deadline: expiry there aborts the request
    /// ([`ToolCallResult::TotalTimeout`]); every other failure — including
    /// the per-tool timeout — resolves to a text result fed back to the
    /// model (module docs).
    pub async fn call(
        &self,
        metrics: &Metrics,
        name: &str,
        arguments_json: &str,
        total_deadline: Option<Instant>,
    ) -> ToolCallResult {
        let Some((_, _, server)) = self.entries.iter().find(|(n, ..)| n == name) else {
            // Callers gate on contains(); keep this total anyway.
            return ToolCallResult::Text(format!("tool '{name}' is not available"));
        };
        let server_name = server.config.name.clone();
        let record = |outcome: CallOutcome| {
            metrics
                .mcp_tool_calls_total
                .with_label_values(&[&server_name, name, outcome.label()])
                .inc();
        };

        let arguments: Value = match arguments_json {
            "" => json!({}),
            text => match serde_json::from_str(text) {
                Ok(Value::Object(map)) => Value::Object(map),
                Ok(other) => other,
                Err(err) => {
                    // The model emitted non-JSON arguments; tell it so —
                    // the same self-correction loop as a tool error.
                    record(CallOutcome::ToolError);
                    return ToolCallResult::Text(format!(
                        "tool call arguments were not valid JSON: {err}"
                    ));
                }
            },
        };

        let tool_deadline = Instant::now() + server.tool_timeout();
        let deadline = match total_deadline {
            Some(total) => tool_deadline.min(total),
            None => tool_deadline,
        };

        let Some(conn) = server.current_conn() else {
            record(CallOutcome::Unavailable);
            return ToolCallResult::Text(format!(
                "tool '{name}' is currently unavailable (MCP server '{server_name}' \
                 is not connected)"
            ));
        };

        let started = Instant::now();
        let outcome = conn
            .request(
                "tools/call",
                json!({"name": name, "arguments": arguments}),
                deadline,
            )
            .await;
        match outcome {
            Ok(result) => {
                let is_error = result
                    .get("isError")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let text = flatten_content(&result);
                record(if is_error {
                    CallOutcome::ToolError
                } else {
                    CallOutcome::Ok
                });
                tracing::debug!(target: "kiln::mcp", server = %server_name, tool = %name,
                    is_error, elapsed_ms = started.elapsed().as_millis() as u64,
                    "MCP tool call completed");
                ToolCallResult::Text(text)
            }
            Err(McpError::Timeout) => {
                // Which deadline fired? If the total deadline has passed,
                // the request is out of budget — abort it. Otherwise this
                // was the per-tool bound: report and continue.
                if let Some(total) = total_deadline
                    && Instant::now() >= total
                {
                    record(CallOutcome::Timeout);
                    return ToolCallResult::TotalTimeout;
                }
                record(CallOutcome::Timeout);
                tracing::warn!(target: "kiln::mcp", server = %server_name, tool = %name,
                    timeout_secs = server.config.tool_timeout_secs,
                    "MCP tool call timed out (per-tool bound); feeding error result to the model");
                ToolCallResult::Text(format!(
                    "tool call timed out after {} seconds",
                    server.config.tool_timeout_secs
                ))
            }
            Err(err) => {
                record(CallOutcome::Error);
                tracing::warn!(target: "kiln::mcp", server = %server_name, tool = %name,
                    error = %err, "MCP tool call failed");
                ToolCallResult::Text(format!("tool call failed: {err}"))
            }
        }
    }
}

/// Outcome of [`McpToolSet::call`].
pub enum ToolCallResult {
    /// Feed this text back to the model as the tool result.
    Text(String),
    /// The request's total-timeout budget expired during the call: abort
    /// the request through the standard 504 path.
    TotalTimeout,
}

/// MCP `tools/call` result → tool-result text: text items verbatim (joined
/// with newlines), any other content kind as its JSON. `structuredContent`
/// is ignored when text content exists (servers mirror one into the other).
fn flatten_content(result: &Value) -> String {
    let items = result.get("content").and_then(Value::as_array);
    let mut parts: Vec<String> = Vec::new();
    for item in items.into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                parts.push(
                    item.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                );
            }
            _ => parts.push(item.to_string()),
        }
    }
    if parts.is_empty()
        && let Some(structured) = result.get("structuredContent")
    {
        return structured.to_string();
    }
    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Supervision
// ---------------------------------------------------------------------------

async fn supervise(
    server: Arc<McpServer>,
    metrics: Arc<Metrics>,
    mut shutdown: watch::Receiver<bool>,
) {
    let name = server.config.name.clone();
    let mut attempt: u32 = 0;
    loop {
        if *shutdown.borrow() {
            return;
        }
        match connect(&server.config).await {
            Ok((conn, protocol_version, tools)) => {
                attempt = 0;
                metrics
                    .mcp_connect_attempts_total
                    .with_label_values(&[&name, "ok"])
                    .inc();
                metrics.mcp_up.with_label_values(&[&name]).set(1);
                tracing::info!(target: "kiln::mcp", server = %name,
                    protocol_version = %protocol_version, tools = tools.len(),
                    "MCP server connected");
                let conn = Arc::new(conn);
                server.set_conn(Some(Arc::clone(&conn)));
                server.set_state(McpState::Connected {
                    tools,
                    protocol_version: protocol_version.clone(),
                });

                // Serve until the transport dies, re-listing on
                // notifications/tools/list_changed. (Helper future: the
                // watch guard must not live across the select arms.)
                async fn wait_broken(mut rx: watch::Receiver<bool>) {
                    let _ = rx.wait_for(|broken| *broken).await;
                }
                loop {
                    tokio::select! {
                        _ = wait_broken(conn.broken.clone()) => break,
                        _ = shutdown.changed() => {
                            conn.close().await;
                            server.set_conn(None);
                            return;
                        }
                        _ = conn.tools_changed.notified() => {
                            match list_tools(&conn).await {
                                Ok(tools) => {
                                    tracing::info!(target: "kiln::mcp", server = %name,
                                        tools = tools.len(), "MCP tool list refreshed");
                                    server.set_state(McpState::Connected {
                                        tools,
                                        protocol_version: protocol_version.clone(),
                                    });
                                }
                                Err(err) => tracing::warn!(target: "kiln::mcp", server = %name,
                                    error = %err, "MCP tools/list refresh failed"),
                            }
                        }
                    }
                }
                server.set_conn(None);
                conn.close().await;
                metrics.mcp_up.with_label_values(&[&name]).set(0);
                server.set_state(McpState::Retrying {
                    attempt: 0,
                    last_error: "connection lost".into(),
                });
                tracing::warn!(target: "kiln::mcp", server = %name,
                    "MCP connection lost; reconnecting with backoff");
            }
            Err(err) => {
                attempt += 1;
                metrics
                    .mcp_connect_attempts_total
                    .with_label_values(&[&name, "error"])
                    .inc();
                metrics.mcp_up.with_label_values(&[&name]).set(0);
                server.set_state(McpState::Retrying {
                    attempt,
                    last_error: err.to_string(),
                });
                let delay = backoff(attempt);
                tracing::warn!(target: "kiln::mcp", server = %name, attempt,
                    delay_ms = delay.as_millis() as u64, error = %err,
                    "MCP connect failed; retrying after backoff");
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    _ = shutdown.changed() => return,
                }
            }
        }
    }
}

/// Full connect: transport up, `initialize` handshake, `initialized`
/// notification, tool discovery. Every step deadline-bounded so a hung
/// server cycles the loop instead of wedging it.
async fn connect(config: &McpServerConfig) -> Result<(Conn, String, Vec<McpTool>), McpError> {
    let conn = match config.transport {
        McpTransportKind::Stdio => Conn::spawn_stdio(config)?,
        McpTransportKind::Http => Conn::http(config),
    };
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let init = conn
        .request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "kiln-gateway",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
            deadline,
        )
        .await
        .inspect_err(|_err| {
            // A failed handshake must not leak a live child process.
            conn.abort();
        })?;
    let protocol_version = init
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(PROTOCOL_VERSION)
        .to_owned();
    conn.set_protocol_version(&protocol_version);
    conn.notify("notifications/initialized", json!({}))
        .await
        .inspect_err(|_err| {
            conn.abort();
        })?;
    let tools = match list_tools(&conn).await {
        Ok(tools) => tools,
        Err(err) => {
            conn.abort();
            return Err(err);
        }
    };
    Ok((conn, protocol_version, tools))
}

/// `tools/list`, following cursor pagination to completion.
async fn list_tools(conn: &Conn) -> Result<Vec<McpTool>, McpError> {
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let params = match &cursor {
            Some(cursor) => json!({"cursor": cursor}),
            None => json!({}),
        };
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        let result = conn.request("tools/list", params, deadline).await?;
        for tool in result
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            tools.push(McpTool {
                name: name.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                input_schema: tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            });
        }
        match result.get("nextCursor").and_then(Value::as_str) {
            Some(next) if !next.is_empty() => cursor = Some(next.to_owned()),
            _ => return Ok(tools),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection: JSON-RPC over stdio or streamable HTTP
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("failed to spawn MCP server: {0}")]
    Spawn(std::io::Error),
    #[error("MCP connection closed")]
    Closed,
    #[error("MCP request timed out")]
    Timeout,
    #[error("MCP transport error: {0}")]
    Transport(String),
    #[error("MCP server error {code}: {message}")]
    Rpc { code: i64, message: String },
}

enum Transport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

struct Conn {
    transport: Transport,
    next_id: AtomicU64,
    /// Flips true when the transport dies; the supervision loop watches it.
    broken: watch::Receiver<bool>,
    broken_tx: watch::Sender<bool>,
    /// Signalled by `notifications/tools/list_changed`.
    tools_changed: Arc<Notify>,
}

impl Conn {
    fn spawn_stdio(config: &McpServerConfig) -> Result<Self, McpError> {
        let mut command = tokio::process::Command::new(&config.command[0]);
        command
            .args(&config.command[1..])
            .envs(&config.env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Own process group (pgid = child pid): the configured command
            // may be a wrapper (`uv run`, `npx`), so teardown must signal
            // the whole group or the real server survives as an orphan —
            // the worker supervisor's spawn discipline exactly
            // (crate::supervisor::signal_group).
            .process_group(0)
            // Safety net if the gateway dies without running teardown.
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(McpError::Spawn)?;
        // All three are piped above; a missing handle is an io-level bug.
        let missing =
            || McpError::Spawn(std::io::Error::other("child spawned without piped stdio"));
        let stdin = child.stdin.take().ok_or_else(missing)?;
        let stdout = child.stdout.take().ok_or_else(missing)?;
        let stderr = child.stderr.take().ok_or_else(missing)?;

        // Forward the server's stderr into the gateway log (debug level:
        // many servers chat on stderr as a matter of course).
        let name = config.name.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "kiln::mcp", server = %name, "stderr: {line}");
            }
        });

        let (broken_tx, broken) = watch::channel(false);
        let tools_changed = Arc::new(Notify::new());
        let stdio = StdioTransport::start(
            stdin,
            stdout,
            Some(child),
            broken_tx.clone(),
            Arc::clone(&tools_changed),
        );
        Ok(Self {
            transport: Transport::Stdio(stdio),
            next_id: AtomicU64::new(1),
            broken,
            broken_tx,
            tools_changed,
        })
    }

    /// Stdio framing over arbitrary IO halves — the unit-test entry point
    /// (tokio duplex pipes stand in for a child process).
    #[cfg(test)]
    fn stdio_for_test(
        writer: impl AsyncWrite + Unpin + Send + 'static,
        reader: impl AsyncRead + Unpin + Send + 'static,
    ) -> Self {
        let (broken_tx, broken) = watch::channel(false);
        let tools_changed = Arc::new(Notify::new());
        let stdio = StdioTransport::start(
            writer,
            reader,
            None,
            broken_tx.clone(),
            Arc::clone(&tools_changed),
        );
        Self {
            transport: Transport::Stdio(stdio),
            next_id: AtomicU64::new(1),
            broken,
            broken_tx,
            tools_changed,
        }
    }

    fn http(config: &McpServerConfig) -> Self {
        let (broken_tx, broken) = watch::channel(false);
        Self {
            transport: Transport::Http(HttpTransport {
                client: reqwest::Client::new(),
                url: config.url.clone().unwrap_or_default(),
                session: Mutex::new(None),
                protocol_version: Mutex::new(None),
            }),
            next_id: AtomicU64::new(1),
            broken,
            broken_tx,
            tools_changed: Arc::new(Notify::new()),
        }
    }

    fn set_protocol_version(&self, version: &str) {
        if let Transport::Http(http) = &self.transport {
            *http
                .protocol_version
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(version.to_owned());
        }
    }

    /// One JSON-RPC request under an absolute deadline. On timeout the
    /// pending slot is cleaned up and a best-effort `notifications/cancelled`
    /// goes to the server; a late response is discarded, not misdelivered.
    async fn request(
        &self,
        method: &str,
        params: Value,
        deadline: Instant,
    ) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        match &self.transport {
            Transport::Stdio(stdio) => stdio.request(id, method, params, deadline).await,
            Transport::Http(http) => {
                let result =
                    tokio::time::timeout_at(deadline, http.request(id, method, params)).await;
                match result {
                    Err(_elapsed) => Err(McpError::Timeout),
                    Ok(Ok(value)) => Ok(value),
                    Ok(Err(err)) => {
                        // Transport-level failures mark the connection
                        // broken so supervision re-initializes; JSON-RPC
                        // errors are the server answering and do not.
                        if matches!(err, McpError::Transport(_) | McpError::Closed) {
                            let _ = self.broken_tx.send(true);
                        }
                        Err(err)
                    }
                }
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        match &self.transport {
            Transport::Stdio(stdio) => stdio.notify(method, params).await,
            Transport::Http(http) => http.notify(method, params).await,
        }
    }

    /// Graceful teardown (shutdown path): stops the IO task, killing the
    /// stdio child.
    async fn close(&self) {
        if let Transport::Stdio(stdio) = &self.transport {
            stdio.close().await;
        }
    }

    /// Synchronous teardown for failed handshakes.
    fn abort(&self) {
        if let Transport::Stdio(stdio) = &self.transport {
            stdio.abort();
        }
    }
}

// --- stdio ---

enum StdioCmd {
    Request {
        id: u64,
        method: String,
        params: Value,
        reply: oneshot::Sender<Result<Value, McpError>>,
    },
    Notify {
        method: String,
        params: Value,
        done: oneshot::Sender<Result<(), McpError>>,
    },
    /// Caller-side timeout: drop the pending slot, tell the server.
    CancelPending {
        id: u64,
    },
    Close,
}

struct StdioTransport {
    cmd_tx: mpsc::Sender<StdioCmd>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl StdioTransport {
    fn start(
        writer: impl AsyncWrite + Unpin + Send + 'static,
        reader: impl AsyncRead + Unpin + Send + 'static,
        child: Option<tokio::process::Child>,
        broken: watch::Sender<bool>,
        tools_changed: Arc<Notify>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let task = tokio::spawn(stdio_io_task(
            writer,
            reader,
            child,
            cmd_rx,
            broken,
            tools_changed,
        ));
        Self {
            cmd_tx,
            task: Mutex::new(Some(task)),
        }
    }

    async fn request(
        &self,
        id: u64,
        method: &str,
        params: Value,
        deadline: Instant,
    ) -> Result<Value, McpError> {
        let (reply, rx) = oneshot::channel();
        self.cmd_tx
            .send(StdioCmd::Request {
                id,
                method: method.to_owned(),
                params,
                reply,
            })
            .await
            .map_err(|_| McpError::Closed)?;
        match tokio::time::timeout_at(deadline, rx).await {
            Err(_elapsed) => {
                let _ = self.cmd_tx.try_send(StdioCmd::CancelPending { id });
                Err(McpError::Timeout)
            }
            Ok(Err(_sender_dropped)) => Err(McpError::Closed),
            Ok(Ok(result)) => result,
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let (done, rx) = oneshot::channel();
        self.cmd_tx
            .send(StdioCmd::Notify {
                method: method.to_owned(),
                params,
                done,
            })
            .await
            .map_err(|_| McpError::Closed)?;
        rx.await.map_err(|_| McpError::Closed)?
    }

    async fn close(&self) {
        let _ = self.cmd_tx.send(StdioCmd::Close).await;
        let task = self
            .task
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(task) = task {
            let _ = task.await;
        }
    }

    fn abort(&self) {
        // Ask the io task to shut down (it runs the group kill on its way
        // out); JoinHandle::abort would skip that teardown and orphan a
        // wrapped server process.
        let _ = self.cmd_tx.try_send(StdioCmd::Close);
    }
}

/// The stdio IO task: owns both pipe halves and the pending-request map.
/// Newline-delimited JSON-RPC: requests out, responses matched by id;
/// server-initiated `ping` answered, `tools/list_changed` surfaced, other
/// notifications ignored. EOF or a write failure drains every pending
/// request with `Closed` and flips `broken`.
async fn stdio_io_task(
    mut writer: impl AsyncWrite + Unpin,
    reader: impl AsyncRead + Unpin,
    child: Option<tokio::process::Child>,
    mut cmd_rx: mpsc::Receiver<StdioCmd>,
    broken: watch::Sender<bool>,
    tools_changed: Arc<Notify>,
) {
    let mut lines = BufReader::new(reader).lines();
    let mut pending: HashMap<u64, oneshot::Sender<Result<Value, McpError>>> = HashMap::new();
    // Child ownership keeps kill_on_drop armed for the task's lifetime;
    // the group kill below targets its pgid on the way out.

    async fn write_line(
        writer: &mut (impl AsyncWrite + Unpin),
        value: &Value,
    ) -> Result<(), std::io::Error> {
        let mut line = value.to_string();
        line.push('\n');
        writer.write_all(line.as_bytes()).await
    }

    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => match cmd {
                None | Some(StdioCmd::Close) => break,
                Some(StdioCmd::Request { id, method, params, reply }) => {
                    let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
                    match write_line(&mut writer, &frame).await {
                        Ok(()) => { pending.insert(id, reply); }
                        Err(err) => {
                            let _ = reply.send(Err(McpError::Transport(err.to_string())));
                            break;
                        }
                    }
                }
                Some(StdioCmd::Notify { method, params, done }) => {
                    let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
                    match write_line(&mut writer, &frame).await {
                        Ok(()) => { let _ = done.send(Ok(())); }
                        Err(err) => {
                            let _ = done.send(Err(McpError::Transport(err.to_string())));
                            break;
                        }
                    }
                }
                Some(StdioCmd::CancelPending { id }) => {
                    if pending.remove(&id).is_some() {
                        let frame = json!({
                            "jsonrpc": "2.0",
                            "method": "notifications/cancelled",
                            "params": {"requestId": id, "reason": "timeout"},
                        });
                        if write_line(&mut writer, &frame).await.is_err() {
                            break;
                        }
                    }
                }
            },
            line = lines.next_line() => match line {
                Ok(Some(line)) => {
                    let Ok(message) = serde_json::from_str::<Value>(&line) else {
                        tracing::debug!(target: "kiln::mcp", "ignoring non-JSON stdio line");
                        continue;
                    };
                    let id = message.get("id").and_then(Value::as_u64);
                    let method = message.get("method").and_then(Value::as_str);
                    match (id, method) {
                        // Response to one of ours.
                        (Some(id), None) => {
                            if let Some(reply) = pending.remove(&id) {
                                let _ = reply.send(parse_rpc_outcome(&message));
                            }
                        }
                        // Server-initiated request: answer pings, decline
                        // the rest (we advertise no client capabilities).
                        (Some(id), Some(method)) => {
                            let response = if method == "ping" {
                                json!({"jsonrpc": "2.0", "id": id, "result": {}})
                            } else {
                                json!({"jsonrpc": "2.0", "id": id, "error":
                                    {"code": -32601, "message": "method not supported"}})
                            };
                            if write_line(&mut writer, &response).await.is_err() {
                                break;
                            }
                        }
                        (None, Some("notifications/tools/list_changed")) => {
                            tools_changed.notify_one();
                        }
                        _ => {}
                    }
                }
                Ok(None) | Err(_) => break, // EOF: server exited
            },
        }
    }
    for (_, reply) in pending.drain() {
        let _ = reply.send(Err(McpError::Closed));
    }
    let _ = broken.send(true);
    // Kill the child's whole process group, not just the direct child: the
    // configured command may be a wrapper (`uv run`), and a server stuck in
    // a long tool call ignores its stdin EOF. `/bin/kill` because signaling
    // a pgid needs libc::kill and unsafe is confined to kiln-mlx — the
    // worker supervisor's exact teardown (crate::supervisor::signal_group).
    if let Some(child) = &child
        && let Some(pgid) = child.id()
    {
        match tokio::process::Command::new("/bin/kill")
            .args(["-9", "--", &format!("-{pgid}")])
            .status()
            .await
        {
            Ok(status) if status.success() => {}
            // Non-zero usually means the group already exited cleanly.
            Ok(_) => tracing::debug!(target: "kiln::mcp", pgid, "MCP process group already gone"),
            Err(err) => tracing::warn!(target: "kiln::mcp", pgid, error = %err,
                "failed to signal MCP process group"),
        }
    }
    // `child` drops here; kill_on_drop reaps the direct child.
}

/// JSON-RPC envelope → result value or typed error.
fn parse_rpc_outcome(message: &Value) -> Result<Value, McpError> {
    if let Some(error) = message.get("error") {
        return Err(McpError::Rpc {
            code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_owned(),
        });
    }
    Ok(message.get("result").cloned().unwrap_or(Value::Null))
}

// --- streamable HTTP ---

struct HttpTransport {
    client: reqwest::Client,
    url: String,
    /// `Mcp-Session-Id` captured from the initialize response, echoed on
    /// every subsequent request (streamable-HTTP session contract).
    session: Mutex<Option<String>>,
    /// Negotiated revision, echoed as `MCP-Protocol-Version` post-init.
    protocol_version: Mutex<Option<String>>,
}

impl HttpTransport {
    fn apply_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = request
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(session) = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref()
        {
            request = request.header("mcp-session-id", session.to_owned());
        }
        if let Some(version) = self
            .protocol_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref()
        {
            request = request.header("mcp-protocol-version", version.to_owned());
        }
        request
    }

    async fn request(&self, id: u64, method: &str, params: Value) -> Result<Value, McpError> {
        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let response = self
            .apply_headers(self.client.post(&self.url))
            .body(frame.to_string())
            .send()
            .await
            .map_err(|err| McpError::Transport(err.to_string()))?;
        if let Some(session) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            *self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(session.to_owned());
        }
        let status = response.status();
        if !status.is_success() {
            return Err(McpError::Transport(format!("HTTP {status}")));
        }
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        if content_type.starts_with("text/event-stream") {
            self.read_sse_response(response, id).await
        } else {
            let body = response
                .text()
                .await
                .map_err(|err| McpError::Transport(err.to_string()))?;
            let message: Value = serde_json::from_str(&body)
                .map_err(|err| McpError::Transport(format!("non-JSON response: {err}")))?;
            parse_rpc_outcome(&message)
        }
    }

    /// Reads the POST-scoped SSE stream until the response bearing our id
    /// arrives (the spec has the server close the stream right after).
    async fn read_sse_response(
        &self,
        mut response: reqwest::Response,
        id: u64,
    ) -> Result<Value, McpError> {
        fn event_response(event: &str, id: u64) -> Option<Result<Value, McpError>> {
            let data: String = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect::<Vec<_>>()
                .join("\n");
            let message = serde_json::from_str::<Value>(&data).ok()?;
            if message.get("id").and_then(Value::as_u64) == Some(id) {
                return Some(parse_rpc_outcome(&message));
            }
            None
        }

        let mut buffer = String::new();
        loop {
            let chunk = response
                .chunk()
                .await
                .map_err(|err| McpError::Transport(err.to_string()))?;
            let Some(chunk) = chunk else {
                // Stream over: a final event may sit unterminated in the
                // buffer (servers that close right after the last frame).
                if let Some(outcome) = event_response(&buffer, id) {
                    return outcome;
                }
                return Err(McpError::Transport(
                    "SSE stream ended without a response".into(),
                ));
            };
            // Normalize CRLF framing (uvicorn-style servers emit \r\n):
            // JSON payloads escape control characters, so the transport
            // layer never carries a meaningful raw CR.
            buffer.push_str(&String::from_utf8_lossy(&chunk).replace('\r', ""));
            // SSE events are blank-line separated; process complete events.
            while let Some(end) = buffer.find("\n\n") {
                let event = buffer[..end].to_owned();
                buffer.drain(..end + 2);
                if let Some(outcome) = event_response(&event, id) {
                    return outcome;
                }
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
        let response = self
            .apply_headers(self.client.post(&self.url))
            .body(frame.to_string())
            .send()
            .await
            .map_err(|err| McpError::Transport(err.to_string()))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(McpError::Transport(format!("HTTP {}", response.status())))
        }
    }
}

// ---------------------------------------------------------------------------
// Admin surface: GET /admin/mcp
// ---------------------------------------------------------------------------

/// Operator listing (SPEC §8.4): every configured server with its live
/// connection state and discovered tools. `active: false` marks a tool
/// shadowed by an earlier server's same-named tool. Client-supplied tools
/// are per-request and never appear here — anything a model can call that
/// is not in this listing came from the client.
pub async fn admin_mcp(
    axum::extract::State(state): axum::extract::State<Arc<crate::app::AppState>>,
) -> axum::Json<Value> {
    axum::Json(status_json(&state.mcp))
}

/// The listing body, separated from the handler for unit testing.
fn status_json(registry: &McpRegistry) -> Value {
    let mut claimed: Vec<String> = Vec::new();
    let servers: Vec<Value> = registry
        .servers()
        .iter()
        .map(|server| {
            let config = &server.config;
            let state = server.state();
            let (status, attempt, last_error, protocol_version, tools) = match &state {
                McpState::Connecting => ("connecting", None, None, None, Vec::new()),
                McpState::Retrying {
                    attempt,
                    last_error,
                } => (
                    "retrying",
                    Some(*attempt),
                    Some(last_error.clone()),
                    None,
                    Vec::new(),
                ),
                McpState::Connected {
                    tools,
                    protocol_version,
                } => (
                    "connected",
                    None,
                    None,
                    Some(protocol_version.clone()),
                    tools.clone(),
                ),
            };
            let tools: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let active = !claimed.contains(&tool.name);
                    if active {
                        claimed.push(tool.name.clone());
                    }
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "active": active,
                    })
                })
                .collect();
            json!({
                "name": config.name,
                "transport": match config.transport {
                    McpTransportKind::Stdio => "stdio",
                    McpTransportKind::Http => "http",
                },
                "command": config.command,
                "url": config.url,
                "tool_timeout_secs": config.tool_timeout_secs,
                "status": status,
                "attempt": attempt,
                "last_error": last_error,
                "protocol_version": protocol_version,
                "tools": tools,
            })
        })
        .collect();
    json!({"servers": servers})
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use tokio::io::DuplexStream;

    /// A scripted MCP server on the far end of duplex pipes: speaks enough
    /// JSON-RPC for the handshake and echoes tool calls.
    struct FakeServer {
        reader: tokio::io::BufReader<tokio::io::ReadHalf<DuplexStream>>,
        writer: tokio::io::WriteHalf<DuplexStream>,
    }

    fn fake_pair() -> (Conn, FakeServer) {
        let (client_side, server_side) = tokio::io::duplex(64 * 1024);
        let (client_read, client_write) = tokio::io::split(client_side);
        let (server_read, server_write) = tokio::io::split(server_side);
        let conn = Conn::stdio_for_test(client_write, client_read);
        (
            conn,
            FakeServer {
                reader: tokio::io::BufReader::new(server_read),
                writer: server_write,
            },
        )
    }

    impl FakeServer {
        async fn recv(&mut self) -> Value {
            let mut line = String::new();
            self.reader.read_line(&mut line).await.expect("read");
            serde_json::from_str(&line).expect("json")
        }

        async fn send(&mut self, value: Value) {
            let mut line = value.to_string();
            line.push('\n');
            self.writer.write_all(line.as_bytes()).await.expect("write");
        }

        /// Answers the standard handshake (initialize + initialized) and
        /// one tools/list with the given tools.
        async fn handshake(&mut self, tools: Value) {
            let init = self.recv().await;
            assert_eq!(init["method"], "initialize");
            self.send(json!({"jsonrpc": "2.0", "id": init["id"],
                "result": {"protocolVersion": PROTOCOL_VERSION,
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "fake", "version": "0"}}}))
                .await;
            let initialized = self.recv().await;
            assert_eq!(initialized["method"], "notifications/initialized");
            let list = self.recv().await;
            assert_eq!(list["method"], "tools/list");
            self.send(json!({"jsonrpc": "2.0", "id": list["id"],
                "result": {"tools": tools}}))
                .await;
        }
    }

    fn weather_tool() -> Value {
        json!({"name": "get_weather", "description": "Weather.",
               "inputSchema": {"type": "object",
                   "properties": {"city": {"type": "string"}}}})
    }

    #[tokio::test]
    async fn stdio_handshake_and_tool_call_round_trip() {
        let (conn, mut server) = fake_pair();
        let server_task = tokio::spawn(async move {
            server.handshake(json!([weather_tool()])).await;
            let call = server.recv().await;
            assert_eq!(call["method"], "tools/call");
            assert_eq!(call["params"]["name"], "get_weather");
            assert_eq!(call["params"]["arguments"]["city"], "Paris");
            server
                .send(json!({"jsonrpc": "2.0", "id": call["id"],
                    "result": {"content": [{"type": "text", "text": "21C"}]}}))
                .await;
            server
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        let init = conn
            .request(
                "initialize",
                json!({"protocolVersion": PROTOCOL_VERSION,
                "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}),
                deadline,
            )
            .await
            .expect("initialize");
        assert_eq!(init["protocolVersion"], PROTOCOL_VERSION);
        conn.notify("notifications/initialized", json!({}))
            .await
            .expect("initialized");
        let tools = list_tools(&conn).await.expect("tools/list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");

        let result = conn
            .request(
                "tools/call",
                json!({"name": "get_weather", "arguments": {"city": "Paris"}}),
                Instant::now() + Duration::from_secs(5),
            )
            .await
            .expect("call");
        assert_eq!(flatten_content(&result), "21C");
        server_task.await.expect("fake server");
    }

    #[tokio::test]
    async fn unanswered_call_times_out_and_late_reply_is_discarded() {
        let (conn, mut server) = fake_pair();
        let deadline = Instant::now() + Duration::from_millis(50);
        let client = conn.request("tools/call", json!({"name": "hang"}), deadline);
        let (outcome, call) = tokio::join!(client, server.recv());
        assert!(matches!(outcome, Err(McpError::Timeout)));
        // The caller-side timeout sends notifications/cancelled.
        let cancelled = server.recv().await;
        assert_eq!(cancelled["method"], "notifications/cancelled");
        assert_eq!(cancelled["params"]["requestId"], call["id"]);
        // A late reply must not confuse a subsequent request.
        server
            .send(json!({"jsonrpc": "2.0", "id": call["id"], "result": {"late": true}}))
            .await;
        let next = conn.request("ping2", json!({}), Instant::now() + Duration::from_secs(5));
        let (outcome, request) = tokio::join!(next, async {
            let request = server.recv().await;
            server
                .send(json!({"jsonrpc": "2.0", "id": request["id"], "result": {"fresh": true}}))
                .await;
            request
        });
        assert_eq!(request["method"], "ping2");
        assert_eq!(outcome.expect("fresh reply")["fresh"], true);
    }

    #[tokio::test]
    async fn server_eof_fails_pending_and_flips_broken() {
        let (conn, mut server) = fake_pair();
        let pending = conn.request(
            "tools/call",
            json!({"name": "x"}),
            Instant::now() + Duration::from_secs(5),
        );
        let (outcome, _) = tokio::join!(pending, async {
            let _ = server.recv().await; // consume the request, then hang up
            drop(server);
        });
        assert!(matches!(outcome, Err(McpError::Closed)));
        let mut broken = conn.broken.clone();
        broken
            .wait_for(|b| *b)
            .await
            .expect("broken watch flips on EOF");
    }

    #[tokio::test]
    async fn server_ping_is_answered_and_list_changed_notifies() {
        let (conn, mut server) = fake_pair();
        server
            .send(json!({"jsonrpc": "2.0", "id": 777, "method": "ping", "params": {}}))
            .await;
        let pong = server.recv().await;
        assert_eq!(pong["id"], 777);
        assert!(pong.get("result").is_some());

        let notified = conn.tools_changed.notified();
        server
            .send(json!({"jsonrpc": "2.0",
                "method": "notifications/tools/list_changed", "params": {}}))
            .await;
        tokio::time::timeout(Duration::from_secs(5), notified)
            .await
            .expect("list_changed surfaces");
    }

    #[tokio::test]
    async fn rpc_error_response_is_a_typed_error() {
        let (conn, mut server) = fake_pair();
        let call = conn.request(
            "tools/call",
            json!({"name": "boom"}),
            Instant::now() + Duration::from_secs(5),
        );
        let (outcome, _) = tokio::join!(call, async {
            let request = server.recv().await;
            server
                .send(json!({"jsonrpc": "2.0", "id": request["id"],
                    "error": {"code": -32602, "message": "bad params"}}))
                .await;
        });
        match outcome {
            Err(McpError::Rpc { code, message }) => {
                assert_eq!(code, -32602);
                assert_eq!(message, "bad params");
            }
            other => panic!("expected Rpc error, got {other:?}"),
        }
    }

    // --- merge / flatten unit coverage ---

    fn tool(name: &str) -> McpTool {
        McpTool {
            name: name.into(),
            description: Some(format!("{name} tool")),
            input_schema: json!({"type": "object", "properties": {}}),
        }
    }

    fn set_of(entries: Vec<(&str, &str)>) -> McpToolSet {
        // (tool, server) pairs; server identity only matters for names.
        McpToolSet {
            entries: entries
                .into_iter()
                .map(|(tool_name, server_name)| {
                    (
                        tool_name.to_owned(),
                        tool(tool_name),
                        Arc::new(McpServer::new(McpServerConfig {
                            name: server_name.into(),
                            transport: McpTransportKind::Stdio,
                            command: vec!["srv".into()],
                            env: BTreeMap::new(),
                            url: None,
                            tool_timeout_secs: 30,
                        })),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn client_tools_shadow_mcp_tools_in_merge() {
        let mut set = set_of(vec![("get_weather", "a"), ("read_file", "a")]);
        let mut tools = vec![json!({"type": "function",
            "function": {"name": "get_weather", "parameters": {}}})];
        set.merge_into(&mut tools, "m");
        // The client's get_weather stands; MCP adds only read_file.
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[1]["function"]["name"], "read_file");
        // The shadowed name left the dispatch set: its calls go to the client.
        assert!(!set.contains("get_weather"));
        assert!(set.contains("read_file"));
    }

    #[test]
    fn openai_def_shape() {
        let def = tool("get_weather").openai_def();
        assert_eq!(def["type"], "function");
        assert_eq!(def["function"]["name"], "get_weather");
        assert_eq!(def["function"]["description"], "get_weather tool");
        assert_eq!(def["function"]["parameters"]["type"], "object");
    }

    /// `GET /admin/mcp` body: connection states, discovered tools, and
    /// cross-server shadowing (`active: false` on a later server's
    /// duplicate) are all operator-visible.
    #[test]
    fn admin_status_json_shows_states_and_shadowing() {
        let make_server = |name: &str| {
            Arc::new(McpServer::new(McpServerConfig {
                name: name.into(),
                transport: McpTransportKind::Stdio,
                command: vec!["srv".into()],
                env: BTreeMap::new(),
                url: None,
                tool_timeout_secs: 30,
            }))
        };
        let a = make_server("a");
        a.set_state(McpState::Connected {
            tools: vec![tool("get_weather"), tool("read_file")],
            protocol_version: PROTOCOL_VERSION.into(),
        });
        let b = make_server("b");
        b.set_state(McpState::Connected {
            tools: vec![tool("get_weather")], // shadowed by a's
            protocol_version: PROTOCOL_VERSION.into(),
        });
        let c = make_server("c");
        c.set_state(McpState::Retrying {
            attempt: 3,
            last_error: "spawn failed".into(),
        });
        let (shutdown, _) = watch::channel(false);
        let registry = McpRegistry {
            servers: vec![a, b, c],
            shutdown,
            tasks: Mutex::new(Vec::new()),
        };

        let body = status_json(&registry);
        let servers = body["servers"].as_array().expect("servers array");
        assert_eq!(servers.len(), 3);
        assert_eq!(servers[0]["status"], "connected");
        assert_eq!(servers[0]["tools"][0]["name"], "get_weather");
        assert_eq!(servers[0]["tools"][0]["active"], true);
        assert_eq!(servers[0]["tools"][1]["active"], true);
        // b's duplicate is listed but inert.
        assert_eq!(servers[1]["tools"][0]["name"], "get_weather");
        assert_eq!(servers[1]["tools"][0]["active"], false);
        // c shows its retry state and the error an operator needs.
        assert_eq!(servers[2]["status"], "retrying");
        assert_eq!(servers[2]["attempt"], 3);
        assert_eq!(servers[2]["last_error"], "spawn failed");
        assert!(servers[2]["tools"].as_array().expect("array").is_empty());
    }

    #[test]
    fn flatten_content_variants() {
        // Text items join with newlines.
        assert_eq!(
            flatten_content(&json!({"content": [
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})),
            "a\nb"
        );
        // Non-text items serialize.
        let flattened = flatten_content(&json!({"content": [
            {"type": "image", "data": "xyz", "mimeType": "image/png"}]}));
        assert!(flattened.contains("image/png"), "{flattened}");
        // structuredContent backfills when content is absent.
        assert_eq!(
            flatten_content(&json!({"structuredContent": {"n": 1}})),
            r#"{"n":1}"#
        );
        assert_eq!(flatten_content(&json!({})), "");
    }
}
