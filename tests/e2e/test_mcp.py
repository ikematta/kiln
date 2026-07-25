"""MCP client e2e (SPEC §8.4): a REAL MCP server (the official `mcp` SDK's
FastMCP reference implementation, tests/e2e/mcp_server.py) connects to the
gateway, its tools merge into requests, and the model's calls execute
gateway-side with the result fed back — the full Phase 7 round trip driven
server-side, proven over both API surfaces with the real `openai` and
`anthropic` SDKs.

Also proven here, per the crate::mcp timeout design:
- a deliberately hung tool (300s sleep) is bounded by `tool_timeout_secs`
  alone — no total timeout configured — via the error-tool-result path and
  the round cap (real timed assertion, not assumed);
- `total_timeout_secs` counts MCP execution: the same hung tool turns into
  a 504 `total_timeout` at the budget, mid-call;
- a server that is unreachable at startup retries on the supervisor
  backoff curve while the gateway serves normally, and a live server in
  the same config is unaffected (the http/streamable transport, spawned
  by this test);
- killing the stdio server's process mid-session reconnects with backoff
  and the tools work again;
- gateway shutdown kills the MCP server's whole process tree (the spawn
  goes through a `uv run` wrapper exactly so this is exercised).

The Qwen3-0.6B rust stack is the model harness: greedy Hermes-format tool
calling proven reliable by the Phase 7 suite for this exact
prompt/tool-shape pair.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import time

import httpx
import pytest

from conftest import (
    QWEN_MODEL_ID,
    build_binaries,
    free_port,
    pinned_model_dir,
    running_stack,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
ADMIN_TOKEN = "kiln-e2e-admin-token"
WEATHER_PROMPT = {"role": "user", "content": "What is the weather in Paris right now?"}


def mcp_argv(*extra: str, tag: str = "") -> list[str]:
    argv = [
        "uv",
        "run",
        "--project",
        "tests/e2e",
        "python",
        "tests/e2e/mcp_server.py",
        *extra,
    ]
    if tag:
        argv += ["--tag", tag]
    return argv


def mcp_block(name: str, argv: list[str], tool_timeout: int | None = None) -> str:
    lines = [
        "[[mcp_server]]",
        f'name = "{name}"',
        'transport = "stdio"',
        f"command = {json.dumps(argv)}",
    ]
    if tool_timeout is not None:
        lines.append(f"tool_timeout_secs = {tool_timeout}")
    return "\n".join(lines) + "\n"


def admin_toml() -> str:
    gateway = build_binaries()
    token_hash = subprocess.run(
        [gateway, "hash-key", ADMIN_TOKEN], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f'[auth]\nadmin_token_hash = "{token_hash}"\n'


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def metric_value(stack, name: str, **labels: str) -> float:
    """Sum of samples of `name` whose labels include all of `labels`."""
    total = 0.0
    found = False
    for line in stack.metrics_text().splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest and rest[0] not in "{ ":
            continue  # a longer metric name sharing the prefix
        if not all(f'{k}="{v}"' in line for k, v in labels.items()):
            continue
        total += float(line.rsplit(" ", 1)[1])
        found = True
    return total if found else 0.0


def wait_for(predicate, timeout_s: float, message: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    pytest.fail(f"timed out waiting for {message}")


def wait_mcp_up(stack, server: str, timeout_s: float = 90) -> None:
    wait_for(
        lambda: metric_value(stack, "kiln_mcp_up", server=server) == 1,
        timeout_s,
        f"MCP server '{server}' to connect",
    )


def assert_no_tagged_processes(tag: str) -> None:
    """The gateway is down: its MCP child (including the python under the
    `uv run` wrapper) must be gone — the process-group teardown mirrored
    from the worker supervisor."""
    result = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True)
    leaked = [int(p) for p in result.stdout.split()]
    for pid in leaked:
        os.kill(pid, signal.SIGKILL)
    assert not leaked, f"gateway shutdown leaked MCP server processes {leaked}"


def qwen_path() -> str:
    path = pinned_model_dir(QWEN_MODEL_ID)
    if path is None:
        pytest.skip(
            f"pinned test model '{QWEN_MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    return str(path)


# ---------------------------------------------------------------------------
# Healthy stdio server: discovery, round trips, merge/shadowing, reconnect
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcp_stack():
    tag = f"kiln-mcp-e2e-{os.getpid()}"
    extra = admin_toml() + mcp_block("e2e", mcp_argv(tag=tag))
    with running_stack(
        [(QWEN_MODEL_ID, "rust", qwen_path())], extra_toml=extra
    ) as stack:
        stack.wait_ready()
        wait_mcp_up(stack, "e2e")
        yield stack, tag
    assert_no_tagged_processes(tag)


@pytest.fixture(scope="module")
def mcp_client(mcp_stack):
    from openai import OpenAI

    stack, _ = mcp_stack
    return OpenAI(
        base_url=f"{stack.base_url}/v1",
        api_key=stack.api_key,
        timeout=240.0,
        max_retries=0,
    )


@pytest.fixture(scope="module")
def mcp_anthropic(mcp_stack):
    import anthropic

    stack, _ = mcp_stack
    return anthropic.Anthropic(
        base_url=stack.base_url,
        api_key=stack.api_key,
        timeout=240.0,
        max_retries=0,
    )


def weather_kwargs(stream: bool = False) -> dict:
    # No `tools` in the request: the only tool in scope is MCP-discovered.
    return {
        "model": QWEN_MODEL_ID,
        "messages": [WEATHER_PROMPT],
        "temperature": 0,
        "max_tokens": 512,
        "stream": stream,
    }


def test_mcp_round_trip_openai(mcp_stack, mcp_client):
    stack, _ = mcp_stack
    ok_before = metric_value(
        stack,
        "kiln_mcp_tool_calls_total",
        server="e2e",
        tool="get_weather",
        outcome="ok",
    )
    completion = mcp_client.chat.completions.create(**weather_kwargs())
    choice = completion.choices[0]
    # The model called the MCP tool, the gateway executed it, and the model
    # answered from the real result — the client sees only a final answer.
    assert choice.finish_reason == "stop", choice
    assert choice.message.tool_calls is None
    content = choice.message.content or ""
    assert "21" in content, content
    ok_after = metric_value(
        stack,
        "kiln_mcp_tool_calls_total",
        server="e2e",
        tool="get_weather",
        outcome="ok",
    )
    assert ok_after > ok_before
    # Usage sums every round the loop ran (prompt grows per round).
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0


def test_mcp_streaming_matches_non_streaming(mcp_client):
    reference = mcp_client.chat.completions.create(**weather_kwargs())
    chunks = mcp_client.chat.completions.create(
        **weather_kwargs(stream=True), stream_options={"include_usage": True}
    )
    content = ""
    finish = None
    usage = None
    for chunk in chunks:
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            finish = choice.finish_reason or finish
            content += choice.delta.content or ""
            # Executed MCP calls are internal: no tool_call deltas leak.
            assert not choice.delta.tool_calls, choice
    assert finish == "stop"
    # Greedy determinism holds through the loop: the streamed content and
    # summed usage equal the non-streaming response exactly.
    assert content == reference.choices[0].message.content
    assert usage is not None
    assert usage.prompt_tokens == reference.usage.prompt_tokens
    assert usage.completion_tokens == reference.usage.completion_tokens


def test_mcp_round_trip_anthropic(mcp_stack, mcp_anthropic):
    response = mcp_anthropic.messages.create(
        model=QWEN_MODEL_ID,
        max_tokens=512,
        messages=[WEATHER_PROMPT],
        temperature=0,
    )
    assert response.stop_reason == "end_turn", response
    kinds = [block.type for block in response.content]
    # Executed calls never surface as tool_use blocks; the thinking block
    # is the Qwen3 adapter behavior from Phase 7, unchanged under the loop.
    assert "tool_use" not in kinds, kinds
    texts = " ".join(block.text for block in response.content if block.type == "text")
    assert "21" in texts, response.content


def test_client_tool_merges_alongside_mcp_tools(mcp_stack, mcp_client):
    """A request-level tool coexists with MCP tools; a weather question
    still routes to the MCP tool and executes gateway-side."""
    stack, _ = mcp_stack
    ok_before = metric_value(
        stack,
        "kiln_mcp_tool_calls_total",
        server="e2e",
        tool="get_weather",
        outcome="ok",
    )
    completion = mcp_client.chat.completions.create(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"],
                    },
                },
            }
        ],
        **weather_kwargs(),
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    assert "21" in (choice.message.content or ""), choice
    assert (
        metric_value(
            stack,
            "kiln_mcp_tool_calls_total",
            server="e2e",
            tool="get_weather",
            outcome="ok",
        )
        > ok_before
    )


def test_client_tool_shadows_mcp_tool(mcp_stack, mcp_client):
    """A client-supplied get_weather wins: the call comes back to the
    client as ordinary tool_calls and the gateway never executes it."""
    stack, _ = mcp_stack
    calls_before = metric_value(stack, "kiln_mcp_tool_calls_total", server="e2e")
    completion = mcp_client.chat.completions.create(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        **weather_kwargs(),
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert (
        metric_value(stack, "kiln_mcp_tool_calls_total", server="e2e") == calls_before
    )


def test_admin_mcp_listing(mcp_stack):
    stack, _ = mcp_stack
    response = httpx.get(
        f"{stack.base_url}/admin/mcp",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        timeout=10,
    )
    assert response.status_code == 200, response.text
    servers = response.json()["servers"]
    assert len(servers) == 1
    server = servers[0]
    assert server["name"] == "e2e"
    assert server["transport"] == "stdio"
    assert server["status"] == "connected"
    assert server["protocol_version"]
    tools = {tool["name"]: tool for tool in server["tools"]}
    assert "get_weather" in tools
    assert tools["get_weather"]["active"] is True
    assert tools["get_weather"]["input_schema"]["type"] == "object"
    # The admin surface stays fail-closed.
    assert httpx.get(f"{stack.base_url}/admin/mcp", timeout=10).status_code == 401


def test_reconnect_after_server_death(mcp_stack, mcp_client):
    """Killing the MCP server process reconnects on the supervisor backoff
    curve (a fresh child) and the tools work again."""
    stack, tag = mcp_stack
    ok_connects_before = metric_value(
        stack, "kiln_mcp_connect_attempts_total", server="e2e", outcome="ok"
    )
    result = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split()]
    assert pids, "expected a live tagged MCP server process"
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    wait_for(
        lambda: (
            metric_value(
                stack, "kiln_mcp_connect_attempts_total", server="e2e", outcome="ok"
            )
            > ok_connects_before
        ),
        30,
        "MCP reconnect after server death",
    )
    wait_mcp_up(stack, "e2e", timeout_s=30)
    completion = mcp_client.chat.completions.create(**weather_kwargs())
    assert completion.choices[0].finish_reason == "stop"
    assert "21" in (completion.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Timeout interaction (crate::mcp module docs), with a really-hung tool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def slow_tool_stack():
    """get_weather hangs 300s; tool_timeout_secs = 2; NO total timeout —
    the per-tool bound must protect the default configuration alone."""
    tag = f"kiln-mcp-slow-{os.getpid()}"
    extra = mcp_block(
        "slow", mcp_argv("--sleep-weather", "300", tag=tag), tool_timeout=2
    )
    with running_stack(
        [(QWEN_MODEL_ID, "rust", qwen_path())], extra_toml=extra
    ) as stack:
        stack.wait_ready()
        wait_mcp_up(stack, "slow")
        yield stack
    assert_no_tagged_processes(tag)


def test_hung_tool_bounded_by_per_tool_timeout(slow_tool_stack):
    """The 300s hang never reaches the client: each call resolves in 2s as
    an error tool-result, the round cap bounds retries, and the request
    completes far under the hang — with no total timeout configured."""
    stack = slow_tool_stack
    started = time.monotonic()
    response = httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {stack.api_key}"},
        json={
            "model": QWEN_MODEL_ID,
            "messages": [WEATHER_PROMPT],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=240.0,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    finish = response.json()["choices"][0]["finish_reason"]
    # Terminal either way: the model answers without the tool ("stop") or
    # retries until the round cap hands its calls back ("tool_calls").
    assert finish in ("stop", "tool_calls"), response.json()
    assert elapsed < 180, f"hung tool stalled the request for {elapsed:.1f}s"
    assert (
        metric_value(
            stack,
            "kiln_mcp_tool_calls_total",
            server="slow",
            tool="get_weather",
            outcome="timeout",
        )
        >= 1
    )


@pytest.fixture(scope="module")
def total_timeout_stack():
    """The same 300s hang, but tool_timeout_secs = 60 and a 12s total
    budget: the total deadline must fire first, mid-call."""
    tag = f"kiln-mcp-total-{os.getpid()}"
    extra = "total_timeout_secs = 12\n" + mcp_block(
        "slow", mcp_argv("--sleep-weather", "300", tag=tag), tool_timeout=60
    )
    with running_stack(
        [(QWEN_MODEL_ID, "rust", qwen_path())], extra_toml=extra
    ) as stack:
        stack.wait_ready()
        wait_mcp_up(stack, "slow")
        yield stack
    assert_no_tagged_processes(tag)


def test_hung_tool_bounded_by_total_timeout(total_timeout_stack):
    """MCP execution spends total-timeout budget (the documented design):
    the request 504s at ~12s even though the tool call itself, at
    tool_timeout_secs = 60, would happily wait far longer."""
    stack = total_timeout_stack
    started = time.monotonic()
    response = httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {stack.api_key}"},
        json={
            "model": QWEN_MODEL_ID,
            "messages": [WEATHER_PROMPT],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 504, response.text
    body = response.json()["error"]
    assert body["code"] == "total_timeout", body
    assert body["type"] == "timeout_error", body
    assert elapsed < 30, f"total timeout took {elapsed:.1f}s to fire"
    assert (
        metric_value(stack, "kiln_timeout_total", model=QWEN_MODEL_ID, scope="total")
        >= 1
    )


# ---------------------------------------------------------------------------
# Unreachable server + streamable-HTTP transport, in one config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_mcp_server():
    """A real streamable-HTTP MCP server owned by the test process."""
    port = free_port()
    # Own session/process group: the argv is a `uv run` wrapper, so
    # teardown must kill the group or the actual python server leaks.
    process = subprocess.Popen(
        mcp_argv("--transport", "http", "--port", str(port)),
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            if process.poll() is not None:
                pytest.fail("http MCP server exited during startup")
            time.sleep(0.25)
    else:
        os.killpg(process.pid, signal.SIGKILL)
        pytest.fail("http MCP server never opened its port")
    yield f"http://127.0.0.1:{port}/mcp"
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def mixed_stack(http_mcp_server):
    """One dead stdio server (nonexistent binary) plus one live HTTP
    server: the dead one must retry forever without disturbing anything."""
    extra = (
        "[[mcp_server]]\n"
        'name = "dead"\n'
        'transport = "stdio"\n'
        'command = ["/nonexistent-kiln-mcp-server"]\n'
        "[[mcp_server]]\n"
        'name = "web"\n'
        'transport = "http"\n'
        f'url = "{http_mcp_server}"\n'
    )
    with running_stack(
        [(QWEN_MODEL_ID, "rust", qwen_path())], extra_toml=extra
    ) as stack:
        stack.wait_ready()
        wait_mcp_up(stack, "web")
        yield stack


def test_unreachable_server_retries_while_gateway_serves(mixed_stack):
    stack = mixed_stack
    # The gateway is fully up (wait_ready passed) with a dead MCP server
    # configured; the supervision loop keeps retrying on the backoff curve.
    wait_for(
        lambda: (
            metric_value(
                stack, "kiln_mcp_connect_attempts_total", server="dead", outcome="error"
            )
            >= 2
        ),
        30,
        "backoff retries against the dead MCP server",
    )
    assert metric_value(stack, "kiln_mcp_up", server="dead") == 0
    assert metric_value(stack, "kiln_mcp_up", server="web") == 1


def test_http_transport_round_trip(mixed_stack):
    """The streamable-HTTP transport end-to-end: discovery over HTTP, the
    model calls the tool, the gateway executes it over HTTP."""
    stack = mixed_stack
    response = httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {stack.api_key}"},
        json={
            "model": QWEN_MODEL_ID,
            "messages": [WEATHER_PROMPT],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=240.0,
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop", choice
    assert "21" in (choice["message"]["content"] or ""), choice
    assert (
        metric_value(
            stack,
            "kiln_mcp_tool_calls_total",
            server="web",
            tool="get_weather",
            outcome="ok",
        )
        >= 1
    )
