"""Full-stack fixture: kiln-gateway + a model worker + pinned tiny model.

The session fixture builds the gateway and Rust worker binaries, writes a
throwaway kiln.toml, starts the gateway (which spawns workers via the
supervisor), and waits for /readyz. Tests then drive it exclusively over
HTTP — the real `openai` SDK for the API surface, plain httpx for
metrics/health (SPEC §11.3).

The `stack` fixture is parametrized over both worker kinds (Phase 3: the
same e2e suite must pass against the Python worker and the Rust worker);
`running_stack` is reusable for multi-model stacks (cross-worker parity).

Requires the pinned Llama-3.2-1B test model (./scripts/fetch-test-model.sh);
skips with an actionable message otherwise.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass

import httpx
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_ID = "llama-3.2-1b-4bit"
QWEN_MODEL_ID = "qwen3-0.6b-4bit"
API_KEY = "kiln-e2e-key"
READY_TIMEOUT_S = 600


def pinned_model_dir(model_id: str) -> pathlib.Path | None:
    root = os.environ.get("KILN_TEST_MODELS") or os.path.expanduser(
        "~/.kiln/test-models"
    )
    candidate = pathlib.Path(root) / model_id
    return candidate if (candidate / "config.json").is_file() else None


def model_dir() -> pathlib.Path | None:
    return pinned_model_dir(MODEL_ID)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Stack:
    base_url: str
    api_key: str
    model_id: str
    worker_kind: str
    runtime_dir: pathlib.Path
    gateway: subprocess.Popen
    log_path: pathlib.Path

    def metrics_text(self) -> str:
        return httpx.get(f"{self.base_url}/metrics", timeout=10).text

    def worker_pids(self) -> list[int]:
        """PIDs of worker processes, identified by the unique runtime dir in
        their --socket argument. Matches the python worker (uv wrapper and
        python module alike) and the rust worker binary."""
        result = subprocess.run(
            ["pgrep", "-f", str(self.runtime_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = []
        for token in result.stdout.split():
            pid = int(token)
            if pid == self.gateway.pid:
                continue
            cmd = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            if "kiln_worker_py" in cmd or "kiln-worker" in cmd:
                pids.append(pid)
        return pids

    def worker_command(self, model_id: str) -> str:
        """Command line of the worker process serving `model_id`, located by
        its unique --socket path (worker-<sha256(id)[:6]>.sock — mirrors the
        gateway's socket_path_for). Empty string if not found."""
        digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
        socket_arg = str(self.runtime_dir / f"worker-{digest}.sock")
        for pid in self.worker_pids():
            cmd = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            if socket_arg in cmd:
                return cmd
        return ""

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.gateway.poll() is not None:
                pytest.fail(
                    f"gateway exited (rc={self.gateway.returncode}); log tail:\n"
                    + tail(self.log_path)
                )
            try:
                response = httpx.get(f"{self.base_url}/readyz", timeout=5)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        pytest.fail(f"stack never became ready; log tail:\n{tail(self.log_path)}")


def tail(path: pathlib.Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text().splitlines()[-lines:])
    except OSError:
        return "<no log>"


def build_binaries() -> pathlib.Path:
    """Builds kiln-gateway + kiln-worker; returns the gateway binary path
    (the worker is found by the gateway as its sibling)."""
    binary = REPO / "target" / "debug" / "kiln-gateway"
    try:
        subprocess.run(
            ["cargo", "build", "-p", "kiln-gateway", "-p", "kiln-worker"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        if not binary.is_file():
            pytest.fail("cargo not on PATH and no prebuilt target/debug/kiln-gateway")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"cargo build failed:\n{exc.stderr[-4000:]}")
    return binary


@contextlib.contextmanager
def running_stack(models: list[tuple], extra_toml: str = ""):
    """Launches a gateway serving `models` — (id, worker_kind) entries serve
    the pinned test model; (id, worker_kind, path) entries name their own
    model directory (auto-routing tests point at doctored configs); a 4th
    element is extra TOML appended inside the model block (e.g. a
    [model.speculative] sub-table or `pinned`/`ttl_seconds` keys).
    `extra_toml` is inserted at top level between [server] and the model
    blocks (e.g. a [memory] section). Tears down with the leaked-worker
    guard. Callers wait_ready()."""
    binary = build_binaries()
    key_hash = subprocess.run(
        [binary, "hash-key", API_KEY], capture_output=True, text=True, check=True
    ).stdout.strip()

    # /tmp, not pytest's tmp_path: worker socket paths must stay under the
    # 104-byte macOS UDS limit.
    runtime_dir = pathlib.Path(tempfile.mkdtemp(prefix="kiln-e2e-", dir="/tmp"))
    port = free_port()
    model_path = model_dir()
    blocks = "\n".join(
        f"""
[[model]]
id = "{entry[0]}"
path = "{entry[2] if len(entry) > 2 else model_path}"
worker = "{entry[1]}"
{entry[3] if len(entry) > 3 else ""}
"""
        for entry in models
    )
    config_path = runtime_dir / "kiln.toml"
    config_path.write_text(
        f"""
[server]
host = "127.0.0.1"
port = {port}
runtime_dir = "{runtime_dir}"
cache_dir = "{runtime_dir}/cache"
{extra_toml}
{blocks}
[[auth.api_keys]]
name = "e2e"
key_hash = "{key_hash}"
"""
    )

    log_path = runtime_dir / "gateway.log"
    with open(log_path, "wb") as log:
        gateway = subprocess.Popen(
            [binary, "--config", str(config_path)],
            cwd=REPO,  # python_worker_argv default is checkout-relative
            stdout=log,
            stderr=log,
        )
    stack = Stack(
        base_url=f"http://127.0.0.1:{port}",
        api_key=API_KEY,
        model_id=models[0][0],
        worker_kind=models[0][1],
        runtime_dir=runtime_dir,
        gateway=gateway,
        log_path=log_path,
    )
    try:
        yield stack
    finally:
        gateway.terminate()
        try:
            gateway.wait(timeout=20)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait(timeout=10)
        time.sleep(0.5)  # let the supervisor's group-kill land
        leaked = stack.worker_pids()
        for pid in leaked:  # never leave strays behind, even when failing
            os.kill(pid, signal.SIGKILL)
        shutil.rmtree(runtime_dir, ignore_errors=True)
        if leaked:
            pytest.fail(
                f"gateway shutdown leaked worker processes {leaked}; the "
                "supervisor must kill the whole worker process group"
            )


@pytest.fixture(scope="session", params=["python", "rust"])
def stack(request):
    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; set KILN_TEST_MODELS and run "
            "./scripts/fetch-test-model.sh"
        )
    with running_stack([(MODEL_ID, request.param)]) as running:
        running.wait_ready()
        yield running


@pytest.fixture(scope="session")
def client(stack):
    from openai import OpenAI

    # max_retries=0: the crash test must see the raw 502, not a silent retry.
    return OpenAI(
        base_url=f"{stack.base_url}/v1",
        api_key=stack.api_key,
        timeout=120.0,
        max_retries=0,
    )


@pytest.fixture(scope="session")
def qwen_stack():
    """Rust-worker Qwen3-0.6B stack: the thinking-trained, Hermes-format
    model (tool-call parsing in test_tool_calls, thinking blocks in
    test_messages). Session-scoped so both suites share one model load."""
    path = pinned_model_dir(QWEN_MODEL_ID)
    if path is None:
        pytest.skip(
            f"pinned test model '{QWEN_MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    with running_stack([(QWEN_MODEL_ID, "rust", str(path))]) as running:
        running.wait_ready()
        yield running
