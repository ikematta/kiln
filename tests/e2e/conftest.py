"""Full-stack fixture: kiln-gateway + the Python worker + pinned tiny model.

The session fixture builds the gateway binary, writes a throwaway kiln.toml,
starts the gateway (which spawns the worker via the supervisor), and waits
for /readyz. Tests then drive it exclusively over HTTP — the real `openai`
SDK for the API surface, plain httpx for metrics/health (SPEC §11.3).

Requires the pinned Llama-3.2-1B test model (./scripts/fetch-test-model.sh);
skips with an actionable message otherwise.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass

import httpx
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_ID = "llama-3.2-1b-4bit"
API_KEY = "kiln-e2e-key"
READY_TIMEOUT_S = 600


def model_dir() -> pathlib.Path | None:
    root = os.environ.get("KILN_TEST_MODELS") or os.path.expanduser(
        "~/.kiln/test-models"
    )
    candidate = pathlib.Path(root) / MODEL_ID
    return candidate if (candidate / "config.json").is_file() else None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Stack:
    base_url: str
    api_key: str
    model_id: str
    runtime_dir: pathlib.Path
    gateway: subprocess.Popen
    log_path: pathlib.Path

    def metrics_text(self) -> str:
        return httpx.get(f"{self.base_url}/metrics", timeout=10).text

    def worker_pids(self) -> list[int]:
        """PIDs of worker processes, identified by the unique runtime dir in
        their --socket argument (matches uv wrapper and python alike)."""
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
            if "kiln_worker_py" in cmd:
                pids.append(pid)
        return pids

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


def build_gateway() -> pathlib.Path:
    binary = REPO / "target" / "debug" / "kiln-gateway"
    try:
        subprocess.run(
            ["cargo", "build", "-p", "kiln-gateway"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        if not binary.is_file():
            pytest.fail("cargo not on PATH and no prebuilt target/debug/kiln-gateway")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"cargo build -p kiln-gateway failed:\n{exc.stderr[-4000:]}")
    return binary


@pytest.fixture(scope="session")
def stack():
    model = model_dir()
    if model is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; set KILN_TEST_MODELS and run "
            "./scripts/fetch-test-model.sh"
        )

    binary = build_gateway()
    key_hash = subprocess.run(
        [binary, "hash-key", API_KEY], capture_output=True, text=True, check=True
    ).stdout.strip()

    # /tmp, not pytest's tmp_path: worker socket paths must stay under the
    # 104-byte macOS UDS limit.
    runtime_dir = pathlib.Path(tempfile.mkdtemp(prefix="kiln-e2e-", dir="/tmp"))
    port = free_port()
    config_path = runtime_dir / "kiln.toml"
    config_path.write_text(
        f"""
[server]
host = "127.0.0.1"
port = {port}
runtime_dir = "{runtime_dir}"

[[model]]
id = "{MODEL_ID}"
path = "{model}"
worker = "python"

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
        model_id=MODEL_ID,
        runtime_dir=runtime_dir,
        gateway=gateway,
        log_path=log_path,
    )
    try:
        stack.wait_ready()
        yield stack
    finally:
        gateway.terminate()
        try:
            gateway.wait(timeout=20)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait(timeout=10)
        shutil.rmtree(runtime_dir, ignore_errors=True)


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
