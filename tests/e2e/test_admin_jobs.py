"""Admin jobs API e2e (SPEC §8.1 `/admin/jobs/*`, §9.1).

Full proxy path with real processes: the gateway spawns `kiln-jobs serve` on
demand, proxies admin JSON to the gRPC jobs service, and a download job runs
against a local stub hub (no network) — files land on disk, status polls
move queued → succeeded, and the SQLite store backs it all.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import os
import subprocess
import threading
import time

import httpx
import pytest
from conftest import REPO, build_binaries, running_stack

ADMIN_TOKEN = "kiln-e2e-admin-token"
STUB_SHA = "e2e0" * 10  # fake 40-char commit sha
STUB_FILES = {
    "config.json": b'{"model_type": "llama"}',
    # Big enough to be a real streamed transfer; advertised with its LFS
    # sha256 so the client verifies content, not just size.
    "model.safetensors": bytes((i * 31 ^ (i >> 8)) & 0xFF for i in range(1 << 20)),
}


class StubHub(http.server.BaseHTTPRequestHandler):
    """Just enough hub REST for one repo: revision resolution, a tree
    listing, and file resolution."""

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _json(self, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/api/models/stub/tiny/revision/main":
            return self._json({"sha": STUB_SHA})
        if self.path.startswith(f"/api/models/stub/tiny/tree/{STUB_SHA}"):
            return self._json(
                [
                    {
                        "type": "file",
                        "path": name,
                        "size": len(data),
                        "lfs": {"oid": hashlib.sha256(data).hexdigest()},
                    }
                    for name, data in STUB_FILES.items()
                ]
            )
        prefix = f"/stub/tiny/resolve/{STUB_SHA}/"
        if self.path.startswith(prefix):
            data = STUB_FILES.get(self.path[len(prefix) :])
            if data is not None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_response(404)
        self.end_headers()


@pytest.fixture(scope="module")
def stub_hub():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@contextlib.contextmanager
def admin_stack(stub_hub: str, with_admin_token: bool = True):
    """A model-less gateway with the admin surface configured. HF_ENDPOINT
    points the (gateway-spawned) kiln-jobs at the stub hub; the job store is
    confined to the stack's throwaway runtime dir."""
    gateway = build_binaries()
    extra = ""
    if with_admin_token:
        token_hash = subprocess.run(
            [gateway, "hash-key", ADMIN_TOKEN],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        extra = f'[auth]\nadmin_token_hash = "{token_hash}"\n'
    previous = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = stub_hub
    try:
        # The conftest template already confines jobs_db to the runtime dir.
        with running_stack([], extra_toml=extra) as stack:
            stack.wait_ready()
            yield stack
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def test_admin_disabled_without_token_hash(stub_hub):
    with admin_stack(stub_hub, with_admin_token=False) as stack:
        response = httpx.get(f"{stack.base_url}/admin/jobs", timeout=10)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "admin_disabled"


def test_admin_requires_the_admin_token(stub_hub):
    with admin_stack(stub_hub) as stack:
        assert httpx.get(f"{stack.base_url}/admin/jobs", timeout=10).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        response = httpx.get(f"{stack.base_url}/admin/jobs", headers=wrong, timeout=10)
        assert response.status_code == 401


def test_download_job_end_to_end_and_quantize_validation(stub_hub):
    with admin_stack(stub_hub) as stack:
        dest = stack.runtime_dir / "downloaded"
        response = httpx.post(
            f"{stack.base_url}/admin/jobs/download",
            headers=admin_headers(),
            json={"repo": "stub/tiny", "dest": str(dest)},
            timeout=30,  # first call spawns kiln-jobs
        )
        assert response.status_code == 202, response.text
        job = response.json()
        job_id = job["id"]
        assert job["kind"] == "download"
        assert job["state"] in ("queued", "running")
        assert job["spec"]["repo"] == "stub/tiny"

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            response = httpx.get(
                f"{stack.base_url}/admin/jobs/{job_id}",
                headers=admin_headers(),
                timeout=10,
            )
            assert response.status_code == 200, response.text
            job = response.json()
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.3)
        assert job["state"] == "succeeded", job
        assert job["detail"]["event"] == "done"

        # The job runner really wrote the repo where we asked.
        for name, data in STUB_FILES.items():
            assert (dest / name).read_bytes() == data
        assert (dest / ".kiln-revision").read_text() == f"stub/tiny@{STUB_SHA}\n"

        # List shows it; unknown ids 404.
        listed = httpx.get(
            f"{stack.base_url}/admin/jobs", headers=admin_headers(), timeout=10
        ).json()
        assert job_id in [entry["id"] for entry in listed["jobs"]]
        assert (
            httpx.get(
                f"{stack.base_url}/admin/jobs/no-such-job",
                headers=admin_headers(),
                timeout=10,
            ).status_code
            == 404
        )

        # Quantize validation proxies as a 400 (no converter run needed).
        response = httpx.post(
            f"{stack.base_url}/admin/jobs/quantize",
            headers=admin_headers(),
            json={"path": "/nonexistent", "bits": 5},
            timeout=10,
        )
        assert response.status_code == 400
        assert "bits" in response.json()["error"]["message"]

        # The job store lives in the stack's runtime dir, not the user's.
        assert (stack.runtime_dir / "jobs.sqlite").is_file()
        assert not REPO.joinpath("jobs.sqlite").exists()
