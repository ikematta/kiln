"""Add-model admin API e2e (`POST /admin/models`, `GET /admin/models/estimate`).

The single add-model flow, driven at the API level with real processes:
register a local model live (no restart), watch kiln.toml gain exactly one
[[model]] block with every pre-existing byte — including a hand edit made
while the gateway runs — preserved, reject duplicates clearly, and walk
the not-downloaded path (structured 409 → download job → retried add).
The browser-driven version of the full flow lives in test_admin_ui.py.
"""

from __future__ import annotations

import contextlib
import difflib
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib

import httpx
import pytest
from conftest import MODEL_ID, build_binaries, model_dir, running_stack
from test_admin_jobs import STUB_FILES, stub_hub  # noqa: F401 (fixture)

ADMIN_TOKEN = "kiln-e2e-add-model-token"


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@contextlib.contextmanager
def dir_hub(repo: str, root: pathlib.Path):
    """Stub hub serving a REAL model directory from disk: revision
    resolution, a tree listing (sizes, no lfs — the downloader verifies by
    size), and streamed file resolution. What huggingface.co would answer
    for `repo`, backed by `root`."""
    sha = "d1r0" * 10

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path == f"/api/models/{repo}/revision/main":
                return self._json({"sha": sha})
            if self.path.startswith(f"/api/models/{repo}/tree/{sha}"):
                return self._json(
                    [
                        {"type": "file", "path": p.name, "size": p.stat().st_size}
                        for p in sorted(root.iterdir())
                        if p.is_file() and not p.name.startswith((".", "README"))
                    ]
                )
            prefix = f"/{repo}/resolve/{sha}/"
            if self.path.startswith(prefix):
                target = root / self.path[len(prefix) :]
                if target.is_file():
                    self.send_response(200)
                    self.send_header("Content-Length", str(target.stat().st_size))
                    self.end_headers()
                    with open(target, "rb") as f:
                        shutil.copyfileobj(f, self.wfile)
                    return
            self.send_response(404)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


@contextlib.contextmanager
def add_stack(models: list[tuple], hub_url: str | None = None):
    """Admin-enabled stack whose model_dir (the default download dest
    root) is a throwaway directory, with the gateway's hub endpoint
    pointed at `hub_url` when given."""
    gateway = build_binaries()
    token_hash = subprocess.run(
        [gateway, "hash-key", ADMIN_TOKEN], capture_output=True, text=True, check=True
    ).stdout.strip()
    dest_root = pathlib.Path(tempfile.mkdtemp(prefix="kiln-e2e-modeldir-", dir="/tmp"))
    # The model_dir line lands inside [server] (extra_toml is inserted
    # among the [server] keys); [auth] then opens its own table.
    extra = f'model_dir = "{dest_root}"\n[auth]\nadmin_token_hash = "{token_hash}"\n'
    previous = os.environ.get("HF_ENDPOINT")
    if hub_url is not None:
        os.environ["HF_ENDPOINT"] = hub_url
    try:
        with running_stack(models, extra_toml=extra) as stack:
            stack.wait_ready()
            yield stack, dest_root
    finally:
        if hub_url is not None:
            if previous is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous
        shutil.rmtree(dest_root, ignore_errors=True)


def assert_only_inserted(before: str, after: str) -> list[str]:
    """The whole config-write contract in one assertion: every line of
    `before` survives verbatim and in order, and the only difference is
    one inserted run of lines. Returns the inserted lines."""
    opcodes = difflib.SequenceMatcher(
        None, before.splitlines(), after.splitlines(), autojunk=False
    ).get_opcodes()
    inserts = [op for op in opcodes if op[0] == "insert"]
    others = [op for op in opcodes if op[0] not in ("equal", "insert")]
    assert not others, f"lines were changed or removed, not just inserted: {opcodes}"
    assert len(inserts) == 1, f"expected exactly one inserted block: {opcodes}"
    _, _, _, lo, hi = inserts[0]
    return after.splitlines()[lo:hi]


def wait_for_status(stack, model_id: str, wanted: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    status = "<never polled>"
    while time.monotonic() < deadline:
        listed = httpx.get(
            f"{stack.base_url}/admin/models", headers=admin_headers(), timeout=10
        ).json()
        by_id = {m["id"]: m for m in listed["models"]}
        status = by_id.get(model_id, {}).get("status", "<absent>")
        if status == wanted:
            return
        time.sleep(1)
    pytest.fail(f"model '{model_id}' never reached '{wanted}' (last: {status})")


def test_add_local_model_live_with_hand_edit_preserved():
    """A hand-edited kiln.toml survives a concurrent add-model write, the
    added model loads and serves with zero gateway restart, and duplicates
    are rejected clearly."""
    path = model_dir()
    if path is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run ./scripts/fetch-test-model.sh"
        )
    with add_stack([(MODEL_ID, "rust")]) as (stack, _dest_root):
        config_path = stack.runtime_dir / "kiln.toml"

        # Hand-edit the config while the gateway is running: the add-model
        # write must preserve this even though the gateway never read it.
        hand_edited = (
            config_path.read_text()
            + "\n# hand-edited while the gateway was running — must survive\n"
        )
        config_path.write_text(hand_edited)

        response = httpx.post(
            f"{stack.base_url}/admin/models",
            headers=admin_headers(),
            json={"id": "added-llama", "path": str(path), "worker": "auto"},
            timeout=30,
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["model"]["status"] == "unloaded (registered)"
        assert body["persisted_to"] == str(config_path)

        # On disk: the hand edit survived and exactly one block was added.
        after = config_path.read_text()
        assert "# hand-edited while the gateway was running" in after
        inserted = assert_only_inserted(hand_edited, after)
        assert any('id = "added-llama"' in line for line in inserted), inserted
        parsed = tomllib.loads(after)
        assert [m["id"] for m in parsed["model"]] == [MODEL_ID, "added-llama"]
        assert parsed["model"][1]["path"] == str(path)

        # Live: both models listed on the admin AND public surfaces, no
        # restart involved.
        listed = httpx.get(
            f"{stack.base_url}/admin/models", headers=admin_headers(), timeout=10
        ).json()
        assert {m["id"] for m in listed["models"]} == {MODEL_ID, "added-llama"}
        public = httpx.get(
            f"{stack.base_url}/v1/models",
            headers={"Authorization": f"Bearer {stack.api_key}"},
            timeout=10,
        ).json()
        assert "added-llama" in {m["id"] for m in public["data"]}

        # Duplicate id: clear 409, config untouched.
        response = httpx.post(
            f"{stack.base_url}/admin/models",
            headers=admin_headers(),
            json={"id": "added-llama", "path": str(path)},
            timeout=30,
        )
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "model_exists"
        assert config_path.read_text() == after

        # Hand-added [[model]] that the gateway has never read: adding the
        # same id must refuse rather than write a duplicate block.
        config_path.write_text(
            after + f'\n[[model]]\nid = "hand-model"\npath = "{path}"\n'
        )
        response = httpx.post(
            f"{stack.base_url}/admin/models",
            headers=admin_headers(),
            json={"id": "hand-model", "path": str(path)},
            timeout=30,
        )
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "config_conflict"

        # The estimate endpoint prices the local dir against the live
        # ledger: the resident model's footprint has already eaten budget.
        estimate = httpx.get(
            f"{stack.base_url}/admin/models/estimate",
            headers=admin_headers(),
            params={"path": str(path)},
            timeout=10,
        ).json()
        assert estimate["source"] == "local"
        assert estimate["weights_bytes"] > 500 * 1024 * 1024  # ~680 MB of weights
        assert estimate["headroom_bytes"] < estimate["budget_bytes"]

        # The added model becomes servable through the existing load
        # machinery — same process throughout.
        response = httpx.post(
            f"{stack.base_url}/admin/models/added-llama/load",
            headers=admin_headers(),
            timeout=10,
        )
        assert response.status_code == 202, response.text
        wait_for_status(stack, "added-llama", "ready", timeout=300)
        completion = httpx.post(
            f"{stack.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {stack.api_key}"},
            json={
                "model": "added-llama",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8,
            },
            timeout=120,
        )
        assert completion.status_code == 200, completion.text
        assert completion.json()["usage"]["completion_tokens"] > 0
        assert stack.gateway.poll() is None, "gateway must not have restarted"


def test_add_model_download_flow_over_the_api(stub_hub):  # noqa: F811
    """The not-downloaded path end-to-end at the API level: hub estimate,
    structured 409 with download coordinates, the standard download job,
    and the retried add resolving to the downloaded files."""
    with add_stack([], stub_hub) as (stack, dest_root):
        # Size estimate straight from the hub listing (HF_ENDPOINT).
        estimate = httpx.get(
            f"{stack.base_url}/admin/models/estimate",
            headers=admin_headers(),
            params={"path": "stub/tiny"},
            timeout=30,
        ).json()
        assert estimate["source"] == "hub"
        assert estimate["weights_bytes"] == len(STUB_FILES["model.safetensors"])
        assert estimate["fits"] is True

        # Adding before downloading: structured 409 carrying the exact
        # coordinates for the job (dest under the configured model_dir).
        payload = {"id": "tiny", "path": "stub/tiny", "worker": "python"}
        response = httpx.post(
            f"{stack.base_url}/admin/models",
            headers=admin_headers(),
            json=payload,
            timeout=30,
        )
        assert response.status_code == 409, response.text
        body = response.json()
        assert body["error"]["code"] == "model_not_downloaded"
        assert body["download"]["repo"] == "stub/tiny"
        dest = body["download"]["dest"]
        assert dest == str(dest_root / "stub--tiny")

        # The standard Phase 10 download job with those coordinates.
        response = httpx.post(
            f"{stack.base_url}/admin/jobs/download",
            headers=admin_headers(),
            json={"repo": "stub/tiny", "dest": dest},
            timeout=30,
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        deadline = time.monotonic() + 60
        state = "queued"
        while time.monotonic() < deadline and state in ("queued", "running"):
            state = (
                httpx.get(
                    f"{stack.base_url}/admin/jobs/{job_id}",
                    headers=admin_headers(),
                    timeout=10,
                )
                .json()
                .get("state")
            )
            time.sleep(0.3)
        assert state == "succeeded"

        # The retried add now resolves to the downloaded dest and
        # registers; kiln.toml records the LOCAL path (what the next boot
        # can load), not the repo id.
        response = httpx.post(
            f"{stack.base_url}/admin/models",
            headers=admin_headers(),
            json=payload,
            timeout=30,
        )
        assert response.status_code == 201, response.text
        assert response.json()["model"]["status"] == "unloaded (registered)"
        parsed = tomllib.loads((stack.runtime_dir / "kiln.toml").read_text())
        assert parsed["model"][0]["id"] == "tiny"
        assert parsed["model"][0]["path"] == dest
        assert parsed["model"][0]["worker"] == "python"


def test_add_model_requires_admin_token(stub_hub):  # noqa: F811
    with add_stack([], stub_hub) as (stack, _dest_root):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = httpx.post(
                f"{stack.base_url}/admin/models",
                headers=headers,
                json={"id": "x", "path": "a/b"},
                timeout=10,
            )
            assert response.status_code == 401, response.text
