"""Phase 6 acceptance (SPEC §12): gateway worker="auto" routing.

One gateway serves the same pinned checkpoint twice under worker="auto":
once from the real model directory (servable -> rust worker), once from a
doctored copy whose config.json carries an intentionally unsupported
quantization mode — a per-module override entry, the mixed-precision form
the rust worker rejects by name (SPEC §7.3). The override's parameters
EQUAL the uniform ones and the weights are symlinks to the same files, so
mlx-lm loads the doctored copy bit-identically; only the routing decision
differs. The unsupported model must transparently route to and serve
correctly from the python worker: same API surface, and greedy output
identical to the rust-served twin.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile

import pytest

from conftest import model_dir, running_stack

RUST_ROUTED = "auto-supported"
PYTHON_ROUTED = "auto-unsupported-quant"

PROMPT = "What is a kiln used for? Answer in one sentence."


def doctored_model_dir(base: pathlib.Path) -> pathlib.Path:
    """A copy of the pinned model whose quantization block gains a
    per-module override with the SAME parameters as the uniform block:
    rejected by the rust matrix, loaded identically by mlx-lm. Weights and
    tokenizer files are symlinked, not copied."""
    doctored = pathlib.Path(
        tempfile.mkdtemp(prefix="kiln-e2e-unsupported-quant-", dir="/tmp")
    )
    for item in base.iterdir():
        if item.name != "config.json":
            os.symlink(item, doctored / item.name)
    config = json.loads((base / "config.json").read_text())
    uniform = dict(config["quantization"])
    assert set(uniform) >= {"group_size", "bits"}, uniform
    config["quantization"] = {**uniform, "model.embed_tokens": uniform}
    (doctored / "config.json").write_text(json.dumps(config))
    return doctored


@pytest.fixture(scope="module")
def routed_stack():
    base = model_dir()
    if base is None:
        pytest.skip("pinned test model not found; run ./scripts/fetch-test-model.sh")
    doctored = doctored_model_dir(base)
    try:
        with running_stack(
            [
                (RUST_ROUTED, "auto"),
                (PYTHON_ROUTED, "auto", str(doctored)),
            ]
        ) as stack:
            stack.wait_ready()
            yield stack
    finally:
        shutil.rmtree(doctored, ignore_errors=True)


@pytest.fixture(scope="module")
def routed_client(routed_stack):
    from openai import OpenAI

    return OpenAI(
        base_url=f"{routed_stack.base_url}/v1",
        api_key=routed_stack.api_key,
        timeout=120.0,
        max_retries=0,
    )


def test_auto_resolves_by_the_support_matrix(routed_stack):
    """The servable config got a rust worker process; the unsupported one a
    python worker process — identified by each worker's unique socket arg."""
    rust_cmd = routed_stack.worker_command(RUST_ROUTED)
    python_cmd = routed_stack.worker_command(PYTHON_ROUTED)
    assert rust_cmd and "kiln_worker_py" not in rust_cmd, (
        f"supported model must be served by the rust worker, got: {rust_cmd!r}"
    )
    assert "kiln_worker_py" in python_cmd, (
        f"unsupported-quant model must be served by the python worker, "
        f"got: {python_cmd!r}"
    )

    log = routed_stack.log_path.read_text()
    assert "worker=auto resolved to rust" in log
    assert "resolved to python (rust worker cannot serve this model)" in log


def test_unsupported_model_serves_transparently_from_python(routed_client):
    """The client sees a plain working model — no hint that routing fell
    back — and, since the doctored checkpoint's weights are the same files,
    greedy output must match the rust-served twin exactly (the cross-worker
    parity invariant)."""
    results = {}
    for model in (RUST_ROUTED, PYTHON_ROUTED):
        response = routed_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=48,
        )
        choice = response.choices[0]
        assert choice.message.content, f"{model}: empty completion"
        results[model] = (
            choice.message.content,
            choice.finish_reason,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    assert results[PYTHON_ROUTED] == results[RUST_ROUTED], (
        f"routed twins diverged:\n"
        f"  rust:   {results[RUST_ROUTED]!r}\n"
        f"  python: {results[PYTHON_ROUTED]!r}"
    )


def test_unsupported_model_serves_completions_too(routed_client):
    """/v1/completions rides the same routing: raw-prompt completion from
    the python-routed model works and matches the rust-routed twin."""
    results = {}
    for model in (RUST_ROUTED, PYTHON_ROUTED):
        response = routed_client.completions.create(
            model=model,
            prompt="A kiln is",
            temperature=0,
            max_tokens=16,
        )
        assert response.choices[0].text, f"{model}: empty completion"
        results[model] = (
            response.choices[0].text,
            response.choices[0].finish_reason,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    assert results[PYTHON_ROUTED] == results[RUST_ROUTED], (
        f"routed twins diverged on /v1/completions:\n"
        f"  rust:   {results[RUST_ROUTED]!r}\n"
        f"  python: {results[PYTHON_ROUTED]!r}"
    )
