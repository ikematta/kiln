"""Phase 8 config wiring (SPEC §6.5/§10): [model.speculative] end-to-end.

A gateway whose kiln.toml carries a [model.speculative] block must spawn
the rust worker with the matching `--draft-model`/`--draft-gamma` argv,
reach READY (the worker enforces the ADR 0005 compat/envelope gates at
attach), and serve normally with the draft/verify loop live. Uses the
tokenizer-compatible pinned pair from the worker-level draft suite:
qwen3-0.6b-8bit target, qwen3-0.6b-4bit draft.

Greedy on-vs-off identity is proven at the engine level (spec_decode) and
the worker level (draft.rs); this suite covers the config path: kiln.toml
-> registry -> supervisor argv -> serving worker. The pair is a
throughput LOSS by design of the pinned fleet (ADR 0006 — sub-1B pairs
lack the size asymmetry speculation needs); correctness, not speed, is
what serving asserts here.
"""

from __future__ import annotations

import pytest

from conftest import pinned_model_dir, running_stack

TARGET_ID = "qwen3-0.6b-8bit"
DRAFT_ID = "qwen3-0.6b-4bit"
SPEC_MODEL = "spec-wired"
GAMMA = 4


@pytest.fixture(scope="module")
def spec_stack():
    target = pinned_model_dir(TARGET_ID)
    draft = pinned_model_dir(DRAFT_ID)
    if target is None or draft is None:
        pytest.skip(
            f"pinned test models '{TARGET_ID}'/'{DRAFT_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    extra = f"""
[model.speculative]
draft = "{draft}"
gamma = {GAMMA}
"""
    with running_stack([(SPEC_MODEL, "rust", str(target), extra)]) as stack:
        stack.wait_ready()
        yield stack


def test_speculative_config_reaches_the_worker_argv(spec_stack):
    """The [model.speculative] block becomes real spawn argv: the worker
    process serving the model carries --draft-model with the resolved
    draft path and --draft-gamma with the configured value."""
    draft = pinned_model_dir(DRAFT_ID)
    cmd = spec_stack.worker_command(SPEC_MODEL)
    assert cmd, "no worker process found for the speculative model"
    assert f"--draft-model {draft}" in cmd, (
        f"worker argv must carry the resolved draft path: {cmd!r}"
    )
    assert f"--draft-gamma {GAMMA}" in cmd, (
        f"worker argv must carry the configured gamma: {cmd!r}"
    )

    log = spec_stack.log_path.read_text()
    assert "draft model loaded" in log, "the worker must report the attached draft"
    assert "speculation envelope applied" in log, (
        "the ADR 0005 envelope clamp must be logged at attach"
    )


def test_speculative_stack_serves_greedy_completions(spec_stack):
    """The speculating stack serves a plain greedy chat completion — the
    draft/verify loop is transparent to the API surface."""
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{spec_stack.base_url}/v1",
        api_key=spec_stack.api_key,
        timeout=120.0,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=SPEC_MODEL,
        messages=[{"role": "user", "content": "What is a kiln used for?"}],
        temperature=0,
        max_tokens=32,
    )
    choice = response.choices[0]
    assert choice.message.content, "empty completion from the speculating stack"
    assert response.usage.completion_tokens > 0
