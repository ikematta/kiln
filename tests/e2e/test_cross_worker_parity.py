"""Cross-worker output parity (Phase 3 closeout acceptance): the same model
served by the Rust worker and the Python worker through one gateway must
produce IDENTICAL greedy output text.

This is a live, full-stack check — distinct from the golden token-id
harness, which compares the Rust implementation against committed mlx-lm
fixtures. Bit-identical greedy decoding across workers is expected, not
hoped for: both run core MLX v0.31.1 (ADR 0001 follow-up B1) and both
tokenize from the same tokenizer.json.
"""

from __future__ import annotations

import pytest

from conftest import model_dir, running_stack

PY_MODEL = "llama-parity-python"
RS_MODEL = "llama-parity-rust"

PROMPTS = [
    "What is a kiln used for? Answer in one sentence.",
    "Count from 1 to 5, one number per line.",
    "Translate to French: the kiln 窯 is very hot \U0001f525.",
]


@pytest.fixture(scope="module")
def parity_client():
    if model_dir() is None:
        pytest.skip("pinned test model not found; run ./scripts/fetch-test-model.sh")
    with running_stack([(PY_MODEL, "python"), (RS_MODEL, "rust")]) as stack:
        stack.wait_ready()
        from openai import OpenAI

        yield OpenAI(
            base_url=f"{stack.base_url}/v1",
            api_key=stack.api_key,
            timeout=120.0,
            max_retries=0,
        )


def _complete(client, model: str, prompt: str, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        **kwargs,
    )
    choice = response.choices[0]
    return (
        choice.message.content,
        choice.finish_reason,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def _stream(client, model: str, prompt: str):
    text, finish = "", None
    events = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        stream=True,
    )
    for event in events:
        for choice in event.choices:
            if choice.delta and choice.delta.content:
                text += choice.delta.content
            if choice.finish_reason:
                finish = choice.finish_reason
    return text, finish


def test_greedy_outputs_identical_across_workers(parity_client):
    for prompt in PROMPTS:
        py = _complete(parity_client, PY_MODEL, prompt)
        rs = _complete(parity_client, RS_MODEL, prompt)
        assert rs == py, (
            f"cross-worker divergence for prompt {prompt!r}:\n"
            f"  python: {py!r}\n  rust:   {rs!r}"
        )
        assert py[0], "empty completion"


def test_streaming_text_identical_across_workers(parity_client):
    prompt = PROMPTS[0]
    py = _stream(parity_client, PY_MODEL, prompt)
    rs = _stream(parity_client, RS_MODEL, prompt)
    assert rs == py, f"streaming divergence:\n  python: {py!r}\n  rust:   {rs!r}"


def test_stop_strings_work_on_both_workers(parity_client):
    # Worker-side matching (python) vs gateway-side matching (rust) must
    # agree on EVERYTHING: matched text excluded, finish_reason "stop", and
    # identical usage — completion_tokens must count tokens up to and
    # including the match, not the rust worker's cancel overshoot (the
    # gateway overrides it; see kiln-gateway chat.rs module docs).
    prompt = "Count from 1 to 9, one number per line."
    results = {}
    for model in (PY_MODEL, RS_MODEL):
        content, finish, prompt_tokens, completion_tokens = _complete(
            parity_client, model, prompt, stop=["4"]
        )
        assert finish == "stop", model
        assert "4" not in (content or ""), (
            f"{model}: stop text leaked into content: {content!r}"
        )
        assert "3" in (content or ""), f"{model}: content ended too early: {content!r}"
        results[model] = (content, finish, prompt_tokens, completion_tokens)
    assert results[RS_MODEL] == results[PY_MODEL], (
        f"stop-string usage/text divergence:\n"
        f"  python: {results[PY_MODEL]!r}\n  rust:   {results[RS_MODEL]!r}"
    )
