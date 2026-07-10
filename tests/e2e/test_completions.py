"""Phase 6: `POST /v1/completions` (SPEC §8.1) via the real openai SDK,
against both worker kinds (the parametrized `stack` fixture). The raw
prompt goes to the model without a chat template; special tokens (BOS) are
supplied by the tokenizer, matching mlx-lm's raw-generate contract."""

from __future__ import annotations

import openai
import pytest

from test_metrics import _counter

PROMPT = "The capital of France is"


def test_completion_non_streaming(stack, client):
    before = stack.metrics_text()
    response = client.completions.create(
        model=stack.model_id,
        prompt=PROMPT,
        temperature=0,
        max_tokens=8,
    )
    assert response.object == "text_completion"
    assert response.id.startswith("cmpl-")
    assert response.model == stack.model_id
    choice = response.choices[0]
    assert choice.text.strip(), "empty completion"
    assert choice.finish_reason in ("stop", "length")
    usage = response.usage
    assert usage.prompt_tokens > 0
    assert 1 <= usage.completion_tokens <= 8
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    if choice.finish_reason == "length":
        assert usage.completion_tokens == 8

    after = stack.metrics_text()
    assert (
        _counter(after, "kiln_completions_total", model=stack.model_id, outcome="ok")
        >= _counter(
            before, "kiln_completions_total", model=stack.model_id, outcome="ok"
        )
        + 1
    )


def test_completion_default_max_tokens_is_16(stack, client):
    # OpenAI legacy default: 16 generated tokens when max_tokens is absent
    # (chat fills the context instead — the endpoints differ by design).
    response = client.completions.create(
        model=stack.model_id, prompt=PROMPT, temperature=0
    )
    assert 1 <= response.usage.completion_tokens <= 16
    if response.choices[0].finish_reason == "length":
        assert response.usage.completion_tokens == 16


def test_completion_streaming_matches_non_streaming(stack, client):
    kwargs = dict(
        model=stack.model_id,
        prompt="Count from 1 to 20, separated by commas:",
        temperature=0,
        max_tokens=24,
    )
    full = client.completions.create(**kwargs)

    parts, finish, usage = [], None, None
    for chunk in client.completions.create(
        stream=True, stream_options={"include_usage": True}, **kwargs
    ):
        assert chunk.object == "text_completion"
        for choice in chunk.choices:
            parts.append(choice.text or "")
            if choice.finish_reason:
                finish = choice.finish_reason
        if chunk.usage:
            usage = chunk.usage

    # Greedy decoding is deterministic per worker, so stream == non-stream.
    assert "".join(parts) == full.choices[0].text
    assert finish == full.choices[0].finish_reason
    assert usage is not None, "include_usage must yield a final usage chunk"
    assert usage.prompt_tokens == full.usage.prompt_tokens
    assert usage.completion_tokens == full.usage.completion_tokens


def test_completion_stop_string(stack, client):
    response = client.completions.create(
        model=stack.model_id,
        prompt="Count: 1, 2, 3, 4, 5, 6, 7,",
        temperature=0,
        max_tokens=32,
        stop=["9"],
    )
    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert "9" not in choice.text, f"stop text leaked: {choice.text!r}"
    assert "8" in choice.text, f"ended too early: {choice.text!r}"


def test_completion_rejections(stack, client):
    for kwargs in (
        {"prompt": ["a", "b"]},  # multiple prompts
        {"prompt": [1, 2, 3]},  # token ids
        {"prompt": PROMPT, "echo": True},
        {"prompt": PROMPT, "n": 2},
        {"prompt": PROMPT, "best_of": 4},
        {"prompt": PROMPT, "logprobs": 3},
        {"prompt": PROMPT, "suffix": "The end."},
    ):
        with pytest.raises(openai.BadRequestError):
            client.completions.create(model=stack.model_id, **kwargs)


def test_completion_unknown_model_is_404(stack, client):
    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="no-such-model", prompt=PROMPT)
