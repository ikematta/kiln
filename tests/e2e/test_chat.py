"""OpenAI SDK conformance: models list, auth, non-stream + stream chat
(Phase 2 acceptance, SPEC §12)."""

from __future__ import annotations

import openai
import pytest


def test_models_list(stack, client):
    models = client.models.list()
    assert models.object == "list"
    ids = [m.id for m in models.data]
    assert stack.model_id in ids
    entry = next(m for m in models.data if m.id == stack.model_id)
    assert entry.object == "model"
    assert entry.owned_by == "kiln"


def test_auth_rejects_bad_and_missing_keys(stack):
    bad = openai.OpenAI(
        base_url=f"{stack.base_url}/v1", api_key="wrong-key", max_retries=0
    )
    with pytest.raises(openai.AuthenticationError):
        bad.models.list()

    import httpx

    response = httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        json={"model": stack.model_id, "messages": [{"role": "user", "content": "x"}]},
        timeout=10,
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "invalid_api_key"
    assert body["error"]["type"] == "invalid_request_error"


def test_chat_completion_non_streaming(stack, client):
    completion = client.chat.completions.create(
        model=stack.model_id,
        messages=[
            {"role": "system", "content": "You answer with single words."},
            {"role": "user", "content": "Name any one primary color."},
        ],
        temperature=0,
        seed=7,
        max_tokens=32,
    )
    assert completion.id.startswith("chatcmpl-")
    assert completion.object == "chat.completion"
    assert completion.model == stack.model_id
    choice = completion.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content and choice.message.content.strip()
    assert choice.finish_reason in ("stop", "length")
    usage = completion.usage
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_greedy_is_reproducible(stack, client):
    def once() -> str:
        return (
            client.chat.completions.create(
                model=stack.model_id,
                messages=[
                    {"role": "user", "content": "Write one sentence about kilns."}
                ],
                temperature=0,
                max_tokens=48,
            )
            .choices[0]
            .message.content
        )

    # CLAUDE.md determinism rule: greedy decoding is reproducible run-to-run.
    assert once() == once()


def test_chat_completion_streaming(stack, client):
    chunks = list(
        client.chat.completions.create(
            model=stack.model_id,
            messages=[{"role": "user", "content": "Count from 1 to 5, digits only."}],
            temperature=0,
            max_tokens=48,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert all(c.object == "chat.completion.chunk" for c in chunks)
    # One shared id across the stream.
    assert len({c.id for c in chunks}) == 1

    # Role preamble first.
    first = chunks[0]
    assert first.choices[0].delta.role == "assistant"

    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "1" in content and "5" in content

    finishes = [
        c.choices[0].finish_reason
        for c in chunks
        if c.choices and c.choices[0].finish_reason
    ]
    assert finishes == ["stop"] or finishes == ["length"]

    # include_usage: exactly one terminal usage chunk with empty choices.
    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[-1] is chunks[-1]
    assert not usage_chunks[0].choices
    assert usage_chunks[0].usage.prompt_tokens > 0
    assert usage_chunks[0].usage.completion_tokens > 0


def test_validation_errors_are_openai_shaped(stack, client):
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(
            model="no-such-model", messages=[{"role": "user", "content": "x"}]
        )
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=stack.model_id,
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=stack.model_id, messages=[])
