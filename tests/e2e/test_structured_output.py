"""Structured output over the OpenAI API (SPEC §8.1 `response_format`,
§11.3, §12 Phase 7): `json_schema` and `json_object` requests against the
rust worker yield schema-valid JSON (llguidance logit masking in the
worker decode step); the python worker lacks CAPABILITY_GRAMMAR in v1, so
the gateway rejects structured-output requests routed to it with a 400
(SPEC §5 capability gating).

The heavyweight 100/100 acceptance sweep lives in the worker-level suite
(crates/kiln-worker/tests/grammar.rs); this file covers the HTTP surface
with the real `openai` SDK.
"""

from __future__ import annotations

import json

import openai
import pytest

SCHEMA = {
    # Compact JSON + bounded fields: output length is structurally bounded.
    "x-guidance": {"whitespace_flexible": False},
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "kind": {"type": "string", "enum": ["cat", "dog", "bird"]},
        "age": {"type": "integer", "minimum": 0, "maximum": 30},
    },
    "required": ["name", "kind", "age"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "pet", "schema": SCHEMA, "strict": True},
}


def _assert_schema_valid(text: str):
    value = json.loads(text)
    assert isinstance(value, dict), text
    assert set(value) == {"name", "kind", "age"}, text
    assert isinstance(value["name"], str) and len(value["name"]) <= 12
    assert value["kind"] in ("cat", "dog", "bird")
    assert isinstance(value["age"], int) and 0 <= value["age"] <= 30


def test_json_schema_response_format(stack, client):
    if stack.worker_kind == "python":
        # SPEC §5: the python worker lacks CAPABILITY_GRAMMAR in v1; the
        # gateway rejects before routing.
        with pytest.raises(openai.BadRequestError) as excinfo:
            client.chat.completions.create(
                model=stack.model_id,
                messages=[{"role": "user", "content": "Describe a pet."}],
                response_format=RESPONSE_FORMAT,
                max_tokens=128,
            )
        assert "structured output" in str(excinfo.value)
        return

    # Full-temperature with distinct seeds: the grammar mask, not the
    # model, must carry validity. 192 tokens comfortably clears the
    # schema's structural worst case (~105 tokens all-escaped).
    for seed in range(1, 7):
        completion = client.chat.completions.create(
            model=stack.model_id,
            messages=[{"role": "user", "content": "Describe one pet as JSON."}],
            response_format=RESPONSE_FORMAT,
            temperature=1.0,
            seed=seed,
            max_tokens=192,
        )
        choice = completion.choices[0]
        assert choice.finish_reason == "stop", choice
        _assert_schema_valid(choice.message.content)


def test_json_schema_streaming(stack, client):
    if stack.worker_kind == "python":
        pytest.skip("python worker lacks CAPABILITY_GRAMMAR in v1")
    chunks = client.chat.completions.create(
        model=stack.model_id,
        messages=[{"role": "user", "content": "Describe one pet as JSON."}],
        response_format=RESPONSE_FORMAT,
        temperature=1.0,
        seed=42,
        max_tokens=192,
        stream=True,
    )
    text = ""
    finish = None
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        text += choice.delta.content or ""
        finish = choice.finish_reason or finish
    assert finish == "stop"
    _assert_schema_valid(text)


def test_json_object_mode(stack, client):
    if stack.worker_kind == "python":
        pytest.skip("python worker lacks CAPABILITY_GRAMMAR in v1")
    # json_object has no structural length bound (any object satisfies
    # it), so bias the model toward a minimal answer: near-greedy plus a
    # brevity instruction, with generous headroom.
    completion = client.chat.completions.create(
        model=stack.model_id,
        messages=[
            {"role": "system", "content": "You reply with minimal, short JSON."},
            {"role": "user", "content": "Give me a JSON object with one short field."},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        seed=3,
        max_tokens=512,
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    assert isinstance(json.loads(choice.message.content), dict)


def test_bad_response_format_is_400(stack, client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=stack.model_id,
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "yaml"},
            max_tokens=8,
        )
    # json_schema without a schema payload is rejected in validation,
    # for both worker kinds.
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=stack.model_id,
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_schema"},
            max_tokens=8,
        )
