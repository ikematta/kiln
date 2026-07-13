"""Tool calling over the OpenAI API (SPEC §8.2, §11.3, §12 Phase 7).

Round-trips a real tool-call request through the real `openai` SDK against
both worker kinds (the session `stack` is parametrized python/rust and both
support tool calls — the gateway owns template rendering and tool-call
parsing for both). The pinned Llama-3.2-1B exercises the Llama python-tag
parser; a rust-worker Qwen3-0.6B stack exercises the Hermes parser (the
XML format has no pinned model and is covered by the kiln-tokenize fixture
suite).

Assertions match the captured parser fixtures
(crates/kiln-tokenize/tests/fixtures/toolcall/): greedy Llama emits
`<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}`
for this prompt, and greedy Qwen3 thinks, then emits the Hermes block.
The Llama prompt embeds today's date (the template's strftime_now), so
exact token output may drift day to day — the asserted call shape is the
stable part. Streaming and non-streaming must agree exactly: same worker
determinism argument as the cross-worker parity suite.
"""

from __future__ import annotations

import json

import pytest

from conftest import QWEN_MODEL_ID

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

USER_MSG = {"role": "user", "content": "What is the weather in Paris right now?"}

FORBIDDEN_MARKERS = ("<|python_tag|>", "<tool_call>", "</tool_call>", "<function=")


def request_kwargs(model_id: str, stream: bool = False) -> dict:
    return {
        "model": model_id,
        "messages": [USER_MSG],
        "tools": TOOLS,
        "temperature": 0,  # greedy: streaming/non-streaming must agree
        "max_tokens": 512,
        "stream": stream,
    }


def assert_weather_call(message, *, expect_content_think: bool = False):
    """The response is a well-formed OpenAI tool call for get_weather with
    no raw family markers leaked into content."""
    calls = message.tool_calls
    assert calls, f"expected tool_calls, got content: {message.content!r}"
    assert len(calls) == 1, calls
    call = calls[0]
    assert call.id.startswith("call_"), call.id
    assert call.type == "function"
    assert call.function.name == "get_weather"
    args = json.loads(call.function.arguments)
    assert args == {"city": "Paris"}, args
    content = message.content or ""
    for marker in FORBIDDEN_MARKERS:
        assert marker not in content, f"raw marker {marker!r} leaked: {content!r}"
    if expect_content_think:
        # Qwen3 thinks before calling; the think block is plain content on
        # this endpoint (Anthropic-shape thinking blocks are /v1/messages,
        # Phase 7 part 3).
        assert "</think>" in content, content
    else:
        assert content == "", content


def collect_stream(chunks):
    """Accumulates SSE deltas the way an SDK user must: content by
    concatenation, tool calls keyed by index. Also asserts the OpenAI
    delta shape: the first delta for a call carries id/type/name and empty
    arguments, later deltas only argument fragments."""
    content = ""
    calls: dict[int, dict] = {}
    finish = None
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        delta = choice.delta
        content += delta.content or ""
        for tc in delta.tool_calls or []:
            if tc.index not in calls:
                assert tc.id and tc.id.startswith("call_"), tc
                assert tc.type == "function"
                assert tc.function.name
                assert tc.function.arguments == "", tc
                calls[tc.index] = {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": "",
                }
            else:
                assert tc.id is None and tc.function.name is None, tc
                calls[tc.index]["arguments"] += tc.function.arguments or ""
    assert sorted(calls) == list(range(len(calls))), calls
    return content, [calls[i] for i in sorted(calls)], finish


def test_tool_call_non_streaming(stack, client):
    completion = client.chat.completions.create(**request_kwargs(stack.model_id))
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    assert_weather_call(choice.message)
    # Tool-calls-only turn: content is null, not "".
    assert choice.message.content is None


def test_tool_call_streaming_matches_non_streaming(stack, client):
    reference = client.chat.completions.create(**request_kwargs(stack.model_id))
    chunks = client.chat.completions.create(
        **request_kwargs(stack.model_id, stream=True)
    )
    content, calls, finish = collect_stream(chunks)

    assert finish == "tool_calls"
    assert content == ""
    assert len(calls) == 1
    ref_call = reference.choices[0].message.tool_calls[0]
    assert calls[0]["name"] == ref_call.function.name
    # The invariant from the fixture suite, end-to-end: deltas reassemble
    # to exactly what the non-streaming path returns.
    assert calls[0]["arguments"] == ref_call.function.arguments


def test_tool_round_trip(stack, client):
    first = client.chat.completions.create(**request_kwargs(stack.model_id))
    message = first.choices[0].message
    assert_weather_call(message)

    call = message.tool_calls[0]
    followup = client.chat.completions.create(
        model=stack.model_id,
        messages=[
            USER_MSG,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"temperature_c": 21, "sky": "clear"}',
            },
        ],
        tools=TOOLS,
        temperature=0,
        max_tokens=512,
    )
    choice = followup.choices[0]
    # The 1B model may answer in text or (captured fixture behavior:
    # llama-after-tool-result) call the tool again — both are well-formed
    # outcomes; raw markers leaking into content are not.
    assert choice.finish_reason in ("stop", "tool_calls"), choice
    content = choice.message.content or ""
    for marker in FORBIDDEN_MARKERS:
        assert marker not in content, f"raw marker {marker!r} leaked: {content!r}"
    if choice.finish_reason == "tool_calls":
        assert choice.message.tool_calls[0].function.name == "get_weather"
        json.loads(choice.message.tool_calls[0].function.arguments)
    else:
        assert content.strip(), choice


def test_tool_choice_none_disables_tools(stack, client):
    completion = client.chat.completions.create(
        model=stack.model_id,
        messages=[USER_MSG],
        tools=TOOLS,
        tool_choice="none",
        temperature=0,
        max_tokens=64,
    )
    choice = completion.choices[0]
    assert choice.message.tool_calls is None
    assert choice.message.content


def test_forced_tool_choice_is_400(stack, client):
    import openai

    with pytest.raises(openai.BadRequestError) as excinfo:
        client.chat.completions.create(
            model=stack.model_id,
            messages=[USER_MSG],
            tools=TOOLS,
            tool_choice="required",
            max_tokens=8,
        )
    assert "tool_choice" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Hermes format: Qwen3-0.6B on the rust worker (stack fixture in conftest,
# shared with test_messages.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_client(qwen_stack):
    from openai import OpenAI

    return OpenAI(
        base_url=f"{qwen_stack.base_url}/v1",
        api_key=qwen_stack.api_key,
        timeout=120.0,
        max_retries=0,
    )


def test_hermes_tool_call_streaming(qwen_stack, qwen_client):
    reference = qwen_client.chat.completions.create(**request_kwargs(QWEN_MODEL_ID))
    choice = reference.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    assert_weather_call(choice.message, expect_content_think=True)

    chunks = qwen_client.chat.completions.create(
        **request_kwargs(QWEN_MODEL_ID, stream=True)
    )
    content, calls, finish = collect_stream(chunks)
    assert finish == "tool_calls"
    assert content == choice.message.content
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == choice.message.tool_calls[0].function.arguments
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
