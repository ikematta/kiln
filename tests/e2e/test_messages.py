"""Anthropic Messages API conformance (`POST /v1/messages`, SPEC §8.1, §11.3).

Drives Kiln through the real `anthropic` SDK — the SDK's pydantic models
validate every response and stream event shape strictly, so a passing run
IS the wire-format conformance check. The session `stack` is parametrized
over both worker kinds (python/rust serving the pinned Llama-3.2-1B); the
shared rust-worker Qwen3-0.6B stack covers the thinking-trained path:
`<think>` regions must arrive as `thinking` content blocks, separated from
`text`, on this endpoint (on the OpenAI endpoint they are plain content —
asserted in test_tool_calls).

All generation is greedy (temperature=0): streaming must reassemble to
exactly the non-streaming response, same as the OpenAI-suite invariants.
"""

from __future__ import annotations

import httpx
import pytest

from conftest import QWEN_MODEL_ID

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }
]

WEATHER_MSG = {"role": "user", "content": "What is the weather in Paris right now?"}

FORBIDDEN_MARKERS = (
    "<|python_tag|>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "<think>",
    "</think>",
)


def anthropic_client(stack):
    import anthropic

    # The SDK appends /v1/messages itself; base_url is the server root.
    # max_retries=0: assertions must see the raw first response.
    return anthropic.Anthropic(
        base_url=stack.base_url,
        api_key=stack.api_key,
        timeout=120.0,
        max_retries=0,
    )


@pytest.fixture(scope="session")
def messages_client(stack):
    return anthropic_client(stack)


@pytest.fixture(scope="session")
def qwen_messages_client(qwen_stack):
    return anthropic_client(qwen_stack)


def msg_kwargs(model_id: str, **overrides) -> dict:
    kwargs = {
        "model": model_id,
        "max_tokens": 256,
        "temperature": 0.0,  # greedy: streaming/non-streaming must agree
        "messages": [
            {"role": "user", "content": "In one short sentence, what is a kiln?"}
        ],
    }
    kwargs.update(overrides)
    return kwargs


def assert_no_marker_leak(message):
    for block in message.content:
        text = (
            block.thinking
            if block.type == "thinking"
            else (block.text if block.type == "text" else "")
        )
        for marker in FORBIDDEN_MARKERS:
            assert marker not in text, f"raw marker {marker!r} leaked: {text!r}"


# ---------------------------------------------------------------------------
# Chat: non-streaming, streaming, system prompts, stop sequences
# ---------------------------------------------------------------------------


def test_non_streaming_message_shape(stack, messages_client):
    response = messages_client.messages.create(**msg_kwargs(stack.model_id))
    assert response.type == "message"
    assert response.role == "assistant"
    assert response.id.startswith("msg_")
    assert response.model == stack.model_id
    assert response.stop_reason in ("end_turn", "max_tokens")
    assert response.stop_sequence is None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.content, response
    assert all(block.type == "text" for block in response.content)
    assert response.content[0].text.strip()
    assert_no_marker_leak(response)


def test_streaming_matches_non_streaming(stack, messages_client):
    reference = messages_client.messages.create(**msg_kwargs(stack.model_id))
    with messages_client.messages.stream(**msg_kwargs(stack.model_id)) as stream:
        final = stream.get_final_message()
    # The SDK accumulated content_block_start/delta/stop into the final
    # message; greedy determinism means byte equality with non-streaming.
    assert [b.type for b in final.content] == [b.type for b in reference.content]
    assert final.content[0].text == reference.content[0].text
    assert final.stop_reason == reference.stop_reason
    assert final.usage.input_tokens == reference.usage.input_tokens
    assert final.usage.output_tokens == reference.usage.output_tokens


def test_system_prompt_steers_the_render(stack, messages_client):
    # Same user turn, different system prompt => different rendered prompt
    # => different greedy completion. Both the string and block forms count.
    baseline = messages_client.messages.create(**msg_kwargs(stack.model_id))
    for system in (
        "You only speak French. Answer every question in French.",
        [
            {
                "type": "text",
                "text": "You only speak French. Answer every question in French.",
            }
        ],
    ):
        steered = messages_client.messages.create(
            **msg_kwargs(stack.model_id, system=system)
        )
        assert steered.content[0].text != baseline.content[0].text


def test_stop_sequence_is_reported(stack, messages_client):
    response = messages_client.messages.create(
        **msg_kwargs(
            stack.model_id,
            messages=[
                {"role": "user", "content": "Count from 1 to 20, one number per line."}
            ],
            stop_sequences=["5"],
        )
    )
    # Both worker paths must attribute the matched sequence: the gateway
    # matcher on the rust path, Finished.matched_stop on the python path.
    assert response.stop_reason == "stop_sequence", response
    assert response.stop_sequence == "5"
    text = "".join(b.text for b in response.content if b.type == "text")
    assert "5" not in text, text


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------


def weather_kwargs(model_id: str, **overrides) -> dict:
    kwargs = msg_kwargs(model_id, max_tokens=512, messages=[WEATHER_MSG], tools=TOOLS)
    kwargs.update(overrides)
    return kwargs


def assert_weather_tool_use(message):
    assert message.stop_reason == "tool_use", message
    tool_uses = [b for b in message.content if b.type == "tool_use"]
    assert len(tool_uses) == 1, message.content
    call = tool_uses[0]
    assert call.id.startswith("toolu_")
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}
    assert_no_marker_leak(message)
    return call


def test_tool_use_non_streaming(stack, messages_client):
    response = messages_client.messages.create(**weather_kwargs(stack.model_id))
    assert_weather_tool_use(response)


def test_tool_use_streaming_matches_non_streaming(stack, messages_client):
    reference = messages_client.messages.create(**weather_kwargs(stack.model_id))
    ref_call = assert_weather_tool_use(reference)

    with messages_client.messages.stream(**weather_kwargs(stack.model_id)) as stream:
        final = stream.get_final_message()
    call = assert_weather_tool_use(final)
    # input_json_delta fragments reassemble to exactly the non-streaming
    # arguments object.
    assert call.name == ref_call.name
    assert call.input == ref_call.input


def test_tool_use_round_trip(stack, messages_client):
    first = messages_client.messages.create(**weather_kwargs(stack.model_id))
    call = assert_weather_tool_use(first)

    followup = messages_client.messages.create(
        **weather_kwargs(
            stack.model_id,
            messages=[
                WEATHER_MSG,
                # Echo the assistant turn back verbatim (SDK block objects),
                # then answer the call — the Anthropic round-trip shape.
                {"role": "assistant", "content": first.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": '{"temperature_c": 21, "sky": "clear"}',
                        }
                    ],
                },
            ],
        )
    )
    # The 1B model may answer in text or call the tool again (captured
    # fixture behavior) — both are well-formed; leaked markers are not.
    assert followup.stop_reason in ("end_turn", "tool_use", "max_tokens"), followup
    assert_no_marker_leak(followup)
    if followup.stop_reason == "tool_use":
        assert_weather_tool_use(followup)
    else:
        text = "".join(b.text for b in followup.content if b.type == "text")
        assert text.strip(), followup


def test_forced_tool_choice_is_400(stack, messages_client):
    import anthropic

    with pytest.raises(anthropic.BadRequestError) as excinfo:
        messages_client.messages.create(
            **weather_kwargs(stack.model_id, tool_choice={"type": "any"})
        )
    assert "tool_choice" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Error envelopes
# ---------------------------------------------------------------------------


def test_error_envelopes_are_anthropic_shaped(stack):
    # Missing max_tokens: 400 with the {"type": "error"} envelope.
    response = httpx.post(
        f"{stack.base_url}/v1/messages",
        headers={"x-api-key": stack.api_key},
        json={"model": stack.model_id, "messages": [{"role": "user", "content": "x"}]},
        timeout=30,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in body["error"]["message"]

    # Bad key: 401 authentication_error (x-api-key is the Anthropic header).
    response = httpx.post(
        f"{stack.base_url}/v1/messages",
        headers={"x-api-key": "not-the-key"},
        json={
            "model": stack.model_id,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
        },
        timeout=30,
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_unknown_model_raises_typed_not_found(stack, messages_client):
    import anthropic

    with pytest.raises(anthropic.NotFoundError):
        messages_client.messages.create(**msg_kwargs("no-such-model"))


# ---------------------------------------------------------------------------
# Thinking blocks: Qwen3-0.6B (thinking-trained) on the rust worker
# ---------------------------------------------------------------------------


def thinking_kwargs(**overrides) -> dict:
    kwargs = msg_kwargs(QWEN_MODEL_ID, max_tokens=1024, messages=[WEATHER_MSG])
    kwargs.update(overrides)
    return kwargs


def test_thinking_blocks_non_streaming(qwen_stack, qwen_messages_client):
    response = qwen_messages_client.messages.create(**thinking_kwargs())
    # Correctly shaped and separated: one thinking block first, tags
    # stripped, then the visible answer as text.
    assert [b.type for b in response.content] == ["thinking", "text"], response.content
    thinking, text = response.content
    assert thinking.thinking.strip()
    assert thinking.signature == ""
    assert text.text.strip()
    assert_no_marker_leak(response)
    assert response.stop_reason == "end_turn", response


def test_thinking_blocks_streaming_matches_non_streaming(
    qwen_stack, qwen_messages_client
):
    reference = qwen_messages_client.messages.create(**thinking_kwargs())

    block_kinds: list[str] = []
    saw_thinking_delta = saw_text_delta = False
    with qwen_messages_client.messages.stream(**thinking_kwargs()) as stream:
        for event in stream:
            if event.type == "content_block_start":
                assert event.index == len(block_kinds)
                block_kinds.append(event.content_block.type)
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    saw_thinking_delta = True
                elif event.delta.type == "text_delta":
                    saw_text_delta = True
        final = stream.get_final_message()

    assert block_kinds == ["thinking", "text"]
    assert saw_thinking_delta and saw_text_delta
    # Deltas reassemble to exactly the non-streaming response.
    assert final.content[0].thinking == reference.content[0].thinking
    assert final.content[1].text == reference.content[1].text
    assert final.stop_reason == reference.stop_reason
    assert final.usage.output_tokens == reference.usage.output_tokens


def test_thinking_precedes_tool_use(qwen_stack, qwen_messages_client):
    # Greedy Qwen3 thinks, then emits the Hermes call (captured parser
    # fixture): thinking and tool_use arrive as separate, ordered blocks.
    response = qwen_messages_client.messages.create(
        **thinking_kwargs(max_tokens=512, tools=TOOLS)
    )
    assert [b.type for b in response.content] == ["thinking", "tool_use"], (
        response.content
    )
    assert response.content[0].thinking.strip()
    call = assert_weather_tool_use(response)
    assert call.input == {"city": "Paris"}


def test_thinking_disabled_renders_non_thinking_prompt(
    qwen_stack, qwen_messages_client
):
    # thinking: disabled maps to the template's enable_thinking=false —
    # Qwen3 then answers directly, so no thinking block can appear.
    response = qwen_messages_client.messages.create(
        **thinking_kwargs(thinking={"type": "disabled"})
    )
    assert [b.type for b in response.content] == ["text"], response.content
    assert response.content[0].text.strip()
    assert_no_marker_leak(response)
