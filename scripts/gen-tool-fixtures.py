#!/usr/bin/env python3
"""gen-tool-fixtures.py — capture real tool-call model output for the
kiln-tokenize streaming tool-call parsers (SPEC §8.2, Phase 7).

Two fixture kinds, both written to crates/kiln-tokenize/tests/fixtures/toolcall/:

  - kind "generation": greedy completions from the pinned test models with
    tools rendered through the model's own chat template — the exact token
    ids and raw text (special tokens included, EOS excluded) a worker
    streams for a real tool-call request. Prompt formats and emission
    quirks (Llama's <|python_tag|> prefix, Qwen3's <think> block, the
    inter-call newline) are captured, not guessed.
  - kind "template": the model family's own chat template rendering of an
    assistant tool_calls message — the training-time serialization of the
    format, used where no pinned model emits it (Qwen3-Coder XML) or to
    pin an alternate legal serialization (Llama's bare JSON, no python
    tag). Rendered with the same jinja2 settings transformers uses.

Run from the repo root (network needed once for the Qwen3-Coder template):

    uv run --project python/kiln_worker_py python scripts/gen-tool-fixtures.py \
        --out crates/kiln-tokenize/tests/fixtures/toolcall

The `expected` block is derived by straightforward per-format extraction
below and MUST be eyeballed against `text` when regenerating: it is the
reference the Rust parsers are held to.
"""

import argparse
import json
import pathlib
import re
import sys
import urllib.request

QWEN3_CODER_TEMPLATE_URL = (
    "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/resolve/main/"
    "tokenizer_config.json"
)
PINNED_DATE_STRING = "26 Jul 2024"  # keep Llama prompts render-stable

WEATHER_TOOL = {
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
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}


def resolve_model_dir(model: str) -> pathlib.Path:
    import os

    candidates = [pathlib.Path(model)]
    root = os.environ.get("KILN_TEST_MODELS")
    if root:
        candidates.append(pathlib.Path(root) / model)
    candidates.append(pathlib.Path.home() / ".kiln/test-models" / model)
    for cand in candidates:
        if (cand / "config.json").is_file():
            return cand
    sys.exit(f"model not found (tried: {', '.join(str(c) for c in candidates)})")


# ---------------------------------------------------------------------------
# Expected-output extraction (reference logic per format; eyeball on regen)
# ---------------------------------------------------------------------------


def json_value_span(text: str, start: int) -> int:
    """End index (exclusive) of the JSON value starting at `start`
    (string/escape-aware balanced scan)."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
                if depth == 0:
                    return i + 1
        elif c == '"':
            in_string = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        elif depth == 0 and c not in " \t\r\n":
            # bare scalar: scan to a JSON delimiter
            for j in range(i, len(text)):
                if text[j] in ",}] \t\r\n":
                    return j
            return len(text)
    raise ValueError(f"unterminated JSON value at {start}: {text[start:]!r}")


def raw_args_span(payload: str, key: str) -> str:
    """The verbatim text of `payload`'s `key` value, exactly as the model
    wrote it (never re-serialized)."""
    key_idx = payload.index(f'"{key}"')
    colon = payload.index(":", key_idx)
    start = colon + 1
    while payload[start] in " \t\r\n":
        start += 1
    return payload[start : json_value_span(payload, start)]


def segment_content(segment: str) -> str:
    """The parser rule for text runs outside tool calls: whitespace-only
    runs are dropped; runs with substance pass through verbatim."""
    return segment if segment.strip() else ""


def expect_llama(text: str) -> dict:
    """Llama 3.x: optional <|python_tag|> then {"name": ..., "parameters": ...},
    `;`-separated for multiple calls; anything else is plain content. The
    model also mirrors the prompt's OpenAI tool shape back —
    {"type": "function", "function": {...}} — so that wrapper unwraps
    (captured in llama-after-tool-result, real 1B output)."""
    stripped = text.removeprefix("<|python_tag|>")
    calls = []
    content = text
    try:
        payloads = [p for p in stripped.split(";") if p.strip()]
        parsed = [json.loads(p) for p in payloads]
        for raw, p in zip(payloads, parsed):
            if "function" in p and isinstance(p["function"], dict):
                raw = raw_args_span(raw, "function")
                p = p["function"]
            key = "parameters" if "parameters" in p else "arguments"
            raw_args = raw_args_span(raw, key)
            assert json.loads(raw_args) == p[key]
            calls.append({"name": p["name"], "arguments": raw_args})
        content = ""
    except (json.JSONDecodeError, ValueError, KeyError):
        calls = []
        content = text
    return {"content": content, "calls": calls}


def expect_hermes(text: str) -> dict:
    """Hermes/Qwen: <tool_call>\\n{json}\\n</tool_call> blocks; text outside
    the blocks is content per `segment_content`."""
    calls = []
    content = []
    pos = 0
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        content.append(segment_content(text[pos : m.start()]))
        payload = json.loads(m.group(1))
        raw_args = raw_args_span(m.group(1), "arguments")
        assert json.loads(raw_args) == payload["arguments"]
        calls.append({"name": payload["name"], "arguments": raw_args})
        pos = m.end()
    content.append(segment_content(text[pos:]))
    return {"content": "".join(content), "calls": calls}


def encode_xml_param(raw: str, declared: str | None) -> str:
    """Reference JSON encoding of an XML parameter value (must mirror the
    Rust parser's `encode_xml_param` exactly): declared string stays a
    quoted string; boolean accepts the template's Python-style True/False;
    integer/number parse numerically; everything else is spliced VERBATIM
    when it is valid JSON (model text is never re-serialized), with a
    quoted-string fallback."""
    quoted = json.dumps(raw, ensure_ascii=False)
    if declared == "string":
        return quoted
    if declared == "boolean":
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered
        return quoted
    if declared == "integer":
        try:
            return str(int(raw.strip()))
        except ValueError:
            return quoted
    if declared == "number":
        try:
            value = float(raw.strip())
        except ValueError:
            return quoted
        if value.is_integer() and "." not in raw and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return quoted


def expect_qwen_xml(text: str, tools: list) -> dict:
    """Qwen3-Coder XML: <tool_call><function=NAME><parameter=K>V</parameter>...
    Values coerced by the tool schema's declared param type; the constructed
    arguments JSON is compact (serde_json style)."""
    types = {}
    for tool in tools:
        fn = tool["function"]
        props = (fn.get("parameters") or {}).get("properties") or {}
        types[fn["name"]] = {k: v.get("type") for k, v in props.items()}
    calls = []
    content = []
    pos = 0
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        content.append(segment_content(text[pos : m.start()]))
        block = m.group(1)
        fn_match = re.search(
            r"<function=([^>]+)>\s*(.*?)\s*</function>", block, re.DOTALL
        )
        name = fn_match.group(1)
        parts = []
        for pm in re.finditer(
            r"<parameter=([^>]+)>\n(.*?)\n</parameter>", fn_match.group(2), re.DOTALL
        ):
            key, raw = pm.group(1), pm.group(2)
            encoded = encode_xml_param(raw, types.get(name, {}).get(key))
            parts.append(f"{json.dumps(key, ensure_ascii=False)}:{encoded}")
        calls.append({"name": name, "arguments": "{" + ",".join(parts) + "}"})
        pos = m.end()
    content.append(segment_content(text[pos:]))
    return {"content": "".join(content), "calls": calls}


# ---------------------------------------------------------------------------
# kind "generation": greedy completions from the pinned models
# ---------------------------------------------------------------------------

# (case name, messages, tools, extra template kwargs)
LLAMA_CASES = [
    (
        "weather-call",
        [{"role": "user", "content": "What is the weather in Paris right now?"}],
        [WEATHER_TOOL],
        {"date_string": PINNED_DATE_STRING},
    ),
    (
        "weather-multibyte",
        [{"role": "user", "content": "What's the weather in São Paulo?"}],
        [WEATHER_TOOL],
        {"date_string": PINNED_DATE_STRING},
    ),
    (
        # Asked NOT to call a tool, the 1B model calls one anyway (with a
        # hallucinated city) — kept as real unforced-call behavior.
        "unforced-call",
        [
            {
                "role": "user",
                "content": "Do not call any function. Just answer in plain "
                "English: what is a kiln?",
            }
        ],
        [WEATHER_TOOL],
        {"date_string": PINNED_DATE_STRING},
    ),
    (
        # The realistic plain-text case under tools: the turn after a tool
        # result. Also exercises tool_calls/tool message rendering.
        "after-tool-result",
        [
            {"role": "user", "content": "What is the weather in Paris right now?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature_c": 21, "sky": "clear"}'},
        ],
        [WEATHER_TOOL],
        {"date_string": PINNED_DATE_STRING},
    ),
]

QWEN_CASES = [
    (
        "weather-call",
        [{"role": "user", "content": "What is the weather in Paris right now?"}],
        [WEATHER_TOOL, TIME_TOOL],
        {"enable_thinking": False},
    ),
    (
        "weather-think-call",
        [{"role": "user", "content": "What is the weather in Paris right now?"}],
        [WEATHER_TOOL, TIME_TOOL],
        {},
    ),
    (
        "two-calls",
        [
            {
                "role": "user",
                "content": "What is the weather in Paris, and what time is it "
                "in Tokyo? Call the tools you need in one go.",
            }
        ],
        [WEATHER_TOOL, TIME_TOOL],
        {"enable_thinking": False},
    ),
    (
        "weather-multibyte",
        [{"role": "user", "content": "What's the weather in São Paulo?"}],
        [WEATHER_TOOL, TIME_TOOL],
        {"enable_thinking": False},
    ),
    (
        "no-tool-text",
        [{"role": "user", "content": "Just say hello, no tools needed."}],
        [WEATHER_TOOL, TIME_TOOL],
        {"enable_thinking": False},
    ),
    (
        "after-tool-result",
        [
            {"role": "user", "content": "What is the weather in Paris right now?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature_c": 21, "sky": "clear"}'},
        ],
        [WEATHER_TOOL, TIME_TOOL],
        {"enable_thinking": False},
    ),
]


def generate_fixtures(model_name: str, family: str, cases, expect, out_dir):
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import generate_step

    model_dir = resolve_model_dir(model_name)
    model, tokenizer = mlx_lm.load(str(model_dir))
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    for name, messages, tools, kwargs in cases:
        prompt_ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, **kwargs
        )
        # The reference rendering (transformers): the gateway's minijinja
        # render must reproduce it byte-for-byte (parity-tested).
        prompt_text = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs
        )
        generated: list[int] = []
        for token, _ in generate_step(mx.array(prompt_ids), model, max_tokens=1024):
            if int(token) in eos:
                break  # workers do not stream their stop token
            generated.append(int(token))
        else:
            sys.exit(f"{model_name}/{name}: no EOS within 1024 tokens; not a fixture")

        text = tokenizer.decode(generated, skip_special_tokens=False)
        fixture = {
            "family": family,
            "source": {
                "kind": "generation",
                "model": model_name,
                "mlx_lm_version": mlx_lm.__version__,
                "template_kwargs": kwargs,
            },
            "tools": tools,
            "messages": messages,
            "prompt_text": prompt_text,
            "token_ids": generated,
            "text": text,
            "expected": expect(text) if family != "qwen_xml" else expect(text, tools),
        }
        path = out_dir / f"{family}-{name}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
        print(f"  {path} ({len(generated)} tokens)", file=sys.stderr)


# ---------------------------------------------------------------------------
# kind "template": the family template's own tool_calls serialization
# ---------------------------------------------------------------------------


def transformers_jinja_env():
    import jinja2

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent
    )
    env.globals["raise_exception"] = _raise
    return env


def _raise(message):
    raise ValueError(message)


def render_assistant_turn(template_src: str, messages, tools, marker: str) -> str:
    """Renders the conversation and slices the final assistant turn — the
    exact text the model was trained to emit for it. `marker` is the
    assistant-turn opener up to and including the newline the generation
    prompt supplies."""
    env = transformers_jinja_env()
    rendered = env.from_string(template_src).render(
        messages=messages, tools=tools, add_generation_prompt=False
    )
    start = rendered.rindex(marker) + len(marker)
    turn = rendered[start:]
    end = turn.index("<|im_end|>") if "<|im_end|>" in turn else turn.index("<|eot_id|>")
    return turn[:end]


def qwen3_coder_template(cache_path: pathlib.Path) -> str:
    if not cache_path.is_file():
        print(f"fetching {QWEN3_CODER_TEMPLATE_URL}", file=sys.stderr)
        with urllib.request.urlopen(QWEN3_CODER_TEMPLATE_URL, timeout=60) as resp:
            cache_path.write_bytes(resp.read())
    return json.loads(cache_path.read_text())["chat_template"]


RUN_TESTS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the project's test suite.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Test file or directory"},
                "verbose": {"type": "boolean"},
                "retries": {"type": "integer"},
                "env": {"type": "object", "description": "Extra environment vars"},
            },
            "required": ["path"],
        },
    },
}

QWEN_XML_TEMPLATE_CASES = [
    (
        "single-call",
        None,
        [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        [WEATHER_TOOL],
    ),
    (
        "typed-params",
        None,
        [
            {
                "name": "run_tests",
                "arguments": {
                    "path": "tests/e2e",
                    "verbose": True,
                    "retries": 2,
                    "env": {"CI": "1"},
                },
            }
        ],
        [RUN_TESTS_TOOL],
    ),
    (
        "multiline-value",
        None,
        [
            {
                "name": "run_tests",
                "arguments": {"path": "tests/a.py\ntests/b.py\ntests/c.py"},
            }
        ],
        [RUN_TESTS_TOOL],
    ),
    (
        "reasoning-then-two-calls",
        "I'll check the weather in both cities.",
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        ],
        [WEATHER_TOOL],
    ),
]


def template_fixtures_qwen_xml(out_dir: pathlib.Path, cache_dir: pathlib.Path):
    template_src = qwen3_coder_template(cache_dir / "qwen3-coder-tokenizer_config.json")
    for name, content, calls, tools in QWEN_XML_TEMPLATE_CASES:
        messages = [
            {"role": "user", "content": "placeholder"},
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {"type": "function", "function": call} for call in calls
                ],
            },
        ]
        # The coder template opens the turn with '<|im_start|>assistant' and
        # puts '\n' before content/calls; generation starts after
        # '<|im_start|>assistant\n', so slice past that newline.
        text = render_assistant_turn(
            template_src, messages, tools, "<|im_start|>assistant\n"
        )
        fixture = {
            "family": "qwen_xml",
            "source": {
                "kind": "template",
                "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "note": "official chat template serialization; no pinned "
                "model emits this format",
            },
            "tools": tools,
            "messages": messages[:1],
            "token_ids": None,
            "text": text,
            "expected": expect_qwen_xml(text, tools),
        }
        path = out_dir / f"qwen_xml-{name}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
        print(f"  {path} (template)", file=sys.stderr)


def template_fixture_llama_bare_json(out_dir: pathlib.Path):
    """Llama's template serializes assistant tool calls as bare JSON (no
    python tag) — pin that alternate legal form as a parser input."""
    model_dir = resolve_model_dir("llama-3.2-1b-4bit")
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    messages = [
        {"role": "user", "content": "What is the weather in Paris right now?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
    ]
    env = transformers_jinja_env()
    env.globals["strftime_now"] = lambda fmt: PINNED_DATE_STRING
    rendered = env.from_string(config["chat_template"]).render(
        messages=messages,
        tools=[WEATHER_TOOL],
        add_generation_prompt=False,
        bos_token="<|begin_of_text|>",
    )
    marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    start = rendered.rindex(marker) + len(marker)
    text = rendered[start:].removesuffix("<|eot_id|>")
    fixture = {
        "family": "llama",
        "source": {
            "kind": "template",
            "model": "llama-3.2-1b-4bit",
            "note": "the template's own tool_calls serialization: bare JSON, "
            "no <|python_tag|>",
        },
        "tools": [WEATHER_TOOL],
        "messages": messages[:1],
        "token_ids": None,
        "text": text,
        "expected": expect_llama(text),
    }
    path = out_dir / "llama-bare-json.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  {path} (template)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cache-dir",
        default="/tmp",
        help="where the downloaded Qwen3-Coder tokenizer_config.json is cached",
    )
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_fixtures_qwen_xml(out_dir, pathlib.Path(args.cache_dir))
    template_fixture_llama_bare_json(out_dir)
    generate_fixtures("llama-3.2-1b-4bit", "llama", LLAMA_CASES, expect_llama, out_dir)
    generate_fixtures("qwen3-0.6b-4bit", "hermes", QWEN_CASES, expect_hermes, out_dir)
    print("done — eyeball each fixture's `expected` before committing", file=sys.stderr)


if __name__ == "__main__":
    main()
