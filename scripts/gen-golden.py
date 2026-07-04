#!/usr/bin/env python3
"""gen-golden.py — generate golden-token parity fixtures from mlx-lm (SPEC §11.2).

Run ONLY when explicitly instructed (CLAUDE.md): committed fixtures are the
reference the Rust workers must reproduce exactly, and regenerating them on a
different MLX core silently moves the bar.

The reference stack is the Python worker venv, whose pins are aligned with the
core MLX that kiln-mlx builds (docs/decisions/0001-mlx-c-pin.md, follow-up B1):

    uv run --project python/kiln_worker_py python scripts/gen-golden.py \
        --model llama-3.2-1b-4bit --out tests/golden/llama-3.2-1b-4bit/

This script hard-refuses to run on any other mlx.core version.

Fixture semantics (the Rust golden harness in kiln-models must mirror them):
  - prompt ids:
      chat_template=true  -> tokenizer.apply_chat_template(
                                 [{role: user, content: prompt}],
                                 add_generation_prompt=True,
                                 date_string=PINNED_DATE_STRING)
                             (renders the template, encodes WITHOUT special
                              tokens — the template itself emits BOS; see the
                              kiln-tokenize crate docs, "BOS contract")
      chat_template=false -> tokenizer.encode(prompt) with special tokens,
                             i.e. the tokenizer's own post-processor adds BOS
  - decoding: greedy (argmax), exactly max_tokens tokens, NO eos stopping —
    post-EOS tokens are still deterministic under greedy and comparing the
    full length keeps the fixture unambiguous.
  - date_string is pinned because Llama 3.x templates otherwise interpolate
    strftime_now() and the rendered prompt would change daily.
"""

import argparse
import json
import pathlib
import sys

EXPECTED_MLX_CORE = "0.31.1"  # docs/decisions/0001-mlx-c-pin.md (mlx-c v0.6.0)
PINNED_DATE_STRING = "26 Jul 2024"  # the Llama template's own non-strftime fallback

# Architecture-independent case list; every model gets the same probes.
# (name, prompt, chat_template, max_tokens)
CASES = [
    (
        "chat-basic",
        "What is a kiln used for? Answer in one sentence.",
        True,
        64,
    ),
    (
        "chat-code",
        "Write a Python function that reverses a singly linked list.",
        True,
        128,
    ),
    (
        "chat-multibyte",
        "Translate to French: the kiln 窯 is very hot \U0001f525.",
        True,
        64,
    ),
    (
        "raw-continuation",
        "The three primary colors are",
        False,
        64,
    ),
    (
        "raw-long-prefill",
        "Pottery is one of the oldest human inventions, originating before the "
        "Neolithic period. Ceramic objects like figurines were made as early as "
        "29,000 BC, and vessels for storing water and food followed as settled "
        "communities emerged. The potter's wheel was invented in Mesopotamia "
        "sometime between 6,000 and 4,000 BC and revolutionised production. "
        "Firing transforms shaped clay into a permanent material: as the "
        "temperature inside a kiln rises, clay minerals lose their chemically "
        "bound water, sinter, and finally vitrify into a glassy matrix. "
        "Earthenware fires at roughly 1,000 degrees Celsius, stoneware near "
        "1,200, and porcelain at 1,300 or above, which is why porcelain is "
        "translucent, vitrified, and strong. Glazes — suspensions of silica, "
        "fluxes, and colouring oxides — melt onto the surface during the glost "
        "firing and seal the body. Reduction and oxidation atmospheres inside "
        "the kiln change how those oxides develop colour, which is why the "
        "same copper glaze can emerge green in oxidation and blood-red in "
        "reduction. Given all of this, the single most important variable a "
        "potter controls during a firing is",
        False,
        64,
    ),
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local name under $KILN_TEST_MODELS (or a path)")
    parser.add_argument("--out", required=True, help="output dir, e.g. tests/golden/<model>/")
    parser.add_argument("--weights-revision", help="override; default reads <model>/.kiln-revision")
    args = parser.parse_args()

    import mlx.core as mx

    if mx.__version__ != EXPECTED_MLX_CORE:
        sys.exit(
            f"mlx.core is {mx.__version__}, but fixtures must be generated on "
            f"{EXPECTED_MLX_CORE} — the core MLX that kiln-mlx builds "
            f"(docs/decisions/0001-mlx-c-pin.md). Refusing to generate."
        )

    import mlx_lm
    from mlx_lm.generate import generate_step

    model_dir = resolve_model_dir(args.model)
    revision = args.weights_revision
    if revision is None:
        marker = model_dir / ".kiln-revision"
        if not marker.is_file():
            sys.exit(f"{marker} missing; pass --weights-revision explicitly")
        revision = marker.read_text().strip()

    print(f"reference: mlx.core {mx.__version__}, mlx-lm {mlx_lm.__version__}", file=sys.stderr)
    print(f"model: {model_dir} ({revision})", file=sys.stderr)

    model, tokenizer = mlx_lm.load(str(model_dir))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, prompt, chat, max_tokens in CASES:
        if chat:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                date_string=PINNED_DATE_STRING,
            )
        else:
            prompt_ids = tokenizer.encode(prompt)

        generated: list[int] = []
        for token, _logprobs in generate_step(
            mx.array(prompt_ids), model, max_tokens=max_tokens
        ):
            generated.append(int(token))
        assert len(generated) == max_tokens, (name, len(generated))

        fixture = {
            "prompt": prompt,
            "chat_template": chat,
            "max_tokens": max_tokens,
            "expected_token_ids": generated,
            "mlx_lm_version": mlx_lm.__version__,
            "weights_revision": revision,
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=True) + "\n")
        preview = tokenizer.decode(generated[:24]).replace("\n", "\\n")
        print(f"  {path}  ({len(prompt_ids)} prompt tok) -> {preview!r}...", file=sys.stderr)

    print(f"wrote {len(CASES)} fixtures to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
