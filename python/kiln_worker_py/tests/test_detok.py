"""Unit tests for the Kiln streaming decoder (tokenizer only, no GPU).

The contract under test: segments handed out via ``last_segment``
concatenate to EXACTLY ``tokenizer.decode(ids)`` (special tokens included,
no cleanup) — including the leading space of a space-prefixed first token,
which mlx-lm's own detokenizers trim (the bug that motivated detok.py), and
with no partial UTF-8 codepoint ever emitted mid-stream.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from kiln_worker_py.detok import REPLACEMENT_CHAR, StreamingDecoder

MODEL_NAME = "llama-3.2-1b-4bit"


@pytest.fixture(scope="module")
def tokenizer():
    root = os.environ.get("KILN_TEST_MODELS")
    model_dir = pathlib.Path(root).expanduser() / MODEL_NAME if root else None
    if model_dir is None or not (model_dir / "tokenizer.json").is_file():
        pytest.skip(
            f"set KILN_TEST_MODELS and fetch {MODEL_NAME} "
            "(./scripts/fetch-test-model.sh) to run detok tests"
        )
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(model_dir))


def full_decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def drive(tokenizer, ids: list[int]) -> list[str]:
    """Feed ids one at a time (the engine's cadence); return the per-token
    segments, with the finalize tail appended as the last element."""
    decoder = StreamingDecoder(tokenizer)
    segments = []
    for token in ids:
        decoder.add_token(token)
        segments.append(decoder.last_segment)
    decoder.finalize()
    segments.append(decoder.last_segment)
    return segments


CORPUS = [
    "a device used to dry and cure materials, such as wood",
    "1\n2\n3\n4\n5",
    "the kiln 窯 is very hot \U0001f525\U0001f9e8 — où? naïve!",
    "\U0001f469‍\U0001f469‍\U0001f467‍\U0001f466 family; \U0001f1eb\U0001f1f7 flag",
]


@pytest.mark.parametrize("text", CORPUS)
def test_segments_concatenate_to_full_decode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    segments = drive(tokenizer, ids)
    assert "".join(segments) == full_decode(tokenizer, ids)
    # Nothing emitted mid-codepoint: the replacement char may only appear if
    # the true decode contains it (it does not, for this corpus).
    assert REPLACEMENT_CHAR not in "".join(segments)


def test_leading_space_of_first_token_is_kept(tokenizer):
    # The regression that motivated this module: a raw-completions
    # continuation almost always starts with a space-prefixed token.
    # mlx-lm's detokenizer returns "a device..." here; the contract (and the
    # gateway's rust-worker decoder) says " a device...".
    ids = tokenizer.encode(" a device used to", add_special_tokens=False)
    segments = drive(tokenizer, ids)
    joined = "".join(segments)
    assert joined == full_decode(tokenizer, ids)
    assert joined.startswith(" "), f"leading space lost: {joined!r}"

    reference = tokenizer.decode(ids)  # mlx-lm-free sanity anchor
    assert joined.lstrip(" ") in (reference.lstrip(" "), reference)


def test_split_codepoint_is_held_then_released(tokenizer):
    # A single emoji usually spans several BPE byte-fallback tokens; the
    # decoder must hold text while the codepoint is incomplete.
    ids = tokenizer.encode("\U0001f525", add_special_tokens=False)
    decoder = StreamingDecoder(tokenizer)
    emitted = []
    for token in ids:
        decoder.add_token(token)
        emitted.append(decoder.last_segment)
    assert "".join(emitted) == "\U0001f525"
    for partial in emitted[:-1]:
        assert REPLACEMENT_CHAR not in partial


def test_special_tokens_are_not_skipped(tokenizer):
    # The engine never feeds the stop token, but any special id that IS fed
    # must surface (decode with skip_special_tokens=False, like the rust
    # decoder).
    bos = tokenizer.bos_token_id
    ids = [bos] + tokenizer.encode("hello", add_special_tokens=False)
    segments = drive(tokenizer, ids)
    assert "".join(segments) == full_decode(tokenizer, ids)
    assert tokenizer.bos_token in "".join(segments)
