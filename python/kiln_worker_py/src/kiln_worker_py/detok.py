"""Incremental, UTF-8-safe detokenization for streamed token ids.

A port of kiln-tokenize's ``StreamingDecoder`` (the decoder the gateway uses
for Rust workers), so both workers compute ``TokenChunk.text`` with the SAME
two-offset algorithm over the same tokenizer semantics — the cross-worker
text-parity invariant (asserted by the e2e suite) then holds by
construction. Decoding is raw: special tokens included, no
``clean_up_tokenization_spaces`` (the tokenizers crate the gateway uses has
no such cleanup).

mlx-lm's streaming detokenizers are deliberately NOT used here: they trim
the leading space of the first generated segment (display behavior for the
mlx-lm CLI), which violates the proto contract that the concatenated
``TokenChunk.text`` equals the exact detokenization of the generated ids.
Chat traffic rarely notices — a templated response's first token is seldom
space-prefixed — but a raw ``/v1/completions`` prompt almost always
continues with one.

This is a Kiln-owned module, not a patch of mlx-lm (CLAUDE.md: no
monkey-patching).
"""

from __future__ import annotations

REPLACEMENT_CHAR = "�"


class StreamingDecoder:
    """Two-offset incremental decoder (HF text-generation-inference scheme).

    Presents the same ``reset`` / ``add_token`` / ``finalize`` /
    ``last_segment`` interface as mlx-lm's detokenizers so the engine's
    decode loop stays agnostic.

    The sliding window handles both hard cases from CLAUDE.md:
    - multi-token UTF-8: a codepoint split across tokens decodes to U+FFFD
      until its continuation bytes arrive — text is held, never emitted as
      a partial codepoint;
    - context-dependent decoders (SentencePiece leading-space rules,
      byte-fallback): new text is always ``decode(window+new) -
      decode(window)``, never tokens decoded in isolation.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self._ids: list[int] = []
        # Start of the decode window: ids before this are already emitted
        # and no longer influence new text.
        self._prefix_offset = 0
        # End of the already-emitted region within the window.
        self._read_offset = 0
        self.text = ""
        self._segment_start = 0

    @property
    def last_segment(self) -> str:
        """Text produced since the last access (mlx-lm interface)."""
        segment = self.text[self._segment_start :]
        self._segment_start = len(self.text)
        return segment

    def _decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def add_token(self, token: int) -> None:
        self._ids.append(token)
        prefix = self._decode(self._ids[self._prefix_offset : self._read_offset])
        full = self._decode(self._ids[self._prefix_offset :])
        # Hold while nothing new appeared or the window ends mid-codepoint
        # (trailing replacement char) — same conditions as the Rust decoder.
        if len(full) > len(prefix) and not full.endswith(REPLACEMENT_CHAR):
            self.text += full[len(prefix) :]
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)

    def finalize(self) -> None:
        """Releases whatever is still held (call once the stream is over).
        Genuinely incomplete trailing bytes surface as U+FFFD — a complete
        codepoint, so still safe for SSE."""
        prefix = self._decode(self._ids[self._prefix_offset : self._read_offset])
        full = self._decode(self._ids[self._prefix_offset :])
        self.text += full[len(prefix) :]
        self._prefix_offset = len(self._ids)
        self._read_offset = len(self._ids)
