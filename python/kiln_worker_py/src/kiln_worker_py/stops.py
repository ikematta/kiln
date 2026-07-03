"""Incremental stop-string matching over a streamed detokenized text.

Text is pushed in arbitrary segments (as the streaming detokenizer releases
them). The matcher never releases text that is, or could become, part of a
stop string: a buffer suffix that is a prefix of any stop string is held back
until disambiguated. Matched stop text is dropped entirely (SPEC: "matched
text excluded").
"""

from __future__ import annotations


class StopStringMatcher:
    """Stateful matcher for one request's stream."""

    def __init__(self, stop_strings: list[str]) -> None:
        self._stops = [s for s in stop_strings if s]
        self._buffer = ""

    def push(self, text: str) -> tuple[str, str | None]:
        """Feed a new text segment.

        Returns ``(released, matched)``: ``released`` is text now safe to
        emit; ``matched`` is the stop string that fired, or ``None``. After a
        match the matcher drops everything from the match onward.
        """
        if not self._stops:
            return text, None
        self._buffer += text

        earliest: tuple[int, str] | None = None
        for stop in self._stops:
            idx = self._buffer.find(stop)
            if idx != -1 and (earliest is None or idx < earliest[0]):
                earliest = (idx, stop)
        if earliest is not None:
            released = self._buffer[: earliest[0]]
            self._buffer = ""
            return released, earliest[1]

        hold_len = 0
        max_hold = min(len(self._buffer), max(len(s) for s in self._stops) - 1)
        for k in range(max_hold, 0, -1):
            suffix = self._buffer[-k:]
            if any(s.startswith(suffix) for s in self._stops):
                hold_len = k
                break
        if hold_len:
            released = self._buffer[:-hold_len]
            self._buffer = self._buffer[-hold_len:]
        else:
            released = self._buffer
            self._buffer = ""
        return released, None

    def flush(self) -> str:
        """Release any held text (call once the stream finished unmatched)."""
        released, self._buffer = self._buffer, ""
        return released
