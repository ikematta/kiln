"""Unit tests for the incremental stop-string matcher (no model needed)."""

from kiln_worker_py.stops import StopStringMatcher


def drive(stops: list[str], segments: list[str]) -> tuple[str, str | None]:
    """Push segments; return (all released text, matched stop or None)."""
    matcher = StopStringMatcher(stops)
    out = ""
    for seg in segments:
        released, matched = matcher.push(seg)
        out += released
        if matched is not None:
            return out, matched
    return out + matcher.flush(), None


def test_no_stops_passes_through():
    assert drive([], ["hello", " world"]) == ("hello world", None)


def test_match_within_one_segment():
    assert drive(["STOP"], ["abc STOP def"]) == ("abc ", "STOP")


def test_match_split_across_segments():
    assert drive(["STOP"], ["abc S", "TO", "P def"]) == ("abc ", "STOP")


def test_partial_match_is_held_until_disambiguated():
    matcher = StopStringMatcher(["STOP"])
    released, matched = matcher.push("abc ST")
    assert released == "abc "  # "ST" could still become "STOP"
    assert matched is None
    released, matched = matcher.push("ATIC")
    assert released == "STATIC"
    assert matched is None


def test_flush_releases_held_tail():
    matcher = StopStringMatcher(["STOP"])
    released, _ = matcher.push("abc ST")
    assert released == "abc "
    assert matcher.flush() == "ST"


def test_earliest_of_multiple_stops_wins():
    assert drive(["bbb", "aa"], ["xx aa yy bbb"]) == ("xx ", "aa")


def test_released_text_never_contains_stop():
    text, matched = drive(["\n\n"], ["para one\n", "\nafter"])
    assert matched == "\n\n"
    assert text == "para one"


def test_stop_at_segment_start():
    assert drive(["END"], ["END now"]) == ("", "END")


def test_empty_stop_strings_are_ignored():
    assert drive([""], ["text"]) == ("text", None)
