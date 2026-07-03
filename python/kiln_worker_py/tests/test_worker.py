"""Integration tests: the worker protocol against a real tiny model over UDS.

Phase 1 acceptance (SPEC §12): streams tokens, cancel mid-stream stops within
2 engine steps, health reports memory numbers.
"""

from __future__ import annotations

import uuid

import pytest

from kiln_worker_py._gen import worker_pb2 as pb

PROMPT = "The capital of France is"


def make_submit(
    text: str = PROMPT,
    max_tokens: int = 24,
    temperature: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    stop_strings: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    request_id: str | None = None,
) -> pb.SubmitRequest:
    return pb.SubmitRequest(
        request_id=request_id or str(uuid.uuid4()),
        raw_text=text,
        sampling=pb.SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, seed=seed
        ),
        stopping=pb.StoppingParams(
            max_tokens=max_tokens,
            stop_strings=stop_strings or [],
            stop_token_ids=stop_token_ids or [],
        ),
    )


def run(stub, request: pb.SubmitRequest):
    """Drain a Submit stream into (admitted, chunk list, finished)."""
    admitted, chunks, finished = None, [], None
    for event in stub.Submit(request):
        kind = event.WhichOneof("event")
        if kind == "admitted":
            assert admitted is None, "Admitted must be sent once, first"
            assert finished is None
            admitted = event.admitted
        elif kind == "tokens":
            chunks.append(event.tokens)
        elif kind == "finished":
            assert finished is None, "stream must contain exactly one Finished"
            finished = event.finished
    assert finished is not None, "stream must end with Finished"
    return admitted, chunks, finished


def generated_ids(chunks) -> list[int]:
    return [t for c in chunks for t in c.token_ids]


def generated_text(chunks) -> str:
    return "".join(c.text for c in chunks)


# -- GetInfo / Health ----------------------------------------------------------


def test_get_info(worker):
    info = worker.stub.GetInfo(pb.InfoRequest())
    assert info.model_id == worker.model_id
    assert info.architecture == "llama"
    assert info.worker_kind == "python"
    assert info.worker_version
    assert info.dtype == "q4_g64"
    assert info.max_context_len > 0
    assert info.vocab_size > 0
    assert info.weights_fingerprint
    assert info.chat_template_hash
    assert pb.CAPABILITY_TOKENIZER_OWNED in info.capabilities


def test_health_reports_memory_numbers(worker):
    status = worker.stub.Health(pb.HealthRequest())
    assert status.state == pb.WORKER_STATE_READY
    assert status.uptime_ms > 0
    mem = status.memory
    assert mem.weights_bytes > 100_000_000  # the 1B-4bit weights are ~700 MB
    assert mem.mlx_active_bytes > 100_000_000  # weights resident in MLX
    assert mem.mlx_peak_bytes >= mem.mlx_active_bytes
    assert mem.process_rss_bytes > mem.mlx_active_bytes // 2


# -- Submit: streaming, determinism, sampling ----------------------------------


def test_submit_streams_tokens(worker):
    admitted, chunks, finished = run(worker.stub, make_submit(max_tokens=24))
    assert admitted is not None
    assert admitted.queue_position == 0
    assert admitted.prompt_tokens > 0
    assert len(chunks) >= 2, "tokens must stream incrementally, not one blob"
    assert all(len(c.token_ids) >= 1 for c in chunks)
    assert finished.finish_reason in (pb.FINISH_REASON_STOP, pb.FINISH_REASON_LENGTH)
    assert finished.prompt_tokens == admitted.prompt_tokens
    assert 0 < finished.completion_tokens <= 24
    assert finished.completion_tokens == len(generated_ids(chunks))
    assert generated_text(chunks)
    assert finished.timings.decode_tokens_per_sec > 0
    assert finished.timings.prefill_ms + finished.timings.decode_ms > 0


def test_greedy_is_deterministic(worker):
    first = run(worker.stub, make_submit(max_tokens=16))
    second = run(worker.stub, make_submit(max_tokens=16))
    assert generated_ids(first[1]) == generated_ids(second[1])
    assert generated_text(first[1]) == generated_text(second[1])


def test_seeded_sampling_is_reproducible(worker):
    req = dict(max_tokens=16, temperature=0.9, top_p=0.95, seed=1234)
    first = run(worker.stub, make_submit(**req))
    second = run(worker.stub, make_submit(**req))
    assert generated_ids(first[1]) == generated_ids(second[1])
    assert first[2].seed_used == 1234
    assert second[2].seed_used == 1234


def test_seed_zero_lets_worker_choose(worker):
    _, _, finished = run(worker.stub, make_submit(max_tokens=4, seed=0))
    assert finished.seed_used != 0


def test_top_k_one_matches_greedy(worker):
    greedy = run(worker.stub, make_submit(max_tokens=12))
    topk = run(worker.stub, make_submit(max_tokens=12, temperature=0.8, top_k=1))
    assert generated_ids(greedy[1]) == generated_ids(topk[1])


# -- Stops ---------------------------------------------------------------------


def test_stop_string_ends_generation_and_is_excluded(worker):
    _, chunks, _ = run(worker.stub, make_submit(max_tokens=24))
    baseline_text = generated_text(chunks)
    assert len(baseline_text) >= 6, "baseline too short to pick a stop string"
    mid = len(baseline_text) // 3
    stop = baseline_text[mid : mid + 3]

    _, chunks2, finished2 = run(
        worker.stub, make_submit(max_tokens=24, stop_strings=[stop])
    )
    text2 = generated_text(chunks2)
    assert finished2.finish_reason == pb.FINISH_REASON_STOP
    assert finished2.matched_stop == stop
    assert stop not in text2
    assert text2 == baseline_text[: baseline_text.index(stop)]


def test_stop_token_id_ends_generation(worker):
    _, chunks, _ = run(worker.stub, make_submit(max_tokens=8))
    baseline_ids = generated_ids(chunks)
    assert len(baseline_ids) == 8
    stop_id = baseline_ids[3]

    _, chunks2, finished2 = run(
        worker.stub, make_submit(max_tokens=8, stop_token_ids=[stop_id])
    )
    assert finished2.finish_reason == pb.FINISH_REASON_STOP
    assert finished2.matched_stop
    assert generated_ids(chunks2) == baseline_ids[:4]
    assert finished2.completion_tokens == 4


# -- Cancel --------------------------------------------------------------------


def test_cancel_mid_stream_stops_within_two_steps(worker):
    request = make_submit(max_tokens=512)
    stream = worker.stub.Submit(request)

    event = next(stream)
    assert event.WhichOneof("event") == "admitted"
    seen_tokens = 0
    while seen_tokens < 3:
        event = next(stream)
        if event.WhichOneof("event") == "tokens":
            seen_tokens += len(event.tokens.token_ids)

    # Grab the live request object before Finished removes it from the
    # registry, so we can assert the step-level stop guarantee.
    req_obj = worker.engine._registry.get(request.request_id)
    assert req_obj is not None

    ack = worker.stub.Cancel(pb.CancelRequest(request_id=request.request_id))
    assert ack.found

    finished = None
    delivered = seen_tokens
    for event in stream:
        kind = event.WhichOneof("event")
        if kind == "tokens":
            delivered += len(event.tokens.token_ids)
        elif kind == "finished":
            finished = event.finished
    assert finished is not None
    assert finished.finish_reason == pb.FINISH_REASON_CANCELLED
    assert finished.completion_tokens < 512

    assert req_obj.cancel_step is not None
    steps_after_cancel = req_obj.steps_done - req_obj.cancel_step
    assert steps_after_cancel <= 2, (
        f"engine ran {steps_after_cancel} steps after Cancel"
    )
    assert delivered <= finished.completion_tokens


def test_cancel_unknown_request_reports_not_found(worker):
    ack = worker.stub.Cancel(pb.CancelRequest(request_id="no-such-request"))
    assert not ack.found


# -- Queueing ------------------------------------------------------------------


def test_second_request_queues_behind_first(worker):
    first_stream = worker.stub.Submit(make_submit(max_tokens=48))
    first_admitted = next(first_stream)
    assert first_admitted.admitted.queue_position == 0

    second_stream = worker.stub.Submit(make_submit(max_tokens=4))
    second_admitted = next(second_stream)
    assert second_admitted.admitted.queue_position >= 1

    finish = [e.finished for e in first_stream if e.WhichOneof("event") == "finished"]
    assert finish and finish[0].finish_reason != pb.FINISH_REASON_ERROR
    finish = [e.finished for e in second_stream if e.WhichOneof("event") == "finished"]
    assert finish and finish[0].finish_reason != pb.FINISH_REASON_ERROR


# -- Validation: malformed input yields Finished{error}, never a crash ----------


def expect_single_error(stub, request: pb.SubmitRequest, code) -> pb.Finished:
    events = list(stub.Submit(request))
    assert len(events) == 1
    finished = events[0].finished
    assert finished.finish_reason == pb.FINISH_REASON_ERROR
    assert finished.error_code == code
    return finished


def test_zero_max_tokens_is_invalid(worker):
    request = make_submit(max_tokens=0)
    expect_single_error(worker.stub, request, pb.WORKER_ERROR_INVALID_REQUEST)


def test_missing_input_is_invalid(worker):
    request = pb.SubmitRequest(
        request_id=str(uuid.uuid4()),
        stopping=pb.StoppingParams(max_tokens=4),
    )
    expect_single_error(worker.stub, request, pb.WORKER_ERROR_INVALID_REQUEST)


def test_context_overflow_is_rejected(worker):
    info = worker.stub.GetInfo(pb.InfoRequest())
    request = make_submit(max_tokens=info.max_context_len + 1)
    expect_single_error(worker.stub, request, pb.WORKER_ERROR_CTX_OVERFLOW)


def test_grammar_is_unsupported(worker):
    request = make_submit(max_tokens=4)
    request.grammar.json_schema = '{"type": "object"}'
    expect_single_error(worker.stub, request, pb.WORKER_ERROR_GRAMMAR_UNSUPPORTED)


def test_worker_survives_bad_requests(worker):
    status = worker.stub.Health(pb.HealthRequest())
    assert status.state == pb.WORKER_STATE_READY
    _, _, finished = run(worker.stub, make_submit(max_tokens=2))
    assert finished.finish_reason != pb.FINISH_REASON_ERROR


# -- Tokenize ------------------------------------------------------------------


def test_tokenize_matches_submit_prompt_count(worker):
    tokenized = worker.stub.Tokenize(
        pb.TokenizeRequest(text=PROMPT, add_special_tokens=True)
    )
    assert len(tokenized.token_ids) > 0
    admitted, _, _ = run(worker.stub, make_submit(text=PROMPT, max_tokens=1))
    assert admitted.prompt_tokens == len(tokenized.token_ids)


def test_tokenize_special_tokens_flag(worker):
    with_special = worker.stub.Tokenize(
        pb.TokenizeRequest(text=PROMPT, add_special_tokens=True)
    )
    without = worker.stub.Tokenize(
        pb.TokenizeRequest(text=PROMPT, add_special_tokens=False)
    )
    assert len(with_special.token_ids) > len(without.token_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
