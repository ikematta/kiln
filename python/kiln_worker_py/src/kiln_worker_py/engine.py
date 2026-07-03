"""Sequential mlx-lm generation engine.

One dedicated engine thread owns every MLX operation (load, prefill, decode,
RNG seeding); gRPC handler threads only enqueue requests and read event
queues. Requests run one at a time (SPEC §9.2: correctness first; batching in
the Python worker is a Phase 9 improvement).

Cancellation: ``mlx_lm.generate.generate_step`` yields one evaluated token per
engine step, so checking the cancel flag between yields bounds the overshoot
to the one step already pipelined — within the 2-step budget the proto
promises for Cancel.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field

import mlx.core as mx
import psutil
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from ._gen import worker_pb2 as pb
from .modelinfo import ModelInfo, read_model_info
from .stops import StopStringMatcher

_LOG = logging.getLogger(__name__)

# Proto: repetition_window == 0 means "worker default (64)".
DEFAULT_REPETITION_WINDOW = 64
PREFILL_CHUNK = 2048


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    sampling: pb.SamplingParams
    stopping: pb.StoppingParams
    enqueued_at: float
    events: queue.Queue[pb.TokenEvent] = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    # Engine steps executed for this request; written by the engine thread,
    # snapshotted by Cancel so tests can assert the ≤2-step stop guarantee.
    steps_done: int = 0
    cancel_step: int | None = None
    finished: bool = False


def _token_chunk(token_ids: list[int], text: str) -> pb.TokenEvent:
    return pb.TokenEvent(tokens=pb.TokenChunk(token_ids=token_ids, text=text))


class Engine:
    """Owns the model, the request queue, and the single MLX thread."""

    def __init__(self, model_path: str, model_id: str) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.info: ModelInfo | None = None
        self.tokenizer = None
        self._model = None
        self._state = pb.WORKER_STATE_LOADING
        self._detail = ""
        self._lock = threading.Lock()
        self._registry: dict[str, Request] = {}
        self._queue: queue.Queue[Request | None] = queue.Queue()
        self._running: Request | None = None
        self._started_at = time.monotonic()
        self._process = psutil.Process()
        self._thread = threading.Thread(
            target=self._main, name="kiln-engine", daemon=True
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _main(self) -> None:
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 — report, don't crash the process
            _LOG.exception("model load failed for %s", self.model_path)
            with self._lock:
                self._state = pb.WORKER_STATE_UNHEALTHY
                self._detail = f"model load failed: {type(exc).__name__}"
            return
        with self._lock:
            self._state = pb.WORKER_STATE_READY
        _LOG.info("model ready: %s (%s)", self.model_id, self.model_path)
        while True:
            req = self._queue.get()
            if req is None:
                return
            with self._lock:
                self._running = req
            try:
                self._run(req)
            except Exception as exc:  # noqa: BLE001 — engine thread must survive
                _LOG.exception("request %s failed outside generation", req.request_id)
                self._finish(
                    req,
                    pb.Finished(
                        finish_reason=pb.FINISH_REASON_ERROR,
                        error_code=pb.WORKER_ERROR_INTERNAL,
                        error_detail=type(exc).__name__,
                        prompt_tokens=len(req.prompt_ids),
                    ),
                )
            finally:
                with self._lock:
                    self._running = None

    def _load(self) -> None:
        self.info = read_model_info(self.model_path)
        _LOG.info("loading model %s from %s", self.model_id, self.model_path)
        model, tokenizer = mlx_load(self.model_path)
        self._model = model
        self.tokenizer = tokenizer

    # -- submission / cancellation (called from gRPC threads) -----------------

    def submit(self, req: Request) -> int | None:
        """Enqueue a request; returns its queue position, None on duplicate id."""
        with self._lock:
            if req.request_id in self._registry:
                return None
            self._registry[req.request_id] = req
            position = self._queue.qsize() + (1 if self._running is not None else 0)
        self._queue.put(req)
        return position

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            req = self._registry.get(request_id)
            if req is None or req.finished:
                return False
            req.cancel_step = req.steps_done
            req.cancelled.set()
            return True

    # -- health (called from gRPC threads) ------------------------------------

    @property
    def state(self) -> int:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    def uptime_ms(self) -> int:
        return int((time.monotonic() - self._started_at) * 1000)

    def queue_counts(self) -> tuple[int, int]:
        """(waiting, running) request counts."""
        with self._lock:
            return self._queue.qsize(), 1 if self._running is not None else 0

    def memory_report(self) -> pb.MemoryReport:
        # mx.get_*_memory are allocator stat reads; safe off the engine thread.
        return pb.MemoryReport(
            weights_bytes=self.info.weights_bytes if self.info else 0,
            mlx_active_bytes=mx.get_active_memory(),
            mlx_cache_bytes=mx.get_cache_memory(),
            mlx_peak_bytes=mx.get_peak_memory(),
            process_rss_bytes=self._process.memory_info().rss,
        )

    # -- generation (engine thread only) ---------------------------------------

    def _run(self, req: Request) -> None:
        started_at = time.monotonic()
        prompt_tokens = len(req.prompt_ids)
        s = req.sampling
        stopping = req.stopping

        seed_used = s.seed if s.seed != 0 else secrets.randbits(63)
        base = pb.Finished(
            prompt_tokens=prompt_tokens,
            seed_used=seed_used,
            timings=pb.Timings(
                queued_ms=int((started_at - req.enqueued_at) * 1000),
            ),
        )
        if req.cancelled.is_set():
            base.finish_reason = pb.FINISH_REASON_CANCELLED
            self._finish(req, base)
            return

        mx.random.seed(seed_used)
        sampler = make_sampler(
            temp=s.temperature,
            top_p=s.top_p if 0.0 < s.top_p < 1.0 else 0.0,
            min_p=s.min_p,
            top_k=s.top_k,
        )
        window = s.repetition_window or DEFAULT_REPETITION_WINDOW
        processors = make_logits_processors(
            repetition_penalty=(
                s.repetition_penalty if s.repetition_penalty not in (0.0, 1.0) else None
            ),
            repetition_context_size=window,
            frequency_penalty=s.frequency_penalty or None,
            frequency_context_size=window,
            presence_penalty=s.presence_penalty or None,
            presence_context_size=window,
        )

        stop_ids = set(stopping.stop_token_ids)
        eos_ids = set() if stopping.ignore_eos else set(self.tokenizer.eos_token_ids)
        matcher = StopStringMatcher(list(stopping.stop_strings))
        detok = self.tokenizer.detokenizer
        detok.reset()

        completion = 0
        finish_reason = pb.FINISH_REASON_LENGTH
        matched_stop = ""
        first_token_at: float | None = None

        try:
            for token, _logprobs in generate_step(
                mx.array(req.prompt_ids),
                self._model,
                max_tokens=stopping.max_tokens,
                sampler=sampler,
                logits_processors=processors,
                prefill_step_size=PREFILL_CHUNK,
            ):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                completion += 1
                req.steps_done += 1

                if req.cancelled.is_set():
                    finish_reason = pb.FINISH_REASON_CANCELLED
                    break

                if token in eos_ids or token in stop_ids:
                    finish_reason = pb.FINISH_REASON_STOP
                    detok.finalize()
                    released, matched = matcher.push(detok.last_segment)
                    matched_stop = (
                        matched
                        if matched is not None
                        else self.tokenizer.decode([token])
                    )
                    if matched is None:
                        released += matcher.flush()
                    req.events.put(_token_chunk([token], released))
                    break

                detok.add_token(token)
                last_step = completion == stopping.max_tokens
                if last_step:
                    detok.finalize()
                released, matched = matcher.push(detok.last_segment)
                if matched is not None:
                    finish_reason = pb.FINISH_REASON_STOP
                    matched_stop = matched
                    req.events.put(_token_chunk([token], released))
                    break
                if last_step:
                    released += matcher.flush()
                req.events.put(_token_chunk([token], released))
        except Exception as exc:  # noqa: BLE001 — malformed input must not kill us
            _LOG.exception("generation failed for request %s", req.request_id)
            base.finish_reason = pb.FINISH_REASON_ERROR
            base.error_code = pb.WORKER_ERROR_INTERNAL
            # Never include prompt content in error_detail; the class name is
            # enough for the gateway, the traceback stays in worker logs.
            base.error_detail = type(exc).__name__
            base.completion_tokens = completion
            self._finish(req, base)
            return

        ended_at = time.monotonic()
        first = first_token_at if first_token_at is not None else ended_at
        prefill_s = first - started_at
        decode_s = ended_at - first
        base.finish_reason = finish_reason
        base.completion_tokens = completion
        base.matched_stop = matched_stop
        base.timings.prefill_ms = int(prefill_s * 1000)
        base.timings.decode_ms = int(decode_s * 1000)
        if prefill_s > 0:
            base.timings.prefill_tokens_per_sec = prompt_tokens / prefill_s
        if decode_s > 0 and completion > 1:
            base.timings.decode_tokens_per_sec = (completion - 1) / decode_s
        self._finish(req, base)

    def _finish(self, req: Request, finished: pb.Finished) -> None:
        with self._lock:
            req.finished = True
            self._registry.pop(req.request_id, None)
        req.events.put(pb.TokenEvent(finished=finished))
