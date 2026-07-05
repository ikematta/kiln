"""gRPC servicer implementing the Kiln worker protocol (Phase 1 surface).

Implemented: GetInfo, Health, Submit (server-streamed), Cancel, Tokenize
(required by CAPABILITY_TOKENIZER_OWNED). Drain and Stats inherit the
generated UNIMPLEMENTED behavior until their phases.

Validation failures never crash the worker or abort the RPC: per the proto,
malformed input yields a single ``Finished{finish_reason=ERROR}`` event.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from collections.abc import Iterator

import grpc

from ._gen import worker_pb2 as pb
from ._gen import worker_pb2_grpc
from .engine import PREFILL_CHUNK, Engine, Request

_WORKER_VERSION = importlib.metadata.version("kiln-worker-py")


def _error_event(code: int, detail: str) -> pb.TokenEvent:
    return pb.TokenEvent(
        finished=pb.Finished(
            finish_reason=pb.FINISH_REASON_ERROR,
            error_code=code,
            error_detail=detail,
        )
    )


class WorkerServicer(worker_pb2_grpc.WorkerServicer):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def GetInfo(
        self, request: pb.InfoRequest, context: grpc.ServicerContext
    ) -> pb.WorkerInfo:
        info = self._engine.info
        resp = pb.WorkerInfo(
            model_id=self._engine.model_id,
            worker_kind="python",
            worker_version=_WORKER_VERSION,
            capabilities=[pb.CAPABILITY_TOKENIZER_OWNED],
            kv_block_size=0,
            # ADR 0002 B': 0 = greedy determinism under batching is not
            # guaranteed by this worker (mlx-lm path, no sub-batching).
            max_deterministic_decode_width=0,
        )
        if info is not None:
            resp.model_path = info.model_path
            resp.architecture = info.architecture
            resp.weights_fingerprint = info.weights_fingerprint
            resp.max_context_len = info.max_context_len
            resp.vocab_size = info.vocab_size
            resp.dtype = info.dtype
            resp.chat_template_hash = info.chat_template_hash
        return resp

    def Health(
        self, request: pb.HealthRequest, context: grpc.ServicerContext
    ) -> pb.HealthStatus:
        waiting, running = self._engine.queue_counts()
        return pb.HealthStatus(
            state=self._engine.state,
            memory=self._engine.memory_report(),
            requests_waiting=waiting,
            requests_running=running,
            uptime_ms=self._engine.uptime_ms(),
            detail=self._engine.detail,
        )

    def Submit(
        self, request: pb.SubmitRequest, context: grpc.ServicerContext
    ) -> Iterator[pb.TokenEvent]:
        engine = self._engine
        if engine.state != pb.WORKER_STATE_READY:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"worker not ready (state={pb.WorkerState.Name(engine.state)})",
            )

        if not request.request_id:
            yield _error_event(pb.WORKER_ERROR_INVALID_REQUEST, "missing request_id")
            return
        if request.HasField("grammar"):
            yield _error_event(
                pb.WORKER_ERROR_GRAMMAR_UNSUPPORTED,
                "python worker does not support grammars",
            )
            return
        if request.echo_prompt:
            yield _error_event(
                pb.WORKER_ERROR_INVALID_REQUEST,
                "echo_prompt is not supported by the python worker",
            )
            return
        if request.stopping.max_tokens == 0:
            yield _error_event(
                pb.WORKER_ERROR_INVALID_REQUEST, "max_tokens must be >= 1"
            )
            return

        input_kind = request.WhichOneof("input")
        if input_kind == "raw_text":
            prompt_ids = list(engine.tokenizer.encode(request.raw_text))
        elif input_kind == "token_ids":
            prompt_ids = list(request.token_ids.ids)
        else:
            yield _error_event(
                pb.WORKER_ERROR_INVALID_REQUEST, "missing input (raw_text|token_ids)"
            )
            return
        if not prompt_ids:
            yield _error_event(pb.WORKER_ERROR_INVALID_REQUEST, "empty prompt")
            return

        max_ctx = engine.info.max_context_len if engine.info else 0
        if max_ctx and len(prompt_ids) + request.stopping.max_tokens > max_ctx:
            yield _error_event(
                pb.WORKER_ERROR_CTX_OVERFLOW,
                f"prompt ({len(prompt_ids)}) + max_tokens "
                f"({request.stopping.max_tokens}) exceeds context ({max_ctx})",
            )
            return

        req = Request(
            request_id=request.request_id,
            prompt_ids=prompt_ids,
            sampling=request.sampling,
            stopping=request.stopping,
            enqueued_at=time.monotonic(),
        )
        position = engine.submit(req)
        if position is None:
            yield _error_event(
                pb.WORKER_ERROR_INVALID_REQUEST,
                f"request_id already active: {request.request_id}",
            )
            return

        # Client disconnect (or stream cancellation) cancels the request; a
        # no-op if it already finished.
        context.add_callback(lambda: engine.cancel(request.request_id))

        yield pb.TokenEvent(
            admitted=pb.RequestAdmitted(
                queue_position=position,
                prompt_tokens=len(prompt_ids),
                prefill_chunks_estimated=math.ceil(len(prompt_ids) / PREFILL_CHUNK),
            )
        )
        while True:
            event = req.events.get()
            yield event
            if event.WhichOneof("event") == "finished":
                return

    def Cancel(
        self, request: pb.CancelRequest, context: grpc.ServicerContext
    ) -> pb.CancelAck:
        return pb.CancelAck(found=self._engine.cancel(request.request_id))

    def Tokenize(
        self, request: pb.TokenizeRequest, context: grpc.ServicerContext
    ) -> pb.TokenizeResponse:
        engine = self._engine
        if engine.state != pb.WORKER_STATE_READY:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"worker not ready (state={pb.WorkerState.Name(engine.state)})",
            )
        ids = engine.tokenizer.encode(
            request.text, add_special_tokens=request.add_special_tokens
        )
        return pb.TokenizeResponse(token_ids=ids)
