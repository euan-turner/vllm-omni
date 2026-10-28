# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-tick timing records for streaming (duplex) sessions at Stage 0.

Purpose: prototype instrumentation for long-context full-duplex profiling.
One record per tick, where a tick is one streaming input update (one client
append, one prefill) plus the decode steps that follow it until the session's
segment stops. Records are written as JSON lines, one file per EngineCore
process, and only when ``VLLM_OMNI_DUPLEX_TICK_TRACE_DIR`` is set.

Metric definitions over these records live in
``benchmarks/duplex_session/long_context/tick_metrics.py``.

Dependencies: vLLM v1 ``Request`` / ``SchedulerOutput`` (read-only access).
Timestamps are ``time.monotonic()`` (CLOCK_MONOTONIC, shared by processes on
one host), so they can be joined with a client running on the same node.
Each closed tick also records this process's CUDA allocator state (debugging
GPU memory growth in long sessions); it is read only when CUDA is already
initialised in the process, so tracing never creates a CUDA context.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import IO, Any

TICK_TRACE_DIR_ENV = "VLLM_OMNI_DUPLEX_TICK_TRACE_DIR"


@dataclass
class TickRecord:
    request_id: str
    external_request_id: str | None
    tick_index: int
    # Scheduler receipt of the append; None for an append that was already
    # queued when its predecessor stopped is impossible, so always set.
    t_arrival: float
    # Frontend arrival (wall clock, vLLM ``Request.arrival_time``).
    arrival_time_wall: float | None
    # Number of appends already queued behind the running tick at receipt.
    queue_depth_at_arrival: int
    t_apply: float | None = None
    context_tokens_before: int | None = None
    prefill_tokens: int | None = None
    t_first_scheduled: float | None = None
    t_prefill_done: float | None = None
    t_end: float | None = None
    num_prefill_steps: int = 0
    num_decode_steps: int = 0
    num_scheduled_tokens: int = 0
    sampled_token_ids: list[int] = field(default_factory=list)
    status: str = "open"
    # torch.cuda.memory_allocated / memory_reserved at t_end; None without CUDA.
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None


def cuda_memory_bytes() -> tuple[int | None, int | None]:
    """(allocated, reserved) bytes of this process's CUDA caching allocator, or (None, None)."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None, None
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


class DuplexTickTracer:
    """Collects tick records from scheduler hooks and appends them to a JSONL file.

    Disabled (every hook is a no-op) unless ``VLLM_OMNI_DUPLEX_TICK_TRACE_DIR``
    is set and the scheduler runs Stage 0. Only resumable (streaming-input)
    requests are traced.
    """

    def __init__(
        self,
        stage_id: int | None,
        *,
        trace_dir: str | None = None,
        clock=time.monotonic,
        memory_probe=cuda_memory_bytes,
    ) -> None:
        trace_dir = trace_dir if trace_dir is not None else os.environ.get(TICK_TRACE_DIR_ENV)
        self.enabled = bool(trace_dir) and (stage_id or 0) == 0
        self._clock = clock
        self._memory_probe = memory_probe
        self._open: dict[str, TickRecord] = {}
        self._pending: dict[str, list[TickRecord]] = {}
        self._tick_counts: dict[str, int] = {}
        self._file: IO[str] | None = None
        self.path: str | None = None
        if self.enabled:
            os.makedirs(trace_dir, exist_ok=True)
            self.path = os.path.join(trace_dir, f"ticks_stage{stage_id or 0}_pid{os.getpid()}.jsonl")
            self._file = open(self.path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

    def _new_record(self, request: Any, *, queue_depth: int) -> TickRecord:
        rid = request.request_id
        index = self._tick_counts.get(rid, 0)
        self._tick_counts[rid] = index + 1
        return TickRecord(
            request_id=rid,
            external_request_id=getattr(request, "external_req_id", None),
            tick_index=index,
            t_arrival=self._clock(),
            arrival_time_wall=getattr(request, "arrival_time", None),
            queue_depth_at_arrival=queue_depth,
        )

    def on_add_request(self, request: Any, existing: Any | None) -> None:
        """Record receipt of a session's first request or of a later append."""
        if not self.enabled:
            return
        if existing is None:
            if not getattr(request, "resumable", False):
                return
            record = self._new_record(request, queue_depth=0)
            record.t_apply = record.t_arrival
            record.context_tokens_before = 0
            record.prefill_tokens = len(getattr(request, "prompt_token_ids", None) or ())
            self._open[request.request_id] = record
            return
        if request.request_id not in self._tick_counts:
            return
        queue = getattr(existing, "streaming_queue", None)
        self._pending.setdefault(request.request_id, []).append(
            self._new_record(request, queue_depth=len(queue) if queue else 0)
        )

    def on_apply_update(self, session: Any, new_tokens: int) -> None:
        """Open the tick of the oldest pending append when it is applied to the session."""
        if not self.enabled:
            return
        pending = self._pending.get(session.request_id)
        if not pending:
            return
        record = pending.pop(0)
        record.t_apply = self._clock()
        record.context_tokens_before = max(int(session.num_prompt_tokens), int(session.num_computed_tokens))
        record.prefill_tokens = int(new_tokens)
        self._open[session.request_id] = record

    def on_schedule(self, num_scheduled_tokens: dict[str, int]) -> None:
        if not self.enabled or not self._open:
            return
        now = self._clock()
        for rid, record in self._open.items():
            if rid in num_scheduled_tokens and record.t_first_scheduled is None:
                record.t_first_scheduled = now

    def on_step_output(self, num_scheduled_tokens: dict[str, int]) -> None:
        """Count one finished model step for every open tick it served."""
        if not self.enabled or not self._open:
            return
        now = self._clock()
        for rid, record in self._open.items():
            tokens = num_scheduled_tokens.get(rid)
            # A step scheduled before this tick's append was applied (async
            # scheduling keeps one in flight) belongs to the previous tick.
            if not tokens or record.t_first_scheduled is None:
                continue
            record.num_scheduled_tokens += int(tokens)
            if record.t_prefill_done is None:
                record.num_prefill_steps += 1
                # The prefill is done once all of its tokens have been scheduled.
                if record.num_scheduled_tokens >= (record.prefill_tokens or 0):
                    record.t_prefill_done = now
            else:
                record.num_decode_steps += 1

    def on_segment_stop(self, request: Any, *, status: str = "stop") -> None:
        """Close the request's open tick with the tokens sampled in it."""
        if not self.enabled:
            return
        record = self._open.pop(request.request_id, None)
        if record is None:
            return
        record.t_end = self._clock()
        record.sampled_token_ids = [int(t) for t in (getattr(request, "output_token_ids", None) or ())]
        record.status = status
        record.cuda_allocated_bytes, record.cuda_reserved_bytes = self._memory_probe()
        self._write(record)

    def on_overflow(self, request_id: str) -> None:
        """Close the request's open and pending ticks after a context overflow."""
        if not self.enabled:
            return
        now = self._clock()
        record = self._open.pop(request_id, None)
        if record is not None:
            record.t_end = now
            record.status = "overflow"
            self._write(record)
        for pending in self._pending.pop(request_id, []):
            pending.t_end = now
            pending.status = "overflow_dropped"
            self._write(pending)

    def _write(self, record: TickRecord) -> None:
        if self._file is not None:
            self._file.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
