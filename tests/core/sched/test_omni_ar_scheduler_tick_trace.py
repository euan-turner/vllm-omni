# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 per-tick trace records (long-context prototype instrumentation)."""

from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.duplex_tick_trace import TICK_TRACE_DIR_ENV, DuplexTickTracer
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _records(tracer: DuplexTickTracer) -> list[dict]:
    with open(tracer.path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _session(rid: str = "s0", prompt_len: int = 30) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        external_req_id=f"ext-{rid}",
        arrival_time=1.0,
        resumable=True,
        prompt_token_ids=list(range(prompt_len)),
        num_prompt_tokens=prompt_len,
        num_computed_tokens=prompt_len,
        output_token_ids=[],
        streaming_queue=deque(),
    )


def test_disabled_without_env_or_off_stage_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(TICK_TRACE_DIR_ENV, raising=False)
    assert not DuplexTickTracer(0).enabled
    assert not DuplexTickTracer(1, trace_dir=str(tmp_path)).enabled
    monkeypatch.setenv(TICK_TRACE_DIR_ENV, str(tmp_path))
    assert DuplexTickTracer(0).enabled


def test_first_tick_then_appended_tick_lifecycle(tmp_path) -> None:
    clock = FakeClock()
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=clock)
    session = _session(prompt_len=30)

    tracer.on_add_request(session, None)
    clock.now = 0.1
    tracer.on_schedule({"s0": 30})
    clock.now = 0.3
    tracer.on_step_output({"s0": 30})
    clock.now = 0.35
    tracer.on_step_output({"s0": 1})
    session.output_token_ids = [7, 99]
    tracer.on_segment_stop(session)

    clock.now = 1.0
    tracer.on_add_request(session, session)
    session.num_prompt_tokens, session.num_computed_tokens = 31, 31
    clock.now = 1.01
    tracer.on_apply_update(session, 12)
    clock.now = 1.02
    tracer.on_schedule({"s0": 12, "other": 5})
    clock.now = 1.10
    tracer.on_step_output({"s0": 12, "other": 5})
    session.output_token_ids = [5]
    clock.now = 1.12
    tracer.on_segment_stop(session)

    first, second = _records(tracer)
    assert first["tick_index"] == 0 and first["prefill_tokens"] == 30 and first["context_tokens_before"] == 0
    assert (first["t_first_scheduled"], first["t_prefill_done"], first["t_end"]) == (0.1, 0.3, 0.35)
    assert first["num_prefill_steps"] == 1 and first["num_decode_steps"] == 1
    assert first["sampled_token_ids"] == [7, 99] and first["status"] == "stop"
    assert second["tick_index"] == 1 and second["t_arrival"] == 1.0 and second["t_apply"] == 1.01
    assert second["context_tokens_before"] == 31 and second["prefill_tokens"] == 12
    assert (second["t_first_scheduled"], second["t_prefill_done"], second["t_end"]) == (1.02, 1.10, 1.12)
    assert second["num_decode_steps"] == 0 and second["sampled_token_ids"] == [5]


def test_closed_tick_records_cuda_memory_from_probe(tmp_path) -> None:
    readings = iter([(100, 200), (None, None)])
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=FakeClock(), memory_probe=lambda: next(readings))
    session = _session()
    tracer.on_add_request(session, None)
    tracer.on_segment_stop(session)
    tracer.on_add_request(session, session)
    tracer.on_apply_update(session, 12)
    tracer.on_segment_stop(session)
    first, second = _records(tracer)
    assert (first["cuda_allocated_bytes"], first["cuda_reserved_bytes"]) == (100, 200)
    assert (second["cuda_allocated_bytes"], second["cuda_reserved_bytes"]) == (None, None)


def test_chunked_prefill_finishes_when_all_tokens_scheduled(tmp_path) -> None:
    clock = FakeClock()
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=clock)
    session = _session(prompt_len=100)
    tracer.on_add_request(session, None)
    tracer.on_schedule({"s0": 60})
    clock.now = 0.2
    tracer.on_step_output({"s0": 60})
    clock.now = 0.4
    tracer.on_step_output({"s0": 40})
    clock.now = 0.5
    tracer.on_step_output({"s0": 1})
    tracer.on_segment_stop(session)
    (record,) = _records(tracer)
    assert record["num_prefill_steps"] == 2 and record["num_decode_steps"] == 1
    assert record["t_prefill_done"] == 0.4


def test_step_finishing_before_the_tick_is_scheduled_is_not_counted(tmp_path) -> None:
    clock = FakeClock()
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=clock)
    session = _session(prompt_len=30)
    tracer._tick_counts["s0"] = 1
    tracer.on_add_request(session, session)
    tracer.on_apply_update(session, 12)
    tracer.on_step_output({"s0": 1})
    tracer.on_schedule({"s0": 12})
    clock.now = 0.2
    tracer.on_step_output({"s0": 12})
    tracer.on_segment_stop(session)
    (record,) = _records(tracer)
    assert record["num_prefill_steps"] == 1 and record["num_scheduled_tokens"] == 12
    assert record["t_prefill_done"] == 0.2


def test_queue_depth_and_overflow_close_open_and_pending_ticks(tmp_path) -> None:
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=FakeClock())
    session = _session()
    tracer.on_add_request(session, None)
    tracer.on_add_request(session, session)
    session.streaming_queue.append(object())
    tracer.on_add_request(session, session)
    tracer.on_overflow("s0")
    records = _records(tracer)
    assert [r["status"] for r in records] == ["overflow", "overflow_dropped", "overflow_dropped"]
    assert [r["queue_depth_at_arrival"] for r in records] == [0, 0, 1]


def test_non_resumable_requests_are_not_traced(tmp_path) -> None:
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=FakeClock())
    request = _session()
    request.resumable = False
    tracer.on_add_request(request, None)
    tracer.on_add_request(request, request)
    tracer.on_segment_stop(request)
    assert _records(tracer) == []


def _scheduler(tracer: DuplexTickTracer) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched._hold_payloadless_outputs = False
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0, session_mode="turn"))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.hash_block_size = 16
    sched.chunk_transfer_adapter = None
    sched.kv_holding_waiting = set()
    sched.deferred_waiting = set()
    sched._free_request_blocks = MagicMock()
    sched.encoder_cache_manager = MagicMock()
    sched._inflight_prefills = set()
    sched._tick_tracer = tracer
    return sched


def _request() -> Request:
    return Request(
        request_id="req-tick",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _update(tokens: list[int]) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=tokens,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_scheduler_session_update_opens_tick_with_context_and_prefill(tmp_path) -> None:
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=FakeClock())
    sched = _scheduler(tracer)
    session = _request()
    session.resumable = True
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0
    tracer._tick_counts[session.request_id] = 1
    tracer.on_add_request(session, session)

    sched._update_request_as_session(session, _update([10, 20]))

    record = tracer._open[session.request_id]
    assert record.context_tokens_before == 6
    assert record.prefill_tokens == 2


def test_scheduler_overflow_closes_the_tick(tmp_path) -> None:
    tracer = DuplexTickTracer(0, trace_dir=str(tmp_path), clock=FakeClock())
    sched = _scheduler(tracer)
    sched.max_model_len = 8
    sched.finish_requests = MagicMock()
    session = _request()
    session.resumable = True
    session.num_computed_tokens = 6
    tracer._tick_counts[session.request_id] = 1
    tracer.on_add_request(session, session)
    tracer.on_apply_update(session, 4)

    assert sched._streaming_update_overflows(session, _update([1, 2, 3, 4]))

    (record,) = _records(tracer)
    assert record["status"] == "overflow"
