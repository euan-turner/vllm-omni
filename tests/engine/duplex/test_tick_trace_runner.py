# SPDX-License-Identifier: Apache-2.0
"""Tests the duplex session runner's opt-in tick-trace records (ingress, dequeue, start, accept).

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: none (recording stage port from test_session_runner).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine.duplex.test_session_runner import SESSION_ID, append_audio, close_harness, open_harness
from vllm_omni.engine.duplex.tick_trace import DUPLEX_TICK_TRACE_ENV
from vllm_omni.utils import jsonl_trace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _records(tmp_path: Path) -> list[list]:
    for writer in jsonl_trace._writers.values():
        writer.close()
    return [json.loads(line) for path in sorted(tmp_path.glob("*.jsonl")) for line in path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_each_stage_append_is_traced_with_its_trigger_and_seq(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(jsonl_trace, "_writers", {})
    monkeypatch.setenv(DUPLEX_TICK_TRACE_ENV, str(tmp_path))
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.run(append_audio())
        await h.run(append_audio(samples=8000))
        await h.run(append_audio(samples=8000))
        request_id = h.stage0_request_id(epoch=0)
    finally:
        await close_harness(h)

    records = _records(tmp_path)
    by_kind = {kind: [row for row in records if row[0] == kind] for kind in ("in", "dq", "start", "acc")}
    assert [row[2:] for row in by_kind["in"]] == [[SESSION_ID, n] for n in (1, 2, 3, 4)]
    assert [row[2:] for row in by_kind["dq"]] == [[SESSION_ID, n] for n in (1, 2, 3, 4)]
    # The half-unit third append buffers; the fourth completes the unit that becomes stage append 3.
    assert [row[2:] for row in by_kind["start"]] == [
        [SESSION_ID, 1, 1, False],
        [SESSION_ID, 2, 2, False],
        [SESSION_ID, 3, 4, False],
    ]
    assert [(row[3], row[4], row[5], row[6]) for row in by_kind["acc"]] == [
        (1, 0, 1, request_id),
        (2, 0, 2, request_id),
        (3, 0, 3, request_id),
    ]
    for start, accept in zip(by_kind["start"], by_kind["acc"], strict=True):
        assert start[1] <= accept[7] <= accept[1]
    for ingress, dequeue in zip(by_kind["in"], by_kind["dq"], strict=True):
        assert ingress[1] <= dequeue[1]


@pytest.mark.asyncio
async def test_runner_writes_nothing_when_tracing_is_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(jsonl_trace, "_writers", {})
    monkeypatch.delenv(DUPLEX_TICK_TRACE_ENV, raising=False)
    h = await open_harness()
    try:
        await h.run(append_audio())
        assert h.runner._tick_trace is None
    finally:
        await close_harness(h)

    assert jsonl_trace._writers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("traced", [False, True])
async def test_orchestrator_forward_is_traced_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, traced: bool
) -> None:
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.tick_trace import duplex_tick_tracer
    from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator
    from vllm_omni.engine.orchestrator import Orchestrator

    monkeypatch.setattr(jsonl_trace, "_writers", {})
    if traced:
        monkeypatch.setenv(DUPLEX_TICK_TRACE_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(DUPLEX_TICK_TRACE_ENV, raising=False)
    calls = []

    async def base_forward(self, req_id, src_stage_id, output, req_state, **kwargs):
        calls.append((req_id, src_stage_id, kwargs))

    monkeypatch.setattr(Orchestrator, "_forward_to_next_stage", base_forward)
    orchestrator = object.__new__(DuplexOrchestrator)
    orchestrator._tick_trace = duplex_tick_tracer("orchestrator")

    await orchestrator._forward_to_next_stage(
        "req", 0, object(), SimpleNamespace(session_id="sess"), is_streaming_session=True
    )

    assert calls == [("req", 0, {"is_streaming_session": True})]
    records = _records(tmp_path)
    if traced:
        assert len(records) == 1
        kind, started, session_id, stage_id, request_id, ended = records[0]
        assert (kind, session_id, stage_id, request_id) == ("fwd", "sess", 0, "req")
        assert started <= ended
    else:
        assert records == []
