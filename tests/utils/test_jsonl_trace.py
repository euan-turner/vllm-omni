# SPDX-License-Identifier: Apache-2.0
"""Tests the opt-in per-process JSONL trace writer and the duplex tick tracer switch.

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: none.
"""

from __future__ import annotations

import json
import os
import queue
from pathlib import Path

import pytest

from vllm_omni.engine.duplex.tick_trace import DUPLEX_TICK_TRACE_ENV, duplex_tick_tracer
from vllm_omni.utils import jsonl_trace
from vllm_omni.utils.jsonl_trace import JsonlTraceWriter, jsonl_trace_writer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ENV = "VLLM_OMNI_TEST_JSONL_TRACE_DIR"


@pytest.fixture(autouse=True)
def _fresh_writers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jsonl_trace, "_writers", {})
    yield
    for writer in jsonl_trace._writers.values():
        writer.close()


def _lines(path: Path) -> list[list]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(DUPLEX_TICK_TRACE_ENV, raising=False)

    assert jsonl_trace_writer(ENV, "unit") is None
    assert duplex_tick_tracer("orchestrator") is None


def test_writes_one_file_per_process_and_flushes_on_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV, str(tmp_path))

    writer = jsonl_trace_writer(ENV, "unit")
    assert writer is not None
    assert jsonl_trace_writer(ENV, "unit") is writer
    writer.emit(("in", 1, "sess", 1))
    writer.emit(("out", 2, "sess", 0, "req", True, 1))
    writer.close()

    assert writer.path == tmp_path / f"unit-{os.getpid()}.jsonl"
    assert _lines(writer.path) == [["in", 1, "sess", 1], ["out", 2, "sess", 0, "req", True, 1]]


def test_full_queue_drops_and_reports_count(tmp_path: Path) -> None:
    writer = JsonlTraceWriter(tmp_path / "drops.jsonl")
    live_queue = writer._queue
    full: queue.Queue = queue.Queue(1)
    full.put_nowait(("filler",))
    writer._queue = full

    writer.emit(("lost",))
    writer.emit(("lost",))
    writer._queue = live_queue
    writer.close()

    assert writer.dropped == 2
    assert _lines(writer.path) == [["trace_dropped", 2]]


def test_duplex_tracer_uses_its_own_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DUPLEX_TICK_TRACE_ENV, str(tmp_path))

    writer = duplex_tick_tracer("stage0")

    assert writer is not None
    assert writer.path.name == f"duplex-ticks-stage0-{os.getpid()}.jsonl"
