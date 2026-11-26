# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The Stage-0 worker's opt-in ``ev`` tick-trace record of dropped window units.

Both drop paths are covered: KV compaction (``compact``, which can cut a unit
part-way) and prompt replacement (``replace``, whole units). Each unit carries
the ``seq`` of the append that opened it, so a reader can tell which input
chunk left the model's context and when.

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: torch (embedding rows only).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
    _MiniCPMO45WindowUnit,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import MiniCPMO45DuplexWorkerHelper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNIT, AUDIO, IMAGE, TEXT = 1, 2, 3, 4


class _RecordingTracer:
    def __init__(self) -> None:
        self.records: list[tuple] = []

    def emit(self, record: tuple) -> None:
        self.records.append(record)


def _runtime(tracer: _RecordingTracer | None) -> MiniCPMO45Stage0DuplexRuntime:
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime._tick_trace = tracer
    runtime.audio_placeholder_token_id = AUDIO
    runtime.tokenizer = SimpleNamespace(unk_token_id=IMAGE)
    runtime._embed_token = lambda token_id: torch.zeros(1, 4)
    return runtime


def _unit(seq: int, token_ids: list[int]) -> _MiniCPMO45WindowUnit:
    return _MiniCPMO45WindowUnit(embeds=[torch.zeros(1, 4) for _ in token_ids], token_ids=list(token_ids), seq=seq)


def _state() -> _MiniCPMO45Stage0SessionState:
    # unit 7: <unit> + 2 image + 3 audio + 2 text = 8 tokens; unit 8 likewise.
    tokens = [UNIT, IMAGE, IMAGE, AUDIO, AUDIO, AUDIO, TEXT, TEXT]
    return _MiniCPMO45Stage0SessionState(
        session_id="sess-1",
        context_token_ids=[0] * 10,
        window_units=[_unit(7, tokens), _unit(8, tokens), _unit(9, tokens)],
    )


def test_a_compaction_records_whole_and_partial_unit_drops() -> None:
    tracer = _RecordingTracer()
    state = _state()

    # Window units start right after the 10-token prefix; drop 12 tokens:
    # all of unit 7 and the first 4 of unit 8 (<unit>, 2 image, 1 audio).
    _runtime(tracer)._evict_window_units_for_reanchor(
        state, {"delta": 12, "sink_end": 10, "prefix_tokens": 10}, epoch=0, seq=21
    )

    assert [unit.seq for unit in state.window_units] == [8, 9]
    assert len(state.window_units[0].token_ids) == 4
    [(kind, _t, session_id, epoch, seq, path, kv, rows)] = tracer.records
    assert (kind, session_id, epoch, seq, path) == ("ev", "sess-1", 0, 21, "compact")
    assert kv == {"delta": 12, "sink_end": 10, "moved_from": None, "prefix_tokens": 10}
    assert rows == [[7, 8, 8, 3, 2], [8, 8, 4, 1, 2]]


def test_a_compaction_records_the_kv_geometry_when_the_sink_ends_before_the_window() -> None:
    """``sink_end < prefix_tokens``: the KV loses suffix tokens the worker cannot attribute.

    The worker clamps the drop to the window start, so the rows alone look
    consistent; the recorded geometry is what lets a reader detect the gap.
    """
    tracer = _RecordingTracer()
    state = _state()

    _runtime(tracer)._evict_window_units_for_reanchor(
        state, {"delta": 8, "sink_end": 16, "moved_from": 24, "prefix_tokens": 18}, epoch=0, seq=21
    )

    [(_kind, _t, _session, _epoch, _seq, _path, kv, rows)] = tracer.records
    assert kv == {"delta": 8, "sink_end": 16, "moved_from": 24, "prefix_tokens": 18}
    assert kv["sink_end"] < kv["prefix_tokens"]
    assert rows == [[7, 8, 8, 3, 2]]


def test_a_replacement_records_the_whole_units_it_drops() -> None:
    tracer = _RecordingTracer()
    state = _state()

    _runtime(tracer)._window_replacement_parts(
        state, {"replace": True, "drop_units": 2, "mode": "basic"}, epoch=0, seq=30
    )

    assert [unit.seq for unit in state.window_units] == [9]
    [(kind, _t, session_id, epoch, seq, path, kv, rows)] = tracer.records
    assert (kind, session_id, epoch, seq, path, kv) == ("ev", "sess-1", 0, 30, "replace", None)
    assert rows == [[7, 8, 8, 3, 2], [8, 8, 8, 3, 2]]


def test_a_replacement_that_fails_validation_records_nothing() -> None:
    tracer = _RecordingTracer()
    window = {"replace": True, "drop_units": 1, "mode": "basic", "replacement_prompt_len": 999}

    with pytest.raises(RuntimeError, match="rebuild length mismatch"):
        _runtime(tracer)._window_replacement_parts(_state(), window, epoch=0, seq=30)

    assert tracer.records == []


def test_nothing_is_recorded_without_a_drop() -> None:
    tracer = _RecordingTracer()
    state = _state()
    runtime = _runtime(tracer)

    runtime._window_replacement_parts(state, {"replace": True, "drop_units": 0, "mode": "basic"}, epoch=0, seq=30)
    runtime._evict_window_units_for_reanchor(state, {"delta": 0, "sink_end": 10, "prefix_tokens": 10})

    assert tracer.records == []


def test_drops_are_applied_but_not_recorded_when_tracing_is_off() -> None:
    state = _state()

    _runtime(None)._evict_window_units_for_reanchor(state, {"delta": 8, "sink_end": 10, "prefix_tokens": 10})

    assert [unit.seq for unit in state.window_units] == [8, 9]


def test_a_finalized_unit_keeps_the_seq_of_the_append_that_opened_it() -> None:
    state = _MiniCPMO45Stage0SessionState(session_id="sess-1", window_enabled=True)
    state.pending_window_unit = _MiniCPMO45WindowUnit(embeds=[torch.zeros(1, 4)], token_ids=[UNIT], seq=12)
    state.pending_window_generated_tokens = [TEXT]

    _runtime(None)._finalize_window_unit(state, closure_token_ids=[TEXT])

    assert [(unit.seq, unit.token_ids) for unit in state.window_units] == [(12, [UNIT, TEXT, TEXT])]


def test_the_worker_reanchor_hands_the_append_identity_to_the_eviction() -> None:
    tracer = _RecordingTracer()
    state = _state()
    runtime = _runtime(tracer)
    runtime.sessions = {"sess-1": state}
    runner = SimpleNamespace(
        model=SimpleNamespace(
            _minicpmo45_duplex_data_plane_helper=runtime,
            _minicpmo45_duplex_request_sessions={"req-1": "sess-1"},
        ),
        input_batch=SimpleNamespace(req_ids=["req-1"], num_reqs=1, num_computed_tokens_cpu=[64]),
        model_intermediate_buffer={
            "req-1": {
                "duplex": {
                    "session_id": "sess-1",
                    "epoch": 0,
                    "seq": 5,
                    "stage0_reanchor": {
                        "delta": 16,
                        "moved_from": 32,
                        "sink_blocks": 1,
                        "sink_end": 16,
                        "prefix_tokens": 10,
                        "old_computed_tokens": 64,
                    },
                }
            }
        },
        requests={},
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(block_size=16),
        kv_caches=[],
    )

    MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)

    [(_kind, _t, _session, epoch, seq, path, kv, rows)] = tracer.records
    assert (epoch, seq, path, kv["delta"]) == (0, 5, "compact", 16)
    # Window offset 6 (sink_end 16 - prefix 10): the last 2 tokens of unit 7, then 8 of unit 8, then 6 of unit 9.
    assert rows == [[7, 8, 2, 0, 0], [8, 8, 8, 3, 2], [9, 8, 6, 3, 2]]
