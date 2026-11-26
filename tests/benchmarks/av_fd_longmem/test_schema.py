# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""av_fd_longmem records survive a JSONL/JSON round trip and refuse foreign versions.

External dependencies: none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.av_fd_longmem.schema import (
    SCHEMA_VERSION,
    ClipRef,
    Item,
    ItemClips,
    Probe,
    QueryClip,
    Segment,
    SessionPlan,
    from_dict,
    read_json,
    read_jsonl,
    to_dict,
    write_json,
    write_jsonl,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _item() -> Item:
    return Item(
        item_id="oi-1q1a-0108-0",
        source="omniinteract",
        lang="zh",
        clips=ItemClips(
            evidence=[ClipRef(path="clips/evidence/oi-1q1a-0108-0__e0.mp4", chunks=4)],
            query=QueryClip(
                path="clips/query/oi-1q1a-0108-0.mp4",
                chunks=34,
                query_onset_chunk=0,
                query_end_ms=2400,
                answer_chunks=30,
            ),
        ),
        query_text="刚才那副墨镜标价多少钱？",
        query_text_original="请问这一个墨镜要多少钱",
        answer="168元",
        modality_declared="visual",
        provenance={"onset_s": 8.64},
    )


def _plan() -> SessionPlan:
    return SessionPlan(
        plan_id="pilot-chained",
        kind="chained",
        seed=0,
        condition={"window_mode": "basic"},
        segments=[Segment(role="evidence", clip="clips/evidence/x.mp4", play_chunks=4, stream_chunk0=30)],
        probes=[
            Probe(
                item_id="oi-1q1a-0108-0",
                evidence_chunks=[[30, 33]],
                query_onset_chunk=94,
                query_end_ms=96400,
                answer_window_chunks=[94, 127],
            )
        ],
        total_chunks=128,
    )


def test_items_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"

    write_jsonl(path, [_item()])

    assert list(read_jsonl(path, Item)) == [_item()]


def test_plans_round_trip_through_json(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"

    write_json(path, _plan())

    assert read_json(path, SessionPlan) == _plan()


def test_a_probe_gap_is_counted_between_evidence_end_and_query_onset() -> None:
    assert _plan().probes[0].gap_chunks == 60


def test_a_record_from_another_schema_version_is_refused() -> None:
    data = to_dict(_item()) | {"schema_version": SCHEMA_VERSION + 1}

    with pytest.raises(ValueError, match="schema_version"):
        from_dict(Item, data)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        from_dict(Item, to_dict(_item()) | {"extra": 1})
