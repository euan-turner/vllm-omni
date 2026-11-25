# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Key-token scoring and evidence residency from a synthetic duplex tick trace.

External dependencies: zhconv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.av_fd_longmem.residency import (
    TraceRejectedError,
    append_chunk,
    load_session_trace,
    probe_residency,
    seq_chunk,
    unit_spans,
)
from benchmarks.av_fd_longmem.schema import ClipRef, Item, ItemClips, Probe, QueryClip
from benchmarks.av_fd_longmem.score import TimedText, answer_keys, carries_key, judge_probe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION = "duplex-abc"


def _item(answer: str) -> Item:
    return Item(
        item_id="i",
        source="test",
        lang="zh",
        clips=ItemClips(
            evidence=[ClipRef("e.mp4", 4)],
            query=QueryClip("q.mp4", 33, 0, 3000, 30),
        ),
        query_text="q",
        query_text_original="q",
        answer=answer,
        modality_declared="visual",
        provenance={},
    )


def _probe(query_onset: int = 100) -> Probe:
    return Probe(
        item_id="i",
        evidence_chunks=[[30, 33]],
        query_onset_chunk=query_onset,
        query_end_ms=query_onset * 1000 + 3000,
        answer_window_chunks=[query_onset, query_onset + 32],
    )


@pytest.mark.parametrize(
    ("answer", "reply", "expected"),
    [
        ("4号车13号门", "是4车13门。", True),
        ("4号车13号门", "是14车13门。", False),
        ("400-097-8586", "电话是4000978586", True),
        ("400-097-8586", "电话是4000-097-8586", False),
        ("2元一颗", "标价是2元一颗", True),
        ("2元一颗", "标价是12元", False),
        ("2元一颗", "是2.99美元一颗。", False),
        ("2元一颗", "是两点九九美元一颗", False),
        ("麗枫酒店（LAVANDE）", "叫做 Lavande丽帆酒店", True),
        ("麗枫酒店（LAVANDE）", "叫做丽枫酒店", True),
        ("麗枫酒店（LAVANDE）", "叫做如家酒店", False),
        ("168元", "标价是一百六十八元", True),
        ("2元一颗", "标价是两元一颗", True),
        ("4号车13号门", "十四车十三门", False),
        ("400-097-8586", "四零零零九七八五八六", True),
        ("629元", "六百二十九元", True),
        ("2元一颗", "我记得是2块9毛9一颗哦。", False),
        ("2元一颗", "两元九角九分一颗", False),
        ("2元一颗", "两块一颗", True),
        ("2元一颗", "2块钱一颗", True),
    ],
)
def test_a_reply_is_correct_when_it_carries_an_answer_key(answer: str, reply: str, expected: bool) -> None:
    assert carries_key(reply, answer_keys(answer)) is expected


def test_only_text_starting_in_the_answer_window_is_judged_and_earlier_answers_flag_rehearsal() -> None:
    chunks = [
        TimedText(50.0, 52.0, "看，168元的墨镜"),
        TimedText(104.0, 106.0, "是168元。"),
        TimedText(140.0, 141.0, "好"),
    ]

    verdict = judge_probe(_item("168元"), _probe(), chunks, channel="transcript")

    assert (verdict.verdict, verdict.reply, verdict.rehearsed, verdict.first_reply_s) == (
        "correct",
        "是168元。",
        True,
        104.0,
    )


def test_no_text_in_the_window_is_silent() -> None:
    verdict = judge_probe(_item("168元"), _probe(), [TimedText(10.0, 11.0, "你好")], channel="transcript")

    assert verdict.verdict == "silent"


def _write_trace(path: Path, ev_rows: list[list]) -> None:
    """One unit per second: unit seq k holds stream chunk k-1 (client appends 5k-4 .. 5k)."""
    rows: list[list] = []
    for seq in range(1, 201):
        rows.append(["start", seq * 10, SESSION, seq, seq * 5, False])
        rows.append(["acc", seq * 10 + 1, SESSION, seq, 0, seq, "req", seq * 10])
    rows += ev_rows
    rows.append(["acc", 1, "duplex-other", 1, 0, 1, "req", 0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _ev(at_seq: int, units: list[list[int]]) -> list:
    return ["ev", 10_000 + at_seq, SESSION, 0, at_seq, "compact", {"delta": 0}, units]


def test_units_span_five_appends_and_evidence_maps_to_its_units(tmp_path: Path) -> None:
    _write_trace(tmp_path / "t.jsonl", [])
    records = load_session_trace(tmp_path, SESSION)

    spans = unit_spans(records)

    assert (spans[0].first_append, spans[0].last_append) == (1, 5)
    assert probe_residency(_probe(), records).evidence_units == [31, 32, 33, 34]


@pytest.mark.parametrize(
    ("ev_rows", "label", "frac"),
    [
        ([], "in", 0.0),
        (
            [_ev(90, [[31, 79, 79, 10, 64], [32, 79, 79, 10, 64], [33, 79, 79, 10, 64], [34, 79, 69, 10, 56]])],
            "out",
            None,
        ),
        ([_ev(90, [[31, 79, 79, 10, 64], [32, 79, 40, 5, 30]])], "partial", None),
        (
            [_ev(120, [[31, 79, 79, 10, 64], [32, 79, 79, 10, 64], [33, 79, 79, 10, 64], [34, 79, 79, 10, 64]])],
            "in",
            0.0,
        ),
    ],
    ids=["no-drops", "all-out-with-sink-residue", "partial", "drop-after-query-onset"],
)
def test_probe_labels_count_only_drops_before_the_query(
    tmp_path: Path, ev_rows: list[list], label: str, frac: float | None
) -> None:
    _write_trace(tmp_path / "t.jsonl", ev_rows)

    residency = probe_residency(_probe(query_onset=100), load_session_trace(tmp_path, SESSION))

    assert residency.label == label
    if label == "out":
        assert residency.evicted_token_frac == pytest.approx((79 * 3 + 69) / (79 * 4))
        assert residency.first_evict_chunk == residency.last_evict_chunk == 89  # seq 90 is chunk 89
        assert residency.chunks_since_eviction == 11
    elif label == "partial":
        assert residency.evicted_token_frac is None
    else:
        assert residency.evicted_token_frac == frac


def test_unit_chunks_follow_the_append_layout(tmp_path: Path) -> None:
    _write_trace(tmp_path / "t.jsonl", [])
    spans = unit_spans(load_session_trace(tmp_path, SESSION))

    assert [append_chunk(n) for n in (1, 5, 6, 10)] == [0, 0, 1, 1]
    assert seq_chunk(spans, 1) == 0 and seq_chunk(spans, 100) == 99


def test_a_drop_on_a_silence_continuation_unit_counts_at_the_previous_client_chunk(tmp_path: Path) -> None:
    _write_trace(tmp_path / "t.jsonl", [])
    path = tmp_path / "t.jsonl"
    silence = [["start", 905, SESSION, 999, 0, True], ["acc", 906, SESSION, 999, 0, 90_5, "req", 905]]
    drop = _ev(90_5, [[31, 79, 79, 10, 64], [32, 79, 79, 10, 64], [33, 79, 79, 10, 64], [34, 79, 79, 10, 64]])
    path.write_text(path.read_text() + "".join(json.dumps(r) + "\n" for r in [*silence, drop]), encoding="utf-8")
    records = load_session_trace(tmp_path, SESSION)

    assert seq_chunk(unit_spans(records), 90_5) == 199
    residency = probe_residency(_probe(query_onset=250), records)
    assert residency.label == "out" and residency.last_evict_chunk == 199


def test_a_trace_with_dropped_records_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "t.jsonl").write_text(json.dumps(["trace_dropped", 3]) + "\n", encoding="utf-8")

    with pytest.raises(TraceRejectedError):
        load_session_trace(tmp_path, SESSION)
