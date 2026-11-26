# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""M1 window analysis from a synthetic tick trace; M0 onset room and StreamArena rules, windows and filler.

External dependencies: none (speech segmentation is faked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.av_fd_longmem import audit
from benchmarks.av_fd_longmem.schema import FillerClip
from benchmarks.av_fd_longmem.sources import omniinteract as oi
from benchmarks.av_fd_longmem.sources import streamarena as sa
from benchmarks.av_fd_longmem.window import plan_filler_only, prompt_lengths, resident_span, unit_windows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION = "duplex-abc"


def _trace(n_units: int, evictions: dict[int, list[int]]) -> dict[str, list[list]]:
    """One 79-token unit per chunk (5 appends each); ``evictions`` maps handling seq -> unit seqs dropped whole."""
    records: dict[str, list[list]] = {"start": [], "acc": [], "ev": [], "w": []}
    t = 0
    for seq in range(1, n_units + 1):
        records["start"].append(["start", t, SESSION, seq, seq * 5, False])
        records["acc"].append(["acc", t + 1, SESSION, seq, 0, seq, "r", t])
        records["w"].append(["w", t + 2, SESSION, 0, seq, 100 + 79 * seq, 0, 0, False])
        if seq in evictions:
            units = [[u, 79, 79, 10, 64] for u in evictions[seq]]
            records["ev"].append(["ev", t + 3, SESSION, 0, seq, "compact", {}, units])
        t += 10
    return records


def test_units_get_their_chunk_token_split_and_eviction_lag() -> None:
    records = _trace(20, {11: [1, 2, 3], 16: [4, 5, 6]})

    units = {u.seq: u for u in unit_windows(records)}

    assert units[1].chunk == 0 and units[11].chunk == 10
    assert (units[1].tokens, units[1].audio_tokens, units[1].image_tokens) == (79, 10, 64)
    assert units[1].out_chunk == 10 and units[1].eviction_lag == 10
    assert units[6].eviction_lag == 10 and units[4].eviction_lag == 12
    assert units[7].out_chunk is None and units[7].eviction_lag is None


def test_a_partly_dropped_unit_is_not_out_until_only_a_sink_residue_remains() -> None:
    records = _trace(12, {})
    records["ev"] = [
        ["ev", 1000, SESSION, 0, 10, "compact", {}, [[1, 79, 40, 5, 30]]],
        ["ev", 1001, SESSION, 0, 12, "compact", {}, [[1, 39, 30, 5, 25]]],
    ]

    unit = next(u for u in unit_windows(records) if u.seq == 1)

    assert unit.first_drop_chunk == 9
    assert unit.out_chunk == 11  # 70 of 79 dropped: 9 left, within the 16-token sink residue
    assert (unit.audio_tokens, unit.image_tokens) == (10, 55)


def test_resident_span_is_a_sawtooth_after_the_first_eviction() -> None:
    units = unit_windows(_trace(20, {11: [1, 2, 3], 16: [4, 5, 6]}))

    spans = resident_span(units, 20)

    assert spans[:10] == [None] * 10
    assert spans[10:16] == [8, 9, 10, 11, 12, 10]


def test_prompt_lengths_take_the_longest_prefill_chunk_per_append() -> None:
    records = _trace(3, {})
    records["w"].append(["w", 999, SESSION, 0, 2, 50, 0, 0, False])

    assert prompt_lengths(records, unit_windows(records)) == [(0, 179), (1, 258), (2, 337)]


def test_a_filler_only_plan_has_no_probes_and_plays_clip_prefixes() -> None:
    filler = [FillerClip(f"f{i}", f"f{i}.mp4", 40, "test", None, {}) for i in range(3)]

    plan = plan_filler_only("m1-basic", 100, filler, condition={"window_mode": "basic"})

    assert plan.probes == [] and plan.total_chunks == 100
    assert [(s.clip, s.play_chunks, s.stream_chunk0) for s in plan.segments] == [
        ("f0.mp4", 40, 0),
        ("f1.mp4", 40, 40),
        ("f2.mp4", 20, 80),
    ]


def _annotations(root: Path, clip: str, rows: list[dict[str, object]]) -> None:
    (root / "1q1a" / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "1q1a" / "annotations" / f"{clip}.json").write_text(json.dumps(rows), encoding="utf-8")


def _qa(time: str, text: str, *, interrupted: bool = False) -> dict[str, object]:
    return {"question_time": time, "question_text": text, "answer_time": time, "answer_text": "a",
            "question_type": "realtime", "is_interrupted": interrupted}  # fmt: skip


def test_onset_statuses_and_room_bounded_by_other_questions_speech(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _qa("00:10", "第一个问题"),
        _qa("00:40", "第二个问题"),
        _qa("00:44", "第三个问题", interrupted=True),
        _qa("bad", "坏"),
        _qa("01:10", "这个酒店叫什么名字"),
        _qa("01:40", "你还记不记得那个塔"),
        _qa("02:10", "第七个问题是什么"),
    ]
    _annotations(tmp_path, "0001", rows)
    monkeypatch.setattr(audit, "read_audio_f32", lambda _path: [0.0] * 16000 * 200)
    spoken = [
        (9.5, 11.0, "第一个问题"),
        (39.5, 41.0, "第二个问题"),
        (43.5, 45.0, "第三个问题"),
        (66.5, 68.0, "这个酒店叫什"),  # an earlier, garbled utterance of the same question
        (69.5, 71.0, "这个酒店叫什么名字"),
        (99.5, 101.0, "你还记不记得那个塔"),
        (115.0, 131.0, "我先说一段话然后第七个问题是什么"),  # already under way when the search window opens
    ]

    def segmenter(_samples, _rate, offset_s):
        end_s = offset_s + oi.ONSET_SEARCH_BEFORE_S + oi.ONSET_SEARCH_AFTER_S
        return [
            oi.SpeechSegment(max(lo, offset_s), hi, text) for lo, hi, text in spoken if lo < end_s and offset_s < hi
        ]

    out = audit.audit_clip(tmp_path, "1q1a", "0001", segmenter)

    assert [r.status for r in out] == [
        "ok", "ok", "question_in_evidence", "unparsable_time", "question_in_evidence", "recall_phrased",
        "onset_censored",
    ]  # fmt: skip  # item 2: the near-identical 第二个问题 is spoken inside its evidence window
    assert out[0].onset_s == 9.5 and out[0].onset_delta_s == -0.5
    assert out[0].pre_onset_room_s == pytest.approx(9.2)
    assert out[1].pre_onset_room_s == pytest.approx(39.5 - 0.3 - 11.0)
    assert out[2].pre_onset_room_s == pytest.approx(43.5 - 0.3 - 41.0)
    assert out[4].onset_s == 69.5
    summary = audit.summarise_onsets(out)
    assert summary["realtime_ok_not_interrupted"] == 2
    assert summary["realtime_yield_by_lookback_s"]["10"] == {"items": 1, "clips": 1}


@pytest.mark.parametrize(
    ("question", "recall"),
    [
        ("你还记不记得前面出现的是哪个生肖？", True),
        ("刚刚我一共问了几个问题？", True),
        ("请问这个耳机卖多少钱", False),
        ("请问我前面有几个人", False),  # 前面 is spatial in these clips
    ],
)
def test_recall_phrased_questions_are_flagged(question: str, recall: bool) -> None:
    assert oi.is_recall_phrased(question) is recall


def _row(**overrides) -> sa.RecallRow:
    base = dict(video_id="v", qid=1, lang="zh", domain="影视", question_zh="之前画面里出现的书叫什么？",
                question_en="What was the book shown earlier?", answer_zh="a", answer_en="a", ask_sec=600,
                evidence_ts_sec=[100], evidence_hint=[], horizon_layer="L2")  # fmt: skip
    return sa.RecallRow(**{**base, **overrides})


@pytest.mark.parametrize(
    ("overrides", "reasons"),
    [
        ({}, []),
        ({"lang": "other"}, ["lang_other"]),
        ({"question_zh": "5分钟前出现的那本书叫什么？"}, ["relative_window_phrasing"]),
        ({"question_en": "What was the book shown a few minutes ago?"}, ["relative_window_phrasing"]),
        ({"evidence_ts_sec": [100, 700]}, ["evidence_after_ask"]),
        ({"question_zh": "到目前为止一共进了几个球？"}, ["so_far_phrasing"]),
        ({"question_en": "How many goals so far?"}, ["so_far_phrasing"]),
        ({"question_zh": "过去5分钟里镜头切了几次？"}, ["relative_window_phrasing"]),
        ({"question_en": "In the last ten minutes, who spoke?"}, ["relative_window_phrasing"]),
    ],
)
def test_streamarena_rule_exclusions(overrides: dict[str, object], reasons: list[str]) -> None:
    assert sa.exclusion_reasons(_row(**overrides)) == reasons


def test_zh_en_videos_are_english_queries_in_their_own_stratum_and_recency_wording_needs_a_rewrite() -> None:
    row = _row(lang="zh+en", question_en="Which book did the host just show?")

    assert row.item_lang == "en" and row.question == "Which book did the host just show?"
    assert row.stratum == "zh+en" and _row(lang="en").stratum == "en" and _row(lang="other").stratum is None
    assert sa.needs_lag_neutral_rewrite(_row(question_zh="刚才出现的书叫什么？"))
    assert not sa.needs_lag_neutral_rewrite(_row())


def test_evidence_windows_extend_over_cues_cap_at_15s_each_side_merge_and_stop_at_ask() -> None:
    assert sa.evidence_windows([100], 600) == [(95, 105)]
    assert sa.evidence_windows([100], 600, [(103.5, 108.2)]) == [(95, 109)]
    assert sa.evidence_windows([100], 600, [(70.0, 130.0)]) == [(85, 115)]
    assert sa.evidence_windows([100, 108], 600) == [(95, 113)]
    assert sa.evidence_windows([598], 600) == [(593, 600)]


def test_streamarena_filler_keeps_60s_clear_of_annotated_times_in_300s_clips() -> None:
    clips = sa.filler_runs(1000.0, [500])

    assert clips == [(0, 300), (300, 140), (560, 300), (860, 140)]
    assert sa.filler_runs(1000.0, [100, 250]) == [(310, 300), (610, 300), (910, 90)]


def test_vtt_cues_are_parsed(tmp_path: Path) -> None:
    vtt = tmp_path / "a.vtt"
    vtt.write_text("WEBVTT\n\n1\n00:00:16.000 --> 00:00:24.000\n文字\n\n00:01:01.550 --> 00:01:01.560 align:start\nx\n")

    assert sa.vtt_cues([vtt]) == [(16.0, 24.0), (61.55, 61.56)]
