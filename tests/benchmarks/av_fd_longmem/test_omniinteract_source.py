# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniInteract adapter: annotation parsing, amended query onset and evidence window.

External dependencies: jiwer; speech segmentation is faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.av_fd_longmem.sources.omniinteract import (
    SpeechSegment,
    annotated_question,
    evidence_window,
    match_question_onset,
    parse_annotation_time,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_annotation_times_are_parsed_from_mm_ss(tmp_path: Path) -> None:
    annotations = tmp_path / "1q1a" / "annotations"
    annotations.mkdir(parents=True)
    row = {
        "question_time": "03:13",
        "question_text": "请问驯龙高手：没牙仔卖多少钱一件",
        "answer_time": "03:16",
        "answer_text": "629元",
        "question_type": "realtime",
    }
    (annotations / "0124.json").write_text(json.dumps([{}, row]), encoding="utf-8")

    question = annotated_question(tmp_path, "1q1a", "0124", 1)

    assert (question.question_time_s, question.answer_time_s) == (193.0, 196.0)
    assert question.answer_text == "629元"


def test_the_onset_is_the_start_of_the_segment_that_matches_the_question() -> None:
    segments = [
        SpeechSegment(156.0, 158.2, "挑战thedestination"),
        SpeechSegment(163.67, 169.24, "请问一下现在这个是几号车几号门"),
    ]

    match = match_question_onset("请问一下现在这个是几号车几号门", segments)

    assert match is not None
    assert (match.onset_s, match.cer) == (163.67, 0.0)


def test_a_lead_in_and_asr_errors_still_match_and_move_the_onset_earlier() -> None:
    segments = [SpeechSegment(191.87, 196.32, "可以吗请问驯龙高手门牙仔卖多少钱一件")]

    match = match_question_onset("请问驯龙高手：没牙仔卖多少钱一件", segments)

    assert match is not None
    assert match.onset_s == 191.87
    assert 0 < match.cer <= 0.5


def test_no_matching_segment_means_no_onset() -> None:
    assert match_question_onset("请问一下现在这个是几号车几号门", [SpeechSegment(1.0, 2.0, "他那边的人太多了")]) is None
    assert match_question_onset("请问一下现在这个是几号车几号门", []) is None


@pytest.mark.parametrize(
    ("onset_s", "expected"),
    [(8.64, (4, 8)), (217.86, (213, 217)), (2.42, (0, 2)), (262.3, (258, 262)), (0.5, None)],
)
def test_evidence_ends_on_the_last_whole_second_before_the_onset(
    onset_s: float, expected: tuple[int, int] | None
) -> None:
    assert evidence_window(onset_s, lookback_s=4) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("03:13", 193.0), ("01:02:03", 3723.0), (12, 12.0), ("7.5", 7.5), ("3:x", None), ("", None), (None, None)],
)
def test_annotation_times_accept_numbers_mm_ss_and_hh_mm_ss(value: object, expected: float | None) -> None:
    assert parse_annotation_time(value) == expected
