# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniInteract adapter: annotation rows, amended query onset and the evidence window.

The user speaks each question on the clip soundtrack, close to the annotated
``question_time``. The amended query time is the onset of the speech segment
whose transcript matches the annotated question; evidence ends just before it,
so the native question is never streamed. Items with no matching segment are
excluded.

External dependencies: jiwer (character error rate); speech segmentation is
injected (``asr.FunasrSpeechSegmenter`` in production).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jiwer

#: Seconds searched around the annotated question time for the spoken question.
ONSET_SEARCH_BEFORE_S = 5.0
ONSET_SEARCH_AFTER_S = 8.0
#: A segment matches the question when its CER against the question text is at most this.
ONSET_MAX_CER = 0.5
#: Evidence ends this long before the onset, then is floored to a whole second.
EVIDENCE_GUARD_S = 0.3

_PUNCTUATION = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech span (clip seconds) and its transcript."""

    start_s: float
    end_s: float
    text: str


#: (mono float32 samples, sample rate, offset of the samples in clip seconds) -> segments in clip seconds.
SpeechSegmenter = Callable[[Sequence[float], int, float], list[SpeechSegment]]


@dataclass(frozen=True)
class AnnotatedQuestion:
    clip: str
    qa_index: int
    question_time_s: float
    answer_time_s: float
    question_text: str
    answer_text: str
    question_type: str


@dataclass(frozen=True)
class OnsetMatch:
    onset_s: float
    segment: SpeechSegment
    cer: float


def parse_annotation_time(value: object) -> float | None:
    """Seconds from an annotation time: a number, ``mm:ss`` or ``hh:mm:ss``; ``None`` if unparseable.

    Same rules as ``vllm_omni.benchmarks.omniinteract_eval`` uses for these annotations.
    """
    if isinstance(value, int | float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        pass
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        return numbers[0] * 60.0 + numbers[1]
    return numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2]


def annotation_rows(data_root: Path, subset: str, clip: str) -> list[dict[str, object]]:
    path = data_root / subset / "annotations" / f"{clip}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of QA rows")
    return rows


def annotated_question(data_root: Path, subset: str, clip: str, qa_index: int) -> AnnotatedQuestion:
    row = annotation_rows(data_root, subset, clip)[qa_index]
    question_time = parse_annotation_time(row.get("question_time"))
    answer_time = parse_annotation_time(row.get("answer_time"))
    if question_time is None or answer_time is None:
        raise ValueError(f"{subset}/{clip}[{qa_index}]: unparseable question/answer time")
    return AnnotatedQuestion(
        clip=clip,
        qa_index=qa_index,
        question_time_s=question_time,
        answer_time_s=answer_time,
        question_text=str(row["question_text"]),
        answer_text=str(row["answer_text"]),
        question_type=str(row.get("question_type", "")),
    )


def video_path(data_root: Path, subset: str, clip: str) -> Path:
    return data_root / subset / "videos" / f"{clip}.mp4"


def _normalise(text: str) -> str:
    return _PUNCTUATION.sub("", text)


def question_cer(question_text: str, transcript: str) -> float:
    """Character error rate of a transcript against the question, ignoring spaces and punctuation."""
    reference = _normalise(question_text)
    hypothesis = _normalise(transcript)
    if not reference:
        raise ValueError("empty question text")
    if not hypothesis:
        return 1.0
    return float(jiwer.cer(reference, hypothesis))


def match_question_onset(question_text: str, segments: Sequence[SpeechSegment]) -> OnsetMatch | None:
    """The segment that best matches the question, if its CER is within ``ONSET_MAX_CER``.

    A segment may carry a short lead-in before the question (e.g. "可以吗"); the
    onset is the segment start, so the lead-in is cut from the evidence too.
    """
    scored = [(question_cer(question_text, segment.text), segment) for segment in segments]
    if not scored:
        return None
    cer, best = min(scored, key=lambda pair: (pair[0], pair[1].start_s))
    if cer > ONSET_MAX_CER:
        return None
    return OnsetMatch(onset_s=best.start_s, segment=best, cer=cer)


def evidence_window(onset_s: float, *, lookback_s: int) -> tuple[int, int] | None:
    """Whole-second evidence span ``[start, end)`` ending before the spoken question.

    ``None`` when the onset leaves no whole second of evidence.
    """
    end = math.floor(onset_s - EVIDENCE_GUARD_S)
    start = max(0, end - lookback_s)
    if end <= start:
        return None
    return start, end
