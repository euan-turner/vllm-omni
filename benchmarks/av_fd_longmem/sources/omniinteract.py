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

#: Seconds searched around the annotated question time for the spoken question. The span before
#: covers the evidence lookback, so speech inside the evidence window is seen too.
ONSET_SEARCH_BEFORE_S = 10.0
ONSET_SEARCH_AFTER_S = 8.0
#: A segment matches the question when its CER against the question text is at most this.
ONSET_MAX_CER = 0.5
#: Evidence ends this long before the onset, then is floored to a whole second.
EVIDENCE_GUARD_S = 0.3
#: Evidence lookback ``W`` in seconds (decision 2026-11-25).
EVIDENCE_LOOKBACK_S = 4
#: A matched segment starting this close to a search window that does not start at 0 s was cut by the window.
CENSORED_ONSET_S = 0.2

_PUNCTUATION = re.compile(r"[\s\W_]+", re.UNICODE)
#: Realtime questions already about the past ("你还记不记得…", "刚刚…"); their evidence is not just before the onset.
#: 前面 is left out: in these clips it is spatial ("我前面的男生"). Checked by hand against all 1q1a matches.
_RECALL_PHRASING = re.compile(
    r"刚刚|刚才|之前|记不记得|还记得|一共问|问了几|上次|方才|一开始|我问过|上一个|刚看到|刚出现|刚经过|刚路过"
)


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
    """Seconds from an annotation time: a number, ``mm:ss`` or ``hh:mm:ss``; ``None`` if unparsable.

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
        raise ValueError(f"{subset}/{clip}[{qa_index}]: unparsable question/answer time")
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


def substring_cer(question_text: str, transcript: str) -> float:
    """CER of the question against its best-aligned substring of the transcript, ignoring spaces and punctuation.

    Speech before and after the question in the same transcript costs nothing,
    so a question merged with neighbouring speech (an announcement, a follow-up)
    still matches.
    """
    reference = _normalise(question_text)
    hypothesis = _normalise(transcript)
    if not reference:
        raise ValueError("empty question text")
    previous = [0] * (len(hypothesis) + 1)  # free start anywhere in the transcript
    for i, ref_char in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_char in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_char != hyp_char))
        previous = current
    return min(previous) / len(reference)  # free end anywhere


def match_question_onset(question_text: str, segments: Sequence[SpeechSegment]) -> OnsetMatch | None:
    """The segment that best holds the question (lowest substring CER, ties to the earliest), within ``ONSET_MAX_CER``.

    The onset is the segment start, so any speech merged before the question in
    the same segment (a lead-in such as "可以吗", an announcement) is cut from the
    evidence too.
    """
    scored = [(substring_cer(question_text, segment.text), segment) for segment in segments]
    if not scored:
        return None
    cer, best = min(scored, key=lambda pair: (pair[0], pair[1].start_s))
    if cer > ONSET_MAX_CER:
        return None
    return OnsetMatch(onset_s=best.start_s, segment=best, cer=cer)


def is_recall_phrased(question_text: str) -> bool:
    """Whether a question refers back in time, so its evidence cannot be cut from just before its onset."""
    return bool(_RECALL_PHRASING.search(question_text))


def find_question_onset(
    question: AnnotatedQuestion, samples: Sequence[float], sample_rate: int, segmenter: SpeechSegmenter
) -> tuple[OnsetMatch | None, list[SpeechSegment]]:
    """The spoken question's onset within the search window around ``question_time``, and every segment seen.

    ``samples`` is the whole clip's mono audio at ``sample_rate``.
    """
    search_start = max(0.0, question.question_time_s - ONSET_SEARCH_BEFORE_S)
    search_end = question.question_time_s + ONSET_SEARCH_AFTER_S
    window = samples[int(search_start * sample_rate) : int(search_end * sample_rate)]
    segments = segmenter(window, sample_rate, search_start)
    return match_question_onset(question.question_text, segments), segments


def onset_censored(match: OnsetMatch, search_start_s: float) -> bool:
    """Whether the matched speech was already under way when the search window opened (its true start is unknown)."""
    return search_start_s > 0 and match.segment.start_s - search_start_s < CENSORED_ONSET_S


def evidence_window(onset_s: float, *, lookback_s: int) -> tuple[float, int] | None:
    """Evidence ``(start_s, chunks)``: whole seconds ending exactly ``EVIDENCE_GUARD_S`` before the spoken question.

    Up to ``lookback_s`` chunks, fewer when the clip start is closer; ``None``
    when not even one second fits. The start is fractional so the last chunk
    (and its mid-second frame) sits as close to the onset as the guard allows.
    """
    end = onset_s - EVIDENCE_GUARD_S
    chunks = min(lookback_s, math.floor(end))
    if chunks < 1:
        return None
    return round(end - chunks, 3), chunks
