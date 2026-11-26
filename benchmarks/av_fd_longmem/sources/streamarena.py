# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StreamArena adapter: HR annotation rows, rule-based exclusions and evidence windows.

Annotations are read from the pinned revision's ``question{,.en}.jsonl`` and
``video_meta.jsonl``, joined on ``(video_id, qid)``. Only ``HR`` (historical
recall) rows become items. The rule-based exclusions of plan §1a step 4 are
applied here; the LLM rules (residual relative-window phrasing, non-standalone
questions) are not implemented yet.

Evidence windows: each ``evidence_ts_sec`` point becomes ``[ts - 5, ts + 5]``,
extended over overlapping subtitle cues, kept within ``ts ± 15`` (30 s),
merged, ended no later than ``ask_sec`` and snapped outward to whole seconds.

Design: plan §1a (``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``).
External dependencies: none.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REVISION = "2960c360deba57b717a17ab66a3ae80f5358a6a3"
EVIDENCE_HALF_WIDTH_S = 5.0
EVIDENCE_MAX_HALF_WIDTH_S = 15.0
#: Video language -> query language. ``zh+en`` videos are English-query items (decision 2026-11-25), but they
#: form their own stratum (decision 2026-11-26): their audio is mostly Chinese, so an English query over them is
#: a cross-language probe, not an English one.
ITEM_LANGS = {"zh": "zh", "en": "en", "zh+en": "en"}

_SO_FAR = re.compile(
    r"到目前为止|到现在为止|目前为止|截至目前|截止目前|截至现在|截止到?现在|迄今|至今"
    r"|\bso far\b|\buntil now\b|\bup to now\b|\bto date\b|\bby now\b",
    re.IGNORECASE,
)
_COUNT = r"[\d一二三四五六七八九十两半几]+"
_COUNT_EN = r"(\d+|a few|few|several|a couple of|one|two|three|four|five|ten|fifteen|twenty|thirty)"
_RELATIVE_WINDOW = re.compile(
    rf"(过去|最近|之前的?|前面的?)\s*(的)?\s*{_COUNT}\s*(个)?\s*(分钟|小时|秒)"
    rf"|{_COUNT}\s*(个)?\s*(分钟|小时|秒钟?)(之?前|以前)"
    rf"|\b(past|last|previous)\s+{_COUNT_EN}\s+(minutes?|seconds?|hours?)\b"
    rf"|\b{_COUNT_EN}\s+(minutes?|seconds?|hours?)\s+ago\b",
    re.IGNORECASE,
)
#: Recency wording that turns false once a long lag is inserted; such queries need a lag-neutral rewrite.
_RECENCY = re.compile(
    r"刚才|刚刚|方才|\bjust now\b|\ba moment ago\b|\bjust (saw|seen|showed|shown|mentioned)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class RecallRow:
    video_id: str
    qid: int
    lang: str  # video language from video_meta: zh | en | zh+en | other
    domain: str  # L1
    question_zh: str
    question_en: str
    answer_zh: str
    answer_en: str
    ask_sec: int
    evidence_ts_sec: list[int]
    evidence_hint: list[str]
    horizon_layer: str | None

    @property
    def item_lang(self) -> str | None:
        """Language the query is spoken in; ``None`` when the video's language is not zh or en alone."""
        return ITEM_LANGS.get(self.lang)

    @property
    def stratum(self) -> str | None:
        """Reporting and sampling stratum: the video language (``zh``, ``en`` or ``zh+en``) of a usable item."""
        return self.lang if self.item_lang is not None else None

    @property
    def question(self) -> str:
        return self.question_en if self.item_lang == "en" else self.question_zh


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recall_rows(annotations: Path) -> list[RecallRow]:
    """Every HR row, joined with its English translation and its video's language and ``L1``."""
    meta = {row["video_id"]: row for row in _jsonl(annotations / "video_meta.jsonl")}
    english = {(row["video_id"], row["qid"]): row for row in _jsonl(annotations / "question.en.jsonl")}
    rows = []
    for row in _jsonl(annotations / "question.jsonl"):
        if row["qtype"] != "HR":
            continue
        video = meta[row["video_id"]]
        en = english[(row["video_id"], row["qid"])]
        rows.append(
            RecallRow(
                video_id=str(row["video_id"]),
                qid=int(row["qid"]),
                lang=str(video["language"]),
                domain=str(video["L1"]),
                question_zh=str(row["question"]),
                question_en=str(en["question"]),
                answer_zh=str(row["answer"]),
                answer_en=str(en["answer"]),
                ask_sec=int(row["ask_sec"]),
                evidence_ts_sec=[int(t) for t in row["evidence_ts_sec"] or []],
                evidence_hint=[str(h) for h in row["evidence_hint"] or []],
                horizon_layer=row.get("horizon_layer"),
            )
        )
    return rows


def exclusion_reasons(row: RecallRow) -> list[str]:
    """Rule-based exclusion reasons (empty when the row survives); every rule is checked."""
    reasons = []
    if row.item_lang is None:
        reasons.append(f"lang_{row.lang}")
    if not row.evidence_ts_sec:
        reasons.append("no_evidence")
    elif any(t > row.ask_sec for t in row.evidence_ts_sec):
        reasons.append("evidence_after_ask")
    if _SO_FAR.search(row.question_zh) or _SO_FAR.search(row.question_en):
        reasons.append("so_far_phrasing")
    if _RELATIVE_WINDOW.search(row.question_zh) or _RELATIVE_WINDOW.search(row.question_en):
        reasons.append("relative_window_phrasing")
    return reasons


_CUE_TIME = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})\.(\d{3})")


def vtt_cues(paths: Sequence[Path]) -> list[tuple[float, float]]:
    """Sorted ``(start_s, end_s)`` of every cue in the given WebVTT files (cue text is ignored)."""
    cues = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _CUE_TIME.match(line.strip())
            if match:
                h0, m0, s0, ms0, h1, m1, s1, ms1 = (int(g) for g in match.groups())
                cues.append((h0 * 3600 + m0 * 60 + s0 + ms0 / 1000, h1 * 3600 + m1 * 60 + s1 + ms1 / 1000))
    return sorted(set(cues))


FILLER_CLEARANCE_S = 60
FILLER_CLIP_CHUNKS = 300
FILLER_MIN_CHUNKS = 60


def annotated_times(annotations: Path, video_id: str) -> list[int]:
    """Every ask, reference and evidence time annotated for a video, over all question types."""
    times: list[int] = []
    for row in _jsonl(annotations / "question.jsonl"):
        if row["video_id"] != video_id:
            continue
        times += [int(row[key]) for key in ("ask_sec", "ref_sec") if row.get(key) is not None]
        times += [int(t) for t in row.get("evidence_ts_sec") or []]
    return sorted(times)


def filler_runs(duration_s: float, blocked_times: Sequence[int]) -> list[tuple[int, int]]:
    """``(start_s, chunks)`` filler clips at least ``FILLER_CLEARANCE_S`` from every blocked time.

    Each clear run is cut into ``FILLER_CLIP_CHUNKS`` clips; a shorter last clip
    is kept when it has at least ``FILLER_MIN_CHUNKS``.
    """
    end_s = math.floor(duration_s)
    edges = [(t - FILLER_CLEARANCE_S, t + FILLER_CLEARANCE_S) for t in sorted(blocked_times)]
    clips: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in [*edges, (end_s, end_s)]:
        start, stop = cursor, min(lo, end_s)
        while stop - start >= FILLER_MIN_CHUNKS:
            chunks = min(FILLER_CLIP_CHUNKS, stop - start)
            clips.append((start, chunks))
            start += chunks
        cursor = max(cursor, hi)
    return clips


def needs_lag_neutral_rewrite(row: RecallRow) -> bool:
    """Whether the spoken query's wording ("刚才", "just now") assumes a short lag."""
    return bool(_RECENCY.search(row.question))


_ZH_RECENCY = re.compile(r"刚才|刚刚|方才")
_EN_JUST_VERB = re.compile(r"\bjust (saw|seen|showed|shown|mentioned)\b", re.IGNORECASE)
_EN_JUST_NOW = re.compile(r"\b(just now|a moment ago)\b", re.IGNORECASE)


def lag_neutral(question: str) -> str:
    """The question with its recency words made true at any lag: 刚才/刚刚/方才 -> 之前, "just now" -> "earlier".

    "just <verb>" becomes "<verb> earlier". Everything else is kept verbatim.
    """
    text = _ZH_RECENCY.sub("之前", question)
    text = _EN_JUST_VERB.sub(lambda m: f"{m.group(1)} earlier", text)
    return _EN_JUST_NOW.sub("earlier", text)


def evidence_windows(
    points: Sequence[int], ask_sec: int, cues: Sequence[tuple[float, float]] = ()
) -> list[tuple[int, int]]:
    """Merged whole-second evidence windows ``[start, end)`` for evidence points before ``ask_sec``."""
    spans: list[tuple[float, float]] = []
    for ts in sorted(points):
        lo, hi = ts - EVIDENCE_HALF_WIDTH_S, ts + EVIDENCE_HALF_WIDTH_S
        for cue_lo, cue_hi in cues:
            if cue_lo < hi and lo < cue_hi:
                lo, hi = min(lo, cue_lo), max(hi, cue_hi)
        lo = max(lo, ts - EVIDENCE_MAX_HALF_WIDTH_S, 0.0)
        hi = min(hi, ts + EVIDENCE_MAX_HALF_WIDTH_S, float(ask_sec))
        if hi > lo:
            spans.append((lo, hi))
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        start, end = math.floor(lo), math.ceil(hi)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
