# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""M1 window characterisation: how much of the stream the Thinker's window holds, chunk by chunk.

A window session streams filler only (no probes) under one window mode. From
its tick trace this module reports:

- tokens per Stage-0 unit, split into audio and image tokens (units that left
  the window, read from ``ev`` drop rows);
- the eviction lag of every unit: chunks from the unit's own chunk to the chunk
  at which it left the window (same out rule as ``residency.py``);
- the resident span at every chunk ``t``: ``t`` minus the newest unit chunk
  that is out by ``t``. A probe with gap ``G`` is in-window for certain when
  ``G + evidence chunks`` stays below the minimum span, and out for certain
  when ``G`` exceeds the maximum span. In ``basic`` mode the span is a sawtooth
  between the low and high token marks;
- the Thinker prompt length per chunk (``w`` rows), the last chunk the Stage-0
  accepted, and when the model spoke (output transcript), for ``off`` mode
  exhaustion and degradation.

Design: plan §8 M1 (``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``).
External dependencies: none.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.av_fd_longmem.planner import _filler_segments
from benchmarks.av_fd_longmem.residency import (
    APPEND_MS,
    SINK_RESIDUE_TOKENS,
    UnitSpan,
    load_session_trace,
    unit_spans,
)
from benchmarks.av_fd_longmem.schema import FillerClip, SessionPlan

#: Chunks of filler per window session, by window mode.
WINDOW_PLAN_CHUNKS = {"basic": 900, "off": 600}


def plan_filler_only(
    plan_id: str, total_chunks: int, filler: Sequence[FillerClip], *, condition: dict[str, object]
) -> SessionPlan:
    """A probe-free session of ``total_chunks`` filler chunks, clips in pool order."""
    return SessionPlan(
        plan_id=plan_id,
        kind="window",
        seed=0,
        condition=condition,
        segments=_filler_segments((0, total_chunks), iter(filler)),
        probes=[],
        total_chunks=total_chunks,
    )


def window_plans(filler: Sequence[FillerClip]) -> list[SessionPlan]:
    """One greedy filler-only plan per window mode in ``WINDOW_PLAN_CHUNKS``, all on the same filler order."""
    return [
        plan_filler_only(f"m1-{mode}", chunks, filler, condition={"window_mode": mode, "temperature": 0.0})
        for mode, chunks in WINDOW_PLAN_CHUNKS.items()
    ]


@dataclass(frozen=True)
class UnitWindow:
    seq: int
    chunk: int  # stream chunk holding the unit's last client append
    tokens: int | None  # length when first dropped from; None if never dropped
    audio_tokens: int | None  # only for units that left the window
    image_tokens: int | None
    first_drop_chunk: int | None
    out_chunk: int | None

    @property
    def eviction_lag(self) -> int | None:
        return None if self.out_chunk is None else self.out_chunk - self.chunk


def _span_chunk(span: UnitSpan) -> int:
    return ((span.last_append - 1) * APPEND_MS) // 1000


def unit_windows(records: dict[str, list[list]]) -> list[UnitWindow]:
    """Per accepted unit with client input: its chunk, token split and when it left the window."""
    spans = unit_spans(records)
    chunk_of = {span.seq: _span_chunk(span) for span in spans}
    first_len: dict[int, int] = {}
    dropped: dict[int, int] = defaultdict(int)
    audio: dict[int, int] = defaultdict(int)
    image: dict[int, int] = defaultdict(int)
    first_drop: dict[int, int] = {}
    out_at: dict[int, int] = {}
    for row in sorted(records["ev"], key=lambda r: r[1]):
        at = chunk_of.get(row[4])
        if at is None:
            continue
        for unit_seq, unit_tokens, dropped_tokens, dropped_audio, dropped_image in row[7]:
            first_len.setdefault(unit_seq, unit_tokens)
            dropped[unit_seq] += dropped_tokens
            audio[unit_seq] += dropped_audio
            image[unit_seq] += dropped_image
            first_drop.setdefault(unit_seq, at)
            if unit_seq not in out_at and dropped[unit_seq] >= first_len[unit_seq] - SINK_RESIDUE_TOKENS:
                out_at[unit_seq] = at
    return [
        UnitWindow(
            seq=span.seq,
            chunk=chunk_of[span.seq],
            tokens=first_len.get(span.seq),
            audio_tokens=audio[span.seq] if span.seq in out_at else None,
            image_tokens=image[span.seq] if span.seq in out_at else None,
            first_drop_chunk=first_drop.get(span.seq),
            out_chunk=out_at.get(span.seq),
        )
        for span in spans
    ]


def resident_span(units: Sequence[UnitWindow], total_chunks: int) -> list[int | None]:
    """Per chunk t: t minus the newest unit chunk out by t; ``None`` before the first eviction."""
    out_by: dict[int, int] = {}
    for unit in units:
        if unit.out_chunk is not None:
            out_by[unit.out_chunk] = max(out_by.get(unit.out_chunk, -1), unit.chunk)
    spans: list[int | None] = []
    newest_out: int | None = None
    for t in range(total_chunks):
        if t in out_by:
            newest_out = max(newest_out if newest_out is not None else -1, out_by[t])
        spans.append(None if newest_out is None else t - newest_out)
    return spans


def prompt_lengths(records: dict[str, list[list]], units: Sequence[UnitWindow]) -> list[tuple[int, int]]:
    """``(chunk, Thinker prompt length)`` per accepted append, from ``w`` rows (max over prefill chunks)."""
    chunk_of = {unit.seq: unit.chunk for unit in units}
    longest: dict[int, int] = {}
    for row in records["w"]:
        seq, prompt_len = row[4], row[5]
        longest[seq] = max(longest.get(seq, 0), prompt_len)
    return sorted((chunk_of[seq], length) for seq, length in longest.items() if seq in chunk_of)


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": ordered[int(0.05 * (len(ordered) - 1))],
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
    }


def speech_chunks(transcript: dict[str, object]) -> list[tuple[float, float, str]]:
    """``(start_s, end_s, text)`` of every output sentence in a ``wav_transcript.json``."""
    rows = []
    for chunk in transcript.get("chunks", []) or []:
        start, end = chunk["timestamp"]
        rows.append((float(start), float(end), str(chunk["text"])))
    return rows


def _only_trace_session(trace_dir: Path) -> str:
    sessions = set()
    for path in trace_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else None
            if row and row[0] in {"w", "acc"}:
                sessions.add(row[2])
    if len(sessions) != 1:
        raise ValueError(f"{trace_dir}: session id unknown and the trace holds {len(sessions)} sessions")
    return sessions.pop()


def window_report(session_dir: Path, trace_dir: Path) -> dict[str, object]:
    """M1 measurements for one window session (see the module docstring).

    A failed session (e.g. ``off`` mode reaching ``max_model_len``) records no
    session id; it is then taken from the trace, which must hold only that session.
    """
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    total = int(session["total_chunks"])
    session_id = session["session_id"] or _only_trace_session(trace_dir)
    records = load_session_trace(trace_dir, session_id)
    units = unit_windows(records)
    out_units = [u for u in units if u.out_chunk is not None]
    spans = resident_span(units, total)
    steady = [s for s in spans if s is not None]
    prompts = prompt_lengths(records, units)
    transcript_path = session_dir / "wav_transcript.json"
    speech = speech_chunks(json.loads(transcript_path.read_text(encoding="utf-8"))) if transcript_path.exists() else []
    compactions = defaultdict(int)
    for row in records["ev"]:
        compactions[row[5]] += 1
    return {
        "plan_id": session["plan_id"],
        "condition": session["condition"],
        "session_id": session_id,
        "success": session["success"],
        "error": session["error"],
        "total_chunks": total,
        "units_with_input": len(units),
        "last_unit_chunk": max((u.chunk for u in units), default=None),
        "ev_records_by_path": dict(compactions),
        "units_out": len(out_units),
        "unit_tokens": _stats([u.tokens for u in out_units if u.tokens is not None]),
        "unit_audio_tokens": _stats([u.audio_tokens for u in out_units if u.audio_tokens is not None]),
        "unit_image_tokens": _stats([u.image_tokens for u in out_units if u.image_tokens is not None]),
        "eviction_lag_chunks": _stats([u.eviction_lag for u in out_units if u.eviction_lag is not None]),
        "first_eviction_chunk": min((u.out_chunk for u in out_units if u.out_chunk is not None), default=None),
        "resident_span_chunks": _stats(steady),
        "prompt_len": _stats([length for _, length in prompts]),
        "prompt_len_last": prompts[-1] if prompts else None,
        "speech_sentences": len(speech),
        "speech_by_minute": _per_minute(speech, total),
        "per_chunk": {
            "resident_span": spans,
            "prompt_len": prompts,
            "units": [asdict(u) for u in units],
            "speech": speech,
        },
    }


def _per_minute(speech: Sequence[tuple[float, float, str]], total_chunks: int) -> list[int]:
    counts = [0] * ((total_chunks + 59) // 60)
    for start, _end, _text in speech:
        minute = int(start // 60)
        if 0 <= minute < len(counts):
            counts[minute] += 1
    return counts


def summary_markdown(report: dict[str, object]) -> str:
    def fmt(stats: object) -> str:
        if not isinstance(stats, dict):
            return "—"
        spread = f"min {stats['min']}, p05 {stats['p05']}, p95 {stats['p95']}, max {stats['max']}; n={stats['n']}"
        return f"median {stats['median']} ({spread})"

    lines = [
        f"# {report['plan_id']}  ({report['condition']})",
        "",
        f"- session success: {report['success']} {report['error'] or ''}",
        f"- chunks streamed: {report['total_chunks']}; last accepted unit at chunk {report['last_unit_chunk']}",
        f"- `ev` records: {report['ev_records_by_path']}; units out of the window: {report['units_out']}",
        f"- tokens per unit (evicted units): {fmt(report['unit_tokens'])}",
        f"  - audio: {fmt(report['unit_audio_tokens'])}",
        f"  - image: {fmt(report['unit_image_tokens'])}",
        f"- first eviction at chunk {report['first_eviction_chunk']}",
        f"- eviction lag (chunks): {fmt(report['eviction_lag_chunks'])}",
        f"- resident span after the first eviction (chunks): {fmt(report['resident_span_chunks'])}",
        f"- Thinker prompt length: {fmt(report['prompt_len'])}; last (chunk, length): {report['prompt_len_last']}",
        f"- output sentences: {report['speech_sentences']}; per minute: {report['speech_by_minute']}",
    ]
    return "\n".join(lines) + "\n"
