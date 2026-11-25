# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session planner: lay out (evidence, query) probes and tile the rest with filler.

A chained plan holds several probes in one stream. Probes are placed longest
lag first, each at the earliest stream chunk where neither its evidence nor its
query clip (query plus silent answer tail) comes within ``margin`` chunks of
anything already placed. Every remaining span is filler: whole filler clips,
then a prefix of one more, so no clip is ever seeked and no filler clip plays
twice in a session.

External dependencies: none.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from benchmarks.av_fd_longmem.schema import FillerClip, Item, Probe, Segment, SessionPlan

FILLER_PREFIX_CHUNKS = 30
PROBE_MARGIN_CHUNKS = 10


@dataclass(frozen=True)
class PlannedProbe:
    item: Item
    gap_chunks: int


@dataclass(frozen=True)
class _Placed:
    item: Item
    evidence: tuple[int, int]  # [start, end) in stream chunks
    query: tuple[int, int]


def _evidence_chunks(item: Item) -> int:
    if len(item.clips.evidence) != 1:
        raise ValueError(f"{item.item_id}: the pilot planner takes exactly one evidence clip")
    return item.clips.evidence[0].chunks


def _clear(span: tuple[int, int], taken: Sequence[tuple[int, int]], margin: int) -> bool:
    return all(span[1] + margin <= lo or hi + margin <= span[0] for lo, hi in taken)


def _place(probes: Sequence[PlannedProbe], *, start: int, margin: int) -> list[_Placed]:
    placed: list[_Placed] = []
    taken: list[tuple[int, int]] = []
    for probe in sorted(probes, key=lambda p: (-p.gap_chunks, p.item.item_id)):
        n_evidence = _evidence_chunks(probe.item)
        n_query = probe.item.clips.query.chunks
        e0 = start
        while True:
            evidence = (e0, e0 + n_evidence)
            q0 = evidence[1] + probe.gap_chunks
            query = (q0, q0 + n_query)
            if _clear(evidence, taken, margin) and _clear(query, taken, margin):
                break
            e0 += 1
        placed.append(_Placed(probe.item, evidence, query))
        taken += [evidence, query]
    return placed


def _filler_segments(gap: tuple[int, int], pool: Iterator[FillerClip]) -> list[Segment]:
    segments = []
    cursor, end = gap
    while cursor < end:
        clip = next(pool, None)
        if clip is None:
            raise ValueError("filler pool exhausted")
        play = min(clip.chunks, end - cursor)
        segments.append(Segment(role="filler", clip=clip.path, play_chunks=play, stream_chunk0=cursor))
        cursor += play
    return segments


def plan_chained(
    plan_id: str,
    probes: Sequence[PlannedProbe],
    filler: Sequence[FillerClip],
    *,
    condition: dict[str, object],
    seed: int = 0,
    margin: int = PROBE_MARGIN_CHUNKS,
    filler_prefix: int = FILLER_PREFIX_CHUNKS,
) -> SessionPlan:
    """One stream holding every probe at its gap, with filler everywhere else."""
    placed = _place(probes, start=filler_prefix, margin=margin)
    spans = sorted(
        [(p.evidence, "evidence", p.item.clips.evidence[0].path) for p in placed]
        + [(p.query, "query", p.item.clips.query.path) for p in placed]
    )
    pool = iter(filler)
    segments: list[Segment] = []
    cursor = 0
    for (lo, hi), role, clip in spans:
        segments += _filler_segments((cursor, lo), pool)
        segments.append(Segment(role=role, clip=clip, play_chunks=hi - lo, stream_chunk0=lo))
        cursor = hi
    total = cursor
    plan_probes = [
        Probe(
            item_id=p.item.item_id,
            evidence_chunks=[[p.evidence[0], p.evidence[1] - 1]],
            query_onset_chunk=p.query[0] + p.item.clips.query.query_onset_chunk,
            query_end_ms=p.query[0] * 1000 + p.item.clips.query.query_end_ms,
            answer_window_chunks=[p.query[0], p.query[1] - 1],
        )
        for p in sorted(placed, key=lambda p: p.query[0])
    ]
    return SessionPlan(
        plan_id=plan_id,
        kind="chained",
        seed=seed,
        condition=condition,
        segments=segments,
        probes=plan_probes,
        total_chunks=total,
    )
