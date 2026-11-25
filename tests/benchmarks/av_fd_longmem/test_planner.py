# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chained planner invariants: exact gaps, a gapless stream, separated probes, no repeated filler.

External dependencies: none.
"""

from __future__ import annotations

import pytest

from benchmarks.av_fd_longmem.planner import PlannedProbe, plan_chained
from benchmarks.av_fd_longmem.schema import ClipRef, FillerClip, Item, ItemClips, QueryClip

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LAGS = {"a": 600, "b": 300, "c": 180, "d": 60, "e": 30, "f": 10}


def _item(name: str, evidence_chunks: int = 4) -> Item:
    return Item(
        item_id=name,
        source="test",
        lang="zh",
        clips=ItemClips(
            evidence=[ClipRef(path=f"evidence/{name}.mp4", chunks=evidence_chunks)],
            query=QueryClip(
                path=f"query/{name}.mp4", chunks=33, query_onset_chunk=0, query_end_ms=2400, answer_chunks=30
            ),
        ),
        query_text="q",
        query_text_original="q",
        answer="a",
        modality_declared="visual",
        provenance={},
    )


def _plan():
    probes = [PlannedProbe(_item(name), lag) for name, lag in LAGS.items()]
    filler = [
        FillerClip(filler_id=f"f{i}", path=f"filler/f{i}.mp4", chunks=97, source="test", domain=None, provenance={})
        for i in range(40)
    ]
    return plan_chained("pilot", probes, filler, condition={"window_mode": "basic"})


def test_every_probe_sits_at_its_exact_gap() -> None:
    plan = _plan()

    assert {probe.item_id: probe.gap_chunks for probe in plan.probes} == LAGS


def test_segments_tile_the_stream_without_holes() -> None:
    plan = _plan()

    cursor = 0
    for segment in plan.segments:
        assert segment.stream_chunk0 == cursor
        assert segment.play_chunks > 0
        cursor += segment.play_chunks
    assert cursor == plan.total_chunks


def test_no_evidence_or_query_comes_within_the_margin_of_another_probe() -> None:
    plan = _plan()
    spans = []
    for probe in plan.probes:
        spans.append((probe.evidence_chunks[0][0], probe.evidence_chunks[0][1] + 1))
        spans.append((probe.answer_window_chunks[0], probe.answer_window_chunks[1] + 1))
    spans.sort()

    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        assert start - prev_end >= 10


def test_the_stream_opens_with_filler_and_no_filler_clip_repeats() -> None:
    plan = _plan()
    filler = [segment for segment in plan.segments if segment.role == "filler"]

    assert plan.segments[0].role == "filler"
    assert plan.segments[0].play_chunks >= 30
    assert len({segment.clip for segment in filler}) == len(filler)


def test_an_exhausted_filler_pool_is_an_error() -> None:
    with pytest.raises(ValueError, match="filler pool exhausted"):
        plan_chained("pilot", [PlannedProbe(_item("a"), 600)], [], condition={})
