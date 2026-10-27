# SPDX-License-Identifier: Apache-2.0
"""Stitch short Omni-DuplexEval samples into one long duplex input stream.

Purpose: prototype for long-context full-duplex profiling (branch
long-context-prototype). Each sample becomes one segment of the stream,
padded with silence to a whole number of 1 s units, so segment boundaries
coincide with tick boundaries. Samples are cycled until the stream reaches a
target duration; a sample can therefore appear at several depths.

External dependencies: none at import time. ``stitch_prepared`` consumes
``PreparedSample`` objects from ``vllm_omni.benchmarks.duplex``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

PCM16_SAMPLE_RATE = 16000
PCM16_BYTES_PER_SAMPLE = 2
UNIT_S = 1


@dataclass(frozen=True)
class Segment:
    """One sample's place in the stitched stream (stream media time, seconds)."""

    index: int
    sample_index: int
    sample_id: str
    split: str
    pass_index: int
    offset_s: int
    duration_s: int

    @property
    def end_s(self) -> int:
        return self.offset_s + self.duration_s

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_duration_s(audio_bytes: int, video_duration_s: float | None) -> int:
    """Whole units covering both the sample's audio and its video."""
    audio_s = audio_bytes / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    longest = max(audio_s, float(video_duration_s or 0.0))
    return max(UNIT_S, math.ceil(longest / UNIT_S - 1e-9) * UNIT_S)


def plan_segments(
    sample_ids: Sequence[str],
    splits: Sequence[str],
    durations_s: Sequence[int],
    *,
    target_s: float,
    start_index: int = 0,
) -> list[Segment]:
    """Cycle through samples from ``start_index`` until the stream lasts ``target_s``."""
    if not sample_ids:
        raise ValueError("no samples to stitch")
    if not len(sample_ids) == len(splits) == len(durations_s):
        raise ValueError("sample_ids, splits and durations_s must have the same length")
    if target_s <= 0:
        raise ValueError("target_s must be positive")
    count = len(sample_ids)
    segments: list[Segment] = []
    offset = 0
    position = 0
    while offset < target_s:
        sample_index = (start_index + position) % count
        segments.append(
            Segment(
                index=position,
                sample_index=sample_index,
                sample_id=sample_ids[sample_index],
                split=splits[sample_index],
                pass_index=position // count,
                offset_s=offset,
                duration_s=int(durations_s[sample_index]),
            )
        )
        offset += int(durations_s[sample_index])
        position += 1
    return segments


def stitch_pcm_and_frames(
    pcms: Sequence[bytes],
    frames: Sequence[Sequence[tuple[float, bytes]]],
    segments: Sequence[Segment],
    *,
    with_frames: bool,
) -> tuple[bytes, tuple[tuple[float, bytes], ...]]:
    """Concatenate per-sample audio (silence-padded) and stream-time-shifted frames."""
    audio = bytearray()
    shifted: list[tuple[float, bytes]] = []
    for segment in segments:
        pcm = pcms[segment.sample_index]
        size = segment.duration_s * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
        audio += pcm[:size]
        audio += bytes(size - min(size, len(pcm)))
        if with_frames:
            shifted.extend(
                (segment.offset_s + timestamp, frame)
                for timestamp, frame in frames[segment.sample_index]
                if timestamp < segment.duration_s
            )
    return bytes(audio), tuple(shifted)


def stitch_prepared(prepared: Sequence[Any], segments: Sequence[Segment], *, with_frames: bool) -> Any:
    """Build one ``PreparedSample`` for the whole stitched stream."""
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample

    pcm, frames = stitch_pcm_and_frames(
        [item.pcm for item in prepared], [item.frames for item in prepared], segments, with_frames=with_frames
    )
    reference = prepared[0]
    return PreparedSample(
        pcm=pcm,
        frames=frames,
        duration=float(segments[-1].end_s),
        ref_audio=reference.ref_audio,
        ref_audio_sha256=reference.ref_audio_sha256,
    )


def split_sentences(
    timed_sentences: Sequence[dict[str, Any]], segments: Sequence[Segment]
) -> dict[int, list[dict[str, Any]]]:
    """Assign each timed sentence to the segment its start falls in, on the segment's own clock."""
    by_segment: dict[int, list[dict[str, Any]]] = {segment.index: [] for segment in segments}
    for sentence in timed_sentences:
        start = float(sentence.get("start", 0.0))
        for segment in segments:
            if segment.offset_s <= start < segment.end_s:
                by_segment[segment.index].append(
                    {
                        **sentence,
                        "start": start - segment.offset_s,
                        "end": float(sentence.get("end", start)) - segment.offset_s,
                    }
                )
                break
    return by_segment
