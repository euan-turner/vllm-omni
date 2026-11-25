# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Clips cut from a synthetic source render into exact one-second units.

External dependencies: ffmpeg/ffprobe (libx264, aac), PyAV, numpy.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.av_fd_longmem.materialise import cut_query_clip, cut_source_clip, probe_clip
from benchmarks.av_fd_longmem.render import render_units
from benchmarks.av_fd_longmem.schema import Segment, SessionPlan

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required"),
]


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """20 s of testsrc video at 25 fps with a 440 Hz tone, 44.1 kHz stereo."""
    path = tmp_path_factory.mktemp("src") / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25:duration=20",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=20",
         "-ac", "2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="module")
def speech(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tts") / "query.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=2.5",
         "-ar", "24000", str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_a_source_clip_is_whole_seconds_at_one_frame_per_chunk(source: Path, tmp_path: Path) -> None:
    probe = probe_clip(cut_source_clip(source, 3, 7, tmp_path / "evidence.mp4"))

    assert probe.duration_s == pytest.approx(7.0, abs=0.05)
    assert probe.video_frames == 7
    assert (probe.audio_sample_rate, probe.audio_channels) == (16000, 1)
    assert max(probe.video_width, probe.video_height) == 640


def test_a_query_clip_carries_speech_at_the_onset_and_silence_after(source: Path, speech: Path, tmp_path: Path) -> None:
    clip = cut_query_clip(source, 5, 12, speech, 2, tmp_path / "query.mp4")
    plan = SessionPlan("p", "single", 0, {}, [Segment("query", clip.name, 12, 0)], [], 12)

    levels = [int(np.abs(np.frombuffer(pcm, np.int16)).mean()) for pcm, _ in render_units(plan, tmp_path)]

    assert levels[0] == levels[1] == 0
    assert all(level > 500 for level in levels[2:4])
    assert all(level == 0 for level in levels[6:])


def test_a_plan_renders_one_second_of_audio_and_one_frame_per_chunk(source: Path, tmp_path: Path) -> None:
    cut_source_clip(source, 0, 6, tmp_path / "filler.mp4")
    cut_source_clip(source, 10, 4, tmp_path / "evidence.mp4")
    plan = SessionPlan(
        "p",
        "single",
        0,
        {},
        [
            Segment("filler", "filler.mp4", 5, 0),
            Segment("evidence", "evidence.mp4", 4, 5, audio="mute", video="freeze"),
        ],
        [],
        9,
    )

    units = list(render_units(plan, tmp_path))

    assert len(units) == 9
    assert {len(pcm) for pcm, _ in units} == {32000}
    assert all(frame[:2] == b"\xff\xd8" for _, frame in units)
    assert int(np.abs(np.frombuffer(units[0][0], np.int16)).mean()) > 500
    assert all(not any(pcm) for pcm, _ in units[5:])
    assert len({frame for _, frame in units[5:]}) == 1


def test_segments_that_leave_a_hole_are_refused(source: Path, tmp_path: Path) -> None:
    cut_source_clip(source, 0, 3, tmp_path / "a.mp4")
    plan = SessionPlan("p", "single", 0, {}, [Segment("filler", "a.mp4", 3, 1)], [], 4)

    with pytest.raises(ValueError, match="expected 0"):
        list(render_units(plan, tmp_path))
