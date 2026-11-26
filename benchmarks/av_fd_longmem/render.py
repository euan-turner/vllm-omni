# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render a session plan into the duplex client's stream units.

Each plan segment is decoded once, from its start, for ``play_chunks`` seconds,
and yields ``(pcm16 1 s, jpeg)`` per chunk: the unit contract of
``RealtimeDuplexClient.stream_av_units``. ``mute`` zeroes the audio; ``freeze``
repeats the segment's first frame. Frames are re-encoded with the PIL JPEG
quality that ``vllm_omni.benchmarks.duplex.omni_duplex_eval_media.extract_jpeg``
maps ffmpeg ``-q:v 5`` to.

``render_preview`` writes the same units to an mp4 (1 fps, 16 kHz mono) for
checking a plan by eye and ear.

External dependencies: PyAV, Pillow, numpy; ffmpeg on PATH for ``render_preview``.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from benchmarks.av_fd_longmem.schema import Segment, SessionPlan

SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE


def ffmpeg_q_to_pil_quality(quality: int) -> int:
    """Map ffmpeg ``-q:v`` (1..31, lower = better) to PIL JPEG quality (1..100, higher = better)."""
    ffmpeg_q = max(1, min(31, int(quality)))
    return max(1, min(100, round(95 - (ffmpeg_q - 1) * (93 / 30))))


JPEG_QUALITY = ffmpeg_q_to_pil_quality(5)

Unit = tuple[bytes, bytes]


def _decode_audio(container: av.container.InputContainer, chunks: int) -> np.ndarray:
    """Mono int16 samples on the clip's timeline (encoder priming removed), exactly ``chunks`` seconds."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    total = chunks * CHUNK_SAMPLES
    out = np.zeros(total, dtype=np.int16)
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            samples = resampled.to_ndarray().reshape(-1)
            start = round(float(resampled.pts * resampled.time_base) * SAMPLE_RATE) if resampled.pts is not None else 0
            lo, hi = max(0, start), min(total, start + len(samples))
            if lo < hi:
                out[lo:hi] = samples[lo - start : hi - start]
    return out


def _decode_frames(container: av.container.InputContainer, chunks: int) -> list[bytes]:
    frames: list[bytes] = []
    for frame in container.decode(video=0):
        buffer = io.BytesIO()
        frame.to_image().save(buffer, format="JPEG", quality=JPEG_QUALITY)
        frames.append(buffer.getvalue())
        if len(frames) == chunks:
            break
    if len(frames) < chunks:
        raise ValueError(f"clip has {len(frames)} frames, segment plays {chunks}")
    return frames


def segment_units(segment: Segment, root: Path) -> Iterator[Unit]:
    path = root / segment.clip
    with av.open(str(path)) as container:
        audio = _decode_audio(container, segment.play_chunks)
    with av.open(str(path)) as container:
        frames = _decode_frames(container, segment.play_chunks)
    if segment.audio == "mute":
        audio = np.zeros_like(audio)
    if segment.video == "freeze":
        frames = [frames[0]] * len(frames)
    for k in range(segment.play_chunks):
        yield audio[k * CHUNK_SAMPLES : (k + 1) * CHUNK_SAMPLES].tobytes(), frames[k]


def render_units(plan: SessionPlan, root: Path) -> Iterator[Unit]:
    """All units of ``plan`` in stream order; segments must tile ``[0, total_chunks)`` exactly."""
    expected = 0
    for segment in plan.segments:
        if segment.stream_chunk0 != expected:
            raise ValueError(f"segment {segment.clip} starts at {segment.stream_chunk0}, expected {expected}")
        yield from segment_units(segment, root)
        expected += segment.play_chunks
    if expected != plan.total_chunks:
        raise ValueError(f"segments cover {expected} chunks, plan has {plan.total_chunks}")


_PREVIEW_FIT = "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2"


def render_preview(plan: SessionPlan, root: Path, out: Path) -> Path:
    """``plan``'s stream units as an mp4: frame k shown for second k, the PCM exactly as sent.

    Frames are letterboxed to 640x640, so segments of different shapes share one video stream.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pcm") as pcm:
        frames = []
        for chunk_pcm, jpeg in render_units(plan, root):
            pcm.write(chunk_pcm)
            frames.append(jpeg)
        pcm.flush()
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "image2pipe", "-framerate", "1", "-c:v", "mjpeg", "-i", "-",
             "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", pcm.name,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", _PREVIEW_FIT, "-r", "1",
             "-c:a", "aac", "-b:a", "96k", "-shortest", str(out)],
            input=b"".join(frames), check=True,
        )  # fmt: skip
    return out
