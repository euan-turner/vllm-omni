# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cut and encode av_fd_longmem clips with ffmpeg.

Every clip is a whole number of seconds: H.264 at 1 fps with every frame a
keyframe (``-g 1``), longest side 640, and AAC-LC 16 kHz mono audio. Clip frame
k is the source frame at ``start + k + 0.5`` s, the mid-second sampling used by
``vllm_omni/benchmarks/omniinteract.py`` ``prepare_media``. Audio is either the
source span, or silence with an overlay WAV laid in (query clips).

External dependencies: ffmpeg and ffprobe on PATH (libx264, aac).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Pinned encode settings, written into every record's provenance.
ENCODE_SETTINGS: dict[str, object] = {
    "video": {
        "codec": "libx264",
        "fps": 1,
        "gop": 1,
        "crf": 18,
        "preset": "medium",
        "pix_fmt": "yuv420p",
        "fit": 640,
        "frame_offset_s": 0.5,
    },
    "audio": {"codec": "aac", "sample_rate": 16000, "channels": 1, "bitrate": "96k"},
}
SAMPLE_RATE = 16000
_FIT = 640
_TIMEOUT_S = 600


@dataclass(frozen=True)
class ClipProbe:
    duration_s: float
    video_frames: int
    video_width: int
    video_height: int
    audio_sample_rate: int
    audio_channels: int


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()}")
    return result.stdout


def _video_args(chunks: int) -> list[str]:
    scale = f"scale={_FIT}:{_FIT}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    return [
        "-vf", f"fps=1,{scale}",
        "-frames:v", str(chunks),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "1", "-pix_fmt", "yuv420p",
    ]  # fmt: skip


def _audio_args(chunks: int) -> list[str]:
    return ["-af", f"apad=whole_dur={chunks}", "-t", str(chunks), "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "aac", "-b:a", "96k"]  # fmt: skip


def cut_source_clip(source: Path, start_s: float, chunks: int, out: Path) -> Path:
    """``chunks`` seconds of ``source`` from ``start_s`` (may be fractional), with its own audio."""
    if start_s < 0 or chunks <= 0:
        raise ValueError(f"invalid span start={start_s} chunks={chunks}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{start_s + 0.5:.3f}", "-t", str(chunks), "-i", str(source),
            "-ss", f"{start_s:.3f}", "-t", str(chunks), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0",
            *_video_args(chunks), *_audio_args(chunks),
            str(out),
        ]
    )  # fmt: skip
    return out


def cut_query_clip(backdrop: Path, start_s: int, chunks: int, speech_wav: Path, onset_chunk: int, out: Path) -> Path:
    """Backdrop video with the source audio muted and ``speech_wav`` laid in at ``onset_chunk``."""
    if onset_chunk < 0 or onset_chunk >= chunks:
        raise ValueError(f"onset_chunk {onset_chunk} outside a {chunks}-chunk clip")
    out.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = onset_chunk * 1000
    _run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{start_s + 0.5:.3f}", "-t", str(chunks), "-i", str(backdrop),
            "-i", str(speech_wav),
            "-map", "0:v:0",
            *_video_args(chunks),
            "-filter_complex",
            f"[1:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono,adelay={delay_ms},apad=whole_dur={chunks}[a]",
            "-map", "[a]", "-t", str(chunks), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "aac", "-b:a", "96k",
            str(out),
        ]
    )  # fmt: skip
    return out


def probe_clip(path: Path) -> ClipProbe:
    """Duration, decoded video frame count and audio format of a clip."""
    info = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "format=duration:stream=codec_type,nb_read_frames,width,height,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ]
        )  # fmt: skip
    )
    streams = {stream["codec_type"]: stream for stream in info["streams"]}
    return ClipProbe(
        duration_s=float(info["format"]["duration"]),
        video_frames=int(streams["video"]["nb_read_frames"]),
        video_width=int(streams["video"]["width"]),
        video_height=int(streams["video"]["height"]),
        audio_sample_rate=int(streams["audio"]["sample_rate"]),
        audio_channels=int(streams["audio"]["channels"]),
    )


def media_duration_s(path: Path) -> float:
    """Container duration of any media file, without decoding it."""
    info = json.loads(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]))
    return float(info["format"]["duration"])
