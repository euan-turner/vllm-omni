# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spoken queries: Qwen3-TTS through a vLLM-Omni ``/v1/audio/speech`` server, with an ASR round trip.

Each query is synthesised once with a fixed built-in voice and cached as a WAV
next to a JSON sidecar (model, voice, language, text, round-trip transcript and
CER). A query whose round-trip CER exceeds ``MAX_ROUND_TRIP_CER`` is rejected:
the model would be answering a different question.

External dependencies: httpx; a running Qwen3-TTS CustomVoice server; a
transcriber callable (Paraformer-zh in production).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from benchmarks.av_fd_longmem.sources.omniinteract import question_cer

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TTS_VOICES = {"zh": ("vivian", "Chinese"), "en": ("ryan", "English")}
MAX_ROUND_TRIP_CER = 0.15

#: WAV path -> transcript.
Transcriber = Callable[[Path], str]


@dataclass(frozen=True)
class SpokenQuery:
    wav: str
    text: str
    model: str
    voice: str
    language: str
    round_trip_text: str
    round_trip_cer: float


def _cache_stem(text: str, voice: str, language: str) -> str:
    return hashlib.sha256(f"{TTS_MODEL}\0{voice}\0{language}\0{text}".encode()).hexdigest()[:16]


def synthesise(
    text: str,
    lang: str,
    *,
    base_url: str,
    cache_dir: Path,
    transcribe: Transcriber,
    timeout_s: float = 120.0,
) -> SpokenQuery:
    """The cached spoken query for ``text``, synthesising and checking it on a cache miss."""
    voice, language = TTS_VOICES[lang]
    stem = _cache_stem(text, voice, language)
    wav = cache_dir / f"{stem}.wav"
    sidecar = cache_dir / f"{stem}.json"
    if sidecar.exists() and wav.exists():
        return SpokenQuery(**json.loads(sidecar.read_text(encoding="utf-8")))
    payload = {"model": TTS_MODEL, "input": text, "voice": voice, "language": language, "response_format": "wav"}
    response = httpx.post(f"{base_url.rstrip('/')}/v1/audio/speech", json=payload, timeout=timeout_s)
    response.raise_for_status()
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(response.content)
    heard = transcribe(wav)
    query = SpokenQuery(
        wav=str(wav),
        text=text,
        model=TTS_MODEL,
        voice=voice,
        language=language,
        round_trip_text=heard,
        round_trip_cer=question_cer(text, heard),
    )
    if query.round_trip_cer > MAX_ROUND_TRIP_CER:
        wav.unlink()
        raise ValueError(f"TTS round trip failed for {text!r}: heard {heard!r} (CER {query.round_trip_cer:.2f})")
    sidecar.write_text(json.dumps(asdict(query), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return query
