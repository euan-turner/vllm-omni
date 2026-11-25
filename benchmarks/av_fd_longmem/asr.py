# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech segmentation for amended query onsets: fsmn-vad segments, each transcribed by Paraformer-zh.

Models load from the Hugging Face hub (``funasr/fsmn-vad``, ``funasr/paraformer-zh``)
on first use; ModelScope is not used.

External dependencies: funasr (pinned dev dependency), numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from benchmarks.av_fd_longmem.sources.omniinteract import SpeechSegment


class FunasrSpeechSegmenter:
    """Callable ``(samples, sample_rate, offset_s) -> [SpeechSegment]`` in clip seconds."""

    def __init__(self, *, device: str = "cuda:1") -> None:
        self._device = device
        self._vad: Any | None = None
        self._asr: Any | None = None

    def _models(self) -> tuple[Any, Any]:
        if self._vad is None or self._asr is None:
            from funasr import AutoModel

            self._vad = AutoModel(model="fsmn-vad", device=self._device, disable_update=True, hub="hf")
            self._asr = AutoModel(model="paraformer-zh", device=self._device, disable_update=True, hub="hf")
        return self._vad, self._asr

    def __call__(self, samples: Sequence[float], sample_rate: int, offset_s: float) -> list[SpeechSegment]:
        vad, asr = self._models()
        wav = np.asarray(samples, dtype=np.float32)
        spans = vad.generate(input=wav, fs=sample_rate)[0]["value"]
        segments = []
        for start_ms, end_ms in spans:
            piece = wav[int(start_ms * sample_rate / 1000) : int(end_ms * sample_rate / 1000)]
            text = asr.generate(input=piece, fs=sample_rate)[0]["text"].replace(" ", "")
            segments.append(SpeechSegment(offset_s + start_ms / 1000, offset_s + end_ms / 1000, text))
        return segments
