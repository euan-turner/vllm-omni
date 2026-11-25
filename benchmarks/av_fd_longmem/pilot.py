# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build the OmniInteract pilot: items, filler, query backdrops and session plans.

Items come from the hand-picked ``pilot/omniinteract_pilot.json``. Each item's
query time is amended to the measured spoken-question onset and its evidence
ends just before it. Queries are the past-tense rewrites, spoken by TTS over a
``1qna`` cooking backdrop with a silent answer tail. Filler is every run of at
least ``FILLER_MIN_CHUNKS`` in the other ``1q1a`` clips that keeps clear of
``[question_time - 5, question_time + 15]`` of every annotated question.

External dependencies: ffmpeg; funasr (onsets, TTS round trip); a Qwen3-TTS
server (see ``tts.py``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from benchmarks.av_fd_longmem import materialise
from benchmarks.av_fd_longmem.asr import SAMPLE_RATE, FunasrSpeechSegmenter, read_audio_f32
from benchmarks.av_fd_longmem.planner import PlannedProbe, plan_chained
from benchmarks.av_fd_longmem.schema import (
    ClipRef,
    FillerClip,
    Item,
    ItemClips,
    QueryClip,
    SessionPlan,
    write_json,
    write_jsonl,
)
from benchmarks.av_fd_longmem.sources import omniinteract as oi
from benchmarks.av_fd_longmem.tts import synthesise

PILOT_SPEC = Path(__file__).parent / "pilot" / "omniinteract_pilot.json"
EVIDENCE_LOOKBACK_S = 4
ANSWER_TAIL_CHUNKS = 30
FILLER_EXCLUDE_BEFORE_S = 5.0
FILLER_EXCLUDE_AFTER_S = 15.0
FILLER_MIN_CHUNKS = 30
FILLER_MAX_CHUNKS = 300
BACKDROP_DIR = Path("1qna/videos_bench/captaincook4d")
BACKDROP_START_S = 60
WINDOW_CONDITION = {"window_mode": "basic", "temperature": 0.0}


@dataclass(frozen=True)
class PilotPaths:
    data_root: Path  # OmniInteract ``data`` directory
    out_root: Path  # av_fd_longmem dataset root

    @property
    def source_root(self) -> Path:
        return self.out_root / "omniinteract"

    @property
    def plans_root(self) -> Path:
        return self.out_root / "plans" / "pilot"


def _spec() -> dict[str, object]:
    return json.loads(PILOT_SPEC.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_items(paths: PilotPaths, *, tts_url: str, segmenter: FunasrSpeechSegmenter) -> list[Item]:
    spec = _spec()
    subset, lang = str(spec["subset"]), str(spec["lang"])
    root = paths.source_root
    backdrops = sorted((paths.data_root / BACKDROP_DIR).glob("*.mp4"))
    items: list[Item] = []
    for index, entry in enumerate(spec["items"]):
        clip, qa_index = str(entry["clip"]), int(entry["qa_index"])
        item_id = f"oi-{subset}-{clip}-{qa_index}"
        question = oi.annotated_question(paths.data_root, subset, clip, qa_index)
        video = oi.video_path(paths.data_root, subset, clip)
        search_start = max(0.0, question.question_time_s - oi.ONSET_SEARCH_BEFORE_S)
        samples = read_audio_f32(video)
        window = samples[
            int(search_start * SAMPLE_RATE) : int((question.question_time_s + oi.ONSET_SEARCH_AFTER_S) * SAMPLE_RATE)
        ]
        onset = oi.match_question_onset(question.question_text, segmenter(window, SAMPLE_RATE, search_start))
        if onset is None:
            raise ValueError(f"{item_id}: no speech segment matches the annotated question")
        span = oi.evidence_window(onset.onset_s, lookback_s=EVIDENCE_LOOKBACK_S)
        if span is None:
            raise ValueError(f"{item_id}: onset {onset.onset_s:.2f}s leaves no evidence")
        evidence = materialise.cut_source_clip(
            video, span[0], span[1] - span[0], root / "clips" / "evidence" / f"{item_id}__e0.mp4"
        )
        spoken = synthesise(
            str(entry["query_text"]),
            lang,
            base_url=tts_url,
            cache_dir=root / "cache" / "tts",
            transcribe=segmenter.transcribe_file,
        )
        speech_ms = round(materialise.media_duration_s(Path(spoken.wav)) * 1000)
        query_chunks = math.ceil(speech_ms / 1000) + ANSWER_TAIL_CHUNKS
        backdrop = backdrops[index]
        query = materialise.cut_query_clip(
            backdrop, BACKDROP_START_S, query_chunks, Path(spoken.wav), 0, root / "clips" / "query" / f"{item_id}.mp4"
        )
        items.append(
            Item(
                item_id=item_id,
                source="omniinteract",
                lang=lang,
                clips=ItemClips(
                    evidence=[ClipRef(path=_rel(evidence, paths.out_root), chunks=span[1] - span[0])],
                    query=QueryClip(
                        path=_rel(query, paths.out_root),
                        chunks=query_chunks,
                        query_onset_chunk=0,
                        query_end_ms=speech_ms,
                        answer_chunks=query_chunks - math.ceil(speech_ms / 1000),
                    ),
                ),
                query_text=spoken.text,
                query_text_original=question.question_text,
                answer=str(entry.get("answer_override") or question.answer_text),
                modality_declared=str(entry["modality"]),
                provenance={
                    "video": f"{subset}/videos/{clip}.mp4",
                    "qa_index": qa_index,
                    "question_time_s": question.question_time_s,
                    "onset_s": onset.onset_s,
                    "onset_transcript": onset.segment.text,
                    "onset_cer": onset.cer,
                    "evidence_source_s": list(span),
                    "backdrop": str(backdrop.relative_to(paths.data_root)),
                    "backdrop_start_s": BACKDROP_START_S,
                    "tts": {
                        "model": spoken.model,
                        "voice": spoken.voice,
                        "language": spoken.language,
                        "round_trip_cer": spoken.round_trip_cer,
                    },
                    "lag_chunks": int(entry["lag_chunks"]),
                    "encode": materialise.ENCODE_SETTINGS,
                },
            )
        )
    write_jsonl(root / "items.jsonl", items)
    return items


def filler_runs(data_root: Path, subset: str, exclude_clips: set[str]) -> list[tuple[str, int, int]]:
    """``(clip, start_s, chunks)`` filler runs of every non-excluded clip, longest runs capped."""
    runs: list[tuple[str, int, int]] = []
    for annotation in sorted((data_root / subset / "annotations").glob("*.json")):
        clip = annotation.stem
        if clip in exclude_clips:
            continue
        duration = materialise.media_duration_s(oi.video_path(data_root, subset, clip))
        blocked = sorted(
            (max(0.0, t - FILLER_EXCLUDE_BEFORE_S), t + FILLER_EXCLUDE_AFTER_S)
            for row in oi.annotation_rows(data_root, subset, clip)
            if (t := oi.parse_annotation_time(row.get("question_time"))) is not None
        )
        cursor = 0.0
        for lo, hi in [*blocked, (duration, duration)]:
            start, end = math.ceil(cursor), math.floor(min(lo, duration))
            while end - start >= FILLER_MIN_CHUNKS:
                chunks = min(FILLER_MAX_CHUNKS, end - start)
                runs.append((clip, start, chunks))
                start += chunks
            cursor = max(cursor, hi)
    return runs


def build_filler(paths: PilotPaths, *, needed_chunks: int) -> list[FillerClip]:
    """Materialise filler runs in clip order until ``needed_chunks`` are covered."""
    spec = _spec()
    subset = str(spec["subset"])
    pilot_clips = {str(entry["clip"]) for entry in spec["items"]}
    filler: list[FillerClip] = []
    covered = 0
    for clip, start, chunks in filler_runs(paths.data_root, subset, pilot_clips):
        if covered >= needed_chunks:
            break
        filler_id = f"oi-{subset}-{clip}-f{start:04d}"
        out = materialise.cut_source_clip(
            oi.video_path(paths.data_root, subset, clip),
            start,
            chunks,
            paths.source_root / "clips" / "filler" / f"{filler_id}.mp4",
        )
        filler.append(
            FillerClip(
                filler_id=filler_id,
                path=_rel(out, paths.out_root),
                chunks=chunks,
                source="omniinteract",
                domain=None,
                provenance={
                    "video": f"{subset}/videos/{clip}.mp4",
                    "src_t0": start,
                    "encode": materialise.ENCODE_SETTINGS,
                },
            )
        )
        covered += chunks
    write_jsonl(paths.source_root / "filler.jsonl", filler)
    return filler


def build_plans(paths: PilotPaths, items: list[Item], filler: list[FillerClip]) -> list[SessionPlan]:
    """The chained pilot session and one G=0 validity control per item (context mode, 60 units)."""
    lags = {item.item_id: int(item.provenance["lag_chunks"]) for item in items}
    chained = plan_chained(
        "pilot-chained", [PlannedProbe(item, lags[item.item_id]) for item in items], filler, condition=WINDOW_CONDITION
    )
    plans = [chained]
    control_condition = {"window_mode": "context", "context_max_units": 60, "temperature": 0.0}
    for index, item in enumerate(items):
        # Each control draws its own filler, offset so controls do not share an opening clip.
        pool = filler[index:] + filler[:index]
        control = plan_chained(f"pilot-g0-{item.item_id}", [PlannedProbe(item, 0)], pool, condition=control_condition)
        plans.append(replace(control, kind="control_g0"))
    for plan in plans:
        write_json(paths.plans_root / f"{plan.plan_id}.json", plan)
    return plans
