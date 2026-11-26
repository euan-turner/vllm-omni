# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build a named item set (e.g. ``m3``) for one source: annotate candidates, pick items, cut clips, write plans.

Stages, each resumable (LLM/VLM calls are cached, clips are cut deterministically):

1. ``*_candidates``: query rewrite (OmniInteract: LLM; StreamArena: recency words
   replaced by ``sources.streamarena.lag_neutral``), the OmniInteract reference answer
   (``annotate.SHORT_ANSWER``: shortest answer plus aliases), LLM tags, evidence clip, ASR of the
   evidence audio and grounding, one record per candidate in
   ``<source>/sets/<set>/candidates.jsonl`` (excluded candidates keep their reason).
   *Grounded* = the VLM answers the question correctly from the evidence frames
   without seeing the answer, or the LLM finds the answer stated in the evidence
   transcript and its quote really occurs in the transcript (the LLM otherwise
   invents quotes). The VLM's answer-given verdict is recorded but not used: on
   the M3 StreamArena candidates it said "no" even where its own observation
   stated the answer.
2. ``build_items``: picks ``n`` grounded candidates (StreamArena: quotas per stratum ``zh``,
   ``en``, ``zh+en``), speaks their queries (TTS),
   lays each over a leakage-checked backdrop and writes ``items.jsonl`` and the
   set's ``filler.jsonl``.
3. ``build_plans``: per item, a G=0 control, its blind control and both modality
   ablations (context window, 60 units), and single-probe retention plans
   (basic window). Every filler clip an item's plans play is checked for leakage
   by the VLM (frames) and by an ASR keyword scan (speech). A hit adds the clip
   to the item's ``filler_exclude`` and all of that item's plans are rebuilt, so
   filler stays nested across lags.

Filler: OmniInteract uses runs of 1q1a clips that are not the source of any
item in the set (domain unset: the OmniInteract taxonomy is open, plan Q2).
StreamArena uses filler of the item's stratum (video language: ``zh``, ``en``
or ``zh+en``) from videos of a different ``L1``. Backdrops are removed from the filler pool of the whole set.

Design: plan §1a, §1b, §2, §3 and §8 M3 (``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``).
External dependencies: an LLM and a VLM (``llm.CachedChat``), a Qwen3-TTS server (``tts.py``),
Paraformer-zh and Whisper-large-v3 (transcriber callables), ffmpeg.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.av_fd_longmem import annotate as an
from benchmarks.av_fd_longmem import materialise
from benchmarks.av_fd_longmem.asr import read_audio_f32
from benchmarks.av_fd_longmem.audit import OnsetRow, omniinteract_handcheck_sample
from benchmarks.av_fd_longmem.llm import CachedChat
from benchmarks.av_fd_longmem.pilot import BACKDROP_DIR, BACKDROP_START_S, PILOT_SPEC
from benchmarks.av_fd_longmem.pilot import filler_runs as omniinteract_filler_runs
from benchmarks.av_fd_longmem.planner import PlannedProbe, ablation_variant, blind_variant, plan_chained
from benchmarks.av_fd_longmem.schema import (
    ClipRef,
    FillerClip,
    Item,
    ItemClips,
    QueryClip,
    SessionPlan,
    read_jsonl,
    write_json,
    write_jsonl,
)
from benchmarks.av_fd_longmem.score import answer_keys, carries_key, normalise
from benchmarks.av_fd_longmem.sources import omniinteract as oi
from benchmarks.av_fd_longmem.sources import streamarena as sa
from benchmarks.av_fd_longmem.tts import synthesise

ANSWER_TAIL_CHUNKS = 30
CONTROL_CONDITION = {"window_mode": "context", "context_max_units": 60, "temperature": 0.0}
RETENTION_CONDITION = {"window_mode": "basic", "temperature": 0.0}
#: Retention lags per source (Euan, 2026-11-25): SA evidence is ~15 chunks, so G=60 would be partly evicted.
RETENTION_LAGS = {"omniinteract": (60, 300), "streamarena": (30, 300)}
LEAKAGE_WINDOW_FRAMES = 30
OMNIINTERACT_FILLER_POOL_CHUNKS = 4000
MAX_LEAKAGE_ROUNDS = 5

#: (lang, mono f32 16 kHz samples) -> transcript.
Transcribe = Callable[[str, Sequence[float]], str]


@dataclass(frozen=True)
class SetPaths:
    out_root: Path  # av_fd_longmem dataset root
    source: str
    set_name: str

    @property
    def source_root(self) -> Path:
        return self.out_root / self.source

    @property
    def set_root(self) -> Path:
        return self.source_root / "sets" / self.set_name

    @property
    def plans_root(self) -> Path:
        return self.out_root / "plans" / self.set_name

    def cache(self, kind: str) -> Path:
        return self.source_root / "cache" / kind

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.out_root))


@dataclass(frozen=True)
class Models:
    llm: CachedChat
    vlm: CachedChat
    transcribe: Transcribe


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pilot_clips() -> set[str]:
    return {str(entry["clip"]) for entry in json.loads(PILOT_SPEC.read_text(encoding="utf-8"))["items"]}


# --- stage 1: candidates -------------------------------------------------------------------------------------


def _ground(
    models: Models, lang: str, clip: Path, chunks: int, query: str, answer: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = an.clip_frames(clip, chunks)
    read = an.visual_read(models.vlm, frames, query)
    read_match = an.answers_match(models.llm, query, answer, str(read.output["answer"]))
    given = an.ground_visual(models.vlm, frames, query, answer)
    transcript = models.transcribe(lang, read_audio_f32(clip))
    heard = an.ground_audio(models.llm, transcript, query, answer)
    visual = bool(read.output["answerable"]) and bool(read_match.output["match"])
    quote = normalise(str(heard.output["quote"]))
    audio = bool(heard.output["stated"]) and bool(quote) and quote in normalise(transcript)
    grounding = {
        "visual_read": read.output,
        "visual_read_match": read_match.output["match"],
        "visual_given": given.output,
        "audio_transcript": transcript,
        "audio": heard.output,
        "audio_quote_in_transcript": bool(quote) and quote in normalise(transcript),
        "visual": visual,
        "audio_stated": audio,
        "grounded": visual or audio,
    }
    return grounding, [read.provenance, read_match.provenance, given.provenance, heard.provenance]


def _finish(record: dict[str, Any], models: Models, clip: Path, chunks: int, provenance: list) -> dict[str, Any]:
    grounding, ground_prov = _ground(models, record["lang"], clip, chunks, record["query"], record["answer"])
    record |= {"grounding": grounding, "llm_provenance": provenance + ground_prov}
    if record["status"] == "candidate" and not grounding["grounded"]:
        record |= {"status": "excluded", "reason": "not_grounded"}
    return record


def omniinteract_candidates(
    paths: SetPaths, data_root: Path, models: Models, *, n: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Annotate ``n`` sampled realtime items (W=4, at most one per clip, pilot clips left out)."""
    onsets = [OnsetRow(**row) for row in _read_rows(paths.source_root / "audit" / "1q1a" / "onsets.jsonl")]
    sample = omniinteract_handcheck_sample(onsets, n=n, exclude_clips=_pilot_clips(), seed=seed)
    records = []
    for row in sample:
        item_id = f"oi-1q1a-{row.clip}-{row.qa_index}"
        question = oi.annotated_question(data_root, "1q1a", row.clip, row.qa_index)
        rewrite = an.rewrite_omniinteract(models.llm, question.question_text, question.answer_text)
        short = an.short_answer(models.llm, question.question_text, question.answer_text)
        assert row.onset_s is not None
        span = oi.evidence_window(row.onset_s, lookback_s=oi.EVIDENCE_LOOKBACK_S)
        assert span is not None
        start_s, chunks = span
        clip = materialise.cut_source_clip(
            oi.video_path(data_root, "1q1a", row.clip),
            start_s,
            chunks,
            paths.source_root / "clips" / "evidence" / f"{item_id}__e0.mp4",
        )
        record = {
            "item_id": item_id,
            "source": "omniinteract",
            "lang": "zh",
            "stratum": "zh",
            "domain": None,
            "question_original": question.question_text,
            "answer_original": question.answer_text,
            "query": rewrite.output["query"],
            "answer": an.reference_answer(short.output["answer"], short.output["aliases"]),
            "answer_rewrite": rewrite.output["answer"],
            "modality_declared": rewrite.output["modality"],
            "flags": {"standalone": rewrite.output["standalone"]},
            "evidence": {"path": paths.rel(clip), "chunks": chunks, "source_s": [start_s, start_s + chunks]},
            "source_video": f"1q1a/videos/{row.clip}.mp4",
            "source_clip": row.clip,
            "onset_s": row.onset_s,
            "status": "candidate" if rewrite.output["standalone"] else "excluded",
            "reason": None if rewrite.output["standalone"] else "non_standalone",
        }
        records.append(_finish(record, models, clip, chunks, [rewrite.provenance, short.provenance]))
    _write_rows(paths.set_root / "candidates.jsonl", records)
    return records


def _streamarena_reports(paths: SetPaths) -> list[dict[str, Any]]:
    reports = []
    for path in sorted((paths.source_root / "audit").glob("*.json")):
        if path.name not in {"summary.json", "codec.json"}:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def streamarena_candidates(paths: SetPaths, models: Models) -> list[dict[str, Any]]:
    """Annotate every rule survivor of the fetched videos (``audit/<video_id>.json`` reports)."""
    rows = {(r.video_id, r.qid): r for r in sa.recall_rows(paths.source_root / "annotations")}
    records = []
    for report in _streamarena_reports(paths):
        for entry in report["items"]:
            qid = int(entry["item_id"].rsplit("-", 1)[1])
            row = rows[(report["video_id"], qid)]
            tags = an.annotate_streamarena(models.llm, row.question, entry["answer"], row.evidence_hint)
            reasons = [
                name
                for name, hit in (
                    ("multi_window", len(entry["windows"]) != 1),
                    ("relative_time", tags.output["relative_time"]),
                    ("non_standalone", tags.output["non_standalone"]),
                    ("present_referent", tags.output["present_referent"]),
                )
                if hit
            ]
            lo, hi = entry["windows"][0]
            record = {
                "item_id": entry["item_id"],
                "source": "streamarena",
                "lang": row.item_lang,
                "stratum": row.stratum,
                "domain": row.domain,
                "question_original": row.question,
                "answer_original": entry["answer"],
                "query": sa.lag_neutral(row.question),
                "answer": entry["answer"],
                "modality_declared": tags.output["modality"],
                "flags": {k: tags.output[k] for k in ("relative_time", "non_standalone", "present_referent")},
                "evidence": {"path": entry["clips"][0], "chunks": hi - lo, "source_s": [lo, hi]},
                "source_video": report["video_id"],
                "source_clip": report["video_id"],
                "revision": sa.REVISION,
                "status": "excluded" if reasons else "candidate",
                "reason": ",".join(reasons) or None,
            }
            if "multi_window" in reasons:
                records.append(record | {"llm_provenance": [tags.provenance]})
                continue
            clip = paths.out_root / entry["clips"][0]
            records.append(_finish(record, models, clip, hi - lo, [tags.provenance]))
    _write_rows(paths.set_root / "candidates.jsonl", records)
    return records


# --- stage 2: items ------------------------------------------------------------------------------------------


def pick(candidates: Sequence[dict[str, Any]], *, n: int, per_stratum: dict[str, int] | None = None) -> list[dict]:
    """``n`` grounded candidates in candidate order; with ``per_stratum``, that many per stratum, modalities spread.

    Guessable answers are left to the model's own blind control (decision 2026-11-26).
    """
    grounded = [c for c in candidates if c["status"] == "candidate"]
    if per_stratum is None:
        return grounded[:n]
    picked = []
    for stratum, count in per_stratum.items():
        pool = sorted((c for c in grounded if c["stratum"] == stratum), key=lambda c: c["item_id"])
        chosen: list[dict] = []
        while len(chosen) < count and pool:
            seen = {c["modality_declared"] for c in chosen}
            fresh = [c for c in pool if c["modality_declared"] not in seen]
            nxt = (fresh or pool)[0]
            chosen.append(nxt)
            pool.remove(nxt)
        picked += chosen
    return picked


def _leak_hits(
    models: Models, clip: Path, chunks: int, query: str, answer: str, lang: str, asr_cache: Path
) -> dict[str, Any]:
    """VLM leakage over ``LEAKAGE_WINDOW_FRAMES``-frame windows, plus an ASR keyword scan of the clip's speech."""
    frames = an.clip_frames(clip, chunks)
    windows = range(0, len(frames), LEAKAGE_WINDOW_FRAMES)
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(
            pool.map(
                lambda s: an.leakage(models.vlm, frames[s : s + LEAKAGE_WINDOW_FRAMES], s, query, answer).output,
                windows,
            )
        )
    visual = [r for r in replies if r["shows_answer"]]
    key = hashlib.sha256(f"{lang}\0{clip.stat().st_size}\0{clip}".encode()).hexdigest()[:16]
    cached = asr_cache / f"{key}.json"
    if cached.exists():
        transcript = json.loads(cached.read_text(encoding="utf-8"))["text"]
    else:
        transcript = models.transcribe(lang, read_audio_f32(clip))
        asr_cache.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"clip": str(clip), "lang": lang, "text": transcript}, ensure_ascii=False))
    spoken = carries_key(transcript, answer_keys(answer))
    return {"visual": visual, "spoken": spoken, "leaks": bool(visual) or spoken}


def _omniinteract_backdrops(data_root: Path) -> list[Path]:
    return sorted((data_root / BACKDROP_DIR).glob("*.mp4"))


def _query_clip(
    paths: SetPaths, item_id: str, spoken_wav: Path, backdrop: Path, backdrop_start_s: int
) -> tuple[Path, QueryClip]:
    speech_ms = round(materialise.media_duration_s(spoken_wav) * 1000)
    chunks = math.ceil(speech_ms / 1000) + ANSWER_TAIL_CHUNKS
    out = materialise.cut_query_clip(
        backdrop, backdrop_start_s, chunks, spoken_wav, 0, paths.source_root / "clips" / "query" / f"{item_id}.mp4"
    )
    return out, QueryClip(
        path=paths.rel(out),
        chunks=chunks,
        query_onset_chunk=0,
        query_end_ms=speech_ms,
        answer_chunks=chunks - math.ceil(speech_ms / 1000),
    )


def omniinteract_filler(paths: SetPaths, data_root: Path, item_clips: set[str], *, seed: int = 0) -> list[FillerClip]:
    """A shuffled pool of 1q1a filler runs from clips that are not the source of any item in the set."""
    runs = omniinteract_filler_runs(data_root, "1q1a", item_clips)
    random.Random(seed).shuffle(runs)
    pool: list[FillerClip] = []
    covered = 0
    for clip, start, chunks in runs:
        if covered >= OMNIINTERACT_FILLER_POOL_CHUNKS:
            break
        filler_id = f"oi-1q1a-{clip}-f{start:04d}"
        out = paths.source_root / "clips" / "filler" / f"{filler_id}.mp4"
        if not out.exists():
            materialise.cut_source_clip(oi.video_path(data_root, "1q1a", clip), start, chunks, out)
        pool.append(
            FillerClip(
                filler_id=filler_id,
                path=paths.rel(out),
                chunks=chunks,
                source="omniinteract",
                domain=None,
                provenance={"video": f"1q1a/videos/{clip}.mp4", "src_t0": start, "encode": materialise.ENCODE_SETTINGS},
            )
        )
        covered += chunks
    return pool


def streamarena_filler(paths: SetPaths) -> list[FillerClip]:
    """Every materialised filler clip of the fetched videos, labelled with its video's ``L1``, language and stratum."""
    meta = {row["video_id"]: row for row in _read_rows(paths.source_root / "annotations" / "video_meta.jsonl")}
    pool = []
    for report in _streamarena_reports(paths):
        video = meta[report["video_id"]]
        for entry in report["filler"]:
            pool.append(
                FillerClip(
                    filler_id=Path(entry["path"]).stem,
                    path=entry["path"],
                    chunks=int(entry["chunks"]),
                    source="streamarena",
                    domain=str(video["L1"]),
                    provenance={
                        "video_id": report["video_id"],
                        "src_t0": int(entry["src_t0"]),
                        "lang": sa.ITEM_LANGS.get(str(video["language"])),
                        "stratum": str(video["language"]),
                        "revision": sa.REVISION,
                        "encode": materialise.ENCODE_SETTINGS,
                    },
                )
            )
    return pool


def item_pool(item: Item, filler: Sequence[FillerClip], *, exclude: set[str]) -> list[FillerClip]:
    """The filler an item may play, in an order fixed by the item id (so items do not all share one opening clip)."""
    if item.source == "streamarena":
        stratum = item.provenance["stratum"]
        eligible = [f for f in filler if f.provenance["stratum"] == stratum and f.domain != item.domain]
    else:
        eligible = list(filler)
    eligible = [f for f in eligible if f.filler_id not in exclude and f.filler_id not in item.filler_exclude]
    random.Random(item.item_id).shuffle(eligible)
    return eligible


def build_items(
    paths: SetPaths,
    candidates: Sequence[dict[str, Any]],
    models: Models,
    *,
    data_root: Path | None,
    tts_url: str,
    transcribe_file: Callable[[str, Path], str],
) -> tuple[list[Item], list[FillerClip]]:
    """Spoken queries over leakage-checked backdrops.

    Writes ``items.jsonl``, ``filler.jsonl`` and ``backdrops.jsonl``.
    """
    if paths.source == "omniinteract":
        assert data_root is not None
        filler = omniinteract_filler(paths, data_root, {c["source_clip"] for c in candidates})
        backdrops = [(path, BACKDROP_START_S, None) for path in _omniinteract_backdrops(data_root)]
    else:
        filler = streamarena_filler(paths)
        backdrops = []
    used_backdrops: set[str] = set()
    backdrop_rows = []
    items = []
    for candidate in candidates:
        spoken = synthesise(
            candidate["query"],
            candidate["lang"],
            base_url=tts_url,
            cache_dir=paths.source_root / "cache" / "tts",
            transcribe=lambda wav, lang=candidate["lang"]: transcribe_file(lang, wav),
        )
        if paths.source == "streamarena":
            options = [
                (paths.out_root / f.path, 0, f.filler_id)
                for f in sorted(filler, key=lambda f: f.filler_id)
                if f.provenance["stratum"] == candidate["stratum"] and f.domain != candidate["domain"]
            ]
        else:
            options = backdrops
        for backdrop, start_s, filler_id in options:
            key = filler_id or str(backdrop)
            if key in used_backdrops:
                continue
            query_path, query = _query_clip(paths, candidate["item_id"], Path(spoken.wav), backdrop, start_s)
            check = _leak_hits(
                models, query_path, query.chunks, candidate["query"], candidate["answer"], candidate["lang"],
                paths.cache("asr"),
            )  # fmt: skip
            backdrop_rows.append({"item_id": candidate["item_id"], "backdrop": key, **check})
            if not check["leaks"]:
                break
        else:
            raise ValueError(f"{candidate['item_id']}: no leakage-free backdrop left")
        used_backdrops.add(key)
        items.append(
            Item(
                item_id=candidate["item_id"],
                source=candidate["source"],
                lang=candidate["lang"],
                clips=ItemClips(
                    evidence=[ClipRef(path=candidate["evidence"]["path"], chunks=candidate["evidence"]["chunks"])],
                    query=query,
                ),
                query_text=spoken.text,
                query_text_original=candidate["question_original"],
                answer=candidate["answer"],
                modality_declared=candidate["modality_declared"],
                domain=candidate["domain"],
                provenance={
                    "set": paths.set_name,
                    "stratum": candidate["stratum"],
                    "video": candidate["source_video"],
                    "evidence_source_s": candidate["evidence"]["source_s"],
                    "onset_s": candidate.get("onset_s"),
                    "revision": candidate.get("revision"),
                    "answer_original": candidate["answer_original"],
                    "grounding": {k: candidate["grounding"][k] for k in ("visual", "audio_stated")},
                    "backdrop": key if paths.source == "streamarena" else str(backdrop.relative_to(data_root)),
                    "backdrop_start_s": start_s,
                    "tts": {
                        "model": spoken.model,
                        "voice": spoken.voice,
                        "language": spoken.language,
                        "round_trip_text": spoken.round_trip_text,
                        "round_trip_cer": spoken.round_trip_cer,
                    },
                    "llm": candidate["llm_provenance"],
                    "encode": materialise.ENCODE_SETTINGS,
                },
            )
        )
    filler = [f for f in filler if f.filler_id not in used_backdrops]
    write_jsonl(paths.set_root / "items.jsonl", items)
    write_jsonl(paths.set_root / "filler.jsonl", filler)
    _write_rows(paths.set_root / "backdrops.jsonl", backdrop_rows)
    return items, filler


# --- stage 3: plans ------------------------------------------------------------------------------------------


def item_plans(paths: SetPaths, item: Item, pool: Sequence[FillerClip]) -> list[SessionPlan]:
    """G=0 control, blind control, audio/video ablations and the retention plans of one item, on one filler order."""
    prefix = f"{paths.set_name}-{item.item_id}"
    control = plan_chained(f"{prefix}-g0", [PlannedProbe(item, 0)], pool, condition=CONTROL_CONDITION)
    control = replace(control, kind="control_g0")
    plans = [
        control,
        blind_variant(control, pool, plan_id=f"{prefix}-blind"),
        ablation_variant(control, mode="audio", plan_id=f"{prefix}-ablate-audio"),
        ablation_variant(control, mode="video", plan_id=f"{prefix}-ablate-video"),
    ]
    for lag in RETENTION_LAGS[item.source]:
        plan = plan_chained(f"{prefix}-g{lag}", [PlannedProbe(item, lag)], pool, condition=RETENTION_CONDITION)
        plans.append(replace(plan, kind="single"))
    return plans


def build_plans(
    paths: SetPaths, items: Sequence[Item], filler: Sequence[FillerClip], models: Models
) -> tuple[list[Item], list[SessionPlan]]:
    """Leakage-checked plans for every item; rewrites ``items.jsonl`` with each item's ``filler_exclude``."""
    by_path = {f.path: f for f in filler}
    leakage_rows = []
    done_items, all_plans = [], []
    for item in items:
        checked: set[str] = set()
        for _round in range(MAX_LEAKAGE_ROUNDS):
            plans = item_plans(paths, item, item_pool(item, filler, exclude=set()))
            played = {s.clip for p in plans for s in p.segments if s.role == "filler"}
            new_hits = []
            for clip_path in sorted(played - checked):
                clip = by_path[clip_path]
                check = _leak_hits(
                    models, paths.out_root / clip.path, clip.chunks, item.query_text, item.answer, item.lang,
                    paths.cache("asr"),
                )  # fmt: skip
                leakage_rows.append({"item_id": item.item_id, "filler_id": clip.filler_id, **check})
                checked.add(clip_path)
                if check["leaks"]:
                    new_hits.append(clip.filler_id)
            if not new_hits:
                break
            item = replace(item, filler_exclude=sorted({*item.filler_exclude, *new_hits}))
        else:
            raise ValueError(f"{item.item_id}: leakage hits did not settle in {MAX_LEAKAGE_ROUNDS} rounds")
        done_items.append(item)
        all_plans += plans
    ids = [p.plan_id for p in all_plans]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate plan ids")
    for plan in all_plans:
        write_json(paths.plans_root / f"{plan.plan_id}.json", plan)
    write_jsonl(paths.set_root / "items.jsonl", done_items)
    _write_rows(paths.set_root / "leakage.jsonl", leakage_rows)
    return done_items, all_plans


def load_set(paths: SetPaths) -> tuple[list[Item], list[FillerClip]]:
    return (
        list(read_jsonl(paths.set_root / "items.jsonl", Item)),
        list(read_jsonl(paths.set_root / "filler.jsonl", FillerClip)),
    )
