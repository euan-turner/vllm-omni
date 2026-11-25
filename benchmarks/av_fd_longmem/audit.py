# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""M0 audits: OmniInteract question onsets and evidence room; StreamArena exclusions and a 3-video sample.

OmniInteract:

For every annotated question of a subset, the audit locates the spoken
question (``sources.omniinteract.find_question_onset``) and records the onset,
its offset from the annotated ``question_time`` and the matched transcript.
It also records the *pre-onset room*: seconds between the evidence guard
before the onset and the end of the latest other question spoken before it
(its matched speech segment), or the clip start. The model's answers are not
on the clip soundtrack, so only the user's questions bound the room. The room
bounds the evidence lookback ``W``. Items whose speech began before the search
window, whose question is also spoken inside the evidence window, or (realtime)
whose question already refers to the past are excluded by status.

Speech segments are cached per clip under ``<out>/segments/<clip>.json`` so the
audit resumes and matching can change without ASR; rows go to ``onsets.jsonl``.
``summarise_onsets`` turns the rows into exclusion counts, the onset
offset distribution and the item yield per candidate ``W``.

StreamArena: ``summarise_streamarena`` counts the rule-based exclusions over
all HR rows; ``materialise_streamarena_video`` cuts the evidence and filler
clips of one fetched video and measures their size.

Query codec: ``query_codec_check`` compares ASR error on spoken queries as
synthesised, after the clip AAC encode and after a lossless FLAC encode.

Design: plan §0 and §8 M0 (``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``).
External dependencies: a speech segmenter (``asr.FunasrSpeechSegmenter``); ffmpeg and ffprobe; for the
codec check a Qwen3-TTS server, Paraformer-zh and Whisper-large-v3 (``seed_tts_eval``).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmarks.av_fd_longmem import materialise
from benchmarks.av_fd_longmem.asr import SAMPLE_RATE, FunasrSpeechSegmenter, read_audio_f32
from benchmarks.av_fd_longmem.sources import omniinteract as oi
from benchmarks.av_fd_longmem.sources import streamarena as sa
from benchmarks.av_fd_longmem.tts import synthesise

#: Assumed length of a question's speech when its onset was not matched.
QUESTION_SPEECH_FALLBACK_S = 5.0
#: Candidate evidence lookbacks reported by the audit.
CANDIDATE_LOOKBACK_S = (2, 4, 6, 8, 10)


@dataclass(frozen=True)
class OnsetRow:
    clip: str
    qa_index: int
    question_type: str
    is_interrupted: bool
    question_time_s: float | None
    answer_time_s: float | None
    question_text: str
    onset_s: float | None
    onset_delta_s: float | None
    onset_cer: float | None
    onset_transcript: str | None
    qualifying_segments: int  # segments within the onset CER threshold (>1: repeated or similar question nearby)
    segments: list[list[object]]  # [start_s, end_s, text] of every segment in the search window
    pre_onset_room_s: float | None
    status: str  # see ``onset_rows``


def _speech_span(question: oi.AnnotatedQuestion | None, match: oi.OnsetMatch | None) -> tuple[float, float] | None:
    """Where an item's question is spoken: its matched segment, else ``QUESTION_SPEECH_FALLBACK_S`` from its time."""
    if match is not None:
        return match.segment.start_s, match.segment.end_s
    if question is not None:
        return question.question_time_s, question.question_time_s + QUESTION_SPEECH_FALLBACK_S
    return None


def _room(onset_s: float, index: int, spans: Sequence[tuple[float, float] | None]) -> float:
    """Seconds before the evidence guard back to the end of the latest other question spoken before the onset."""
    bound = 0.0
    for other_index, span in enumerate(spans):
        if span is None or other_index == index or span[0] >= onset_s:
            continue
        bound = max(bound, span[1])
    return max(0.0, onset_s - oi.EVIDENCE_GUARD_S - bound)


def _question_in_evidence(question_text: str, onset_s: float, match: oi.OnsetMatch, seen: Sequence[oi.SpeechSegment]) -> bool:
    """Whether another segment inside the ``EVIDENCE_LOOKBACK_S`` evidence window also holds the question."""
    window = oi.evidence_window(onset_s, lookback_s=oi.EVIDENCE_LOOKBACK_S)
    if window is None:
        return False
    start, chunks = window
    return any(
        segment is not match.segment
        and segment.start_s < start + chunks
        and start < segment.end_s
        and oi.substring_cer(question_text, segment.text) <= oi.ONSET_MAX_CER
        for segment in seen
    )


def _clip_questions(data_root: Path, subset: str, clip: str) -> tuple[list[dict], list[oi.AnnotatedQuestion | None]]:
    rows = oi.annotation_rows(data_root, subset, clip)
    questions: list[oi.AnnotatedQuestion | None] = []
    for index in range(len(rows)):
        try:
            questions.append(oi.annotated_question(data_root, subset, clip, index))
        except ValueError:
            questions.append(None)
    return rows, questions


def onset_rows(
    data_root: Path, subset: str, clip: str, segments: Sequence[Sequence[oi.SpeechSegment] | None]
) -> list[OnsetRow]:
    """Onset rows of a clip from the speech segments seen around each question (``None``: time unparsable).

    Status, first applicable: ``unparsable_time``, ``no_onset_match``,
    ``onset_censored`` (the speech began before the search window), then for
    realtime items ``question_in_evidence`` (the question is also spoken inside
    the evidence window) and ``recall_phrased``; otherwise ``ok``.
    """
    rows, questions = _clip_questions(data_root, subset, clip)
    matches = [
        None if question is None or seen is None else oi.match_question_onset(question.question_text, seen)
        for question, seen in zip(questions, segments, strict=True)
    ]
    spans = [_speech_span(question, match) for question, match in zip(questions, matches, strict=True)]
    out: list[OnsetRow] = []
    for index, (row, question, seen, match) in enumerate(zip(rows, questions, segments, matches, strict=True)):
        base = {
            "clip": clip,
            "qa_index": index,
            "question_type": str(row.get("question_type", "")),
            "is_interrupted": bool(row.get("is_interrupted", False)),
            "question_text": str(row.get("question_text", "")),
        }
        if question is None or seen is None:
            out.append(OnsetRow(**base, question_time_s=None, answer_time_s=None, onset_s=None, onset_delta_s=None,
                                onset_cer=None, onset_transcript=None, qualifying_segments=0, segments=[],
                                pre_onset_room_s=None, status="unparsable_time"))  # fmt: skip
            continue
        qualifying = sum(1 for s in seen if oi.substring_cer(question.question_text, s.text) <= oi.ONSET_MAX_CER)
        common = {
            **base,
            "question_time_s": question.question_time_s,
            "answer_time_s": question.answer_time_s,
            "segments": [[round(s.start_s, 2), round(s.end_s, 2), s.text] for s in seen],
            "qualifying_segments": qualifying,
        }
        if match is None:
            out.append(OnsetRow(**common, onset_s=None, onset_delta_s=None, onset_cer=None, onset_transcript=None,
                                pre_onset_room_s=None, status="no_onset_match"))  # fmt: skip
            continue
        search_start = max(0.0, question.question_time_s - oi.ONSET_SEARCH_BEFORE_S)
        realtime = question.question_type == "realtime"
        if oi.onset_censored(match, search_start):
            status = "onset_censored"
        elif realtime and _question_in_evidence(question.question_text, match.onset_s, match, seen):
            status = "question_in_evidence"
        elif realtime and oi.is_recall_phrased(question.question_text):
            status = "recall_phrased"
        else:
            status = "ok"
        out.append(
            OnsetRow(
                **common,
                onset_s=round(match.onset_s, 3),
                onset_delta_s=round(match.onset_s - question.question_time_s, 3),
                onset_cer=round(match.cer, 3),
                onset_transcript=match.segment.text,
                pre_onset_room_s=round(_room(match.onset_s, index, spans), 3),
                status=status,
            )
        )
    return out


def clip_segments(
    data_root: Path, subset: str, clip: str, segmenter: oi.SpeechSegmenter
) -> list[list[oi.SpeechSegment] | None]:
    """Speech segments in each question's onset search window (``None`` when its time is unparsable)."""
    _rows, questions = _clip_questions(data_root, subset, clip)
    samples = read_audio_f32(oi.video_path(data_root, subset, clip))
    return [
        None if question is None else oi.find_question_onset(question, samples, SAMPLE_RATE, segmenter)[1]
        for question in questions
    ]


def audit_clip(data_root: Path, subset: str, clip: str, segmenter: oi.SpeechSegmenter) -> list[OnsetRow]:
    return onset_rows(data_root, subset, clip, clip_segments(data_root, subset, clip, segmenter))


def audit_subset(data_root: Path, subset: str, out_dir: Path, segmenter: oi.SpeechSegmenter) -> list[OnsetRow]:
    """Onset rows for every clip of ``subset``.

    The speech segments of each clip are cached in ``out_dir/segments_b<before>_a<after>/<clip>.json``
    (keyed by the onset search window), so re-running with a changed matching rule needs no ASR.
    """
    cache = out_dir / f"segments_b{oi.ONSET_SEARCH_BEFORE_S:g}_a{oi.ONSET_SEARCH_AFTER_S:g}"
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[OnsetRow] = []
    for annotation in sorted((data_root / subset / "annotations").glob("*.json")):
        clip = annotation.stem
        path = cache / f"{clip}.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            segments = [None if seen is None else [oi.SpeechSegment(*s) for s in seen] for seen in stored]
        else:
            segments = clip_segments(data_root, subset, clip, segmenter)
            stored = [None if seen is None else [[s.start_s, s.end_s, s.text] for s in seen] for seen in segments]
            path.write_text(json.dumps(stored, ensure_ascii=False) + "\n", encoding="utf-8")
        rows += onset_rows(data_root, subset, clip, segments)
    with (out_dir / "onsets.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return rows


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[round(q * (len(ordered) - 1))], 3)

    return {"n": len(ordered), "min": at(0), "p05": at(0.05), "p25": at(0.25), "median": at(0.5), "p75": at(0.75),
            "p95": at(0.95), "max": at(1), "mean": round(statistics.fmean(ordered), 3)}  # fmt: skip


def summarise_onsets(rows: Sequence[OnsetRow]) -> dict[str, object]:
    """Exclusion counts, onset offsets (matched onsets) and the realtime item yield per candidate lookback ``W``.

    Yield counts realtime items with status ``ok``, not interrupted, whose room fits ``W``.
    """
    by_type: dict[str, dict[str, object]] = {}
    for question_type in sorted({r.question_type for r in rows}):
        typed = [r for r in rows if r.question_type == question_type]
        matched = [r for r in typed if r.onset_s is not None]
        by_type[question_type] = {
            "items": len(typed),
            "status": dict(Counter(r.status for r in typed)),
            "onset_match_rate": round(len(matched) / len(typed), 3) if typed else None,
            "onset_delta_s": _quantiles([r.onset_delta_s for r in matched if r.onset_delta_s is not None]),
            "abs_delta_over_1_1s": sum(
                1 for r in matched if r.onset_delta_s is not None and abs(r.onset_delta_s) > 1.1
            ),
            "onset_cer": _quantiles([r.onset_cer for r in matched if r.onset_cer is not None]),
        }
    eligible = [r for r in rows if r.question_type == "realtime" and r.status == "ok" and not r.is_interrupted]
    rooms = [r.pre_onset_room_s for r in eligible if r.pre_onset_room_s is not None]
    yield_by_w = {
        str(w): {
            "items": sum(1 for room in rooms if room >= w),
            "clips": len({r.clip for r in eligible if r.pre_onset_room_s is not None and r.pre_onset_room_s >= w}),
        }
        for w in CANDIDATE_LOOKBACK_S
    }
    return {
        "items": len(rows),
        "clips": len({r.clip for r in rows}),
        "question_speech_fallback_s": QUESTION_SPEECH_FALLBACK_S,
        "by_question_type": by_type,
        "realtime_ok_not_interrupted": len(eligible),
        "realtime_pre_onset_room_s": _quantiles(rooms),
        "realtime_no_room": sum(1 for room in rooms if room < 1),
        "realtime_yield_by_lookback_s": yield_by_w,
    }


def summarise_streamarena(rows: Sequence[sa.RecallRow]) -> dict[str, object]:
    """Rule-based exclusion counts (each rule, and first-failing-rule waterfall) and the survivors' strata."""
    reasons = {(r.video_id, r.qid): sa.exclusion_reasons(r) for r in rows}
    survivors = [r for r in rows if not reasons[(r.video_id, r.qid)]]
    windows = {(r.video_id, r.qid): sa.evidence_windows(r.evidence_ts_sec, r.ask_sec) for r in survivors}
    evidence_s = [sum(hi - lo for lo, hi in w) for w in windows.values()]
    return {
        "hr_items": len(rows),
        "videos": len({r.video_id for r in rows}),
        "rule_hits": dict(Counter(reason for rs in reasons.values() for reason in rs)),
        "first_rule_waterfall": dict(Counter(rs[0] for rs in reasons.values() if rs)),
        "survivors": len(survivors),
        "survivor_videos": len({r.video_id for r in survivors}),
        "survivors_by_lang": dict(Counter(r.item_lang for r in survivors)),
        "survivors_by_domain": dict(Counter(r.domain for r in survivors)),
        "survivors_by_horizon_layer": dict(Counter(r.horizon_layer for r in survivors)),
        "survivor_evidence_windows": dict(Counter(len(w) for w in windows.values())),
        "survivor_evidence_s_without_cues": _quantiles(evidence_s),
        "survivor_evidence_span_s": _quantiles([w[-1][1] - w[0][0] for w in windows.values() if w]),
        "survivors_evidence_span_over_60s": sum(1 for w in windows.values() if w and w[-1][1] - w[0][0] > 60),
        "survivors_needing_lag_neutral_rewrite": sum(1 for r in survivors if sa.needs_lag_neutral_rewrite(r)),
        "survivors_evidence_within_10s_of_ask": sum(
            1
            for r in survivors
            if windows[(r.video_id, r.qid)] and windows[(r.video_id, r.qid)][-1][1] > r.ask_sec - 10
        ),
        "not_implemented": ["llm_relative_window_phrasing", "llm_non_standalone", "present_referent_tag"],
    }


def materialise_streamarena_video(
    annotations: Path, raw_dir: Path, video_id: str, out_root: Path, rows: Sequence[sa.RecallRow]
) -> dict[str, object]:
    """Evidence clips for the video's surviving items and its filler clips; sizes and window extension stats.

    ``raw_dir`` holds the extracted tar (``<video_id>.mp4`` and its ``.vtt`` files).
    Clips go to ``out_root/clips/{evidence,filler}`` (``out_root`` is the source root).
    """
    video = raw_dir / f"{video_id}.mp4"
    cues = sa.vtt_cues(sorted(raw_dir.glob("*.vtt")))
    duration = materialise.media_duration_s(video)
    items = []
    for row in rows:
        if row.video_id != video_id or sa.exclusion_reasons(row):
            continue
        bare = sa.evidence_windows(row.evidence_ts_sec, row.ask_sec)
        extended = sa.evidence_windows(row.evidence_ts_sec, row.ask_sec, cues)
        item_id = f"sa-{video_id}-{row.qid}"
        paths = []
        for k, (lo, hi) in enumerate(extended):
            clip = materialise.cut_source_clip(
                video, lo, hi - lo, out_root / "clips" / "evidence" / f"{item_id}__e{k}.mp4"
            )
            paths.append(str(clip.relative_to(out_root.parent)))
        items.append(
            {
                "item_id": item_id,
                "lang": row.item_lang,
                "question": row.question,
                "answer": row.answer_zh if row.item_lang == "zh" else row.answer_en,
                "ask_sec": row.ask_sec,
                "evidence_ts_sec": row.evidence_ts_sec,
                "evidence_hint": row.evidence_hint,
                "windows_without_cues": bare,
                "windows": extended,
                "clips": paths,
            }
        )
    filler = []
    for start, chunks in sa.filler_runs(duration, sa.annotated_times(annotations, video_id)):
        out = out_root / "clips" / "filler" / f"sa-{video_id}-f{start:05d}.mp4"
        materialise.cut_source_clip(video, start, chunks, out)
        filler.append(
            {
                "path": str(out.relative_to(out_root.parent)),
                "src_t0": start,
                "chunks": chunks,
                "bytes": out.stat().st_size,
            }
        )
    filler_chunks = sum(f["chunks"] for f in filler)
    filler_bytes = sum(f["bytes"] for f in filler)
    return {
        "video_id": video_id,
        "duration_s": duration,
        "vtt_cues": len(cues),
        "items": items,
        "filler_clips": len(filler),
        "filler_chunks": filler_chunks,
        "filler_mb_per_hour": round(filler_bytes / 1e6 / (filler_chunks / 3600), 1) if filler_chunks else None,
        "filler": filler,
    }


#: Query audio encodings compared by ``query_codec_check``: the clip setting and a lossless alternative.
CODEC_ARGS = {
    "aac": (".m4a", ["-c:a", "aac", "-b:a", "96k"]),
    "flac": (".flac", ["-c:a", "flac"]),
}


def _encode(wav: Path, out: Path, codec_args: list[str]) -> Path:
    import subprocess

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", str(SAMPLE_RATE), *codec_args,
         str(out)],
        check=True,
    )  # fmt: skip
    return out


def query_codec_check(
    queries: Sequence[tuple[str, str]],
    *,
    tts_url: str,
    work_dir: Path,
    transcribers: dict[str, Callable[[np.ndarray], str]],
) -> dict[str, object]:
    """ASR error of spoken queries as synthesised (16 kHz WAV), and after AAC and FLAC encoding.

    ``queries`` are ``(lang, text)``; ``transcribers`` map a language to an ASR
    callable over 16 kHz mono float32. Error is the Seed-TTS WER protocol
    (character level for zh). Queries failing the TTS round-trip gate are counted.
    """
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import process_one_official

    rows = []
    rejected = 0
    for lang, text in queries:
        transcribe = transcribers[lang]
        try:
            spoken = synthesise(text, lang, base_url=tts_url, cache_dir=work_dir / "tts",
                                transcribe=lambda path, t=transcribe: t(read_audio_f32(path)))  # fmt: skip
        except ValueError:
            rejected += 1
            continue
        wav = Path(spoken.wav)
        row: dict[str, object] = {"lang": lang, "text": text}
        for name, path in [("wav", wav)] + [
            (codec, _encode(wav, work_dir / "encoded" / f"{wav.stem}{suffix}", args))
            for codec, (suffix, args) in CODEC_ARGS.items()
        ]:
            heard = transcribe(read_audio_f32(path))
            row[f"{name}_asr"] = heard
            row[f"{name}_err"] = round(process_one_official(heard, text, lang)[0], 4)
        rows.append(row)
    summary: dict[str, object] = {"queries": len(queries), "tts_round_trip_rejected": rejected}
    for lang in sorted({r["lang"] for r in rows}):
        typed = [r for r in rows if r["lang"] == lang]
        summary[lang] = {
            "n": len(typed),
            **{
                f"{name}_mean_err": round(statistics.fmean(r[f"{name}_err"] for r in typed), 4)
                for name in ("wav", *CODEC_ARGS)
            },  # fmt: skip
            "aac_worse_than_wav": sum(1 for r in typed if r["aac_err"] > r["wav_err"]),
            "aac_better_than_wav": sum(1 for r in typed if r["aac_err"] < r["wav_err"]),
            "flac_differs_from_wav": sum(1 for r in typed if r["flac_asr"] != r["wav_asr"]),
        }
    summary["rows"] = rows
    return summary


def omniinteract_handcheck_sample(
    rows: Sequence[OnsetRow], *, n: int, exclude_clips: set[str], seed: int = 0
) -> list[OnsetRow]:
    """``n`` realtime, non-interrupted items with room for ``oi.EVIDENCE_LOOKBACK_S``, at most one per clip."""
    import random

    eligible = [
        r for r in rows
        if r.question_type == "realtime" and r.status == "ok" and not r.is_interrupted
        and r.pre_onset_room_s is not None and r.pre_onset_room_s >= oi.EVIDENCE_LOOKBACK_S
        and r.clip not in exclude_clips
    ]  # fmt: skip
    rng = random.Random(seed)
    rng.shuffle(eligible)
    picked: dict[str, OnsetRow] = {}
    for row in eligible:
        picked.setdefault(row.clip, row)
    return sorted(list(picked.values())[:n], key=lambda r: (r.clip, r.qa_index))


def contact_sheet(clip: Path, chunks: int, out: Path) -> Path:
    """One JPEG tiling every frame of a 1 fps clip, 4 per row, 320 px wide each."""
    import subprocess

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = (chunks + 3) // 4
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-vf", f"scale=320:-2,tile=4x{rows}:padding=4",
         "-frames:v", "1", "-q:v", "3", str(out)],
        check=True,
    )  # fmt: skip
    return out


def handcheck_page(entries: Sequence[dict[str, object]], out: Path) -> Path:
    """A local HTML index of evidence contact sheets with question, answer and the evidence audio's ASR."""
    import html

    cards = []
    for e in entries:
        cards.append(
            f"<section><h3>{html.escape(str(e['item_id']))}</h3>"
            f"<p><b>Q</b> {html.escape(str(e['question']))}<br><b>A</b> {html.escape(str(e['answer']))}<br>"
            f"<b>Evidence</b> {html.escape(str(e['evidence']))}<br><b>Audio ASR</b> {html.escape(str(e['asr']))}</p>"
            f"<img src='{html.escape(str(e['sheet']))}' loading='lazy'>"
            "<p>visible? <input type='checkbox'> audible? <input type='checkbox'></p></section>"
        )
    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>av_fd_longmem M0 evidence hand check</title>"
        "<style>body{font-family:sans-serif;max-width:1340px;margin:auto}section{border-top:1px solid #ccc;"
        "padding:8px 0}img{max-width:100%}</style>" + "".join(cards),
        encoding="utf-8",
    )
    return out


def _handcheck_entry(
    item_id: str, question: str, answer: str, evidence: str, clip: Path, chunks: int, asr_text: str, out_dir: Path
) -> dict[str, object]:
    sheet = contact_sheet(clip, chunks, out_dir / "sheets" / f"{item_id}.jpg")
    return {
        "item_id": item_id,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "asr": asr_text,
        "sheet": str(sheet.relative_to(out_dir)),
    }


def build_handcheck(dataset_root: Path, omniinteract_root: Path, asr: FunasrSpeechSegmenter, *, n: int) -> Path:
    """The M0 hand-check page under ``dataset_root/audit_handcheck``.

    OmniInteract: ``n`` sampled items (``omniinteract_handcheck_sample``, pilot
    clips excluded), evidence cut with ``oi.EVIDENCE_LOOKBACK_S``. StreamArena:
    every evidence clip listed by the ``audit-streamarena`` video reports.
    Each entry has a contact sheet and the Paraformer transcript of the
    evidence audio (zh only).
    """
    out_dir = dataset_root / "audit_handcheck"
    pilot_spec = json.loads((Path(__file__).parent / "pilot" / "omniinteract_pilot.json").read_text(encoding="utf-8"))
    pilot_clips = {str(entry["clip"]) for entry in pilot_spec["items"]}
    onsets_path = dataset_root / "omniinteract" / "audit" / "1q1a" / "onsets.jsonl"
    onsets = [OnsetRow(**json.loads(line)) for line in onsets_path.read_text(encoding="utf-8").splitlines() if line]
    entries = []
    for row in omniinteract_handcheck_sample(onsets, n=n, exclude_clips=pilot_clips):
        start, chunks = oi.evidence_window(float(row.onset_s), lookback_s=oi.EVIDENCE_LOOKBACK_S)
        end = start + chunks
        item_id = f"oi-1q1a-{row.clip}-{row.qa_index}"
        video = oi.video_path(omniinteract_root, "1q1a", row.clip)
        clip = materialise.cut_source_clip(video, start, chunks, out_dir / "clips" / f"{item_id}.mp4")
        answer = oi.annotated_question(omniinteract_root, "1q1a", row.clip, row.qa_index).answer_text
        evidence = f"source s {start}-{end} (onset {row.onset_s}); frame k is source s {start} + k + 0.5"
        asr_text = asr.transcribe_samples(read_audio_f32(clip))
        entries.append(
            _handcheck_entry(item_id, row.question_text, answer, evidence, clip, chunks, asr_text, out_dir)
        )
    for report_path in sorted((dataset_root / "streamarena" / "audit").glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for item in report.get("items", []):
            for k, (rel, (start, end)) in enumerate(zip(item["clips"], item["windows"], strict=True)):
                clip = dataset_root / rel
                evidence = f"source s {start}-{end}; points {item['evidence_ts_sec']}; hint {item['evidence_hint']}"
                asr_text = asr.transcribe_samples(read_audio_f32(clip)) if item["lang"] == "zh" else "(not transcribed)"
                entries.append(
                    _handcheck_entry(
                        f"{item['item_id']}__e{k}", item["question"], item["answer"], evidence, clip, end - start,
                        asr_text, out_dir,
                    )
                )  # fmt: skip
    return handcheck_page(entries, out_dir / "index.html")
