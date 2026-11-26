# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Label and score every session of a run, and write a per-probe summary.

For each session of ``runs/<run_id>``: evidence residency at query onset from
the copied tick trace (``labels.jsonl``), and a key-token verdict on two answer
channels (``verdicts.jsonl``): the model's output text (``wav_transcript.json``)
and ASR of ``output.wav`` over each answer window (Paraformer-zh for zh items,
Whisper-large-v3 for en items). ``summary.md``
lists one row per probe, with the plan kind. Sessions with an epoch change or a
rejected trace are reported and skipped.

With an ``llm_match`` callable, each non-silent reply also gets an LLM verdict
(``annotate.answers_match``). From M4 that verdict is the primary one and the
key-token verdict is secondary (decision 2026-11-26): ``summary.md`` lists the
LLM verdict first and every channel where the two disagree, for review by ear.
It is still uncalibrated (one call, no abstained/hallucinated split); the plan's
judge (M5) replaces it before the M6 benchmark runs. Without ``llm_match`` (the
pilot) the key-token verdict is the primary one. Key-token matching is never the
benchmark's final score; from M6 it is only a column that flags judge disagreements.

External dependencies: funasr and transformers Whisper (ASR channel); zhconv.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from benchmarks.av_fd_longmem.asr import SAMPLE_RATE, BilingualTranscriber, read_audio_f32
from benchmarks.av_fd_longmem.residency import TraceRejectedError, load_session_trace, probe_residency
from benchmarks.av_fd_longmem.schema import Item, SessionPlan, read_json, read_jsonl
from benchmarks.av_fd_longmem.score import TimedText, judge_probe, transcript_chunks, verdict_rows


def _plan_for(session: dict[str, object], dataset_root: Path) -> SessionPlan:
    recorded = session.get("plan_path")
    if isinstance(recorded, str) and Path(recorded).exists():
        return read_json(Path(recorded), SessionPlan)
    matches = sorted((dataset_root / "plans").glob(f"*/{session['plan_id']}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"plan {session['plan_id']} not found under {dataset_root / 'plans'}")
    return read_json(matches[0], SessionPlan)


def _items(dataset_root: Path) -> dict[str, Item]:
    """Every item of every source and item set; an id defined twice is an error."""
    items: dict[str, Item] = {}
    for path in sorted([*dataset_root.glob("*/items.jsonl"), *dataset_root.glob("*/sets/*/items.jsonl")]):
        for item in read_jsonl(path, Item):
            if item.item_id in items:
                raise ValueError(f"item {item.item_id} is defined twice (again in {path})")
            items[item.item_id] = item
    return items


#: (item, reply text) -> whether the reply gives the item's answer.
LlmMatch = Callable[[Item, str], bool]


def evaluate_run(
    dataset_root: Path, run_id: str, *, asr: BilingualTranscriber, llm_match: LlmMatch | None = None
) -> list[dict[str, object]]:
    run_dir = dataset_root / "runs" / run_id
    items = _items(dataset_root)
    labels: list[dict[str, object]] = []
    verdicts: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for session_path in sorted(run_dir.glob("sessions/*/session.json")):
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session_dir = session_path.parent
        if not session["success"] or session["epoch_changed"]:
            rows.append(
                {"plan_id": session["plan_id"], "skipped": "failed" if not session["success"] else "epoch_changed"}
            )
            continue
        plan = _plan_for(session, dataset_root)
        try:
            records = load_session_trace(run_dir / "trace", str(session["session_id"]))
        except TraceRejectedError as exc:
            rows.append({"plan_id": plan.plan_id, "skipped": f"trace rejected: {exc}"})
            continue
        text_chunks = transcript_chunks(json.loads((session_dir / "wav_transcript.json").read_text(encoding="utf-8")))
        output_audio = read_audio_f32(session_dir / "output.wav")
        for probe in plan.probes:
            item = items[probe.item_id]
            residency = probe_residency(probe, records)
            if plan.kind == "control_blind":  # its evidence span plays filler: residency does not apply
                residency = replace(residency, label="n/a")
            window = (probe.query_onset_chunk, probe.answer_window_chunks[1] + 1)
            heard = asr(item.lang, output_audio[window[0] * SAMPLE_RATE : window[1] * SAMPLE_RATE])
            asr_chunks = [TimedText(float(window[0]), float(window[1]), heard)] if heard else []
            by_text = judge_probe(item, probe, text_chunks, channel="transcript")
            by_audio = judge_probe(item, probe, asr_chunks, channel="asr")
            label = {"run_id": run_id, "plan_id": plan.plan_id, "gap_chunks": probe.gap_chunks, **residency.__dict__}
            labels.append(label)
            llm = {
                verdict.channel: (llm_match(item, verdict.reply) if llm_match and verdict.reply.strip() else None)
                for verdict in (by_text, by_audio)
            }
            for verdict in verdict_rows([by_text, by_audio]):
                verdicts.append(
                    {
                        "run_id": run_id,
                        "plan_id": plan.plan_id,
                        "plan_kind": plan.kind,
                        "gap_chunks": probe.gap_chunks,
                        **verdict,
                        "llm_match": llm[verdict["channel"]],
                    }
                )
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "kind": plan.kind,
                    "item_id": item.item_id,
                    "gap": probe.gap_chunks,
                    "residency": residency.label,
                    "evicted_frac": residency.evicted_token_frac,
                    "transcript": by_text.verdict,
                    "asr": by_audio.verdict,
                    "llm_transcript": llm["transcript"],
                    "llm_asr": llm["asr"],
                    "rehearsed": by_text.rehearsed,
                    "reply": by_text.reply,
                    "answer": item.answer,
                }
            )
    (run_dir / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels))
    (run_dir / "verdicts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts))
    (run_dir / "summary.md").write_text(
        _summary_markdown(run_id, rows, llm_primary=llm_match is not None), encoding="utf-8"
    )
    return rows


def _llm(match: object) -> str:
    return "" if match is None else "correct" if match else "incorrect"


def _disagreements(rows: list[dict[str, object]]) -> list[str]:
    lines = []
    for row in rows:
        for channel in ("transcript", "asr"):
            llm = row.get(f"llm_{channel}")
            if llm is not None and (row[channel] == "correct") != llm:
                lines.append(
                    f"| {row['plan_id']} | {channel} | {_llm(llm)} | {row[channel]} | "
                    f"{str(row['reply']).replace('|', '/')} | {row['answer']} |"
                )
    return lines


def _summary_markdown(run_id: str, rows: list[dict[str, object]], *, llm_primary: bool) -> str:
    primary = (
        "Primary verdict: LLM match (uncalibrated). Key-token verdicts are secondary."
        if llm_primary
        else "Primary verdict: key-token (no LLM match was run)."
    )
    lines = [
        f"# {run_id}",
        "",
        primary,
        "",
        "| plan | kind | item | gap | residency | evicted frac | llm (transcript / asr) | key-token (transcript / asr) "
        "| rehearsed | reply | answer |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "skipped" in row:
            lines.append(f"| {row['plan_id']} | skipped: {row['skipped']} ||||||||||")
            continue
        frac = row["evicted_frac"]
        lines.append(
            f"| {row['plan_id']} | {row['kind']} | {row['item_id']} | {row['gap']} | {row['residency']} | "
            f"{'' if frac is None else f'{frac:.2f}'} | {_llm(row['llm_transcript'])} / {_llm(row['llm_asr'])} | "
            f"{row['transcript']} / {row['asr']} | {row['rehearsed']} | "
            f"{str(row['reply']).replace('|', '/')} | {row['answer']} |"
        )
    disagreements = _disagreements([row for row in rows if "skipped" not in row])
    if disagreements:
        lines += [
            "",
            "## Scorer disagreements (listen to these)",
            "",
            "| plan | channel | llm | key-token | reply | answer |",
            "|---|---|---|---|---|---|",
            *disagreements,
        ]
    return "\n".join(lines) + "\n"
