# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Label and score every session of a run, and write a per-probe summary.

For each session of ``runs/<run_id>``: evidence residency at query onset from
the copied tick trace (``labels.jsonl``), and a key-token verdict on two answer
channels (``verdicts.jsonl``): the model's output text (``wav_transcript.json``)
and Paraformer ASR of ``output.wav`` over each answer window. ``summary.md``
lists one row per probe. Sessions with an epoch change or a rejected trace are
reported and skipped.

External dependencies: funasr (ASR channel); zhconv.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.av_fd_longmem.asr import SAMPLE_RATE, FunasrSpeechSegmenter, read_audio_f32
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
    items: dict[str, Item] = {}
    for path in dataset_root.glob("*/items.jsonl"):
        items.update({item.item_id: item for item in read_jsonl(path, Item)})
    return items


def evaluate_run(dataset_root: Path, run_id: str, *, asr: FunasrSpeechSegmenter) -> list[dict[str, object]]:
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
            window = (probe.query_onset_chunk, probe.answer_window_chunks[1] + 1)
            heard = asr.transcribe_samples(output_audio[window[0] * SAMPLE_RATE : window[1] * SAMPLE_RATE])
            asr_chunks = [TimedText(float(window[0]), float(window[1]), heard)] if heard else []
            by_text = judge_probe(item, probe, text_chunks, channel="transcript")
            by_audio = judge_probe(item, probe, asr_chunks, channel="asr")
            label = {"run_id": run_id, "plan_id": plan.plan_id, "gap_chunks": probe.gap_chunks, **residency.__dict__}
            labels.append(label)
            for verdict in verdict_rows([by_text, by_audio]):
                verdicts.append({"run_id": run_id, "plan_id": plan.plan_id, "gap_chunks": probe.gap_chunks, **verdict})
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "item_id": item.item_id,
                    "gap": probe.gap_chunks,
                    "residency": residency.label,
                    "evicted_frac": residency.evicted_token_frac,
                    "transcript": by_text.verdict,
                    "asr": by_audio.verdict,
                    "rehearsed": by_text.rehearsed,
                    "reply": by_text.reply,
                    "answer": item.answer,
                }
            )
    (run_dir / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels))
    (run_dir / "verdicts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts))
    (run_dir / "summary.md").write_text(_summary_markdown(run_id, rows), encoding="utf-8")
    return rows


def _summary_markdown(run_id: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"# {run_id}",
        "",
        "| plan | item | gap | residency | evicted frac | transcript | asr | rehearsed | reply | answer |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "skipped" in row:
            lines.append(f"| {row['plan_id']} | skipped: {row['skipped']} |||||||||")
            continue
        frac = row["evicted_frac"]
        lines.append(
            f"| {row['plan_id']} | {row['item_id']} | {row['gap']} | {row['residency']} | "
            f"{'' if frac is None else f'{frac:.2f}'} | {row['transcript']} | {row['asr']} | {row['rehearsed']} | "
            f"{str(row['reply']).replace('|', '/')} | {row['answer']} |"
        )
    return "\n".join(lines) + "\n"
