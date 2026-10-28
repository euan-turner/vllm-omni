# SPDX-License-Identifier: Apache-2.0
"""Analyse a long-context duplex run: tick latency vs context, deadline misses, quality vs depth.

Purpose: prototype analysis for branch long-context-prototype. Joins the
server's Stage-0 tick records (``ticks_stage0_pid*.jsonl``) with the sessions
of a ``run_long_session.py`` run, and writes ``analysis/``:

- ``ticks.csv``: one row per complete tick (metrics from ``tick_metrics.py``),
  with its session and segment.
- ``summary.json``: per-context-bin percentiles by tick kind, linear fits of
  service time vs context, overflow counts and, if scored, quality vs depth.
- ``*.png`` plots when matplotlib is installed.

Ticks are matched to sessions by order of first arrival: tick records whose
first tick arrived outside every session's [start, end] window (warm-up
sessions, other clients) are dropped. Tick ``k`` of a session is input unit
``k``, i.e. stream media time ``[k, k+1)`` s.

External dependencies: stdlib; matplotlib (optional, plots); transformers
(optional, ``--model`` to resolve the listen token id).

Usage:
    python benchmarks/duplex_session/long_context/analyze.py RUN_DIR --tick-trace-dir TRACE_DIR \\
        [--model openbmb/MiniCPM-o-4_5] [--baseline-score-root SCORES/RTD_Omni]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tick_metrics import (  # noqa: E402
    TICK_DEADLINE_S,
    Tick,
    compute_ticks,
    linear_fit,
    percentile,
    read_records,
    summarize_by_context,
)


def load_sessions(run_dir: Path) -> list[dict[str, Any]]:
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_dir.glob("session*/manifest.json"))]
    return sorted(sessions, key=lambda manifest: manifest["t_start_monotonic"])


def match_requests_to_sessions(records: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> dict[str, int]:
    """request_id -> session_index, by first-tick arrival inside a session's window, in start order."""
    first_arrival = {
        record["request_id"]: float(record["t_arrival"]) for record in records if record.get("tick_index") == 0
    }
    candidates = sorted(
        (arrival, rid)
        for rid, arrival in first_arrival.items()
        if any(s["t_start_monotonic"] <= arrival <= s["t_end_monotonic"] for s in sessions)
    )
    if len(candidates) != len(sessions):
        print(
            f"warning: {len(candidates)} traced requests in session windows for {len(sessions)} sessions; "
            "matching in start order",
            file=sys.stderr,
        )
    return {rid: session["session_index"] for (_, rid), session in zip(candidates, sessions, strict=False)}


def segment_at(segments: list[dict[str, Any]], media_s: float) -> dict[str, Any] | None:
    for segment in segments:
        if segment["offset_s"] <= media_s < segment["offset_s"] + segment["duration_s"]:
            return segment
    return None


def resolve_listen_token_id(model: str | None, explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    if not model:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    token_id = tokenizer.convert_tokens_to_ids("<|listen|>")
    return int(token_id) if isinstance(token_id, int) and token_id != tokenizer.unk_token_id else None


def segment_score(score: dict[str, Any]) -> dict[str, float]:
    content = score.get("content") or {}
    temporal = (score.get("temporal") or {}).get("summary") or {}
    return {
        "content_score": float(content.get("content_score", math.nan)),
        "temporal_score": float(temporal.get("avg_temporal_score", math.nan)),
    }


def quality_vs_depth(run_dir: Path, sessions: list[dict[str, Any]], baseline_root: Path | None) -> list[dict[str, Any]]:
    """One row per scored segment: its depth (stream offset) and score, minus the sample's depth-0 baseline."""
    rows = []
    for session in sessions:
        score_dir = run_dir / f"session{session['session_index']}" / "scores"
        for segment in session["segments"]:
            path = score_dir / f"{segment['index']:04d}_{segment['sample_id']}.json"
            if not path.exists():
                continue
            row = {
                "session_index": session["session_index"],
                "segment_index": segment["index"],
                "sample_id": segment["sample_id"],
                "depth_s": segment["offset_s"],
                **segment_score(json.loads(path.read_text(encoding="utf-8"))),
            }
            baseline = baseline_root / f"{segment['sample_id']}.json" if baseline_root else None
            if baseline is not None and baseline.exists():
                base = segment_score(json.loads(baseline.read_text(encoding="utf-8")))
                row["content_delta"] = row["content_score"] - base["content_score"]
                row["temporal_delta"] = row["temporal_score"] - base["temporal_score"]
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(out_dir: Path, ticks: list[Tick], tick_rows: list[dict[str, Any]], quality: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots", file=sys.stderr)
        return
    steady = [tick for tick in ticks if tick.tick_index > 0]
    colors = {"listen": "#4C78A8", "speak": "#F58518"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, color in colors.items():
        members = [tick for tick in steady if tick.kind == kind]
        ax.scatter(
            [t.context_tokens for t in members], [t.service_s for t in members], s=4, alpha=0.4, color=color, label=kind
        )
    ax.axhline(TICK_DEADLINE_S, color="#B0413E", linestyle="--", linewidth=1, label="tick deadline")
    ax.axvline(40960, color="#888888", linestyle=":", linewidth=1, label="HF max context")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("tick service time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "service_vs_context.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, color in colors.items():
        members = [tick for tick in steady if tick.kind == kind]
        ax.scatter(
            [t.context_tokens for t in members],
            [t.prefill_s for t in members],
            s=3,
            alpha=0.3,
            color=color,
            marker="o",
            label=f"{kind} prefill",
        )
        ax.scatter(
            [t.context_tokens for t in members],
            [t.decode_s for t in members],
            s=3,
            alpha=0.3,
            color=color,
            marker="x",
            label=f"{kind} decode",
        )
    ax.set_xlabel("context tokens")
    ax.set_ylabel("seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prefill_decode_vs_context.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for session_index in sorted({row["session_index"] for row in tick_rows}):
        rows = [row for row in tick_rows if row["session_index"] == session_index]
        ax.plot(
            [row["tick_index"] for row in rows],
            [row["latency_s"] for row in rows],
            linewidth=1,
            label=f"session {session_index} latency",
        )
        ax.plot(
            [row["tick_index"] for row in rows],
            [row["queue_s"] for row in rows],
            linewidth=1,
            linestyle=":",
            label=f"session {session_index} queue",
        )
    ax.axhline(TICK_DEADLINE_S, color="#B0413E", linestyle="--", linewidth=1, label="tick deadline")
    ax.set_xlabel("tick index")
    ax.set_ylabel("seconds (arrival to end / to first step)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_queue_by_tick.png", dpi=150)
    plt.close(fig)

    if quality:
        key = "content_delta" if any("content_delta" in row for row in quality) else "content_score"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([row["depth_s"] / 60 for row in quality], [row.get(key, math.nan) for row in quality], s=10)
        ax.set_xlabel("probe depth (minutes of preceding stream)")
        ax.set_ylabel(key)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_vs_depth.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tick-trace-dir", type=Path, help="defaults to run.json tick_trace_dir")
    parser.add_argument("--model", help="tokenizer to resolve <|listen|>; otherwise any 1-token tick is listen")
    parser.add_argument("--listen-token-id", type=int)
    parser.add_argument("--baseline-score-root", type=Path, help="depth-0 omni-duplex-eval scores for the split")
    parser.add_argument("--bin-tokens", type=int, default=2048)
    args = parser.parse_args(argv)

    run_meta = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    trace_dir = args.tick_trace_dir or (Path(run_meta["tick_trace_dir"]) if run_meta.get("tick_trace_dir") else None)
    if trace_dir is None:
        parser.error("--tick-trace-dir is required when run.json has no tick_trace_dir")
    records = read_records(sorted(trace_dir.glob("ticks_stage0_pid*.jsonl")))
    sessions = load_sessions(args.run_dir)
    owner = match_requests_to_sessions(records, sessions)
    records = [record for record in records if record["request_id"] in owner]
    ticks = compute_ticks(records, listen_token_id=resolve_listen_token_id(args.model, args.listen_token_id))

    by_index = {session["session_index"]: session for session in sessions}
    tick_rows = []
    for tick in ticks:
        session_index = owner[tick.request_id]
        segment = segment_at(by_index[session_index]["segments"], float(tick.tick_index))
        tick_rows.append(
            {
                "session_index": session_index,
                "segment_index": segment["index"] if segment else None,
                "sample_id": segment["sample_id"] if segment else None,
                **tick.as_dict(),
            }
        )

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "ticks.csv", tick_rows)
    quality = quality_vs_depth(args.run_dir, sessions, args.baseline_score_root)
    write_csv(out_dir / "quality_vs_depth.csv", quality)
    overflow = [record for record in records if str(record.get("status", "")).startswith("overflow")]
    steady = [tick for tick in ticks if tick.tick_index > 0]
    summary = {
        "modality": run_meta.get("modality"),
        "tick_deadline_s": TICK_DEADLINE_S,
        "num_ticks": len(ticks),
        "max_context_tokens": max((tick.context_tokens for tick in ticks), default=0),
        "deadline_miss_rate": (sum(t.deadline_miss for t in steady) / len(steady)) if steady else None,
        "overflow_records": len(overflow),
        "queue_p99_s": percentile([t.queue_s for t in steady], 0.99) if steady else None,
        "suspected_continuation_ticks": sum(t.suspected_continuation for t in ticks),
        "by_context": summarize_by_context(ticks, bin_tokens=args.bin_tokens),
        "linear_fit": {kind: linear_fit(ticks, kind=kind) for kind in ("listen", "speak")},
        "quality": {
            "scored_segments": len(quality),
            "content_score_mean": statistics.fmean(r["content_score"] for r in quality) if quality else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot(out_dir, ticks, tick_rows, quality)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_context"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
