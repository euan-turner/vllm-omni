#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-tick listen/speak latency and slack from duplex tick-trace files.

Reads the ``duplex-ticks-*.jsonl`` files written when a server runs with
``VLLM_OMNI_DUPLEX_TICK_TRACE_DIR`` set (record schema in
``vllm_omni/engine/duplex/tick_trace.py``) and writes ``ticks.csv`` and
``summary.csv``, binned by Stage-0 context like llama.cpp-omni's perf-duplex
sweep summary.

A tick is one accepted Stage-0 append. A listen tick's segment is a single
Stage-0 output (the listen token); a speak tick streams several. Its decision
is the first Stage-0 output of the matching segment (segments are counted per request, closed by
``segment_finished``); a request whose segment count differs from its append
count is reported and excluded rather than guessed. Latency starts at server
ingress of the ``AppendAudio`` that triggered the append (the append's own
start time for silence continuations); slack is the tick period minus latency.

A speak tick whose Talker handoffs (``h`` records between its first output and
the next segment's) all carry zero hidden rows is ``speak_empty``: the model
chose to speak but emitted no words, so no audio can follow. It gets no first
audio, keeping that model behaviour out of the system first-audio latency.

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: standard library only; imports nothing from the server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

TICK_PERIOD_MS = 1000.0
CONTEXT_BIN = 4096
_NS_PER_MS = 1e6


@dataclass
class Tick:
    session_id: str
    request_id: str
    epoch: int | None
    seq: int | None
    kind: str
    context: int | None
    window_replaced: bool
    t_in_ms: float
    mailbox_ms: float | None
    accept_ms: float
    decision_ms: float
    segment_end_ms: float
    first_audio_ms: float | None
    stage0_outputs: int
    handoff_rows: int | None

    @property
    def decision_latency_ms(self) -> float:
        return self.decision_ms - self.t_in_ms

    @property
    def audio_latency_ms(self) -> float | None:
        return None if self.first_audio_ms is None else self.first_audio_ms - self.t_in_ms


@dataclass
class TraceRecords:
    ingress: dict[tuple[str, int], int] = field(default_factory=dict)
    dequeue: dict[tuple[str, int], int] = field(default_factory=dict)
    starts: dict[tuple[str, int], tuple[int, int, bool]] = field(default_factory=dict)
    accepts: dict[tuple[str, int], tuple[int, int | None, int | None, str | None]] = field(default_factory=dict)
    outputs: list[tuple[int, str, int, str, bool, int]] = field(default_factory=list)
    prompts: dict[tuple[str, int | None, int | None], tuple[int | None, bool]] = field(default_factory=dict)
    handoffs: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    dropped: int = 0


@dataclass
class JoinResult:
    ticks: list[Tick]
    mismatches: list[str]


def load_records(paths: Iterable[Path]) -> TraceRecords:
    """Parse trace files; the first Stage-0 prompt record per ``(session, epoch, seq)`` wins."""
    records = TraceRecords()
    prompt_times: dict[tuple[str, int | None, int | None], int] = {}
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                kind = row[0]
                if kind == "trace_dropped":
                    records.dropped += int(row[1])
                elif kind == "in":
                    records.ingress[(row[2], row[3])] = row[1]
                elif kind == "dq":
                    records.dequeue[(row[2], row[3])] = row[1]
                elif kind == "start":
                    records.starts[(row[2], row[3])] = (row[1], row[4], bool(row[5]))
                elif kind == "acc":
                    records.accepts[(row[2], row[3])] = (row[1], row[4], row[5], row[6])
                elif kind == "out":
                    records.outputs.append((row[1], row[2], row[3], row[4], bool(row[5]), int(row[6])))
                elif kind == "w":
                    key = (row[2], row[3], row[4])
                    if key not in prompt_times or row[1] < prompt_times[key]:
                        prompt_times[key] = row[1]
                        records.prompts[key] = (row[5], bool(row[8]))
                elif kind == "h":
                    records.handoffs[row[2]].append((row[1], int(row[5])))
    records.outputs.sort()
    for handoffs in records.handoffs.values():
        handoffs.sort()
    return records


def _segments(outputs: Sequence[tuple[int, str, int, str, bool, int]]) -> list[list[tuple[int, int]]]:
    """Stage-0 outputs of one request split at ``segment_finished`` (an unfinished tail is dropped)."""
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for t, _session, _stage, _request, finished, n_tokens in outputs:
        current.append((t, n_tokens))
        if finished:
            segments.append(current)
            current = []
    return segments


def _handoff_rows(handoffs: Sequence[tuple[int, int]], start_t: int, end_t: float) -> int | None:
    """Hidden rows handed to the Talker in ``[start_t, end_t)``; ``None`` when no handoff was recorded."""
    rows = [hidden_rows for t, hidden_rows in handoffs if start_t <= t < end_t]
    return sum(rows) if rows else None


def join_ticks(records: TraceRecords) -> JoinResult:
    """Attribute each accepted Stage-0 append to its segment and first audio."""
    stage0_by_request: dict[str, list[tuple[int, str, int, str, bool, int]]] = defaultdict(list)
    final_by_session: dict[str, list[int]] = defaultdict(list)
    final_stage = max((stage for _, _, stage, _, _, _ in records.outputs), default=0)
    for output in records.outputs:
        t, session_id, stage_id, request_id, _, _ = output
        if stage_id == 0:
            stage0_by_request[request_id].append(output)
        elif stage_id == final_stage:
            final_by_session[session_id].append(t)

    appends_by_request: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, (_t, _epoch, _seq, request_id) in sorted(records.accepts.items(), key=lambda item: item[1][0]):
        if request_id is not None:
            appends_by_request[request_id].append(key)

    ticks: list[Tick] = []
    mismatches: list[str] = []
    for request_id, append_keys in appends_by_request.items():
        segments = _segments(stage0_by_request.get(request_id, ()))
        if len(segments) != len(append_keys):
            mismatches.append(f"{request_id}: {len(append_keys)} appends vs {len(segments)} finished segments")
            continue
        handoffs = records.handoffs.get(request_id, ())
        segment_ends = [next_segment[0][0] for next_segment in segments[1:]] + [math.inf]
        for append_key, segment, segment_end_t in zip(append_keys, segments, segment_ends, strict=True):
            session_id, attempt = append_key
            accept_t, epoch, seq, _ = records.accepts[append_key]
            start_t, dq_n, silence = records.starts.get(append_key, (accept_t, 0, False))
            ingress_t = None if silence else records.ingress.get((session_id, dq_n))
            dequeue_t = None if silence else records.dequeue.get((session_id, dq_n))
            t_in = ingress_t if ingress_t is not None else start_t
            decision_t = segment[0][0]
            listen = len(segment) == 1
            handoff_rows = None if listen else _handoff_rows(handoffs, decision_t, segment_end_t)
            if silence:
                kind = "silence"
            elif listen:
                kind = "listen"
            else:
                kind = "speak_empty" if handoff_rows == 0 else "speak"
            first_audio = None
            if not listen and kind != "speak_empty":
                first_audio = next((t for t in final_by_session.get(session_id, ()) if t >= decision_t), None)
            context, window_replaced = records.prompts.get((session_id, epoch, seq), (None, False))
            ticks.append(
                Tick(
                    session_id=session_id,
                    request_id=request_id,
                    epoch=epoch,
                    seq=seq,
                    kind=kind,
                    context=context,
                    window_replaced=window_replaced,
                    t_in_ms=t_in / _NS_PER_MS,
                    mailbox_ms=None if ingress_t is None or dequeue_t is None else (dequeue_t - ingress_t) / _NS_PER_MS,
                    accept_ms=accept_t / _NS_PER_MS,
                    decision_ms=decision_t / _NS_PER_MS,
                    segment_end_ms=segment[-1][0] / _NS_PER_MS,
                    first_audio_ms=None if first_audio is None else first_audio / _NS_PER_MS,
                    stage0_outputs=len(segment),
                    handoff_rows=handoff_rows,
                )
            )
    ticks.sort(key=lambda tick: tick.t_in_ms)
    return JoinResult(ticks=ticks, mismatches=mismatches)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summarize(ticks: Sequence[Tick], *, period_ms: float = TICK_PERIOD_MS, bin_size: int = CONTEXT_BIN) -> list[dict]:
    """Per kind and context bin: tick count, latency percentiles, slack and on-time fraction."""
    groups: dict[tuple[str, int], list[Tick]] = defaultdict(list)
    for tick in ticks:
        if tick.context is None:
            continue
        groups[(tick.kind, tick.context // bin_size * bin_size)].append(tick)
    rows = []
    for (kind, lo), group in sorted(groups.items()):
        latency = [tick.decision_latency_ms for tick in group]
        audio = [value for tick in group if (value := tick.audio_latency_ms) is not None]
        mailbox = [tick.mailbox_ms for tick in group if tick.mailbox_ms is not None]
        rows.append(
            {
                "kind": kind,
                "ctx_bin_start": lo,
                "ctx_bin_end": lo + bin_size,
                "n_ticks": len(group),
                "decision_p50_ms": _percentile(latency, 0.5),
                "decision_p95_ms": _percentile(latency, 0.95),
                "decision_max_ms": max(latency),
                "first_audio_p50_ms": _percentile(audio, 0.5),
                "first_audio_p95_ms": _percentile(audio, 0.95),
                "first_audio_max_ms": max(audio, default=math.nan),
                "slack_p5_ms": period_ms - _percentile(latency, 0.95),
                "slack_min_ms": period_ms - max(latency),
                "on_time_frac": sum(value < period_ms for value in latency) / len(group),
                "mailbox_p95_ms": _percentile(mailbox, 0.95),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", type=Path, help="Directory given as VLLM_OMNI_DUPLEX_TICK_TRACE_DIR.")
    parser.add_argument(
        "--output-dir", type=Path, help="Where to write ticks.csv and summary.csv (default: trace_dir)."
    )
    args = parser.parse_args(argv)

    paths = sorted(args.trace_dir.glob("duplex-ticks-*.jsonl"))
    if not paths:
        print(f"no duplex-ticks-*.jsonl files in {args.trace_dir}", file=sys.stderr)
        return 1
    records = load_records(paths)
    result = join_ticks(records)
    output_dir = args.output_dir or args.trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tick_rows = [
        {**asdict(tick), "decision_latency_ms": tick.decision_latency_ms, "audio_latency_ms": tick.audio_latency_ms}
        for tick in result.ticks
    ]
    _write_csv(output_dir / "ticks.csv", tick_rows)
    summary = summarize(result.ticks)
    _write_csv(output_dir / "summary.csv", summary)

    print(f"{len(result.ticks)} ticks from {len(paths)} files; dropped records: {records.dropped}")
    for mismatch in result.mismatches:
        print(f"EXCLUDED {mismatch}", file=sys.stderr)
    print(
        f"{'kind':11} {'ctx bin':>13} {'n':>5} {'dec50':>6} {'dec95':>6} {'decmax':>7} "
        f"{'aud50':>6} {'aud95':>6} {'slack5':>7} {'ontime':>6}"
    )
    for row in summary:
        print(
            f"{row['kind']:11} {row['ctx_bin_start']:>6}-{row['ctx_bin_end']:<6} {row['n_ticks']:>5} "
            f"{row['decision_p50_ms']:6.0f} {row['decision_p95_ms']:6.0f} {row['decision_max_ms']:7.0f} "
            f"{row['first_audio_p50_ms']:6.0f} {row['first_audio_p95_ms']:6.0f} {row['slack_p5_ms']:7.0f} "
            f"{row['on_time_frac']:6.2f}"
        )
    return 2 if result.mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
