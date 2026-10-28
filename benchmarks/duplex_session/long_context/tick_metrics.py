# SPDX-License-Identifier: Apache-2.0
"""Tick and tick-deadline metric definitions for long-context duplex profiling.

Purpose: turn the Stage-0 tick records written by
``vllm_omni/core/sched/duplex_tick_trace.py`` (one JSON line per tick) into
per-tick metrics and context-length summaries. Prototype on branch
long-context-prototype; definitions here are the ones the plots use.

Definitions (all times in seconds, ``time.monotonic`` in the EngineCore):

- **Tick**: one streaming input update applied to a session: its prefill
  plus the decode steps that follow it until the segment stops. Only records
  with ``status == "stop"`` are complete ticks. An update is either a 1 s
  client append or a 1 s silence unit the server injects to continue a
  response after the model chose to listen mid-response (see
  ``vllm_omni/engine/duplex/session/model_channel.py``). Both are real
  Stage-0 work and count as ticks, so ticks are not one per second of input.
- **context_tokens**: KV length once the tick's prefill is done
  (``context_tokens_before + prefill_tokens``).
- **service_s** = ``t_end - t_first_scheduled``: the tick's prefill and decodes.
- **prefill_s** = ``t_prefill_done - t_first_scheduled``.
- **decode_s** = ``t_end - t_prefill_done``.
- **queue_s** = ``t_first_scheduled - t_arrival``: wait between the append
  reaching the scheduler and its first scheduled step (includes waiting for
  the previous tick of the same session).
- **latency_s** = ``t_end - t_arrival``.
- **kind**: ``listen`` when the tick sampled exactly one token and it is the
  listen token (any single token when the listen id is unknown);
  ``degenerate`` when at least ``DEGENERATE_MIN_REPEATS`` (3) of the sampled
  tokens, and at least ``DEGENERATE_MIN_FRACTION`` (half) of them, are
  ``<|tts_bos|>`` (only when its id is known); otherwise ``speak``. Degenerate
  ticks are the repeated-``<|tts_bos|>`` runs seen past ~15-25k tokens, which
  produce no text and usually hit the 20-token cap; they are separated so the
  ``speak`` profile measures normal speech.
- **Tick deadline**: ``TICK_DEADLINE_S`` = 1.0, the input unit period. A tick
  misses it when ``service_s > TICK_DEADLINE_S``; ticks are serial within a
  session, so sustained misses grow the input backlog without bound.
- **suspected_continuation**: the tick arrived less than
  ``CONTINUATION_GAP_S`` (0.5 s) after the session's previous update. Client
  appends arrive one per second, so such a tick is most likely a server
  silence continuation. Heuristic: a late client append followed by an
  on-time one is flagged too; the trace does not mark continuations.
- **Backlog**: ``queue_s`` growing over a session. There is no separate lag
  metric: tick index is not input time (see Tick).
- **cuda_allocated_bytes / cuda_reserved_bytes**: Stage-0 process CUDA
  allocator state when the tick closed (None when not recorded).

External dependencies: none (stdlib only).
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TICK_DEADLINE_S = 1.0
UNIT_S = 1.0
CONTINUATION_GAP_S = 0.5 * UNIT_S
DEGENERATE_MIN_REPEATS = 3
DEGENERATE_MIN_FRACTION = 0.5
TICK_KINDS = ("listen", "speak", "degenerate")


@dataclass(frozen=True)
class Tick:
    request_id: str
    tick_index: int
    kind: str
    context_tokens: int
    prefill_tokens: int
    num_sampled_tokens: int
    num_decode_steps: int
    queue_depth_at_arrival: int
    service_s: float
    prefill_s: float
    decode_s: float
    queue_s: float
    latency_s: float
    deadline_miss: bool
    suspected_continuation: bool
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def tick_kind(
    sampled_token_ids: Sequence[int], listen_token_id: int | None, tts_bos_token_id: int | None = None
) -> str:
    if len(sampled_token_ids) == 1 and (listen_token_id is None or sampled_token_ids[0] == listen_token_id):
        return "listen"
    if tts_bos_token_id is not None:
        repeats = sum(token == tts_bos_token_id for token in sampled_token_ids)
        if repeats >= DEGENERATE_MIN_REPEATS and repeats >= DEGENERATE_MIN_FRACTION * len(sampled_token_ids):
            return "degenerate"
    return "speak"


def compute_ticks(
    records: Iterable[dict[str, Any]], *, listen_token_id: int | None = None, tts_bos_token_id: int | None = None
) -> list[Tick]:
    """Complete ticks in (request, tick_index) order; incomplete or overflowed records are skipped."""
    complete = [
        record
        for record in records
        if record.get("status") == "stop"
        and all(record.get(key) is not None for key in ("t_first_scheduled", "t_prefill_done", "t_end"))
    ]
    complete.sort(key=lambda record: (record["request_id"], record["tick_index"]))
    # Gaps use every received update, complete or not, so a dropped record
    # cannot make its successor look like a continuation.
    arrivals: dict[tuple[str, int], float] = {
        (record["request_id"], int(record["tick_index"])): float(record["t_arrival"]) for record in records
    }
    ticks = []
    for record in complete:
        rid = record["request_id"]
        index = int(record["tick_index"])
        previous = arrivals.get((rid, index - 1))
        service = record["t_end"] - record["t_first_scheduled"]
        sampled = list(record.get("sampled_token_ids") or ())
        ticks.append(
            Tick(
                request_id=rid,
                tick_index=index,
                kind=tick_kind(sampled, listen_token_id, tts_bos_token_id),
                context_tokens=int(record.get("context_tokens_before") or 0) + int(record.get("prefill_tokens") or 0),
                prefill_tokens=int(record.get("prefill_tokens") or 0),
                num_sampled_tokens=len(sampled),
                num_decode_steps=int(record.get("num_decode_steps") or 0),
                queue_depth_at_arrival=int(record.get("queue_depth_at_arrival") or 0),
                service_s=service,
                prefill_s=record["t_prefill_done"] - record["t_first_scheduled"],
                decode_s=record["t_end"] - record["t_prefill_done"],
                queue_s=record["t_first_scheduled"] - record["t_arrival"],
                latency_s=record["t_end"] - record["t_arrival"],
                deadline_miss=service > TICK_DEADLINE_S,
                suspected_continuation=previous is not None
                and float(record["t_arrival"]) - previous < CONTINUATION_GAP_S,
                cuda_allocated_bytes=record.get("cuda_allocated_bytes"),
                cuda_reserved_bytes=record.get("cuda_reserved_bytes"),
            )
        )
    return ticks


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_by_context(
    ticks: Iterable[Tick], *, bin_tokens: int = 2048, skip_first_tick: bool = True
) -> list[dict[str, Any]]:
    """Service/queue/latency percentiles and deadline-miss rate per (context bin, kind)."""
    groups: dict[tuple[int, str], list[Tick]] = {}
    for tick in ticks:
        if skip_first_tick and tick.tick_index == 0:
            continue
        groups.setdefault((tick.context_tokens // bin_tokens, tick.kind), []).append(tick)
    rows = []
    for (bin_index, kind), members in sorted(groups.items()):
        service = [tick.service_s for tick in members]
        rows.append(
            {
                "context_lo": bin_index * bin_tokens,
                "context_hi": (bin_index + 1) * bin_tokens,
                "kind": kind,
                "count": len(members),
                "service_p50_s": percentile(service, 0.5),
                "service_p90_s": percentile(service, 0.9),
                "service_p99_s": percentile(service, 0.99),
                "prefill_p50_s": percentile([tick.prefill_s for tick in members], 0.5),
                "decode_p50_s": percentile([tick.decode_s for tick in members], 0.5),
                "queue_p99_s": percentile([tick.queue_s for tick in members], 0.99),
                "latency_p99_s": percentile([tick.latency_s for tick in members], 0.99),
                "suspected_continuations": sum(tick.suspected_continuation for tick in members),
                "sampled_tokens_mean": statistics.fmean(tick.num_sampled_tokens for tick in members),
                "deadline_miss_rate": sum(tick.deadline_miss for tick in members) / len(members),
            }
        )
    return rows


def linear_fit(ticks: Iterable[Tick], *, kind: str, skip_first_tick: bool = True) -> dict[str, float] | None:
    """Least-squares ``service_s ~ a + b * context_tokens`` for one tick kind.

    ``deadline_context_tokens`` extrapolates the context at which the fitted
    service time reaches the tick deadline (``inf`` when the slope is not positive).
    """
    points = [
        (tick.context_tokens, tick.service_s)
        for tick in ticks
        if tick.kind == kind and not (skip_first_tick and tick.tick_index == 0)
    ]
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    var_x = sum((x - mean_x) ** 2 for x, _ in points)
    if var_x == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var_x
    intercept = mean_y - slope * mean_x
    return {
        "intercept_s": intercept,
        "slope_s_per_1k_tokens": slope * 1000,
        "deadline_context_tokens": (TICK_DEADLINE_S - intercept) / slope if slope > 0 else math.inf,
        "count": len(points),
    }
