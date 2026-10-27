# SPDX-License-Identifier: Apache-2.0
"""Tick and tick-deadline metric definitions for long-context duplex profiling.

Purpose: turn the Stage-0 tick records written by
``vllm_omni/core/sched/duplex_tick_trace.py`` (one JSON line per tick) into
per-tick metrics and context-length summaries. Prototype on branch
long-context-prototype; definitions here are the ones the plots use.

Definitions (all times in seconds, ``time.monotonic`` in the EngineCore):

- **Tick**: one streaming input update (one 1 s client append) applied to a
  session: its prefill plus the decode steps that follow it until the
  segment stops. Only records with ``status == "stop"`` are complete ticks.
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
  listen token (any single token when the listen id is unknown); otherwise
  ``speak``.
- **Tick deadline**: ``TICK_DEADLINE_S`` = 1.0, the input unit period. A tick
  misses it when ``service_s > TICK_DEADLINE_S``; ticks are serial within a
  session, so sustained misses grow the input backlog without bound.
- **realtime_lag_s** = ``t_end - (t_arrival[first tick] + tick_index * UNIT_S)``:
  how far behind the real-time input schedule the session's output is.

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
    realtime_lag_s: float
    deadline_miss: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def tick_kind(sampled_token_ids: Sequence[int], listen_token_id: int | None) -> str:
    if len(sampled_token_ids) == 1 and (listen_token_id is None or sampled_token_ids[0] == listen_token_id):
        return "listen"
    return "speak"


def compute_ticks(records: Iterable[dict[str, Any]], *, listen_token_id: int | None = None) -> list[Tick]:
    """Complete ticks in (request, tick_index) order; incomplete or overflowed records are skipped."""
    complete = [
        record
        for record in records
        if record.get("status") == "stop"
        and all(record.get(key) is not None for key in ("t_first_scheduled", "t_prefill_done", "t_end"))
    ]
    complete.sort(key=lambda record: (record["request_id"], record["tick_index"]))
    first_arrival: dict[str, float] = {}
    for record in records:
        rid = record["request_id"]
        if record.get("tick_index") == 0:
            first_arrival[rid] = float(record["t_arrival"])
    ticks = []
    for record in complete:
        rid = record["request_id"]
        anchor = first_arrival.get(rid, float(record["t_arrival"]))
        service = record["t_end"] - record["t_first_scheduled"]
        sampled = list(record.get("sampled_token_ids") or ())
        ticks.append(
            Tick(
                request_id=rid,
                tick_index=int(record["tick_index"]),
                kind=tick_kind(sampled, listen_token_id),
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
                realtime_lag_s=record["t_end"] - (anchor + int(record["tick_index"]) * UNIT_S),
                deadline_miss=service > TICK_DEADLINE_S,
            )
        )
    return ticks


def _percentile(values: Sequence[float], q: float) -> float:
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
    """Service-time percentiles and deadline-miss rate per (context bin, kind)."""
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
                "service_p50_s": _percentile(service, 0.5),
                "service_p90_s": _percentile(service, 0.9),
                "service_p99_s": _percentile(service, 0.99),
                "prefill_p50_s": _percentile([tick.prefill_s for tick in members], 0.5),
                "decode_p50_s": _percentile([tick.decode_s for tick in members], 0.5),
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
