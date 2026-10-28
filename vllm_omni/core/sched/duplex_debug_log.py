# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Periodic debug logging for long duplex sessions (prototype).

Purpose: diagnose host-memory growth and turn-state behaviour in long
MiniCPM-o duplex sessions (branch long-context-prototype). Off unless
``VLLM_OMNI_DUPLEX_DEBUG_LOG_S`` is set to a positive interval in seconds;
each caller then emits one ``[duplex-debug <name>] {json}`` log line at most
once per interval.

Dependencies: torch (tensor sizes), vLLM logger; reads ``/proc/self/statm``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DEBUG_LOG_ENV = "VLLM_OMNI_DUPLEX_DEBUG_LOG_S"


class PeriodicDebugLog:
    """Rate-limited JSON log lines; every method is a cheap no-op when disabled."""

    def __init__(self, name: str, *, interval_s: float | None = None, clock=time.monotonic) -> None:
        if interval_s is None:
            try:
                interval_s = float(os.environ.get(DEBUG_LOG_ENV) or 0)
            except ValueError:
                interval_s = 0.0
        self.name = name
        self.interval_s = interval_s
        self.enabled = interval_s > 0
        self._clock = clock
        self._next = clock() + interval_s

    def due(self) -> bool:
        if not self.enabled:
            return False
        now = self._clock()
        if now < self._next:
            return False
        self._next = now + self.interval_s
        return True

    def emit(self, fields: dict[str, Any]) -> None:
        logger.info("[duplex-debug %s] %s", self.name, json.dumps(fields, default=str, separators=(",", ":")))


def rss_bytes() -> int | None:
    """Resident set size of this process, or None off Linux."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def host_alloc_stats() -> dict[str, int] | None:
    """Current bytes held by torch's pinned-host caching allocator, or None if unavailable."""
    stats_fn = getattr(torch.cuda, "host_memory_stats", None)
    if stats_fn is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    try:
        stats = stats_fn()
    except RuntimeError:
        return None
    return {key: int(value) for key, value in stats.items() if key.endswith(".current") and "bytes" in key}


def top_fields(fields: dict[str, Any], *, n: int = 4, max_depth: int = 10) -> dict[str, dict[str, int]]:
    """The ``n`` entries of ``fields`` with the largest footprint (host bytes, then scalar items)."""
    sizes = {name: footprint(value, max_depth=max_depth) for name, value in fields.items()}
    ranked = sorted(sizes.items(), key=lambda kv: (kv[1]["cpu_bytes"] + 8 * kv[1]["items"]), reverse=True)
    return {name: size for name, size in ranked[:n] if size["cpu_bytes"] or size["items"] or size["device_bytes"]}


def footprint(obj: Any, *, max_depth: int = 6) -> dict[str, int]:
    """Tensor/array bytes (host / device), tensor count and scalar items reachable from ``obj``.

    Walks dicts, lists, tuples, sets and plain objects' ``__dict__``; a list of
    scalars counts as ``items`` without visiting each element. numpy arrays and
    bytes count as host bytes. Shared objects are counted once.
    """
    totals = {"cpu_bytes": 0, "device_bytes": 0, "tensors": 0, "items": 0}
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if id(value) in seen or depth > max_depth:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            key = "cpu_bytes" if value.device.type == "cpu" else "device_bytes"
            totals[key] += value.numel() * value.element_size()
            totals["tensors"] += 1
        elif isinstance(value, bytes | bytearray):
            totals["cpu_bytes"] += len(value)
        elif hasattr(value, "__array_interface__") and hasattr(value, "nbytes"):
            totals["cpu_bytes"] += int(value.nbytes)
            totals["tensors"] += 1
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, list | tuple | set | frozenset):
            if value and isinstance(next(iter(value)), int | float | str | bool):
                totals["items"] += len(value)
                return
            for item in value:
                visit(item, depth + 1)
        elif hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module | type):
            visit(vars(value), depth + 1)

    visit(obj, 0)
    return totals
