# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, per-process JSONL trace files for diagnostics (no effect when disabled).

A trace is enabled by pointing an environment variable at a directory; each
process then appends records to its own ``<prefix>-<pid>.jsonl`` there. Records
are tuples, serialized as JSON arrays by a background thread, so callers on an
event loop or engine step never block on file I/O. The queue is bounded: when
it is full, records are dropped and counted, and the count is written as a
``["trace_dropped", n]`` record.

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: none beyond the standard library.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

_QUEUE_CAPACITY = 65536
_FLUSH_INTERVAL_S = 0.5

_writers: dict[tuple[str, str, int], JsonlTraceWriter] = {}
_writers_lock = threading.Lock()


class JsonlTraceWriter:
    """Appends tuples as JSON lines to one file from a daemon thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.dropped = 0
        self._queue: queue.Queue[tuple | None] = queue.Queue(_QUEUE_CAPACITY)
        self._reported_dropped = 0
        self._thread = threading.Thread(target=self._run, name=f"jsonl-trace-{path.stem}", daemon=True)
        self._thread.start()

    def emit(self, record: tuple) -> None:
        """Queue one record without blocking; drop and count it when the queue is full."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout_s: float = 5.0) -> None:
        """Write every queued record and stop the writer thread."""
        self._queue.put(None)
        self._thread.join(timeout_s)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            while True:
                try:
                    record = self._queue.get(timeout=_FLUSH_INTERVAL_S)
                except queue.Empty:
                    self._write_dropped(file)
                    file.flush()
                    continue
                if record is None:
                    self._write_dropped(file)
                    file.flush()
                    return
                file.write(json.dumps(record, separators=(",", ":")))
                file.write("\n")

    def _write_dropped(self, file) -> None:
        if self.dropped != self._reported_dropped:
            self._reported_dropped = self.dropped
            file.write(json.dumps(["trace_dropped", self.dropped]))
            file.write("\n")


def jsonl_trace_writer(env_var: str, prefix: str) -> JsonlTraceWriter | None:
    """The calling process's writer for ``prefix`` when ``env_var`` names a directory, else ``None``.

    Writers are created lazily and cached per process id, so a forked or spawned
    worker opens its own file and thread.
    """
    directory = os.environ.get(env_var)
    if not directory:
        return None
    pid = os.getpid()
    key = (env_var, prefix, pid)
    writer = _writers.get(key)
    if writer is not None:
        return writer
    with _writers_lock:
        writer = _writers.get(key)
        if writer is None:
            writer = JsonlTraceWriter(Path(directory) / f"{prefix}-{pid}.jsonl")
            _writers[key] = writer
    return writer
