# SPDX-License-Identifier: Apache-2.0
"""Merge the per-stage torch profiler traces of a profile_run.sh run into one
Chrome/Perfetto trace, so all stages share a single timeline in ui.perfetto.dev.

Design doc: docs/design/fullduplex.md; profiling pattern:
docs/contributing/profiling.md.
External dependencies: none (stdlib only). Inputs are Kineto chrome traces
written by the stage engines under <run>/traces/*_stage<N>_rank<R>_*/.

Usage:
    python benchmarks/duplex_session/merge_stage_traces.py RUN_DIR [RUN_DIR ...]
    python benchmarks/duplex_session/merge_stage_traces.py /tmp/duplex_session_runs/*/
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

MERGED_NAME = "merged_trace.json.gz"
STAGE_DIR_RE = re.compile(r"_stage(\d+)_rank(\d+)_")
# Above Linux pid_max (2**22), so remapped device pids never collide with OS pids.
DEVICE_PID_BASE = 10_000_000
DEVICE_PIDS_PER_TRACE = 1_000
FLOW_ID_STRIDE = 1 << 32
SORT_INDEX_STRIDE = 1_000


def find_stage_traces(run_dir: Path) -> list[tuple[str, Path]]:
    """Return (label, path) for each stage trace under run_dir/traces, ordered by stage then rank."""
    found = []
    for path in run_dir.glob("traces/*/trace_rank*.json.gz"):
        match = STAGE_DIR_RE.search(path.parent.name)
        if match:
            stage, rank = int(match[1]), int(match[2])
            found.append(((stage, rank), f"stage{stage} rank{rank}", path))
    return [(label, path) for _, label, path in sorted(found)]


def missing_stages(run_dir: Path) -> list[int]:
    """Return the stages listed in run_dir/run.json that have no trace; empty if run.json is absent."""
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return []
    expected = json.loads(run_json.read_text())["stages"]
    traced = {int(STAGE_DIR_RE.search(path.parent.name)[1]) for _, path in find_stage_traces(run_dir)}
    return sorted(set(expected) - traced)


def _remap_events(trace: dict, index: int, label: str) -> Iterator[dict]:
    """Yield trace events with device/synthetic pids, flow ids and sort order namespaced by trace index."""
    events = trace["traceEvents"]
    cpu_pids = {e["pid"] for e in events if e.get("name") == "process_labels" and e["args"].get("labels") == "CPU"}
    pid_map: dict[object, int] = {}

    def new_pid(pid: object) -> int:
        if pid not in pid_map:
            if len(pid_map) >= DEVICE_PIDS_PER_TRACE:
                raise ValueError(f"{label}: more than {DEVICE_PIDS_PER_TRACE} non-CPU pids")
            pid_map[pid] = DEVICE_PID_BASE + index * DEVICE_PIDS_PER_TRACE + len(pid_map)
        return pid_map[pid]

    named = set()
    for event in events:
        pid = event.get("pid")
        if pid is not None and pid not in cpu_pids:
            event["pid"] = new_pid(pid)
        if event.get("ph") in ("s", "t", "f") and "id" in event:
            event["id"] += index * FLOW_ID_STRIDE
        if event.get("ph") == "M":
            if event.get("name") == "process_name":
                named.add(event["pid"])
            elif event.get("name") == "process_sort_index":
                within = 0 if pid in cpu_pids else 1 + event["pid"] % DEVICE_PIDS_PER_TRACE
                event["args"]["sort_index"] = index * SORT_INDEX_STRIDE + within
        yield event

    for original, pid in pid_map.items():
        if pid not in named:
            yield {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"{label} pid {original!r}"},
            }


def merge_run(run_dir: Path) -> Path:
    """Write run_dir/merged_trace.json.gz combining every stage trace; return its path."""
    stage_traces = find_stage_traces(run_dir)
    if not stage_traces:
        raise FileNotFoundError(f"no traces/*_stage<N>_rank<R>_*/trace_rank*.json.gz under {run_dir}")
    out_path = run_dir / MERGED_NAME
    base_time = None
    first = True
    with gzip.open(out_path, "wt") as out:
        out.write('{"traceEvents": [')
        for index, (label, path) in enumerate(stage_traces):
            with gzip.open(path, "rt") as f:
                trace = json.load(f)
            trace_base = trace.get("baseTimeNanoseconds")
            if base_time is None:
                base_time = trace_base
            elif trace_base != base_time:
                raise ValueError(f"{path}: baseTimeNanoseconds {trace_base} != {base_time}; clocks differ")
            for event in _remap_events(trace, index, label):
                out.write(("" if first else ",\n") + json.dumps(event, separators=(",", ":")))
                first = False
            del trace
        out.write("],\n")
        tail = {
            "displayTimeUnit": "ms",
            "baseTimeNanoseconds": base_time,
            "mergedFrom": [{"label": label, "path": str(path)} for label, path in stage_traces],
        }
        out.write(json.dumps(tail)[1:])
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="profile_run.sh run directories")
    args = parser.parse_args(argv)
    status = 0
    for run_dir in args.run_dirs:
        missing = missing_stages(run_dir)
        if missing:
            print(f"WARNING: {run_dir}: no trace for stage(s) {missing} listed in run.json", file=sys.stderr)
        try:
            print(merge_run(run_dir))
        except (FileNotFoundError, ValueError) as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
