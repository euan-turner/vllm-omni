# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/duplex_session/merge_stage_traces.py on synthetic Kineto traces.

Design doc: docs/design/fullduplex.md.
External dependencies: none.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "merge_stage_traces", REPO_ROOT / "benchmarks/duplex_session/merge_stage_traces.py"
)
merge_stage_traces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_stage_traces)

BASE_NS = 1_790_857_026_000_000_000


def _stage_trace(os_pid: int, stage: int, ts: float, flow_id: int) -> dict:
    name = f"VLLM::StageEngineCoreProc_stage{stage}"
    return {
        "baseTimeNanoseconds": BASE_NS,
        "displayTimeUnit": "ms",
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": os_pid, "tid": 0, "args": {"name": name}},
            {"ph": "M", "name": "process_labels", "pid": os_pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": "process_sort_index", "pid": os_pid, "tid": 0, "args": {"sort_index": os_pid}},
            {"ph": "M", "name": "process_name", "pid": 0, "tid": 0, "args": {"name": name}},
            {"ph": "M", "name": "process_labels", "pid": 0, "tid": 0, "args": {"labels": "GPU 0"}},
            {"ph": "M", "name": "process_sort_index", "pid": 0, "tid": 0, "args": {"sort_index": 5000000}},
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": os_pid,
                "tid": os_pid,
                "ts": ts,
                "dur": 1.0,
            },
            {"ph": "s", "cat": "ac2g", "name": "ac2g", "id": flow_id, "pid": os_pid, "tid": os_pid, "ts": ts},
            {"ph": "X", "cat": "kernel", "name": "gemm", "pid": 0, "tid": 7, "ts": ts + 2, "dur": 5.0},
            {"ph": "f", "cat": "ac2g", "name": "ac2g", "id": flow_id, "pid": 0, "tid": 7, "ts": ts + 2, "bp": "e"},
            {"ph": "X", "cat": "overhead", "name": "Buffer Flush", "pid": -1, "tid": 0, "ts": ts, "dur": 1.0},
        ],
    }


def _write_run(run_dir: Path, traces: dict[str, dict]) -> None:
    for dirname, trace in traces.items():
        path = run_dir / "traces" / dirname / "trace_rank0.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)


def _merged_events(run_dir: Path) -> list[dict]:
    with gzip.open(merge_stage_traces.merge_run(run_dir), "rt") as f:
        return json.load(f)["traceEvents"]


@pytest.fixture
def two_stage_run(tmp_path: Path) -> Path:
    # Same flow id in both stages, and both stages report device pid 0.
    _write_run(
        tmp_path,
        {
            "20261022-173336_stage1_rank0_1": _stage_trace(200, 1, 2000.0, flow_id=42),
            "20261022-173335_stage0_rank0_1": _stage_trace(100, 0, 1000.0, flow_id=42),
        },
    )
    return tmp_path


def test_device_pids_are_separated_per_stage_and_cpu_pids_kept(two_stage_run: Path) -> None:
    events = _merged_events(two_stage_run)
    kernel_pids = {e["ts"]: e["pid"] for e in events if e.get("cat") == "kernel"}
    assert kernel_pids[1002.0] != kernel_pids[2002.0]
    assert {e["pid"] for e in events if e.get("cat") == "cuda_runtime"} == {100, 200}
    assert not {kernel_pids[1002.0], kernel_pids[2002.0]} & {100, 200}


def test_flow_pairs_stay_within_their_stage(two_stage_run: Path) -> None:
    events = _merged_events(two_stage_run)
    flows: dict[int, set[float]] = {}
    for e in events:
        if e.get("ph") in ("s", "f"):
            flows.setdefault(e["id"], set()).add(e["ts"] // 1000)
    assert len(flows) == 2
    assert all(len(stage_ts) == 1 for stage_ts in flows.values())


def test_timestamps_unchanged_and_rows_sorted_by_stage(two_stage_run: Path) -> None:
    events = _merged_events(two_stage_run)
    assert sorted(e["ts"] for e in events if e.get("cat") == "kernel") == [1002.0, 2002.0]
    sort_index = {e["pid"]: e["args"]["sort_index"] for e in events if e.get("name") == "process_sort_index"}
    gpu_pids = {e["ts"]: e["pid"] for e in events if e.get("cat") == "kernel"}
    assert sort_index[100] < sort_index[gpu_pids[1002.0]] < sort_index[200] < sort_index[gpu_pids[2002.0]]


def test_unnamed_synthetic_pids_get_a_stage_label(two_stage_run: Path) -> None:
    events = _merged_events(two_stage_run)
    overhead_pids = {e["pid"] for e in events if e.get("cat") == "overhead"}
    names = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    assert sorted(names[pid] for pid in overhead_pids) == ["stage0 rank0 pid -1", "stage1 rank0 pid -1"]


def test_mismatched_clock_base_is_rejected(tmp_path: Path) -> None:
    skewed = _stage_trace(200, 1, 2000.0, flow_id=1)
    skewed["baseTimeNanoseconds"] = BASE_NS + 1
    _write_run(tmp_path, {"x_stage0_rank0_1": _stage_trace(100, 0, 1000.0, 1), "x_stage1_rank0_1": skewed})
    with pytest.raises(ValueError, match="baseTimeNanoseconds"):
        merge_stage_traces.merge_run(tmp_path)


def test_main_warns_when_a_run_json_stage_has_no_trace(two_stage_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (two_stage_run / "run.json").write_text(json.dumps({"stages": [0, 1, 2]}))
    assert merge_stage_traces.main([str(two_stage_run)]) == 0
    assert "no trace for stage(s) [2]" in capsys.readouterr().err
    assert (two_stage_run / merge_stage_traces.MERGED_NAME).exists()


def test_main_is_silent_when_all_run_json_stages_are_traced(
    two_stage_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (two_stage_run / "run.json").write_text(json.dumps({"stages": [0, 1]}))
    assert merge_stage_traces.main([str(two_stage_run)]) == 0
    assert capsys.readouterr().err == ""


def test_main_reports_run_without_traces(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert merge_stage_traces.main([str(tmp_path)]) == 1
    assert "no traces" in capsys.readouterr().err
