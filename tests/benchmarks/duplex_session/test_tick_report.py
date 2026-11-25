# SPDX-License-Identifier: Apache-2.0
"""Tests the duplex tick-trace joiner in benchmarks/duplex_session/tick_report.py.

Design doc: docs/contributing/profiling.md (Duplex Tick Trace).
External dependencies: none (the script is loaded from its file).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmarks/duplex_session/tick_report.py"
MS = 1_000_000


def _load():
    spec = importlib.util.spec_from_file_location("duplex_tick_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tick_report = _load()


def _write(tmp_path: Path, name: str, rows: list[list]) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _append(rows: list[list], *, attempt: int, dq: int, t_in: int, seq: int, silence: bool = False) -> None:
    """One client append (unless silence) and the Stage-0 append it triggers, accepted 5 ms later."""
    if not silence:
        rows.append(["in", t_in * MS, "s", dq])
        rows.append(["dq", (t_in + 2) * MS, "s", dq])
    rows.append(["start", (t_in + 3) * MS, "s", attempt, dq, silence])
    rows.append(["acc", (t_in + 5) * MS, "s", attempt, 0, seq, "r0", (t_in + 3) * MS])


def _trace(tmp_path: Path) -> list[Path]:
    orch: list[list] = []
    # Tick 1 (listen): one-token segment 80 ms after ingress at t=1000.
    _append(orch, attempt=1, dq=1, t_in=1000, seq=1)
    orch.append(["out", 1080 * MS, "s", 0, "r0", True, 1])
    # Tick 2 (speak): ingress at 2000; decision at 2100; audio at 2300.
    _append(orch, attempt=2, dq=2, t_in=2000, seq=2)
    orch.append(["out", 2100 * MS, "s", 0, "r0", False, 2])
    orch.append(["out", 2150 * MS, "s", 1, "r0", False, 4])
    orch.append(["out", 2300 * MS, "s", 2, "r0", False, 0])
    # Tick 3 is accepted before tick 2's segment finishes (send-ahead).
    _append(orch, attempt=3, dq=3, t_in=3000, seq=3)
    orch.append(["out", 3010 * MS, "s", 0, "r0", True, 3])
    orch.append(["out", 3400 * MS, "s", 0, "r0", True, 1])
    # Tick 4: a silence continuation with no client input.
    _append(orch, attempt=4, dq=3, t_in=4000, seq=4, silence=True)
    orch.append(["out", 4090 * MS, "s", 0, "r0", True, 1])
    worker = [
        ["w", 1001 * MS, "s", 0, 1, 1000, 900, 79, False],
        ["w", 2001 * MS, "s", 0, 2, 5000, 4900, 79, False],
        ["w", 2002 * MS, "s", 0, 2, 9999, 9999, 79, False],
        ["w", 3001 * MS, "s", 0, 3, 5100, 5000, 79, False],
        ["w", 4001 * MS, "s", 0, 4, 5200, 5100, 79, True],
    ]
    return [
        _write(tmp_path, "duplex-ticks-orchestrator-1.jsonl", orch),
        _write(tmp_path, "duplex-ticks-stage0-2.jsonl", worker),
    ]


def test_ticks_attribute_segments_by_order_not_time(tmp_path: Path) -> None:
    result = tick_report.join_ticks(tick_report.load_records(_trace(tmp_path)))

    assert result.mismatches == []
    assert [(tick.seq, tick.kind) for tick in result.ticks] == [
        (1, "listen"),
        (2, "speak"),
        (3, "listen"),
        (4, "silence"),
    ]
    listen, speak, late_listen, silence = result.ticks
    assert listen.decision_latency_ms == pytest.approx(80)
    assert listen.mailbox_ms == pytest.approx(2)
    # The segment-2 output at 3010 ms belongs to tick 2, so tick 3's decision is 3400 ms.
    assert speak.decision_latency_ms == pytest.approx(100)
    assert speak.segment_end_ms == pytest.approx(3010)
    assert speak.audio_latency_ms == pytest.approx(300)
    assert late_listen.decision_latency_ms == pytest.approx(400)
    # A silence tick starts at its own append, not at the last client ingress.
    assert silence.decision_latency_ms == pytest.approx(4090 - 4003)
    assert silence.mailbox_ms is None


def test_speak_without_handoff_rows_is_empty_and_gets_no_first_audio(tmp_path: Path) -> None:
    orch: list[list] = []
    # Tick 1: a speak decision whose handoffs carry no hidden rows; no audio follows.
    _append(orch, attempt=1, dq=1, t_in=1000, seq=1)
    orch.append(["out", 1080 * MS, "s", 0, "r0", False, 1])
    orch.append(["h", 1081 * MS, "r0", 1, 0, 0])
    orch.append(["out", 1200 * MS, "s", 0, "r0", True, 2])
    orch.append(["h", 1201 * MS, "r0", 1, 0, 0])
    # Tick 2: words; its first handoff is forwarded after its first output; audio at 2300.
    _append(orch, attempt=2, dq=2, t_in=2000, seq=2)
    orch.append(["out", 2090 * MS, "s", 0, "r0", False, 3])
    orch.append(["h", 2091 * MS, "r0", 1, 0, 4])
    orch.append(["out", 2150 * MS, "s", 0, "r0", True, 5])
    orch.append(["h", 2151 * MS, "r0", 1, 0, 0])
    orch.append(["out", 2300 * MS, "s", 1, "r0", False, 0])

    result = tick_report.join_ticks(
        tick_report.load_records([_write(tmp_path, "duplex-ticks-orchestrator-1.jsonl", orch)])
    )

    empty, speak = result.ticks
    assert (empty.kind, empty.handoff_rows, empty.audio_latency_ms) == ("speak_empty", 0, None)
    assert (speak.kind, speak.handoff_rows) == ("speak", 4)
    assert speak.audio_latency_ms == pytest.approx(300)


def test_speak_without_handoff_records_stays_speak(tmp_path: Path) -> None:
    ticks = tick_report.join_ticks(tick_report.load_records(_trace(tmp_path))).ticks

    assert [tick.handoff_rows for tick in ticks] == [None, None, None, None]


def test_first_prompt_record_wins_and_window_flag_is_kept(tmp_path: Path) -> None:
    result = tick_report.join_ticks(tick_report.load_records(_trace(tmp_path)))

    assert [tick.context for tick in result.ticks] == [1000, 5000, 5100, 5200]
    assert [tick.window_replaced for tick in result.ticks] == [False, False, False, True]


def test_segment_count_mismatch_excludes_the_request(tmp_path: Path) -> None:
    orch: list[list] = []
    _append(orch, attempt=1, dq=1, t_in=1000, seq=1)
    _append(orch, attempt=2, dq=2, t_in=2000, seq=2)
    orch.append(["out", 1080 * MS, "s", 0, "r0", True, 1])

    result = tick_report.join_ticks(
        tick_report.load_records([_write(tmp_path, "duplex-ticks-orchestrator-1.jsonl", orch)])
    )

    assert result.ticks == []
    assert result.mismatches == ["r0: 2 appends vs 1 finished segments"]


def test_summary_bins_by_context_and_reports_slack(tmp_path: Path) -> None:
    ticks = tick_report.join_ticks(tick_report.load_records(_trace(tmp_path))).ticks

    rows = {(row["kind"], row["ctx_bin_start"]): row for row in tick_report.summarize(ticks)}

    assert set(rows) == {("listen", 0), ("speak", 4096), ("listen", 4096), ("silence", 4096)}
    late = rows[("listen", 4096)]
    assert late["n_ticks"] == 1
    assert late["slack_min_ms"] == pytest.approx(600)
    assert late["on_time_frac"] == 1.0
    assert rows[("speak", 4096)]["first_audio_p50_ms"] == pytest.approx(300)


def test_cli_writes_csvs_and_flags_mismatch(tmp_path: Path) -> None:
    _trace(tmp_path)

    assert tick_report.main([str(tmp_path)]) == 0
    assert (tmp_path / "ticks.csv").read_text().count("\n") == 5
    assert (tmp_path / "summary.csv").exists()
