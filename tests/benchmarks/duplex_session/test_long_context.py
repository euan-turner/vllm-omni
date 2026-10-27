# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/duplex_session/long_context (stitching, tick metrics, analysis, driver artifacts).

Prototype on branch long-context-prototype.
External dependencies: none (matplotlib plots are skipped when not installed).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
LC_DIR = REPO_ROOT / "benchmarks/duplex_session/long_context"


def _load(name: str):
    sys.path.insert(0, str(LC_DIR))
    spec = importlib.util.spec_from_file_location(name, LC_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stitch = _load("stitch")
tick_metrics = _load("tick_metrics")
analyze = _load("analyze")
driver = _load("run_long_session")

BYTES_PER_S = 16000 * 2


def test_segment_duration_rounds_up_to_whole_units() -> None:
    assert stitch.segment_duration_s(BYTES_PER_S * 3, 24.2) == 25
    assert stitch.segment_duration_s(BYTES_PER_S * 30 + 2, 24.0) == 31
    assert stitch.segment_duration_s(0, None) == 1


def test_plan_cycles_samples_until_target() -> None:
    segments = stitch.plan_segments(["a", "b", "c"], ["S"] * 3, [10, 20, 30], target_s=95, start_index=1)
    assert [s.sample_id for s in segments] == ["b", "c", "a", "b", "c"]
    assert [s.offset_s for s in segments] == [0, 20, 50, 60, 80]
    assert [s.pass_index for s in segments] == [0, 0, 0, 1, 1]
    assert segments[-1].end_s >= 95


def test_stitch_pads_audio_and_shifts_frames() -> None:
    pcms = [b"\x01\x00" * 16000, b"\x02\x00" * 48000]
    frames = [((0.0, b"A0"), (1.0, b"A1"), (5.0, b"late")), ((0.0, b"B0"), (2.0, b"B2"))]
    segments = stitch.plan_segments(["a", "b"], ["S", "S"], [2, 3], target_s=5)
    audio, shifted = stitch.stitch_pcm_and_frames(pcms, frames, segments, with_frames=True)
    assert len(audio) == 5 * BYTES_PER_S
    assert audio[:BYTES_PER_S] == pcms[0] and audio[BYTES_PER_S : 2 * BYTES_PER_S] == bytes(BYTES_PER_S)
    assert audio[2 * BYTES_PER_S :] == pcms[1]
    assert shifted == ((0.0, b"A0"), (1.0, b"A1"), (2.0, b"B0"), (4.0, b"B2"))
    _, no_frames = stitch.stitch_pcm_and_frames(pcms, frames, segments, with_frames=False)
    assert no_frames == ()


def test_split_sentences_rebases_to_segment_clock() -> None:
    segments = stitch.plan_segments(["a", "b"], ["S", "S"], [10, 10], target_s=20)
    timed = [{"sentence": "x", "start": 3.0, "end": 4.0}, {"sentence": "y", "start": 12.5, "end": 14.0}]
    by_segment = stitch.split_sentences(timed, segments)
    assert by_segment[0] == [{"sentence": "x", "start": 3.0, "end": 4.0}]
    assert by_segment[1] == [{"sentence": "y", "start": 2.5, "end": 4.0}]


def _record(rid: str, index: int, *, t0: float, context: int, service: float, sampled: list[int], **extra) -> dict:
    return {
        "request_id": rid,
        "tick_index": index,
        "t_arrival": t0,
        "t_first_scheduled": t0 + 0.01,
        "t_prefill_done": t0 + 0.01 + service / 2,
        "t_end": t0 + 0.01 + service,
        "context_tokens_before": context - 12,
        "prefill_tokens": 12,
        "num_decode_steps": len(sampled) - 1,
        "queue_depth_at_arrival": 0,
        "sampled_token_ids": sampled,
        "status": "stop",
        **extra,
    }


def test_tick_metrics_definitions() -> None:
    records = [
        _record("r", 0, t0=10.0, context=500, service=0.5, sampled=[1]),
        _record("r", 1, t0=11.0, context=600, service=0.2, sampled=[1]),
        _record("r", 2, t0=12.0, context=700, service=1.4, sampled=[2, 3, 4]),
        {**_record("r", 3, t0=13.0, context=800, service=0.1, sampled=[]), "status": "overflow"},
    ]
    ticks = tick_metrics.compute_ticks(records, listen_token_id=1)
    assert [t.kind for t in ticks] == ["listen", "listen", "speak"]
    third = ticks[2]
    assert third.context_tokens == 700
    assert math.isclose(third.service_s, 1.4) and math.isclose(third.queue_s, 0.01)
    assert math.isclose(third.prefill_s + third.decode_s, third.service_s)
    assert third.deadline_miss and not ticks[1].deadline_miss
    assert math.isclose(third.realtime_lag_s, (12.0 + 0.01 + 1.4) - (10.0 + 2))
    assert tick_metrics.tick_kind([5], None) == "listen" and tick_metrics.tick_kind([5], 1) == "speak"


def test_summaries_skip_first_tick_and_fit_slope() -> None:
    records = [_record("r", 0, t0=0.0, context=100, service=5.0, sampled=[1])]
    records += [
        _record("r", i, t0=float(i), context=1000 * i, service=0.1 + 0.05 * i, sampled=[1]) for i in range(1, 9)
    ]
    ticks = tick_metrics.compute_ticks(records, listen_token_id=1)
    rows = tick_metrics.summarize_by_context(ticks, bin_tokens=4096)
    assert sum(row["count"] for row in rows) == 8
    fit = tick_metrics.linear_fit(ticks, kind="listen")
    assert math.isclose(fit["slope_s_per_1k_tokens"], 0.05, rel_tol=1e-9)
    assert math.isclose(fit["deadline_context_tokens"], 18000, rel_tol=1e-9)
    assert tick_metrics.linear_fit(ticks, kind="speak") is None


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    trace_dir = tmp_path / "ticks"
    trace_dir.mkdir()
    segments = stitch.plan_segments(["a", "b"], ["RTD_Omni"] * 2, [2, 2], target_s=4)
    (run_dir / "session0" / "scores").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"modality": "av", "tick_trace_dir": str(trace_dir)}))
    (run_dir / "session0" / "manifest.json").write_text(
        json.dumps(
            {
                "session_index": 0,
                "t_start_monotonic": 100.0,
                "t_end_monotonic": 200.0,
                "segments": [s.as_dict() for s in segments],
            }
        )
    )
    (run_dir / "session0" / "scores" / "0001_b.json").write_text(
        json.dumps({"content": {"content_score": 2.0}, "temporal": {"summary": {"avg_temporal_score": 1.5}}})
    )
    warmup = [_record("warm", 0, t0=50.0, context=100, service=0.3, sampled=[1])]
    measured = [_record("live", i, t0=101.0 + i, context=100 + 12 * i, service=0.3, sampled=[1]) for i in range(4)]
    (trace_dir / "ticks_stage0_pid1.jsonl").write_text("\n".join(json.dumps(r) for r in warmup + measured) + "\n")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "b.json").write_text(
        json.dumps({"content": {"content_score": 3.0}, "temporal": {"summary": {"avg_temporal_score": 2.0}}})
    )
    return run_dir, baseline


def test_analyze_joins_ticks_to_sessions_and_scores(tmp_path) -> None:
    run_dir, baseline = _write_run(tmp_path)
    assert analyze.main([str(run_dir), "--listen-token-id", "1", "--baseline-score-root", str(baseline)]) == 0
    summary = json.loads((run_dir / "analysis" / "summary.json").read_text())
    assert summary["num_ticks"] == 4
    lines = (run_dir / "analysis" / "ticks.csv").read_text().splitlines()
    assert len(lines) == 5 and all(",live," in line for line in lines[1:])
    assert [line.split(",")[2] for line in lines[1:]] == ["a", "a", "b", "b"]
    quality = (run_dir / "analysis" / "quality_vs_depth.csv").read_text().splitlines()
    assert quality[0].split(",")[-2:] == ["content_delta", "temporal_delta"]
    assert quality[1].split(",")[3:] == ["2", "2.0", "1.5", "-1.0", "-0.5"]


def test_write_segment_artifacts_produce_judge_inputs(tmp_path) -> None:
    segments = stitch.plan_segments(["a", "b"], ["RTD_Omni"] * 2, [10, 10], target_s=20)
    timed = [{"sentence": "hello.", "start": 11.0, "end": 12.0}]
    driver.write_segment_artifacts(tmp_path, segments, timed, {"clock": "media", "pace": "realtime"})
    meta = json.loads((tmp_path / "segments" / "0001_b.meta.json").read_text())
    assert meta["id"] == "b" and meta["clock"] == "media" and meta["segment"]["offset_s"] == 10
    assert json.loads((tmp_path / "segments" / "0001_b.json").read_text()) == [
        {"sentence": "hello.", "start": 1.0, "end": 2.0}
    ]
    assert json.loads((tmp_path / "segments" / "0000_a.json").read_text()) == []


def test_session_start_indices_rotate() -> None:
    assert driver.session_start_indices(0, 3, 7, 50) == [0, 7, 14]
    assert driver.session_start_indices(45, 2, 7, 50) == [45, 2]


def test_driver_parser_requires_target_for_generate() -> None:
    args = driver.build_parser().parse_args(
        ["generate", "--dataset", "d", "--out-dir", "o", "--target-minutes", "10", "--modality", "audio"]
    )
    assert args.modality == "audio" and args.target_minutes == 10.0 and args.sessions == 1
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["generate", "--dataset", "d", "--out-dir", "o"])
