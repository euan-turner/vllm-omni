# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/duplex_session/serve.sh and its 2-GPU deploy overlay.

Design doc: docs/design/fullduplex.md (Serving).
External dependencies: bash; the overlay test needs vllm (skipped otherwise).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SH = REPO_ROOT / "benchmarks/duplex_session/serve.sh"
DEPLOY_DIR = REPO_ROOT / "benchmarks/duplex_session/deploy"


def _dry_run(**env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SERVE_SH)],
        env={**os.environ, "DRY_RUN": "1", **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _argv(**env: str) -> list[str]:
    result = _dry_run(**env)
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def test_default_layout_selects_2gpu_overlay() -> None:
    argv = _argv()
    deploy = REPO_ROOT / "benchmarks/duplex_session/deploy/minicpmo_4_5_2gpu_32gb.yaml"
    assert _flag(argv, "--deploy-config") == str(deploy)
    assert deploy.is_file()


def test_profile_off_adds_no_overrides_or_nsys() -> None:
    argv = _argv(PROFILE="off", PROFILE_STAGES="0,1")
    assert "--stage-overrides" not in argv
    assert argv[0] == "vllm-omni"


def test_torch_profile_configures_exactly_the_listed_stages() -> None:
    argv = _argv(PROFILE="torch", PROFILE_STAGES="0,2", PROFILE_DIR="/tmp/p")
    overrides = json.loads(_flag(argv, "--stage-overrides"))
    assert set(overrides) == {"0", "2"}
    for stage in overrides.values():
        assert stage["profiler_config"]["profiler"] == "torch"
        assert stage["profiler_config"]["torch_profiler_dir"] == "/tmp/p/torch"


def test_cuda_profile_wraps_server_in_nsys_capture_range() -> None:
    argv = _argv(PROFILE="cuda", PROFILE_STAGES="1")
    assert argv[:2] == ["nsys", "profile"]
    assert "--capture-range=cudaProfilerApi" in argv
    overrides = json.loads(_flag(argv, "--stage-overrides"))
    assert overrides == {"1": {"profiler_config": {"profiler": "cuda"}}}


def test_extra_arguments_pass_through_to_serve() -> None:
    argv = _argv(PROFILE="off")
    result = subprocess.run(
        ["bash", str(SERVE_SH), "--disable-log-stats"],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert shlex.split(result.stdout) == [*argv, "--disable-log-stats"]


@pytest.mark.parametrize(
    "env",
    [
        {"LAYOUT": "1gpu"},
        {"LAYOUT": "3gpu"},
        {"PROFILE": "nsys"},
        {"PROFILE": "torch", "PROFILE_STAGES": "0,3"},
        {"PROFILE": "torch", "PROFILE_STAGES": "0,,1"},
    ],
)
def test_invalid_settings_exit_before_launch(env: dict[str, str]) -> None:
    result = _dry_run(**env)
    assert result.returncode == 2
    assert result.stdout == ""


def _resolved_stages(name: str) -> tuple[dict, dict[int, dict]]:
    pytest.importorskip("vllm")
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    merged = resolve_deploy_yaml(DEPLOY_DIR / name)
    return merged, {stage["stage_id"]: stage for stage in merged["stages"]}


def test_2gpu_overlay_keeps_session_cap_and_session_sized_audio_graphs() -> None:
    merged, stages = _resolved_stages("minicpmo_4_5_2gpu_32gb.yaml")

    assert merged["session_mode"] == "duplex"
    assert merged["connectors"]["connector_of_shared_memory"]["extra"]["codec_chunk_frames"] == 25
    assert {stage["max_num_seqs"] for stage in stages.values()} == {merged["duplex_session"]["max_sessions"]} == {4}
    assert stages[0]["hf_overrides"] == {"duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions": True}


def test_2gpu_overlay_keeps_shipped_placement() -> None:
    _, stages = _resolved_stages("minicpmo_4_5_2gpu_32gb.yaml")

    assert [stages[i]["devices"] for i in range(3)] == ["0", "1", "1"]
    assert [stages[i]["gpu_memory_utilization"] for i in range(3)] == [0.9, 0.55, 0.35]


def test_2gpu_overlay_builds_vision_tower_with_bounded_video_profiling() -> None:
    _, stages = _resolved_stages("minicpmo_4_5_2gpu_32gb.yaml")

    assert stages[0]["max_model_len"] == 16384
    limits = stages[0]["limit_mm_per_prompt"]
    assert limits["image"] >= 1
    assert limits["video"] == {"count": 1, "num_frames": 32}
