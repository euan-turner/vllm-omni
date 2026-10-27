# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/duplex_session/serve.sh and its 1-GPU and 2-GPU deploy overlays.

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


def test_1gpu_layout_selects_1gpu_overlay() -> None:
    argv = _argv(LAYOUT="1gpu")
    deploy = REPO_ROOT / "benchmarks/duplex_session/deploy/minicpmo_4_5_1gpu_32gb.yaml"
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


@pytest.mark.parametrize(("with_stack", "expected"), [(None, False), ("0", False), ("1", True)])
def test_torch_profile_stacks_are_opt_in_and_cuda_time_dump_is_off(with_stack: str | None, expected: bool) -> None:
    env = {"PROFILE": "torch"} | ({"TORCH_WITH_STACK": with_stack} if with_stack is not None else {})
    config = json.loads(_flag(_argv(**env), "--stage-overrides"))["0"]["profiler_config"]
    assert config["torch_profiler_with_stack"] is expected
    assert config["torch_profiler_dump_cuda_time_total"] is False
    assert config["torch_profiler_use_gzip"] is True


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
        {"LAYOUT": "3gpu"},
        {"PROFILE": "nsys"},
        {"PROFILE": "torch", "PROFILE_STAGES": "0,3"},
        {"PROFILE": "torch", "PROFILE_STAGES": "0,,1"},
        {"PROFILE": "torch", "TORCH_WITH_STACK": "yes"},
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


def test_1gpu_overlay_caps_two_sessions_on_one_device() -> None:
    merged, stages = _resolved_stages("minicpmo_4_5_1gpu_32gb.yaml")

    assert merged["session_mode"] == "duplex"
    assert {stage["devices"] for stage in stages.values()} == {"0"}
    assert {stage["max_num_seqs"] for stage in stages.values()} == {merged["duplex_session"]["max_sessions"]} == {2}
    assert stages[0]["hf_overrides"] == {"duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions": True}
    extra = merged["connectors"]["connector_of_shared_memory"]["extra"]
    assert extra["micro_batch_size"] == merged["duplex_session"]["max_sessions"]
    assert extra["codec_chunk_frames"] == 25


def test_1gpu_overlay_is_audio_only_with_8192_context() -> None:
    _, stages = _resolved_stages("minicpmo_4_5_1gpu_32gb.yaml")

    assert stages[0]["max_model_len"] == 8192
    assert stages[0]["max_num_batched_tokens"] == 4096
    assert stages[0]["limit_mm_per_prompt"] == {"image": 0, "video": 0, "audio": 64}


def test_1gpu_overlay_fixed_kv_pools() -> None:
    merged, stages = _resolved_stages("minicpmo_4_5_1gpu_32gb.yaml")

    cuda_kv = {s["stage_id"]: s["kv_cache_memory_bytes"] for s in merged["platforms"]["cuda"]["stages"]}
    assert cuda_kv == {0: 1342177280, 1: 536870912}


def _dry_run_env(**env: str) -> str:
    result = _dry_run(**env)
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_gh200_layout_overflows_hf_context_explicitly() -> None:
    env = {"LAYOUT": "gh200", "TICK_TRACE_DIR": "/tmp/ticks"}
    argv = _argv(**env)
    deploy = DEPLOY_DIR / "minicpmo_4_5_gh200_long_context.yaml"
    assert _flag(argv, "--deploy-config") == str(deploy)
    assert deploy.is_file()
    assert json.loads(_flag(argv, "--stage-overrides")) == {"0": {"max_model_len": 131072}}
    stderr = _dry_run_env(**env)
    assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" in stderr
    assert "VLLM_OMNI_DUPLEX_TICK_TRACE_DIR=/tmp/ticks" in stderr


def test_max_model_len_within_hf_limit_merges_with_profiler_overrides() -> None:
    env = {"LAYOUT": "gh200", "MAX_MODEL_LEN": "40960", "PROFILE": "torch", "PROFILE_STAGES": "0,1"}
    overrides = json.loads(_flag(_argv(**env), "--stage-overrides"))
    assert overrides["0"]["max_model_len"] == 40960
    assert "profiler_config" in overrides["0"] and "profiler_config" in overrides["1"]
    assert "max_model_len" not in overrides["1"]
    assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" not in _dry_run_env(**env)


@pytest.mark.parametrize("value", ["0", "abc", "-5"])
def test_invalid_max_model_len_is_rejected(value: str) -> None:
    result = _dry_run(LAYOUT="gh200", MAX_MODEL_LEN=value)
    assert result.returncode == 2
    assert "MAX_MODEL_LEN" in result.stderr
