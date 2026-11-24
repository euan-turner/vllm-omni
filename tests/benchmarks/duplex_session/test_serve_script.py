# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests the command built by benchmarks/duplex_session/serve.sh.

Design doc: docs/design/fullduplex.md (Serving).
External dependencies: bash.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SH = REPO_ROOT / "benchmarks/duplex_session/serve.sh"
REVISION = "02b54c54c36f8b48e97501bb0fde5d178c62df38"


def _served(tmp_path: Path, *args: str, **env: str) -> tuple[list[str], str]:
    """Run serve.sh against a stub CLI; return its arguments and the PYTORCH_CUDA_ALLOC_CONF it saw."""
    stub = tmp_path / "vllm-omni"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" "ALLOC=$PYTORCH_CUDA_ALLOC_CONF"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    clean_env = {k: v for k, v in os.environ.items() if k not in {"PORT", "PYTORCH_CUDA_ALLOC_CONF"}}
    result = subprocess.run(
        ["bash", str(SERVE_SH), *args],
        env={**clean_env, "PATH": f"{tmp_path}:{os.environ['PATH']}", **env},
        capture_output=True,
        check=True,
    )
    *argv, alloc = result.stdout.decode().split("\0")[:-1]
    return argv, alloc.removeprefix("ALLOC=")


def _served_argv(tmp_path: Path, *args: str, **env: str) -> list[str]:
    return _served(tmp_path, *args, **env)[0]


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_serves_pinned_gptq_checkpoint_with_its_deploy_config(tmp_path: Path) -> None:
    argv = _served_argv(tmp_path)

    assert argv[:3] == ["serve", "openbmb/MiniCPM-o-4_5-GPTQ", "--omni"]
    assert _flag(argv, "--revision") == REVISION
    assert _flag(argv, "--code-revision") == REVISION
    assert Path(_flag(argv, "--deploy-config")) == (
        REPO_ROOT / "benchmarks/duplex_session/deploy/minicpmo_4_5_gptq_32gb.yaml"
    )
    assert Path(_flag(argv, "--chat-template")).is_file()
    assert _flag(argv, "--chat-template-content-format") == "openai"
    assert _flag(argv, "--port") == "8099"


def test_port_and_extra_arguments_reach_the_cli(tmp_path: Path) -> None:
    argv = _served_argv(tmp_path, "--log-level", "debug", PORT="8123")

    assert _flag(argv, "--port") == "8123"
    assert argv[-2:] == ["--log-level", "debug"]


def test_expandable_segments_is_the_default_allocator_config(tmp_path: Path) -> None:
    assert _served(tmp_path)[1] == "expandable_segments:True"


def test_caller_allocator_config_is_kept(tmp_path: Path) -> None:
    assert _served(tmp_path, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")[1] == "max_split_size_mb:128"
