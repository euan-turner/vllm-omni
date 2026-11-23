# SPDX-License-Identifier: Apache-2.0
"""Tests the command built by benchmarks/duplex_session/serve.sh.

Design doc: docs/design/fullduplex.md (Serving).
External dependencies: bash.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SH = REPO_ROOT / "benchmarks/duplex_session/serve.sh"
REVISION = "02b54c54c36f8b48e97501bb0fde5d178c62df38"


def _served_argv(tmp_path: Path, *args: str, **env: str) -> list[str]:
    """Run serve.sh against a stub CLI and return the arguments it received."""
    stub = tmp_path / "vllm-omni"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    clean_env = {k: v for k, v in os.environ.items() if k != "PORT"}
    result = subprocess.run(
        ["bash", str(SERVE_SH), *args],
        env={**clean_env, "PATH": f"{tmp_path}:{os.environ['PATH']}", **env},
        capture_output=True,
        check=True,
    )
    return result.stdout.decode().split("\0")[:-1]


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
