# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/duplex_session/profile_run.sh in DRY_RUN mode.

Design doc: docs/design/fullduplex.md; plan: duplex_session re-scope (2026-10-22).
External dependencies: bash.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_RUN_SH = REPO_ROOT / "benchmarks/duplex_session/profile_run.sh"
BASE_ENV = {"DRY_RUN": "1", "DATASET": "/data/RTD_OCR.parquet", "REF_AUDIO": "/data/ref.wav"}


def _dry_run(**env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROFILE_RUN_SH)],
        env={**os.environ, **BASE_ENV, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _commands(**env: str) -> list[list[str]]:
    result = _dry_run(**env)
    assert result.returncode == 0, result.stderr
    return [shlex.split(line) for line in result.stdout.splitlines() if line and not line.startswith("#")]


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _kind(argv: list[str]) -> str:
    if argv[0] == "curl":
        return argv[argv.index("-X") + 2].rsplit("/", 1)[-1]
    if argv[:3] == ["vllm", "bench", "omni-duplex-eval"]:
        return "generate:" + Path(_flag(argv, "--response-root")).name
    return argv[0]


def test_measured_run_is_bracketed_by_profile_calls_after_warmup() -> None:
    kinds = [_kind(argv) for argv in _commands(CONCURRENCY="2")]
    assert kinds == [
        "mkdir",
        "generate:warmup",
        "reset_prefix_cache",
        "reset_mm_cache",
        "reset_encoder_cache",
        "touch",
        "start_profile",
        "generate:measured",
        "stop_profile",
    ]


def test_measured_run_uses_concurrency_and_limit_and_warmup_is_serial() -> None:
    commands = [argv for argv in _commands(CONCURRENCY="4", LIMIT="8", WARMUP="2") if argv[0] == "vllm"]
    warmup, measured = commands
    assert (_flag(warmup, "--limit"), _flag(warmup, "--concurrency")) == ("2", "1")
    assert (_flag(measured, "--limit"), _flag(measured, "--concurrency")) == ("8", "4")
    assert _flag(measured, "--dataset") == BASE_ENV["DATASET"]
    assert _flag(measured, "--ref-audio") == BASE_ENV["REF_AUDIO"]


def test_limit_defaults_to_concurrency() -> None:
    measured = [argv for argv in _commands(CONCURRENCY="3") if argv[0] == "vllm"][-1]
    assert _flag(measured, "--limit") == "3"


def test_caches_are_reset_after_warmup_and_before_profiling() -> None:
    kinds = [_kind(argv) for argv in _commands()]
    resets = [kinds.index(name) for name in ("reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache")]
    assert kinds.index("generate:warmup") < min(resets)
    assert max(resets) < kinds.index("start_profile")


def test_warmup_zero_skips_warmup_generate() -> None:
    kinds = [_kind(argv) for argv in _commands(WARMUP="0")]
    assert "generate:warmup" not in kinds
    assert kinds.index("start_profile") < kinds.index("generate:measured")


def test_profile_body_names_the_requested_stages() -> None:
    start = next(argv for argv in _commands(STAGES="0,1") if _kind(argv) == "start_profile")
    assert _flag(start, "-d") == '{"stages":[0,1]}'


@pytest.mark.parametrize(
    "env",
    [
        {"STAGES": "3"},
        {"STAGES": "0;1"},
        {"CONCURRENCY": "0"},
        {"LIMIT": "x"},
        {"WARMUP": "-1"},
        {"DATASET": ""},
        {"REF_AUDIO": ""},
    ],
)
def test_invalid_settings_are_rejected(env: dict[str, str]) -> None:
    result = _dry_run(**env)
    assert result.returncode != 0
    assert result.stdout == ""
