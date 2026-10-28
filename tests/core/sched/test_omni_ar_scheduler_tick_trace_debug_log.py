# SPDX-License-Identifier: Apache-2.0
"""Tests for the prototype duplex debug log helper and the Stage-0 listen->tts_bos rewrite counter.

Prototype on branch long-context-prototype. External dependencies: torch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.sched.duplex_debug_log import DEBUG_LOG_ENV, PeriodicDebugLog, footprint, rss_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_disabled_without_env(monkeypatch) -> None:
    monkeypatch.delenv(DEBUG_LOG_ENV, raising=False)
    log = PeriodicDebugLog("x")
    assert not log.enabled and not log.due()
    monkeypatch.setenv(DEBUG_LOG_ENV, "not-a-number")
    assert not PeriodicDebugLog("x").enabled


def test_due_at_most_once_per_interval() -> None:
    now = [0.0]
    log = PeriodicDebugLog("x", interval_s=30, clock=lambda: now[0])
    assert not log.due()
    now[0] = 30.0
    assert log.due() and not log.due()
    now[0] = 59.0
    assert not log.due()
    now[0] = 60.0
    assert log.due()


def test_footprint_counts_tensors_once_and_scalar_lists_as_items() -> None:
    shared = torch.zeros(4, dtype=torch.float32)
    state = {"a": {"t": shared, "again": shared}, "hist": [1, 2, 3], "obj": SimpleNamespace(t=torch.zeros(2))}
    assert footprint(state) == {"cpu_bytes": 16 + 8, "device_bytes": 0, "tensors": 2, "items": 3}


def test_rss_bytes_is_positive_on_linux() -> None:
    value = rss_bytes()
    assert value is None or value > 0


def test_stage0_counter_separates_rewritten_listens_from_sampled_tts_bos(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    monkeypatch.setenv(DEBUG_LOG_ENV, "3600")
    model = SimpleNamespace()
    count = MiniCPMO45OmniForConditionalGeneration._minicpmo45_duplex_debug_count
    ids = {"listen_token_id": 1, "tts_bos_token_id": 9}
    for raw, final in [(1, 9), (1, 9), (1, 1), (9, 9), (5, 5)]:
        count(model, raw, final, ids)
    _, counts = model._minicpmo45_duplex_debug
    assert counts == {"samples": 5, "listen_sampled": 3, "listen_rewritten_to_tts_bos": 2, "tts_bos_sampled": 1}
