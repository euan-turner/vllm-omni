# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Basic-mode KV compaction keeps the whole first-append context, suffix included.

The first unit starts after the system prefix, reference audio *and* the session
suffix (``<|audio_end|><|im_end|>`` ...). A compaction plan anchored on the
prefix alone puts the sink and the unit boundaries ``len(suffix)`` tokens too
early, so the freed KV range starts inside the suffix and every cut lands
``len(suffix)`` tokens before a real unit start.

Design doc: docs/design/fullduplex.md (Stage-0 sliding window).
External dependencies: none.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
    MiniCPMO45DuplexSchedulerHelper,
    MiniCPMO45DuplexWindowManager,
    MiniCPMO45DuplexWindowPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = 16
PREFIX_TOKENS = 64  # system prefix + ref audio: a whole number of blocks
SUFFIX_TOKENS = 2  # <|audio_end|><|im_end|>
CONTEXT_TOKENS = PREFIX_TOKENS + SUFFIX_TOKENS
UNIT_TOKENS = 79  # 64 image + 10 audio + markers, as measured on a live session
HIGH_WATERMARK = 8000
LOW_WATERMARK = 6000


def _runtime_config() -> dict[str, object]:
    return {
        "duplex_first_append_context_tokens": CONTEXT_TOKENS,
        "duplex_window_prefix_tokens": PREFIX_TOKENS,
        "duplex_window_config": {
            "sliding_window_mode": "basic",
            "basic_window_high_tokens": HIGH_WATERMARK,
            "basic_window_low_tokens": LOW_WATERMARK,
        },
    }


def _plan(num_units: int):
    computed = CONTEXT_TOKENS + num_units * UNIT_TOKENS
    session = SimpleNamespace(num_computed_tokens=computed)
    update = SimpleNamespace(
        prompt_token_ids=[0] * UNIT_TOKENS,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "seq": num_units + 1,
                "runtime_config": _runtime_config(),
            }
        },
    )
    return MiniCPMO45DuplexWindowPolicy.plan_reanchor(
        session=session,
        update=update,
        block_size=BLOCK_SIZE,
        max_model_len=32000,
        unit_tokens=[UNIT_TOKENS] * num_units,
    )


def test_compaction_never_frees_the_session_suffix() -> None:
    plan = _plan(num_units=101)

    assert plan is not None
    assert plan.sink_end >= CONTEXT_TOKENS


def test_compaction_cuts_on_a_real_unit_start() -> None:
    plan = _plan(num_units=101)

    assert plan is not None
    assert (plan.moved_from - CONTEXT_TOKENS) % UNIT_TOKENS == 0
    assert plan.moved_from % BLOCK_SIZE == 0
    assert plan.sink_end - CONTEXT_TOKENS < BLOCK_SIZE
    assert CONTEXT_TOKENS + 101 * UNIT_TOKENS + UNIT_TOKENS - plan.delta <= LOW_WATERMARK


class _BlockListManager(MiniCPMO45DuplexWindowManager):
    def __init__(self) -> None:
        self.block_size = BLOCK_SIZE
        self.blocks = list(range(1024))
        self.req_to_blocks = {"req-1": self.blocks}

    def compact_request_blocks(self, request_id: str, *, sink_blocks: int, num_blocks: int) -> int:
        del self.blocks[sink_blocks : sink_blocks + num_blocks]
        return num_blocks * BLOCK_SIZE


def test_scheduler_unit_ledger_matches_the_kv_after_a_trim() -> None:
    """The scheduler's unit list keeps measuring from the end of the suffix after compaction."""
    closed_units = 101
    open_start = CONTEXT_TOKENS + closed_units * UNIT_TOKENS
    generated = [42]
    # The closing unit spans open_start .. prompt end + generated token + terminator + </unit>.
    computed = open_start + UNIT_TOKENS - len(generated) - 2
    session = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(computed)),
        _all_token_ids=list(range(computed)),
        num_prompt_tokens=computed,
        num_computed_tokens=computed,
        _minicpmo45_window_open_start=open_start,
        _minicpmo45_window_units=[{"length": UNIT_TOKENS, "generated_token_ids": []} for _ in range(closed_units)],
    )
    update = SimpleNamespace(
        prompt_token_ids=[0] * UNIT_TOKENS,
        model_intermediate_buffer={"duplex": {"data_plane": True, "seq": 2, "runtime_config": _runtime_config()}},
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        model_config=SimpleNamespace(max_model_len=32000),
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[_BlockListManager()])),
    )

    plan = MiniCPMO45DuplexSchedulerHelper.apply_session_window(
        scheduler, session, update, segment_output_ids=generated, completed_terminator=None
    )

    assert plan is not None
    lengths = [unit["length"] for unit in session._minicpmo45_window_units]
    # The oldest unit's head stays in the sink page; every later unit is whole.
    assert lengths[0] == plan.sink_end - CONTEXT_TOKENS
    assert set(lengths[1:]) == {UNIT_TOKENS}
    assert CONTEXT_TOKENS + sum(lengths) == session._minicpmo45_window_open_start
