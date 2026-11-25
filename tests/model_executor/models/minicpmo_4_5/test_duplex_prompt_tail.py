# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests that the MiniCPM-o duplex Thinker publishes only its prompt tail to the frontend.

``preprocess`` reports ``duplex_prompt_token_ids`` on every step, and llm2tts
reads only its last token. Publishing the whole padded resumable prompt made
every step serialize O(context) ids. This test drives the duplex prefill branch
with a stubbed Stage-0 helper.

Design doc: docs/design/fullduplex.md.
External dependencies: none (CPU tensors, stubbed helper).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PAD = 0
HIDDEN = 4


class _Helper:
    def __init__(self, append_ids: list[int]) -> None:
        self.append_ids = append_ids

    def take_staged_prefill(self, state, epoch, seq):
        rows = len(self.append_ids)
        return {
            "success": True,
            "inputs_embeds": torch.ones(rows, HIDDEN),
            "input_token_ids": list(self.append_ids),
            "num_input_tokens": rows,
        }

    def stage_padding_token_id(self) -> int:
        return PAD

    def trace_prompt(self, *args) -> None:
        return None


def _model(helper: _Helper) -> MiniCPMO45OmniForConditionalGeneration:
    model = object.__new__(MiniCPMO45OmniForConditionalGeneration)
    object.__setattr__(model, "_modules", {})
    object.__setattr__(model, "_parameters", {})
    object.__setattr__(model, "_buffers", {})
    model.model_stage = "llm"
    model._duplex_data_plane_helper = lambda: helper
    model._commit_minicpmo45_duplex_pending_samples = lambda **_: None
    model._minicpmo45_duplex_session_state = lambda *_: SimpleNamespace()
    model._minicpmo45_duplex_request_session_map = lambda: {}
    model.get_input_embeddings = lambda ids: torch.zeros(int(ids.shape[0]), HIDDEN)
    return model


@pytest.mark.parametrize(("prompt_len", "computed", "span"), [(20, 0, 20), (20, 12, 8), (5000, 4990, 10)])
def test_duplex_prefill_publishes_only_the_prompt_tail(prompt_len: int, computed: int, span: int) -> None:
    append_ids = [501, 502, 503, 504, 505, 506, 507, 508]
    model = _model(_Helper(append_ids))

    _, _, info = model.preprocess(
        torch.zeros(span, dtype=torch.long),
        None,
        duplex={"data_plane": True, "session_id": "s", "epoch": 0, "seq": 3, "payload": {}},
        _omni_prompt_len=prompt_len,
        _omni_num_computed_tokens=computed,
        req_id="r",
    )

    assert info["duplex"]["duplex_prompt_token_ids"] == [508]
