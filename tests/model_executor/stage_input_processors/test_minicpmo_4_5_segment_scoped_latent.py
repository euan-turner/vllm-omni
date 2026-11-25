# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests that bounded Stage-0 output accumulation gives llm2tts the same Talker condition.

A native-duplex Stage-0 request accumulates its outputs in the frontend
``OmniRequestState``. The MiniCPM-o plugin declares the hidden-row ledger keys
segment-scoped (restart at each applied streaming update) and the re-sent
prompt tail and special ids snapshot keys (latest value only). These tests
script a multi-append session through a real ``OmniRequestState`` with and
without those policies and check that every speak handoff is identical, that
retained rows stay bounded by one segment, and that the prompt ids llm2tts
reads stay one token instead of growing with every step.

Design doc: docs/design/fullduplex.md.
External dependencies: none (CPU tensors only).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SPEAK, LISTEN, CHUNK_EOS, TURN_EOS = 9304, 9303, 9308, 9310
_META = {
    "tts_bos_token_id": 9301,
    "tts_eos_token_id": 9302,
    "listen_token_id": LISTEN,
    "speak_token_id": SPEAK,
    "chunk_eos_token_id": CHUNK_EOS,
    "chunk_tts_eos_token_id": 9309,
    "turn_eos_token_id": TURN_EOS,
}
HIDDEN = 4


@dataclass
class _Segment:
    """One append: its prompt rows (prefilled in chunks) and the unit it decodes."""

    prompt_ids: list[int]
    output_ids: list[int]
    prefill_chunks: int = 1
    forward_terminator: bool = False


def _state() -> OmniRequestState:
    detok = MagicMock(output_token_ids=[0], get_next_output_text=MagicMock(return_value=""))
    detok.num_output_tokens = MagicMock(return_value=1)
    return OmniRequestState(
        request_id="r",
        external_req_id="r",
        parent_req=None,
        request_index=0,
        lora_request=None,
        prompt=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        logprobs_processor=MagicMock(logprobs=None, cumulative_logprob=None, prompt_logprobs=None),
        detokenizer=detok,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        output_kind=RequestOutputKind.CUMULATIVE,
    )


class _Session:
    """Feeds Thinker steps into one request state, as the Stage-0 output processor would."""

    def __init__(self, *, segment_scoped: bool) -> None:
        self.state = _state()
        if segment_scoped:
            self.state.segment_scoped_keys = MiniCPMO45DuplexPlugin.stage0_segment_scoped_output_keys
            self.state.snapshot_keys = MiniCPMO45DuplexPlugin.stage0_snapshot_output_keys
        self.position = 0
        self.pending_lookahead: list[int] = []
        self.prompt_tail: list[int] = []

    def _step(self, token_ids: list[int]) -> None:
        rows = len(token_ids)
        start = self.position
        self.position += rows
        latent = torch.arange(start * HIDDEN, self.position * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN)
        self.state.add_multimodal_tensor(
            {
                "latent": latent,
                "latent_input_ids": torch.tensor(token_ids).reshape(-1, 1),
                "latent_positions": torch.arange(start, self.position).reshape(-1, 1),
                # The V1 Thinker re-sends the append's prompt tail and the
                # tokenizer's special ids on every step.
                "duplex_prompt_token_ids": torch.tensor(self.prompt_tail),
                "meta": {key: torch.tensor([value]) for key, value in _META.items()},
            },
            mm_type=OutputModalityNames.LATENT,
        )

    def run(self, segment: _Segment) -> dict:
        """Prefill the append (after any lookahead row), decode the unit, return the finish snapshot."""
        prompt_rows = self.pending_lookahead + segment.prompt_ids
        self.pending_lookahead = []
        self.prompt_tail = segment.prompt_ids[-1:]
        bounds = torch.linspace(0, len(prompt_rows), segment.prefill_chunks + 1).round().int().tolist()
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            if hi > lo:
                self._step(prompt_rows[lo:hi])
        # The first unit token is sampled from the last prompt row; each sampled
        # token is fed back by the next decode step, except the terminator.
        for token in segment.output_ids[:-1]:
            self._step([token])
        if segment.forward_terminator:
            self.pending_lookahead = [segment.output_ids[-1]]
        output = self.state.make_request_output([1], None, FinishReason.STOP, None)
        snapshot = dict(output.outputs[0].multimodal_output)
        self.state.apply_streaming_update(
            SimpleNamespace(final=False, prompt=None, prompt_token_ids=[0], arrival_time=0.0)
        )
        return snapshot

    def retained_rows(self) -> int:
        latent = self.state.mm_accumulated.tensors.get("latent")
        if latent is None:
            return 0
        return sum(int(t.shape[0]) for t in latent) if isinstance(latent, list) else int(latent.shape[0])


def _handoff(snapshot: dict, segment: _Segment) -> dict:
    completion = SimpleNamespace(token_ids=segment.output_ids, text="", multimodal_output=snapshot)
    source = SimpleNamespace(request_id="r", prompt_token_ids=segment.prompt_ids, outputs=[completion])
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 0, "model_turn_id": 1}})
    return llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]


def _audio(n: int, base: int) -> list[int]:
    return [base + k for k in range(n)]


SCRIPT = [
    _Segment(_audio(12, 1000), [LISTEN]),
    _Segment(_audio(12, 1100), [SPEAK, 21, 22, CHUNK_EOS], forward_terminator=True),
    _Segment(_audio(12, 1200), [LISTEN]),
    _Segment(_audio(40, 1300), [SPEAK, 31, TURN_EOS, 77, CHUNK_EOS], prefill_chunks=3),
    _Segment(_audio(12, 1400), [LISTEN], prefill_chunks=2),
    _Segment(_audio(12, 1500), [SPEAK, 21, 22, CHUNK_EOS]),
    _Segment(_audio(12, 1600), [SPEAK, 21, 22, CHUNK_EOS], forward_terminator=True),
]


def test_plugin_declares_the_three_ledger_keys_segment_scoped() -> None:
    assert MiniCPMO45DuplexPlugin.stage0_segment_scoped_output_keys == {
        "latent",
        "latent_input_ids",
        "latent_positions",
    }
    assert MiniCPMO45DuplexPlugin.stage0_snapshot_output_keys == {"duplex_prompt_token_ids", "meta"}


def test_every_speak_handoff_matches_full_history() -> None:
    full, scoped = _Session(segment_scoped=False), _Session(segment_scoped=True)
    speak_handoffs = 0
    for segment in SCRIPT:
        full_snapshot, scoped_snapshot = full.run(segment), scoped.run(segment)
        if segment.output_ids == [LISTEN]:
            continue
        reference, candidate = _handoff(full_snapshot, segment), _handoff(scoped_snapshot, segment)
        # Only the latest prompt tail reaches llm2tts; without the policy every
        # step's copy is concatenated, growing with the session.
        assert torch.as_tensor(scoped_snapshot["duplex_prompt_token_ids"]).numel() == 1
        assert torch.as_tensor(full_snapshot["duplex_prompt_token_ids"]).numel() > 1
        # The native Talker request carries the handoff, not the Thinker prompt.
        assert "prompt_token_ids" not in candidate and "prompt" not in candidate["ids"]
        assert candidate["ids"]["tts"] == reference["ids"]["tts"]
        assert torch.equal(
            torch.as_tensor(candidate["hidden_states"]["tts"]), torch.as_tensor(reference["hidden_states"]["tts"])
        )
        speak_handoffs += 1
    assert speak_handoffs == 4


def test_retained_rows_are_bounded_by_one_segment() -> None:
    full, scoped = _Session(segment_scoped=False), _Session(segment_scoped=True)
    for segment in SCRIPT * 5:
        full_snapshot = full.run(segment)
        scoped_snapshot = scoped.run(segment)
        # After the update only a lookahead row can be held, and it arrives
        # with the next step, so nothing is retained between segments.
        assert scoped.retained_rows() == 0
        assert torch.as_tensor(scoped_snapshot["duplex_prompt_token_ids"]).numel() == 1
    assert full.retained_rows() == full.position
    # Unbounded: one prompt-tail copy per step for the whole session.
    assert torch.as_tensor(full_snapshot["duplex_prompt_token_ids"]).numel() > len(SCRIPT * 5)
