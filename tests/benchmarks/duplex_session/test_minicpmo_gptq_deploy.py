# SPDX-License-Identifier: Apache-2.0
"""Tests the GPTQ deploy against the checkpoint served by serve.sh.

Checks the Talker quantization override and that the Thinker KV pool physically
holds every admitted session at full context.

External dependencies: huggingface_hub (reads the pinned checkpoint's weight
index and config from the cache or the Hub; skipped when neither is reachable), vllm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from vllm_omni.config.stage_config import resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SH = REPO_ROOT / "benchmarks/duplex_session/serve.sh"
DEPLOY = REPO_ROOT / "benchmarks/duplex_session/deploy/minicpmo_4_5_gptq_32gb.yaml"


def _served_checkpoint() -> tuple[str, str]:
    script = SERVE_SH.read_text()
    model = re.search(r"^MODEL=(\S+)$", script, re.MULTILINE)
    revision = re.search(r"^MODEL_REVISION=(\S+)$", script, re.MULTILINE)
    assert model is not None and revision is not None
    return model.group(1), revision.group(1)


def _checkpoint_file(name: str) -> Path:
    model, revision = _served_checkpoint()
    try:
        return Path(hf_hub_download(model, name, revision=revision))
    except LocalEntryNotFoundError:
        pytest.skip(f"{name} for {model}@{revision} is not cached and the Hub is unreachable")


def _gptq_modules() -> set[str]:
    """Top-level checkpoint modules holding GPTQ-packed weights."""
    weight_map = json.loads(_checkpoint_file("model.safetensors.index.json").read_text())["weight_map"]
    return {key.split(".", 1)[0] for key in weight_map if key.endswith(".qweight")}


def test_talker_override_matches_served_checkpoint() -> None:
    stages = {stage["stage_id"]: stage for stage in resolve_deploy_yaml(DEPLOY)["stages"]}
    talker_overrides = stages[1].get("hf_overrides", {})

    assert "quantization_config" in talker_overrides and talker_overrides["quantization_config"] is None
    assert _gptq_modules() == {"llm"}, (
        "The Stage 1 quantization_config override assumes only llm.* is GPTQ-quantized; "
        "update the deploy config for this checkpoint."
    )


#: vLLM's default KV page size; the deploy does not override ``block_size``.
KV_BLOCK_TOKENS = 16
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def test_thinker_kv_holds_every_session_at_full_context() -> None:
    # The duplex Stage-0 attention reports a sliding-window spec, so vLLM's
    # startup fit check and its logged "KV cache size" overstate physical
    # capacity; an undersized pool boots and then stalls sessions. Size it here.
    resolved = resolve_deploy_yaml(DEPLOY)
    thinker = next(stage for stage in resolved["stages"] if stage["stage_id"] == 0)
    cuda_thinker = next(stage for stage in resolved["platforms"]["cuda"]["stages"] if stage["stage_id"] == 0)
    assert "block_size" not in thinker
    config = json.loads(_checkpoint_file("config.json").read_text())
    bytes_per_token = (
        2
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * config["head_dim"]
        * _DTYPE_BYTES[config.get("dtype") or config["torch_dtype"]]
    )

    pages = cuda_thinker["kv_cache_memory_bytes"] // (bytes_per_token * KV_BLOCK_TOKENS)

    assert pages * KV_BLOCK_TOKENS >= resolved["duplex_session"]["max_sessions"] * thinker["max_model_len"]
