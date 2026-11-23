# SPDX-License-Identifier: Apache-2.0
"""Tests that the GPTQ deploy's Talker override matches the checkpoint served by serve.sh.

Design doc: docs/design/fullduplex.md (Serving).
External dependencies: huggingface_hub (reads the pinned checkpoint's weight
index from the cache or the Hub; skipped when neither is reachable), vllm.
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


def _gptq_modules() -> set[str]:
    """Top-level checkpoint modules holding GPTQ-packed weights."""
    model, revision = _served_checkpoint()
    try:
        index = hf_hub_download(model, "model.safetensors.index.json", revision=revision)
    except LocalEntryNotFoundError:
        pytest.skip(f"weight index for {model}@{revision} is not cached and the Hub is unreachable")
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    return {key.split(".", 1)[0] for key in weight_map if key.endswith(".qweight")}


def test_talker_override_matches_served_checkpoint() -> None:
    stages = {stage["stage_id"]: stage for stage in resolve_deploy_yaml(DEPLOY)["stages"]}
    talker_overrides = stages[1].get("hf_overrides", {})

    assert "quantization_config" in talker_overrides and talker_overrides["quantization_config"] is None
    assert _gptq_modules() == {"llm"}, (
        "The Stage 1 quantization_config override assumes only llm.* is GPTQ-quantized; "
        "update the deploy config for this checkpoint."
    )
