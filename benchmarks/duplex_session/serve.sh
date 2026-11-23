#!/usr/bin/env bash
# Serve the W4A16 GPTQ MiniCPM-o 4.5 duplex pipeline on one 32 GB GPU.
#
# Design doc: docs/design/fullduplex.md (Serving).
# External dependencies: vllm-omni CLI, step-audio2==1.0.0 (Stage 2).
#
# Environment: PORT (default 8099).
# Extra arguments are passed through to `vllm-omni serve`.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=openbmb/MiniCPM-o-4_5-GPTQ
MODEL_REVISION=02b54c54c36f8b48e97501bb0fde5d178c62df38

cd "$REPO_ROOT"
exec vllm-omni serve "$MODEL" --omni \
  --revision "$MODEL_REVISION" \
  --code-revision "$MODEL_REVISION" \
  --deploy-config "$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_gptq_32gb.yaml" \
  --trust-remote-code \
  --chat-template "$REPO_ROOT/vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja" \
  --chat-template-content-format openai \
  --host 0.0.0.0 \
  --port "${PORT:-8099}" \
  "$@"
