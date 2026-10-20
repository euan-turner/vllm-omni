#!/usr/bin/env bash
# Launch a MiniCPM-o 4.5 duplex server for duplex_session benchmarks, with
# optional per-stage torch or Nsight Systems profiling.
#
# Design doc: docs/design/fullduplex.md (Serving); profiling pattern:
# docs/contributing/profiling.md ("Profiling Omni AR/TTS stages").
# External dependencies: vllm-omni CLI, step-audio2==1.0.0 (Stage 2),
# nsys (PROFILE=cuda only).
#
# Environment:
#   LAYOUT          1gpu | 2gpu (default 2gpu)
#   PROFILE         off | torch | cuda (default off)
#   PROFILE_STAGES  comma-separated stage ids given profiler_config (default 0)
#   PROFILE_DIR     output root for traces (default /tmp/duplex_session_profile)
#   MODEL, PORT     default openbmb/MiniCPM-o-4_5, 8099
#   DRY_RUN=1       print the command instead of running it
# Extra arguments are passed through to `vllm-omni serve`.
set -euo pipefail

LAYOUT=${LAYOUT:-2gpu}
PROFILE=${PROFILE:-off}
PROFILE_STAGES=${PROFILE_STAGES:-0}
PROFILE_DIR=${PROFILE_DIR:-/tmp/duplex_session_profile}
MODEL=${MODEL:-openbmb/MiniCPM-o-4_5}
PORT=${PORT:-8099}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

case "$LAYOUT" in
  1gpu) DEPLOY="$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_1gpu_32gb.yaml" ;;
  2gpu) DEPLOY="$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_2gpu_32gb.yaml" ;;
  *) echo "LAYOUT must be 1gpu or 2gpu, got '$LAYOUT'" >&2; exit 2 ;;
esac

if ! [[ "$PROFILE_STAGES" =~ ^[0-2](,[0-2])*$ ]]; then
  echo "PROFILE_STAGES must be comma-separated ids in 0..2, got '$PROFILE_STAGES'" >&2
  exit 2
fi

case "$PROFILE" in
  off) PROFILER_CONFIG="" ;;
  torch)
    PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR/torch\",\
\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":false,\
\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true}"
    ;;
  cuda) PROFILER_CONFIG='{"profiler":"cuda"}' ;;
  *) echo "PROFILE must be off, torch or cuda, got '$PROFILE'" >&2; exit 2 ;;
esac

CMD=(vllm-omni serve "$MODEL" --omni
  --deploy-config "$DEPLOY"
  --trust-remote-code
  --chat-template "$REPO_ROOT/vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"
  --chat-template-content-format openai
  --host 0.0.0.0 --port "$PORT")

if [[ -n "$PROFILER_CONFIG" ]]; then
  OVERRIDES=""
  IFS=',' read -ra STAGE_IDS <<< "$PROFILE_STAGES"
  for stage_id in "${STAGE_IDS[@]}"; do
    OVERRIDES+="${OVERRIDES:+,}\"$stage_id\":{\"profiler_config\":$PROFILER_CONFIG}"
  done
  CMD+=(--stage-overrides "{$OVERRIDES}")
fi

CMD+=("$@")

if [[ "$PROFILE" == cuda ]]; then
  CMD=(nsys profile --trace=cuda,nvtx,osrt --sample=process-tree
    --cpuctxsw=process-tree --trace-fork-before-exec=true
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown
    --cuda-graph-trace=node
    -o "$PROFILE_DIR/nsys_${LAYOUT}_$(date +%Y%m%d_%H%M%S)"
    "${CMD[@]}")
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

if [[ "$PROFILE" != off ]]; then
  mkdir -p "$PROFILE_DIR"
fi
cd "$REPO_ROOT"
exec "${CMD[@]}"
