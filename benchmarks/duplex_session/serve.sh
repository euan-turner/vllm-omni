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
#   LAYOUT          1gpu | 2gpu | gh200 (default 2gpu)
#   MAX_MODEL_LEN   Stage-0 max_model_len override (gh200 default 131072). Above
#                   the HF limit (40960) it exports VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
#   TICK_TRACE_DIR  writes Stage-0 per-tick JSONL records there (default off)
#   PROFILE         off | torch | cuda (default off)
#   PROFILE_STAGES  comma-separated stage ids given profiler_config (default 0)
#   PROFILE_DIR     output root for traces (default /tmp/duplex_session_profile)
#   TORCH_WITH_STACK  1 records Python stacks in torch traces (default 0; slows
#                   /stop_profile and enlarges traces)
#   MODEL, PORT     default openbmb/MiniCPM-o-4_5, 8099
#   VLLM_SERVER_DEV_MODE  default 1, exposes the cache-reset endpoints that
#                   profile_run.sh calls; local research servers only
#   DRY_RUN=1       print the command instead of running it
# Extra arguments are passed through to `vllm-omni serve`.
set -euo pipefail

LAYOUT=${LAYOUT:-2gpu}
PROFILE=${PROFILE:-off}
PROFILE_STAGES=${PROFILE_STAGES:-0}
PROFILE_DIR=${PROFILE_DIR:-/tmp/duplex_session_profile}
TORCH_WITH_STACK=${TORCH_WITH_STACK:-0}
MODEL=${MODEL:-openbmb/MiniCPM-o-4_5}
PORT=${PORT:-8099}
export VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-1}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HF_MAX_MODEL_LEN=40960

case "$LAYOUT" in
  1gpu) DEPLOY="$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_1gpu_32gb.yaml" ;;
  2gpu) DEPLOY="$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_2gpu_32gb.yaml" ;;
  gh200)
    DEPLOY="$REPO_ROOT/benchmarks/duplex_session/deploy/minicpmo_4_5_gh200_long_context.yaml"
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
    # Stage-0 max_position_embeddings in that deploy file (RoPE table size).
    MAX_POSITIONS=131072
    ;;
  *) echo "LAYOUT must be 1gpu, 2gpu or gh200, got '$LAYOUT'" >&2; exit 2 ;;
esac
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}

if [[ -n "$MAX_MODEL_LEN" ]]; then
  if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_MODEL_LEN must be a positive integer, got '$MAX_MODEL_LEN'" >&2
    exit 2
  fi
  if [[ -n "${MAX_POSITIONS:-}" ]] && (( MAX_MODEL_LEN > MAX_POSITIONS )); then
    echo "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the deploy file's max_position_embeddings ($MAX_POSITIONS)" >&2
    exit 2
  fi
  if (( MAX_MODEL_LEN > HF_MAX_MODEL_LEN )); then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  fi
fi

if [[ -n "${TICK_TRACE_DIR:-}" ]]; then
  export VLLM_OMNI_DUPLEX_TICK_TRACE_DIR="$TICK_TRACE_DIR"
fi

if ! [[ "$PROFILE_STAGES" =~ ^[0-2](,[0-2])*$ ]]; then
  echo "PROFILE_STAGES must be comma-separated ids in 0..2, got '$PROFILE_STAGES'" >&2
  exit 2
fi

case "$TORCH_WITH_STACK" in
  0) WITH_STACK=false ;;
  1) WITH_STACK=true ;;
  *) echo "TORCH_WITH_STACK must be 0 or 1, got '$TORCH_WITH_STACK'" >&2; exit 2 ;;
esac

case "$PROFILE" in
  off) PROFILER_CONFIG="" ;;
  torch)
    PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR/torch\",\
\"torch_profiler_with_stack\":$WITH_STACK,\"torch_profiler_record_shapes\":false,\
\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false}"
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

declare -A STAGE_FIELDS=()
if [[ -n "$PROFILER_CONFIG" ]]; then
  IFS=',' read -ra STAGE_IDS <<< "$PROFILE_STAGES"
  for stage_id in "${STAGE_IDS[@]}"; do
    STAGE_FIELDS[$stage_id]="\"profiler_config\":$PROFILER_CONFIG"
  done
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  STAGE_FIELDS[0]="${STAGE_FIELDS[0]:+${STAGE_FIELDS[0]},}\"max_model_len\":$MAX_MODEL_LEN"
fi
if (( ${#STAGE_FIELDS[@]} > 0 )); then
  OVERRIDES=""
  for stage_id in $(printf '%s\n' "${!STAGE_FIELDS[@]}" | sort -n); do
    OVERRIDES+="${OVERRIDES:+,}\"$stage_id\":{${STAGE_FIELDS[$stage_id]}}"
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
  echo "env: VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-} VLLM_OMNI_DUPLEX_TICK_TRACE_DIR=${VLLM_OMNI_DUPLEX_TICK_TRACE_DIR:-}" >&2
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

if [[ "$PROFILE" != off ]]; then
  mkdir -p "$PROFILE_DIR"
fi
cd "$REPO_ROOT"
exec "${CMD[@]}"
