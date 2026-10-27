#!/usr/bin/env bash
#SBATCH --job-name=duplex-long-context
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Add your project's account/partition with sbatch flags, e.g. `sbatch -A <project> ...`.
#
# Long-context full-duplex run on one Isambard-AI GH200: serve MiniCPM-o 4.5
# with Stage-0 tick tracing, warm up, stream stitched RTD_Omni sessions in real
# time, then analyse ticks. Prototype (branch long-context-prototype).
#
# External dependencies: Slurm; one of podman-hpc (RUNTIME=podman-hpc, image
# from Containerfile), apptainer (RUNTIME=apptainer, SIF from that image) or a
# local venv (RUNTIME=venv); model and dataset already in $HF_HOME (compute
# nodes are assumed offline, so HF_HUB_OFFLINE=1).
#
# Environment:
#   REPO           repo checkout (default: $SLURM_SUBMIT_DIR, i.e. submit from the repo root)
#   RUNTIME        podman-hpc | apptainer | venv (default podman-hpc)
#   IMAGE          podman-hpc image name or apptainer .sif path (default vllm-omni-lc)
#   VENV           venv path for RUNTIME=venv (default $REPO/.venv)
#   MODALITY       av | audio (default av)
#   TARGET_MINUTES stream length (default 30 for av, 120 for audio)
#   SESSIONS       concurrent long sessions (default 1)
#   MAX_MODEL_LEN  Stage-0 context (default 131072; >40960 is explicit overflow)
#   DATASET        local Omni-DuplexEval mirror (default $HF_HOME/omni_duplex_eval)
#   REF_AUDIO      MiniCPM-o reference voice wav (required)
#   OUT_ROOT       results root (default $REPO/runs/long_context)
set -euo pipefail

# Slurm runs a copy of this script from its spool directory, so locate the repo
# from the submit directory rather than the script path.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
[[ -f "$REPO/benchmarks/duplex_session/serve.sh" ]] || { echo "REPO=$REPO is not the vllm-omni checkout; submit from the repo root or set REPO" >&2; exit 2; }
RUNTIME=${RUNTIME:-podman-hpc}
IMAGE=${IMAGE:-vllm-omni-lc}
VENV=${VENV:-$REPO/.venv}
MODALITY=${MODALITY:-av}
if [[ "$MODALITY" == av ]]; then TARGET_MINUTES=${TARGET_MINUTES:-30}; else TARGET_MINUTES=${TARGET_MINUTES:-120}; fi
SESSIONS=${SESSIONS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
DATASET=${DATASET:-$HF_HOME/omni_duplex_eval}
REF_AUDIO=${REF_AUDIO:?set REF_AUDIO to the MiniCPM-o reference voice wav}
OUT_ROOT=${OUT_ROOT:-$REPO/runs/long_context}
PORT=${PORT:-8099}

RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODALITY}_${TARGET_MINUTES}min_s${SESSIONS}_${SLURM_JOB_ID:-local}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
TRACE_DIR="$RUN_DIR/ticks"
mkdir -p "$TRACE_DIR"
export HF_HUB_OFFLINE=1

# podman-hpc --gpu mounts the host driver libraries (libnvidia-ml, libcuda) in
# /usr/lib64, which the Ubuntu-based image does not search, so vLLM fails with
# "Failed to infer device type". Prepend it inside the container.
GPU_LIB_DIR=/usr/lib64
GPU_LIB_SHIM='export LD_LIBRARY_PATH="'"$GPU_LIB_DIR"'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec "$@"'

in_env() {
  case "$RUNTIME" in
    podman-hpc)
      podman-hpc run --rm --gpu --net host --ipc host \
        -v "$REPO:$REPO" -v "$HF_HOME:$HF_HOME" -v "$OUT_ROOT:$OUT_ROOT" -v "$(dirname "$REF_AUDIO"):$(dirname "$REF_AUDIO")" \
        -e HF_HOME -e HF_HUB_OFFLINE -e PYTHONPATH="$REPO" -w "$REPO" "$IMAGE" bash -c "$GPU_LIB_SHIM" _ "$@" ;;
    apptainer)
      apptainer exec --nv --bind "$REPO,$HF_HOME,$OUT_ROOT,$(dirname "$REF_AUDIO")" \
        --env PYTHONPATH="$REPO" --pwd "$REPO" "$IMAGE" "$@" ;;
    venv) (cd "$REPO" && PATH="$VENV/bin:$PATH" PYTHONPATH="$REPO" "$@") ;;
    *) echo "RUNTIME must be podman-hpc, apptainer or venv" >&2; exit 2 ;;
  esac
}

in_env env LAYOUT=gh200 MAX_MODEL_LEN="$MAX_MODEL_LEN" TICK_TRACE_DIR="$TRACE_DIR" PORT="$PORT" \
  bash benchmarks/duplex_session/serve.sh > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 180); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server exited; see $RUN_DIR/server.log" >&2; exit 1; }
  sleep 10
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "server not healthy after 30 min" >&2; exit 1; }

DRIVER=benchmarks/duplex_session/long_context/run_long_session.py
COMMON=(--dataset "$DATASET" --split RTD_Omni --ref-audio "$REF_AUDIO" --url "ws://localhost:$PORT/v1/realtime?duplex=1" --modality "$MODALITY")

# Warm-up: its ticks land in the same trace files but outside the measured
# sessions' time windows, so analyze.py drops them.
in_env python "$DRIVER" generate "${COMMON[@]}" --target-minutes 1 --out-dir "$RUN_DIR/warmup" || true

in_env python "$DRIVER" generate "${COMMON[@]}" --target-minutes "$TARGET_MINUTES" --sessions "$SESSIONS" \
  --tick-trace-dir "$TRACE_DIR" --out-dir "$RUN_DIR/measured" || echo "driver reported session errors (see manifests)"

in_env python benchmarks/duplex_session/long_context/analyze.py "$RUN_DIR/measured" \
  --tick-trace-dir "$TRACE_DIR" --model openbmb/MiniCPM-o-4_5
echo "results: $RUN_DIR"
