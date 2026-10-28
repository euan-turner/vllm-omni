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
#   MAX_MODEL_LEN  Stage-0 context (default 131072; >40960 extends RoPE: diagnostics only)
#   DATASET        local Omni-DuplexEval mirror (default $HF_HOME/omni_duplex_eval)
#   REF_AUDIO      MiniCPM-o reference voice wav (required)
#   OUT_ROOT       results root (default $REPO/runs/long_context)
#   INPUT_WAIT_TIMEOUT_S  server stage input-wait timeout (default 0 = off: a
#                  silent model starves Stage 2 for minutes, which is not a hang)
#   ALLOC_CONF     server PYTORCH_CUDA_ALLOC_CONF (default expandable_segments:True)
#   TICK_STALL_S   driver hang detection: abort after this long without a
#                  Stage-0 tick record (default 120; 0 disables)
#   DEBUG_LOG_S    server [duplex-debug] lines every N s: Stage-0 listen->tts_bos
#                  rewrites, Talker state sizes, per-stage RSS (default 30; 0 off)
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
INPUT_WAIT_TIMEOUT_S=${INPUT_WAIT_TIMEOUT_S:-0}
# Stage 0 allocates per-tick temporaries sized by context; with fixed-size
# segments the caching allocator reserves ~25 GiB extra by 30k tokens and
# starves Stage 1/2 (allocated stays flat). Expandable segments reuse memory.
ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True}
TICK_STALL_S=${TICK_STALL_S:-120}
DEBUG_LOG_S=${DEBUG_LOG_S:-30}

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
  VLLM_OMNI_INPUT_WAIT_TIMEOUT_S="$INPUT_WAIT_TIMEOUT_S" PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  VLLM_OMNI_DUPLEX_DEBUG_LOG_S="$DEBUG_LOG_S" bash benchmarks/duplex_session/serve.sh > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
# Per-process GPU memory every 30 s (host nvidia-smi sees the container's stage processes).
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv -l 30 \
  > "$RUN_DIR/gpu_mem.csv" 2>&1 &
GPU_MON_PID=$!
# Host memory every 30 s: RSS of this user's 8 largest processes, plus
# /dev/shm usage and the job cgroup's total (Slurm's OOM killer acts on the
# cgroup, which also charges shared memory that no process RSS shows; stages
# exchange data through shared memory with --ipc host).
CGROUP_MEM=/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)/memory.current
[[ -r "$CGROUP_MEM" ]] || CGROUP_MEM=/sys/fs/cgroup/memory$(awk -F: '$2~/memory/{print $3}' /proc/self/cgroup)/memory.usage_in_bytes
while sleep 30; do
  ts=$(date +%FT%T)
  ps -u "$USER" -o rss=,pid=,args= --sort=-rss | head -8 \
    | awk -v ts="$ts" '{a=$3; for(i=4;i<=NF;i++) a=a" "$i; printf "%s,%d,%s,%s\n", ts, $1/1024, $2, substr(a,1,80)}'
  echo "$ts,$(df -m --output=used /dev/shm | tail -1 | tr -d ' '),-,[/dev/shm used]"
  [[ -r "$CGROUP_MEM" ]] && echo "$ts,$(( $(cat "$CGROUP_MEM") / 1048576 )),-,[job cgroup total]"
done > "$RUN_DIR/host_mem.csv" 2>&1 &
HOST_MON_PID=$!
# Keep the job's exit status: under set -e, the killed server's 143 from `wait` would replace it.
trap 'rc=$?; kill $GPU_MON_PID $HOST_MON_PID $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true; exit $rc' EXIT

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
  --tick-trace-dir "$TRACE_DIR" --tick-stall-s "$TICK_STALL_S" --out-dir "$RUN_DIR/measured" || echo "driver reported session errors (see manifests)"

in_env python benchmarks/duplex_session/long_context/analyze.py "$RUN_DIR/measured" \
  --tick-trace-dir "$TRACE_DIR" --model openbmb/MiniCPM-o-4_5
echo "results: $RUN_DIR"
