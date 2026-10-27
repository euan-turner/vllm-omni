#!/usr/bin/env bash
# Profile concurrent MiniCPM-o duplex sessions on a running serve.sh server:
# warm up, reset the prefix, multimodal and encoder caches, then bracket one
# concurrent omni-duplex-eval generate run with /start_profile and
# /stop_profile, collect the new torch traces, summarise each one and merge
# the stage traces into one Perfetto file (merge_stage_traces.py).
# The cache resets are vLLM dev endpoints: the server needs
# VLLM_SERVER_DEV_MODE=1 (serve.sh sets it).
#
# Design doc: docs/design/fullduplex.md; profiling pattern:
# docs/contributing/profiling.md. Plan: duplex_session re-scope (2026-10-22).
# External dependencies: vllm bench omni-duplex-eval, curl, git,
# .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py.
#
# Environment:
#   DATASET      required: local Omni-DuplexEval .parquet/dir or HF id
#   REF_AUDIO    required: assistant voice WAV (e.g. the model's HT_ref_audio.wav)
#   STAGES       comma-separated stage ids to trace (default 0); must be a
#                subset of the server's PROFILE_STAGES
#   CONCURRENCY  concurrent sessions (default 1)
#   LIMIT        measured samples, one turn each (default CONCURRENCY)
#   WARMUP       warm-up samples before profiling (default 1)
#   SPLIT, FPS   omni-duplex-eval --split (default all) and --fps (default 1)
#   PROFILE_DIR  must match serve.sh (default /tmp/duplex_session_profile)
#   OUT_ROOT     run directories (default /tmp/duplex_session_runs)
#   MODEL, PORT  default openbmb/MiniCPM-o-4_5, 8099
#   TRACE_SETTLE_POLL_S, TRACE_SETTLE_TIMEOUT_S
#                poll interval and limit while traces finish writing
#                after /stop_profile (default 10, 300)
#   DRY_RUN=1    print the commands instead of running them
set -euo pipefail

: "${DATASET:?DATASET must name a local Omni-DuplexEval parquet/dir or HF id}"
: "${REF_AUDIO:?REF_AUDIO must name the assistant voice WAV}"
STAGES=${STAGES:-0}
CONCURRENCY=${CONCURRENCY:-1}
LIMIT=${LIMIT:-$CONCURRENCY}
WARMUP=${WARMUP:-1}
SPLIT=${SPLIT:-all}
FPS=${FPS:-1}
PROFILE_DIR=${PROFILE_DIR:-/tmp/duplex_session_profile}
OUT_ROOT=${OUT_ROOT:-/tmp/duplex_session_runs}
MODEL=${MODEL:-openbmb/MiniCPM-o-4_5}
PORT=${PORT:-8099}
TRACE_SETTLE_POLL_S=${TRACE_SETTLE_POLL_S:-10}
TRACE_SETTLE_TIMEOUT_S=${TRACE_SETTLE_TIMEOUT_S:-300}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ANALYZER="$REPO_ROOT/.claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py"
MERGER="$REPO_ROOT/benchmarks/duplex_session/merge_stage_traces.py"

if ! [[ "$STAGES" =~ ^[0-2](,[0-2])*$ ]]; then
  echo "STAGES must be comma-separated ids in 0..2, got '$STAGES'" >&2
  exit 2
fi
for name in CONCURRENCY LIMIT WARMUP; do
  if ! [[ "${!name}" =~ ^[0-9]+$ ]] || { [[ "$name" != WARMUP ]] && (( ${!name} < 1 )); }; then
    echo "$name must be a positive integer (WARMUP may be 0), got '${!name}'" >&2
    exit 2
  fi
done

BASE="http://localhost:$PORT"
RUN_DIR="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_c${CONCURRENCY}_n${LIMIT}"
BODY="{\"stages\":[$STAGES]}"

run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

generate() {
  local samples=$1 concurrency=$2 root=$3
  run vllm bench omni-duplex-eval --omni generate \
    --url "ws://localhost:$PORT/v1/realtime?duplex=1" --model "$MODEL" \
    --ref-audio "$REF_AUDIO" --dataset "$DATASET" --split "$SPLIT" \
    --fps "$FPS" --limit "$samples" --concurrency "$concurrency" \
    --response-root "$root"
}

profile() {
  run curl -sSf -X POST "$BASE/$1" -H 'Content-Type: application/json' -d "$BODY"
}

reset_caches() {
  local endpoint
  for endpoint in reset_prefix_cache reset_mm_cache reset_encoder_cache; do
    if ! run curl -sSf -X POST "$BASE/$endpoint"; then
      echo "POST /$endpoint failed; start the server with VLLM_SERVER_DEV_MODE=1" >&2
      exit 1
    fi
  done
}

run mkdir -p "$RUN_DIR/traces"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  curl -sSf "$BASE/health" >/dev/null
  cat > "$RUN_DIR/run.json" <<EOF
{"git_sha": "$(git -C "$REPO_ROOT" rev-parse HEAD)", "git_dirty": $([[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]] && echo true || echo false),
 "dataset": "$DATASET", "split": "$SPLIT", "fps": $FPS, "stages": [$STAGES],
 "concurrency": $CONCURRENCY, "limit": $LIMIT, "warmup": $WARMUP, "model": "$MODEL"}
EOF
fi

if (( WARMUP > 0 )); then
  generate "$WARMUP" 1 "$RUN_DIR/warmup"
fi
reset_caches

run touch "$RUN_DIR/.profile_start"
profile start_profile
trap 'profile stop_profile >/dev/null || true' EXIT
start_s=$(date +%s)
generate "$LIMIT" "$CONCURRENCY" "$RUN_DIR/measured"
echo "measured window: $(( $(date +%s) - start_s )) s" >&2
profile stop_profile
trap - EXIT

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "# then wait for traces newer than .profile_start under $PROFILE_DIR/torch to settle," \
    "move them to $RUN_DIR/traces, summarise each and merge them into $RUN_DIR/merged_trace.json.gz"
  exit 0
fi

# Stages keep writing (background gzip, op tables) after /stop_profile returns.
# Settled means two polls with identical sizes and mtimes and no uncompressed
# trace_rank*.json left. An uncompressed trace that stays unchanged for three
# more polls means the server has torch_profiler_use_gzip off.
previous=""
stable_uncompressed=0
deadline=$(( $(date +%s) + TRACE_SETTLE_TIMEOUT_S ))
while true; do
  current=$(find "$PROFILE_DIR/torch" -type f -newer "$RUN_DIR/.profile_start" -printf '%s %T@ %p\n' 2>/dev/null | sort)
  if [[ -n "$current" && "$current" == "$previous" ]]; then
    if ! grep -qE '/trace_rank[0-9]+\.json$' <<< "$current"; then
      break
    fi
    if (( ++stable_uncompressed >= 3 )); then
      echo "WARNING: uncompressed trace_rank*.json left unchanged; torch_profiler_use_gzip looks off," \
        "so traces are large and settling relies on file stability alone" >&2
      break
    fi
  else
    stable_uncompressed=0
  fi
  if (( $(date +%s) >= deadline )); then
    echo "traces under $PROFILE_DIR/torch did not settle within ${TRACE_SETTLE_TIMEOUT_S} s" >&2
    exit 1
  fi
  previous=$current
  sleep "$TRACE_SETTLE_POLL_S"
done

mapfile -t traces < <(find "$PROFILE_DIR/torch" -type f -newer "$RUN_DIR/.profile_start" 2>/dev/null)
if (( ${#traces[@]} == 0 )); then
  echo "no new torch traces under $PROFILE_DIR/torch (PROFILE=cuda servers write nsys on shutdown)" >&2
  exit 0
fi
for trace in "${traces[@]}"; do
  dest="$RUN_DIR/traces/${trace#"$PROFILE_DIR/torch/"}"
  mkdir -p "$(dirname "$dest")"
  mv "$trace" "$dest"
  if [[ "$dest" == *trace*.json* ]]; then
    python "$ANALYZER" "$dest" > "$dest.summary.txt" || echo "trace_analyzer failed on $dest" >&2
  fi
done
python "$MERGER" "$RUN_DIR" >/dev/null || echo "merge_stage_traces failed on $RUN_DIR" >&2
echo "run directory: $RUN_DIR" >&2
