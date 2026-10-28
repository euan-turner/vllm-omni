#!/usr/bin/env bash
# Compact plain-text report of one long-context run, for pasting into review:
# job state, driver outcome, summary table, server errors, Stage-0 allocator
# memory by tick, per-process GPU memory, stage timings and decoded late ticks.
#
# Prototype (branch long-context-prototype). Run from the directory the job was
# submitted from (the repo root), on a login node:
#   benchmarks/duplex_session/isambard/report.sh runs/long_context/<run id>
# External dependencies: python3 (stdlib), awk; optional sacct; optional
# podman-hpc and the vllm-omni-lc image (IMAGE) to decode late-tick text.
set -uo pipefail

RUN=${1:?usage: report.sh <run dir>}
RUN=${RUN%/}
IMAGE=${IMAGE:-vllm-omni-lc}
JOB=${RUN##*_}
TICKS=("$RUN"/ticks/ticks_stage0_*.jsonl)

echo "== run $RUN"
command -v sacct >/dev/null && sacct -j "$JOB" -X -o JobID,Elapsed,State,ExitCode

echo "== driver"
grep -hE '^session|TickStall|oom_kill|server exited|not healthy|^Traceback|Error:' "duplex-long-context-$JOB.out" 2>/dev/null \
  | cut -c1-300 || echo "(no duplex-long-context-$JOB.out here)"

echo "== summary"
python3 - "$RUN/measured/analysis/summary.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except OSError as exc:
    sys.exit(f"(no summary: {exc})")
print(json.dumps({k: v for k, v in s.items() if k != "by_context"}, default=str))
print("ctx_lo  kind    n    svc_p50  svc_p99  queue_p99  sampled  cont")
for r in s.get("by_context", []):
    print(f"{r['context_lo']:>6}  {r['kind']:6} {r['count']:>4}  {r['service_p50_s']:7.3f}  {r['service_p99_s']:7.3f}"
          f"  {r.get('queue_p99_s', float('nan')):9.3f}  {r['sampled_tokens_mean']:7.1f}  {r.get('suspected_continuations', '-')}")
EOF

echo "== server errors (first 20)"
grep -nE 'CUDACachingAllocator|out of memory|OutOfMemory|^Traceback|is dead|closed duplex session|FINISHED_ERROR|ERROR' \
  "$RUN/server.log" 2>/dev/null | grep -v '_C' | cut -c1-250 | head -20

echo "== stage-0 memory by tick (measured session)"
python3 - "${TICKS[@]}" <<'EOF'
import collections, json, sys
recs = [json.loads(l) for f in sys.argv[1:] if not f.endswith("*.jsonl") for l in open(f)]
if not recs:
    sys.exit("(no tick records)")
rid = collections.Counter(r["request_id"] for r in recs).most_common(1)[0][0]
t = [r for r in recs if r["request_id"] == rid]
gib = lambda b: "-" if b is None else f"{b / 2**30:.2f}"
step = max(1, len(t) // 20)
print("tick    ctx  alloc_GiB  reserved_GiB  sampled  status")
for r in t[::step] + t[-1:]:
    ctx = (r.get("context_tokens_before") or 0) + (r.get("prefill_tokens") or 0)
    print(f"{r['tick_index']:>5} {ctx:>6}  {gib(r.get('cuda_allocated_bytes')):>9}  {gib(r.get('cuda_reserved_bytes')):>12}"
          f"  {len(r.get('sampled_token_ids') or ()):>7}  {r.get('status')}")
EOF

echo "== GPU memory per process (MiB)"
awk -F', ' 'NR>1 && $4 ~ /MiB/ {m=$4+0; p=$2; if(!(p in f)){f[p]=m; t0[p]=$1}; l[p]=m; t1[p]=$1; if(m>x[p])x[p]=m}
  END{for(p in f) printf "pid %s  first %d  last %d  max %d  (%s -> %s)\n", p, f[p], l[p], x[p], t0[p], t1[p]}' \
  "$RUN/gpu_mem.csv" 2>/dev/null

echo "== host RSS per process (MiB): first, last, max"
awk -F, '{k=$3" "$4; if(!(k in f)){f[k]=$2; t0[k]=$1}; l[k]=$2; t1[k]=$1; if($2>x[k])x[k]=$2}
  END{for(k in f) if(x[k]>=1024) printf "%7d %7d %7d  %s  (%s -> %s)\n", f[k], l[k], x[k], k, t0[k], t1[k]}' \
  "$RUN/host_mem.csv" 2>/dev/null | sort -k3 -nr | head -10

echo "== stage timings per response (11 evenly spaced)"
grep -o 'OmniTiming.*' "$RUN/server.log" 2>/dev/null \
  | sed -E 's/.*total=([0-9.]+s).*stages=\[([^]]*)\].*/total=\1 stages=\2/' \
  | awk '{l[NR]=$0} END{print "responses:", NR; for(i=0;i<=10;i++){k=int(1+i*(NR-1)/10); if(k>=1 && !s[k]++) print k": "l[k]}}'

echo "== late ticks, decoded"
if command -v podman-hpc >/dev/null; then
  podman-hpc run --rm -i -v "$PWD:$PWD" -v "$HF_HOME:$HF_HOME" -e HF_HOME -e HF_HUB_OFFLINE=1 -w "$PWD" "$IMAGE" \
    python - "${TICKS[@]}" <<'EOF' 2>&1 | grep -vE 'Warning|warn|^\s*$' | tail -16
import collections, json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-4_5", trust_remote_code=True)
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
rid = collections.Counter(r["request_id"] for r in recs).most_common(1)[0][0]
t = [r for r in recs if r["request_id"] == rid and len(r.get("sampled_token_ids") or ()) > 1]
for r in t[-15:]:
    ctx = r["context_tokens_before"] + r["prefill_tokens"]
    print(r["tick_index"], ctx, len(r["sampled_token_ids"]), repr(tok.decode(r["sampled_token_ids"]))[:110])
EOF
else
  echo "(podman-hpc not found; skipped)"
fi
