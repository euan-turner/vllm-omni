# Long-context duplex profiling (prototype)

Prototype on branch `long-context-prototype`. Measures how MiniCPM-o 4.5 tick
latency grows with context, split by listening vs speaking, and whether answer
quality drops with depth in a long stream. Model: one GH200 on Isambard-AI.

## Pieces

| File | Role |
| --- | --- |
| `vllm_omni/core/sched/duplex_tick_trace.py` | Stage-0 scheduler writes one JSON line per tick when `VLLM_OMNI_DUPLEX_TICK_TRACE_DIR` is set |
| [`tick_metrics.py`](tick_metrics.py) | Tick, tick-deadline (1 s), queue and continuation definitions |
| [`stitch.py`](stitch.py) | Cycles RTD_Omni samples into one stream, segments aligned to 1 s units |
| [`run_long_session.py`](run_long_session.py) | `generate`: stream stitched sessions in real time; `evaluate`: judge every segment |
| [`analyze.py`](analyze.py) | Joins ticks with sessions; `analysis/` CSVs, `summary.json`, plots |
| [`../deploy/minicpmo_4_5_gh200_long_context.yaml`](../deploy/minicpmo_4_5_gh200_long_context.yaml) | `serve.sh LAYOUT=gh200` |
| [`../isambard/`](../isambard/) | Containerfile, arm64 availability check, sbatch job |

`serve.sh LAYOUT=gh200` sets Stage-0 `max_model_len` to `MAX_MODEL_LEN`
(default 131072). Above the HF limit of 40960 it exports
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`: context past 40960 is an explicit
overflow for systems measurements; quality there is not meaningful. A session
that outgrows `max_model_len` still ends with `context_length_exceeded`, and
its tick record has `status: overflow`.

At 1 fps an audio+video unit adds about 78 tokens and an audio-only unit about
12 (plus generated tokens), so 131072 tokens is about 28 min AV or 2–3 h audio.

## Isambard-AI

```bash
# login node: does the base image / do the wheels exist for arm64?
benchmarks/duplex_session/isambard/check_arm64.sh
podman-hpc build -f benchmarks/duplex_session/isambard/Containerfile -t vllm-omni-lc .
podman-hpc migrate vllm-omni-lc   # makes the image visible on compute nodes

# interactive node with internet: model + data into $HF_HOME
hf download openbmb/MiniCPM-o-4_5
hf download Hothan/Omni-DuplexEval data/RTD_Omni-00000-of-00001.parquet \
    --repo-type dataset --local-dir "$HF_HOME/omni_duplex_eval"

REF_AUDIO=/path/HT_ref_audio.wav MODALITY=av TARGET_MINUTES=30 \
    sbatch -A <project> benchmarks/duplex_session/isambard/long_context.sbatch
REF_AUDIO=/path/HT_ref_audio.wav MODALITY=audio TARGET_MINUTES=120 \
    sbatch -A <project> benchmarks/duplex_session/isambard/long_context.sbatch
```

`RUNTIME=apptainer IMAGE=/path/vllm-omni-lc.sif` or `RUNTIME=venv` replace
podman-hpc. Results land in `runs/long_context/<run id>/` (`server.log`,
`ticks/`, `warmup/`, `measured/`, `measured/analysis/`).

## Local probe (quality vs depth)

Each segment of a stitched session is one RTD_Omni question at depth
`offset_s`. Score the segments, then compare with the same samples at depth 0
(a normal `vllm bench omni-duplex-eval generate` + `evaluate` run):

```bash
python benchmarks/duplex_session/long_context/run_long_session.py evaluate \
    --dataset "$HF_HOME/omni_duplex_eval" --out-dir runs/long_context/<id>/measured \
    --judge-base-url http://127.0.0.1:8000 --judge-model <judge>
python benchmarks/duplex_session/long_context/analyze.py runs/long_context/<id>/measured \
    --model openbmb/MiniCPM-o-4_5 --baseline-score-root <depth0 score root>/RTD_Omni
```

Use `--start-index` (or `--sessions` with `--session-stride`) so that each
sample is measured at several depths.
