# duplex_session: MiniCPM-o 4.5 multi-turn duplex benchmarks

Research harness for latency and profiling experiments on MiniCPM-o 4.5 in
full-duplex mode. Design record: [`docs/design/fullduplex.md`](../../docs/design/fullduplex.md);
model recipe: [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../recipes/OpenBMB/MiniCPM-o-4_5.md).

## Serving

[`serve.sh`](serve.sh) launches `vllm-omni serve` in duplex mode. Run it from any
directory.

| `LAYOUT` | Deploy config | Placement | Sessions |
| --- | --- | --- | --- |
| `2gpu` (default) | [`deploy/minicpmo_4_5_2gpu_32gb.yaml`](deploy/minicpmo_4_5_2gpu_32gb.yaml) | Thinker GPU 0; Talker + Code2Wav GPU 1 | 4 |
| `1gpu` | [`deploy/minicpmo_4_5_1gpu_32gb.yaml`](deploy/minicpmo_4_5_1gpu_32gb.yaml) | all stages GPU 0 | 2 |

The `2gpu` overlay inherits the shipped two-GPU profile and adapts it to 32 GB
cards and the edge target of at most 4 concurrent sessions:

- **Session-sized audio encoder graphs.** Stage 0 captures batch sizes 1, 2 and 4
  instead of 1 to 32 (0.76 GiB instead of 3.59 GiB).
- **Video with bounded profiling.** `image: 1` builds the SigLIP tower that
  duplex `video_frames` use. `video: {count: 1, num_frames: 32}` bounds start-up
  profiling; with the unbounded 64-frame dummy video, profiling runs out of memory.
  The reserved vision path leaves 3.08 GiB of KV, so **the Thinker is capped at
  `max_model_len: 16384`**.
- **Context overflow ends the session's request.** With the default
  `sliding_window_mode: off`, a Stage-0 extension past 16384 tokens finishes the
  request (`_streaming_update_overflows` in
  `vllm_omni/core/sched/omni_ar_scheduler.py`). Long multi-turn runs need a
  session window or fewer turns.

The `1gpu` overlay fits all three stages into one 32 GB card. Unlike `2gpu`:

- **Audio only** (`image: 0`, so no vision tower), and **the Thinker is capped
  at `max_model_len: 8192`** with a fixed 1.25 GiB KV pool (9070 tokens, shared
  by all sessions).
- **At most 2 concurrent sessions.** Session-sized buffers follow: Stage-0 audio
  encoder graphs for batch sizes 1 and 2, a 2-row Code2Wav CFM arena
  (`micro_batch_size: 2`) and a 0.5 GiB Talker KV pool.
- Prefill batches are 4096 tokens on Stages 0 and 1.

Compare the two layouts only on audio-only workloads that stay below 8192
tokens per session, at no more than 2 concurrent sessions.

Stage 2 requires `step-audio2==1.0.0` (the `dev` extra in `pyproject.toml`;
`uv pip install step-audio2==1.0.0`). Do not install `stepaudio2-minicpmo` as
the recipe says: its `stepaudio2` package conflicts and lacks `cosyvoice2`, so
start-up fails with `ImportError: There is no such class as cosyvoice2...`.

**Status (2026-10-20):** `2gpu` boots and serves audio and audio + video
duplex turns (warm first audio 156 ms audio-only, 177–198 ms with five 1 fps
frames), and torch profiling of stages 0 and 1 works. `1gpu` served one
warm-up and six rounds of 2 concurrent audio sessions (one turn each). Only
230 MiB of the card stayed free, so watch for Code2Wav or Thinker out-of-memory
errors on longer multi-turn runs.

The first session after start-up is slow: its first audio took 2–28 s across
boots, against ~160 ms when warm. Always run warm-up sessions before measuring
or calling `/start_profile`.

```bash
LAYOUT=2gpu benchmarks/duplex_session/serve.sh
curl -s localhost:8099/health
```

Set `DRY_RUN=1` to print the resolved command without launching it. Any extra
arguments are appended to `vllm-omni serve`.

## Profiling

`PROFILE=torch|cuda` adds `profiler_config` to the stages in `PROFILE_STAGES`
(0 Thinker, 1 Talker, 2 Code2Wav). Model start-up and warm-up are not traced:
capture happens only between `/start_profile` and `/stop_profile`.

```bash
LAYOUT=2gpu PROFILE=torch PROFILE_STAGES=0,1 benchmarks/duplex_session/serve.sh

# after at least one warm-up session:
curl -sS -X POST localhost:8099/start_profile -H 'Content-Type: application/json' -d '{"stages":[0,1]}'
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url 'ws://localhost:8099/v1/realtime?duplex=1' --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav --output-dir /tmp/duplex_demo
curl -sS -X POST localhost:8099/stop_profile -H 'Content-Type: application/json' -d '{"stages":[0,1]}'
```

- Torch traces go to `$PROFILE_DIR/torch` (default
  `/tmp/duplex_session_profile/torch`), one per stage (`stage{N}_rank{R}`). Open
  them in [Perfetto](https://ui.perfetto.dev/).
- `PROFILE=cuda` wraps the server in `nsys profile --capture-range=cudaProfilerApi`
  and writes `$PROFILE_DIR/nsys_<layout>_<timestamp>.nsys-rep`. The report is
  only valid if both profile calls succeed.
- The `stages` list in the curl body must be a subset of `PROFILE_STAGES`. A
  stage without `profiler_config` makes `/start_profile` return HTTP 500.
- Profiled latency is diagnostic only. Take performance numbers from a
  `PROFILE=off` server.

## Profiling concurrent sessions

[`profile_run.sh`](profile_run.sh) drives a running `serve.sh` server with
`vllm bench omni-duplex-eval generate`. Each sample is one full-duplex session
with one question (audio + 1 fps video by default). The script runs `WARMUP`
serial warm-up sessions, then calls `/start_profile`, runs `LIMIT` sessions at
`CONCURRENCY`, and calls `/stop_profile`. Between warm-up and profiling it resets the prefix,
multimodal and encoder caches, so samples repeated from warm-up are not cheaper;
these are vLLM dev endpoints, which `serve.sh` enables with
`VLLM_SERVER_DEV_MODE=1`. It moves the new torch traces into the
run directory and writes a `trace_analyzer.py` summary next to each one: GPU
busy and idle time for that stage process, the longest idle gaps with the CPU
work overlapping them, and the kernels with the most total time. Finally it
merges the stage traces into `merged_trace.json.gz` in the run directory, one
Perfetto timeline with a CPU and a GPU row per stage, and warns when a stage
in `run.json` has no trace. To merge earlier runs:
`python benchmarks/duplex_session/merge_stage_traces.py /tmp/duplex_session_runs/*/`.

The full Omni-DuplexEval dataset is about 18 GB, so mirror one split locally.
RTD_Omni has 50 samples, 669 MB, with videos of 25 s median and 53 s max:

```bash
hf download Hothan/Omni-DuplexEval data/RTD_Omni-00000-of-00001.parquet \
    --repo-type dataset --local-dir "$HF_HOME/omni_duplex_eval"
```

`load_samples` needs `datasets` (`uv pip install --python .venv/bin/python datasets==3.6.0`).

```bash
LAYOUT=2gpu PROFILE=torch PROFILE_STAGES=0,1,2 benchmarks/duplex_session/serve.sh

DATASET="$HF_HOME/omni_duplex_eval" SPLIT=RTD_Omni STAGES=0,1,2 CONCURRENCY=2 \
    REF_AUDIO=/path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
    benchmarks/duplex_session/profile_run.sh
```

- Output goes to `$OUT_ROOT/<timestamp>_c<CONCURRENCY>_n<LIMIT>/`: `run.json`
  (git sha and settings), `warmup/` and `measured/` responses with
  `duplex_metrics.json`, `traces/` and `merged_trace.json.gz`.
- Warm-up and measured runs both start from the first sample, so they overlap;
  the cache resets in between keep the overlap from making measured sessions
  cheaper.
- `PROFILE_DIR` must match the server's. With `PROFILE=cuda` the nsys report is
  written only when the server shuts down, so the script finds no torch traces.
- **2gpu only for now.** omni-duplex-eval always sends video (`--fps 0` divides
  by zero), and the `1gpu` overlay is audio-only.
- Traces of several 25 s sessions are large (stage 0: 29 MB gzipped for four
  sessions), so start with a small `LIMIT`.
- `/stop_profile` is slow. Stages stop one after another, and each builds a
  `key_averages` op table. With stacks and the CUDA-time dump on, one session
  on stages 0 and 1 took 11 min; with `serve.sh`'s defaults (both off), four
  sessions on stages 0 to 2 took 80 s (2026-10-22). `TORCH_WITH_STACK=1`
  brings the slow path back.
- Stages keep writing after `/stop_profile` returns, so the script waits until
  the new files stop changing (`TRACE_SETTLE_POLL_S`, default 10 s;
  `TRACE_SETTLE_TIMEOUT_S`, default 300 s; observed about 25 s).
