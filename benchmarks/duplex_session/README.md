# duplex_session: MiniCPM-o 4.5 multi-turn duplex benchmarks

Research harness for latency and profiling experiments on MiniCPM-o 4.5 in
full-duplex mode. Design record: [`docs/design/fullduplex.md`](../../docs/design/fullduplex.md);
model recipe: [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../recipes/OpenBMB/MiniCPM-o-4_5.md).

`metrics.py` in this directory defines what "faster" means and is human-owned.

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
