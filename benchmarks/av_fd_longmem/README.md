# av_fd_longmem: full-duplex long-horizon A/V memory benchmark

Measures whether a full-duplex omni model (MiniCPM-o 4.5 in vLLM-Omni) can
recall something it saw or heard earlier **when the question arrives on its
live input stream**. The question is spoken into the stream, not sent as text,
after a controlled lag of unrelated filler.

Each probe has three parts:

1. **Evidence:** a few seconds of source video and audio that show the answer.
2. **Filler:** `G` seconds of unrelated video (the lag, counted in chunks).
3. **Query:** a past-tense spoken question, followed by a silent 30 s answer window.

One **chunk** is one second of stream: 1 s of 16 kHz audio plus one frame.

The harness is a client around a deployed `vllm-omni serve` duplex server. It
records each session's output text and audio. The server's tick trace says
whether the evidence was still in the model's context window when the question
arrived.

## Status

Only the **OmniInteract pilot** is implemented: six hand-picked Chinese
(`1q1a`) items in one chained session, plus a validity control per item. Not
yet implemented:

- StreamArena and the bulk OmniInteract dataset
- LLM rewriting and VLM grounding of items
- the LLM judge
- blind controls, proactive probes and modality ablations

The pilot scores answers by matching key tokens, a stand-in for the LLM judge
(see [Scoring](#scoring)).

## Requirements

- 2× 32 GB GPU (developed on 2× RTX 5090), `ffmpeg` and `ffprobe` on PATH.
- The repo's uv environment (`.venv`), plus these pinned dev dependencies:

  ```bash
  uv pip install --python .venv/bin/python "funasr==1.3.29" "jiwer==4.0.0" "zhconv==1.4.3"
  ```

- These models in the Hugging Face cache (`HF_HOME`):
  - `openbmb/MiniCPM-o-4_5-GPTQ` at revision `02b54c54…`, the model under test
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, for the spoken queries
  - `funasr/fsmn-vad` and `funasr/paraformer-zh`, for question onsets and ASR; they download on first use
- OmniInteract extracted locally. Point `--omniinteract-root` at its `data/`
  directory, which holds `1q1a/` and `1qna/`.

Outputs go to `~/Documents/datasets/av_fd_longmem` by default; change this with
`--dataset-root`. Run every command below from the repository root.

```bash
export HF_HOME=/path/to/hf_home
export PATH=$PWD/.venv/bin:$PATH
OI=/path/to/OmniInteract/data
REF=$HF_HOME/hub/models--openbmb--MiniCPM-o-4_5-GPTQ/snapshots/02b54c54c36f8b48e97501bb0fde5d178c62df38/assets/HT_ref_audio.wav
DEPLOY=benchmarks/duplex_session/deploy/minicpmo_4_5_gptq_32gb.yaml
```

## Reproducing the pilot

### 1. Build items, filler and plans

This step needs the TTS server running on one GPU:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091 \
  --stage-overrides '{"1": {"max_num_seqs": 1}}' &

python -m benchmarks.av_fd_longmem build-pilot --omniinteract-root "$OI"
```

The build does the following:

1. For each item in [`pilot/omniinteract_pilot.json`](pilot/omniinteract_pilot.json), it finds where the spoken question starts. fsmn-vad splits the audio into speech segments, Paraformer transcribes them, and the segment that best matches the annotated question (CER ≤ 0.5) gives the start time.
2. It cuts the evidence as `[onset − 4 s, ⌊onset − 0.3 s⌋)`, so the original question is never streamed.
3. It speaks the past-tense query with voice `vivian`, then checks it by transcribing the result (CER ≤ 0.15).
4. It lays the query over a muted `1qna` cooking clip, which serves as the backdrop, and adds a silent 30 s answer tail.
5. It cuts filler from the other `1q1a` clips. A filler run must be at least 30 s long and must avoid `[question_time − 5 s, question_time + 15 s]` around every annotated question.
6. It writes the plans:
   - `plans/pilot/pilot-chained.json`: one 670-chunk session holding all six probes at lags 10, 30, 60, 180, 300 and 600 chunks
   - `plans/pilot/pilot-g0-<item>.json`: one G=0 validity control per item

Stop the TTS server when the build finishes.

### 2. Serve MiniCPM-o with the tick trace on

Start one server per GPU. Each server runs one session at a time.

```bash
mkdir -p /tmp/trace_gpu0 /tmp/trace_gpu1
CUDA_VISIBLE_DEVICES=0 PORT=8099 VLLM_OMNI_DUPLEX_TICK_TRACE_DIR=/tmp/trace_gpu0 benchmarks/duplex_session/serve.sh &
CUDA_VISIBLE_DEVICES=1 PORT=8100 VLLM_OMNI_DUPLEX_TICK_TRACE_DIR=/tmp/trace_gpu1 benchmarks/duplex_session/serve.sh &
until curl -sf localhost:8099/health && curl -sf localhost:8100/health; do sleep 5; done
```

### 3. Run the sessions

Sessions stream in real time. The chained session takes about 11 minutes and
each control about 70 s.

```bash
D=~/Documents/datasets/av_fd_longmem/plans/pilot
# GPU 0, in the background: the baseline, then an identical repeat to check determinism
for run in pilot-baseline-1 pilot-baseline-2; do
  python -m benchmarks.av_fd_longmem run $D/pilot-chained.json --run-id $run \
    --base-url http://127.0.0.1:8099 --ref-audio "$REF" --trace-dir /tmp/trace_gpu0 --deploy-config $DEPLOY
done &
# GPU 1, in parallel: the G=0 validity controls
python -m benchmarks.av_fd_longmem run $D/pilot-g0-*.json --run-id pilot-controls-g0 \
  --base-url http://127.0.0.1:8100 --ref-audio "$REF" --trace-dir /tmp/trace_gpu1 --deploy-config $DEPLOY
wait
```

Each session is configured from its plan's `condition`:

| Plan | `sliding_window_mode` | Window size | Temperature |
|---|---|---|---|
| Chained baseline | `basic` | 8000/6000 tokens, about 100 s | `0.0` (greedy) |
| G=0 controls | `context` | `context_max_units: 60` | `0.0` (greedy) |

Always set the window mode explicitly. The server default, `off`, lets the context grow until the model degrades.

### 4. Score

```bash
python -m benchmarks.av_fd_longmem score pilot-baseline-1 pilot-baseline-2 pilot-controls-g0 --asr-device cuda:0
```

Use `--asr-device cpu` while the GPUs are still serving.

Each run directory gets the following files:

| File | Contents |
|---|---|
| `labels.jsonl` | Whether each probe's evidence was in or out of the window at query onset |
| `verdicts.jsonl` | One verdict per probe and answer channel |
| `summary.md` | One row per probe |

To check determinism, compare `sessions/pilot-chained/wav_transcript.json`
between the two baseline runs.

## Output layout

```text
<dataset-root>/
  omniinteract/items.jsonl  filler.jsonl  clips/{evidence,query,filler}/*.mp4  cache/tts/
  plans/pilot/<plan_id>.json
  runs/<run_id>/
    run.json                       server, model, deploy YAML, ref audio, git HEAD and dirty flag
    sessions/<plan_id>/            output.wav, wav_transcript.json, events.json, result.json, session.json
    trace/                         server tick-trace files (duplex-ticks-*.jsonl), copied after each session
    labels.jsonl  verdicts.jsonl  summary.md
```

Plans contain clip references only. Every time inside a plan is a stream chunk
index. A probe's gap is `query_onset_chunk − last evidence chunk − 1`.

## How it works

- **Streaming.** [`runner.py`](runner.py) renders a plan into PCM and one JPEG per chunk ([`render.py`](render.py)). It then streams them with `vllm_omni.benchmarks.omniinteract.run_omniinteract_case`, the same code path as the OmniInteract serving benchmark. Audio goes out in 200 ms appends, and frame `k` travels with the append that crosses `k + 0.5 s`. The client acknowledges playback as it would in a live session. A session whose `epoch` ever changes is marked and skipped when scoring.
- **Residency.** [`residency.py`](residency.py) works out whether each probe's evidence was still in the window at query onset:
  - Stream chunk `c` is client appends `5c+1 … 5c+5`.
  - Trace records link appends to Stage-0 units: `start` and `acc`, joined on the attempt number, give each unit's `seq`.
  - The opt-in `ev` trace record logs every window drop by unit `seq`. Its format is documented in `vllm_omni/engine/duplex/tick_trace.py`.
  - A probe is **in** the window if none of its evidence units has lost tokens before query onset. It is **out** if every evidence unit has lost everything except a sink residue of at most one KV block. Anything in between is **partial**.
  - A trace containing `trace_dropped` is rejected, because a lost `ev` record would make evicted evidence look in-window.

### Scoring

Two answer channels are scored for every probe:

- the model's output text in `wav_transcript.json`
- Paraformer ASR of `output.wav` over the answer window

Text counts toward a probe if it starts inside the window that runs from the query onset to the end of the answer tail. Output before the query onset that already carries the answer is flagged as `rehearsed`.

[`score.py`](score.py) matches **key tokens**. It is valid only for short factual answers:

- A numeric answer matches on its digit groups. `4号车13号门` needs `4` then `13` as whole groups, so `14车13门` and `2.99` are both rejected. Chinese numerals (`一百六十八`, `两`, `点`) are converted first.
- Any other answer matches on its text after conversion to Simplified Chinese. A parenthetical gives an accepted alternative, so `麗枫酒店（LAVANDE）` accepts either name.

Every verdict carries the reply text so it can be reviewed by hand. The plan's LLM judge replaces this before any bulk runs.

## Pilot results

Results from 2026-11-25 on GPTQ, one session per server, `basic` window, greedy decoding. The pilot is a plumbing check, not a measurement: one item per lag and one run each.

| Item | Lag (chunks) | Evidence at query | G=0 control (context-60) | Chained baseline reply | Verdict |
|---|---|---|---|---|---|
| 0127 hotel name | 10 | in | correct (`Lavande丽帆酒店`) | `薰衣草酒店` ("Lavender hotel") | incorrect |
| 0102 car / door | 30 | in | correct | `4车13门` | correct |
| 0108 sunglasses price | 60 | in | correct | `168元` | correct |
| 0115 egg unit price | 180 | out | correct | `2.99美元一颗` (invented) | incorrect |
| 0130 hotline | 300 | out | correct | `1-800-5555-0123` (invented) | incorrect |
| 0124 toy price | 600 | out | **not perceived** (answers about Lego) | about table-tennis bats | excluded |

- **Retention:** two of the three in-window probes are answered, and none of the out-of-window ones. The miss at lag 10 is a translation of the brand name.
- **Eviction timing:** the window evicts evidence between lags 60 and 180, consistent with about 100 s of A/V (about 79 tokens per chunk: 64 image + 10 audio).
- **No abstention:** once the evidence is evicted, the model invents a confident answer rather than saying it doesn't know.
- **Determinism:** the repeat run is token-identical. It gives the same 7 responses and the same Stage-0 token count in all 670 segments.
- **Answer channels:** text and ASR agree on all six chained probes. They disagree on three of the six G=0 controls:
  - `4车` is heard as `十四车`.
  - The `Lavande` brand name is lost in the audio.
  - The phone number is garbled in speech.
- **Pilot health:** every probe got a residency label, there were no epoch changes, and no trace records were dropped.
