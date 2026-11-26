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

Implemented:

- the **OmniInteract pilot**: six hand-picked Chinese (`1q1a`) items in one
  chained session, plus a validity control per item;
- the **M0 data audit** of OmniInteract and StreamArena (see [M0](#m0-data-audit));
- the **M1 window characterisation** (see [M1](#m1-window-characterisation));
- the **M3 item-set pipeline and smoke** (see [M3](#m3-smoke-item-sets-with-llm-rewrite-and-vlm-grounding)):
  LLM rewrite and tags, VLM grounding, blind pre-check, leakage checks, and the
  control, ablation and retention plans.

Not yet implemented:

- bulk item construction (the M3 pipeline is run on 5 items per source)
- the calibrated LLM judge (M5)
- proactive probes

The pilot scores answers by matching key tokens, a stand-in for the LLM judge
(see [Scoring](#scoring)).

## Requirements

- 2× 32 GB GPU (developed on 2× RTX 5090), `ffmpeg` and `ffprobe` on PATH.
- The repo's uv environment (`.venv`), plus these pinned dev dependencies:

  ```bash
  uv pip install --python .venv/bin/python "funasr==1.3.29" "jiwer==4.0.0" "zhconv==1.4.3" "zhon==2.1.1"
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
3. It speaks the lag-neutral past-tense query (之前看到的…) with voice `vivian`, then checks it by transcribing the result (CER ≤ 0.15).
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
| --- | --- | --- | --- |
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
| --- | --- |
| `labels.jsonl` | Whether each probe's evidence was in or out of the window at query onset |
| `verdicts.jsonl` | One verdict per probe and answer channel |
| `summary.md` | One row per probe |

To check determinism, compare `sessions/pilot-chained/wav_transcript.json`
between the two baseline runs.

## M0 data audit

The audit counts what survives each rule. It also measures the spoken-question
onsets and clip sizes, and builds a hand-check sample. Outputs go under
`<dataset-root>/{omniinteract,streamarena}/audit/` and
`<dataset-root>/audit_handcheck/`.

```bash
# OmniInteract onsets: fsmn-vad + Paraformer on CPU (~10 min); segments are cached per clip
python -m benchmarks.av_fd_longmem audit-omniinteract --omniinteract-root "$OI"

# StreamArena: fetch the annotations at the pinned revision, then count exclusions.
# --videos also materialises the evidence and filler clips of fetched, extracted tars.
python -c "from huggingface_hub import hf_hub_download as d; import os
for f in ['question.jsonl','question.en.jsonl','video_meta.jsonl','video_meta.en.jsonl','videos/videos_manifest.jsonl']:
    d('hkuzxc/StreamArena', f, repo_type='dataset', revision='2960c360deba57b717a17ab66a3ae80f5358a6a3',
      local_dir=os.path.expanduser('~/Documents/datasets/av_fd_longmem/streamarena/annotations'))"
python -m benchmarks.av_fd_longmem audit-streamarena --raw-root <extracted-tars> --videos jvEuEu65YE8 p-PVprguijI j_ZPV10bu54

# Query codec: TTS 25 zh + 25 en StreamArena questions, ASR them as WAV, AAC and FLAC (TTS server on GPU 1)
python -m benchmarks.av_fd_longmem audit-codec --asr-device cuda:1

# Hand-check page: 30 OmniInteract + the StreamArena evidence clips, with contact sheets and evidence ASR
python -m benchmarks.av_fd_longmem audit-handcheck --omniinteract-root "$OI"
```

Each tar must be checked against its sha256 in `videos_manifest.jsonl` before
it is extracted, and deleted once its clips are cut.

### Results (2026-11-25)

**OmniInteract `1q1a`** (942 questions: 638 realtime, 304 proactive):

- **Onsets.**
    - VAD runs over `[question_time − 10 s, question_time + 8 s]`.
    - A question matches the segment whose transcript holds it as a substring,
      with CER ≤ 0.5. Its onset is the segment start. Merged speech before the
      question (an announcement, an earlier question, "哦不，我想问的是…") is cut
      with it.
    - 634 of 638 realtime questions match, and all six pilot onsets reproduce.
    - Δ = onset − annotated time: median −0.16 s, p05 −5.3 s, p95 +0.7 s.
- **Realtime status:**

  | Status | Items | Meaning |
  | --- | --- | --- |
  | `ok` | 552 | |
  | `recall_phrased` | 66 | already about the past (刚刚…, 你还记不记得…); 前面 is spatial here and is not counted |
  | `question_in_evidence` | 10 | the question, or a near-identical one, is also spoken inside the evidence window |
  | `onset_censored` | 6 | the speech began before the search window |
  | `no_onset_match` | 4 | |

- **Room before the onset** runs back to the end of the previous question's
  measured speech. The model's answers are not on the soundtrack: the text
  that matched answers turned out to be questions echoing their wording.
- **Yield** (`ok`, not interrupted, room ≥ W): 349 items at W=2, **298 at
  W=4 (137 clips)**, 265 at W=8. W = 4 s is the chosen lookback. The
  evidence ends exactly 0.3 s before the onset, so its last frame is at most
  0.8 s before the question.
- **Hand check** of 19 present-tense items, by eye on 8-frame contact sheets
  (an earlier 8 s lookback):
    - 9 show the answer (4 of them only in the last 1–2 s before the onset)
    - 5 are marginal (small or blurred text or logos)
    - 5 don't show it
    - The VLM grounding step is still needed.

**StreamArena** (877 HR items, 235 videos; `zh+en` videos count as English):

- **Rule exclusions.** An item can hit several rules:

  | Rule | Items |
  | --- | --- |
  | video language `other` | 7 |
  | evidence after `ask_sec` | 23 |
  | "so far" phrasing | 108 |
  | "past N minutes" / "N minutes ago" phrasing | 22 |

- **731 survive**: 609 zh and 122 en.
    - 48 use recency wording (刚才, "just now"), which is false at long lags. Their queries need a lag-neutral rewrite.
    - Evidence is 10 s at the median before subtitle extension. On the 14 fetched items it is 10–17 s after extension (median 15 s).
    - Items with several evidence windows: 60 span more than 60 s from first to last window, so they cannot fit the context-60 validity window.
    - The LLM rules (residual relative phrasing, non-standalone) are not applied yet.
- **Three fetched videos** gave 14 survivors.
    - The evidence-clip ASR holds the answer for the zh audio items (生日贺卡, 手风琴, 五种口味, 二十袋, 六十包).
    - Screen-recording text, such as a GitHub star count, is not legible at 640 px.

**Clips and codec:**

- **Clip size:** 214 MB per hour for OmniInteract phone footage and 107–136 MB
  per hour for StreamArena. A full StreamArena filler pool would be about 40 GB.
- **StreamArena filler** (clear of every annotated time by 60 s): 35% of the
  three videos' running time.
- **Codec:** ASR error on the 50 spoken queries is the same for WAV, AAC and
  FLAC. One English query does better with AAC. Clips keep AAC.

## M1 window characterisation

Each window mode streams filler only. The trace then shows how many chunks the
Thinker's window holds.

```bash
python -m benchmarks.av_fd_longmem window-plans          # plans/m1/m1-basic.json (900 chunks), m1-off.json (600)
# serve as in "Reproducing the pilot", step 2, then:
python -m benchmarks.av_fd_longmem run $D/../m1/m1-basic.json --run-id m1-basic --base-url http://127.0.0.1:8099 \
  --ref-audio "$REF" --trace-dir /tmp/trace_gpu0 --deploy-config $DEPLOY
python -m benchmarks.av_fd_longmem run $D/../m1/m1-off.json --run-id m1-off --base-url http://127.0.0.1:8100 \
  --ref-audio "$REF" --trace-dir /tmp/trace_gpu1 --deploy-config $DEPLOY
python -m benchmarks.av_fd_longmem window-report m1-basic
python -m benchmarks.av_fd_longmem window-report m1-off
```

`window-report` writes `window_report.json` (per chunk) and `window_summary.md` in the run directory.

### Results (2026-11-25, GPTQ, greedy, OmniInteract filler)

- **Tokens per chunk:** 79, made of 10 audio, 64 image and 5 marker tokens.
- **`basic` (8000/6000 tokens):**
    - The window compacts every 32 chunks.
    - The resident span is a sawtooth from **100 chunks down to 69**; it was 63
    at the first compaction.
    - Evidence ending G chunks before the query is therefore always in the window
    when `G + evidence chunks ≤ 69`, and always out when `G ≥ 100`. Between the
    two, it depends on the phase.
    - The model's own speech adds tokens: the pilot, which had answers, reached a
    minimum span of 60.
- **`off`:** the last unit is accepted at **chunk 398**. The prompt is then
  31,589 tokens, and Stage 0 ends the request with `context_length_exceeded`
  at `max_model_len` 32000. The session fails and keeps no transcript.

### Single-probe check of the lag grid (2026-11-25)

A single-probe session is `filler 30 | evidence | filler G | query + 30 s tail`.
It meets only the first compaction, at chunk 100, which drops the oldest ~37
chunks. Item `0108` (sunglasses, 4 evidence chunks at 30–33, `basic`, greedy):

```bash
python -m benchmarks.av_fd_longmem single-plans --item oi-1q1a-0108-0 --lags 40 60
python -m benchmarks.av_fd_longmem run $D/../single/single-oi-1q1a-0108-0-g40.json --run-id single-g40 ...
python -m benchmarks.av_fd_longmem run $D/../single/single-oi-1q1a-0108-0-g60.json --run-id single-g60 ...
```

| G | Query onset | Label at onset | Compaction | Reply |
| --- | --- | --- | --- | --- |
| 40 | chunk 74 | in | chunk 100, drops chunks 0–37 | 168元 at 78.5 s: correct |
| 60 | chunk 94 | in | chunk 100, drops chunks 0–37 | 168元 at 98.0 s: correct |

- **G = 60 is the grid point (decision).** Residency is judged at the query
  onset.
- **The G = 60 margin is small.** The evidence was evicted 2 chunks after the
  answer began. A longer query, longer evidence or slower answer can push the
  eviction ahead of the reply.

## M3 smoke: item sets with LLM rewrite and VLM grounding

M3 builds a named **item set** (`m3`) of 5 probes per source through the full
annotation pipeline. Each probe gets six plans:

| Plan | Window | Purpose |
| --- | --- | --- |
| `-g0` | `context`, 60 units | validity: perceived at G=0 |
| `-blind` | `context`, 60 units | validity: evidence replaced by unused filler of the same length |
| `-ablate-audio`, `-ablate-video` | `context`, 60 units | empirical modality: evidence muted, or frozen on its first frame |
| `-g60`/`-g30`, `-g300` | `basic` | retention, in-window and out-of-window |

StreamArena uses G=30, not 60. Its evidence is about 15 chunks, so at G=60 the
query lands after the first compaction (chunk 100) and the probe is partly
evicted (decision, 2026-11-25).

The pilot's `omniinteract/{items,filler}.jsonl` are left untouched. An item set
lives in `<source>/sets/<set>/`, its plans in `plans/<set>/`. Clips share
`<source>/clips/`, keyed by item or filler id, and item ids must be unique
across every set: `score` refuses duplicates.

### Models

| Role | Model | Revision |
| --- | --- | --- |
| LLM: OmniInteract rewrite, tags, blind pre-check, answer match | `cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` (community 4-bit, 17 GB) | `8d3f99ed` |
| VLM: grounding, backdrop and filler leakage | `Qwen/Qwen2.5-VL-7B-Instruct` bf16 | `cc594898` |
| ASR: evidence grounding, leakage scan, TTS check, answer channel | Paraformer-zh; `openai/whisper-large-v3` for en | |

Qwen's own FP8 checkpoint of the LLM is 31 GB and does not fit one 32 GB GPU,
so the community AWQ quant is used. Calls run at temperature 0. Each reply is a
JSON object that the client checks against the prompt's schema. Server-side
guided decoding is **not** used: with xgrammar, compact JSON derailed both
models and whitespace-tolerant JSON let them loop. Replies are cached under
`<source>/cache/{llm,vlm}/`, keyed on model, revision, prompt id and version,
and the full request. The server flags are not in the key, so clear the cache
if you change them.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
  --revision 8d3f99eda08b18a8090de5289ff02e6dc51d47e2 --port 8200 --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --max-num-seqs 16 --seed 0 &
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --port 8201 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 40, "video": 0}' --gpu-memory-utilization 0.9 --max-num-seqs 8 --seed 0 &
```

### Building the set

1. **StreamArena videos.** Fetch and cut the videos with `audit-streamarena`
   (see [M0](#m0-data-audit)). M3 uses the three M0 videos plus two English
   ones: `H3KnMyojEQU` (launch event) and `J7eS9BRw2Y8` (animation). The extra
   two give English items filler in their own language from another domain.
2. **Annotate** (LLM and VLM servers):

   ```bash
   python -m benchmarks.av_fd_longmem set-annotate --source streamarena
   python -m benchmarks.av_fd_longmem set-annotate --source omniinteract --omniinteract-root "$OI" --n 20
   ```

   This writes `candidates.jsonl`, one record per candidate with every LLM and
   VLM output.
   - **Query rewrite.**
     - OmniInteract: the LLM rewrites the present-tense question to a
       lag-neutral past (之前看到的…) and strips the answer down.
     - StreamArena: a fixed rule replaces 刚才/刚刚/方才 with 之前, and "just
       now" with "earlier" (`sources.streamarena.lag_neutral`). LLM rewrites
       switched the language and pasted timestamps from the evidence hint
       into the question.
   - **StreamArena tags** (LLM): `relative_time`, `non_standalone` and
     `present_referent` each exclude an item. Items with several evidence
     windows are excluded too, because the planner takes one evidence clip.
   - **Grounding.** An item is grounded when either check passes:
     - the VLM, **not shown the answer**, answers correctly from the evidence
       frames as they are streamed, and the LLM confirms the match;
     - the LLM finds the answer in the evidence's ASR transcript, **and its
       quote really occurs in that transcript**. The LLM invents quotes otherwise.

     The VLM's verdict when it is shown the answer is recorded but not used. It
     said "no" for all 16 SA candidates, and for 5 of the 9 OI candidates that
     it read correctly unprompted. Several of its own observations stated the
     answer ("The GitHub page shows a skill with 1.3k stars.").
   - **Blind pre-check:** the LLM answers the query with no media. A match is
     only a flag. Such items sort last when picking, and the model's own blind
     control decides.
3. **Items** (TTS server, plus the VLM for the backdrop check):

   ```bash
   python -m benchmarks.av_fd_longmem set-items --source streamarena --per-lang en=2 zh=3 --asr-device cuda:0
   python -m benchmarks.av_fd_longmem set-items --source omniinteract --omniinteract-root "$OI" --n 5 --asr-device cuda:0
   ```

   - **Picking.** Grounded candidates are taken in order, with modalities
     spread within each language.
   - **Query clips.** The spoken query goes over a backdrop with a silent 30 s
     tail, and the cut query clip is checked for leakage:
     - OmniInteract backdrops are `1qna` cooking clips.
     - StreamArena backdrops are filler from a different-domain video in the
       item's language, then removed from the whole set's filler pool.
   - **Filler.**
     - OmniInteract: a shuffled 4,000-chunk pool of 1q1a runs that excludes
       every clip that is the source of an item in the set.
     - StreamArena: every filler clip of the fetched videos, restricted per
       item to its language and another `L1`.
4. **Plans** (VLM):

   ```bash
   python -m benchmarks.av_fd_longmem set-plans --source streamarena --asr-device cuda:0
   python -m benchmarks.av_fd_longmem set-plans --source omniinteract --asr-device cuda:0
   ```

   Every filler clip an item's plans play is checked in two ways: by the VLM in
   30-frame windows, and by an ASR keyword scan for the answer. A hit puts the
   clip in the item's `filler_exclude`, and all six of its plans are rebuilt, so
   filler stays nested across lags. Results go to `leakage.jsonl`.
5. **Check by eye and ear.** This renders a plan to an mp4, exactly as it is
   streamed:

   ```bash
   python -m benchmarks.av_fd_longmem render-preview plans/m3/m3-sa-jvEuEu65YE8-5-g30.json --out previews/m3/sa.mp4
   ```

### Running and scoring

Serve and run as in the pilot, one source per GPU, after one warm-up session per
server. Then score:

```bash
python -m benchmarks.av_fd_longmem run $P/m3-sa-*.json --run-id m3-sa --base-url http://127.0.0.1:8099 ...
python -m benchmarks.av_fd_longmem run $P/m3-oi-*.json --run-id m3-oi --base-url http://127.0.0.1:8100 ...
python -m benchmarks.av_fd_longmem score m3-sa m3-oi --asr-device cuda:0 --llm-url http://127.0.0.1:8200
```

`--llm-url` adds an LLM answer-match verdict. Since 2026-11-26 it is the
**primary** verdict and the key-token verdict is secondary; `summary.md` lists
every disagreement for listening. It is **uncalibrated**; the plan's judge is M5. English output audio is
transcribed with Whisper.

### Smoke results (2026-11-25, runs `m3-sa`, `m3-oi`)

The setup matches the pilot (GPTQ, greedy, one session per server, k=1), with
one warm-up session per server first. All 60 sessions succeeded, with no epoch
changes and no dropped trace records. The tables use the LLM verdict on the
transcript; † marks a verdict where the key-token scorer disagrees, because it
cannot match a verbose or paraphrased answer.

**Residency.** Every G=0, G=30 and G=60 probe was in the window at query onset,
with 0.00 of its evidence evicted. Every G=300 probe was out, with 1.00
evicted. That is the M1 prediction (`G + evidence ≤ 69` in, `G ≥ 100` out). The
blind control's span plays filler, so its residency is reported as `n/a`.

**StreamArena** (in-window lag G=30):

| Item | Answer | Declared | G=0 | Blind | Status | G=30 | G=300 | Audio muted | Video frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J7eS9BRw2Y8-8 (en) | BLEACH | visual | ✗ Monster | ✗ | not perceived | ✗ | ✗ | ✗ | ✗ |
| j_ZPV10bu54-16 (en) | Financial Technology | audio | ✗ computer science | ✗ | not perceived | ✗ | ✗ | ✗ | ✗ |
| jvEuEu65YE8-15 | 手风琴 | audio | ✓ | ✗ 二胡 | valid | silent | ✗ 二胡 | ✓ | ✓ |
| jvEuEu65YE8-5 | 生日贺卡 | visual | ✓ | ✗ | valid | ✓ | ✗ 鲜花 | ✗ 一幅画 | ✓ |
| p-PVprguijI-13 | 60包 | av | ✓ | ✗ | valid | ✓ | ✗ 20包 | ✗ 20包 | ✓ |

**OmniInteract** (in-window lag G=60):

| Item | Answer | G=0 | Blind | Status | G=60 | G=300 | Audio muted | Video frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0036-6 | 两辆电动车 | ✓ | **✓** | prior-answerable | ✓ | ✗ 没有出现过 | ✗ 1辆 | ✗ 一辆 |
| 0038-4 | 粉色 (verbose reference) | ✓† | **✓†** | prior-answerable | ✓† | ✓† | ✓† | ✓† |
| 0041-3 | 轮椅租借处 | ✓ | ✗ | valid | ✓ | ✗ | ✓ | ✗ 食物简史 |
| 0055-3 | 浅粉色 | ✗ 回收箱 | ✗ | not perceived | ✓† | ✗ 绿色 | ✓† | ✓† |
| 0060-1 | UGG | ✓ | ✗ LOVE | valid | ✓ | ✗ UMBO | ✓ | ✓ |

- **Valid probes: 5 of 10.** Of these, 4 of 5 are answered in the window. The
  fifth (jvEuEu65YE8-15 at G=30) got no reply: the model spoke once at chunk 1,
  about the filler, and then stayed silent through the query and tail. **None**
  of the 5 is answered out of the window. Every out-of-window reply is a
  confident invention, as in the pilot.
- **Both English items were not perceived.** One is a word on a cartoon bottle,
  the other a spoken fact in a tutorial. Both went through grounding, the
  speech check and the Whisper answer channel. Two items say nothing about
  English as a whole, but the en yield needs watching in M4.
- **The blind control catches guessable answers.** 0036 (a count of two) and
  0038 (pink flowers) are answered with no evidence, so the control works as
  designed. Small answer spaces (colours, small counts) will cost yield.
- **One G=0 control failure is not repeatable.** 0055 fails its G=0 control
  but is answered at G=60 and in both ablations. All three G=0-length plans run
  in `context` mode. At k=1 a single miss decides validity; this is the plan's
  accepted noise, shown here on 1 of 10 items.
- **Declared modality differs from the ablation.**
    - jvEuEu65YE8-5 is declared visual, but it fails with the audio muted. The
      evidence speech says 生日贺卡.
    - p-PVprguijI-13 (declared av) also needs audio.
    - The ablation, not the tag, gives the modality, as planned.
- **The video freeze is weak on short static evidence.** OmniInteract evidence
  is 4 frames, and frame 0 often already shows the answer (0060 shows UGG in
  every frame). The `ablate_video` result there says little.
- **Answer channels.** The transcript and the ASR (Paraformer for zh, Whisper
  for en) disagree on 3 of 60 probes. 0041 muted-audio is one: the Talker's
  audio lost the answer.
- **The scorer.** The key-token scorer marks 9 probes incorrect where the LLM
  marks them correct, all on 0038 and 0055 colour answers.

**Pipeline yield.**

- StreamArena: 12 of 16 candidates were grounded.
- OmniInteract: 9 of 20.
- VLM agreement, by Claude's eye (not human labels; Euan's M0 labels were not
  saved per item):
    - 9 of the OmniInteract evidence clips (5 grounded, 4 excluded): 8 agree,
      and 1 is a borderline colour-naming miss (a wine-red umbrella read as purple)
    - 2 StreamArena items: both agree
- The leakage loop made 56 clip checks and found 1 hit (a card shown in
  live-commerce filler, for the 生日贺卡 item). It was excluded and re-planned.

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

The first pilot's plans and records (before the lag-neutral queries) are kept in
`<dataset-root>/archive/pilot-v1/`.

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

Every verdict carries the reply text so it can be reviewed by hand. Key-token matching is the primary verdict **only for the pilot**:

- **M4 and experiments:** the LLM match is primary and key-token is secondary.
- **M6 benchmark runs:** the calibrated M5 judge gives the score. Key-token stays only as an extra column that flags disagreements with the judge; it is too fragile to be the benchmark's score (2026-11-26).

## Pilot results

### Rerun with lag-neutral queries (2026-11-25, runs `pilot2-baseline`, `pilot2-controls-g0`)

Same setup as below. Changes since the first run:

- the queries are lag-neutral (之前看到的…)
- the evidence ends exactly 0.3 s before the onset
- the scorer reads spoken money amounts (2块9毛9 is 2.99)

| Item | Lag (chunks) | Evidence at query | G=0 control (context-60) | Chained baseline reply | Verdict |
| --- | --- | --- | --- | --- | --- |
| 0127 hotel name | 10 | in | correct (`Lavande丽枫酒店`) | `LAVANDE露枫酒店` | correct |
| 0102 car / door | 30 | in | correct | `四车十三门` | correct |
| 0108 sunglasses price | 60 | in | correct | `168元` | correct |
| 0115 egg unit price | 180 | out | correct | `2块9毛9一颗` (invented) | incorrect |
| 0130 hotline | 300 | out | **not perceived** (`四个零，九七八五八六`) | `400-6695-0036` (invented) | excluded |
| 0124 toy price | 600 | out | correct (`629元`) | `15美元` (invented) | incorrect |

- **Retention:** all 3 in-window probes are answered and none of the 2 valid
  out-of-window ones. The model still invents an answer rather than abstaining.
- **Validity flips against the first run:** 0124 is now perceived and 0130 is
  not. That run's evidence for 0124 ended at 190 s, before 629 was legible
  (≈190.9 s); this cut ends at 191.57 s. The 0130 control misreads the
  number. One run each, so this flip is not attributed further.
- **Answer channels:** text and ASR agree on 5 of 6 chained probes and 5 of 6
  controls. The `Lavande` brand name is lost in the audio, as before.
- **Health:** every session succeeded, with no epoch changes and no dropped
  trace records. No determinism repeat was run this time.

### First run (2026-11-25, runs `pilot-baseline-1`, `-2`, `pilot-controls-g0`)

Results from 2026-11-25 on GPTQ, one session per server, `basic` window, greedy decoding. These runs used the first query wording (刚才…, "just now") and the floored evidence cut. Both have since changed, and the pilot plans have been rebuilt; the old plans are in `archive/pilot-v1/`. The pilot is a plumbing check, not a measurement: one item per lag and one run each.

| Item | Lag (chunks) | Evidence at query | G=0 control (context-60) | Chained baseline reply | Verdict |
| --- | --- | --- | --- | --- | --- |
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
