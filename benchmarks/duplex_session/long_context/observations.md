# Long-context duplex observations (MiniCPM-o 4.5, GH200)

Prototype results, branch `long-context-prototype`, 2026-10-28. One session per
run on one Isambard-AI GH200 (all three stages on one GPU), stitched RTD_Omni
streamed in real time at 1 unit/s. Metric definitions are in
[`tick_metrics.py`](tick_metrics.py).

| Run | Modality | Reached | Ended by |
| --- | --- | --- | --- |
| `7793458` | audio | 47.2k tokens, ~53 min | host-RAM OOM kill (Stage 1) |
| `7793983` | audio + video, 1 fps | 75.2k tokens, ~15 min | host-RAM OOM kill (Stage 1, 2) |

Context past the HF limit of 40960 was reached by extending the RoPE table
(plain extrapolation). Extending context is not a goal (see 3); the extension
is kept only to observe degenerate behaviour past the limit.

## 1. Listen ticks leave latency slack within the context length

| Kind | Audio slope | AV slope | Audio / AV at ~39k |
| --- | --- | --- | --- |
| listen | 0.93 ms / 1k tokens | 0.94 ms / 1k tokens | ~80 ms / ~80 ms |
| speak | 2.5 ms / 1k tokens | 2.9 ms / 1k tokens | (too few ticks) |

- Listen service grows linearly and identically across modalities; video adds
  a constant ~12 ms (intercept 50 ms vs 38 ms).
- At the 40960 limit a listen tick uses under 10% of the 1 s tick deadline;
  the fit reaches 1 s only at ~1M tokens. Stage-0 compute alone does not break
  real time within the context length.
- Needs re-confirming on an edge device; GH200 numbers do not transfer.

## 2. Speak ticks are too sparse for a reliable fit

- The model mostly stops speaking normally after ~12k tokens: 226 (audio) and
  69 (AV) speak ticks against 2,702 and 545 listen ticks.
- The speak slope is therefore noisy. More speak samples are needed, e.g. from
  prompts that elicit longer answers, or by measuring speak cost at fixed
  context depths.

## 3. Sessions lock into an open turn inside the context length

- **Behaviour:** after a long listening stretch the model emits a single
  `<|speak|>` (opening a turn) and then keeps sampling `<|listen|>` without ever
  emitting `<|turn_eos|>`. Because the turn is open, the host sampling policy
  rewrites each `<|listen|>` to `<|tts_bos|>` and decoding continues
  ([`minicpmo_4_5_omni.py`](../../../vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py),
  `_finalize_minicpmo45_native_duplex_sample`), so ticks become text-free runs
  of `<|tts_bos|>` at the 20-token cap ("degenerate" ticks in the metrics).
  Confirmed by a sampler counter: every `<|tts_bos|>` in these runs was a
  rewritten `<|listen|>`; the model never sampled `<|tts_bos|>` itself.
- **Onset:** ~38.9k tokens in both audio (`7793458`) and AV capped at 40960
  (`7798235`); ~41–45k in AV runs with the extended RoPE. In audio the lock-in
  never ended; in one extended AV run (`7798408`) the model closed the turn at
  ~64.7k and listen ticks returned.
- **Onset is the main result.** The lock-in starts before the 40960 limit, so
  extending context artificially is pointless. The onset gives a target for
  context compaction/compression, or at least an upper watermark by which
  compaction must trigger. The turn-taking policy is a second lever: it
  converts a model that wants to listen into forced speech.
- **Cost:** a degenerate tick costs 0.22–0.30 s (3–4× a listen tick at the
  same context), and ~85% are followed by a server-inserted silence
  continuation unit, so Stage 0 runs ~2 updates per second of input.

## 4. Real time is first lost in the degenerate phase (audio)

- `queue_p99` (append arrival to first scheduled step) is ~0 before
  degeneration in both runs. In the audio degenerate phase it is 1.4–1.7 s,
  above the 1 s unit period; in the AV degenerate phase 0.2–0.5 s.
- Two ~0.22 s updates per second should not produce a 1.5 s wait, so
  something else serialises in the audio session. **Next step:** inspect the
  audio run's raw tick timeline before attributing the backlog to
  degeneration alone.

## Open issues, not results

- **Host-RAM OOM** ends every long session that reaches the lock-in: Stage 1's
  RSS grows ~1.5 GB per 10 min in normal operation and ~9 GB per 30 s during
  the lock-in, until Slurm kills it at the ~115 GB job limit. The growth is not
  in the Talker model's per-request state. The only Python-visible data that
  grows with it is a scalar list reachable from Stage 1's request (~30M items),
  too small on its own to explain ~90 GB; the source is still unidentified.
- **GPU memory** was a measurement artefact, now fixed: Stage 0's caching
  allocator reserved ~25 GiB of context-sized temporaries by 30k tokens
  (allocated stayed flat at ~45.5 GiB); `expandable_segments:True` keeps
  reserved at ~47–49 GiB.
- **Silence continuations** (`vllm_omni/engine/duplex/session/model_channel.py`)
  can race the client's next append and insert unheard silence (~15 s over the
  first 20 min of one audio run); `suspected_continuation` in the metrics is a
  heuristic for them.
