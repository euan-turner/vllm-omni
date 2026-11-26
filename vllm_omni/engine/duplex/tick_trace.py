# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in per-tick timing trace for duplex sessions.

Enable by setting ``VLLM_OMNI_DUPLEX_TICK_TRACE_DIR`` to a directory. Every
process that records ticks (the orchestrator and the Stage-0 workers) writes its
own ``duplex-ticks-<role>-<pid>.jsonl`` there; ``benchmarks/duplex_session/
tick_report.py`` joins them into per-tick latency and slack. Timestamps are
``time.monotonic_ns()``, which is comparable across processes on one host.

Records (JSON arrays, first element is the kind):

- ``["in", t, session_id, n]``: the n-th ``AppendAudio`` entered the session mailbox.
- ``["dq", t, session_id, n]``: the n-th ``AppendAudio`` left the mailbox.
- ``["start", t, session_id, attempt, dq_n, silence]``: a stage append was started
  while handling mailbox item ``dq_n`` (0 if none); ``silence`` marks a
  runtime silence continuation.
- ``["acc", t, session_id, attempt, epoch, seq, request_id, submit_t]``: the stage
  accepted that append.
- ``["out", t, session_id, stage_id, request_id, segment_finished, n_tokens]``:
  a stage output reached the orchestrator, before any session filtering;
  ``n_tokens`` is the length of its first completion's ``token_ids`` (cumulative
  per request for streaming Stage-0 requests).
- ``["w", t, session_id, epoch, seq, prompt_len, num_computed, num_input_tokens,
  window_replaced]``: Stage-0 built the prompt for append ``(epoch, seq)``
  (chunked prefill repeats it per chunk).
- ``["ev", t, session_id, epoch, seq, path, kv, units]``: Stage-0 dropped
  window units while handling append ``(epoch, seq)``. ``path`` is ``compact``
  (KV re-anchor; ``kv`` is ``{delta, sink_end, moved_from, prefix_tokens}``,
  the freed KV range and the worker's window start) or ``replace`` (validated
  prompt rebuild; ``kv`` is ``None``). Each ``units`` row is ``[unit_seq,
  unit_tokens, dropped_tokens, dropped_audio_tokens, dropped_image_tokens]``:
  ``unit_seq`` is the append whose input opened the unit, ``unit_tokens`` its
  length before this drop. Reading rules: a compaction can cut a unit
  part-way, so sum rows per ``unit_seq`` across records; a unit is out of the
  window once none of its audio/image tokens remain, not once every token is
  gone (a few may stay in the sink or the kept tail). An append that closes no
  model unit (buffering) never appears here; its audio rides in the next
  append's unit. A ``trace_dropped`` record means ``ev`` records may be missing.
- ``["fwd", t_start, session_id, src_stage_id, request_id, t_end]``: the
  orchestrator forwarded a stage output to the next stage (input processing,
  e.g. ``llm2tts``, plus the submit), inline on its loop.
- ``["h", t, request_id, source_prompt_ids, handoff_prompt_ids, hidden_rows]``:
  ``llm2tts`` built a native-duplex Talker handoff; ``source_prompt_ids`` is
  the size of the accumulated ``duplex_prompt_token_ids`` it read (the Thinker
  prompt tail, 1) and ``handoff_prompt_ids`` the prompt ids copied into the
  Talker request (0: native handoffs carry none).
- ``["w1", t, request_id, prompt_len, num_computed, condition_rows, span_len]``:
  the Talker built a native-duplex prefill span from a handoff (chunked
  prefill repeats it per chunk).

Design doc: docs/design/fullduplex.md; usage in docs/contributing/profiling.md.
External dependencies: none.
"""

from __future__ import annotations

from vllm_omni.utils.jsonl_trace import JsonlTraceWriter, jsonl_trace_writer

DUPLEX_TICK_TRACE_ENV = "VLLM_OMNI_DUPLEX_TICK_TRACE_DIR"


def duplex_tick_tracer(role: str) -> JsonlTraceWriter | None:
    """This process's duplex tick trace writer for ``role``, or ``None`` when tracing is off."""
    return jsonl_trace_writer(DUPLEX_TICK_TRACE_ENV, f"duplex-ticks-{role}")
