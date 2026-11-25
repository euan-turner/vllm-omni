# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evidence residency at query onset, from the server's duplex tick trace.

Join: the runner sends 200 ms appends, so stream chunk c spans client appends
``5c+1 .. 5c+5`` and its frame rides on append ``5c+3`` (the first to cross
``c + 0.5`` s). A Stage-0 unit is started while handling mailbox item ``dq_n``
(``start``), accepted as ``(epoch, seq)`` (``acc``, joined on ``attempt``), and
holds the client audio after the previous unit's ``dq_n`` up to its own. Window
drops arrive as ``ev`` rows per unit ``seq``.

A unit is *out* once its cumulative dropped tokens reach its length less one
KV block (a sink residue may stay, see ``tick_trace.py``), *partial* once any
token is dropped, else *in*. A probe is in-window when every evidence unit is
in, out-of-window when every one is out, else partial. Only drops recorded
before the query onset count; a drop is dated by the chunk of the unit it was
handled with (a silence-continuation unit takes the chunk of the client unit
before it). A trace with ``trace_dropped`` is rejected: a lost ``ev`` would
read as in-window.

External dependencies: none.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.av_fd_longmem.schema import Probe

APPEND_MS = 200
APPENDS_PER_CHUNK = 1000 // APPEND_MS
FRAME_APPEND_OFFSET = 3  # frame of chunk c rides on append 5c + 3
SINK_RESIDUE_TOKENS = 16


class TraceRejectedError(ValueError):
    pass


@dataclass(frozen=True)
class UnitSpan:
    seq: int
    first_append: int  # client append index (1-based), inclusive
    last_append: int


@dataclass(frozen=True)
class ProbeResidency:
    item_id: str
    label: str  # in | out | partial
    evidence_units: list[int]
    units_out: int
    units_partial: int
    evicted_token_frac: float | None
    first_evict_chunk: int | None
    last_evict_chunk: int | None
    chunks_since_eviction: int | None


def load_session_trace(trace_dir: Path, session_id: str) -> dict[str, list[list]]:
    records: dict[str, list[list]] = defaultdict(list)
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row[0] == "trace_dropped":
                raise TraceRejectedError(f"{path.name}: {row[1]} trace records were dropped")
            if len(row) > 2 and row[2] == session_id:
                records[row[0]].append(row)
    return records


def unit_spans(records: dict[str, list[list]]) -> list[UnitSpan]:
    """Client-append span of every accepted Stage-0 unit, in seq order."""
    trigger = {row[3]: row[4] for row in records["start"]}  # attempt -> dq_n
    accepted = sorted((row[5], trigger.get(row[3])) for row in records["acc"] if row[5] is not None)
    spans: list[UnitSpan] = []
    previous_end = 0
    for seq, dq_n in accepted:
        if not dq_n:  # silence continuation: no client audio
            continue
        spans.append(UnitSpan(seq=seq, first_append=previous_end + 1, last_append=dq_n))
        previous_end = dq_n
    return spans


def chunk_units(spans: list[UnitSpan], chunk: int) -> set[int]:
    """Seqs of the units holding stream chunk ``chunk``'s audio or frame."""
    first = chunk * APPENDS_PER_CHUNK + 1
    last = (chunk + 1) * APPENDS_PER_CHUNK
    return {span.seq for span in spans if span.first_append <= last and first <= span.last_append}


def append_chunk(append: int) -> int:
    """Stream chunk of client append ``append`` (1-based): chunk c is appends ``5c+1 .. 5c+5``."""
    return ((append - 1) * APPEND_MS) // 1000


def seq_chunk(spans: list[UnitSpan], seq: int) -> int | None:
    """Stream chunk at which the unit accepted as ``seq`` was handled.

    A client unit maps to the chunk of its last append. A silence-continuation
    unit has no client input; it maps to the latest client unit before it.
    ``None`` only when no client unit precedes ``seq``.
    """
    latest: UnitSpan | None = None
    for span in spans:
        if span.seq > seq:
            break
        latest = span
    return None if latest is None else append_chunk(latest.last_append)


def probe_residency(probe: Probe, records: dict[str, list[list]]) -> ProbeResidency:
    spans = unit_spans(records)
    evidence_units: set[int] = set()
    for c0, c1 in probe.evidence_chunks:
        for chunk in range(c0, c1 + 1):
            evidence_units |= chunk_units(spans, chunk)
    if not evidence_units:
        raise ValueError(f"{probe.item_id}: no Stage-0 unit holds the evidence chunks")
    first_len: dict[int, int] = {}
    dropped: dict[int, int] = defaultdict(int)
    evict_chunks: list[int] = []
    for row in sorted(records["ev"], key=lambda r: r[1]):
        at_chunk = seq_chunk(spans, row[4])
        if at_chunk is None:
            raise TraceRejectedError(f"ev record at seq {row[4]} precedes every client unit")
        if at_chunk >= probe.query_onset_chunk:
            continue
        for unit_seq, unit_tokens, dropped_tokens, _audio, _image in row[7]:
            if unit_seq in evidence_units:
                first_len.setdefault(unit_seq, unit_tokens)
                dropped[unit_seq] += dropped_tokens
                evict_chunks.append(at_chunk)
    out = {seq for seq in evidence_units if seq in first_len and dropped[seq] >= first_len[seq] - SINK_RESIDUE_TOKENS}
    partial = {seq for seq in evidence_units if dropped[seq] > 0} - out
    label = "in" if not out and not partial else "out" if out == evidence_units else "partial"
    known = [seq for seq in evidence_units if seq in first_len]
    frac = sum(dropped[seq] for seq in known) / sum(first_len[seq] for seq in known) if known else 0.0
    return ProbeResidency(
        item_id=probe.item_id,
        label=label,
        evidence_units=sorted(evidence_units),
        units_out=len(out),
        units_partial=len(partial),
        evicted_token_frac=frac if len(known) == len(evidence_units) or not known else None,
        first_evict_chunk=min(evict_chunks) if evict_chunks else None,
        last_evict_chunk=max(evict_chunks) if evict_chunks else None,
        chunks_since_eviction=probe.query_onset_chunk - max(evict_chunks) if evict_chunks else None,
    )


def residency_rows(probes: list[Probe], records: dict[str, list[list]]) -> list[dict[str, object]]:
    return [asdict(probe_residency(probe, records)) for probe in probes]
