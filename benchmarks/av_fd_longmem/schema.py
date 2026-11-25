# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Records of the av_fd_longmem dataset and session plans, stored as JSONL/JSON.

Items link an item's clip files to its expected plaintext answer; filler clips
supply the lag; a session plan lists clip references to play back to back. All
times inside a plan are stream chunk indices (1 chunk = 1 s = one append).
Every file carries ``schema_version``.

External dependencies: none.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1

AudioMode = Literal["keep", "mute"]
VideoMode = Literal["keep", "freeze"]
SegmentRole = Literal["filler", "evidence", "query"]


@dataclass(frozen=True)
class ClipRef:
    """One materialised clip: ``chunks`` whole seconds, clip second k is chunk k."""

    path: str
    chunks: int


@dataclass(frozen=True)
class QueryClip:
    """Backdrop video with the spoken query laid in, then a silent answer tail.

    ``query_onset_chunk`` and ``query_end_ms`` are relative to the clip start;
    ``answer_chunks`` is the length of the silent tail after the query.
    """

    path: str
    chunks: int
    query_onset_chunk: int
    query_end_ms: int
    answer_chunks: int


@dataclass(frozen=True)
class ItemClips:
    evidence: list[ClipRef]
    query: QueryClip


@dataclass(frozen=True)
class Item:
    """A reactive probe: evidence clip(s), a past-tense spoken query and the expected answer."""

    item_id: str
    source: str
    lang: str
    clips: ItemClips
    query_text: str
    query_text_original: str
    answer: str
    modality_declared: str
    provenance: dict[str, Any]
    status: str = "candidate"
    task: str = "reactive"
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class FillerClip:
    filler_id: str
    path: str
    chunks: int
    source: str
    domain: str | None
    provenance: dict[str, Any]
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class Segment:
    """``play_chunks`` is a prefix of the clip; ``stream_chunk0`` its first stream chunk."""

    role: SegmentRole
    clip: str
    play_chunks: int
    stream_chunk0: int
    audio: AudioMode = "keep"
    video: VideoMode = "keep"


@dataclass(frozen=True)
class Probe:
    """Stream-chunk coordinates of one item inside a plan.

    ``evidence_chunks`` are inclusive ``[c0, c1]`` ranges; the gap of the probe
    is ``query_onset_chunk - evidence_chunks[-1][1] - 1``.
    """

    item_id: str
    evidence_chunks: list[list[int]]
    query_onset_chunk: int
    query_end_ms: int
    answer_window_chunks: list[int]

    @property
    def gap_chunks(self) -> int:
        return self.query_onset_chunk - self.evidence_chunks[-1][1] - 1


@dataclass(frozen=True)
class SessionPlan:
    plan_id: str
    kind: str
    seed: int
    condition: dict[str, Any]
    segments: list[Segment]
    probes: list[Probe]
    total_chunks: int
    schema_version: int = SCHEMA_VERSION


T = TypeVar("T")


def _from_value(tp: Any, value: Any) -> Any:
    if is_dataclass(tp) and isinstance(value, dict):
        return from_dict(tp, value)
    origin = get_origin(tp)
    if origin is list and isinstance(value, list):
        (inner,) = get_args(tp)
        return [_from_value(inner, v) for v in value]
    return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a record from its JSON form, rejecting a different ``schema_version``."""
    if "schema_version" in data and data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{cls.__name__}: schema_version {data['schema_version']} != {SCHEMA_VERSION}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    return cls(**{name: _from_value(hints[name], value) for name, value in data.items()})


def to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls: type[T]) -> Iterator[T]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield from_dict(cls, json.loads(line))


def write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, cls: type[T]) -> T:
    return from_dict(cls, json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "SCHEMA_VERSION",
    "ClipRef",
    "FillerClip",
    "Item",
    "ItemClips",
    "Probe",
    "QueryClip",
    "Segment",
    "SessionPlan",
    "from_dict",
    "read_json",
    "read_jsonl",
    "to_dict",
    "write_json",
    "write_jsonl",
]
