# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run session plans against a deployed duplex server, one realtime session per plan.

A plan is rendered into one PCM stream and one JPEG per chunk, then streamed by
``vllm_omni.benchmarks.omniinteract.run_omniinteract_case``, which owns the
realtime clock (200 ms appends, frame k with the append that crosses k + 0.5 s),
playback acknowledgements and output collection. The session is configured
from the plan's condition: window mode always sent explicitly, temperature,
reference audio. A session whose epoch ever changes is marked and excluded.

Per plan the run directory gets ``sessions/<plan_id>/{output.wav,
wav_transcript.json, events.json, result.json, session.json}``; the server's
tick-trace files are copied to ``trace/`` after every session.

External dependencies: a running vLLM-Omni duplex server with
``VLLM_OMNI_DUPLEX_TICK_TRACE_DIR`` set; PyAV (rendering).
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.av_fd_longmem.render import render_units
from benchmarks.av_fd_longmem.schema import SessionPlan, read_json
from vllm_omni.benchmarks.data_modules.omniinteract_dataset import OmniInteractCase, OmniInteractPreparedInput
from vllm_omni.benchmarks.omniinteract import (
    OmniInteractBenchmarkConfig,
    publish_deferred_case_artifacts,
    run_omniinteract_case,
)
from vllm_omni.clients.duplex import reference_audio_data_url

RUN_SUBSET = "av_fd_longmem"


@dataclass(frozen=True)
class ServerTarget:
    base_url: str
    model: str
    ref_audio: str | None
    trace_dir: Path | None


def session_extra_body(condition: dict[str, object]) -> dict[str, object]:
    """Window settings for ``session.update``; the mode is always explicit (the server default is ``off``)."""
    mode = condition.get("window_mode")
    if mode not in {"off", "basic", "context"}:
        raise ValueError(f"plan condition needs window_mode off|basic|context, got {mode!r}")
    extra: dict[str, object] = {"sliding_window_mode": mode}
    for key in (
        "basic_window_high_tokens",
        "basic_window_low_tokens",
        "context_previous_max_tokens",
        "context_max_units",
    ):
        if key in condition:
            extra[key] = condition[key]
    return extra


def prepared_input(plan: SessionPlan, root: Path, ref_audio: str | None) -> OmniInteractPreparedInput:
    pcm = bytearray()
    frames: list[str] = []
    for chunk_pcm, jpeg in render_units(plan, root):
        pcm += chunk_pcm
        frames.append(base64.b64encode(jpeg).decode("ascii"))
    return OmniInteractPreparedInput(
        duration_s=float(plan.total_chunks),
        pcm16=bytes(pcm),
        video_frames=tuple(frames),
        ref_audio_data_url=reference_audio_data_url(ref_audio),
    )


def _session_case(plan: SessionPlan, plan_path: Path) -> OmniInteractCase:
    return OmniInteractCase(
        subset=RUN_SUBSET,
        video_rel=f"{plan.plan_id}.mp4",
        video_path=plan_path,
        annotation_path=plan_path,
        scene_type=plan.kind,
    )


def _epochs(events: list[dict[str, object]]) -> list[int]:
    seen: list[int] = []
    for event in events:
        epoch = event.get("epoch")
        if isinstance(epoch, int) and epoch not in seen:
            seen.append(epoch)
    return seen


def _session_id(events: list[dict[str, object]]) -> str | None:
    for event in events:
        if event.get("type") == "session.created":
            session = event.get("session")
            if isinstance(session, dict) and isinstance(session.get("id"), str):
                return session["id"]
    return None


def git_revision(repo: Path) -> dict[str, object]:
    """HEAD and whether the working tree had uncommitted changes."""
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"head": head, "dirty": bool(status.strip())}


async def run_plan(plan_path: Path, dataset_root: Path, run_dir: Path, target: ServerTarget) -> dict[str, object]:
    """Stream one plan in real time and write its session artefacts; returns ``session.json``."""
    plan = read_json(plan_path, SessionPlan)
    temperature = plan.condition.get("temperature")
    config = OmniInteractBenchmarkConfig(
        model=target.model,
        base_url=target.base_url,
        output_root=run_dir / "sessions",
        ref_audio=target.ref_audio,
        extra_body=session_extra_body(plan.condition),
        temperature=float(temperature) if isinstance(temperature, int | float) else None,
        max_video_duration_s=float(plan.total_chunks + 60),
    )
    case = _session_case(plan, plan_path)
    prepared = prepared_input(plan, dataset_root, target.ref_audio)
    started = time.time()
    result = await run_omniinteract_case(case, config, request_index=plan.plan_id, prepared_input=prepared)
    publish_deferred_case_artifacts(config.output_root, case, result)
    published = Path(result.output_dir)
    session_dir = run_dir / "sessions" / plan.plan_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    published.rename(session_dir)
    events_path = session_dir / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
    epochs = _epochs(events)
    session = {
        "plan_id": plan.plan_id,
        "plan_path": str(plan_path.resolve()),
        "plan_kind": plan.kind,
        "condition": plan.condition,
        "session_id": _session_id(events),
        "success": result.success,
        "error": result.error,
        "epochs": epochs,
        "epoch_changed": len(epochs) > 1,
        "started_unix_s": started,
        "total_chunks": plan.total_chunks,
    }
    (session_dir / "session.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    if target.trace_dir is not None:
        trace_out = run_dir / "trace"
        trace_out.mkdir(parents=True, exist_ok=True)
        for trace_file in target.trace_dir.glob("*.jsonl"):
            shutil.copy2(trace_file, trace_out / trace_file.name)
    return session
