# SPDX-License-Identifier: Apache-2.0
"""Drive long stitched Omni-DuplexEval streams through a duplex server, then score each segment.

Purpose: prototype for long-context full-duplex profiling (branch
long-context-prototype). ``generate`` streams one or more stitched sessions
in real time (see ``stitch.py``) and writes, per session, a manifest, the full
timed-sentence output and one judge-format response per segment. ``evaluate``
scores every segment with the existing Omni-DuplexEval judge, so a sample's
score can be compared across its depths in the stream (local probe).

Per-tick timing is written by the server (``serve.sh TICK_TRACE_DIR=...``),
not by this driver; ``analyze.py`` joins the two by session start order.
With ``--tick-trace-dir``, the driver also treats ``--tick-stall-s`` without a
new Stage-0 tick record as a server hang and ends every session as aborted
(the server's own input-wait timeout is off for long runs, since a model that
listens silently for minutes is legitimate).

External dependencies: vllm_omni.benchmarks.duplex (client, dataset, judge),
``datasets`` for local Hugging Face splits, a running duplex server and, for
``evaluate``, an OpenAI-compatible judge server.

Usage:
    python benchmarks/duplex_session/long_context/run_long_session.py generate \\
        --dataset "$HF_HOME/omni_duplex_eval" --split RTD_Omni --modality av \\
        --target-minutes 12 --ref-audio /path/HT_ref_audio.wav --out-dir runs/av_12min
    python benchmarks/duplex_session/long_context/run_long_session.py evaluate \\
        --dataset "$HF_HOME/omni_duplex_eval" --split RTD_Omni --out-dir runs/av_12min \\
        --judge-base-url http://127.0.0.1:8000 --judge-model Qwen/Qwen2.5-VL-72B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stitch import Segment, plan_segments, segment_duration_s, split_sentences, stitch_prepared  # noqa: E402

DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"
STITCHED_SPLIT = "stitched"

T = TypeVar("T")


class TickStallError(RuntimeError):
    """No Stage-0 tick record was written for longer than the stall limit."""


def latest_tick_mtime(trace_dir: Path) -> float | None:
    return max((path.stat().st_mtime for path in trace_dir.glob("ticks_stage0_*.jsonl")), default=None)


async def watch_ticks(trace_dir: Path, stall_s: float, *, poll_s: float = 5.0, clock=time.time) -> None:
    """Raise ``TickStallError`` once no tick file has been written for ``stall_s`` (counting from the call)."""
    started = clock()
    while True:
        await asyncio.sleep(poll_s)
        idle = clock() - max(latest_tick_mtime(trace_dir) or started, started)
        if idle > stall_s:
            raise TickStallError(f"no Stage-0 tick for {idle:.0f}s (limit {stall_s:.0f}s); server presumed hung")


async def until_stall(work: Awaitable[T], stall: asyncio.Future) -> T:
    """Await ``work``; if ``stall`` finishes first, cancel ``work`` and raise the stall's exception."""
    task = asyncio.ensure_future(work)
    await asyncio.wait({task, stall}, return_when=asyncio.FIRST_COMPLETED)
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    stall.result()
    raise TickStallError("stall watchdog ended without an error")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def session_start_indices(first: int, sessions: int, stride: int, count: int) -> list[int]:
    """Start sample index of each concurrent session; stride rotates which sample leads."""
    return [(first + j * stride) % count for j in range(sessions)]


def write_segment_artifacts(
    session_dir: Path,
    segments: list[Segment],
    timed_sentences: list[dict[str, Any]],
    base_meta: dict[str, Any],
) -> None:
    """One judge-format response (``<index>_<id>.json`` + ``.meta.json``) per segment."""
    segment_dir = session_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    by_segment = split_sentences(timed_sentences, segments)
    for segment in segments:
        stem = f"{segment.index:04d}_{segment.sample_id}"
        meta = {**base_meta, "id": segment.sample_id, "segment": segment.as_dict()}
        (segment_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (segment_dir / f"{stem}.json").write_text(
            json.dumps(by_segment[segment.index], ensure_ascii=False, indent=2), encoding="utf-8"
        )


async def _generate(args: argparse.Namespace) -> int:
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample, load_samples
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import (
        PreparedReferenceAudio,
        generate_sample,
        prepare_sample,
    )

    samples = load_samples(
        args.dataset, split=args.split, family="rtd", media_root=args.media_root, limit=args.limit, ids=args.ids
    )
    if not samples:
        raise ValueError("no samples selected; check --dataset/--split/--ids/--limit")
    out_dir = Path(args.out_dir)
    media_dir = out_dir / ".media"
    reference = PreparedReferenceAudio.from_path(args.ref_audio) if args.ref_audio else None
    prepared = [
        await asyncio.to_thread(prepare_sample, sample, media_dir=media_dir, ref_audio=reference, fps=args.fps)
        for sample in samples
    ]
    durations = [segment_duration_s(len(item.pcm), item.duration) for item in prepared]
    with_frames = args.modality == "av"
    starts = session_start_indices(args.start_index, args.sessions, args.session_stride, len(samples))

    run_meta = {
        "git_sha": _git_sha(),
        "modality": args.modality,
        "target_minutes": args.target_minutes,
        "fps": args.fps,
        "split": args.split,
        "sessions": args.sessions,
        "stagger_s": args.stagger_s,
        "url": args.url,
        "model": args.model,
        "tick_trace_dir": args.tick_trace_dir,
        "tick_stall_s": args.tick_stall_s,
        "argv": sys.argv,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    loop = asyncio.get_running_loop()
    if args.tick_trace_dir and args.tick_stall_s > 0:
        stall: asyncio.Future = asyncio.ensure_future(watch_ticks(Path(args.tick_trace_dir), args.tick_stall_s))
    else:
        stall = loop.create_future()

    async def run_session(session_index: int, start_index: int) -> dict[str, Any]:
        await asyncio.sleep(session_index * args.stagger_s)
        segments = plan_segments(
            [sample.id for sample in samples],
            [sample.split for sample in samples],
            durations,
            target_s=args.target_minutes * 60,
            start_index=start_index,
        )
        stream = stitch_prepared(prepared, segments, with_frames=with_frames)
        stitched = DuplexSample(
            id=f"session{session_index}",
            split=STITCHED_SPLIT,
            family="rtd",
            task_type=None,
            video=None,
            video_duration=stream.duration,
        )
        session_dir = out_dir / f"session{session_index}"
        t_start = time.monotonic()
        try:
            result = await until_stall(
                generate_sample(
                    stitched,
                    url=args.url,
                    model=args.model,
                    ref_audio=None,
                    output_root=session_dir,
                    fps=args.fps,
                    pace="realtime",
                    overwrite=True,
                    prepared=stream,
                    defer_artifacts=True,
                    instructions=args.instructions or None,
                ),
                stall,
            )
        except Exception as exc:
            # A server-side session close (e.g. a stage failure) surfaces as a
            # websocket error. Keep the session window so the ticks traced up to
            # the failure are still analysed.
            result = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = result.error
        t_end = time.monotonic()
        manifest = {
            "session_index": session_index,
            "start_index": start_index,
            "t_start_monotonic": t_start,
            "t_end_monotonic": t_end,
            "elapsed_s": t_end - t_start,
            "stream_duration_s": stream.duration,
            "num_frames": len(stream.frames),
            "error": error,
            "aborted": result is None,
            "output_tokens": result.output_tokens if result is not None else None,
            "segments": [segment.as_dict() for segment in segments],
            "session_metrics": result.session_metrics if result is not None else None,
            "metadata": result.metadata if result is not None else None,
        }
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        if result is not None:
            (session_dir / "timed_sentences.json").write_text(
                json.dumps(result.timed_sentences, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (session_dir / "request_metrics.json").write_text(
                json.dumps(result.request_metrics, indent=2, default=str), encoding="utf-8"
            )
            write_segment_artifacts(session_dir, segments, list(result.timed_sentences), dict(result.metadata))
        print(
            f"session{session_index}: {stream.duration:.0f}s planned, {t_end - t_start:.0f}s elapsed, "
            f"{len(segments)} segments, error={error or 'none'}",
            flush=True,
        )
        return manifest

    try:
        manifests = await asyncio.gather(*(run_session(index, start) for index, start in enumerate(starts)))
    finally:
        stall.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await stall
    return 0 if all(not manifest["error"] for manifest in manifests) else 1


def _evaluate(args: argparse.Namespace) -> int:
    import concurrent.futures

    from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import load_samples
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import evaluate_sample
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge

    out_dir = Path(args.out_dir)
    samples = {
        sample.id: sample
        for sample in load_samples(args.dataset, split=args.split, family="rtd", media_root=args.media_root)
    }
    judge = DuplexJudge(args.judge_base_url, args.judge_model, api_key=args.judge_api_key)
    jobs = []
    for response_path in sorted(out_dir.glob("session*/segments/*.json")):
        if response_path.name.endswith(".meta.json"):
            continue
        meta = json.loads(response_path.with_name(response_path.stem + ".meta.json").read_text(encoding="utf-8"))
        score_path = response_path.parent.parent / "scores" / response_path.name
        if score_path.exists() and not args.overwrite:
            continue
        jobs.append((samples[meta["id"]], response_path, score_path))

    def evaluate_one(job: tuple[Any, Path, Path]) -> None:
        sample, response_path, score_path = job
        evaluate_sample(
            sample,
            response_path,
            score_path,
            judge,
            judge_fps=args.judge_fps,
            judge_video_mode=args.judge_video_mode,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        list(executor.map(evaluate_one, jobs))
    print(f"scored {len(jobs)} segments", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("generate", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--split", default="RTD_Omni")
        command.add_argument("--media-root")
        command.add_argument("--out-dir", required=True)
    generate = sub.choices["generate"]
    generate.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--ref-audio")
    generate.add_argument("--modality", choices=("av", "audio"), default="av")
    generate.add_argument("--target-minutes", type=float, required=True)
    generate.add_argument("--fps", type=float, default=1.0)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--ids", nargs="*")
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--sessions", type=int, default=1)
    generate.add_argument("--session-stride", type=int, default=7)
    generate.add_argument("--stagger-s", type=float, default=2.0)
    generate.add_argument("--instructions", default="")
    generate.add_argument("--tick-trace-dir", help="recorded in run.json for analyze.py; enables the stall watchdog")
    generate.add_argument(
        "--tick-stall-s",
        type=float,
        default=120.0,
        help="abort all sessions after this long without a new Stage-0 tick record (0 disables)",
    )
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--judge-base-url", default="http://127.0.0.1:8000")
    evaluate.add_argument("--judge-model", required=True)
    evaluate.add_argument("--judge-api-key", default="EMPTY")
    evaluate.add_argument("--judge-video-mode", choices=("video_url", "frame-sample"), default="video_url")
    evaluate.add_argument("--judge-fps", type=int, default=2)
    evaluate.add_argument("--eval-workers", type=int, default=4)
    evaluate.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "generate":
        if args.sessions < 1:
            raise ValueError("--sessions must be at least 1")
        return asyncio.run(_generate(args))
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
