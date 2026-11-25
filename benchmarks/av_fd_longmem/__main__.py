# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""av_fd_longmem command line: ``python -m benchmarks.av_fd_longmem <stage> ...``.

Stages:
- ``build-pilot``: OmniInteract pilot items, filler and plans (needs a Qwen3-TTS server).
- ``run``: stream plans against a deployed duplex server, one session at a time.
- ``score``: residency labels, answer verdicts and a per-probe summary for a run.

External dependencies: see each stage's module.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

DEFAULT_DATASET_ROOT = Path("~/Documents/datasets/av_fd_longmem").expanduser()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_pilot(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.asr import FunasrSpeechSegmenter
    from benchmarks.av_fd_longmem.pilot import PilotPaths, build_filler, build_items, build_plans

    paths = PilotPaths(data_root=args.omniinteract_root, out_root=args.dataset_root)
    segmenter = FunasrSpeechSegmenter(device=args.asr_device)
    items = build_items(paths, tts_url=args.tts_url, segmenter=segmenter)
    filler = build_filler(paths, needed_chunks=args.filler_chunks)
    plans = build_plans(paths, items, filler)
    for plan in plans:
        print(f"{plan.plan_id}: {plan.total_chunks} chunks, probes {[(p.item_id, p.gap_chunks) for p in plan.probes]}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.runner import ServerTarget, git_revision, run_plan

    run_dir: Path = args.dataset_root / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = ServerTarget(base_url=args.base_url, model=args.model, ref_audio=args.ref_audio, trace_dir=args.trace_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        run_json.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "created_unix_s": time.time(),
                    "base_url": args.base_url,
                    "model": args.model,
                    "deploy_config": args.deploy_config,
                    "ref_audio": args.ref_audio,
                    "git_revision": git_revision(REPO_ROOT),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    failures = 0
    for plan_path in args.plans:
        session = asyncio.run(run_plan(plan_path, args.dataset_root, run_dir, target))
        ok = session["success"] and not session["epoch_changed"]
        failures += not ok
        print(f"{session['plan_id']}: success={session['success']} epochs={session['epochs']} {session['error'] or ''}")
    return 1 if failures else 0


def _score(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.asr import FunasrSpeechSegmenter
    from benchmarks.av_fd_longmem.evaluate import evaluate_run

    for run_id in args.run_ids:
        evaluate_run(args.dataset_root, run_id, asr=FunasrSpeechSegmenter(device=args.asr_device))
        print((args.dataset_root / "runs" / run_id / "summary.md").read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.av_fd_longmem", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    stages = parser.add_subparsers(dest="stage", required=True)

    build = stages.add_parser("build-pilot", help="Build the OmniInteract pilot items, filler and plans.")
    build.add_argument("--omniinteract-root", type=Path, required=True, help="OmniInteract `data` directory.")
    build.add_argument("--tts-url", default="http://127.0.0.1:8091")
    build.add_argument("--asr-device", default="cuda:1")
    build.add_argument("--filler-chunks", type=int, default=2000, help="Filler seconds to materialise.")
    build.set_defaults(handler=_build_pilot)

    run = stages.add_parser("run", help="Stream plans against a deployed duplex server.")
    run.add_argument("plans", type=Path, nargs="+", help="Plan JSON files, run in order.")
    run.add_argument("--run-id", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8099")
    run.add_argument("--model", default="openbmb/MiniCPM-o-4_5-GPTQ")
    run.add_argument("--ref-audio", help="Reference WAV for the output voice (MiniCPM-o: assets/HT_ref_audio.wav).")
    run.add_argument(
        "--trace-dir", type=Path, help="The server's VLLM_OMNI_DUPLEX_TICK_TRACE_DIR, copied after each session."
    )
    run.add_argument("--deploy-config", help="Deploy YAML the server was started with (recorded in run.json).")
    run.set_defaults(handler=_run)

    score = stages.add_parser("score", help="Label, judge and summarise finished runs.")
    score.add_argument("run_ids", nargs="+")
    score.add_argument("--asr-device", default="cuda:1")
    score.set_defaults(handler=_score)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
