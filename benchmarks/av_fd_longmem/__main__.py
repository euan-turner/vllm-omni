# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""av_fd_longmem command line: ``python -m benchmarks.av_fd_longmem <stage> ...``.

Stages:
- ``build-pilot``: OmniInteract pilot items, filler and plans (needs a Qwen3-TTS server).
- ``run``: stream plans against a deployed duplex server, one session at a time.
- ``score``: residency labels, answer verdicts and a per-probe summary for a run.
- ``single-plans``: single-probe retention plans for one built item at several lags.
- ``audit-omniinteract``: M0 onset audit of an OmniInteract subset (onset match, offsets, pre-onset room).
- ``audit-streamarena``: M0 StreamArena exclusion counts, plus evidence/filler clips of fetched videos.
- ``audit-codec``: M0 ASR error of spoken queries as WAV vs AAC vs FLAC (needs a Qwen3-TTS server).
- ``audit-handcheck``: M0 evidence hand-check sample (contact sheets + evidence ASR) for both sources.
- ``window-plans``: M1 filler-only window plans, one per window mode.
- ``window-report``: M1 window measurements (tokens per unit, eviction lag, resident span) for a run.
- ``set-annotate``: LLM/VLM annotation of a source's candidates into an item set (needs LLM and VLM servers).
- ``set-items``: pick an item set's items, speak their queries, cut query clips (needs TTS and VLM servers).
- ``set-plans``: leakage-checked control, ablation and retention plans for an item set (needs a VLM server).
- ``render-preview``: one plan rendered to an mp4, exactly as streamed, for checking by eye and ear.

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


def _transcriber(device: str):  # -> asr.BilingualTranscriber
    import os

    from benchmarks.av_fd_longmem.asr import BilingualTranscriber, FunasrSpeechSegmenter

    os.environ.setdefault("SEED_TTS_EVAL_DEVICE", device)
    return BilingualTranscriber(FunasrSpeechSegmenter(device=device))


def _score(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.evaluate import evaluate_run

    asr = _transcriber(args.asr_device)
    llm_match = None
    if args.llm_url:
        from benchmarks.av_fd_longmem.annotate import answers_match
        from benchmarks.av_fd_longmem.llm import LLM_MODEL, LLM_REVISION, CachedChat, ChatModel

        chat = CachedChat(ChatModel(args.llm_url, LLM_MODEL, LLM_REVISION), args.dataset_root / "cache" / "llm")

        def llm_match(item, reply: str) -> bool:
            return bool(answers_match(chat, item.query_text, item.answer, reply).output["match"])

    for run_id in args.run_ids:
        evaluate_run(args.dataset_root, run_id, asr=asr, llm_match=llm_match)
        print((args.dataset_root / "runs" / run_id / "summary.md").read_text(encoding="utf-8"))
    return 0


def _single_plans(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.pilot import PilotPaths, build_single_plans
    from benchmarks.av_fd_longmem.schema import FillerClip, Item, read_jsonl

    paths = PilotPaths(data_root=Path(), out_root=args.dataset_root)
    items = list(read_jsonl(paths.source_root / "items.jsonl", Item))
    filler = list(read_jsonl(paths.source_root / "filler.jsonl", FillerClip))
    for plan in build_single_plans(paths, items, filler, item_id=args.item, lags=args.lags):
        print(f"{plan.plan_id}: {plan.total_chunks} chunks, probes {[(p.item_id, p.gap_chunks) for p in plan.probes]}")
    return 0


def _audit_omniinteract(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.asr import FunasrSpeechSegmenter
    from benchmarks.av_fd_longmem.audit import audit_subset, summarise_onsets

    out_dir: Path = args.dataset_root / "omniinteract" / "audit" / args.subset
    rows = audit_subset(args.omniinteract_root, args.subset, out_dir, FunasrSpeechSegmenter(device=args.asr_device))
    summary = summarise_onsets(rows)
    (out_dir / "onset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _audit_streamarena(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.audit import materialise_streamarena_video, summarise_streamarena
    from benchmarks.av_fd_longmem.sources.streamarena import recall_rows

    source_root: Path = args.dataset_root / "streamarena"
    annotations = source_root / "annotations"
    out_dir = source_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = recall_rows(annotations)
    summary = summarise_streamarena(rows)
    for video_id in args.videos:
        raw_dir = args.raw_root / video_id / video_id
        report = materialise_streamarena_video(annotations, raw_dir, video_id, source_root, rows)
        report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (out_dir / f"{video_id}.json").write_text(report_json, encoding="utf-8")
    summary["videos_materialised"] = []
    for report_path in sorted(out_dir.glob("*.json")):
        if report_path.name in {"summary.json", "codec.json"}:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fields = ("video_id", "duration_s", "vtt_cues", "filler_clips", "filler_chunks", "filler_mb_per_hour")
        summary["videos_materialised"].append({key: report[key] for key in fields} | {"items": len(report["items"])})
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _audit_codec(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("SEED_TTS_EVAL_DEVICE", args.asr_device)
    import zhconv

    from benchmarks.av_fd_longmem.asr import FunasrSpeechSegmenter
    from benchmarks.av_fd_longmem.audit import query_codec_check
    from benchmarks.av_fd_longmem.sources.streamarena import exclusion_reasons, recall_rows
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import _transcribe_en_f32_16k

    source_root: Path = args.dataset_root / "streamarena"
    survivors = [r for r in recall_rows(source_root / "annotations") if not exclusion_reasons(r)]
    queries = [("zh", r.question) for r in survivors if r.item_lang == "zh"][: args.per_lang]
    queries += [("en", r.question) for r in survivors if r.item_lang == "en"][: args.per_lang]
    paraformer = FunasrSpeechSegmenter(device=args.asr_device)
    transcribers = {
        "zh": lambda wav: zhconv.convert(paraformer.transcribe_samples(wav), "zh-cn"),
        "en": _transcribe_en_f32_16k,
    }
    out_dir = source_root / "audit"
    report = query_codec_check(queries, tts_url=args.tts_url, work_dir=source_root / "work" / "codec",
                               transcribers=transcribers)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codec.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 0


def _audit_handcheck(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.asr import FunasrSpeechSegmenter
    from benchmarks.av_fd_longmem.audit import build_handcheck

    page = build_handcheck(
        args.dataset_root, args.omniinteract_root, FunasrSpeechSegmenter(device=args.asr_device), n=args.n
    )
    print(page)
    return 0


def _window_plans(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.schema import FillerClip, read_jsonl, write_json
    from benchmarks.av_fd_longmem.window import window_plans

    filler = list(read_jsonl(args.dataset_root / args.source / "filler.jsonl", FillerClip))
    for plan in window_plans(filler):
        path = args.dataset_root / "plans" / "m1" / f"{plan.plan_id}.json"
        write_json(path, plan)
        print(f"{path}: {plan.total_chunks} chunks, {plan.condition}")
    return 0


def _window_report(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.window import summary_markdown, window_report

    run_dir: Path = args.dataset_root / "runs" / args.run_id
    reports = [
        window_report(path.parent, run_dir / "trace") for path in sorted(run_dir.glob("sessions/*/session.json"))
    ]
    (run_dir / "window_report.json").write_text(json.dumps(reports, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = "\n".join(summary_markdown(report) for report in reports)
    (run_dir / "window_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


def _set_models(args: argparse.Namespace):  # -> (build.SetPaths, build.Models)
    from benchmarks.av_fd_longmem.build import Models, SetPaths
    from benchmarks.av_fd_longmem.llm import LLM_MODEL, LLM_REVISION, VLM_MODEL, VLM_REVISION, CachedChat, ChatModel

    paths = SetPaths(out_root=args.dataset_root, source=args.source, set_name=args.set)
    models = Models(
        llm=CachedChat(ChatModel(args.llm_url, LLM_MODEL, LLM_REVISION), paths.cache("llm")),
        vlm=CachedChat(ChatModel(args.vlm_url, VLM_MODEL, VLM_REVISION), paths.cache("vlm")),
        transcribe=_transcriber(args.asr_device),
    )
    return paths, models


def _set_annotate(args: argparse.Namespace) -> int:
    from collections import Counter

    from benchmarks.av_fd_longmem.build import omniinteract_candidates, streamarena_candidates

    paths, models = _set_models(args)
    if args.source == "omniinteract":
        records = omniinteract_candidates(paths, args.omniinteract_root, models, n=args.n)
    else:
        records = streamarena_candidates(paths, models)
    print(json.dumps(Counter(r["reason"] or "candidate" for r in records), ensure_ascii=False))
    for r in records:
        g = r.get("grounding", {})
        print(
            f"{r['item_id']}: {r['status']} {r['reason'] or ''} | {r['query']} -> {r['answer']} | "
            f"vlm={g.get('visual_read', {}).get('answer')!r} visual={g.get('visual')} audio={g.get('audio_stated')}"
        )
    return 0


def _set_items(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.build import build_items, pick

    paths, models = _set_models(args)
    candidates = [json.loads(line) for line in (paths.set_root / "candidates.jsonl").read_text().splitlines()]
    per_lang = {lang: int(count) for lang, count in (pair.split("=") for pair in args.per_lang)} or None
    chosen = pick(candidates, n=args.n, per_lang=per_lang)
    items, filler = build_items(
        paths, chosen, models, data_root=args.omniinteract_root, tts_url=args.tts_url,
        transcribe_file=models.transcribe.transcribe_file,
    )  # fmt: skip
    for item in items:
        print(f"{item.item_id}: {item.query_text} -> {item.answer} ({item.modality_declared}, {item.lang})")
    print(f"filler pool: {len(filler)} clips, {sum(f.chunks for f in filler)} chunks")
    return 0


def _set_plans(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.build import build_plans, load_set

    paths, models = _set_models(args)
    items, filler = load_set(paths)
    items, plans = build_plans(paths, items, filler, models)
    for item in items:
        print(f"{item.item_id}: filler_exclude={item.filler_exclude}")
    for plan in plans:
        print(f"{plan.plan_id}: {plan.kind} {plan.total_chunks} chunks, gaps {[p.gap_chunks for p in plan.probes]}")
    return 0


def _render_preview(args: argparse.Namespace) -> int:
    from benchmarks.av_fd_longmem.render import render_preview
    from benchmarks.av_fd_longmem.schema import SessionPlan, read_json

    plan = read_json(args.plan, SessionPlan)
    out = render_preview(plan, args.dataset_root, args.out)
    print(out)
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
    score.add_argument(
        "--llm-url", help="LLM server for the primary (uncalibrated) answer-match verdict; omit for key-token only."
    )
    score.set_defaults(handler=_score)

    single = stages.add_parser("single-plans", help="Single-probe retention plans for one item at several lags.")
    single.add_argument("--item", required=True, help="item_id from omniinteract/items.jsonl")
    single.add_argument("--lags", type=int, nargs="+", required=True, help="Gaps G in chunks.")
    single.set_defaults(handler=_single_plans)

    audit = stages.add_parser("audit-omniinteract", help="M0 onset audit of an OmniInteract subset.")
    audit.add_argument("--omniinteract-root", type=Path, required=True, help="OmniInteract `data` directory.")
    audit.add_argument("--subset", default="1q1a")
    audit.add_argument("--asr-device", default="cpu")
    audit.set_defaults(handler=_audit_omniinteract)

    sa_audit = stages.add_parser("audit-streamarena", help="M0 StreamArena exclusion counts and clip sample.")
    sa_audit.add_argument("--raw-root", type=Path, help="Directory of extracted tars, <root>/<video_id>/<video_id>/.")
    sa_audit.add_argument("--videos", nargs="*", default=[], help="Fetched video ids to materialise.")
    sa_audit.set_defaults(handler=_audit_streamarena)

    codec = stages.add_parser("audit-codec", help="M0 query ASR error as WAV vs AAC vs FLAC.")
    codec.add_argument("--tts-url", default="http://127.0.0.1:8091")
    codec.add_argument("--asr-device", default="cuda:1")
    codec.add_argument("--per-lang", type=int, default=25, help="StreamArena survivor questions per language.")
    codec.set_defaults(handler=_audit_codec)

    hand = stages.add_parser("audit-handcheck", help="M0 evidence hand-check sample for both sources.")
    hand.add_argument("--omniinteract-root", type=Path, required=True, help="OmniInteract `data` directory.")
    hand.add_argument("--n", type=int, default=30, help="OmniInteract items in the sample.")
    hand.add_argument("--asr-device", default="cpu")
    hand.set_defaults(handler=_audit_handcheck)

    plans = stages.add_parser("window-plans", help="Write the M1 filler-only window plans.")
    plans.add_argument("--source", default="omniinteract", help="Source whose filler.jsonl supplies the filler.")
    plans.set_defaults(handler=_window_plans)

    report = stages.add_parser("window-report", help="M1 window measurements for a finished run.")
    report.add_argument("run_id")
    report.set_defaults(handler=_window_report)

    def set_stage(name: str, help_text: str, handler) -> argparse.ArgumentParser:
        stage = stages.add_parser(name, help=help_text)
        stage.add_argument("--set", default="m3", help="Item set name (<source>/sets/<set>, plans/<set>).")
        stage.add_argument("--source", choices=["omniinteract", "streamarena"], required=True)
        stage.add_argument("--omniinteract-root", type=Path, help="OmniInteract `data` directory.")
        stage.add_argument("--llm-url", default="http://127.0.0.1:8200")
        stage.add_argument("--vlm-url", default="http://127.0.0.1:8201")
        stage.add_argument("--asr-device", default="cpu")
        stage.set_defaults(handler=handler)
        return stage

    annotate = set_stage("set-annotate", "Annotate a source's candidates into an item set.", _set_annotate)
    annotate.add_argument("--n", type=int, default=20, help="OmniInteract candidates to sample.")
    items = set_stage("set-items", "Pick items, speak queries, cut query clips.", _set_items)
    items.add_argument("--n", type=int, default=5)
    items.add_argument("--per-lang", nargs="*", default=[], help="lang=count, e.g. en=2 zh=3 (StreamArena).")
    items.add_argument("--tts-url", default="http://127.0.0.1:8091")
    set_stage("set-plans", "Write leakage-checked plans for an item set.", _set_plans)

    preview = stages.add_parser("render-preview", help="Render one plan to an mp4, exactly as streamed.")
    preview.add_argument("plan", type=Path)
    preview.add_argument("--out", type=Path, required=True)
    preview.set_defaults(handler=_render_preview)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
