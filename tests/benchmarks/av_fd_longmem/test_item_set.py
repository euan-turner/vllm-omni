# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Item-set building: control variants, lag-neutral rewrites, reply checking, caching, picking and filler pools.

External dependencies: none (the chat server is faked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.av_fd_longmem import llm
from benchmarks.av_fd_longmem.annotate import reference_answer
from benchmarks.av_fd_longmem.build import RETENTION_LAGS, SetPaths, item_plans, item_pool, pick
from benchmarks.av_fd_longmem.evaluate import _items
from benchmarks.av_fd_longmem.planner import PlannedProbe, ablation_variant, blind_variant, plan_chained
from benchmarks.av_fd_longmem.schema import ClipRef, FillerClip, Item, ItemClips, QueryClip, write_jsonl
from benchmarks.av_fd_longmem.score import answer_keys, carries_key
from benchmarks.av_fd_longmem.sources.streamarena import lag_neutral

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _item(
    name: str = "x", *, source: str = "omniinteract", lang: str = "zh", stratum: str = "zh", domain: str | None = None
) -> Item:
    return Item(
        item_id=name,
        source=source,
        lang=lang,
        clips=ItemClips(
            evidence=[ClipRef(path=f"evidence/{name}.mp4", chunks=15)],
            query=QueryClip(
                path=f"query/{name}.mp4", chunks=35, query_onset_chunk=0, query_end_ms=4200, answer_chunks=30
            ),
        ),
        query_text="q",
        query_text_original="q",
        answer="a",
        modality_declared="visual",
        provenance={"stratum": stratum},
        domain=domain,
    )


def _filler(
    n: int = 20, *, chunks: int = 97, domain: str | None = None, lang: str = "zh", stratum: str = "zh"
) -> list[FillerClip]:
    return [
        FillerClip(
            filler_id=f"{domain}-f{i}",
            path=f"filler/{domain}-f{i}.mp4",
            chunks=chunks,
            source="test",
            domain=domain,
            provenance={"lang": lang, "stratum": stratum},
        )
        for i in range(n)
    ]


def _control():
    return plan_chained("c", [PlannedProbe(_item(), 0)], _filler(), condition={"window_mode": "context"})


def test_blind_variant_keeps_coordinates_and_plays_unused_filler_in_place_of_evidence() -> None:
    control = _control()
    blind = blind_variant(control, _filler(), plan_id="b")

    assert blind.kind == "control_blind"
    assert blind.probes == control.probes
    assert blind.total_chunks == control.total_chunks
    assert all(segment.role != "evidence" for segment in blind.segments)
    control_clips = {segment.clip for segment in control.segments}
    evidence = next(s for s in control.segments if s.role == "evidence")
    replacing = [s for s in blind.segments if evidence.stream_chunk0 <= s.stream_chunk0 < evidence.stream_chunk0 + 15]
    assert sum(s.play_chunks for s in replacing) == 15
    assert not {s.clip for s in replacing} & control_clips


@pytest.mark.parametrize(("mode", "field", "value"), [("audio", "audio", "mute"), ("video", "video", "freeze")])
def test_ablation_variant_changes_only_the_evidence_segments(mode: str, field: str, value: str) -> None:
    control = _control()
    ablated = ablation_variant(control, mode=mode, plan_id="a")

    assert ablated.kind == f"ablate_{mode}"
    for before, after in zip(control.segments, ablated.segments, strict=True):
        if before.role == "evidence":
            assert getattr(after, field) == value
        else:
            assert after == before


def test_ablation_variant_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError):
        ablation_variant(_control(), mode="both", plan_id="a")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("刚才佩奇妈妈给佩奇带的靴子是什么颜色", "之前佩奇妈妈给佩奇带的靴子是什么颜色"),
        ("猪爷爷刚刚的花园里有几个苹果", "猪爷爷之前的花园里有几个苹果"),
        ("What was shown just now?", "What was shown earlier?"),
        ("The press conference just mentioned, how much?", "The press conference mentioned earlier, how much?"),
        ("前面视频里提到的乐器是什么？", "前面视频里提到的乐器是什么？"),
    ],
)
def test_lag_neutral_replaces_only_recency_words(question: str, expected: str) -> None:
    assert lag_neutral(question) == expected


SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 20},
        "ok": {"type": "boolean"},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "frame_s": {"type": "integer"},
    },
}


def test_reply_format_names_every_key_in_order() -> None:
    text = llm.reply_format(SCHEMA)

    assert text.index('"answer"') < text.index('"ok"') < text.index('"kind"') < text.index('"frame_s"')
    assert '"a" | "b"' in text and "at most 20 characters" in text


def test_parse_reply_takes_the_first_object_and_drops_extra_keys() -> None:
    raw = 'Sure: {"answer": "x", "ok": true, "kind": "b", "frame_s": 3, "extra": 1} trailing'

    assert llm.parse_reply(raw, SCHEMA) == {"answer": "x", "ok": True, "kind": "b", "frame_s": 3}


@pytest.mark.parametrize(
    "raw",
    [
        "no json here",
        '{"answer": "x", "ok": true, "kind": "b"}',
        '{"answer": "x", "ok": "yes", "kind": "b", "frame_s": 3}',
        '{"answer": "x", "ok": true, "kind": "c", "frame_s": 3}',
        '{"answer": "x", "ok": true, "kind": "a", "frame_s": true}',
    ],
)
def test_parse_reply_rejects_a_reply_that_breaks_the_schema(raw: str) -> None:
    with pytest.raises(ValueError):
        llm.parse_reply(raw, SCHEMA)


LIST_SCHEMA = {
    "type": "object",
    "properties": {"aliases": {"type": "array", "items": {"type": "string", "maxLength": 10}, "maxItems": 4}},
}


def test_a_list_of_strings_is_described_and_checked() -> None:
    assert "list of string (at most 10 characters) (at most 4 items)" in llm.reply_format(LIST_SCHEMA)
    assert llm.parse_reply('{"aliases": ["a", "b"]}', LIST_SCHEMA) == {"aliases": ["a", "b"]}
    with pytest.raises(ValueError):
        llm.parse_reply('{"aliases": "a"}', LIST_SCHEMA)
    with pytest.raises(ValueError):
        llm.parse_reply('{"aliases": ["a", 1]}', LIST_SCHEMA)


def test_reference_answer_writes_aliases_as_parentheticals_the_scorer_accepts() -> None:
    answer = reference_answer("粉色", ["粉红色", "pink", "粉色", " ", "红(深)"])

    assert answer == "粉色（粉红色）（pink）"
    assert answer_keys(answer) == ["粉色", "粉红色", "pink"]
    assert carries_key("那些花是粉红色的", answer_keys(answer))
    assert reference_answer("ugg", []) == "ugg"


class _FakeServer:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.calls: list[dict] = []
        self._content = content
        self._finish_reason = finish_reason

    def post(self, url: str, json: dict, timeout: float):  # noqa: A002 - mirrors httpx.post
        self.calls.append(json)
        content, finish = self._content, self._finish_reason

        class _Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

        return _Response()


PROMPT = llm.Prompt(prompt_id="p", version=1, system="s", schema=SCHEMA)
REPLY = '{"answer": "x", "ok": true, "kind": "a", "frame_s": 0}'


def test_cached_chat_answers_a_repeated_call_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _FakeServer(REPLY)
    monkeypatch.setattr(llm.httpx, "post", server.post)
    chat = llm.CachedChat(llm.ChatModel("http://x", "m", "rev"), tmp_path)

    first = chat.ask(PROMPT, [llm.text_part("q"), llm.jpeg_part(b"\xff\xd8jpeg")])
    second = chat.ask(PROMPT, [llm.text_part("q"), llm.jpeg_part(b"\xff\xd8jpeg")])

    assert len(server.calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert first.output == second.output == llm.parse_reply(REPLY, SCHEMA)
    assert server.calls[0]["temperature"] == 0.0
    record = json.loads((tmp_path / f"{first.key}.json").read_text())
    assert "base64" not in json.dumps(record["request"])


def test_cached_chat_key_changes_with_revision_prompt_version_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm.httpx, "post", _FakeServer(REPLY).post)
    base = llm.CachedChat(llm.ChatModel("http://x", "m", "rev"), tmp_path)
    other_rev = llm.CachedChat(llm.ChatModel("http://x", "m", "rev2"), tmp_path)
    bumped = llm.Prompt(prompt_id="p", version=2, system="s", schema=SCHEMA)

    keys = {
        base.ask(PROMPT, [llm.jpeg_part(b"a")]).key,
        base.ask(PROMPT, [llm.jpeg_part(b"b")]).key,
        other_rev.ask(PROMPT, [llm.jpeg_part(b"a")]).key,
        base.ask(bumped, [llm.jpeg_part(b"a")]).key,
    }

    assert len(keys) == 4


def test_cached_chat_rejects_a_reply_cut_at_max_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.httpx, "post", _FakeServer('{"answer": "xxxx', finish_reason="length").post)
    chat = llm.CachedChat(llm.ChatModel("http://x", "m", "rev"), tmp_path)

    with pytest.raises(ValueError, match="max_tokens"):
        chat.ask(PROMPT, "q")
    assert not list(tmp_path.iterdir())


def _candidate(item_id: str, stratum: str, modality: str, *, status: str = "candidate") -> dict:
    return {"item_id": item_id, "stratum": stratum, "modality_declared": modality, "status": status}


def test_pick_skips_excluded_candidates_and_keeps_candidate_order() -> None:
    candidates = [
        _candidate("a", "zh", "visual"),
        _candidate("b", "zh", "visual", status="excluded"),
        _candidate("c", "zh", "visual"),
        _candidate("d", "zh", "visual"),
    ]

    assert [c["item_id"] for c in pick(candidates, n=3)] == ["a", "c", "d"]


def test_pick_per_stratum_keeps_mixed_language_videos_apart_and_spreads_modalities() -> None:
    candidates = [
        _candidate("e1", "en", "audio"),
        _candidate("e2", "en", "audio"),
        _candidate("e3", "en", "visual"),
        _candidate("m1", "zh+en", "visual"),
        _candidate("m2", "zh+en", "audio"),
        _candidate("z1", "zh", "av"),
        _candidate("z2", "zh", "av"),
        _candidate("z3", "zh", "audio"),
        _candidate("z4", "zh", "visual"),
    ]

    picked = [c["item_id"] for c in pick(candidates, n=7, per_stratum={"en": 2, "zh+en": 1, "zh": 3})]

    assert picked == ["e1", "e3", "m1", "z1", "z3", "z4"]


def test_streamarena_pool_is_same_stratum_other_domain_and_honours_exclusions() -> None:
    filler = _filler(4, domain="anime") + _filler(4, domain="shop") + _filler(4, domain="talk", lang="en", stratum="en")
    item = _item(source="streamarena", lang="zh", domain="anime")
    item = Item(**{**item.__dict__, "filler_exclude": ["shop-f1"]})

    pool = item_pool(item, filler, exclude={"shop-f2"})

    assert sorted(f.filler_id for f in pool) == ["shop-f0", "shop-f3"]


def test_english_items_never_get_filler_from_mixed_language_videos() -> None:
    filler = _filler(4, domain="film", lang="en", stratum="zh+en") + _filler(4, domain="talk", lang="en", stratum="en")
    item = _item(source="streamarena", lang="en", stratum="en", domain="tutorial")

    assert sorted(f.filler_id for f in item_pool(item, filler, exclude=set())) == [f"talk-f{i}" for i in range(4)]


def test_item_pool_order_depends_on_the_item_only() -> None:
    filler = _filler(30)

    assert item_pool(_item("a"), filler, exclude=set()) == item_pool(_item("a"), filler, exclude=set())
    assert item_pool(_item("a"), filler, exclude=set()) != item_pool(_item("b"), filler, exclude=set())


def test_item_plans_share_one_filler_order_across_lags() -> None:
    paths = SetPaths(out_root=Path("/data"), source="streamarena", set_name="m3")
    item = _item("x", source="streamarena")
    plans = item_plans(paths, item, _filler(30, chunks=60))

    kinds = [plan.kind for plan in plans]
    assert kinds == ["control_g0", "control_blind", "ablate_audio", "ablate_video", "single", "single"]
    assert [plan.probes[0].gap_chunks for plan in plans[4:]] == list(RETENTION_LAGS["streamarena"])
    fillers = [[s.clip for s in plan.segments if s.role == "filler"] for plan in (plans[0], *plans[4:])]
    for shorter, longer in zip(fillers, fillers[1:]):
        assert longer[: len(shorter) - 1] == shorter[:-1]
    assert len({plan.plan_id for plan in plans}) == len(plans)


def test_items_reads_item_sets_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "omniinteract" / "items.jsonl", [_item("pilot")])
    write_jsonl(tmp_path / "omniinteract" / "sets" / "m3" / "items.jsonl", [_item("m3")])

    assert set(_items(tmp_path)) == {"pilot", "m3"}

    write_jsonl(tmp_path / "streamarena" / "sets" / "m3" / "items.jsonl", [_item("pilot")])
    with pytest.raises(ValueError, match="defined twice"):
        _items(tmp_path)
