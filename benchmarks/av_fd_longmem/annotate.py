# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LLM and VLM annotation of candidate items: query rewrite, tags, grounding and leakage.

Each function sends one versioned prompt through ``llm.CachedChat`` and
returns the parsed reply with its provenance. The text-only LLM rewrites and
tags questions, compares answers and reads evidence
transcripts. The VLM looks at a clip's frames exactly as the stream delivers
them (1 fps, the rendered JPEGs) and says whether the answer can be seen.

Design: plan §1a steps 4–6 and §1b steps 3–4 and 7, §3 confound controls
(``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``).
External dependencies: an LLM and a VLM behind ``llm.CachedChat``; PyAV via ``render``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.av_fd_longmem.llm import CachedChat, Prompt, jpeg_part, text_part
from benchmarks.av_fd_longmem.render import segment_units
from benchmarks.av_fd_longmem.schema import Segment

MODALITIES = ["audio", "visual", "av"]


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_BOOL = {"type": "boolean"}


def _text(max_chars: int) -> dict[str, Any]:
    """A string bounded in length, so a greedy repetition loop cannot run to ``max_tokens``."""
    return {"type": "string", "maxLength": max_chars}


def _list(item: dict[str, Any], max_items: int) -> dict[str, Any]:
    return {"type": "array", "items": item, "maxItems": max_items}


_MODALITY = {"type": "string", "enum": MODALITIES}
_VERDICT = {"type": "string", "enum": ["yes", "no", "unclear"]}

REWRITE_OMNIINTERACT = Prompt(
    prompt_id="rewrite_omniinteract",
    version=2,
    system=(
        "你在为一个视频记忆基准改写问题。原问题是用户在第一视角视频里实时提出的（例如“这个多少钱”）。"
        "在基准中，这个问题会在相关画面过去很久之后才被提出，可能是几秒、几分钟或一小时。\n"
        "请把问题改写成：\n"
        "1. 指向过去、但不限定时间远近：用“之前看到的…”“之前…”开头，"
        "禁止使用“刚才”“刚刚”“现在”“这个”等只在短时间内成立的词；\n"
        "2. 独立可懂：用原问题里的信息说清楚指的是什么物体或场景（如“那副墨镜”“地铁上那个位置”），"
        "不要编造原问题没有的细节；\n"
        "3. 不泄露答案，问的内容与原问题相同，一句话，口语化。\n"
        "同时给出精简答案：去掉答案里的寒暄和复述（如“好的”“我看到”），只保留回答本身，保留数字和单位。\n"
        "modality：回答需要看画面填 visual，需要听声音填 audio，两者都需要填 av。\n"
        "standalone：改写后的问题是否能在不看原上下文时明确指代对象。"
    ),
    schema=_object({"query": _text(200), "answer": _text(120), "modality": _MODALITY, "standalone": _BOOL}),
)

#: The reference answer is set by its own prompt, not by the rewrite, so changing it leaves the queries untouched.
SHORT_ANSWER = Prompt(
    prompt_id="short_answer",
    version=3,
    system=(
        "你在为一个视频记忆基准整理参考答案。给你一个问题和它的原答案。\n"
        "answer：能回答问题的最短说法。\n"
        "- 去掉寒暄、对问题的复述（如“好的”“我看到”“这上面写的是”“主色调是”）。\n"
        "- 只保留问题问的那一项；原答案在主答案之外补充的次要细节要去掉"
        "（“主色调是粉色，部分偏浅粉色”→“粉色”）。问题本身要求列出多项时才保留多项，用“，”分隔。\n"
        "- 是非题只答“是”或“不是”。\n"
        "- 保留数字和单位；不要加入原答案没有的内容。\n"
        "aliases：同一个答案的其他写法，只限繁简体、中英文、大小写、全称与简称"
        "（“UGG”→“ugg”，“粉色”→“pink”，“上海三菱电梯”→“三菱电梯”），最多4个。"
        "不要给同义词或近似的说法（“酒红色”不能写成“深红色”，“浅粉色”不能写成“粉色”）。没有就给空列表。"
    ),
    schema=_object({"answer": _text(40), "aliases": _list(_text(40), 4)}),
    max_tokens=128,
)

ANNOTATE_STREAMARENA = Prompt(
    prompt_id="annotate_streamarena",
    version=4,
    system=(
        "You tag recall questions for a long-video memory benchmark. Each question is asked aloud long after "
        "the evidence was shown: the delay can be seconds, minutes or an hour, and other video plays in between. "
        "Recency words in the question ('刚才', 'just now') are replaced with neutral past wording ('之前', "
        "'earlier') before it is asked, so ignore them.\n"
        "Judge the QUESTION TEXT only. The evidence hint is the annotators' private note (it may contain "
        "timestamps); use it only to understand what the evidence shows, never as part of the question.\n"
        "- relative_time: true ONLY if the question text itself names a position or duration in the video that a "
        "long inserted delay would make false ('in the last two minutes', '最后一段', 'at the very beginning of "
        "the video'). Words such as '之前', '前面', 'earlier', 'previous' do NOT count.\n"
        "- non_standalone: true if a listener who watched the whole video could still not tell what the question "
        "refers to, e.g. it points at something on screen at the moment it is asked ('this', 'him here'), or "
        "depends on an earlier question. A question that names its referent ('猪爷爷的花园', 'the teacher') is "
        "standalone.\n"
        "- present_referent: true if it refers to something expected to still be on screen when asked "
        "('this person', '这个人', 'the man on the left now').\n"
        "- modality: what perceiving the evidence needs: 'audio' (speech or sound), 'visual' (on-screen only), "
        "'av' (both)."
    ),
    schema=_object({"relative_time": _BOOL, "non_standalone": _BOOL, "present_referent": _BOOL, "modality": _MODALITY}),
)

ANSWER_MATCH = Prompt(
    prompt_id="answer_match",
    version=2,
    system=(
        "Decide whether a candidate answer gives the same answer as the reference answer to the question. Numbers "
        "must agree (units may be omitted or written differently). Names match if they refer to the same thing in "
        "any language or script. A candidate that lists several options matches only if it commits to the right one."
    ),
    schema=_object({"match": _BOOL}),
    max_tokens=64,
)

GROUND_AUDIO = Prompt(
    prompt_id="ground_audio",
    version=3,
    system=(
        "You get the automatic speech transcript of a short video clip, a question about the clip and the reference "
        "answer. Decide whether a listener could get the reference answer from the speech alone. ASR errors are "
        "expected: accept homophones and near-miss spellings of the answer. quote: the few transcript words (at most "
        "20) that state the answer, or an empty string."
    ),
    schema=_object({"quote": _text(80), "stated": _BOOL}),
    max_tokens=1024,
)

GROUND_VISUAL = Prompt(
    prompt_id="ground_visual",
    version=2,
    system=(
        "You see consecutive frames of a video clip, one per second, labelled by time. A question about the video and "
        "its reference answer follow. Decide whether a viewer of these frames alone (no sound) could read or see the "
        "reference answer. 'yes' only if the answer itself is visible and legible (text, number, object, colour, "
        "action). 'unclear' if it might be there but is too small, blurred or ambiguous. First describe what the "
        "relevant frames show, then give the verdict and the best frame's time in seconds (-1 if none)."
    ),
    schema=_object({"observation": _text(400), "visible": _VERDICT, "frame_s": {"type": "integer"}}),
    max_tokens=384,
)

VISUAL_READ = Prompt(
    prompt_id="visual_read",
    version=2,
    system=(
        "You see consecutive frames of a video clip, one per second, labelled by time, with no sound. Answer the "
        "question about the clip from these frames alone. Read text and numbers exactly as shown. If the frames do "
        "not show enough to answer, set answerable to false and give your best guess anyway. First describe what "
        "the relevant frames show, then answer briefly in the question's language."
    ),
    schema=_object({"observation": _text(400), "answer": _text(200), "answerable": _BOOL}),
    max_tokens=384,
)

LEAKAGE = Prompt(
    prompt_id="leakage",
    version=2,
    system=(
        "You see consecutive frames of an unrelated filler video, one per second, labelled by time. A question and "
        "its reference answer about a DIFFERENT video follow. Decide whether these frames would let a viewer answer "
        "the question: the same object, text or number visible together with what the question asks about. "
        "Coincidental similar objects without the answer do not count. Describe briefly, then decide."
    ),
    schema=_object({"observation": _text(400), "shows_answer": _BOOL, "frame_s": {"type": "integer"}}),
    max_tokens=256,
)


@dataclass(frozen=True)
class Annotation:
    output: dict[str, Any]
    provenance: dict[str, Any]


def _annotation(chat: CachedChat, prompt: Prompt, content: str | Sequence[dict[str, Any]]) -> Annotation:
    result = chat.ask(prompt, content)
    return Annotation(result.output, result.provenance())


def rewrite_omniinteract(llm: CachedChat, question: str, answer: str) -> Annotation:
    return _annotation(llm, REWRITE_OMNIINTERACT, f"原问题：{question}\n原答案：{answer}")


def short_answer(llm: CachedChat, question: str, answer: str) -> Annotation:
    return _annotation(llm, SHORT_ANSWER, f"问题：{question}\n原答案：{answer}")


def reference_answer(answer: str, aliases: Sequence[str]) -> str:
    """The answer with each distinct alias as a parenthetical, the alias form ``score.answer_keys`` reads.

    Aliases that contain brackets themselves are dropped, since they would split into wrong keys.
    """
    extra = [a.strip() for a in aliases if a.strip() and a.strip() != answer.strip() and not set(a) & set("（）()")]
    return answer.strip() + "".join(f"（{alias}）" for alias in dict.fromkeys(extra))


def annotate_streamarena(llm: CachedChat, question: str, answer: str, hints: Sequence[str]) -> Annotation:
    hint = " / ".join(hints) or "(none)"
    return _annotation(llm, ANNOTATE_STREAMARENA, f"Question: {question}\nAnswer: {answer}\nEvidence hint: {hint}")


def answers_match(llm: CachedChat, question: str, reference: str, candidate: str) -> Annotation:
    return _annotation(
        llm, ANSWER_MATCH, f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"
    )


def ground_audio(llm: CachedChat, transcript: str, question: str, answer: str) -> Annotation:
    return _annotation(
        llm,
        GROUND_AUDIO,
        f"Transcript: {transcript or '(no speech)'}\nQuestion: {question}\nReference answer: {answer}",
    )


def clip_frames(clip: Path, chunks: int, *, start: int = 0) -> list[bytes]:
    """The stream JPEGs of clip seconds ``[start, chunks)``, as ``render`` delivers them to the model."""
    segment = Segment(role="filler", clip=clip.name, play_chunks=chunks, stream_chunk0=0)
    frames = [jpeg for _pcm, jpeg in segment_units(segment, clip.parent)]
    return frames[start:]


def _frames_content(frames: Sequence[bytes], first_s: int, text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for k, jpeg in enumerate(frames):
        content += [text_part(f"t={first_s + k}s"), jpeg_part(jpeg)]
    return [*content, text_part(text)]


def ground_visual(vlm: CachedChat, frames: Sequence[bytes], question: str, answer: str) -> Annotation:
    return _annotation(
        vlm, GROUND_VISUAL, _frames_content(frames, 0, f"Question: {question}\nReference answer: {answer}")
    )


def visual_read(vlm: CachedChat, frames: Sequence[bytes], question: str) -> Annotation:
    """The VLM's own answer from the frames, without being shown the reference answer."""
    return _annotation(vlm, VISUAL_READ, _frames_content(frames, 0, f"Question: {question}"))


def leakage(vlm: CachedChat, frames: Sequence[bytes], first_s: int, question: str, answer: str) -> Annotation:
    return _annotation(
        vlm, LEAKAGE, _frames_content(frames, first_s, f"Question: {question}\nReference answer: {answer}")
    )
