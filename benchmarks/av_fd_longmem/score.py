# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Key-token scoring of probe answers (pilot judge stand-in).

A probe's reply is every output text chunk that starts inside its answer
window (query onset to the end of the silent answer tail). Pilot answers are
crisp, so a reply is correct when it carries one of the answer's keys. A
numeric key is the answer's digit groups (``4号车13号门`` -> ``4 13``), matched
as consecutive equal groups in the reply or as one reply group equal to all of
them run together. Chinese numerals in a reply (ASR writes ``一百六十八``,
``两``, ``四零零``) are read as digits first. Any other key is the answer text
itself (Simplified Chinese, punctuation ignored). A parenthetical in the answer is an
accepted alternative, e.g. ``麗枫酒店（LAVANDE）`` accepts either name. The
same check over output before the query onset flags rehearsal. This is not the plan's LLM judge; it
holds only for short factual answers and every verdict carries the matched
text for review.

External dependencies: zhconv.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import zhconv

from benchmarks.av_fd_longmem.schema import Item, Probe

_PARENTHETICAL = re.compile(r"[（(]([^）)]*)[）)]")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_DIGIT_GROUPS = re.compile(r"\d+(?:\.\d+)?")  # a decimal is one group: "2.99" never matches "2"
_CN_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CN_NUMBER = re.compile(f"[{''.join(_CN_DIGITS)}{''.join(_CN_UNITS)}]+")


def _cn_number(run: str) -> str:
    """A run of Chinese numerals as digits: positional if it has 十/百/千/万, else digit by digit."""
    if not any(char in _CN_UNITS for char in run):
        return "".join(str(_CN_DIGITS[char]) for char in run)
    total, section, digit = 0, 0, 0
    for char in run:
        if char in _CN_DIGITS:
            digit = _CN_DIGITS[char]
        elif char == "万":
            total += (section + digit) * 10000
            section, digit = 0, 0
        else:
            section += (digit or 1) * _CN_UNITS[char]
            digit = 0
    return str(total + section + digit)


_CN_DECIMAL_POINT = re.compile(r"(?<=\d)点(?=\d)")
#: Spoken money: 2块9毛9 / 2元9角9分 -> 2.99. The tenths need 毛/角: in 两块九一颗 the 一 may mean "each".
_SPOKEN_MONEY = re.compile(r"(\d+)(?:块|元)(\d)(?:毛|角)(?:(\d)分?)?")


def chinese_numerals_to_digits(text: str) -> str:
    """Chinese numerals as digits; 点 between digits and spoken money amounts become decimals.

    两点九九 -> 2.99; 两块九毛九 -> 2.99 (not 两块九, which stays ambiguous). A decimal is one
    digit group, so it never matches a whole-number key.
    """
    digits = _CN_DECIMAL_POINT.sub(".", _CN_NUMBER.sub(lambda match: _cn_number(match.group(0)), text))
    return _SPOKEN_MONEY.sub(lambda m: f"{m.group(1)}.{m.group(2)}{m.group(3) or ''}", digits)


@dataclass(frozen=True)
class TimedText:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class Verdict:
    item_id: str
    channel: str
    verdict: str  # correct | incorrect | silent
    reply: str
    keys: list[str]
    rehearsed: bool
    first_reply_s: float | None


def normalise(text: str) -> str:
    return _NON_WORD.sub("", zhconv.convert(text, "zh-cn")).lower()


def _key(text: str) -> str:
    groups = _DIGIT_GROUPS.findall(text)
    return " ".join(groups) if groups else normalise(text)


def answer_keys(answer: str) -> list[str]:
    """Keys of the answer and of each parenthetical alternative: digits if numeric, else normalised text."""
    keys = [_key(_PARENTHETICAL.sub("", answer))] + [_key(alias) for alias in _PARENTHETICAL.findall(answer)]
    return [key for key in dict.fromkeys(keys) if key]


def _carries_digits(groups: list[str], key: list[str]) -> bool:
    joined = "".join(key)
    if joined in groups:
        return True
    return any(groups[i : i + len(key)] == key for i in range(len(groups) - len(key) + 1))


def carries_key(text: str, keys: Sequence[str]) -> bool:
    groups = _DIGIT_GROUPS.findall(chinese_numerals_to_digits(text))
    words = normalise(text)
    for key in keys:
        parts = key.split(" ")
        if all(_DIGIT_GROUPS.fullmatch(part) for part in parts):
            if _carries_digits(groups, parts):
                return True
        elif key in words:
            return True
    return False


def judge_probe(item: Item, probe: Probe, chunks: Sequence[TimedText], *, channel: str) -> Verdict:
    window_start = probe.query_onset_chunk
    window_end = probe.answer_window_chunks[1] + 1
    reply_chunks = [chunk for chunk in chunks if window_start <= chunk.start_s < window_end]
    before = " ".join(chunk.text for chunk in chunks if chunk.start_s < window_start)
    reply = " ".join(chunk.text for chunk in reply_chunks)
    keys = answer_keys(item.answer)
    verdict = "silent" if not reply.strip() else "correct" if carries_key(reply, keys) else "incorrect"
    return Verdict(
        item_id=item.item_id,
        channel=channel,
        verdict=verdict,
        reply=reply,
        keys=keys,
        rehearsed=carries_key(before, keys),
        first_reply_s=reply_chunks[0].start_s if reply_chunks else None,
    )


def transcript_chunks(wav_transcript: dict[str, object]) -> list[TimedText]:
    """Output text chunks of an OmniInteract ``wav_transcript.json``, on the session's stream clock."""
    chunks = []
    for chunk in wav_transcript.get("chunks", []):
        start, end = chunk["timestamp"]
        chunks.append(TimedText(float(start), float(end), str(chunk["text"])))
    return chunks


def verdict_rows(verdicts: Sequence[Verdict]) -> list[dict[str, object]]:
    return [asdict(verdict) for verdict in verdicts]
