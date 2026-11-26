# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cached, deterministic JSON chat calls to an OpenAI-compatible server (the LLM and VLM of the item pipeline).

Every call runs at temperature 0 and asks for a JSON object; the reply is
parsed and checked against the prompt's schema on the client (required keys,
types, enums), and a reply that fails the check raises. Server-side guided
decoding is not used: with xgrammar, compact JSON derailed both models and
whitespace-tolerant JSON let them loop on whitespace (2026-11-25). Replies are
cached on disk under ``<cache_dir>/<sha256>.json``. The key covers the model,
its revision, the prompt id and version, the messages (images included, as
data URLs) and the decoding parameters, so a changed prompt or model misses
the cache instead of returning a stale answer. Each cached record keeps the
key's inputs (images replaced by their sha256) next to the parsed reply.

Design: plan §1 ("Every LLM or VLM output is cached with model, revision and
prompt hash"), ``~/.claude/plans/we-previously-worked-on-wiggly-octopus-…md``.
External dependencies: httpx; a ``vllm serve`` (or other OpenAI-compatible) server.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

LLM_MODEL = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
LLM_REVISION = "8d3f99eda08b18a8090de5289ff02e6dc51d47e2"
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


@dataclass(frozen=True)
class ChatModel:
    """A served model: ``model`` is the name the server was started with, ``revision`` its pinned snapshot."""

    base_url: str
    model: str
    revision: str


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt: bump ``version`` whenever the wording or schema changes."""

    prompt_id: str
    version: int
    system: str
    schema: dict[str, Any]
    max_tokens: int = 512


@dataclass(frozen=True)
class ChatResult:
    output: dict[str, Any]
    key: str
    model: str
    revision: str
    prompt_id: str
    prompt_version: int
    cached: bool

    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "prompt": f"{self.prompt_id}@v{self.prompt_version}",
            "key": self.key,
        }


def jpeg_part(jpeg: bytes) -> dict[str, Any]:
    """An OpenAI ``image_url`` content part holding ``jpeg`` as a data URL."""
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _redact_images(content: Any) -> Any:
    if isinstance(content, list):
        return [_redact_images(part) for part in content]
    if isinstance(content, dict) and content.get("type") == "image_url":
        url = content["image_url"]["url"]
        return {"type": "image_url", "sha256": hashlib.sha256(url.encode()).hexdigest()}
    return content


def _describe(spec: dict[str, Any]) -> str:
    if "enum" in spec:
        return " | ".join(json.dumps(value) for value in spec["enum"])
    kind = {"string": "string", "boolean": "true | false", "integer": "integer"}[spec["type"]]
    return f"{kind} (at most {spec['maxLength']} characters)" if "maxLength" in spec else kind


def reply_format(schema: dict[str, Any]) -> str:
    """The reply instruction appended to every system prompt: one JSON object with the schema's keys, in order."""
    keys = ", ".join(f'"{name}": {_describe(spec)}' for name, spec in schema["properties"].items())
    return f"Reply with exactly one JSON object and nothing else, with these keys in this order: {{{keys}}}."


def parse_reply(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    """The first JSON object in ``raw``, checked against ``schema``'s keys, types and enums; raises if it fails."""
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
    output, _end = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(output, dict):
        raise ValueError(f"reply is not a JSON object: {raw[:200]!r}")
    types = {"string": str, "boolean": bool, "integer": int}
    for name, spec in schema["properties"].items():
        if name not in output:
            raise ValueError(f"reply lacks {name!r}: {raw[:200]!r}")
        value = output[name]
        expected = types[spec["type"]]
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ValueError(f"reply {name!r} is not {spec['type']}: {raw[:200]!r}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"reply {name!r}={value!r} not in {spec['enum']}")
    return {name: output[name] for name in schema["properties"]}


class CachedChat:
    """``ask(prompt, user_content)`` returns the parsed JSON reply, from the cache if the same call was made before."""

    def __init__(self, model: ChatModel, cache_dir: Path, *, timeout_s: float = 600.0) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._timeout_s = timeout_s

    def _request(self, prompt: Prompt, user_content: str | Sequence[dict[str, Any]]) -> dict[str, Any]:
        content = user_content if isinstance(user_content, str) else list(user_content)
        return {
            "model": self._model.model,
            "messages": [
                {"role": "system", "content": prompt.system + "\n\n" + reply_format(prompt.schema)},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": prompt.max_tokens,
        }

    def ask(self, prompt: Prompt, user_content: str | Sequence[dict[str, Any]]) -> ChatResult:
        request = self._request(prompt, user_content)
        identity = {
            "revision": self._model.revision,
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.version,
            "request": request,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            return self._result(record["output"], key, prompt, cached=True)
        response = httpx.post(
            f"{self._model.base_url.rstrip('/')}/v1/chat/completions", json=request, timeout=self._timeout_s
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        raw = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise ValueError(f"{prompt.prompt_id}: reply cut at max_tokens={prompt.max_tokens}: {raw[-200:]!r}")
        output = parse_reply(raw, prompt.schema)
        record = {
            **identity,
            "request": {
                **request,
                "messages": [{**m, "content": _redact_images(m["content"])} for m in request["messages"]],
            },
            "raw": raw,
            "output": output,
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return self._result(output, key, prompt, cached=False)

    def _result(self, output: dict[str, Any], key: str, prompt: Prompt, *, cached: bool) -> ChatResult:
        return ChatResult(
            output=output,
            key=key,
            model=self._model.model,
            revision=self._model.revision,
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.version,
            cached=cached,
        )
