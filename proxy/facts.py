"""Fact extraction: ask the upstream LLM to turn a user message into a short
list of search queries, and parse the answer defensively.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from .config import Config

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[[^\]]*\]", re.DOTALL)


def _clean(data: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(data, list):
        return out
    for item in data:
        if isinstance(item, str):
            s = item.strip()
        elif isinstance(item, dict):
            s = str(item.get("query") or item.get("text") or item.get("fact") or "").strip()
        else:
            s = str(item).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def extract_json_array(text: str | None) -> list[str]:
    """Defensively parse a JSON array of strings out of a model response."""
    if not text:
        return []
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _clean(data)
    except Exception:
        pass
    m = _ARRAY_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return _clean(data)
        except Exception:
            pass
    return []


async def extract_facts(
    client: httpx.AsyncClient,
    cfg: Config,
    base_url: str,
    api_key: str,
    user_text: str,
) -> list[str]:
    """Return up to ``max_facts`` short search queries for ``user_text``.

    Any failure results in an empty list (the caller then forwards unchanged).
    """
    fe = cfg.fact_extraction
    if not fe.enabled or not user_text:
        return []
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": fe.model,
        "messages": [
            {"role": "system", "content": fe.prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": fe.temperature,
        "max_tokens": fe.max_tokens,
    }
    resp = await client.post(
        base_url.rstrip("/") + "/chat/completions", json=payload, headers=headers
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    facts = extract_json_array(content)
    return facts[: fe.max_facts]
