"""Prompt augmentation: build the grounding instruction and splice it into the
request messages (as a system message, or prepended to the first user message).
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .kiwix import Article


def build_help_text(cfg: Config, facts: list[str], articles: list[Article]) -> str:
    facts_str = ", ".join(facts) if facts else "(none)"
    if articles:
        parts = [f"### {a.title}\n{a.text}" for a in articles]
        articles_str = "\n\n".join(parts)
    else:
        articles_str = "(no relevant articles found)"
    return cfg.help_prompt.template.replace("{facts}", facts_str).replace(
        "{articles}", articles_str
    )


def augment_messages(
    cfg: Config,
    messages: list[dict[str, Any]],
    facts: list[str],
    articles: list[Article],
) -> list[dict[str, Any]]:
    help_text = build_help_text(cfg, facts, articles)
    position = (cfg.help_prompt.position or "system").lower()

    if position == "user_prefix":
        new = list(messages)
        for i, m in enumerate(new):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    new[i] = {**m, "content": help_text + "\n\n" + content}
                elif isinstance(content, list):
                    new[i] = {
                        **m,
                        "content": [{"type": "text", "text": help_text}] + list(content),
                    }
                else:
                    new[i] = {**m, "content": help_text}
                break
        else:
            # No user message found; fall back to a system message.
            return [{"role": "system", "content": help_text}] + new
        return new

    # Default: insert as an (additional) system message at the front.
    return [{"role": "system", "content": help_text}] + list(messages)
