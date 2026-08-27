"""Query-relevance-aware article text selection.

A plain prefix truncation keeps the article intro and throws away everything
else -- which is exactly where the payload of a list article (the crew list,
the list of names/dates/items) usually lives. This module splits the article
into sections, scores each section against the query, and fills the character
budget with the intro plus the most relevant sections. Inside a section, list
items (``li``) are kept first and individually bounded, so a list of names
survives even under a tight budget.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .html2text import sections_to_text, truncate

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "so",
        "as",
        "it",
        "its",
        "itself",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "mine",
        "tell",
        "give",
        "show",
        "list",
        "name",
        "names",
        "many",
        "much",
        "all",
        "any",
        "some",
        "no",
        "not",
        "yes",
    }
)

# Max length of a single list item once it is placed in the budget.
_MAX_ITEM_CHARS = 200
# Line kinds that carry dense, list-like facts (names, cells, entries).
_DATA_KINDS = {"li", "td", "th"}
# Per-term match-count caps used when scoring sections.
_COUNT_CAP = 15
_HEADING_CAP = 3
# Minimum number of bulleted items for a section to count as a list section.
_MIN_LIST_ITEMS = 5
# Maximum average item length: short items signal an enumeration of
# names/places (the payload of list-type questions); long items are
# descriptions, not a compact list.
_MAX_AVG_ITEM_CHARS = 100
# Headings whose list items are references/links/media rather than content.
_NON_CONTENT_HEADINGS = frozenset(
    {
        "references",
        "bibliography",
        "further reading",
        "external links",
        "see also",
        "notes",
        "footnotes",
        "sources",
        "citations",
        "gallery",
        "media",
        "referenties",
        "bronnen",
        "literatuur",
        "voetnoten",
        "externe links",
        "zie ook",
        "galerij",
        "mediabestanden",
    }
)


def _is_list_section(heading: str, lines: Sequence[tuple[str, str]]) -> bool:
    """True for compact bulleted lists of content items (e.g. the settlements
    of a municipality). Such sections carry the payload of list-type questions
    and are kept in the budget even when the query terms score them low."""
    items = [t for k, t in lines if k == "li"]
    if len(items) < _MIN_LIST_ITEMS:
        return False
    if heading.strip().lower() in _NON_CONTENT_HEADINGS:
        return False
    return sum(len(t) for t in items) / len(items) <= _MAX_AVG_ITEM_CHARS


def _variants(word: str) -> set[str]:
    """Light stemming: plural / -ed / -ing variants for fuzzy term matching."""
    v = {word}
    if word.endswith("s") and len(word) > 3:
        v.add(word[:-1])
    if word.endswith("es") and len(word) > 4:
        v.add(word[:-2])
    if word.endswith("ed") and len(word) > 4:
        v.add(word[:-2])
    if word.endswith("ing") and len(word) > 5:
        v.add(word[:-3])
    return {x for x in v if len(x) >= 3}


def query_terms(query: str) -> list[str]:
    """Meaningful, de-duplicated words of a query (lowercased, no stopwords)."""
    terms: list[str] = []
    seen: set[str] = set()
    for w in _WORD_RE.findall(query.lower()):
        if len(w) < 3 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        terms.append(w)
    return terms


def _score(heading: str, lines: Sequence[tuple[str, str]], terms: list[str]) -> int:
    """Count-based relevance: how often query terms occur in the section.

    A table section full of matching rows (e.g. the list of Moon walkers)
    outranks a short section that merely mentions the terms once.
    """
    words = _WORD_RE.findall(" ".join([heading] + [t for _, t in lines]).lower())
    hwords = _WORD_RE.findall(heading.lower())
    score = 0
    for t in terms:
        vs = _variants(t)
        score += min(sum(1 for w in words if w in vs), _COUNT_CAP)
        score += 3 * min(sum(1 for w in hwords if w in vs), _HEADING_CAP)
    return score


def _section_block(heading: str, lines: Sequence[tuple[str, str]], budget: int) -> str:
    """Render one section into at most ``budget`` chars.

    Data lines (list items, table cells) are placed first, each bounded to
    ``_MAX_ITEM_CHARS``, because they carry the densest facts for list-type
    questions; prose fills whatever budget remains.
    """
    if budget <= 0 or not lines:
        return ""
    parts: list[str] = []
    if heading:
        parts.append(heading)
    used = sum(len(p) for p in parts)

    data = [t for k, t in lines if k in _DATA_KINDS]
    prose = [t for k, t in lines if k not in _DATA_KINDS]
    chosen: list[str] = []
    for t in data:
        item = t if len(t) <= _MAX_ITEM_CHARS else truncate(t, _MAX_ITEM_CHARS)
        if chosen and used + len(item) + 1 > budget:
            break
        chosen.append(item)
        used += len(item) + 1
    for t in prose:
        if chosen and used + len(t) + 1 > budget:
            break
        chosen.append(t)
        used += len(t) + 1
    if not chosen:
        return ""
    return "\n".join(parts + chosen)


def select_article_text(
    sections: list[tuple[str, list[tuple[str, str]]]], query: str, max_chars: int
) -> str:
    """Pick the part of an article most relevant to ``query`` within budget.

    Falls back to a plain prefix truncation when the query carries no
    meaningful terms or nothing in the article matches them.
    """
    if max_chars <= 0 or not sections:
        return ""
    terms = query_terms(query)
    if not terms:
        return truncate(sections_to_text(sections), max_chars)

    ranked: list[tuple[int, int]] = []
    list_idx: list[int] = []
    for idx, (heading, lines) in enumerate(sections):
        if idx != 0 and _is_list_section(heading, lines):
            list_idx.append(idx)
            continue
        s = _score(heading, lines, terms)
        if s > 0:
            ranked.append((s, idx))
    if not ranked and not list_idx:
        return truncate(sections_to_text(sections), max_chars)

    # Bulleted list sections enumerate the sub-items of the entity (e.g. the
    # settlements of a municipality) and are the payload of list-type
    # questions. They are ranked as if they had scored three quarters of the
    # best-scoring section: high enough to beat low-relevance filler, low
    # enough to stay behind a section that truly matches the query (e.g. a
    # crew table).
    floor = max((s for s, _ in ranked), default=0) * 3 // 4
    for idx in list_idx:
        s = _score(sections[idx][0], sections[idx][1], terms)
        ranked.append((max(s, floor), idx))

    intro_budget = max(200, max_chars // 3)
    blocks: list[str] = []
    used = 0
    # Intro (title + lead) first for context, then sections by relevance.
    order = [0]
    order += [idx for _, idx in sorted(ranked, key=lambda x: (-x[0], x[1])) if idx != 0]
    for idx in order:
        if used >= max_chars:
            break
        heading, lines = sections[idx]
        budget = intro_budget if idx == 0 else max_chars - used
        block = _section_block(heading, lines, budget)
        if not block:
            continue
        blocks.append(block)
        used += len(block)
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = truncate(text, max_chars)
    return text
