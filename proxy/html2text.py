"""Dependency-free HTML -> plain text conversion (stdlib html.parser).

Good enough for extracting readable prose out of Wikipedia ZIM articles: it
drops scripts/styles/navigation (sidebars, navboxes, hatnotes, reference
lists), turns block elements into lines, unescapes entities, and collapses
whitespace.

It also exposes a structured view (``parse_sections``) that splits an article
into (heading, lines) sections with list items kept as individual lines. The
relevance module uses that structure to pick the parts of a long article that
matter for a given query, instead of blindly truncating the prefix.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Elements whose *content* should be discarded entirely.
_SKIP_CONTENT = {"script", "style", "head", "noscript", "template", "svg", "math"}
# Table class fragments that mark navigation chrome (sidebars, navboxes).
_SKIP_TABLE_CLASSES = ("sidebar", "navbox")
# Div class fragments that mark editorial chrome (hatnotes, edit sections).
_SKIP_DIV_CLASSES = ("hatnote", "mw-editsection", "ambox")
# List class fragments that mark the reference list.
_SKIP_LIST_CLASSES = ("references", "reflist")
# Tags whose content forms a single logical line (list items stay atomic).
_LINE_TAGS = {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption", "td", "th"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

_EDIT_RE = re.compile(r"\[edit(?:\s+source)?\]", re.IGNORECASE)
_CITE_RE = re.compile(r"\[\d+\]")
_CITATION_NEEDED_RE = re.compile(r"\[citation needed\]", re.IGNORECASE)
_NOTE_RE = re.compile(r"\[note\s*\d+\]", re.IGNORECASE)

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_markers(text: str) -> str:
    """Remove wiki editorial markers: [edit], [1], [citation needed], [note 2]."""
    text = _EDIT_RE.sub(" ", text)
    text = _CITE_RE.sub(" ", text)
    text = _CITATION_NEEDED_RE.sub(" ", text)
    text = _NOTE_RE.sub(" ", text)
    return text


def _collapse(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


class _SectionParser(HTMLParser):
    """Parse HTML into sections: (heading, [(line_kind, text), ...])."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._table_skip = 0
        self._div_skip = 0
        self._in_heading = False
        self._heading_buf: list[str] = []
        self._line_kind: str | None = None
        self._line_buf: list[str] = []
        self._loose_buf: list[str] = []
        self._sections: list[tuple[str, list[tuple[str, str]]]] = []
        self._cur: list | None = None

    # -- helpers -------------------------------------------------------------
    def _ensure(self) -> list:
        if self._cur is None:
            self._cur = ["", []]
        return self._cur

    def _flush_line(self) -> None:
        if self._line_kind is not None:
            text = clean_markers(_collapse(" ".join(self._line_buf)))
            if text:
                self._ensure()[1].append((self._line_kind, text))
            self._line_kind = None
            self._line_buf = []

    def _flush_loose(self) -> None:
        text = clean_markers(_collapse(" ".join(self._loose_buf)))
        if text:
            self._ensure()[1].append(("loose", text))
        self._loose_buf = []

    def _close_section(self) -> None:
        if self._cur is not None and (self._cur[0] or self._cur[1]):
            self._sections.append((self._cur[0], self._cur[1]))
        self._cur = None

    # -- HTMLParser hooks ------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        d = dict(attrs)
        cls = d.get("class") or ""
        role = d.get("role") or ""
        if tag == "table" and (any(c in cls for c in _SKIP_TABLE_CLASSES) or role == "navigation"):
            self._table_skip += 1
            return
        if self._table_skip:
            if tag == "table":
                self._table_skip += 1
            return
        if tag == "div" and any(c in cls for c in _SKIP_DIV_CLASSES):
            self._div_skip += 1
            return
        if self._div_skip:
            if tag == "div":
                self._div_skip += 1
            return
        if tag in _HEADING_TAGS:
            self._flush_loose()
            self._flush_line()
            self._in_heading = True
            self._heading_buf = []
            return
        if tag in _LINE_TAGS:
            self._flush_loose()
            self._flush_line()
            self._line_kind = tag
            self._line_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if tag == "table" and self._table_skip:
            self._table_skip -= 1
            return
        if tag == "div" and self._div_skip:
            self._div_skip -= 1
            return
        if self._skip_depth or self._table_skip or self._div_skip:
            return
        if tag in _HEADING_TAGS and self._in_heading:
            self._in_heading = False
            heading = clean_markers(_collapse(" ".join(self._heading_buf)))
            self._close_section()
            self._cur = [heading, []]
            return
        if tag in _LINE_TAGS and self._line_kind == tag:
            self._flush_line()

    def handle_data(self, data: str) -> None:
        if not data or self._skip_depth or self._table_skip or self._div_skip:
            return
        if self._in_heading:
            self._heading_buf.append(data)
        elif self._line_kind is not None:
            self._line_buf.append(data)
        else:
            self._loose_buf.append(data)

    def finish(self) -> list[tuple[str, list[tuple[str, str]]]]:
        self._flush_loose()
        self._flush_line()
        self._close_section()
        return self._sections


def parse_sections(html: str | bytes) -> list[tuple[str, list[tuple[str, str]]]]:
    """Split article HTML into ``(heading, [(kind, text), ...])`` sections.

    ``kind`` is the originating tag (``li``, ``p``, ``td``, ...) or ``loose``
    for text outside block elements. List items (``li``) are kept as single
    lines so callers can treat them as atomic entries.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", "replace")
    if not html:
        return []
    parser = _SectionParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Malformed HTML: fall back to a crude tag strip.
        text = clean_markers(_collapse(re.sub(r"<[^>]+>", " ", html)))
        return [("", [("loose", text)])] if text else []
    return parser.finish()


def html_to_sections(html: str | bytes) -> list[tuple[str, list[str]]]:
    """Public, untyped-kind view of :func:`parse_sections`."""
    return [(heading, [text for _, text in lines]) for heading, lines in parse_sections(html)]


def sections_to_text(sections: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Render parsed sections back to plain text (headings inline)."""
    parts: list[str] = []
    for heading, lines in sections:
        if heading:
            parts.append(heading)
        parts.extend(text for _, text in lines)
    text = "\n".join(parts)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def html_to_text(html: str | bytes) -> str:
    return sections_to_text(parse_sections(html))


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer cutting at a sentence/word boundary to avoid mid-word splits.
    for sep in (". ", ".\n", "\n", " "):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + (1 if sep == " " else 0)].rstrip()
    return cut
