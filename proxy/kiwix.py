"""Kiwix/ZIM lookup orchestration.

Turns a list of fact queries into a de-duplicated, budget-bounded list of
plain-text Wikipedia articles, with an in-memory LRU cache keyed by the
normalized query.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config, KiwixCfg
from .html2text import parse_sections, sections_to_text, truncate
from .lru import LRUCache
from .relevance import select_article_text
from .zim import ZimCollection

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    return _WS_RE.sub(" ", query.strip()).lower()


def resolve_zim_paths(kcfg: KiwixCfg) -> list[str]:
    """Resolve the configured ZIM source(s) to a list of file paths.

    ``zim_dir`` (a directory of ``*.zim`` files, sorted by file name) is
    scanned first; ``zim_path`` (a single file) is added as well when set and
    not already included.
    """
    paths: list[str] = []
    if kcfg.zim_dir:
        d = Path(kcfg.zim_dir)
        if d.is_dir():
            paths = [
                str(p)
                for p in sorted(d.iterdir(), key=lambda p: p.name)
                if p.is_file() and p.suffix.lower() == ".zim"
            ]
        else:
            log.warning("kiwix.zim_dir %s is not a directory", d)
    if kcfg.zim_path:
        p = str(kcfg.zim_path)
        if p not in paths:
            paths.append(p)
    return paths


@dataclass
class Article:
    title: str
    text: str
    path: str = ""
    source: str = ""


class KiwixLookup:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.zim = ZimCollection(resolve_zim_paths(cfg.kiwix))
        self._cache = LRUCache(cfg.kiwix.cache_size)

    def close(self) -> None:
        self.zim.close()

    def _fetch(
        self,
        path: str | None = None,
        title: str | None = None,
        query: str = "",
        source: str = "",
    ) -> Article | None:
        raw = self.zim.get_article(path=path, title=title, source=source or None)
        if not raw:
            return None
        sections = parse_sections(raw.get("content") or b"")
        # Query-aware selection keeps the parts of the article that match the
        # query (e.g. the crew list of a "List of ..." article) instead of a
        # blind prefix cut; fall back to a plain prefix if it yields nothing.
        text = select_article_text(
            sections, query or title or "", self.cfg.kiwix.max_chars_per_article
        )
        if not text.strip():
            text = truncate(sections_to_text(sections), self.cfg.kiwix.max_chars_per_article)
        if not text.strip():
            return None
        real_path = raw.get("path") or path or ""
        return Article(
            title=raw.get("title") or title or path or "",
            text=text,
            path=real_path,
            source=raw.get("source") or source or "",
        )

    def _search_one(self, query: str) -> list[Article]:
        norm = normalize_query(query)
        cached = self._cache.get(norm)
        if cached is not None:
            return cached

        limit = self.cfg.kiwix.max_articles_per_fact
        articles: list[Article] = []
        seen: set[str] = set()

        def add(path: str | None, title: str | None, source: str = "") -> None:
            if len(articles) >= limit:
                return
            key = normalize_query(title or "")
            if key and key in seen:
                return
            art = self._fetch(path=path, title=title, source=source, query=query)
            if art is None:
                return
            # De-duplicate by the resolved title (redirects may change it), so
            # the same article is not fetched twice under different names, and
            # the same article in two archives counts once.
            key = normalize_query(art.title) or (path or "")
            if key in seen:
                return
            seen.add(key)
            articles.append(art)

        # 1) An exact article-title match is the most relevant hit for entity
        #    queries, so it always takes the first slot (when it exists).
        add(None, query.strip())
        # 2) Fill the remaining slots with full-text search results (merged
        #    across all archives, best score first).
        if self.zim.has_fulltext_index:
            for r in self.zim.search(query, limit * 3):
                add(r.get("path"), r.get("title"), r.get("source") or "")
                if len(articles) >= limit:
                    break

        self._cache.set(norm, articles)
        return articles

    def lookup(self, queries: list[str]) -> list[Article]:
        """Look up every query, de-duplicate across facts, enforce the budget."""
        budget = self.cfg.kiwix.total_char_budget
        out: list[Article] = []
        seen: set[str] = set()
        used = 0
        for q in queries:
            if not q or not q.strip():
                continue
            for art in self._search_one(q):
                # De-duplicate by resolved title across facts and archives.
                key = normalize_query(art.title) or art.path
                if key in seen:
                    continue
                seen.add(key)
                cost = len(art.text)
                # Always allow at least one article; stop once the budget is met.
                if out and used + cost > budget:
                    break
                out.append(art)
                used += cost
                if used >= budget:
                    break
            if used >= budget:
                break
        log.debug("lookup: %d queries -> %d articles (%d chars)", len(queries), len(out), used)
        return out
