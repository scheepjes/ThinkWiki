"""ctypes binding to the native libzim wrapper (native/libzim_wrapper.so).

Provides a small, dependency-free interface to open a ZIM archive, run
full-text searches, and fetch article content by path or title.
"""

from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path
from typing import Any, Self

_LIB_NAME = "libzim_wrapper.so"

# Meta-refresh redirect stubs: <meta http-equiv="refresh" content="0;URL='...'">
_REFRESH_RE = re.compile(r"http-equiv=[\"']refresh[\"'][^>]*URL=[\"']([^\"']+)", re.IGNORECASE)


def _load_lib() -> ctypes.CDLL:
    here = Path(__file__).resolve().parent.parent / "native"
    lib_path = here / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(f"native lib not found at {lib_path}; run `bash native/build.sh` first")
    lib = ctypes.CDLL(str(lib_path))

    lib.zimw_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.zimw_open.restype = ctypes.c_void_p
    lib.zimw_close.argtypes = [ctypes.c_void_p]
    lib.zimw_close.restype = None
    lib.zimw_article_count.argtypes = [ctypes.c_void_p]
    lib.zimw_article_count.restype = ctypes.c_long
    lib.zimw_entry_count.argtypes = [ctypes.c_void_p]
    lib.zimw_entry_count.restype = ctypes.c_long
    lib.zimw_has_fulltext_index.argtypes = [ctypes.c_void_p]
    lib.zimw_has_fulltext_index.restype = ctypes.c_int
    lib.zimw_get_metadata.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.zimw_get_metadata.restype = ctypes.c_void_p
    for name in ("zimw_get_article_by_path", "zimw_get_article_by_title"):
        getattr(lib, name).argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),  # out_title
            ctypes.POINTER(ctypes.c_void_p),  # out_path
            ctypes.POINTER(ctypes.c_void_p),  # out_mimetype
            ctypes.POINTER(ctypes.c_void_p),  # out_content
            ctypes.POINTER(ctypes.c_long),  # out_content_len
        ]
        getattr(lib, name).restype = ctypes.c_int
    lib.zimw_search.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.zimw_search.restype = ctypes.c_void_p
    lib.zimw_search_count.argtypes = [ctypes.c_void_p]
    lib.zimw_search_count.restype = ctypes.c_int
    lib.zimw_search_result.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zimw_search_result.restype = ctypes.c_int
    lib.zimw_search_free.argtypes = [ctypes.c_void_p]
    lib.zimw_search_free.restype = None
    lib.zimw_free.argtypes = [ctypes.c_void_p]
    lib.zimw_free.restype = None
    return lib


class Zim:
    """A thin, safe wrapper around an opened ZIM archive."""

    def __init__(self, path: str) -> None:
        self._lib = _load_lib()
        self._path = str(path)
        err = ctypes.create_string_buffer(512)
        handle = self._lib.zimw_open(os.fsencode(self._path), err, 512)
        if not handle:
            raise RuntimeError(
                f"failed to open ZIM {self._path}: {err.value.decode('utf-8', 'replace')}"
            )
        self._h = handle

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._h:
            self._lib.zimw_close(self._h)
            self._h = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # -- helpers -----------------------------------------------------------
    def _cstr(self, ptr: ctypes.c_void_p | None) -> str | None:
        if not ptr:
            return None
        try:
            return ctypes.string_at(ptr).decode("utf-8", "replace")
        finally:
            self._lib.zimw_free(ptr)

    # -- info --------------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    @property
    def article_count(self) -> int:
        return int(self._lib.zimw_article_count(self._h))

    @property
    def entry_count(self) -> int:
        return int(self._lib.zimw_entry_count(self._h))

    @property
    def has_fulltext_index(self) -> bool:
        return bool(self._lib.zimw_has_fulltext_index(self._h))

    def metadata(self, name: str) -> str | None:
        ok = ctypes.c_int(0)
        ptr = self._lib.zimw_get_metadata(self._h, name.encode("utf-8"), ctypes.byref(ok))
        return self._cstr(ptr)

    # -- article access ----------------------------------------------------
    def get_article(
        self, path: str | None = None, title: str | None = None, _depth: int = 0
    ) -> dict[str, Any] | None:
        """Fetch an article, transparently following ZIM redirect stubs.

        Some ZIMs store redirects as tiny meta-refresh HTML stubs; when one is
        returned, the target article is fetched instead (bounded to avoid
        redirect loops).
        """
        result = self._get_article_raw(path=path, title=title)
        if result is None or _depth >= 3:
            return result
        content = result.get("content") or b""
        if len(content) > 2000:
            return result
        m = _REFRESH_RE.search(content.decode("utf-8", "replace"))
        if not m:
            return result
        target = m.group(1).split("#", 1)[0].split("?", 1)[0].lstrip("./")
        if not target or "://" in target or target == result.get("path"):
            return result
        followed = self.get_article(path=target, _depth=_depth + 1)
        return followed or result

    def _get_article_raw(
        self, path: str | None = None, title: str | None = None
    ) -> dict[str, Any] | None:
        if path is None and title is None:
            raise ValueError("provide either path or title")
        out_title = ctypes.c_void_p()
        out_path = ctypes.c_void_p()
        out_mime = ctypes.c_void_p()
        out_content = ctypes.c_void_p()
        out_len = ctypes.c_long()
        if path is not None:
            fn = self._lib.zimw_get_article_by_path
            key = path
        else:
            fn = self._lib.zimw_get_article_by_title
            key = title  # type: ignore[assignment]
        ok = fn(
            self._h,
            key.encode("utf-8"),
            ctypes.byref(out_title),
            ctypes.byref(out_path),
            ctypes.byref(out_mime),
            ctypes.byref(out_content),
            ctypes.byref(out_len),
        )
        if not ok:
            return None
        result = {
            "title": self._cstr(out_title),
            "path": self._cstr(out_path),
            "mimetype": self._cstr(out_mime),
            "content": ctypes.string_at(out_content, out_len.value) if out_len.value else b"",
        }
        self._lib.zimw_free(out_content)
        return result

    # -- search ------------------------------------------------------------
    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        if not query:
            return []
        err = ctypes.create_string_buffer(512)
        rs = self._lib.zimw_search(self._h, query.encode("utf-8"), int(max_results), err, 512)
        if not rs:
            return []
        try:
            n = self._lib.zimw_search_count(rs)
            results: list[dict[str, Any]] = []
            for i in range(n):
                t = ctypes.c_void_p()
                p = ctypes.c_void_p()
                s = ctypes.c_int()
                sn = ctypes.c_void_p()
                if self._lib.zimw_search_result(
                    rs,
                    i,
                    ctypes.byref(t),
                    ctypes.byref(p),
                    ctypes.byref(s),
                    ctypes.byref(sn),
                ):
                    results.append(
                        {
                            "title": self._cstr(t),
                            "path": self._cstr(p),
                            "score": int(s.value),
                            "snippet": self._cstr(sn),
                        }
                    )
            return results
        finally:
            self._lib.zimw_search_free(rs)
