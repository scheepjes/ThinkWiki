"""OpenAI-compatible proxy server with Wikipedia (Kiwix/ZIM) grounding.

Drop-in for the OpenAI Chat Completions API. Before forwarding a request it
optionally extracts facts, looks them up in a local ZIM, and injects the
resulting articles into the prompt. Responses (JSON or SSE) are proxied back
to the client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .augment import augment_messages
from .config import Config, load_config
from .debug import DebugStore
from .facts import extract_facts
from .kiwix import KiwixLookup

log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _latest_user_text(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            joined = "\n".join(x for x in parts if x)
            return joined or None
    return None


def _accumulate_sse(acc: dict[str, Any], chunk: bytes) -> None:
    """Fold SSE `data:` chunks into ``acc`` (content, finish_reason, chunk count)."""
    acc["buf"] += chunk.decode("utf-8", "replace")
    while "\n" in acc["buf"]:
        line, acc["buf"] = acc["buf"].split("\n", 1)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            log.debug("skipping malformed SSE data line: %r", data[:120])
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                acc["content"] += content
            finish = choice.get("finish_reason")
            if finish:
                acc["finish_reason"] = finish


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    _configure_logging(cfg.logging.level)

    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["http"] = httpx.AsyncClient(timeout=cfg.upstream.timeout_seconds)
        state["debug"] = DebugStore(cfg.debug.max_entries) if cfg.debug.enabled else None
        if cfg.kiwix.zim_path or cfg.kiwix.zim_dir:
            try:
                state["kiwix"] = KiwixLookup(cfg)
                log.info(
                    "opened %d ZIM archive(s) (%d articles): %s",
                    len(state["kiwix"].zim.names),
                    state["kiwix"].zim.article_count,
                    ", ".join(state["kiwix"].zim.names),
                )
            except Exception as e:  # pragma: no cover - depends on data file
                log.error("failed to open ZIM archives: %s", e)
                state["kiwix"] = None
        else:
            log.warning("kiwix.zim_path / kiwix.zim_dir not set; enrichment disabled")
            state["kiwix"] = None
        yield
        await state["http"].aclose()
        if state.get("kiwix") is not None:
            state["kiwix"].close()

    app = FastAPI(title="ThinkWiki proxy", lifespan=lifespan)

    def _headers() -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.upstream.api_key:
            h["Authorization"] = f"Bearer {cfg.upstream.api_key}"
        return h

    def _base() -> str:
        return cfg.upstream.base_url.rstrip("/")

    def _resolve_model(requested: str | None) -> str:
        """Map a requested (public) model id to the upstream model to call.

        If a model catalog is configured, a known id is mapped to its upstream
        model and an unknown/absent id falls back to the catalog default. With no
        catalog configured, the legacy behavior (passthrough / upstream default)
        is used.
        """
        models = cfg.models
        if not models.entries:
            return requested or cfg.upstream.default_model
        if not requested:
            requested = models.default or models.entries[0].id
        for entry in models.entries:
            if entry.id == requested:
                return entry.upstream_model
        return requested

    def _model_objects() -> list[dict[str, Any]]:
        now = int(time.time())
        return [
            {
                "id": e.id,
                "object": "model",
                "created": now,
                "owned_by": "thinkwiki",
                "description": e.description or None,
            }
            for e in cfg.models.entries
        ]

    async def _enrich(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list]:
        """Return ``(payload, facts, articles)``.

        The payload is augmented when enrichment succeeds; on any failure the
        original payload is returned unchanged with empty facts/articles.
        """
        if not cfg.fact_extraction.enabled:
            return payload, [], []
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload, [], []
        user_text = _latest_user_text(messages)
        if not user_text:
            return payload, [], []
        kiwix: KiwixLookup | None = state.get("kiwix")
        if kiwix is None:
            return payload, [], []
        try:
            facts = await extract_facts(
                state["http"], cfg, _base(), cfg.upstream.api_key, user_text
            )
            if not facts:
                return payload, [], []
            articles = await asyncio.to_thread(kiwix.lookup, facts)
            if not articles:
                return payload, facts, []
            log.info("enriching with %d facts -> %d articles", len(facts), len(articles))
            out = dict(payload)
            out["messages"] = augment_messages(cfg, messages, facts, articles)
            return out, facts, articles
        except Exception as e:
            log.warning("enrichment failed; forwarding unchanged: %s", e)
            return payload, [], []

    def _debug_entry(
        *,
        request: dict[str, Any],
        model_requested: str | None,
        model_upstream: str,
        facts: list[str],
        articles: list,
        upstream_request: dict[str, Any],
        response: dict[str, Any] | None,
        status_code: int,
        latency_ms: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "request": request,
            "model_requested": model_requested,
            "model_upstream": model_upstream,
            "facts": facts,
            "articles": [
                {
                    "title": a.title,
                    "path": a.path,
                    "source": a.source,
                    "chars": len(a.text),
                }
                for a in articles
            ],
            "upstream_request": upstream_request,
            "response": response,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }

    if "/v1/chat/completions" in cfg.endpoints:

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse({"error": "request body must be an object"}, status_code=400)

            client_request = dict(body)
            model_requested = body.get("model")
            body["model"] = _resolve_model(model_requested)
            model_upstream = body["model"]

            t0 = time.perf_counter()
            enriched, facts, articles = await _enrich(body)
            if cfg.upstream.temperature is not None:
                enriched["temperature"] = cfg.upstream.temperature
            url = _base() + "/chat/completions"
            debug: DebugStore | None = state.get("debug")

            if enriched.get("stream"):
                req = state["http"].build_request("POST", url, json=enriched, headers=_headers())
                upstream = await state["http"].send(req, stream=True)
                acc: dict[str, Any] = {"content": "", "finish_reason": None, "chunks": 0, "buf": ""}

                async def gen():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            acc["chunks"] += 1
                            _accumulate_sse(acc, chunk)
                            yield chunk
                    finally:
                        await upstream.aclose()
                        if debug is not None:
                            debug.record(
                                _debug_entry(
                                    request=client_request,
                                    model_requested=model_requested,
                                    model_upstream=model_upstream,
                                    facts=facts,
                                    articles=articles,
                                    upstream_request=enriched,
                                    response={
                                        "content": acc["content"],
                                        "finish_reason": acc["finish_reason"],
                                        "stream": True,
                                        "chunks": acc["chunks"],
                                    },
                                    status_code=upstream.status_code,
                                    latency_ms=(time.perf_counter() - t0) * 1000,
                                )
                            )

                return StreamingResponse(
                    gen(),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "text/event-stream"),
                )

            resp = await state["http"].post(url, json=enriched, headers=_headers())
            latency_ms = (time.perf_counter() - t0) * 1000
            try:
                data = resp.json()
            except Exception:
                if debug is not None:
                    debug.record(
                        _debug_entry(
                            request=client_request,
                            model_requested=model_requested,
                            model_upstream=model_upstream,
                            facts=facts,
                            articles=articles,
                            upstream_request=enriched,
                            response=None,
                            status_code=resp.status_code,
                            latency_ms=latency_ms,
                            error="upstream returned non-JSON",
                        )
                    )
                return JSONResponse(
                    {"error": "upstream returned non-JSON", "status": resp.status_code},
                    status_code=resp.status_code,
                )
            if debug is not None:
                choice = (data.get("choices") or [{}])[0]
                debug.record(
                    _debug_entry(
                        request=client_request,
                        model_requested=model_requested,
                        model_upstream=model_upstream,
                        facts=facts,
                        articles=articles,
                        upstream_request=enriched,
                        response={
                            "content": choice.get("message", {}).get("content"),
                            "finish_reason": choice.get("finish_reason"),
                            "usage": data.get("usage"),
                        },
                        status_code=resp.status_code,
                        latency_ms=latency_ms,
                    )
                )
            return JSONResponse(data, status_code=resp.status_code)

    if "/v1/models" in cfg.endpoints:

        @app.get("/v1/models")
        async def models():
            # If the proxy defines its own model catalog, serve it; otherwise
            # fall back to forwarding the upstream's model list.
            if cfg.models.entries:
                return JSONResponse({"object": "list", "data": _model_objects()})
            resp = await state["http"].get(_base() + "/models", headers=_headers())
            try:
                data = resp.json()
            except Exception:
                data = {"object": "list", "data": [{"id": cfg.upstream.default_model}]}
            return JSONResponse(data, status_code=resp.status_code)

    if cfg.debug.enabled:

        @app.get("/debug/requests")
        async def debug_requests():
            store: DebugStore | None = state.get("debug")
            if store is None:
                return JSONResponse({"error": "debug not enabled"}, status_code=404)
            return {"object": "list", "count": len(store), "data": store.entries()}

        @app.get("/debug/requests/{entry_id}")
        async def debug_request_one(entry_id: int):
            store = state.get("debug")
            if store is None:
                return JSONResponse({"error": "debug not enabled"}, status_code=404)
            entry = store.get(entry_id)
            if entry is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            return entry

        @app.delete("/debug/requests")
        async def debug_clear():
            store = state.get("debug")
            if store is None:
                return JSONResponse({"error": "debug not enabled"}, status_code=404)
            return {"cleared": store.clear()}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "enrichment": state.get("kiwix") is not None,
            "debug": state.get("debug") is not None,
        }

    return app


app = create_app()
