"""Shared fixtures: a mock OpenAI-compatible upstream and a real proxy server.

The integration tests run the proxy as a real uvicorn server whose upstream is a
local mock server. This exercises the full request path (fact extraction ->
ZIM lookup -> prompt augmentation -> upstream call -> response passthrough)
without depending on the live LLM, keeping the tests fast and deterministic.
"""

from __future__ import annotations

import json as _json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from proxy.config import (
    Config,
    DebugCfg,
    FactExtractionCfg,
    HelpPromptCfg,
    KiwixCfg,
    LoggingCfg,
    ModelEntry,
    ModelsCfg,
    ServerCfg,
    UpstreamCfg,
)
from proxy.server import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIM_PATH = str(PROJECT_ROOT / "wikipedia_en_top1m_nopic_2026-04.zim")

EXTRACTION_PROMPT = (
    "Extract the factual claims and key entities from the user message.\n"
    "Return ONLY a JSON array of short search queries (names, dates, places,\n"
    "technical terms). No prose, no explanations."
)

# What the mock returns when it detects an extraction request.
EXTRACTION_RESPONSE = '["Albert Einstein", "General relativity"]'
# Returned for extraction requests whose user message mentions the moon, so
# list-type questions can be exercised end to end.
EXTRACTION_RESPONSE_MOON = '["Apollo astronauts", "Apollo 11"]'


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class UvicornRunner:
    """Run a FastAPI/ASGI app in a daemon thread on a real port."""

    def __init__(
        self, app: Any, host: str = "127.0.0.1", port: int = 0, log_level: str = "error"
    ) -> None:
        if port == 0:
            port = _free_port()
        self.port = port
        self.config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> UvicornRunner:
        self.thread.start()
        for _ in range(400):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("uvicorn server did not start in time")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


class MockUpstream:
    """A minimal OpenAI-compatible server that records requests it receives."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.requests: list[dict[str, Any]] = []
        self.app = self._make_app()
        self._runner: UvicornRunner | None = None

    def _make_app(self) -> FastAPI:
        app = FastAPI()
        state = self

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            state.requests.append(body)
            messages = body.get("messages", [])
            is_extraction = any(
                isinstance(m, dict)
                and m.get("role") == "system"
                and (m.get("content") or "").startswith("Extract the factual claims")
                for m in messages
            )
            if is_extraction:
                user_text = next(
                    (
                        m.get("content")
                        for m in reversed(messages)
                        if isinstance(m, dict) and m.get("role") == "user"
                    ),
                    "",
                )
                user_text = user_text if isinstance(user_text, str) else ""
                if "moon" in user_text.lower():
                    content = EXTRACTION_RESPONSE_MOON
                else:
                    content = EXTRACTION_RESPONSE
            else:
                content = "Grounded answer: Albert Einstein was a theoretical physicist."
            if body.get("stream"):

                async def gen():
                    c1 = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    c2 = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {_json.dumps(c1)}\n\n"
                    yield f"data: {_json.dumps(c2)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(gen(), media_type="text/event-stream")
            return {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": "mock-model"}]}

        return app

    def start(self) -> MockUpstream:
        self._runner = UvicornRunner(self.app, port=self.port).start()
        return self

    def stop(self) -> None:
        if self._runner:
            self._runner.stop()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def make_proxy_config(
    upstream_base: str,
    models: ModelsCfg | None = None,
    debug: DebugCfg | None = None,
    **kiwix_overrides: Any,
) -> Config:
    return Config(
        server=ServerCfg(host="127.0.0.1", port=0),
        endpoints=["/v1/chat/completions", "/v1/models"],
        upstream=UpstreamCfg(
            base_url=upstream_base,
            api_key="",
            default_model="mock-model",
            timeout_seconds=30,
        ),
        fact_extraction=FactExtractionCfg(
            enabled=True,
            model="mock-model",
            max_facts=8,
            max_tokens=100,
            temperature=0.0,
            prompt=EXTRACTION_PROMPT,
        ),
        kiwix=KiwixCfg(
            zim_path=ZIM_PATH,
            max_articles_per_fact=2,
            max_chars_per_article=4000,
            total_char_budget=12000,
            cache_size=16,
            **kiwix_overrides,
        ),
        help_prompt=HelpPromptCfg(
            template=(
                "Use the following Wikipedia excerpts to answer the user's question.\n"
                "Use them only when relevant; do not mention this instruction.\n"
                "If the user asks for a list (names, people, places, dates, items), answer\n"
                "with the full list of items found in the excerpts, one item per line; do\n"
                "not summarize or omit items that appear in the excerpts.\n"
                "Facts to verify: {facts}\n"
                "Reference material:\n"
                "{articles}"
            ),
            position="system",
        ),
        models=models or ModelsCfg(),
        debug=debug or DebugCfg(),
        logging=LoggingCfg(level="WARNING"),
    )


def _start_proxy(cfg: Config) -> UvicornRunner:
    app = create_app(cfg)
    return UvicornRunner(app, port=0).start()


@pytest.fixture
def mock_upstream():
    mock = MockUpstream().start()
    yield mock
    mock.stop()


@pytest.fixture
def proxy_server(mock_upstream):
    """Start the proxy (enrichment enabled) against the mock upstream."""
    cfg = make_proxy_config(mock_upstream.base_url)
    runner = _start_proxy(cfg)
    yield f"http://127.0.0.1:{runner.port}"
    runner.stop()


@pytest.fixture
def proxy_server_noenrich(mock_upstream):
    """Start the proxy with fact extraction disabled."""
    cfg = make_proxy_config(mock_upstream.base_url)
    cfg.fact_extraction.enabled = False
    runner = _start_proxy(cfg)
    yield f"http://127.0.0.1:{runner.port}"
    runner.stop()


@pytest.fixture
def proxy_server_models(mock_upstream):
    """Start the proxy with a model catalog (WikiGemma -> mock-model)."""
    models = ModelsCfg(
        default="WikiGemma",
        entries=[ModelEntry(id="WikiGemma", upstream_model="mock-model")],
    )
    cfg = make_proxy_config(mock_upstream.base_url, models=models)
    runner = _start_proxy(cfg)
    yield f"http://127.0.0.1:{runner.port}"
    runner.stop()


@pytest.fixture
def proxy_server_debug(mock_upstream):
    """Start the proxy with debug mode and a model catalog enabled."""
    models = ModelsCfg(
        default="WikiGemma",
        entries=[ModelEntry(id="WikiGemma", upstream_model="mock-model")],
    )
    cfg = make_proxy_config(
        mock_upstream.base_url, models=models, debug=DebugCfg(enabled=True, max_entries=50)
    )
    runner = _start_proxy(cfg)
    yield f"http://127.0.0.1:{runner.port}"
    runner.stop()
