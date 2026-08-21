"""Live end-to-end test against the real local LLM (http://localhost:9001/v1).

Skipped automatically if the upstream LLM is not reachable, so the suite still
passes in environments without the model running.
"""

import time

import httpx
import pytest
from conftest import UvicornRunner

from proxy.config import load_config
from proxy.server import create_app

LIVE_BASE = "http://localhost:9001/v1"


def _llm_reachable(base: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(base + "/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _llm_reachable(LIVE_BASE),
    reason="live LLM at http://localhost:9001/v1 is not reachable",
)


@pytest.fixture
def live_proxy():
    cfg = load_config()
    assert cfg.upstream.base_url == LIVE_BASE, "expected config to target the live LLM"
    app = create_app(cfg)
    runner = UvicornRunner(app, port=0).start()
    yield f"http://127.0.0.1:{runner.port}"
    runner.stop()


def test_live_end_to_end(live_proxy):
    t0 = time.time()
    r = httpx.post(
        live_proxy + "/v1/chat/completions",
        json={
            "model": cfg_model(),
            "messages": [
                {
                    "role": "user",
                    "content": "In one sentence, who was Albert Einstein?",
                }
            ],
            "stream": False,
        },
        timeout=180,
    )
    dt = time.time() - t0
    assert r.status_code == 200, r.text
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 0
    # The answer should be grounded in the Einstein article.
    assert "einstein" in content.lower()
    print(f"\n[live] answered in {dt:.1f}s: {content[:200]}")


def cfg_model() -> str:
    return load_config().upstream.default_model
