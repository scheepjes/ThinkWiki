"""Tests for the model catalog (WikiGemma) and debug mode."""

import httpx

EXTRACT_PREFIX = "Extract the factual claims"


def _is_extraction(body) -> bool:
    return any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and (m.get("content") or "").startswith(EXTRACT_PREFIX)
        for m in body.get("messages", [])
    )


# --- Model catalog ---------------------------------------------------------


def test_models_endpoint_returns_catalog(proxy_server_models):
    r = httpx.get(proxy_server_models + "/v1/models", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "WikiGemma" in ids
    wiki = next(m for m in data["data"] if m["id"] == "WikiGemma")
    assert wiki["object"] == "model"


def test_wikigemma_maps_to_upstream_model(proxy_server_models, mock_upstream):
    r = httpx.post(
        proxy_server_models + "/v1/chat/completions",
        json={
            "model": "WikiGemma",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls, "expected a forwarded chat call"
    # The public id WikiGemma must be translated to the upstream model.
    assert main_calls[-1]["model"] == "mock-model"
    assert main_calls[-1]["model"] != "WikiGemma"


def test_omitted_model_uses_default(proxy_server_models, mock_upstream):
    r = httpx.post(
        proxy_server_models + "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls[-1]["model"] == "mock-model"


def test_legacy_passthrough_without_catalog(proxy_server, mock_upstream):
    # No model catalog configured: the requested model passes through unchanged.
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={
            "model": "some-raw-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls[-1]["model"] == "some-raw-model"


# --- Debug mode ------------------------------------------------------------


def test_debug_captures_nonstream(proxy_server_debug, mock_upstream):
    httpx.post(
        proxy_server_debug + "/v1/chat/completions",
        json={
            "model": "WikiGemma",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "stream": False,
        },
        timeout=60,
    )
    r = httpx.get(proxy_server_debug + "/debug/requests", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    entry = data["data"][0]  # most recent first
    # The client's original query is captured.
    assert entry["request"]["messages"] == [{"role": "user", "content": "Who was Einstein?"}]
    assert entry["model_requested"] == "WikiGemma"
    assert entry["model_upstream"] == "mock-model"
    # Facts extracted and articles retrieved are recorded.
    assert entry["facts"], "expected extracted facts to be captured"
    assert entry["articles"], "expected retrieved articles to be captured"
    # The augmented upstream request (the grounding prompt) is captured.
    sys_msgs = [m for m in entry["upstream_request"]["messages"] if m.get("role") == "system"]
    assert sys_msgs and "Facts to verify:" in sys_msgs[0]["content"]
    # The response is captured.
    assert entry["response"]["content"].startswith("Grounded answer")
    assert entry["status_code"] == 200
    assert entry["latency_ms"] >= 0


def test_debug_captures_stream(proxy_server_debug, mock_upstream):
    with httpx.stream(
        "POST",
        proxy_server_debug + "/v1/chat/completions",
        json={
            "model": "WikiGemma",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=60,
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())
    r = httpx.get(proxy_server_debug + "/debug/requests", timeout=10)
    entry = r.json()["data"][0]
    assert entry["response"]["stream"] is True
    assert entry["response"]["chunks"] >= 1
    assert "Grounded answer" in entry["response"]["content"]


def test_debug_get_single_and_clear(proxy_server_debug):
    httpx.post(
        proxy_server_debug + "/v1/chat/completions",
        json={
            "model": "WikiGemma",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        timeout=60,
    )
    listing = httpx.get(proxy_server_debug + "/debug/requests", timeout=10).json()
    entry_id = listing["data"][0]["id"]

    single = httpx.get(f"{proxy_server_debug}/debug/requests/{entry_id}", timeout=10)
    assert single.status_code == 200
    assert single.json()["id"] == entry_id

    missing = httpx.get(f"{proxy_server_debug}/debug/requests/999999", timeout=10)
    assert missing.status_code == 404

    cleared = httpx.delete(proxy_server_debug + "/debug/requests", timeout=10)
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] >= 1
    assert httpx.get(proxy_server_debug + "/debug/requests", timeout=10).json()["count"] == 0


def test_debug_disabled_by_default(proxy_server):
    # The plain proxy fixture has debug disabled -> endpoints are absent.
    r = httpx.get(proxy_server + "/debug/requests", timeout=10)
    assert r.status_code == 404
    health = httpx.get(proxy_server + "/health", timeout=10).json()
    assert health["debug"] is False
