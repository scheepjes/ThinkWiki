"""End-to-end integration tests: real proxy server + mock OpenAI upstream.

Verifies the full path: fact extraction -> ZIM lookup -> prompt augmentation ->
upstream call -> response passthrough (JSON and SSE), plus the passthrough
behavior when enrichment is disabled.
"""

import httpx

EXTRACT_PREFIX = "Extract the factual claims"


def _is_extraction(body) -> bool:
    return any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and (m.get("content") or "").startswith(EXTRACT_PREFIX)
        for m in body.get("messages", [])
    )


def test_health(proxy_server):
    r = httpx.get(proxy_server + "/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["enrichment"] is True


def test_models_passthrough(proxy_server):
    r = httpx.get(proxy_server + "/v1/models", timeout=10)
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "mock-model"


def test_chat_completions_enriched(proxy_server, mock_upstream):
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"].startswith("Grounded answer")

    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert len(main_calls) == 1, "expected exactly one forwarded chat call"
    main = main_calls[0]

    # The forwarded request must contain an augmented system message with
    # the retrieved Wikipedia article.
    sys_msgs = [m for m in main["messages"] if m.get("role") == "system"]
    assert sys_msgs, "expected an augmented system message"
    content = sys_msgs[0]["content"]
    assert "Facts to verify:" in content
    assert "Albert Einstein" in content
    # The original user message is preserved.
    assert any(
        m.get("role") == "user" and "Who was Einstein?" in (m.get("content") or "")
        for m in main["messages"]
    )
    # Model defaulting: client specified a model, it is preserved.
    assert main["model"] == "mock-model"


def test_model_defaults_to_configured(proxy_server, mock_upstream):
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls[-1]["model"] == "mock-model"


def test_streaming_sse_passthrough(proxy_server, mock_upstream):
    with httpx.stream(
        "POST",
        proxy_server + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=60,
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        body = b"".join(r.iter_bytes()).decode("utf-8")
    assert "data:" in body
    assert "[DONE]" in body
    assert "Grounded answer" in body


def test_no_enrichment_when_disabled(proxy_server_noenrich, mock_upstream):
    before = len(mock_upstream.requests)
    r = httpx.post(
        proxy_server_noenrich + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    new = mock_upstream.requests[before:]
    # No extraction call and no augmented system message: exactly one call,
    # forwarded unchanged.
    assert len(new) == 1
    assert not _is_extraction(new[0])
    assert not any(m.get("role") == "system" for m in new[0]["messages"])
    assert new[0]["messages"] == [{"role": "user", "content": "Who was Einstein?"}]


def test_list_question_gets_named_list(proxy_server, mock_upstream):
    """A list-type question must be grounded with the actual named list.

    The mock upstream extracts ["Apollo astronauts", "Apollo 11"] for moon
    questions; the forwarded prompt must carry the crew names from the ZIM
    article, not just a generic intro paragraph.
    """
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Which astronauts landed on the moon?"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls, "expected a forwarded chat call"
    main = main_calls[-1]

    sys_msgs = [m for m in main["messages"] if m.get("role") == "system"]
    assert sys_msgs, "expected an augmented system message"
    content = sys_msgs[0]["content"]
    # Concrete astronaut names from the ZIM list article must be present.
    assert "Neil Armstrong" in content
    assert "Aldrin" in content
    # The grounding instruction tells the model to answer with the full list.
    assert "complete list" in content.lower()


def test_zim_dir_enriches_end_to_end(proxy_server_zimdir, mock_upstream):
    """A proxy configured with a ZIM directory (not a single file) must still
    ground answers from the archives found in that directory."""
    r = httpx.post(
        proxy_server_zimdir + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls, "expected a forwarded chat call"
    main = main_calls[-1]
    sys_msgs = [m for m in main["messages"] if m.get("role") == "system"]
    assert sys_msgs, "expected an augmented system message"
    assert "Albert Einstein" in sys_msgs[0]["content"]


def test_facts_not_found_in_zim_forwards_unchanged(proxy_server, mock_upstream):
    """When extracted facts have no ZIM match, the request must be forwarded
    unchanged so the upstream LLM answers on its own (no grounding prompt)."""
    before = len(mock_upstream.requests)
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me about the Qwzx Asdf Frobnicate."}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]

    new = mock_upstream.requests[before:]
    # Facts were extracted (one extraction call) but nothing was found in the
    # ZIM, so exactly one main call goes out, forwarded byte-for-byte.
    assert any(_is_extraction(b) for b in new), "expected a fact-extraction call"
    main_calls = [b for b in new if not _is_extraction(b)]
    assert len(main_calls) == 1, "expected exactly one forwarded chat call"
    main = main_calls[0]
    assert main["messages"] == [
        {"role": "user", "content": "Tell me about the Qwzx Asdf Frobnicate."}
    ]
    assert not any(m.get("role") == "system" for m in main["messages"])


def test_forced_temperature_overrides_client(proxy_server_temperature, mock_upstream):
    """A configured upstream.temperature must override the client's value so
    answers stay deterministic."""
    r = httpx.post(
        proxy_server_temperature + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Who was Einstein?"}],
            "temperature": 0.9,
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls[-1]["temperature"] == 0.0


def test_temperature_passthrough_when_unconfigured(proxy_server, mock_upstream):
    """Without a configured upstream.temperature the client's value passes
    through unchanged."""
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    main_calls = [b for b in mock_upstream.requests if not _is_extraction(b)]
    assert main_calls[-1]["temperature"] == 0.7


def test_forced_temperature_applies_without_enrichment(proxy_server_temperature, mock_upstream):
    """The forced temperature applies even when enrichment is skipped."""
    before = len(mock_upstream.requests)
    r = httpx.post(
        proxy_server_temperature + "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me about the Qwzx Asdf Frobnicate."}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    new = mock_upstream.requests[before:]
    main_calls = [b for b in new if not _is_extraction(b)]
    assert main_calls[-1]["temperature"] == 0.0


def test_invalid_json_body(proxy_server):
    r = httpx.post(
        proxy_server + "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert r.status_code == 400
