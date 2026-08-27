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


# --- consistency of list-type questions ------------------------------------
#
# Repeating the same list question must yield the same, complete answer every
# time: the grounded excerpt carries the full list and the forced temperature
# (0.0 in config.yaml) removes sampling variance.

MEDEMBLIK_PLACES = (
    "Abbekerk",
    "Andijk",
    "Benningbroek",
    "Hauwert",
    "Lambertschaag",
    "Medemblik",
    "Midwoud",
    "Nibbixwoud",
    "Onderdijk",
    "Oostwoud",
    "Opperdoes",
    "Sijbekarspel",
    "Twisk",
    "Wadway",
    "Wervershoof",
    "Wognum",
    "Zwaagdijk-Oost",
    "Zwaagdijk-West",
)

OPMEER_KERNEN = (
    "Opmeer",
    "Hoogwoud",
    "Aartswoud",
    "De Weere",
    "Gouwe",
    "Spanbroek",
    "Wadway",
    "Zandwerven",
)


def _ask(base: str, question: str) -> str:
    r = httpx.post(
        base + "/v1/chat/completions",
        json={
            "model": cfg_model(),
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        timeout=300,
    )
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


def _assert_complete_and_consistent(answers: list[str], expected: tuple[str, ...]) -> None:
    for a in answers:
        low = a.lower()
        missing = [e for e in expected if e.lower() not in low]
        assert not missing, f"missing {missing!r} in answer: {a!r}"
    assert len(set(answers)) == 1, f"answers differ between runs: {answers!r}"


def test_medemblik_places_are_complete_and_consistent(live_proxy):
    question = "uit welke woonplaatsen bestaat de gemeente medemblik"
    answers = [_ask(live_proxy, question) for _ in range(3)]
    for a in answers:
        print(f"\n[live] medemblik: {a[:300]}")
    _assert_complete_and_consistent(answers, MEDEMBLIK_PLACES)


def test_opmeer_kernen_are_complete_and_consistent(live_proxy):
    question = "uit welke kernen bestaat de gemeente Opmeer"
    answers = [_ask(live_proxy, question) for _ in range(2)]
    for a in answers:
        print(f"\n[live] opmeer: {a[:300]}")
    _assert_complete_and_consistent(answers, OPMEER_KERNEN)
