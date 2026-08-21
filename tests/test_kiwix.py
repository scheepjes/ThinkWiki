from conftest import ZIM_PATH

from proxy.config import Config, KiwixCfg
from proxy.kiwix import KiwixLookup, normalize_query


def _cfg(**overrides) -> Config:
    base: dict = {
        "zim_path": ZIM_PATH,
        "max_articles_per_fact": 2,
        "max_chars_per_article": 400,
        "total_char_budget": 2000,
        "cache_size": 8,
    }
    base.update(overrides)
    return Config(kiwix=KiwixCfg(**base))


def test_normalize_query():
    assert normalize_query("  Hello   World  ") == "hello world"
    assert normalize_query("") == ""


def test_lookup_returns_articles():
    k = KiwixLookup(_cfg())
    try:
        arts = k.lookup(["Albert Einstein"])
        assert arts
        assert arts[0].title == "Albert Einstein"
        for a in arts:
            assert len(a.text) <= 400
    finally:
        k.close()


def test_dedup_across_repeated_facts():
    k = KiwixLookup(_cfg())
    try:
        arts = k.lookup(["Albert Einstein", "Albert Einstein"])
        titles = [a.title for a in arts]
        assert len(titles) == len(set(titles))
    finally:
        k.close()


def test_dedup_across_related_facts():
    k = KiwixLookup(_cfg())
    try:
        arts = k.lookup(["Albert Einstein", "General relativity"])
        paths = [a.path for a in arts]
        assert len(paths) == len(set(paths))
    finally:
        k.close()


def test_total_char_budget_respected():
    k = KiwixLookup(_cfg(total_char_budget=800))
    try:
        arts = k.lookup(["Albert Einstein", "General relativity", "Paris"])
        assert len(arts) >= 1
        total = sum(len(a.text) for a in arts)
        # At most one article may overshoot the budget (the first one).
        assert total <= 800 + 400
    finally:
        k.close()


def test_cache_returns_consistent_results():
    k = KiwixLookup(_cfg())
    try:
        first = k.lookup(["Albert Einstein"])
        second = k.lookup(["Albert Einstein"])
        assert [a.title for a in first] == [a.title for a in second]
    finally:
        k.close()


def test_empty_queries_yield_nothing():
    k = KiwixLookup(_cfg())
    try:
        assert k.lookup([]) == []
        assert k.lookup(["", "   "]) == []
    finally:
        k.close()


def test_list_query_recovers_named_list():
    """The crew list of a 'List of ...' article must survive the char budget.

    A blind prefix cut keeps only the intro; query-aware selection must reach
    the actual list of astronauts.
    """
    k = KiwixLookup(_cfg(max_chars_per_article=4000, total_char_budget=12000))
    try:
        arts = k.lookup(["Apollo astronauts"])
        assert arts
        top = arts[0]
        assert top.title == "List of Apollo astronauts"
        assert len(top.text) <= 4000
        # Concrete names from the Moon-walkers table, not just the lead
        # paragraph. (The table lists the twelve men who walked on the Moon.)
        for name in (
            "Neil Armstrong",
            "Aldrin",
            "Pete Conrad",
            "Alan Bean",
            "Alan Shepard",
            "Harrison Schmitt",
        ):
            assert name in top.text, f"missing {name!r}"
    finally:
        k.close()


def test_list_query_keeps_lead_context():
    k = KiwixLookup(_cfg(max_chars_per_article=4000, total_char_budget=12000))
    try:
        arts = k.lookup(["Apollo astronauts"])
        assert arts
        # The lead paragraph (context) is still present alongside the list.
        assert "Apollo program" in arts[0].text
    finally:
        k.close()


def test_list_query_budget_is_hard_capped():
    k = KiwixLookup(_cfg(max_chars_per_article=1500, total_char_budget=3000))
    try:
        arts = k.lookup(["Apollo astronauts"])
        assert arts
        for a in arts:
            assert len(a.text) <= 1500
    finally:
        k.close()


def test_regular_query_still_works_with_small_budget():
    k = KiwixLookup(_cfg(max_chars_per_article=400, total_char_budget=800))
    try:
        arts = k.lookup(["Albert Einstein"])
        assert arts
        assert "physicist" in arts[0].text.lower() or "Einstein" in arts[0].text
    finally:
        k.close()
