from pathlib import Path

from conftest import ZIM_PATH

from proxy.config import Config, KiwixCfg
from proxy.kiwix import KiwixLookup, normalize_query, resolve_zim_paths


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


# --- multi-ZIM directory support -------------------------------------------


def _zim_dir(tmp_path, names=("aaa_en.zim", "bbb_fr.zim")):
    """A temp dir holding symlinks to the real ZIM under the given names."""
    d = tmp_path / "zims"
    d.mkdir()
    real = Path(ZIM_PATH).resolve()
    for n in names:
        (d / n).symlink_to(real)
    (d / "notes.txt").write_text("not a zim")
    return d


def test_resolve_zim_paths_from_dir(tmp_path):
    d = _zim_dir(tmp_path)
    paths = resolve_zim_paths(KiwixCfg(zim_dir=str(d)))
    assert [Path(p).name for p in paths] == ["aaa_en.zim", "bbb_fr.zim"]


def test_resolve_zim_paths_dir_sorted_and_filtered(tmp_path):
    d = _zim_dir(tmp_path, names=("zeta.zim", "alpha.zim"))
    paths = resolve_zim_paths(KiwixCfg(zim_dir=str(d)))
    assert [Path(p).name for p in paths] == ["alpha.zim", "zeta.zim"]


def test_resolve_zim_paths_dir_plus_single_file(tmp_path):
    d = _zim_dir(tmp_path, names=("aaa_en.zim",))
    paths = resolve_zim_paths(KiwixCfg(zim_dir=str(d), zim_path=ZIM_PATH))
    assert [Path(p).name for p in paths] == ["aaa_en.zim", Path(ZIM_PATH).name]


def test_resolve_zim_paths_single_file_only():
    assert resolve_zim_paths(KiwixCfg(zim_path=ZIM_PATH)) == [ZIM_PATH]


def test_resolve_zim_paths_empty_when_unset():
    assert resolve_zim_paths(KiwixCfg()) == []


def test_resolve_zim_paths_missing_dir_falls_back_to_file(tmp_path):
    paths = resolve_zim_paths(KiwixCfg(zim_dir=str(tmp_path / "nope"), zim_path=ZIM_PATH))
    assert paths == [ZIM_PATH]


def test_lookup_via_zim_dir_tags_source(tmp_path):
    d = _zim_dir(tmp_path, names=("wiki.zim",))
    k = KiwixLookup(Config(kiwix=KiwixCfg(zim_dir=str(d))))
    try:
        arts = k.lookup(["Albert Einstein"])
        assert arts
        assert all(a.source == "wiki.zim" for a in arts)
    finally:
        k.close()


def test_lookup_across_two_archives_dedups_by_title(tmp_path):
    """The same article present in two archives must be counted only once."""
    d = _zim_dir(tmp_path)
    k = KiwixLookup(
        Config(
            kiwix=KiwixCfg(
                zim_dir=str(d),
                max_articles_per_fact=3,
                max_chars_per_article=400,
                total_char_budget=2000,
                cache_size=8,
            )
        )
    )
    try:
        arts = k.lookup(["Albert Einstein"])
        # Two identical archives would yield 6 candidate hits; title de-dup
        # must collapse them to the 3 distinct articles.
        assert len(arts) == 3
        titles = [normalize_query(a.title) for a in arts]
        assert len(titles) == len(set(titles))
        assert all(a.source for a in arts)
    finally:
        k.close()
