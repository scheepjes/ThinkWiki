import pytest
from conftest import ZIM_PATH

from proxy.zim import Zim, ZimCollection


@pytest.fixture(scope="module")
def zim():
    z = Zim(ZIM_PATH)
    yield z
    z.close()


@pytest.fixture(scope="module")
def collection():
    # The same archive opened twice stands in for two distinct ZIM files.
    c = ZimCollection([ZIM_PATH, ZIM_PATH])
    yield c
    c.close()


def test_open_reports_counts(zim):
    assert zim.article_count > 0
    assert zim.entry_count >= zim.article_count


def test_has_fulltext_index(zim):
    assert zim.has_fulltext_index is True


def test_metadata(zim):
    title = zim.metadata("Title")
    assert title


def test_search_returns_results(zim):
    results = zim.search("Albert Einstein", 3)
    assert results
    for r in results:
        assert "title" in r
        assert "path" in r
        assert "score" in r


def test_search_empty_query(zim):
    assert zim.search("", 5) == []


def test_get_article_by_title(zim):
    a = zim.get_article(title="Albert Einstein")
    assert a is not None
    assert a["title"] == "Albert Einstein"
    assert a["mimetype"] == "text/html"
    assert a["path"]
    assert len(a["content"]) > 1000


def test_get_article_by_path(zim):
    a = zim.get_article(path="Albert_Einstein")
    assert a is not None
    assert a["title"] == "Albert Einstein"


def test_get_article_follows_redirect_stub(zim):
    """Redirect stubs (meta-refresh HTML) resolve to the target article."""
    a = zim.get_article(title="List of people who have walked on the Moon")
    assert a is not None
    # The stub redirects to the Apollo astronauts article.
    assert a["path"] == "List_of_Apollo_astronauts"
    assert len(a["content"]) > 10000


def test_get_article_missing_returns_none(zim):
    assert zim.get_article(title="This Article Does Not Exist 98765") is None


def test_get_article_requires_path_or_title(zim):
    with pytest.raises(ValueError):
        zim.get_article()


# --- ZimCollection (multiple archives) -------------------------------------


def test_collection_counts_are_summed(collection, zim):
    assert collection.article_count == 2 * zim.article_count
    assert collection.entry_count == 2 * zim.entry_count
    assert collection.has_fulltext_index is True


def test_collection_names_lists_each_archive(collection):
    assert len(collection.names) == 2
    assert all(n for n in collection.names)


def test_collection_search_tags_source_and_merges(collection):
    results = collection.search("Albert Einstein", 3)
    assert results
    for r in results:
        assert r["source"] in collection.names
    # Merged results are de-duplicated by normalized title.
    titles = [" ".join(r["title"].lower().split()) for r in results]
    assert len(titles) == len(set(titles))


def test_collection_search_sorted_by_score(collection):
    results = collection.search("Albert Einstein", 5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_collection_search_empty_query(collection):
    assert collection.search("", 5) == []


def test_collection_get_article_by_title_has_source(collection):
    a = collection.get_article(title="Albert Einstein")
    assert a is not None
    assert a["title"] == "Albert Einstein"
    assert a["source"] in collection.names


def test_collection_get_article_by_path(collection):
    a = collection.get_article(path="Albert_Einstein")
    assert a is not None
    assert a["title"] == "Albert Einstein"


def test_collection_get_article_source_prefers_that_archive(collection):
    # A path resolved via a specific archive should be fetched from there.
    a = collection.get_article(path="Albert_Einstein", source=collection.names[0])
    assert a is not None
    assert a["source"] == collection.names[0]


def test_collection_get_article_missing_returns_none(collection):
    assert collection.get_article(title="This Article Does Not Exist 98765") is None


def test_collection_empty_paths_raise():
    with pytest.raises(ValueError):
        ZimCollection([])


def test_collection_all_bad_paths_raise(tmp_path):
    with pytest.raises(RuntimeError):
        ZimCollection([str(tmp_path / "nope.zim")])


def test_collection_skips_unopenable_files(collection, tmp_path):
    bad = str(tmp_path / "missing.zim")
    c = ZimCollection([bad, ZIM_PATH])
    try:
        assert c.article_count == collection.article_count // 2
        assert c.get_article(title="Albert Einstein") is not None
    finally:
        c.close()
