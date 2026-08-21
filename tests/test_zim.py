import pytest
from conftest import ZIM_PATH

from proxy.zim import Zim


@pytest.fixture(scope="module")
def zim():
    z = Zim(ZIM_PATH)
    yield z
    z.close()


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
