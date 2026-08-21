from proxy.lru import LRUCache


def test_basic_get_set():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert c.get("missing") is None


def test_eviction_of_least_recently_used():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_access_refreshes_recency():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")  # "a" becomes most recently used
    c.set("c", 3)  # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3


def test_len_and_clear():
    c = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    assert len(c) == 2
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None


def test_capacity_floor_is_one():
    c = LRUCache(0)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") is None
    assert c.get("b") == 2
