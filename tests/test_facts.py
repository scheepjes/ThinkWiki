from proxy.facts import extract_json_array


def test_plain_array():
    assert extract_json_array('["a", "b"]') == ["a", "b"]


def test_fenced_json():
    assert extract_json_array('```json\n["a", "b"]\n```') == ["a", "b"]


def test_prose_around_array():
    assert extract_json_array('Sure! Here you go: ["Paris", "France"] Hope that helps.') == [
        "Paris",
        "France",
    ]


def test_no_array_returns_empty():
    assert extract_json_array("no array here, just prose") == []


def test_dedup_is_case_insensitive_and_drops_empty():
    assert extract_json_array('["A", "a", "B", ""]') == ["A", "B"]


def test_dict_entries_use_query_or_text():
    assert extract_json_array('[{"query": "x"}, {"text": "y"}, {"fact": "z"}]') == [
        "x",
        "y",
        "z",
    ]


def test_empty_and_none():
    assert extract_json_array("") == []
    assert extract_json_array(None) == []


def test_non_list_json_ignored():
    assert extract_json_array('{"a": 1}') == []
