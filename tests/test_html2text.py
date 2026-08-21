from proxy.html2text import html_to_sections, html_to_text, parse_sections, truncate


def test_strips_scripts_and_style():
    html = (
        "<html><head><style>a{color:red}</style></head>"
        "<body><script>x=1;</script><p>Hello</p></body></html>"
    )
    assert html_to_text(html) == "Hello"


def test_block_tags_produce_newlines():
    out = html_to_text("<h1>Title</h1><p>Body text</p>")
    assert "Title" in out and "Body text" in out and "\n" in out


def test_unescapes_entities():
    assert "AT&T" in html_to_text("<p>AT&amp;T</p>")


def test_bytes_input():
    assert html_to_text(b"<p>Hi</p>") == "Hi"


def test_empty_input():
    assert html_to_text("") == ""
    assert html_to_text(b"") == ""


def test_malformed_html_does_not_crash():
    assert "abc" in html_to_text("<div>abc <p>unclosed")


def test_truncate_short():
    assert truncate("short", 100) == "short"


def test_truncate_respects_limit():
    out = truncate("aaaa bbbb cccc dddd", 9)
    assert len(out) <= 9


def test_truncate_prefers_word_boundary():
    out = truncate("one two three four", 10)
    assert not out.endswith(" ")
    assert out.startswith("one")


def test_sidebar_table_removed():
    html = (
        '<table class="sidebar sidebar-collapse nomobile" role="navigation"><tbody>'
        "<tr><td>Part of a series on the</td></tr>"
        "<tr><td>United States space program</td></tr></tbody></table>"
        "<p>Real content here.</p>"
    )
    out = html_to_text(html)
    assert "Real content here." in out
    assert "series" not in out
    assert "space program" not in out


def test_navbox_removed_by_role():
    html = '<table role="navigation"><tr><td>Nav junk</td></tr></table><p>Keep me</p>'
    out = html_to_text(html)
    assert "Nav junk" not in out
    assert "Keep me" in out


def test_regular_table_content_kept():
    html = "<table><tr><td>Neil Armstrong</td><td>Apollo 11</td></tr></table>"
    out = html_to_text(html)
    assert "Neil Armstrong" in out
    assert "Apollo 11" in out


def test_edit_and_citation_markers_removed():
    html = "<h2>Section [edit]</h2><p>Fact one.[1] Fact two.[citation needed]</p>"
    out = html_to_text(html)
    assert "[edit]" not in out
    assert "[1]" not in out
    assert "citation needed" not in out
    assert "Fact one." in out
    assert "Fact two." in out


def test_parse_sections_splits_by_heading():
    html = "<h1>Title</h1><p>Lead.</p><h2>People</h2><ul><li>Ada</li><li>Grace</li></ul>"
    secs = parse_sections(html)
    assert secs[0][0] == "Title"
    people = [s for s in secs if s[0] == "People"]
    assert people
    kinds = [k for k, _ in people[0][1]]
    assert kinds == ["li", "li"]
    assert [t for _, t in people[0][1]] == ["Ada", "Grace"]


def test_html_to_sections_plain_view():
    html = "<h1>T</h1><p>Body</p>"
    secs = html_to_sections(html)
    # Content after a heading belongs to that heading's section.
    assert secs == [("T", ["Body"])]


def test_malformed_html_still_returns_text():
    out = html_to_text("<div>abc <p>unclosed <table class='sidebar'>junk")
    assert "abc" in out
    assert "junk" not in out
