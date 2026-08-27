"""Unit tests for query-relevance-aware article text selection."""

from proxy.relevance import select_article_text

# Synthetic article: an intro, a long prose section, a bulleted list of
# settlements (the payload of list-type questions), and a references list
# that must NOT be treated as content.
SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Medemblik (gemeente)",
        [
            ("p", "Medemblik is a municipality in the province of North Holland."),
            ("li", "Mayor: someone"),
            ("li", "Area: 257 km2"),
        ],
    ),
    (
        "Geschiedenis",
        [("p", "The municipality has a long history. " * 60)],
    ),
    (
        "Kernen binnen de gemeente",
        [
            ("p", "De gemeente omvat de volgende kernen:"),
            ("li", "Abbekerk"),
            ("li", "Andijk"),
            ("li", "Benningbroek"),
            ("li", "Hauwert"),
            ("li", "Lambertschaag"),
            ("li", "Twisk"),
            ("li", "Wervershoof"),
        ],
    ),
    (
        "Referenties",
        [
            ("li", "Source one."),
            ("li", "Source two."),
            ("li", "Source three."),
            ("li", "Source four."),
            ("li", "Source five."),
        ],
    ),
]


def test_list_section_kept_even_when_unscored():
    """A bulleted list section must survive the budget even when the query
    terms do not occur in it (e.g. 'Medemblik' vs 'Kernen binnen de
    gemeente')."""
    text = select_article_text(SECTIONS, "Medemblik", 1200)
    for place in ("Abbekerk", "Andijk", "Benningbroek", "Hauwert", "Lambertschaag"):
        assert place in text, f"missing {place!r}"


def test_references_list_not_treated_as_content():
    text = select_article_text(SECTIONS, "Medemblik", 600)
    assert "Source one." not in text


def test_budget_still_hard_capped():
    text = select_article_text(SECTIONS, "Medemblik", 500)
    assert len(text) <= 500


def test_scored_sections_still_ranked_first_after_lists():
    """Query terms matching a prose section must still pull it in."""
    text = select_article_text(SECTIONS, "history municipality", 2000)
    assert "long history" in text.lower()


def test_no_terms_falls_back_to_prefix():
    text = select_article_text(SECTIONS, "the", 200)
    assert "Medemblik is a municipality" in text
