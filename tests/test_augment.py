from proxy.augment import augment_messages, build_help_text
from proxy.config import Config, HelpPromptCfg
from proxy.kiwix import Article


def _cfg(position: str) -> Config:
    return Config(
        help_prompt=HelpPromptCfg(
            template="Facts: {facts}\nArticles:\n{articles}", position=position
        )
    )


def _arts():
    return [Article(title="Albert Einstein", text="Physicist.", path="Albert_Einstein")]


def test_system_position_prepends_system_message():
    cfg = _cfg("system")
    msgs = [{"role": "user", "content": "hi"}]
    out = augment_messages(cfg, msgs, ["f1"], _arts())
    assert out[0]["role"] == "system"
    assert "f1" in out[0]["content"]
    assert "Albert Einstein" in out[0]["content"]
    assert out[1] == msgs[0]


def test_user_prefix_prepends_to_first_user_message():
    cfg = _cfg("user_prefix")
    msgs = [{"role": "system", "content": "orig"}, {"role": "user", "content": "hi"}]
    out = augment_messages(cfg, msgs, ["f1"], _arts())
    assert out[0] == msgs[0]  # leading system untouched
    assert out[1]["content"].startswith("Facts: f1")
    assert out[1]["content"].endswith("hi")


def test_user_prefix_with_multimodal_content():
    cfg = _cfg("user_prefix")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
    out = augment_messages(cfg, msgs, ["f1"], _arts())
    first = out[0]["content"][0]
    assert first["type"] == "text"
    assert "f1" in first["text"]
    assert out[0]["content"][1]["text"] == "look"


def test_no_user_message_falls_back_to_system():
    cfg = _cfg("user_prefix")
    out = augment_messages(cfg, [{"role": "assistant", "content": "x"}], ["f1"], _arts())
    assert out[0]["role"] == "system"


def test_build_help_text_handles_empty():
    cfg = _cfg("system")
    text = build_help_text(cfg, [], [])
    assert "(none)" in text
    assert "(no relevant articles found)" in text


def test_default_template_instructs_list_answers():
    cfg = Config()
    assert "list" in cfg.help_prompt.template.lower()
    assert "{facts}" in cfg.help_prompt.template
    assert "{articles}" in cfg.help_prompt.template


def test_build_help_text_renders_articles():
    cfg = Config()
    arts = [Article(title="List of Apollo astronauts", text="Neil Armstrong\nBuzz Aldrin")]
    text = build_help_text(cfg, ["Apollo astronauts"], arts)
    assert "List of Apollo astronauts" in text
    assert "Neil Armstrong" in text
    assert "Apollo astronauts" in text
