import textwrap

from proxy.config import load_config


def test_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET", "s3cret")
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            upstream:
              base_url: http://x/v1
              api_key: ${MY_SECRET}
              default_model: m
            kiwix:
              zim_path: ${MISSING_VAR:-fallback}
            """
        )
    )
    cfg = load_config(str(p))
    assert cfg.upstream.api_key == "s3cret"
    assert cfg.kiwix.zim_path == "fallback"


def test_env_override(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("upstream:\n  base_url: http://orig/v1\n  default_model: orig\n")
    monkeypatch.setenv("PROXY_UPSTREAM_BASE_URL", "http://override/v1")
    monkeypatch.setenv("PROXY_UPSTREAM_DEFAULT_MODEL", "ovr-model")
    cfg = load_config(str(p))
    assert cfg.upstream.base_url == "http://override/v1"
    assert cfg.upstream.default_model == "ovr-model"


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(str(tmp_path / "nope.yaml"))
    assert cfg.server.port == 8050
    assert cfg.upstream.base_url == "http://10.0.0.10:9001/v1"
    assert cfg.fact_extraction.enabled is True
    assert "/v1/chat/completions" in cfg.endpoints
    assert cfg.kiwix.zim_path == ""
    assert cfg.models.entries == []
    assert cfg.debug.enabled is False


def test_partial_config_fills_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server:\n  port: 9999\n")
    cfg = load_config(str(p))
    assert cfg.server.port == 9999
    assert cfg.server.host == "0.0.0.0"
    assert cfg.upstream.default_model == "Fluppie"


def test_models_and_debug_parsed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              default: WikiGemma
              entries:
                - id: WikiGemma
                  upstream_model: real-model
                  description: grounded
            debug:
              enabled: true
              max_entries: 42
            """
        )
    )
    cfg = load_config(str(p))
    assert cfg.models.default == "WikiGemma"
    assert len(cfg.models.entries) == 1
    assert cfg.models.entries[0].id == "WikiGemma"
    assert cfg.models.entries[0].upstream_model == "real-model"
    assert cfg.models.entries[0].description == "grounded"
    assert cfg.debug.enabled is True
    assert cfg.debug.max_entries == 42
