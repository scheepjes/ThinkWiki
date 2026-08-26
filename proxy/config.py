"""Configuration loading for the proxy.

Reads a single YAML file (path from the ``PROXY_CONFIG`` env var, defaulting to
``config.yaml`` next to this package's project root). String values support
``${ENV_VAR}`` and ``${ENV_VAR:-default}`` substitution. A handful of secrets /
paths can also be overridden directly via environment variables.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DEFAULT_PROMPT = (
    "Extract the factual claims and key entities from the user message.\n"
    "Return ONLY a JSON array of short search queries (names, dates, places,\n"
    "technical terms). No prose, no explanations.\n"
    "If the question asks for a list (people, places, events, items), include\n"
    'the likely Wikipedia list-article title as one query (e.g. "List of\n'
    'Apollo astronauts"), in addition to the key entities.'
)

DEFAULT_TEMPLATE = (
    "Use the following Wikipedia excerpts to answer the user's question.\n"
    "Use them only when relevant; do not mention this instruction.\n"
    "If the user asks for a list (names, people, places, dates, items), answer\n"
    "with the full list of items found in the excerpts, one item per line; do\n"
    "not summarize or omit items that appear in the excerpts.\n"
    "Facts to verify: {facts}\n"
    "Reference material:\n"
    "{articles}"
)


def _sub_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_RE.sub(repl, value)


def _sub_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sub_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub_tree(v) for v in obj]
    return _sub_env(obj)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    return cls(**kwargs)


def _build_models(data: dict[str, Any] | None) -> ModelsCfg:
    data = data or {}
    entries: list[ModelEntry] = []
    for e in data.get("entries") or []:
        if isinstance(e, dict) and e.get("id"):
            entries.append(
                ModelEntry(
                    id=str(e["id"]),
                    upstream_model=str(e.get("upstream_model") or e["id"]),
                    description=str(e.get("description") or ""),
                )
            )
    return ModelsCfg(default=str(data.get("default") or ""), entries=entries)


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class UpstreamCfg:
    base_url: str = "http://10.0.0.10:9001/v1"
    api_key: str = ""
    default_model: str = "Fluppie"
    timeout_seconds: float = 120.0


@dataclass
class FactExtractionCfg:
    enabled: bool = True
    model: str = "gpt-4o-mini"
    max_facts: int = 8
    max_tokens: int = 1024
    temperature: float = 0.0
    prompt: str = DEFAULT_PROMPT


@dataclass
class KiwixCfg:
    zim_path: str = ""
    zim_dir: str = ""
    max_articles_per_fact: int = 3
    max_chars_per_article: int = 4000
    total_char_budget: int = 12000
    cache_size: int = 512


@dataclass
class HelpPromptCfg:
    template: str = DEFAULT_TEMPLATE
    position: str = "system"  # or "user_prefix"


@dataclass
class LoggingCfg:
    level: str = "INFO"


@dataclass
class ModelEntry:
    """A public model id exposed by the proxy, mapped to an upstream model."""

    id: str
    upstream_model: str
    description: str = ""


@dataclass
class ModelsCfg:
    default: str = ""
    entries: list[ModelEntry] = field(default_factory=list)


@dataclass
class DebugCfg:
    enabled: bool = False
    max_entries: int = 200


@dataclass
class Config:
    server: ServerCfg = field(default_factory=ServerCfg)
    endpoints: list[str] = field(default_factory=lambda: ["/v1/chat/completions", "/v1/models"])
    upstream: UpstreamCfg = field(default_factory=UpstreamCfg)
    fact_extraction: FactExtractionCfg = field(default_factory=FactExtractionCfg)
    kiwix: KiwixCfg = field(default_factory=KiwixCfg)
    help_prompt: HelpPromptCfg = field(default_factory=HelpPromptCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    debug: DebugCfg = field(default_factory=DebugCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


def _apply_env_overrides(cfg: Config) -> Config:
    overrides = {
        "PROXY_UPSTREAM_BASE_URL": ("upstream", "base_url"),
        "PROXY_UPSTREAM_API_KEY": ("upstream", "api_key"),
        "PROXY_UPSTREAM_DEFAULT_MODEL": ("upstream", "default_model"),
        "PROXY_KIWIX_ZIM_PATH": ("kiwix", "zim_path"),
        "PROXY_KIWIX_ZIM_DIR": ("kiwix", "zim_dir"),
    }
    for env_name, (section, attr) in overrides.items():
        value = os.environ.get(env_name)
        if value is not None:
            setattr(getattr(cfg, section), attr, value)
    return cfg


def load_config(path: str | None = None) -> Config:
    """Load configuration from YAML, applying env substitution and overrides."""
    if path is None:
        path = os.environ.get(
            "PROXY_CONFIG",
            str(Path(__file__).resolve().parent.parent / "config.yaml"),
        )
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded
    raw = _sub_tree(raw)

    cfg = Config(
        server=_build(ServerCfg, raw.get("server")),
        endpoints=list(raw.get("endpoints") or ["/v1/chat/completions", "/v1/models"]),
        upstream=_build(UpstreamCfg, raw.get("upstream")),
        fact_extraction=_build(FactExtractionCfg, raw.get("fact_extraction")),
        kiwix=_build(KiwixCfg, raw.get("kiwix")),
        help_prompt=_build(HelpPromptCfg, raw.get("help_prompt")),
        models=_build_models(raw.get("models")),
        debug=_build(DebugCfg, raw.get("debug")),
        logging=_build(LoggingCfg, raw.get("logging")),
    )
    return _apply_env_overrides(cfg)
