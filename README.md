# ThinkWiki

Enrich AI queries with Wikipedia data.

An OpenAI-compatible proxy server that grounds LLM answers in a local
Wikipedia archive. It sits between a client and an upstream LLM API: before
forwarding a request it extracts factual claims from the user's message, looks
them up in a local [ZIM](https://openzim.org/) archive (Wikipedia via Kiwix),
and injects the relevant articles into the prompt so the model answers grounded
in retrieved content.

```
client ──▶ proxy ──▶ upstream LLM (http://localhost:9001/v1)
              │
              ├─ 1. extract facts (LLM call)
              ├─ 2. look up articles (local ZIM / libzim)
              └─ 3. augment prompt, forward, stream/return response
```

If fact extraction or lookup fails for any reason, the request is forwarded
**unchanged** — enrichment is best-effort and never blocks the request path.

## Features

- Drop-in for the OpenAI **Chat Completions** API (`POST /v1/chat/completions`)
  and `GET /v1/models`.
- Supports both `stream: false` (JSON) and `stream: true` (SSE passthrough,
  byte-for-byte).
- **Model catalog**: expose friendly model ids (e.g. `WikiGemma`) that are
  mapped to the real upstream model. `GET /v1/models` serves the catalog and a
  request's `model` is translated before forwarding.
- Defensive fact extraction (strict JSON array of short queries). For
  list-type questions the extractor is prompted to also emit the likely
  Wikipedia list-article title (e.g. `List of Apollo astronauts`), which then
  hits the archive by exact title.
- **Multi-ZIM**: point `kiwix.zim_dir` at a directory and every `*.zim` file in
  it is opened and searched. Results from all archives are merged (best score
  first) and de-duplicated by article title across archives, so the same
  article in two ZIMs (e.g. two languages) is used once. Each retrieved
  article records which archive it came from (visible in debug entries).
- ZIM lookup with exact-title-first ranking, cross-fact de-duplication, a global
  character budget, and an in-memory LRU cache keyed by normalized query.
- **Query-relevance-aware article extraction**: each article is split into
  sections, scored against the query, and the character budget is filled with
  the intro plus the most relevant sections. List/table rows (names, dates,
  cells) are kept first and individually bounded, so a list of names survives
  truncation instead of being cut off after the lead paragraph. Navigation
  chrome (sidebars, navboxes, `[edit]`/citation markers) is stripped. ZIM
  redirect stubs are followed transparently.
- **Debug mode**: capture every request/response exchange (client query,
  extracted facts, retrieved articles, the augmented upstream prompt, and the
  response) in a bounded in-memory buffer, browsable via `GET /debug/requests`.
- Single YAML config with `${ENV_VAR}` / `${ENV_VAR:-default}` substitution and
  env-var overrides for secrets.

## Project layout

The repository root is also a Python virtualenv, so the interpreter and
dependencies live in `bin/` and `lib/`.

```
├── DESIGN.txt            # original design spec
├── config.yaml           # runtime configuration
├── requirements.txt      # Python dependencies
├── ruff.toml             # linter config
├── mypy.ini              # type-checker config
├── zims/                 # every *.zim here is opened and searched
│   └── wikipedia_en_top1m_nopic_2026-04.zim   # local Wikipedia archive (~16 GB)
├── native/
│   ├── zim_wrapper.cpp   # C interface over the libzim C++ API
│   ├── build.sh          # builds libzim_wrapper.so
│   └── libzim_wrapper.so # compiled shared library
├── proxy/                # the application package
│   ├── __main__.py       # entry point (python -m proxy)
│   ├── server.py         # FastAPI app + request pipeline
│   ├── config.py         # YAML config loading + env overrides
│   ├── facts.py          # LLM fact extraction + defensive JSON parsing
│   ├── kiwix.py          # ZIM lookup orchestration (dedup, budget, cache)
│   ├── zim.py            # ctypes binding to libzim_wrapper.so (+ redirect follow)
│   ├── html2text.py      # stdlib HTML -> plain text / structured sections
│   ├── relevance.py      # query-relevance-aware section selection
│   ├── augment.py        # prompt augmentation (system / user-prefix)
│   ├── debug.py          # bounded in-memory request/response capture
│   └── lru.py            # tiny thread-safe LRU cache
└── tests/                # pytest suite (unit + integration + live)
```

## Prerequisites

- Python **3.11+**
- A C/C++ compiler (`g++`)
- The **libzim** runtime and development headers:
  - Debian/Ubuntu: `sudo apt-get install libzim-dev`
  - (provides `libzim.so`, the `zim/*.h` headers, and `pkg-config` entries)

> **Why a custom wrapper?** The design originally called for `pykiwik`, but it
> is not published on PyPI. Instead, a small C++ wrapper exposes a C interface
> over the system `libzim` C++ API and is bound from Python with `ctypes` —
> no extra Python packages required for ZIM access.

## Setup

```bash
# 1. Install Python dependencies into the venv
bin/pip install -r requirements.txt

# 2. Build the native libzim wrapper
bash native/build.sh        # -> native/libzim_wrapper.so
```

## Configuration

Configuration is a single YAML file. Its path comes from the `PROXY_CONFIG`
environment variable, defaulting to `config.yaml` in the project root.

A working `config.yaml` is included, already pointed at the local LLM and the
local ZIM. Key sections:

| Section           | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `server`          | Bind host/port for the proxy.                                  |
| `endpoints`       | Which routes the proxy exposes/intercepts.                     |
| `upstream`        | Upstream LLM `base_url`, `api_key`, `default_model`, timeout.  |
| `fact_extraction` | Toggle, extractor model, `max_facts`, and the extraction prompt. |
| `kiwix`           | `zim_dir` (dir of `.zim` files) and/or `zim_path` (single file), per-fact/per-article/total char limits, cache size. |
| `help_prompt`     | The grounding template (`{facts}`, `{articles}`) and `position`. |
| `models`          | Public model catalog: `default` + `entries` (id → upstream model). |
| `debug`           | Toggle request/response capture + `max_entries` buffer size.     |
| `logging`         | Log level.                                                     |

### ZIM archives (single file or directory)

`kiwix` locates its content two ways (either or both):

- `zim_dir` — a directory; **every `*.zim` file in it is opened and searched**.
  Drop new archives into the directory and restart to use them. Files are
  processed in sorted file-name order; a file that fails to open is skipped
  with a warning (the rest still work).
- `zim_path` — a single `.zim` file (the original behavior).

When both are set, the directory's files are used and the single file is added
if it is not already among them. Searches run against all archives and the
results are merged (highest score first) and de-duplicated by article title, so
an article present in several ZIMs is injected once. Each retrieved article
records its source archive (shown in `GET /debug/requests`).

`help_prompt.position` is either `system` (insert an additional system message)
or `user_prefix` (prepend to the first user message).

`models.entries` maps each public `id` to the `upstream_model` actually called.
When a catalog is present, `GET /v1/models` returns those ids and a chat
request's `model` is translated (unknown/absent ids fall back to `models.default`).
With no `models` section the legacy behavior applies: the requested model is
passed through unchanged and `GET /v1/models` forwards to the upstream.

### Environment overrides

Secrets and paths can be set via environment variables (they take precedence
over the YAML values):

- `PROXY_CONFIG` — path to the YAML config file
- `PROXY_UPSTREAM_BASE_URL`
- `PROXY_UPSTREAM_API_KEY`
- `PROXY_UPSTREAM_DEFAULT_MODEL`
- `PROXY_KIWIX_ZIM_PATH`
- `PROXY_KIWIX_ZIM_DIR`

Any string value in the YAML also supports `${ENV_VAR}` and
`${ENV_VAR:-default}` substitution.

## Running

```bash
bin/python -m proxy
```

The server binds to `server.host:server.port` (default `127.0.0.1:8050`) and
opens the ZIM at startup.

### Example: non-streaming

```bash
curl -s http://127.0.0.1:8050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "WikiGemma",
    "messages": [{"role": "user", "content": "Who was Albert Einstein?"}],
    "stream": false
  }'
```

### Example: streaming (SSE)

```bash
curl -sN http://127.0.0.1:8050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "WikiGemma",
    "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
    "stream": true
  }'
```

### Using an OpenAI client

Point any OpenAI SDK at the proxy:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8050/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="WikiGemma",
    messages=[{"role": "user", "content": "Who was Albert Einstein?"}],
)
print(resp.choices[0].message.content)
```

### Debug mode

When `debug.enabled` is `true`, every exchange is captured in a bounded
in-memory buffer (newest first). Inspect it with:

```bash
# list all captured exchanges
curl -s http://127.0.0.1:8050/debug/requests

# fetch one entry by id
curl -s http://127.0.0.1:8050/debug/requests/1

# clear the buffer
curl -s -X DELETE http://127.0.0.1:8050/debug/requests
```

Each entry records the client request, the requested vs. upstream model, the
extracted `facts`, the retrieved `articles`, the full augmented
`upstream_request` (the grounding prompt), and the `response` (for streams the
accumulated content + chunk count). These endpoints return `404` when debug is
disabled.

## Testing

```bash
bin/python -m pytest tests/ -q
```

The suite has three tiers:

- **Unit** — config, HTML→text, LRU, fact parsing, augmentation, and the ZIM
  binding / lookup orchestration (these open the real ZIM, which opens in ~20 ms).
- **Integration** — runs the proxy as a real server against a local mock
  OpenAI upstream, verifying the full path (extraction → lookup → augmentation
  → passthrough) for JSON and SSE, plus the no-enrichment path.
- **Live** — end-to-end against the real local LLM at `http://localhost:9001/v1`.
  Skipped automatically if that endpoint is unreachable.

## Lint & typecheck

```bash
bin/ruff check proxy/ tests/
bin/ruff format --check proxy/ tests/
bin/mypy proxy/ tests/
```

## Notes

- ZIM content comes from `kiwix.zim_dir` (all `*.zim` in it) and/or
  `kiwix.zim_path`. If no archive can be opened, the proxy still starts and
  serves requests, but enrichment is disabled (check `GET /health` →
  `enrichment`).
- The upstream is any OpenAI-compatible API; the included config targets a
  local llama.cpp server. Set `upstream.api_key` for providers that require one.
- The debug buffer is in-memory only and bounded by `debug.max_entries` (oldest
  entries are evicted); it is cleared on restart. It is intended for local
  inspection, not as a durable audit log.
