"""Entry point: ``python -m proxy``."""

from __future__ import annotations

import uvicorn

from .config import load_config
from .server import create_app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=(cfg.logging.level or "info").lower(),
    )


if __name__ == "__main__":
    main()
