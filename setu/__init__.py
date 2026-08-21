"""
Setu — voice-first government scheme discovery.

Loading .env here, at package import, is deliberate. Every entry point (uvicorn,
pytest, eval/run.py, a bare python -c) imports this package before it touches a
key, so there is exactly one place that has to be right. The alternative --
remembering to call load_dotenv() in each entry point -- fails silently and
looks like an authentication problem rather than a missing call.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env(path: Path = ENV_PATH) -> None:
    """
    Minimal .env reader: KEY=value, # comments, blank lines.

    Hand-rolled rather than pulling in python-dotenv, because this is twelve
    lines and one fewer dependency to install on a machine that has already
    crashed once this week.

    Real environment variables always win, so `SETU_LLM_MODE=offline uvicorn ...`
    overrides the file without editing it -- which is exactly what you want when
    switching the demo to offline mode.
    """
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_load_env()
