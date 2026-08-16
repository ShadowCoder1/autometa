"""Environment + model configuration.

The API key lives in the repo-root `.env` (never printed, never logged). Call `load_env()` once at
CLI/server/test-recording startup; library code only ever reads `os.environ`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(os.environ.get("CANOPY_ENV_FILE", REPO_ROOT / ".env"))

#: Model roles used across the pipeline (see docs/superpowers/plans, Global Constraints).
MODELS: dict[str, str] = {
    "primary": "claude-opus-5",         # mapper, extractors, digitizer read-outs
    "secondary": "claude-sonnet-5",     # cross-check / second route / verifier
    "adjudicator": "claude-opus-5",     # adjudication of disagreements
    "adjudicator_max": "claude-fable-5",  # optional max-effort adjudicator (server-side fallback)
}

#: Prompt-cache TTL marker used on document/image blocks.
EPHEMERAL = {"type": "ephemeral"}


def load_env(path: str | Path | None = None, override: bool = False) -> bool:
    """Load `.env` into `os.environ`. Returns True when a file was found.

    Never prints or logs the values. Missing python-dotenv or a missing file is not an error: the
    default (offline) test suite must work without any key at all.
    """
    p = Path(path) if path is not None else ENV_PATH
    if not p.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:                                   # pragma: no cover - dotenv is a dependency
        return False
    load_dotenv(p, override=override)
    return True


def api_key() -> str | None:
    """The Anthropic API key if configured (never log the return value)."""
    return os.environ.get("ANTHROPIC_API_KEY") or None


def live_enabled() -> bool:
    """True when live API calls are allowed (`CANOPY_LIVE=1`)."""
    return os.environ.get("CANOPY_LIVE", "") not in ("", "0", "false", "False")


def record_enabled() -> bool:
    """True when new fixtures should be written (`CANOPY_RECORD=1`)."""
    return os.environ.get("CANOPY_RECORD", "") not in ("", "0", "false", "False")
