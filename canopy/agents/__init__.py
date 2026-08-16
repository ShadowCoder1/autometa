"""Canopy agents: the LLM roles that *locate and label* evidence in a paper.

No agent computes a statistic (see `canopy.llm.schemas`); each one returns JSON that maps onto the
models in `canopy.models`, and every prompt lives in a file under `canopy/llm/prompts/` so it can
be diffed, versioned and reviewed on its own.
"""
from __future__ import annotations

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "llm" / "prompts"
_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")


def load_prompt(name: str) -> str:
    """The raw text of `canopy/llm/prompts/<name>.md`."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no prompt {name!r} in {PROMPT_DIR}")
    return path.read_text()


def render_prompt(name: str, **values: str) -> str:
    """Fill `{{PLACEHOLDER}}` markers. An unfilled marker is a bug, so it raises."""
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    left = sorted(set(_PLACEHOLDER.findall(text)))
    if left:
        raise KeyError(f"prompt {name!r} still has unfilled placeholders {left}")
    return text
