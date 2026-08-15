"""Token prices and cost accounting ($ per million tokens)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

CACHE_WRITE_MULTIPLIER = 1.25     # 5-minute cache write costs 1.25x the input price
CACHE_READ_MULTIPLIER = 0.1       # cache read costs 0.1x the input price


class UnknownModel(KeyError):
    """Raised when a model has no entry in the price table."""


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-fable-5": Price(10.0, 50.0),
    "claude-opus-4-8": Price(5.0, 25.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
}

_MTOK = 1_000_000.0


def price_for(model: str) -> Price:
    """Price of a model id; tolerates the `provider/model` and dated `model-YYYYMMDD` forms."""
    if model in PRICES:
        return PRICES[model]
    name = model.split("/")[-1]
    if name in PRICES:
        return PRICES[name]
    for known in PRICES:                      # e.g. "claude-opus-5-20260401"
        if name.startswith(known):
            return PRICES[known]
    raise UnknownModel(f"no price for model {model!r}; add it to canopy.llm.costs.PRICES")


def estimate_cost(usage: Mapping[str, Any] | None, model: str) -> float:
    """USD for one call from an Anthropic `usage` mapping."""
    if not usage:
        return 0.0
    p = price_for(model)
    inp = float(usage.get("input_tokens") or 0)
    out = float(usage.get("output_tokens") or 0)
    cw = float(usage.get("cache_creation_input_tokens") or 0)
    cr = float(usage.get("cache_read_input_tokens") or 0)
    return (inp * p.input_per_mtok
            + out * p.output_per_mtok
            + cw * p.input_per_mtok * CACHE_WRITE_MULTIPLIER
            + cr * p.input_per_mtok * CACHE_READ_MULTIPLIER) / _MTOK


def approx_tokens(*parts: Any) -> int:
    """Very rough token count (chars/4) — only for budget guards and offline estimates."""
    total = 0
    for part in parts:
        if part is None:
            continue
        text = part if isinstance(part, str) else json.dumps(part, default=str)
        total += len(text)
    return max(1, total // 4)


def estimate_request_cost(model: str, system: Any, messages: Any, max_tokens: int) -> float:
    """Worst-case cost of a call before it is made (input heuristic + full output budget)."""
    usage = {"input_tokens": approx_tokens(system, messages), "output_tokens": int(max_tokens)}
    return estimate_cost(usage, model)
