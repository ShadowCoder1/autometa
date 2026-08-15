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


CHARS_PER_TOKEN = 3.5             # deliberately pessimistic (real text is ~4)
IMAGE_TOKENS_ESTIMATE = 4784      # the cap prepare_for_claude sizes images to (HIGH_RES tier)
PDF_TOKENS_PER_BYTE = 0.15        # measured live: a 125 kB, 5-page paper ≈ 17.8k tokens
FILE_DOCUMENT_TOKENS = 20000      # document sent by file_id and never registered: assume a paper
PDF_TOKENS_PER_PAGE = 2500        # measured live: text + page raster of one journal page

#: `file_id` -> page count, so a whole-document call reserves what the paper actually costs
#: instead of the flat guess above. Whoever uploads the PDF registers it once
#: (`canopy.agents.verify_common.whole_paper`); a run only ever grows this map.
_FILE_PAGES: dict[str, int] = {}


def register_file_pages(file_id: str, n_pages: int) -> None:
    """Record how long the paper behind a Files-API id is (see `_FILE_PAGES`)."""
    if file_id and n_pages and n_pages > 0:
        _FILE_PAGES[str(file_id)] = int(n_pages)


def clear_file_pages() -> None:
    _FILE_PAGES.clear()


def file_document_tokens(file_id: str = "") -> int:
    """Input tokens to reserve for a `document` block sent by `file_id`."""
    pages = _FILE_PAGES.get(str(file_id or ""))
    return pages * PDF_TOKENS_PER_PAGE if pages else FILE_DOCUMENT_TOKENS


def approx_tokens(*parts: Any) -> int:
    """Very rough token count (chars/4) — only for offline estimates."""
    total = 0
    for part in parts:
        if part is None:
            continue
        text = part if isinstance(part, str) else json.dumps(part, default=str)
        total += len(text)
    return max(1, total // 4)


def _strip_binaries(node: Any, acc: list[int]) -> Any:
    """Replace base64/file payloads with their token estimate so they are not counted as text."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "source" and isinstance(v, dict):
                kind = v.get("type")
                if kind == "base64":
                    data = v.get("data") or ""
                    n_bytes = len(data) * 3 // 4
                    if str(v.get("media_type", "")).startswith("image/"):
                        acc[0] += IMAGE_TOKENS_ESTIMATE
                    else:
                        acc[0] += int(n_bytes * PDF_TOKENS_PER_BYTE)
                    out[k] = {kk: vv for kk, vv in v.items() if kk != "data"}
                    continue
                if kind == "file":
                    acc[0] += file_document_tokens(str(v.get("file_id") or ""))
                    out[k] = v
                    continue
            out[k] = _strip_binaries(v, acc)
        return out
    if isinstance(node, (list, tuple)):
        return [_strip_binaries(v, acc) for v in node]
    return node


def estimate_input_tokens(system: Any, messages: Any) -> int:
    """Conservative input-token estimate: text chars/3.5 plus per-image / per-document tokens."""
    acc = [0]
    stripped = _strip_binaries({"system": system, "messages": messages}, acc)
    text = json.dumps(stripped, default=str, ensure_ascii=False)
    return max(1, int(len(text) / CHARS_PER_TOKEN) + acc[0])


def estimate_request_cost(model: str, system: Any, messages: Any, max_tokens: int) -> float:
    """Worst-case cost of a call before it is made — the amount reserved against the budget."""
    usage = {"input_tokens": estimate_input_tokens(system, messages),
             "output_tokens": int(max_tokens)}
    return estimate_cost(usage, model)
