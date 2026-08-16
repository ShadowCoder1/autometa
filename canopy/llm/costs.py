"""Token prices and cost accounting ($ per million tokens)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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


# ----------------------------------------------------------------------------- run accounting
#: `cell_key` prefix -> the pipeline stage that spent the money. A call whose prefix is not here
#: is reported under `other` rather than being dropped or guessed at.
STAGE_OF_PREFIX: dict[str, str] = {
    "map": "map", "map-sources": "map", "map-crosscheck": "map", "map-adjudicate": "map",
    "map-roster": "map",
    "extract-text": "extract", "extract-stats": "extract", "digitize": "digitize",
    "verify": "verify", "orientation": "verify", "adjudicate": "verify",
}


def stage_of(cell_key: str) -> str:
    """Which stage a logged call belongs to, from its `cell_key` prefix."""
    head = str(cell_key or "").split(":", 1)[0].strip()
    return STAGE_OF_PREFIX.get(head, "other" if head else "unattributed")


def _usage_totals(calls: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cached_replies": 0}
    for call in calls:
        total["calls"] += 1
        total["cost_usd"] += float(call.get("cost_usd") or 0.0)
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                    "cache_read_input_tokens"):
            total[key] += int(call.get(key) or 0)
        total["cached_replies"] += 1 if call.get("cached") else 0
    total["cost_usd"] = round(total["cost_usd"], 6)
    return total


def cost_by_stage(calls: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-stage `{calls, cost_usd, tokens…}` from `LLMClient.calls()` (amendment B, task 15)."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for call in calls:
        buckets.setdefault(stage_of(str(call.get("cell_key") or "")), []).append(call)
    return {stage: _usage_totals(rows) for stage, rows in sorted(buckets.items())}


def cache_stats(calls: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """How much of the input the API served from its prompt cache, and what that saved.

    `cache_hit_ratio` is read tokens over ALL input-side tokens (fresh + written + read), so it
    answers "what share of everything we sent did we get at a tenth of the price?". Calls the disk
    cache replayed are excluded: they never reached the API, and counting them would flatter the
    number the run is trying to measure.
    """
    live = [c for c in calls if not c.get("cached")]
    total = _usage_totals(live)
    seen = (total["input_tokens"] + total["cache_creation_input_tokens"]
            + total["cache_read_input_tokens"])
    read = total["cache_read_input_tokens"]
    return {
        "live_calls": total["calls"],
        "input_tokens": total["input_tokens"],
        "cache_creation_input_tokens": total["cache_creation_input_tokens"],
        "cache_read_input_tokens": read,
        "cache_hit_ratio": round(read / seen, 4) if seen else 0.0,
        "saved_usd": round(sum(_saved(c) for c in live), 6),
    }


def _saved(call: Mapping[str, Any]) -> float:
    """USD this call did NOT pay because its prefix was already cached."""
    read = float(call.get("cache_read_input_tokens") or 0)
    if not read:
        return 0.0
    try:
        price = price_for(str(call.get("served_model") or call.get("model") or ""))
    except UnknownModel:                              # pragma: no cover - defensive
        return 0.0
    return read * price.input_per_mtok * (1.0 - CACHE_READ_MULTIPLIER) / _MTOK


def cache_summary_line(stats: Mapping[str, Any]) -> str:
    """The one line `canopy run` prints about caching."""
    return (f"prompt cache: {stats['cache_hit_ratio']:.0%} of input tokens read from cache "
            f"({stats['cache_read_input_tokens']:,} read, "
            f"{stats['cache_creation_input_tokens']:,} written, "
            f"{stats['input_tokens']:,} fresh) — saved about ${stats['saved_usd']:.2f}")


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
