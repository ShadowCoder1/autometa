"""`LLMClient` — structured/text calls with disk cache, fixture replay, cost accounting.

Layers of a call:  disk cache → replay fixtures → provider (live).  Nothing here does arithmetic
on study values; it only moves JSON around and keeps an audit trail of every call.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import live_enabled, record_enabled
from .cache import DiskCache, cache_key, image_hashes, sha256_text
from .costs import approx_tokens, estimate_cost, estimate_request_cost
from .errors import (BudgetExceeded, LiveCallsDisabled, LLMError, MissingFixture, ParseError,
                     RefusalError, TruncatedOutput)
from .providers import (AnthropicProvider, LLMProvider, LLMRequest, ReplayProvider,
                        response_from_record)
from .schemas import assert_no_derived_stats, assert_valid_output_schema

__all__ = ["LLMClient", "LLMResult", "LLMCall", "image_block", "pdf_block", "MissingFixture",
           "RefusalError", "TruncatedOutput", "BudgetExceeded", "LiveCallsDisabled", "ParseError",
           "LLMError"]

STREAM_MAX_TOKENS = 16000          # above this the SDK requires streaming
EPHEMERAL = {"type": "ephemeral"}


# ----------------------------------------------------------------------------- content blocks
def _b64(path: str | Path) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")


def image_block(png_path: str | Path, cache: bool = False) -> dict[str, Any]:
    """base64 PNG image block (images must already be sized by `ingest.images.prepare_for_claude`)."""
    block: dict[str, Any] = {"type": "image",
                             "source": {"type": "base64", "media_type": "image/png",
                                        "data": _b64(png_path)}}
    if cache:
        block["cache_control"] = dict(EPHEMERAL)
    return block


def pdf_block(pdf_path: str | Path | None, file_id: str | None = None,
              cache: bool = True) -> dict[str, Any]:
    """PDF document block — by `file_id` (Files API) when given, else inline base64."""
    if file_id:
        source: dict[str, Any] = {"type": "file", "file_id": file_id}
    else:
        if pdf_path is None:
            raise ValueError("pdf_block needs a path or a file_id")
        source = {"type": "base64", "media_type": "application/pdf", "data": _b64(pdf_path)}
    block: dict[str, Any] = {"type": "document", "source": source}
    if cache:
        block["cache_control"] = dict(EPHEMERAL)
    return block


# ----------------------------------------------------------------------------- results
@dataclass
class LLMResult:
    text: str
    parsed: Any | None
    usage: dict[str, Any]
    cost_usd: float
    model: str
    call_id: str
    cached: bool
    stop_reason: str
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    served_model: str = ""


@dataclass
class LLMCall:
    """Audit record for one call (amendment B)."""

    key: str
    model: str
    served_model: str = ""
    request_id: str = ""
    stop_reason: str = ""
    effort: str | None = None
    max_tokens: int = 0
    prompt_version: str = ""
    schema_hash: str = ""
    cell_key: str = ""
    image_hashes: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    source: str = "live"                     # live | disk_cache | replay
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ----------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, cache_dir: str | Path | None = None, budget_usd: float | None = None,
                 max_concurrency: int = 6, record_dir: str | Path | None = None,
                 replay_dir: str | Path | None = None, default_effort: str = "high",
                 provider: LLMProvider | None = None, client: Any = None,
                 allow_live: bool = False, prompt_version: str = ""):
        self._allow_live = allow_live
        self.cache = DiskCache(cache_dir)
        self.replay = ReplayProvider(replay_dir) if replay_dir is not None else None
        self.record_dir = Path(record_dir) if record_dir is not None else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd = budget_usd
        self.default_effort = default_effort
        self.prompt_version = prompt_version
        self.provider: LLMProvider = provider or AnthropicProvider(client=client)
        self._sem = threading.Semaphore(max_concurrency)
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._reserved = 0.0
        self._calls: list[LLMCall] = []

    # ------------------------------------------------------------------ helpers
    @property
    def live(self) -> bool:
        return self._allow_live or live_enabled()

    def image_block(self, png_path: str | Path, cache: bool = False) -> dict[str, Any]:
        return image_block(png_path, cache=cache)

    def pdf_block(self, pdf_path: str | Path | None = None, file_id: str | None = None,
                  cache: bool = True) -> dict[str, Any]:
        return pdf_block(pdf_path, file_id=file_id, cache=cache)

    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._calls]

    def reserved_usd(self) -> float:
        """USD currently reserved by calls that are in flight."""
        with self._lock:
            return self._reserved

    def check_budget(self, estimated_usd: float) -> None:
        """Raise if a call of this size would not fit — without taking the money."""
        with self._lock:
            self._assert_affordable(estimated_usd)

    def _assert_affordable(self, estimated_usd: float) -> None:
        """Caller must hold `self._lock`."""
        if self.budget_usd is None:
            return
        committed = self._total_cost + self._reserved
        if committed + estimated_usd > self.budget_usd:
            raise BudgetExceeded(
                f"call would cost about ${estimated_usd:.4f}; ${self._total_cost:.4f} spent and "
                f"${self._reserved:.4f} reserved of ${self.budget_usd:.4f}")

    def _reserve(self, estimated_usd: float) -> float:
        """Atomically take `estimated_usd` from the budget for a call about to be made.

        Reservation (not check-then-act) is what keeps concurrent calls from jointly
        overspending: the money is held until the real cost is known.
        """
        with self._lock:
            self._assert_affordable(estimated_usd)
            self._reserved += estimated_usd
        return estimated_usd

    def _release(self, estimated_usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimated_usd)

    def count_tokens(self, *, model: str, system: Any, messages: list[Any]) -> int:
        """Exact count when live calls are allowed, otherwise a chars/4 estimate."""
        counter = getattr(self.provider, "count_tokens", None)
        if counter is not None and (self.live or not getattr(self.provider, "is_live", False)):
            try:
                return int(counter(model=model, system=system, messages=messages))
            except Exception:                       # pragma: no cover - network hiccup
                pass
        return approx_tokens(system, messages)

    # ------------------------------------------------------------------ public calls
    def structured(self, *, model: str, system: Any = "", messages: list[Any],
                   schema: dict, effort: str | None = None, max_tokens: int = 16000,
                   cache_key_extra: str = "", betas: list[str] | None = None,
                   fallbacks: str | None = None, prompt_version: str = "",
                   cell_key: str = "", validate_schema: bool = True) -> LLMResult:
        """One structured-output call.

        The schema is validated before anything is sent: strict objects (amendment/API rules) and
        no derived-statistic fields (amendment A) — agents return raw extracted values, effect
        sizes are computed in `canopy.stats`. Pass `validate_schema=False` to bypass.
        """
        if validate_schema:
            assert_valid_output_schema(schema)
            assert_no_derived_stats(schema)
        return self._call(model=model, system=system, messages=messages, schema=schema,
                          effort=effort, max_tokens=max_tokens, cache_key_extra=cache_key_extra,
                          betas=betas, fallbacks=fallbacks, prompt_version=prompt_version,
                          cell_key=cell_key)

    def text(self, *, model: str, system: Any = "", messages: list[Any],
             effort: str | None = None, max_tokens: int = 8000, cache_key_extra: str = "",
             betas: list[str] | None = None, fallbacks: str | None = None,
             prompt_version: str = "", cell_key: str = "") -> LLMResult:
        return self._call(model=model, system=system, messages=messages, schema=None,
                          effort=effort, max_tokens=max_tokens, cache_key_extra=cache_key_extra,
                          betas=betas, fallbacks=fallbacks, prompt_version=prompt_version,
                          cell_key=cell_key)

    # ------------------------------------------------------------------ engine
    def _call(self, *, model: str, system: Any, messages: list[Any], schema: dict | None,
              effort: str | None, max_tokens: int, cache_key_extra: str,
              betas: list[str] | None, fallbacks: str | None, prompt_version: str,
              cell_key: str, _retry: int = 0) -> LLMResult:
        effort = effort or self.default_effort
        key = cache_key(model=model, system=system, messages=messages, schema=schema,
                        effort=effort, max_tokens=max_tokens, extra=cache_key_extra)
        meta = dict(key=key, model=model, effort=effort, max_tokens=max_tokens,
                    prompt_version=prompt_version or self.prompt_version,
                    schema_hash=sha256_text(json.dumps(schema, sort_keys=True)) if schema else "",
                    cell_key=cell_key, image_hashes=image_hashes(messages))

        cached = self.cache.get(key)
        if cached is not None:
            return self._replayed(cached, meta, source="disk_cache", schema=schema)
        if self.replay is not None:
            record = self.replay.load(key)
            if record is not None:
                return self._replayed(record, meta, source="replay", schema=schema)

        self._guard_live(key)
        request = LLMRequest(model=model, system=system, messages=messages, schema=schema,
                             effort=effort, max_tokens=max_tokens, betas=betas,
                             fallbacks=fallbacks, stream=max_tokens > STREAM_MAX_TOKENS, key=key)

        reservation = self._reserve(estimate_request_cost(model, system, messages, max_tokens))
        try:
            started = time.perf_counter()
            with self._sem:
                response = self.provider.complete(request)
            latency = time.perf_counter() - started
            cost = estimate_cost(response.usage, response.model or model)
            self._log(meta, response=response, latency=latency, cost=cost, cached=False,
                      source="live")
        finally:
            self._release(reservation)

        if response.stop_reason == "refusal":
            raise RefusalError(f"model refused (request {response.request_id or key})")
        if response.stop_reason == "max_tokens":
            if _retry == 0:
                return self._call(model=model, system=system, messages=messages, schema=schema,
                                  effort=effort, max_tokens=max_tokens * 2,
                                  cache_key_extra=cache_key_extra, betas=betas,
                                  fallbacks=fallbacks, prompt_version=prompt_version,
                                  cell_key=cell_key, _retry=1)
            raise TruncatedOutput(
                f"response hit max_tokens={max_tokens} twice (request {response.request_id or key})")

        parsed = self._parse(response.text) if schema is not None else None
        record = {
            "key": key, "model": model, "served_model": response.model,
            "stop_reason": response.stop_reason, "usage": response.usage, "text": response.text,
            "parsed": parsed, "response": response.raw, "request_id": response.request_id,
            "effort": effort, "max_tokens": max_tokens, "cost_usd": cost,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.cache.put(key, record)
        self._write_fixture(key, record)
        return LLMResult(text=response.text, parsed=parsed, usage=dict(response.usage),
                         cost_usd=cost, model=model, call_id=key, cached=False,
                         stop_reason=response.stop_reason, latency_s=latency, raw=response.raw,
                         request_id=response.request_id, served_model=response.model)

    # ------------------------------------------------------------------ internals
    def _guard_live(self, key: str) -> None:
        if self.live:
            return
        if self.replay is not None:
            raise MissingFixture(
                f"no fixture {key}.json in {self.replay.dir} — record one with "
                f"CANOPY_LIVE=1 CANOPY_RECORD=1, or pass a FakeProvider")
        if getattr(self.provider, "is_live", False):
            raise LiveCallsDisabled(
                "live API calls need CANOPY_LIVE=1 (or LLMClient(allow_live=True))")

    def _replayed(self, record: dict[str, Any], meta: dict[str, Any], source: str,
                  schema: dict | None) -> LLMResult:
        response = response_from_record(record, default_model=meta["model"])
        parsed = record.get("parsed")
        if parsed is None and schema is not None and response.text:
            parsed = self._parse(response.text)
        if response.stop_reason == "refusal":
            raise RefusalError(f"cached refusal for {meta['key']}")
        self._log(meta, response=response, latency=0.0, cost=0.0, cached=True, source=source)
        return LLMResult(text=response.text, parsed=parsed, usage=dict(response.usage),
                         cost_usd=0.0, model=meta["model"], call_id=meta["key"], cached=True,
                         stop_reason=response.stop_reason, latency_s=0.0, raw=response.raw,
                         request_id=response.request_id, served_model=response.model)

    def _log(self, meta: dict[str, Any], *, response: Any, latency: float, cost: float,
             cached: bool, source: str) -> None:
        usage = response.usage or {}
        call = LLMCall(
            key=meta["key"], model=meta["model"], served_model=response.model,
            request_id=response.request_id, stop_reason=response.stop_reason,
            effort=meta["effort"], max_tokens=meta["max_tokens"],
            prompt_version=meta["prompt_version"], schema_hash=meta["schema_hash"],
            cell_key=meta["cell_key"], image_hashes=list(meta["image_hashes"]),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            latency_s=round(latency, 4), cost_usd=cost, cached=cached, source=source,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self._lock:
            self._calls.append(call)
            self._total_cost += cost

    def _write_fixture(self, key: str, record: dict[str, Any]) -> None:
        targets: list[Path] = []
        if self.record_dir is not None:
            targets.append(self.record_dir)
        if record_enabled() and self.replay is not None:
            targets.append(self.replay.dir)
        for directory in targets:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))

    @staticmethod
    def _parse(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("```")[1]
                stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            start, end = stripped.find("{"), stripped.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(stripped[start:end + 1])
                except json.JSONDecodeError:
                    pass
            raise ParseError(f"structured response was not JSON: {text[:200]!r}")
