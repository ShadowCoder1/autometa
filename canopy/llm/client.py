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
from typing import Any, Callable

from ..config import live_enabled, record_enabled
from .cache import DiskCache, cache_key, image_hashes, sha256_text
from .costs import approx_tokens, estimate_cost, estimate_request_cost
from .errors import (BudgetExceeded, LiveCallsDisabled, LLMError, MissingFixture, ParseError,
                     RefusalError, TruncatedOutput)
from .providers import (AnthropicProvider, LLMProvider, LLMRequest, ReplayProvider,
                        response_from_record)
from .schemas import assert_no_derived_stats, assert_valid_output_schema

__all__ = ["LLMClient", "LLMResult", "LLMCall", "ToolLoopResult", "image_block", "pdf_block",
           "MissingFixture", "RefusalError", "TruncatedOutput", "BudgetExceeded",
           "LiveCallsDisabled", "ParseError", "LLMError"]

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


def _summarize_tool_output(content: list[dict] | str, limit: int = 400) -> str:
    """One readable line per tool result for the audit trail (image bytes are hashed, not stored)."""
    if isinstance(content, str):
        return content if len(content) <= limit else content[:limit] + "..."
    parts: list[str] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            src = block.get("source") or {}
            parts.append(f"<image {src.get('media_type', '?')}>")
        else:
            parts.append(f"<{kind}>")
    joined = " ".join(parts)
    return joined if len(joined) <= limit else joined[:limit] + "..."


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
    content: list[dict[str, Any]] = field(default_factory=list)   # all blocks (tool_use included)


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
    tool_calls: int = 0                      # tool_use blocks the model emitted on this turn
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ToolLoopResult:
    """One completed tool-use loop (see `LLMClient.tool_loop`)."""

    parsed: dict[str, Any]                   # the `final_tool` call's validated input
    turns: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    call_ids: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(parsed=self.parsed, turns=self.turns, tool_calls=list(self.tool_calls),
                    call_ids=list(self.call_ids), cost_usd=self.cost_usd, models=list(self.models))


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

    # ------------------------------------------------------------------ tool loop
    def tool_loop(self, *, model: str, system: Any = "", messages: list[Any],
                  tools: list[dict], handlers: dict[str, Callable[[dict], list[dict] | str]],
                  final_tool: str, effort: str | None = None, max_tokens: int = 8000,
                  max_tool_calls: int = 8, prompt_version: str = "", cell_key: str = "",
                  cache_key_extra: str = "") -> ToolLoopResult:
        """Run a tool-use conversation until the model calls `final_tool`, one cached turn at a time.

        Every turn goes through `_call`, so each is disk-cached / replayable / live exactly like any
        other request (the tool list and `tool_choice` are part of the cache key). `handlers` map a
        tool name to a callable that receives the tool input and returns either a plain string or a
        list of content blocks (an image block, say). Their outputs go back as `tool_result` blocks.

        The terminal action is `final_tool` — a tool whose `input_schema` is the answer schema, so
        the answer arrives as validated JSON without combining `output_config.format` with tools.
        After `max_tool_calls` non-final tool calls the loop forces `tool_choice` to `final_tool`.
        A turn that calls no tool at all gets exactly one "call the tool" nudge, then raises.
        """
        if not any((t or {}).get("name") == final_tool for t in tools):
            raise ValueError(f"final_tool {final_tool!r} is not in the tool list "
                             f"{[t.get('name') for t in tools]}")
        convo: list[Any] = list(messages)
        logged: list[dict[str, Any]] = []
        call_ids: list[str] = []
        models: list[str] = []
        cost = 0.0
        turns = 0
        non_final = 0
        nudged = False
        max_turns = max_tool_calls + 4                       # forced-submit + nudge + slack
        while True:
            if turns >= max_turns:
                raise LLMError(f"tool loop did not reach {final_tool!r} in {turns} turns")
            forced = non_final >= max_tool_calls
            choice = {"type": "tool", "name": final_tool} if forced else {"type": "auto"}
            result = self._call(model=model, system=system, messages=convo, schema=None,
                                effort=effort, max_tokens=max_tokens,
                                cache_key_extra=cache_key_extra, betas=None, fallbacks=None,
                                prompt_version=prompt_version, cell_key=cell_key,
                                tools=tools, tool_choice=choice)
            turns += 1
            call_ids.append(result.call_id)
            models.append(result.served_model or model)
            cost += result.cost_usd
            blocks = list(result.content) or [{"type": "text", "text": result.text}]
            uses = [b for b in blocks if b.get("type") == "tool_use"]

            if not uses:
                if nudged:
                    raise LLMError(
                        f"model ended its turn without calling {final_tool!r} twice "
                        f"(last text: {result.text[:200]!r})")
                nudged = True
                convo = convo + [
                    {"role": "assistant", "content": blocks},
                    {"role": "user", "content": [{"type": "text", "text":
                     f"You did not call a tool. Call the `{final_tool}` tool now with your "
                     f"best answer; use `unknown`/null fields where you are not sure."}]}]
                continue

            final = next((u for u in uses if u.get("name") == final_tool), None)
            if final is not None:
                parsed = final.get("input")
                parsed = dict(parsed) if isinstance(parsed, dict) else {}
                logged.append({"turn": turns, "name": final_tool, "input": parsed,
                               "output": "", "image_hashes": [], "is_error": False})
                return ToolLoopResult(parsed=parsed, turns=turns, tool_calls=logged,
                                      call_ids=call_ids, cost_usd=cost, models=models)

            results: list[dict[str, Any]] = []
            for use in uses:
                non_final += 1
                name = str(use.get("name") or "")
                payload = use.get("input")
                payload = dict(payload) if isinstance(payload, dict) else {}
                content, is_error = self._run_tool(name, payload, handlers)
                results.append({"type": "tool_result", "tool_use_id": use.get("id", ""),
                                "content": content, "is_error": is_error})
                logged.append({"turn": turns, "name": name, "input": payload,
                               "output": _summarize_tool_output(content),
                               "image_hashes": image_hashes(content), "is_error": is_error})
            convo = convo + [{"role": "assistant", "content": blocks},
                             {"role": "user", "content": results}]

    @staticmethod
    def _run_tool(name: str, payload: dict[str, Any],
                  handlers: dict[str, Callable[[dict], list[dict] | str]]
                  ) -> tuple[list[dict] | str, bool]:
        """Run one tool; a bad name or a raising handler becomes an error result *for the model*."""
        handler = handlers.get(name)
        if handler is None:
            return (f"error: no tool named {name!r}; available tools: "
                    f"{sorted(handlers)}"), True
        try:
            out = handler(payload)
        except Exception as exc:                            # handler errors are the model's problem
            return f"error: {type(exc).__name__}: {exc}", True
        if isinstance(out, str):
            return out, False
        return list(out), False

    # ------------------------------------------------------------------ engine
    def _call(self, *, model: str, system: Any, messages: list[Any], schema: dict | None,
              effort: str | None, max_tokens: int, cache_key_extra: str,
              betas: list[str] | None, fallbacks: str | None, prompt_version: str,
              cell_key: str, tools: list[dict] | None = None,
              tool_choice: dict | None = None, _retry: int = 0) -> LLMResult:
        effort = effort or self.default_effort
        key = cache_key(model=model, system=system, messages=messages, schema=schema,
                        effort=effort, max_tokens=max_tokens, extra=cache_key_extra,
                        tools=tools, tool_choice=tool_choice)
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
                             fallbacks=fallbacks, stream=max_tokens > STREAM_MAX_TOKENS, key=key,
                             tools=tools, tool_choice=tool_choice)

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
                                  cell_key=cell_key, tools=tools, tool_choice=tool_choice,
                                  _retry=1)
            raise TruncatedOutput(
                f"response hit max_tokens={max_tokens} twice (request {response.request_id or key})")

        parsed = self._parse(response.text) if schema is not None else None
        record = {
            "key": key, "model": model, "served_model": response.model,
            "stop_reason": response.stop_reason, "usage": response.usage, "text": response.text,
            "content": response.content,
            "parsed": parsed, "response": response.raw, "request_id": response.request_id,
            "effort": effort, "max_tokens": max_tokens, "cost_usd": cost,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.cache.put(key, record)
        self._write_fixture(key, record)
        return LLMResult(text=response.text, parsed=parsed, usage=dict(response.usage),
                         cost_usd=cost, model=model, call_id=key, cached=False,
                         stop_reason=response.stop_reason, latency_s=latency, raw=response.raw,
                         request_id=response.request_id, served_model=response.model,
                         content=list(response.content))

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
                         request_id=response.request_id, served_model=response.model,
                         content=list(response.content))

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
            tool_calls=sum(1 for b in (getattr(response, "content", None) or [])
                           if isinstance(b, dict) and b.get("type") == "tool_use"),
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
