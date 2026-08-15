"""Provider seam.

`LLMClient` never talks to a vendor SDK directly; it talks to an `LLMProvider`:

* `AnthropicProvider` — the real Messages API (lazy SDK construction, `max_retries=4`);
* `ReplayProvider`    — reads recorded fixtures keyed by request hash (offline tests);
* `FakeProvider`      — canned payloads for unit tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import MissingFixture

#: models that reject `output_config.effort` (verified live: haiku returns 400 for it)
MODELS_WITHOUT_EFFORT: frozenset[str] = frozenset({"claude-haiku-4-5"})


def supports_effort(model: str) -> bool:
    name = model.split("/")[-1]
    return not any(name.startswith(m) for m in MODELS_WITHOUT_EFFORT)


@dataclass
class LLMRequest:
    model: str
    system: Any = ""
    messages: list[Any] = field(default_factory=list)
    schema: dict | None = None
    effort: str | None = "high"
    max_tokens: int = 16000
    betas: list[str] | None = None
    fallbacks: str | None = None
    stream: bool = False
    key: str = ""


@dataclass
class ProviderResponse:
    text: str
    stop_reason: str = "end_turn"
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    is_live: bool

    def complete(self, request: LLMRequest) -> ProviderResponse:
        ...


# ----------------------------------------------------------------------------- helpers
def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {}
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                  "cache_read_input_tokens"):
            v = getattr(usage, k, None)
            if v is not None:
                data[k] = v
    return {k: (v if isinstance(v, (int, float)) else v) for k, v in data.items()}


def _first_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    for block in content or []:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            return getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "") or ""
    return ""


def _raw_dict(message: Any) -> dict[str, Any]:
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:                                  # pragma: no cover - defensive
            return dump()
    return message if isinstance(message, dict) else {}


# ----------------------------------------------------------------------------- anthropic
class AnthropicProvider:
    """The real API. The SDK object is built on first use so importing/creating is key-free."""

    name = "anthropic"
    is_live = True

    def __init__(self, client: Any = None, max_retries: int = 4, api_key: str | None = None):
        self._client = client
        self._max_retries = max_retries
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            from ..config import api_key as env_key
            from ..config import load_env

            load_env()
            kwargs: dict[str, Any] = {"max_retries": self._max_retries}
            key = self._api_key or env_key()
            if key:
                kwargs["api_key"] = key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _kwargs(self, req: LLMRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {}
        if req.effort and supports_effort(req.model):
            output_config["effort"] = req.effort
        if req.schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": req.schema}
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": req.messages,
        }
        if output_config:
            kwargs["output_config"] = output_config
        if req.system:
            kwargs["system"] = req.system
        if req.betas:
            kwargs["betas"] = list(req.betas)
        if req.fallbacks:
            kwargs["fallbacks"] = req.fallbacks
        # NOTE: never pass `temperature` or `thinking` — these models think adaptively.
        return kwargs

    def _endpoint(self, req: LLMRequest) -> Any:
        if req.betas or req.fallbacks:
            return self.client.beta.messages
        return self.client.messages

    def complete(self, request: LLMRequest) -> ProviderResponse:
        kwargs = self._kwargs(request)
        endpoint = self._endpoint(request)
        if request.stream:
            with endpoint.stream(**kwargs) as stream:
                message = stream.get_final_message()
        else:
            message = endpoint.create(**kwargs)
        return ProviderResponse(
            text=_first_text(message),
            stop_reason=getattr(message, "stop_reason", "") or "",
            model=getattr(message, "model", "") or request.model,
            usage=_usage_dict(getattr(message, "usage", None)),
            request_id=str(getattr(message, "_request_id", "") or getattr(message, "id", "") or ""),
            raw=_raw_dict(message),
        )

    def count_tokens(self, *, model: str, system: Any, messages: list[Any]) -> int:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            kwargs["system"] = system
        res = self.client.messages.count_tokens(**kwargs)
        return int(getattr(res, "input_tokens", 0) or 0)

    def upload_file(self, path: str | Path, media_type: str = "application/pdf",
                    betas: list[str] | None = None) -> str:
        p = Path(path)
        kwargs: dict[str, Any] = {}
        if betas:
            kwargs["betas"] = list(betas)
        with open(p, "rb") as fh:
            uploaded = self.client.beta.files.upload(file=(p.name, fh, media_type), **kwargs)
        return str(getattr(uploaded, "id", "") or "")


# ----------------------------------------------------------------------------- replay
class ReplayProvider:
    """Serves recorded fixtures from `<dir>/<key>.json`; never touches the network."""

    name = "replay"
    is_live = False

    def __init__(self, replay_dir: str | Path):
        self.dir = Path(replay_dir)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def complete(self, request: LLMRequest) -> ProviderResponse:
        record = self.load(request.key)
        if record is None:
            raise MissingFixture(
                f"no fixture {request.key}.json in {self.dir} — record one with "
                f"CANOPY_LIVE=1 CANOPY_RECORD=1")
        return response_from_record(record, default_model=request.model)


def response_from_record(record: dict[str, Any], default_model: str = "") -> ProviderResponse:
    """Build a `ProviderResponse` from a cached/recorded fixture dict."""
    text = record.get("text")
    if text is None and record.get("parsed") is not None:
        text = json.dumps(record["parsed"])
    return ProviderResponse(
        text=text or "",
        stop_reason=record.get("stop_reason", "end_turn"),
        model=record.get("served_model") or record.get("model") or default_model,
        usage=record.get("usage") or {},
        request_id=record.get("request_id", ""),
        raw=record.get("response") or {},
    )


# ----------------------------------------------------------------------------- fake
class FakeProvider:
    """Canned responses for unit tests. `payloads` may be dicts (JSON-dumped) or raw strings."""

    name = "fake"
    is_live = False

    def __init__(self, payloads: Any = None, stop_reason: str = "end_turn",
                 usage: dict[str, Any] | None = None, model: str = ""):
        if payloads is None:
            payloads = [{"ok": True}]
        if isinstance(payloads, (dict, str)):
            payloads = [payloads]
        self.payloads = list(payloads)
        self.stop_reason = stop_reason
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 100,
                               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self.model = model
        self.requests: list[LLMRequest] = []
        self.uploads: list[dict[str, Any]] = []

    def complete(self, request: LLMRequest) -> ProviderResponse:
        self.requests.append(request)
        i = min(len(self.requests) - 1, len(self.payloads) - 1)
        payload = self.payloads[i]
        if callable(payload):
            payload = payload(request)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return ProviderResponse(text=text, stop_reason=self.stop_reason,
                                model=self.model or request.model, usage=dict(self.usage),
                                request_id=f"fake_{len(self.requests)}",
                                raw={"id": f"fake_{len(self.requests)}"})

    def count_tokens(self, *, model: str, system: Any, messages: list[Any]) -> int:
        from .costs import approx_tokens

        return approx_tokens(system, messages)

    def upload_file(self, path: str | Path, media_type: str = "application/pdf",
                    betas: list[str] | None = None) -> str:
        import hashlib

        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        self.uploads.append({"path": str(path), "media_type": media_type, "betas": betas})
        return f"file_fake_{digest}"
