"""Task 2: LLM client — cache keys, fixture replay, cost, blocks, budget."""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import pytest

from canopy.llm.cache import DiskCache, cache_key
from canopy.llm.client import (
    BudgetExceeded,
    LLMClient,
    LLMResult,
    MissingFixture,
    RefusalError,
    TruncatedOutput,
    image_block,
    pdf_block,
)
from canopy.llm.providers import (
    AnthropicProvider,
    FakeProvider,
    LLMProvider,
    LLMRequest,
    ProviderResponse,
    ReplayProvider,
)
from canopy.llm.costs import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICES,
    UnknownModel,
    estimate_cost,
    price_for,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}
MSGS = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


# ------------------------------------------------------------------ fake SDK
class _Block:
    def __init__(self, text): self.type = "text"; self.text = text


class _Msg:
    def __init__(self, payload, stop_reason="end_turn", model="claude-opus-5"):
        self.id = "msg_fake"
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.stop_reason = stop_reason
        self.content = [_Block(json.dumps(payload))]
        self.usage = {"input_tokens": 1000, "output_tokens": 100,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    def model_dump(self, mode="json"):
        return {"id": self.id, "type": self.type, "role": self.role, "model": self.model,
                "stop_reason": self.stop_reason,
                "content": [{"type": "text", "text": b.text} for b in self.content],
                "usage": dict(self.usage)}


class _Stream:
    def __init__(self, msg): self._msg = msg

    def __enter__(self): return self

    def __exit__(self, *a): return False

    def get_final_message(self): return self._msg


class _Messages:
    def __init__(self, outer): self._outer = outer

    def create(self, **kw):
        self._outer.calls.append(kw)
        return _Msg(self._outer.payload, stop_reason=self._outer.stop_reason)

    def stream(self, **kw):
        self._outer.calls.append({**kw, "_stream": True})
        return _Stream(_Msg(self._outer.payload, stop_reason=self._outer.stop_reason))

    def count_tokens(self, **kw):
        self._outer.calls.append({**kw, "_count": True})
        return type("T", (), {"input_tokens": 4242})()


class FakeSDK:
    def __init__(self, payload=None, stop_reason="end_turn"):
        self.payload = payload if payload is not None else {"ok": True}
        self.stop_reason = stop_reason
        self.calls: list[dict] = []
        self.messages = _Messages(self)


def _client(tmp_path, **kw) -> LLMClient:
    kw.setdefault("client", FakeSDK())
    return LLMClient(cache_dir=tmp_path / "cache", **kw)


# ------------------------------------------------------------------ (a) cache key
def test_cache_key_stable_across_dict_ordering():
    k1 = cache_key(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA,
                   effort="high", max_tokens=16000)
    reordered_schema = {"additionalProperties": False, "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}}, "type": "object"}
    reordered_msgs = [{"content": [{"text": "hi", "type": "text"}], "role": "user"}]
    k2 = cache_key(schema=reordered_schema, messages=reordered_msgs, system="s",
                   model="claude-opus-5", max_tokens=16000, effort="high")
    assert k1 == k2 and len(k1) == 64


def test_cache_key_hashes_image_bytes_not_base64_and_ignores_cache_control():
    big = base64.standard_b64encode(b"\x89PNG" + b"x" * 5000).decode()
    m1 = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": big}}]}]
    m2 = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": big},
         "cache_control": {"type": "ephemeral"}}]}]
    k1 = cache_key(model="m", system="", messages=m1, schema=None, effort="high", max_tokens=10)
    k2 = cache_key(model="m", system="", messages=m2, schema=None, effort="high", max_tokens=10)
    assert k1 == k2
    other = base64.standard_b64encode(b"\x89PNG" + b"y" * 5000).decode()
    m3 = json.loads(json.dumps(m1).replace(big, other))
    assert cache_key(model="m", system="", messages=m3, schema=None,
                     effort="high", max_tokens=10) != k1


def test_cache_key_changes_with_inputs():
    base = dict(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA,
                effort="high", max_tokens=16000)
    k = cache_key(**base)
    assert cache_key(**{**base, "model": "claude-sonnet-5"}) != k
    assert cache_key(**{**base, "effort": "low"}) != k
    assert cache_key(**{**base, "max_tokens": 8000}) != k
    assert cache_key(**{**base, "extra": "v2"}) != k


# ------------------------------------------------------------------ (b)(c) replay
def _write_fixture(d: Path, key: str, payload: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps({
        "key": key, "model": "claude-opus-5", "stop_reason": "end_turn",
        "usage": {"input_tokens": 1000, "output_tokens": 100,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        "text": json.dumps(payload), "parsed": payload,
        "response": {"id": "msg_fixture"},
    }))


def test_replay_from_fixture_returns_parsed_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    key = cache_key(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA,
                    effort="high", max_tokens=16000)
    fx = tmp_path / "fixtures"
    _write_fixture(fx, key, {"ok": True, "value": 42})
    c = LLMClient(cache_dir=None, replay_dir=fx)      # no SDK client at all
    r = c.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)
    assert isinstance(r, LLMResult)
    assert r.parsed == {"ok": True, "value": 42}
    assert r.cached is True and r.call_id == key
    assert r.cost_usd == 0.0          # replays are free
    assert c.total_cost() == 0.0


def test_missing_fixture_raises_when_not_live(tmp_path, monkeypatch):
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    c = LLMClient(cache_dir=None, replay_dir=tmp_path / "fixtures")
    with pytest.raises(MissingFixture):
        c.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)


def test_record_writes_fixture_and_replays_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    rec = tmp_path / "fixtures"
    sdk = FakeSDK({"ok": True, "value": 7})
    c = LLMClient(cache_dir=None, record_dir=rec, replay_dir=rec, client=sdk)
    r = c.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)
    assert r.parsed["value"] == 7 and r.cached is False
    assert len(sdk.calls) == 1
    # structured outputs are requested via output_config.format, effort included
    oc = sdk.calls[0]["output_config"]
    assert oc["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert oc["effort"] == "high"
    assert "temperature" not in sdk.calls[0] and "thinking" not in sdk.calls[0]
    assert list(rec.glob("*.json"))
    # now offline replay of the recorded fixture
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    c2 = LLMClient(cache_dir=None, replay_dir=rec)
    r2 = c2.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)
    assert r2.parsed == r.parsed and r2.cached is True


def test_disk_cache_prevents_second_api_call(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=tmp_path / "cache", client=sdk)
    a = c.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)
    b = c.structured(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA)
    assert len(sdk.calls) == 1
    assert a.parsed == b.parsed
    assert a.cached is False and b.cached is True
    assert c.total_cost() == pytest.approx(a.cost_usd)   # cache hit adds nothing
    assert len(c.calls()) == 2 and c.calls()[1]["cached"] is True


def test_refusal_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    c = LLMClient(cache_dir=None, client=FakeSDK({"ok": True}, stop_reason="refusal"))
    with pytest.raises(RefusalError):
        c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA)


def test_fallbacks_use_beta_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")

    class _BetaMessages(_Messages):
        pass

    class FakeBetaSDK(FakeSDK):
        def __init__(self):
            super().__init__({"ok": True})
            self.beta = type("B", (), {})()
            self.beta.messages = _BetaMessages(self)

    sdk = FakeBetaSDK()
    c = LLMClient(cache_dir=None, client=sdk)
    c.structured(model="claude-fable-5", system="s", messages=MSGS, schema=SCHEMA,
                 betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    assert sdk.calls[0]["betas"] == ["server-side-fallback-2026-07-01"]
    assert sdk.calls[0]["fallbacks"] == "default"


# ------------------------------------------------------------------ (d) cost
def test_estimate_cost_numbers():
    usage = {"input_tokens": 100_000, "output_tokens": 10_000}
    assert estimate_cost(usage, "claude-opus-5") == pytest.approx(0.75)
    assert estimate_cost(usage, "claude-sonnet-5") == pytest.approx(0.30)
    assert estimate_cost(usage, "claude-haiku-4-5") == pytest.approx(0.15)
    assert estimate_cost(usage, "claude-fable-5") == pytest.approx(1.5)
    assert estimate_cost(usage, "claude-opus-4-8") == pytest.approx(0.75)


def test_estimate_cost_cache_multipliers():
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    expected = 5.0 * CACHE_WRITE_MULTIPLIER + 5.0 * CACHE_READ_MULTIPLIER
    assert estimate_cost(usage, "claude-opus-5") == pytest.approx(expected)
    assert CACHE_WRITE_MULTIPLIER == 1.25 and CACHE_READ_MULTIPLIER == 0.1


def test_price_table():
    assert (PRICES["claude-opus-5"].input_per_mtok, PRICES["claude-opus-5"].output_per_mtok) == (5.0, 25.0)
    assert (PRICES["claude-sonnet-5"].input_per_mtok, PRICES["claude-sonnet-5"].output_per_mtok) == (2.0, 10.0)
    assert (PRICES["claude-fable-5"].input_per_mtok, PRICES["claude-fable-5"].output_per_mtok) == (10.0, 50.0)
    assert (PRICES["claude-haiku-4-5"].input_per_mtok, PRICES["claude-haiku-4-5"].output_per_mtok) == (1.0, 5.0)
    with pytest.raises(UnknownModel):
        price_for("gpt-nope")


# ------------------------------------------------------------------ (e) blocks
def test_image_block_is_valid_base64_png_block(tmp_path):
    png = tmp_path / "p001.png"
    raw = b"\x89PNG\r\n\x1a\n" + b"pretend-pixels"
    png.write_bytes(raw)
    b = image_block(png)
    assert b["type"] == "image"
    assert b["source"]["type"] == "base64"
    assert b["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(b["source"]["data"]) == raw
    assert "cache_control" not in b
    assert image_block(png, cache=True)["cache_control"] == {"type": "ephemeral"}


def test_pdf_block_base64_and_file_id(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    b = pdf_block(pdf)
    assert b["type"] == "document"
    assert b["source"] == {"type": "base64", "media_type": "application/pdf",
                           "data": base64.standard_b64encode(b"%PDF-1.7\n%%EOF\n").decode()}
    assert b["cache_control"] == {"type": "ephemeral"}
    f = pdf_block(pdf, file_id="file_abc", cache=False)
    assert f["source"] == {"type": "file", "file_id": "file_abc"}
    assert "cache_control" not in f


def test_client_block_methods_delegate(tmp_path):
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    c = LLMClient(cache_dir=None)
    assert c.image_block(png) == image_block(png)


# ------------------------------------------------------------------ (f) budget
def test_budget_guard_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=None, budget_usd=1e-9, client=sdk)
    with pytest.raises(BudgetExceeded):
        c.structured(model="claude-opus-5", system="s" * 100, messages=MSGS, schema=SCHEMA)
    assert sdk.calls == []            # nothing was spent
    with pytest.raises(BudgetExceeded):
        c.check_budget(10.0)


def test_no_budget_means_no_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    c = LLMClient(cache_dir=None, client=FakeSDK({"ok": True}))
    c.check_budget(1e9)               # no budget configured -> no raise
    r = c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA)
    assert r.cost_usd > 0


def test_cached_call_is_not_budget_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    key = cache_key(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA,
                    effort="high", max_tokens=16000)
    fx = tmp_path / "fx"
    _write_fixture(fx, key, {"ok": True})
    c = LLMClient(cache_dir=None, replay_dir=fx, budget_usd=1e-9)
    assert c.structured(model="claude-opus-5", system="sys", messages=MSGS,
                        schema=SCHEMA).parsed == {"ok": True}


# ------------------------------------------------------------------ misc
def test_disk_cache_roundtrip(tmp_path):
    d = DiskCache(tmp_path / "c")
    assert d.get("k") is None
    d.put("k", {"a": 1})
    assert d.get("k") == {"a": 1}
    assert DiskCache(None).get("k") is None      # disabled cache is a no-op


def test_text_call_has_no_format(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK("plain")
    c = LLMClient(cache_dir=None, client=sdk)
    r = c.text(model="claude-sonnet-5", system="s", messages=MSGS)
    assert r.parsed is None
    assert "format" not in sdk.calls[0]["output_config"]


def test_client_without_api_key_does_not_construct_sdk(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    LLMClient(cache_dir=tmp_path / "c")           # must not raise


# ------------------------------------------------------------------ provider seam (amendment B)
def test_fake_provider_needs_no_key_no_live_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prov = FakeProvider([{"ok": True, "n": 1}, {"ok": True, "n": 2}])
    c = LLMClient(cache_dir=None, provider=prov)
    assert isinstance(prov, LLMProvider)
    assert c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA).parsed["n"] == 1
    assert c.structured(model="claude-opus-5", system="s2", messages=MSGS, schema=SCHEMA).parsed["n"] == 2
    assert len(prov.requests) == 2
    assert isinstance(prov.requests[0], LLMRequest)
    assert prov.requests[0].schema == SCHEMA and prov.requests[0].effort == "high"


def test_replay_provider_reads_fixture_dir(tmp_path):
    key = cache_key(model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA,
                    effort="high", max_tokens=16000)
    fx = tmp_path / "fx"
    _write_fixture(fx, key, {"ok": True, "value": 3})
    prov = ReplayProvider(fx)
    req = LLMRequest(key=key, model="claude-opus-5", system="sys", messages=MSGS, schema=SCHEMA,
                     effort="high", max_tokens=16000)
    r = prov.complete(req)
    assert isinstance(r, ProviderResponse)
    assert json.loads(r.text)["value"] == 3
    assert prov.is_live is False
    with pytest.raises(MissingFixture):
        prov.complete(LLMRequest(key="nope", model="m", system="", messages=MSGS))


def test_anthropic_provider_is_lazy_and_live(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = AnthropicProvider()          # must not construct the SDK yet
    assert p.is_live is True
    assert p.name == "anthropic"


# ------------------------------------------------------------------ truncation / streaming
def test_truncation_retries_once_with_double_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True}, stop_reason="max_tokens")
    c = LLMClient(cache_dir=None, client=sdk)
    with pytest.raises(TruncatedOutput):
        c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA, max_tokens=1000)
    assert [k["max_tokens"] for k in sdk.calls] == [1000, 2000]


def test_streaming_used_above_16k_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=None, client=sdk)
    c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA, max_tokens=32000)
    assert sdk.calls[0].get("_stream") is True
    sdk2 = FakeSDK({"ok": True})
    LLMClient(cache_dir=None, client=sdk2).structured(model="claude-opus-5", system="s",
                                                      messages=MSGS, schema=SCHEMA, max_tokens=16000)
    assert sdk2.calls[0].get("_stream") is None


# ------------------------------------------------------------------ call log (amendment B)
def test_call_record_has_audit_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    png = tmp_path / "i.png"
    png.write_bytes(b"\x89PNG-bytes")
    msgs = [{"role": "user", "content": [image_block(png), {"type": "text", "text": "hi"}]}]
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=None, client=sdk)
    r = c.structured(model="claude-opus-5", system="s", messages=msgs, schema=SCHEMA,
                     prompt_version="mapper@1", cell_key="ds1/late_adaptation/A")
    rec = c.calls()[-1]
    for field in ("key", "request_id", "stop_reason", "effort", "prompt_version", "schema_hash",
                  "model", "served_model", "image_hashes", "cell_key", "latency_s", "cost_usd",
                  "input_tokens", "output_tokens", "cache_creation_input_tokens",
                  "cache_read_input_tokens", "cached"):
        assert field in rec, field
    assert rec["prompt_version"] == "mapper@1"
    assert rec["cell_key"] == "ds1/late_adaptation/A"
    assert len(rec["image_hashes"]) == 1 and len(rec["image_hashes"][0]) == 64
    assert rec["schema_hash"] and rec["cost_usd"] > 0
    assert r.stop_reason == "end_turn" and r.latency_s >= 0
    assert json.dumps(rec)                     # the log is JSON-serialisable


def test_count_tokens_offline_is_a_heuristic(tmp_path, monkeypatch):
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    c = LLMClient(cache_dir=None, provider=FakeProvider([{"ok": True}]))
    n = c.count_tokens(model="claude-opus-5", system="s" * 400, messages=MSGS)
    assert isinstance(n, int) and n > 0


def test_count_tokens_live_uses_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=None, client=sdk)
    assert c.count_tokens(model="claude-opus-5", system="s", messages=MSGS) == 4242


# ------------------------------------------------------------------ per-model capabilities
def test_effort_is_dropped_for_models_that_reject_it(tmp_path, monkeypatch):
    """Verified live: claude-haiku-4-5 returns 400 for output_config.effort."""
    monkeypatch.setenv("CANOPY_LIVE", "1")
    sdk = FakeSDK({"ok": True})
    c = LLMClient(cache_dir=None, client=sdk)
    c.structured(model="claude-haiku-4-5", system="s", messages=MSGS, schema=SCHEMA)
    assert sdk.calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    c.text(model="claude-haiku-4-5", system="s", messages=MSGS)
    assert "output_config" not in sdk.calls[1]          # empty config is omitted entirely
    c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA)
    assert sdk.calls[2]["output_config"]["effort"] == "high"


# ------------------------------------------------------------------ budget reservation (fix 1)
class SlowProvider(FakeProvider):
    """FakeProvider that blocks inside `complete` so two callers overlap in flight."""

    def __init__(self, payload=None, delay: float = 0.25):
        super().__init__(payload or {"ok": True})
        self.delay = delay
        self.entered = threading.Event()

    def complete(self, request):
        self.entered.set()
        time.sleep(self.delay)
        return super().complete(request)


def _spend(client, results, index, **kw):
    try:
        results[index] = client.structured(model="claude-opus-5", system=f"s{index}",
                                           messages=MSGS, schema=SCHEMA, max_tokens=1000, **kw)
    except Exception as exc:                      # noqa: BLE001 - the test inspects the type
        results[index] = exc


def test_budget_is_reserved_not_just_checked(tmp_path):
    """Two concurrent calls must not jointly overspend: the second is refused while the
    first is still in flight (its cost is not yet recorded)."""
    provider = SlowProvider(delay=0.3)
    # one call reserves ~$0.025 (1000 output tokens on opus-5); the budget allows exactly one
    c = LLMClient(cache_dir=None, provider=provider, budget_usd=0.03, max_concurrency=4)
    results: dict[int, object] = {}
    t1 = threading.Thread(target=_spend, args=(c, results, 1))
    t1.start()
    assert provider.entered.wait(2.0)             # thread 1 is inside the provider call
    _spend(c, results, 2)                         # thread 2 asks while thread 1 is in flight
    t1.join(5)
    assert isinstance(results[1], LLMResult)
    assert isinstance(results[2], BudgetExceeded)
    assert len(provider.requests) == 1            # the refused call never reached the provider
    assert c.total_cost() <= 0.03


def test_reservation_is_released_after_the_call(tmp_path):
    provider = FakeProvider([{"ok": True}] * 5)
    # one reservation is ~$0.025; $0.05 fits two calls only when the first one is released
    c = LLMClient(cache_dir=None, provider=provider, budget_usd=0.05)
    a = c.structured(model="claude-opus-5", system="a", messages=MSGS, schema=SCHEMA,
                     max_tokens=1000)
    assert a.cost_usd == pytest.approx(0.0075)    # actual usage, far below the reservation
    b = c.structured(model="claude-opus-5", system="b", messages=MSGS, schema=SCHEMA,
                     max_tokens=1000)             # only possible if the reservation was freed
    assert b.parsed == {"ok": True}
    assert len(provider.requests) == 2
    assert c.total_cost() == pytest.approx(0.015)


def test_reservation_is_released_when_the_call_raises(tmp_path):
    class Boom(FakeProvider):
        def complete(self, request):
            self.requests.append(request)
            raise RuntimeError("network down")

    provider = Boom()
    c = LLMClient(cache_dir=None, provider=provider, budget_usd=0.03)
    with pytest.raises(RuntimeError):
        c.structured(model="claude-opus-5", system="a", messages=MSGS, schema=SCHEMA,
                     max_tokens=1000)
    assert c.reserved_usd() == 0.0
    c.check_budget(0.02)                          # budget is free again


def test_estimate_counts_images_without_counting_base64(tmp_path):
    from canopy.llm.costs import estimate_input_tokens

    png = tmp_path / "i.png"
    png.write_bytes(b"\x89PNG" + b"x" * 400_000)          # ~533 kB of base64
    msgs = [{"role": "user", "content": [image_block(png), {"type": "text", "text": "hello"}]}]
    tokens = estimate_input_tokens("sys", msgs)
    assert 1_000 < tokens < 20_000                        # image priced as an image, not as text


# ------------------------------------------------------------------ fallbacks (fix 2)
def test_fallbacks_only_allowed_for_fable(tmp_path, monkeypatch):
    monkeypatch.setenv("CANOPY_LIVE", "1")

    class FakeBeta(FakeSDK):
        def __init__(self):
            super().__init__({"ok": True})
            self.beta = type("B", (), {})()
            self.beta.messages = _Messages(self)

    sdk = FakeBeta()
    c = LLMClient(cache_dir=None, client=sdk)
    with pytest.raises(ValueError, match="fallbacks"):
        c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA,
                     betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    assert sdk.calls == []
    # betas alone stay legal for every model (e.g. the Files API beta)
    c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=SCHEMA,
                 betas=["files-api-2025-04-14"])
    assert sdk.calls[0]["betas"] == ["files-api-2025-04-14"]
    assert "fallbacks" not in sdk.calls[0]


# ------------------------------------------------------------------ schema validation (fix 5)
def test_structured_validates_the_output_schema(tmp_path):
    provider = FakeProvider([{"ok": True}])
    c = LLMClient(cache_dir=None, provider=provider)
    loose = {"type": "object", "properties": {"ok": {"type": "boolean"}}}   # no required/apf
    with pytest.raises(AssertionError):
        c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=loose)
    assert provider.requests == []                        # refused before spending anything
    r = c.structured(model="claude-opus-5", system="s", messages=MSGS, schema=loose,
                     validate_schema=False)               # explicit escape hatch
    assert r.parsed == {"ok": True}
