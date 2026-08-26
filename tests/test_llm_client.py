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
    DEGENERATE_MIN_LEN,
    LLMClient,
    LLMResult,
    MissingFixture,
    RefusalError,
    TruncatedOutput,
    degenerate_reply,
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


# --------------------------------------------------------- C12: a reply that did not happen
#: Every orientation ballot the three-paper run produced, copied verbatim out of
#: `runs/rerun-fixed/papers/*/verify.json` (that directory is not in the repo, so the ballots
#: travel here instead). The detector's whole claim is measured against these 16 and nothing else.
BALLOTS = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "runs" / "orientation_ballots.json").read_text()
)["ballots"]

#: The four the decision identifies, with the signature each one carries. A ballot that is not in
#: this table is a real reply, however thin — the detector may not touch it.
DEGENERATE = {
    ("bock", "Tracking root mean square error between target and cursor (RMSE)", "claude-opus-5"):
        ["too_short", "raw_serialisation"],
    ("cressman", "Hand deviation at peak velocity on no-cursor (aftereffect) reaches, expressed in "
                 "degrees and as a percentage of the 30° distortion, with aligned-cursor "
                 "performance as baseline", "claude-opus-5"): ["too_short"],
    ("cressman", "Mean angular deviation of the hand at peak velocity during reach training with "
                 "the misaligned (30° CW rotated) cursor", "claude-opus-5"): ["doubled_words"],
    ("heuer", "Aftereffect (posttest−pretest difference of final movement direction) in the "
              "open-loop test with cued absence of the visuomotor rotation", "claude-sonnet-5"):
        ["control_artefacts"],
}


def test_the_run_really_did_produce_sixteen_orientation_ballots():
    assert len(BALLOTS) == 16
    assert len({(b["paper"], b["measure_name"], b["model"]) for b in BALLOTS}) == 16
    assert {b["paper"] for b in BALLOTS} == {"bock", "cressman", "heuer"}


@pytest.mark.parametrize("ballot", BALLOTS,
                         ids=[f"{b['paper']}-{b['model'][7:]}-{len(b['reason'])}" for b in BALLOTS])
def test_exactly_the_four_degenerate_ballots_are_caught(ballot):
    """C12 acceptance test (iv). Twelve of these are real justifications between 403 and 951
    characters with 2-4 quotes each, and one of the four caught is 707 characters with 4 quotes —
    length and quote count are not what separates them, which is why the earlier `len < 40 or
    quotes == []` sweep found only one of the three anyone had noticed."""
    key = (ballot["paper"], ballot["measure_name"], ballot["model"])
    assert degenerate_reply(ballot["reason"]) == DEGENERATE.get(key, [])


def test_the_detector_is_not_just_a_length_test():
    """Zero false positives on twelve real replies, and the caught set is not the short set."""
    caught = [b for b in BALLOTS if degenerate_reply(b["reason"])]
    assert len(caught) == 4
    clean = [b for b in BALLOTS if not degenerate_reply(b["reason"])]
    assert min(len(b["reason"]) for b in clean) == 403        # a short real reply survives
    assert max(len(b["reason"]) for b in caught) == 1172      # a long fake one does not
    assert sorted(len(b["reason"]) for b in caught) == [11, 37, 707, 1172]


def test_a_control_character_from_a_pdf_text_layer_costs_one_call_not_a_cell():
    """The honest caveat. Heuer's 1172-character ballot is a real argument that happens to quote a
    `\\x08` out of the PDF's text layer, and the detector calls it degenerate anyway. That is the
    designed asymmetry: the caller re-issues once, so over-firing costs a ballot, while under-firing
    costs a pooled cell to a stub that was counted as a witness."""
    heuer = next(b for b in BALLOTS if degenerate_reply(b["reason"]) == ["control_artefacts"])
    assert "\x08" in heuer["reason"] and len(heuer["reason"]) > 1000
    assert degenerate_reply(heuer["reason"].replace("\x08", "")) == []


@pytest.mark.parametrize("text, expected", [
    ("placeholder", ["too_short"]),
    ("placeholder','reason':'',\"reason\":\"\"}", ["too_short", "raw_serialisation"]),
    ("", ["too_short"]),
    ("   \n  ", ["too_short"]),
    ("a" * (DEGENERATE_MIN_LEN - 1), ["too_short"]),
    ("a" * DEGENERATE_MIN_LEN, []),
    ("the reader should report which which group had the higher late-block value here",
     ["doubled_words"]),
    ("tabs\tand newlines\nand returns\r are ordinary text quoted out of a pdf, not debris", []),
    ("an otherwise fine sentence about the measure that carries a stray \\x08 escape", 
     ["control_artefacts"]),
    ("a sentence about a measure with an unbalanced { brace left in it by the serialiser",
     ["raw_serialisation"]),
])
def test_the_degenerate_signatures_one_at_a_time(text, expected):
    assert degenerate_reply(text) == expected


def test_a_refusal_long_enough_to_be_a_sentence_is_an_abstention_not_a_non_reply():
    """Fix round F15, and the controller's ruling on it: `not quotes` was NOT added as a fifth
    signature.

    This 46-character boilerplate refusal passes the detector, quotes or no quotes, and that is
    the decision rather than an oversight. A reader that answers "unknown" has ABSTAINED, and C3
    already refuses to let an abstention settle anything; reclassifying it as a reply that did not
    happen would open C12's single-witness row instead, and let the OTHER reader set the direction
    of the measure alone. Over-firing costs one call — this would cost a witness.
    """
    reply = "N/A. Not enough information was provided here."
    assert len(reply) > DEGENERATE_MIN_LEN
    assert degenerate_reply(reply) == []
    assert degenerate_reply(reply + " no quotes were supplied with it either") == []


def test_the_word_repeat_signature_needs_a_real_repeat():
    """`\\b(\\w{3,})\\s+\\1\\b` — a word that merely STARTS the same is not a repeat, and repeats
    shorter than three letters are left alone, because "of of" is a typo a human makes. Three
    letters and up is a decoding loop, and that is deliberately over-inclusive: it costs a call."""
    assert degenerate_reply("the value of of the measure is defined in the methods here") == []
    assert degenerate_reply("the measure measures adaptation and is defined in the methods") == []
    assert degenerate_reply("the the measure is defined in the methods here") == ["doubled_words"]


# ----------------------------------------------- the stuttered tail (whole-branch review, MAJOR 3)
#: `runs/nine/cache/9e6c5e19a487….json` → `content[1].text`, the raw reply claude-opus-5 gave to
#: the orientation ballot for Bock's aftereffect measure. It opens with a real argument and then
#: runs ~25 sentence tails that start mid-word — `stop_reason: end_turn`, 1131 output tokens, no
#: control characters, no doubled words, balanced braces, well over the length floor. It passed
#: every existing signature, was counted as an orientation witness, and was then put verbatim
#: into the tiebreak prompt as the reasoning of a prior reader.
STUTTERED = (
    "The measure is a signed angular pointing error in degrees (median initial movement "
    "direction), plotted on an axis running from -40 to +60 deg; during adaptation errors are "
    "positive (in the direction of the 60-deg rotation) and during the no-feedback "
    "after-effect phase they fall into the negative region, i.e. the sign carries the "
    "direction of the deviation and the magnitude of the negative excursion is the "
    "after-effect. Hence a larger (less negative) raw value corresponds to a SMALLER "
    "after-effect, so for this review's construct lower raw values mean more recalibration. "
    "The text explicitly states the after-effect was 'similar in both age groups' and "
    "'age-independent', with no statement that one group's after-effect magnitude exceeded "
    "the other's (only that its decline was faster in the young), so the direction on this "
    "raw measure is not stated.error sign not construct magnitude).ular error, so direction "
    "unknown..al angular error).al after-effect magnitude).ular pointing error).al "
    "deviation).ular error).al angular error, so unknown).ular error).al angular pointing "
    "error).ular error).al deviation from baseline).ular error).al after-effect).ular "
    "error).al angular error).ular error).al deviation).ular error).al after-effect "
    "magnitude).ular error).al signed error).ular error).al deviation).ular error).al "
    "after-effect).ular error).al angular error).ular error)."
)


def test_a_stuttered_tail_is_a_reply_that_did_not_happen():
    assert len(STUTTERED) > DEGENERATE_MIN_LEN
    assert degenerate_reply(STUTTERED) == ["stuttered_tail"]


def test_the_argument_this_reply_starts_with_is_a_real_reply_on_its_own():
    """The signature is about the TAIL, not about the reader: everything before the stutter is a
    justification that would have been counted, and is."""
    head = STUTTERED.split("not stated.")[0] + "not stated."
    assert len(head) > 800 and degenerate_reply(head) == []


def test_the_stutter_signature_needs_three_runs_of_it():
    """Two is prose that happens to bracket ("(deg).al" survives a sentence split); three is a
    decoder repeating itself. Over-firing costs one re-issued call, so the floor is low — but it
    is a floor, and an ordinary parenthesis before a lower-case word is not it."""
    assert degenerate_reply(
        "the measure is the angular error at the end of adaptation (deg).the paper states it "
        "plainly in the methods section and the figure legend repeats it") == []
    assert degenerate_reply(
        "the measure is defined in the methods).ular error).al deviation).ular error, so the "
        "direction on this raw measure is not stated anywhere in the paper") == ["stuttered_tail"]


def test_env_extra_body_is_merged_verbatim_and_bad_json_is_refused(monkeypatch):
    """Proxy plumbing: CANOPY_LLM_EXTRA_BODY reaches the request body verbatim (a gateway's
    routing fields, e.g. pinning the serving provider so Files-API references resolve) and a
    typo'd value fails loudly rather than silently unpinning every call."""
    import pytest

    from canopy.llm.providers import AnthropicProvider, LLMRequest

    provider = AnthropicProvider(client=object())
    req = LLMRequest(model="claude-sonnet-5", max_tokens=64,
                     messages=[{"role": "user", "content": "hi"}])
    monkeypatch.setenv("CANOPY_LLM_EXTRA_BODY",
                       '{"provider": {"order": ["Anthropic"], "allow_fallbacks": false}}')
    kwargs = provider._kwargs(req)
    assert kwargs["extra_body"] == {"provider": {"order": ["Anthropic"],
                                                 "allow_fallbacks": False}}
    monkeypatch.setenv("CANOPY_LLM_EXTRA_BODY", "")
    assert "extra_body" not in provider._kwargs(req)
    monkeypatch.setenv("CANOPY_LLM_EXTRA_BODY", "{not json")
    with pytest.raises(ValueError, match="CANOPY_LLM_EXTRA_BODY"):
        provider._kwargs(req)
