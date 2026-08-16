"""Task 6b: `LLMClient.tool_loop` — turn-by-turn tool use on the cached/replayable `_call` engine."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from canopy.llm.cache import cache_key
from canopy.llm.client import LLMClient, LLMError
from canopy.llm.providers import FakeProvider, LLMRequest, ProviderResponse

SUBMIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "number"}},
}
SUBMIT_TOOL = {"name": "submit", "description": "Give the final answer.",
               "strict": True, "input_schema": SUBMIT_SCHEMA}
ZOOM_TOOL = {
    "name": "crop_image",
    "description": "Zoom into a region of the image.",
    "input_schema": {"type": "object", "additionalProperties": False,
                     "required": ["x0", "y0", "x1", "y1"],
                     "properties": {k: {"type": "number"} for k in ("x0", "y0", "x1", "y1")}},
}
TOOLS = [ZOOM_TOOL, SUBMIT_TOOL]
MSGS = [{"role": "user", "content": [{"type": "text", "text": "read the chart"}]}]

PNG_1PX = base64.standard_b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _use(name: str, inp: dict, tid: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _client(payloads, **kw) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider(payloads)
    return LLMClient(provider=provider, cache_dir=None, **kw), provider


# ------------------------------------------------------------------ provider seam
def test_fake_provider_returns_content_blocks_and_tool_use_stop_reason():
    provider = FakeProvider([[_text("looking"), _use("crop_image", {"x0": 0})]])
    res = provider.complete(LLMRequest(model="claude-opus-5", messages=MSGS))
    assert res.stop_reason == "tool_use"
    assert [b["type"] for b in res.content] == ["text", "tool_use"]
    assert res.text == "looking"


def test_anthropic_kwargs_pass_tools_and_tool_choice():
    from canopy.llm.providers import AnthropicProvider

    kwargs = AnthropicProvider()._kwargs(
        LLMRequest(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                   tool_choice={"type": "tool", "name": "submit"}, schema=None))
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit"}
    assert "format" not in kwargs.get("output_config", {})     # never combine tools + json format


def test_cache_key_depends_on_tools_and_tool_choice():
    base = dict(model="claude-opus-5", system="", messages=MSGS, schema=None)
    plain = cache_key(**base)
    assert cache_key(**base, tools=TOOLS) != plain
    assert cache_key(**base, tools=TOOLS) != cache_key(**base, tools=[SUBMIT_TOOL])
    assert (cache_key(**base, tools=TOOLS, tool_choice={"type": "auto"})
            != cache_key(**base, tools=TOOLS, tool_choice={"type": "tool", "name": "submit"}))
    assert cache_key(**base, tools=None, tool_choice=None) == plain   # old keys stay valid


# ------------------------------------------------------------------ the loop
def test_tool_loop_runs_handler_then_returns_submit_input():
    calls: list[dict] = []

    def crop(inp: dict) -> str:
        calls.append(inp)
        return "cropped region shows a bar top at y=120"

    client, provider = _client([
        [_text("let me zoom"), _use("crop_image", {"x0": 1, "y0": 2, "x1": 3, "y1": 4})],
        [_text("done"), _use("submit", {"answer": 12.5}, tid="toolu_2")],
    ])
    out = client.tool_loop(model="claude-opus-5", system="s", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": crop}, final_tool="submit")
    assert out.parsed == {"answer": 12.5}
    assert out.turns == 2
    assert calls == [{"x0": 1, "y0": 2, "x1": 3, "y1": 4}]
    assert [c["name"] for c in out.tool_calls] == ["crop_image", "submit"]
    assert "bar top" in out.tool_calls[0]["output"]
    assert len(out.call_ids) == 2 and len(set(out.call_ids)) == 2
    assert out.models == ["claude-opus-5", "claude-opus-5"]
    # the second request carries the assistant turn and the tool_result
    second = provider.requests[1].messages
    assert second[1]["role"] == "assistant"
    assert second[2]["content"][0]["type"] == "tool_result"
    assert second[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_tool_loop_handler_may_return_content_blocks_and_images_are_hashed():
    def crop(inp: dict) -> list[dict]:
        return [{"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": base64.standard_b64encode(PNG_1PX).decode()}},
                _text("that crop is sent-image px [0..10]x[0..10] at zoom 4")]

    client, provider = _client([
        [_use("crop_image", {"x0": 0, "y0": 0, "x1": 10, "y1": 10})],
        [_use("submit", {"answer": 3.0}, tid="toolu_2")],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": crop}, final_tool="submit")
    assert out.parsed == {"answer": 3.0}
    logged = out.tool_calls[0]
    assert logged["image_hashes"] and len(logged["image_hashes"][0]) == 64
    assert "zoom 4" in logged["output"]
    sent = provider.requests[1].messages[-1]["content"][0]["content"]
    assert [b["type"] for b in sent] == ["image", "text"]


def test_tool_loop_reports_unknown_tool_as_an_error_result():
    client, _ = _client([
        [_use("nope", {})],
        [_use("submit", {"answer": 1.0}, tid="toolu_2")],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={}, final_tool="submit")
    assert out.parsed == {"answer": 1.0}
    assert out.tool_calls[0]["is_error"] is True
    assert "nope" in out.tool_calls[0]["output"]


def test_tool_loop_surfaces_a_failing_handler_to_the_model_not_the_caller():
    def boom(inp: dict) -> str:
        raise ValueError("x0 is outside the image")

    client, _ = _client([
        [_use("crop_image", {"x0": -5})],
        [_use("submit", {"answer": 2.0}, tid="toolu_2")],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": boom}, final_tool="submit")
    assert out.parsed == {"answer": 2.0}
    assert out.tool_calls[0]["is_error"] is True
    assert "outside the image" in out.tool_calls[0]["output"]


def test_tool_loop_forces_the_final_tool_after_the_call_cap():
    payloads = [[_use("crop_image", {"x0": i}, tid=f"toolu_{i}")] for i in range(2)]
    payloads.append([_use("submit", {"answer": 9.0}, tid="toolu_final")])
    client, provider = _client(payloads)
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": lambda i: "ok"}, final_tool="submit",
                           max_tool_calls=2)
    assert out.parsed == {"answer": 9.0}
    choices = [r.tool_choice for r in provider.requests]
    assert choices[0] == {"type": "auto"} and choices[1] == {"type": "auto"}
    assert choices[2] == {"type": "tool", "name": "submit"}   # cap reached -> forced
    assert len([c for c in out.tool_calls if c["name"] == "crop_image"]) == 2


def test_tool_loop_nudges_once_when_a_turn_calls_no_tool_then_raises():
    client, provider = _client([[_text("The mean looks like 12.")]])
    with pytest.raises(LLMError, match="submit"):
        client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                         handlers={}, final_tool="submit")
    assert len(provider.requests) == 2                       # original + one nudge
    nudge = provider.requests[1].messages[-1]
    assert nudge["role"] == "user" and "submit" in json.dumps(nudge["content"])


def test_tool_loop_recovers_when_the_nudge_works():
    client, _ = _client([
        [_text("The mean looks like 12.")],
        [_use("submit", {"answer": 12.0})],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={}, final_tool="submit")
    assert out.parsed == {"answer": 12.0} and out.turns == 2


def test_tool_loop_parallel_tool_calls_return_all_results_in_one_user_turn():
    client, provider = _client([
        [_use("crop_image", {"x0": 0}, tid="t1"), _use("crop_image", {"x0": 9}, tid="t2")],
        [_use("submit", {"answer": 4.0}, tid="t3")],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": lambda i: f"crop at {i['x0']}"},
                           final_tool="submit")
    results = provider.requests[1].messages[-1]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert len([c for c in out.tool_calls if c["name"] == "crop_image"]) == 2
    assert out.parsed == {"answer": 4.0}


# ------------------------------------------------------------------ audit + cost
def test_tool_loop_audits_each_turn_with_a_tool_call_count():
    client, _ = _client([
        [_use("crop_image", {"x0": 0}, tid="t1"), _use("crop_image", {"x0": 5}, tid="t2")],
        [_use("submit", {"answer": 7.0}, tid="t3")],
    ])
    out = client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": lambda i: "ok"}, final_tool="submit",
                           prompt_version="digitize/1", cell_key="ds1/late/A")
    calls = client.calls()
    assert len(calls) == 2
    assert [c["tool_calls"] for c in calls] == [2, 1]
    assert all(c["prompt_version"] == "digitize/1" for c in calls)
    assert all(c["cell_key"] == "ds1/late/A" for c in calls)
    assert all(c["stop_reason"] == "tool_use" for c in calls)
    assert out.cost_usd == pytest.approx(client.total_cost())
    assert out.cost_usd > 0


def test_tool_loop_turns_are_cached_on_disk_and_replayed(tmp_path: Path):
    payloads = [
        [_use("crop_image", {"x0": 0}, tid="t1")],
        [_use("crop_image", {"x0": 40}, tid="t2")],
        [_use("submit", {"answer": 31.5}, tid="t3")],
    ]
    live = LLMClient(provider=FakeProvider(payloads), cache_dir=None, record_dir=tmp_path)
    first = live.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": lambda i: f"crop {i['x0']}"},
                           final_tool="submit")
    assert first.turns == 3
    assert len(list(tmp_path.glob("*.json"))) == 3

    dead = FakeProvider([])                       # would raise IndexError if ever called
    replayed = LLMClient(provider=dead, cache_dir=None, replay_dir=tmp_path)
    again = replayed.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                               handlers={"crop_image": lambda i: f"crop {i['x0']}"},
                               final_tool="submit")
    assert again.parsed == {"answer": 31.5}
    assert again.turns == 3
    assert again.call_ids == first.call_ids
    assert again.cost_usd == 0.0
    assert [c["source"] for c in replayed.calls()] == ["replay"] * 3


def test_tool_loop_requires_live_or_a_fixture(tmp_path: Path):
    from canopy.llm.client import MissingFixture

    client = LLMClient(provider=FakeProvider([]), cache_dir=None, replay_dir=tmp_path)
    with pytest.raises(MissingFixture):
        client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS, handlers={},
                         final_tool="submit")


def test_structured_and_text_calls_send_no_tools():
    client, provider = _client([{"ok": True}])
    client.structured(model="claude-opus-5", messages=MSGS,
                      schema={"type": "object", "additionalProperties": False,
                              "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
    assert provider.requests[0].tools is None
    assert provider.requests[0].tool_choice is None


def test_first_text_picks_the_first_text_block_past_thinking_and_tool_use():
    from canopy.llm.providers import _first_text

    message = {"content": [{"type": "thinking", "thinking": "hmm"}, _use("submit", {}),
                           _text("the answer"), _text("ignored")]}
    assert _first_text(message) == "the answer"
    assert _first_text({"content": [_use("submit", {})]}) == ""


def test_fake_provider_text_is_the_first_text_block_of_a_canned_turn():
    provider = FakeProvider([[{"type": "thinking", "thinking": "hmm"}, _text("first"),
                              _use("crop_image", {}), _text("second")]])
    res = provider.complete(LLMRequest(model="claude-opus-5", messages=MSGS))
    assert res.text == "first"
    assert [b["type"] for b in res.content] == ["thinking", "text", "tool_use", "text"]


def test_kwargs_refuses_tools_and_a_structured_schema_together():
    from canopy.llm.providers import AnthropicProvider

    req = LLMRequest(model="claude-opus-5", messages=MSGS, tools=TOOLS, schema=SUBMIT_SCHEMA)
    with pytest.raises(ValueError, match="cannot be combined"):
        AnthropicProvider()._kwargs(req)
    # each on its own is fine
    AnthropicProvider()._kwargs(LLMRequest(model="claude-opus-5", messages=MSGS, tools=TOOLS))
    AnthropicProvider()._kwargs(LLMRequest(model="claude-opus-5", messages=MSGS,
                                           schema=SUBMIT_SCHEMA))


# ------------------------------------------------------------------ task 15: prompt caching
def _markers(content) -> list[int]:
    """Indices of the blocks in one message's content that carry a `cache_control` marker."""
    return [i for i, b in enumerate(content or []) if isinstance(b, dict) and "cache_control" in b]


def test_tool_loop_marks_the_last_block_of_each_turns_tool_result():
    """Turn N+1 can only read turn N's prefix if turn N's last block carries a marker.

    The live smoke test (task 15 §A) measured this: with a marker only on the first user turn,
    turn 2 read 3 156 tokens and paid full price for the 4 500 tokens the tool result added.
    """
    client, provider = _client([[_text("zoom"), _use("crop_image", {"x0": 0}, "t1")],
                                [_text("again"), _use("crop_image", {"x0": 1}, "t2")],
                                [_use("submit", {"answer": 3.0}, "t3")]])
    client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                     handlers={"crop_image": lambda i: "a crop"}, final_tool="submit")
    second, third = provider.requests[1].messages, provider.requests[2].messages
    # the tool_result the first turn appended is marked, so the second turn reads it back
    assert _markers(second[-1]["content"]) == [len(second[-1]["content"]) - 1]
    assert _markers(third[-1]["content"]) == [len(third[-1]["content"]) - 1]


def test_tool_loop_keeps_at_most_four_cache_markers_per_request():
    """The API allows four breakpoints; a long loop must move the marker, not accumulate them."""
    turns = [[_text(f"zoom {i}"), _use("crop_image", {"x0": i}, f"t{i}")] for i in range(6)]
    client, provider = _client([*turns, [_use("submit", {"answer": 1.0}, "tf")]])
    first = [{"role": "user", "content": [{"type": "text", "text": "read the chart",
                                           "cache_control": {"type": "ephemeral"}}]}]
    client.tool_loop(model="claude-opus-5", messages=first, tools=TOOLS,
                     handlers={"crop_image": lambda i: "a crop"}, final_tool="submit",
                     max_tool_calls=8)
    for request in provider.requests:
        total = sum(len(_markers(m.get("content"))) for m in request.messages
                    if isinstance(m.get("content"), list))
        assert total <= 4, f"{total} cache_control markers in one request"
    # the caller's own marker on the invariant first block survives every turn
    assert _markers(provider.requests[-1].messages[0]["content"]) == [0]


def test_tool_loop_marker_does_not_change_the_cache_key(tmp_path: Path):
    """`cache_key` drops `cache_control`, so marking the growing prefix cannot invalidate a fixture."""
    payloads = [[_text("zoom"), _use("crop_image", {"x0": 0}, "t1")],
                [_use("submit", {"answer": 7.0}, "t2")]]
    client = LLMClient(provider=FakeProvider(payloads), cache_dir=tmp_path)
    client.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                     handlers={"crop_image": lambda i: "a crop"}, final_tool="submit")
    keys = {p.stem for p in tmp_path.glob("*.json")}
    provider2 = FakeProvider(payloads)
    replay = LLMClient(provider=provider2, cache_dir=tmp_path)
    out = replay.tool_loop(model="claude-opus-5", messages=MSGS, tools=TOOLS,
                           handlers={"crop_image": lambda i: "a crop"}, final_tool="submit")
    assert out.parsed == {"answer": 7.0}
    assert provider2.requests == [], "a replayed loop must make no provider call"
    assert {p.stem for p in tmp_path.glob("*.json")} == keys
