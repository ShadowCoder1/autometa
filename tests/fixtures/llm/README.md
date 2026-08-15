# Replayed LLM fixtures

One JSON file per request: the file name is the content-addressed request key
(`canopy.llm.cache.cache_key` — sha256 over model, system, messages with image bytes hashed,
schema, effort, max_tokens). `LLMClient(replay_dir=...)` serves them, so the default test suite
runs with no API key and no network.

```json
{
  "key": "<sha256>", "model": "claude-opus-5", "served_model": "claude-opus-5",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
  "text": "<raw first text block>", "parsed": {"...": "parsed JSON"},
  "response": {"...": "full message dump"}, "request_id": "req_...",
  "effort": "high", "max_tokens": 16000, "cost_usd": 0.0, "created_at": "..."
}
```

Record or refresh one with:

```bash
CANOPY_LIVE=1 CANOPY_RECORD=1 .venv/bin/python -m pytest tests/test_mapper.py -q
```

A missing fixture raises `MissingFixture` unless `CANOPY_LIVE=1` — tests never silently go live.
