"""Content-addressed request keys and the on-disk response cache."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

_IMAGE_MEDIA = ("image/png", "image/jpeg", "image/gif", "image/webp")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_b64(data: str) -> str:
    """sha256 of the *decoded* bytes, so the key does not depend on base64 whitespace."""
    try:
        raw = base64.standard_b64decode(data)
    except Exception:                                # pragma: no cover - defensive
        raw = data.encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


def canonicalize(obj: Any) -> Any:
    """Normalise a request payload for hashing.

    * base64 image/document payloads are replaced by the sha256 of their bytes (keys stay small
      and stable across re-encodings);
    * `cache_control` markers are dropped (prompt caching must not change the cache key);
    * dict ordering is irrelevant (json.dumps(sort_keys=True) at the end).
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "cache_control":
                continue
            if k == "source" and isinstance(v, dict) and v.get("type") == "base64" and "data" in v:
                out[k] = {**{kk: vv for kk, vv in v.items() if kk != "data"},
                          "data_sha256": _hash_b64(v["data"])}
                continue
            out[k] = canonicalize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    return obj


def image_hashes(messages: Any) -> list[str]:
    """sha256 of every image byte-payload in a message list (audit trail for a call)."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            src = node.get("source")
            if node.get("type") == "image" and isinstance(src, dict):
                if src.get("type") == "base64" and src.get("media_type") in _IMAGE_MEDIA:
                    found.append(_hash_b64(src.get("data", "")))
                elif src.get("type") == "file":
                    found.append(str(src.get("file_id", "")))
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(messages)
    return found


def cache_key(*, model: str, system: Any, messages: Any, schema: Any = None,
              effort: str | None = "high", max_tokens: int = 16000, extra: str = "") -> str:
    """sha256 over everything that can change the answer."""
    payload = {
        "model": model,
        "system": canonicalize(system),
        "messages": canonicalize(messages),
        "schema": canonicalize(schema),
        "effort": effort,
        "max_tokens": max_tokens,
        "extra": extra,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)
    return sha256_text(blob)


class DiskCache:
    """`<dir>/<key>.json`; a `None` directory disables the cache (all misses, no writes)."""

    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory is not None else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def path(self, key: str) -> Path | None:
        return None if self.dir is None else self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:                 # corrupted entry: treat as a miss
            return None

    def put(self, key: str, value: dict) -> None:
        p = self.path(key)
        if p is None:
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(p)
