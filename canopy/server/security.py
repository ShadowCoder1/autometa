"""Tokens and path safety — the two things standing between a run directory and the network.

A run directory holds the user's PDFs, everything the models read out of them and the cache of
every request. The server hands out exactly two capabilities:

* **a per-run bearer token**, minted when the run is created and compared in constant time;
* **a path resolver** that will only return a file which is still inside the run directory after
  `resolve()` (so `..`, an absolute path and a symlink all fail), whose suffix is on a short
  allow-list, and which is not in `uploads/` (the user's own PDFs) or `cache/` (raw model
  traffic). Those two directories are never served: nothing a reader needs lives in them.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

__all__ = ["ALLOWED_SUFFIXES", "DENIED_TOP_LEVEL", "ACCESS_COOKIE", "PathRejected", "mint_token",
           "media_type", "safe_run_path", "token_matches", "is_loopback", "is_attachment",
           "access_cookie_value", "access_granted"]

#: the cookie a browser holds once it has given the site's access code
ACCESS_COOKIE = "canopy_access"

#: what a run directory may serve — artefacts a reader looks at, nothing executable
ALLOWED_SUFFIXES: frozenset[str] = frozenset({
    ".png", ".svg", ".pdf", ".csv", ".json", ".jsonl", ".html", ".md", ".txt", ".xlsx",
    ".yaml", ".yml"})

#: never served, whatever the suffix says
DENIED_TOP_LEVEL: frozenset[str] = frozenset({"uploads", "cache"})

_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf",
    ".csv": "text/csv; charset=utf-8", ".json": "application/json",
    ".jsonl": "application/json", ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".yaml": "text/plain; charset=utf-8", ".yml": "text/plain; charset=utf-8"}

#: served as a download rather than rendered in the tab (a PDF or an HTML file is a program)
_ATTACHMENT_SUFFIXES: frozenset[str] = frozenset({".pdf", ".html", ".xlsx", ".csv"})

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1"})


class PathRejected(ValueError):
    """A requested artefact path is not one this run may serve."""


def mint_token() -> str:
    """A per-run bearer token. 32 random bytes: guessing one is not a threat model."""
    return secrets.token_urlsafe(32)


def token_matches(given: str | None, expected: str | None) -> bool:
    """Constant-time comparison; an empty token never matches."""
    if not given or not expected:
        return False
    return hmac.compare_digest(str(given), str(expected))


def is_loopback(host: str | None) -> bool:
    return str(host or "").strip().strip("[]").lower() in _LOOPBACK_HOSTS


def access_cookie_value(code: str) -> str:
    """What a browser holds after giving the right code: a digest of it, never the code itself.

    The code is one environment variable on the server, and it is the only thing between the
    public internet and the API budget. A cookie carrying it verbatim would copy it into every
    request log and every browser's cookie jar. A digest lets the holder prove they once knew the
    code without a log reader learning it.
    """
    return hashlib.sha256(b"canopy-access:" + code.encode("utf-8")).hexdigest()


def access_granted(cookie: str | None, header: str | None, code: str | None) -> bool:
    """Whether a request may pass the site gate.

    No configured code means no gate — the local, loopback-only server stays exactly as it was.
    With one, a request passes on the cookie a correct code earned, or on the code itself in a
    header (a script that never saw the form). Constant-time, and empty never matches.
    """
    if not code:
        return True
    if cookie and hmac.compare_digest(str(cookie), access_cookie_value(code)):
        return True
    return token_matches(header, code)


def media_type(path: str | Path) -> str:
    return _MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def is_attachment(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _ATTACHMENT_SUFFIXES


def safe_run_path(run_dir: str | Path, relative: str) -> Path:
    """`run_dir / relative`, or `PathRejected`.

    Rejects, in order: nothing at all, a NUL or a backslash, an absolute path, any `..` or dotfile
    segment, the two never-served top-level directories, a suffix that is not on the allow-list,
    and finally anything that is not inside `run_dir` once every symlink has been resolved. The
    last check is the one that actually holds: the others make the failure readable.
    """
    text = str(relative or "").strip()
    if not text:
        raise PathRejected("no path given")
    if "\x00" in text or "\\" in text:
        raise PathRejected("illegal character in path")
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise PathRejected("absolute paths are not served")

    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts:
        raise PathRejected("no path given")
    if any(p == ".." or p.startswith(".") for p in parts):
        raise PathRejected(f"path segment not allowed: {text!r}")
    if parts[0] in DENIED_TOP_LEVEL:
        raise PathRejected(f"{parts[0]}/ is never served")

    suffix = Path(parts[-1]).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise PathRejected(f"{suffix or 'no'} suffix is not served "
                           f"(allowed: {sorted(ALLOWED_SUFFIXES)})")

    root = Path(run_dir).resolve()
    candidate = (root / Path(*parts)).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise PathRejected("path escapes the run directory")
    return candidate
