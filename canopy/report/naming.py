"""File names and link targets that survive being used as URLs.

Every value in this review is identified by a candidate id like
`d1:late_adaptation:A:text:table_first:claude-opus-5#1`. Two characters in that string are fatal
once it becomes a file name a browser has to fetch: `:` (Windows, and a scheme separator in a
relative URL) and `#`, which a browser truncates at — a link to `…claude-opus-5#1.png` asks the
server for `…claude-opus-5` and jumps to the fragment `1.png`. That is how every per-value
provenance link in the first cut of the HTML report came to be dead.

So there is exactly ONE sanitiser (`safe_name`) and exactly one way to build a link
(`url_path`), and both live here rather than being re-invented per module:

* `safe_name` — what a file on disk may be called: `[A-Za-z0-9._-]`, everything else folded to `_`.
* `url_path` — what an `href`/`src` may contain: the *relative path* percent-encoded per segment,
  so a name that still holds a space, a `#` or a `%` reaches the file it names. HTML-escaping is a
  separate, later step (the caller's `_e`), because escaping and encoding solve different problems.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

__all__ = ["safe_name", "url_path", "UNSAFE"]

#: everything a file name may NOT contain — candidate ids carry `:` and `#`, paths must not
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """`d1:late:A:digitize:D#2` → `d1_late_A_digitize_D_2` — a name a URL can survive."""
    return UNSAFE.sub("_", str(name)).strip("_") or "x"


def url_path(path: str | Path) -> str:
    """A relative path as a URL: separators kept, everything else percent-encoded.

    `quote` leaves `/` alone and encodes `#`, `?`, `%` and spaces, which is exactly the difference
    between a link that opens the file and one that opens a 404.
    """
    return quote(str(path).replace("\\", "/"), safe="/")
