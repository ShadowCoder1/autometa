"""Ingest one PDF in a child process, so a file built to hang a parser can be killed.

    python -m canopy.server.pdf_probe PAPER.pdf                 # check it, keep nothing
    python -m canopy.server.pdf_probe PAPER.pdf --ingest-to DIR # the real thing, page rasters and all

Either way it prints one JSON line:

    {"ok": true, "n_pages": 5, "n_chars": 21033, "n_figures": 2, "has_text_layer": true}

Both modes run `canopy.ingest.pdf.ingest_pdf` — the same ingestion the pipeline runs. Without
`--ingest-to` the page rasters are skipped (the expensive half, and not needed to decide whether a
file is a paper) and the output is thrown away; with it, the run directory gets exactly what a
direct call would have written. The parent (`canopy.server.uploads`) gives the child a timeout:
a PDF crafted to loop forever takes this process down with it and never touches the server.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def probe(path: str | Path, ingest_to: str | Path | None = None) -> dict[str, Any]:
    """Ingest `path`. Into `ingest_to` (with page rasters) when given, otherwise into a scratch
    directory that is deleted again."""
    from contextlib import nullcontext

    from ..ingest.pdf import ingest_pdf

    keep = ingest_to is not None
    holder = (nullcontext(str(ingest_to)) if keep
              else tempfile.TemporaryDirectory(prefix="canopy-probe-"))
    with holder as scratch:
        paper = ingest_pdf(Path(path), Path(scratch), render_pages=keep)
        return {
            "ok": paper.n_pages > 0,
            "n_pages": paper.n_pages,
            "n_chars": sum(page.n_chars for page in paper.pages),
            "n_figures": len(paper.figures),
            "n_tables": len(paper.tables),
            "has_text_layer": bool(paper.has_text_layer),
            "title": (paper.title or "")[:300],
            "out_dir": str(Path(scratch)) if keep else "",
            "error": "" if paper.n_pages else "the file has no pages",
        }


def main(argv: list[str]) -> int:
    if not argv or (len(argv) == 3 and argv[1] != "--ingest-to") or len(argv) not in (1, 3):
        print(json.dumps({"ok": False,                     # pragma: no cover - usage error
                          "error": "usage: pdf_probe PAPER.pdf [--ingest-to DIR]"}))
        return 2
    try:
        payload = probe(argv[0], argv[2] if len(argv) == 3 else None)
    except Exception as exc:                               # any failure is this file's failure
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300], "n_pages": 0,
                   "n_chars": 0, "n_figures": 0, "has_text_layer": False}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":                                 # pragma: no cover - entry point
    raise SystemExit(main(sys.argv[1:]))
