"""Read one PDF and report what is in it — as a child process, so it can be killed.

`python -m canopy.server.pdf_probe PAPER.pdf` prints one JSON line:

    {"ok": true, "n_pages": 5, "n_chars": 21033, "n_figures": 2, "has_text_layer": true}

This is the same ingestion the pipeline runs (`canopy.ingest.pdf.ingest_pdf`), minus the page
rasters, which is the expensive half and is not needed to decide whether a file is a paper. The
parent (`canopy.server.uploads.probe_pdf`) gives it a timeout: a PDF crafted to loop forever
takes this process down with it and never touches the server's own event loop.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def probe(path: str | Path) -> dict[str, Any]:
    from ..ingest.pdf import ingest_pdf

    with tempfile.TemporaryDirectory(prefix="canopy-probe-") as scratch:
        paper = ingest_pdf(Path(path), Path(scratch), render_pages=False)
        return {
            "ok": paper.n_pages > 0,
            "n_pages": paper.n_pages,
            "n_chars": sum(page.n_chars for page in paper.pages),
            "n_figures": len(paper.figures),
            "n_tables": len(paper.tables),
            "has_text_layer": bool(paper.has_text_layer),
            "title": (paper.title or "")[:300],
            "error": "" if paper.n_pages else "the file has no pages",
        }


def main(argv: list[str]) -> int:
    if len(argv) != 1:                                     # pragma: no cover - usage error
        print(json.dumps({"ok": False, "error": "usage: pdf_probe PAPER.pdf"}))
        return 2
    try:
        payload = probe(argv[0])
    except Exception as exc:                               # any failure is this file's failure
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300], "n_pages": 0,
                   "n_chars": 0, "n_figures": 0, "has_text_layer": False}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":                                 # pragma: no cover - entry point
    raise SystemExit(main(sys.argv[1:]))
