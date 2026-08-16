"""Accepting a folder of PDFs from a browser, safely.

Three checks, in this order, before a byte is kept:

1. **the name** must end in `.pdf` and carry no path of its own (a browser's `webkitdirectory`
   upload sends `some/folder/paper.pdf`, and only the last part is ever used);
2. **the magic bytes** must be `%PDF-` — the declared content type is the uploader's opinion;
3. **the size** must fit the cap (per file and for the whole upload).

Then the file is stored as `<sha256>.pdf`, which is also how the pipeline identifies a paper, so
two uploads of the same paper are one file and no user-supplied string ever becomes a path.

Finally `probe_pdf` runs the **real ingestion** (`canopy.ingest.pdf.ingest_pdf`, page rasters
skipped) in a **child process with a timeout**: a PDF that loops or explodes takes the child with
it and the server answers "this file could not be read" instead of hanging (amendment I).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["PDF_MAGIC", "UploadRejected", "validate_pdf", "save_upload", "probe_pdf",
           "probe_many", "safe_filename", "DEFAULT_MAX_UPLOAD_MB", "DEFAULT_MAX_FILES",
           "DEFAULT_MAX_TOTAL_MB", "DEFAULT_PROBE_TIMEOUT"]

PDF_MAGIC = b"%PDF-"
DEFAULT_MAX_UPLOAD_MB = float(os.environ.get("CANOPY_MAX_UPLOAD_MB", "50") or 50)
DEFAULT_MAX_FILES = int(os.environ.get("CANOPY_MAX_UPLOAD_FILES", "500") or 500)
DEFAULT_MAX_TOTAL_MB = float(os.environ.get("CANOPY_MAX_UPLOAD_TOTAL_MB", "2000") or 2000)
DEFAULT_PROBE_TIMEOUT = float(os.environ.get("CANOPY_PDF_PROBE_TIMEOUT", "60") or 60)


class UploadRejected(Exception):
    """An uploaded file is not something this server will keep. Carries the HTTP status."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def safe_filename(name: str) -> str:
    """The last component of an uploaded name, with nothing that could act as a path."""
    tail = str(name or "").replace("\\", "/").split("/")[-1].strip()
    tail = tail.replace("\x00", "")
    return tail[:200] or "upload.pdf"


def validate_pdf(filename: str, data: bytes, max_bytes: float) -> None:
    """Raise `UploadRejected` unless these bytes are a PDF this server will store."""
    name = safe_filename(filename)
    if not name.lower().endswith(".pdf"):
        raise UploadRejected(f"{name!r} is not a PDF (the name must end in .pdf)")
    if not data:
        raise UploadRejected(f"{name!r} is empty")
    if len(data) > max_bytes:
        raise UploadRejected(
            f"{name!r} is {len(data) / 1e6:.1f} MB, over the {max_bytes / 1e6:.1f} MB limit "
            f"(raise it with CANOPY_MAX_UPLOAD_MB)", status_code=413)
    if not data.startswith(PDF_MAGIC):
        raise UploadRejected(f"{name!r} does not start with %PDF- — it is not a PDF file")


def save_upload(dest_dir: str | Path, data: bytes) -> Path:
    """Store the bytes as `<sha256>.pdf`; the same paper twice is the same file."""
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hashlib.sha256(data).hexdigest()}.pdf"
    if not path.exists():
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- child process
def _child_env() -> dict[str, str]:
    """The child needs to import `canopy`, installed or not."""
    import canopy

    env = dict(os.environ)
    package_parent = str(Path(canopy.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{package_parent}{os.pathsep}{existing}" if existing else package_parent
    env.setdefault("MPLBACKEND", "Agg")
    return env


def probe_pdf(path: str | Path, timeout: float = DEFAULT_PROBE_TIMEOUT) -> dict[str, Any]:
    """Ingest one PDF in a child process. Never raises; never hangs longer than `timeout`.

    Returns `{"ok": bool, "n_pages": int, "n_chars": int, "n_figures": int, "has_text_layer":
    bool, "error": str}`. A file that times out, crashes the child or produces no readable page
    is `ok=False` with the reason — the caller turns that into a 400, not a 500.
    """
    command = [sys.executable, "-m", "canopy.server.pdf_probe", str(path)]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=max(0.0, timeout),
                                   env=_child_env(), check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"reading the PDF timed out after {timeout:g}s",
                "n_pages": 0, "n_chars": 0, "n_figures": 0, "has_text_layer": False}
    except OSError as exc:                                 # pragma: no cover - no interpreter
        return {"ok": False, "error": f"could not start the PDF reader: {exc}",
                "n_pages": 0, "n_chars": 0, "n_figures": 0, "has_text_layer": False}

    text = completed.stdout.decode("utf-8", "replace").strip().splitlines()
    for line in reversed(text):                            # the child prints one JSON line last
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    stderr = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    detail = stderr[-1][:300] if stderr else f"exit code {completed.returncode}"
    return {"ok": False, "error": f"the PDF could not be read: {detail}", "n_pages": 0,
            "n_chars": 0, "n_figures": 0, "has_text_layer": False}


def probe_many(paths: Sequence[Path] | Iterable[Path], timeout: float = DEFAULT_PROBE_TIMEOUT,
               workers: int = 4) -> dict[str, dict[str, Any]]:
    """`probe_pdf` over several files at once (each still in its own child)."""
    items = list(paths)
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        results = list(pool.map(lambda p: probe_pdf(p, timeout), items))
    return {str(path): result for path, result in zip(items, results)}
