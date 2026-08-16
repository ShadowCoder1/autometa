"""Accepting a folder of PDFs from a browser, safely.

`stream_upload` is the only way a file gets in, and it checks as it copies — a megabyte at a time,
so a 500-paper folder never exists in memory:

1. **the name** must end in `.pdf` and carry no path of its own (a browser's `webkitdirectory`
   upload sends `some/folder/paper.pdf`, and only the last part is ever used);
2. **the magic bytes** must be `%PDF-` — checked on the first chunk, before the rest is read,
   because the declared content type is the uploader's opinion;
3. **the size** must fit both caps (this file, and everything uploaded so far), and a file that
   exceeds one stops being read at the limit rather than after it.

The file is then stored as `<sha256>.pdf`, which is also how the pipeline identifies a paper, so
two uploads of the same paper are one file and no user-supplied string ever becomes a path.

Both child-process entry points run the **real ingestion** (`canopy.ingest.pdf.ingest_pdf`) with a
timeout: `probe_pdf` to decide whether an upload is a paper at all (page rasters skipped, output
thrown away), and `ingest_pdf_subprocess` as the pipeline's own ingest step. A PDF that loops or
explodes takes the child process with it; the server answers, and the run carries on (amendment I).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["PDF_MAGIC", "UploadRejected", "stream_upload", "probe_pdf", "ingest_pdf_subprocess",
           "safe_filename", "DEFAULT_MAX_UPLOAD_MB", "DEFAULT_MAX_FILES", "DEFAULT_MAX_TOTAL_MB",
           "DEFAULT_PROBE_TIMEOUT", "DEFAULT_INGEST_TIMEOUT", "CHUNK"]

PDF_MAGIC = b"%PDF-"
DEFAULT_MAX_UPLOAD_MB = float(os.environ.get("CANOPY_MAX_UPLOAD_MB", "50") or 50)
DEFAULT_MAX_FILES = int(os.environ.get("CANOPY_MAX_UPLOAD_FILES", "500") or 500)
DEFAULT_MAX_TOTAL_MB = float(os.environ.get("CANOPY_MAX_UPLOAD_TOTAL_MB", "2000") or 2000)
DEFAULT_PROBE_TIMEOUT = float(os.environ.get("CANOPY_PDF_PROBE_TIMEOUT", "60") or 60)
DEFAULT_INGEST_TIMEOUT = float(os.environ.get("CANOPY_INGEST_TIMEOUT", "300") or 300)
#: bytes read from the wire at a time — an upload is never held in memory whole
CHUNK = 1 << 20


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


def stream_upload(source: Any, dest_dir: str | Path, filename: str, *, max_bytes: float,
                  remaining_bytes: float | None = None) -> tuple[Path, int]:
    """Copy one upload to `<sha256>.pdf` a megabyte at a time; never hold the file in memory.

    The name, the magic bytes and both size caps are checked *while* the bytes arrive, so a file
    that is too big stops being read at the limit rather than after it, and a partial file is
    always removed. Returns the path it wrote and how many bytes it was.
    """
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    if not name.lower().endswith(".pdf"):
        raise UploadRejected(f"{name!r} is not a PDF (the name must end in .pdf)")

    digest = hashlib.sha256()
    total = 0
    handle, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".upload-", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as out:
            while True:
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                if total == 0 and not chunk.startswith(PDF_MAGIC[:len(chunk)]):
                    raise UploadRejected(f"{name!r} does not start with %PDF- — it is not a PDF file")
                total += len(chunk)
                if total > max_bytes:
                    raise UploadRejected(
                        f"{name!r} is over the {max_bytes / 1e6:.1f} MB limit for one file "
                        f"(raise it with CANOPY_MAX_UPLOAD_MB)", status_code=413)
                if remaining_bytes is not None and total > remaining_bytes:
                    raise UploadRejected("this upload is over the total size limit "
                                         "(raise it with CANOPY_MAX_UPLOAD_TOTAL_MB)",
                                         status_code=413)
                digest.update(chunk)
                out.write(chunk)
        if total == 0:
            raise UploadRejected(f"{name!r} is empty")
        if total < len(PDF_MAGIC):
            raise UploadRejected(f"{name!r} does not start with %PDF- — it is not a PDF file")
        target = directory / f"{digest.hexdigest()}.pdf"
        os.replace(tmp, target)
        return target, total
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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


def _last_json(stdout: bytes) -> dict[str, Any] | None:
    """The child prints one JSON line last; anything a library printed before it is ignored."""
    for line in reversed(stdout.decode("utf-8", "replace").strip().splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return None


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

    payload = _last_json(completed.stdout)
    if payload is not None:
        return payload
    stderr = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    detail = stderr[-1][:300] if stderr else f"exit code {completed.returncode}"
    return {"ok": False, "error": f"the PDF could not be read: {detail}", "n_pages": 0,
            "n_chars": 0, "n_figures": 0, "has_text_layer": False}


def ingest_pdf_subprocess(pdf: str | Path, out_dir: str | Path,
                          timeout: float = DEFAULT_INGEST_TIMEOUT) -> Any:
    """`canopy.ingest.pdf.ingest_pdf`, in a child process with a timeout.

    This is what the server hands `run_pipeline` as its `ingest_fn`: a PDF built to make a parser
    loop takes the child down with it, the paper ends `error` with the reason, and the run carries
    on. The child writes into `out_dir` exactly what a direct call would have, so `--resume` and
    every later stage cannot tell the difference.
    """
    from ..ingest.pdf import PaperRecord

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "canopy.server.pdf_probe", str(pdf),
               "--ingest-to", str(target)]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=max(0.0, timeout),
                                   env=_child_env(), check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"ingestion timed out after {timeout:g}s: {Path(pdf).name}") from None
    result = _last_json(completed.stdout)
    if not (result or {}).get("ok"):
        detail = ((result or {}).get("error")
                  or (completed.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1]
                  or f"exit code {completed.returncode}")
        raise RuntimeError(f"ingestion failed: {detail[:300]}")
    return PaperRecord.load(target)
