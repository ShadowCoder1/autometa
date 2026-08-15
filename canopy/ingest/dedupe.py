"""Group duplicate PDFs (identical bytes, or same DOI / near-identical title) before ingestion."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from .pdf import DOI_RE, _guess_title, sha256_of


@dataclass
class PaperGroup:
    representative: Path
    duplicates: list[Path] = field(default_factory=list)
    sha256: str = ""
    title: str = ""
    doi: str = ""
    reason: str = "unique"     # unique | sha256 | doi | title


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def dedupe_pdfs(paths: list[str | Path], title_threshold: float = 0.93) -> list[PaperGroup]:
    groups: list[PaperGroup] = []
    by_sha: dict[str, PaperGroup] = {}
    for p in sorted(Path(x) for x in paths):
        sha = sha256_of(p)
        if sha in by_sha:
            by_sha[sha].duplicates.append(p)
            by_sha[sha].reason = "sha256"
            continue
        try:
            doc = pymupdf.open(p)
            title = _guess_title(doc)
            first = doc[0].get_text() if len(doc) else ""
            m = DOI_RE.search(first)
            doi = m.group(1).rstrip(".;,)").lower() if m else ""
        except Exception:
            title, doi = "", ""
        matched = None
        for g in groups:
            if doi and g.doi and doi == g.doi:
                matched, g.reason = g, "doi"
                break
            if title and g.title and len(title) > 20 and SequenceMatcher(None, _norm_title(title), _norm_title(g.title)).ratio() >= title_threshold:
                matched, g.reason = g, "title"
                break
        if matched:
            matched.duplicates.append(p)
        else:
            g = PaperGroup(representative=p, sha256=sha, title=title, doi=doi)
            groups.append(g)
            by_sha[sha] = g
    return groups
