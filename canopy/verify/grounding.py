"""Quote grounding: is this candidate's evidence actually printed in the paper?

Every extractor returns a verbatim quote with its number. Grounding checks that quote against the
deterministic page text from ingestion, because a quote that is not on the page means the value
beside it was invented — a red flag, not a value.

Matching has to survive what a PDF does to a sentence and nothing more:

* NFKC (ligatures, full-width forms, non-breaking and thin spaces);
* every dash/minus variant folded to ASCII `-`, so "−0.5" and "-0.5" are one number;
* words hyphenated across a line break re-joined ("adap-\\ntation" -> "adaptation");
* numeric spacing normalised: "42.5 ± 6.9" == "42.5±6.9", "1 779" == "1779";
* whitespace collapsed, case folded.

Exact substring wins (similarity 1.0). Otherwise the closest window of the page of the quote's own
length is scored with `difflib.SequenceMatcher`; >= `THRESHOLD` counts as grounded. Anything else
returns the best window it found, so a reviewer can see what the page came closest to saying.

`ground_candidate` additionally repairs an off-by-one page (the mapper's page and the printed page
number often differ by the offset of a journal's front matter), sweeps the whole document as a last
resort, and — for a value the extractor says it read out of a table row — checks that the value
really is in that row of that table.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from ..ingest.pdf import PaperRecord, TableRecord
from ..models import Candidate

__all__ = ["ground_quote", "ground_candidate", "check_table_cell", "normalize", "THRESHOLD"]

#: fuzzy similarity at or above which a quote counts as grounded (plan: 0.95)
THRESHOLD = 0.95
#: how far either side of the best anchor the window is re-tried (characters)
_SHIFTS = (0, -2, 2, -5, 5, -10, 10, -20, 20, -40, 40)

_DASHES = "‐‑‒–—―⁃−－˗"
_DASH_RE = re.compile(f"[{_DASHES}]")
_HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_PLUSMINUS_RE = re.compile(r"\s*±\s*")
_PM_ASCII_RE = re.compile(r"\+\s*/\s*-")
_DIGIT_SPACE_RE = re.compile(r"(?<=\d)[  ](?=\d{3}(?!\d))")
_DECIMAL_SPACE_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """The one normalisation both sides of every comparison go through (see the module docstring)."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _PM_ASCII_RE.sub("±", out)
    out = _DASH_RE.sub("-", out)
    out = _HYPHEN_BREAK_RE.sub("", out)          # before whitespace collapse: needs the newline
    out = _PLUSMINUS_RE.sub("±", out)
    out = _DIGIT_SPACE_RE.sub("", out)           # "1 779" -> "1779"
    out = _DECIMAL_SPACE_RE.sub(".", out)        # "42. 5" -> "42.5"
    return _WS_RE.sub(" ", out).strip().casefold()


def _best_window(needle: str, hay: str) -> tuple[float, str]:
    """The window of `hay` of the needle's length that scores highest, and its ratio.

    The longest common block anchors the search (one O(n·m) pass), then a handful of shifts around
    that anchor are scored — a plain sliding window over a whole page is quadratic for no gain.
    """
    if not needle or not hay:
        return 0.0, ""
    size = len(needle)
    matcher = SequenceMatcher(None, needle, hay, autojunk=False)
    block = matcher.find_longest_match(0, size, 0, len(hay))
    anchor = max(0, min(block.b - block.a, len(hay) - size))
    starts = {max(0, min(anchor + shift, max(0, len(hay) - size))) for shift in _SHIFTS}
    best_ratio, best_text = 0.0, hay[anchor:anchor + size]
    for start in sorted(starts):
        window = hay[start:start + size]
        ratio = SequenceMatcher(None, needle, window, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio, best_text = ratio, window
    return best_ratio, best_text


def ground_quote(quote: str, page_text: str) -> tuple[bool, float, str]:
    """`(grounded, similarity, matched_text)` for one quote against one page's text."""
    needle, hay = normalize(quote), normalize(page_text)
    if not needle or not hay:
        return False, 0.0, ""
    if needle in hay:
        return True, 1.0, needle
    ratio, window = _best_window(needle, hay)
    return ratio >= THRESHOLD, round(ratio, 4), window


# ----------------------------------------------------------------------------- table cells
def _row_text(row: list[str]) -> str:
    return normalize(" | ".join(str(cell or "") for cell in row))


def _column_index(table: TableRecord, col_header: str) -> int | None:
    """The column whose header cell names `col_header` (headers are usually the first row)."""
    wanted = normalize(col_header)
    if not wanted:
        return None
    for row in table.rows[:2]:
        for index, cell in enumerate(row):
            cell_text = normalize(str(cell or ""))
            if cell_text and (cell_text == wanted or wanted in cell_text or cell_text in wanted):
                return index
    return None


def _find_row(paper: PaperRecord, page: int | None, row_header: str
              ) -> tuple[TableRecord, list[str]] | None:
    """The ingested table row whose cells name `row_header` — tables on `page` are tried first."""
    wanted = normalize(row_header)
    if not wanted:
        return None
    tables = sorted(paper.tables, key=lambda t: (page is None or t.page != page, t.id))
    for table in tables:
        for row in table.rows:
            cells = [normalize(str(cell or "")) for cell in row]
            if any(cell and (cell == wanted or wanted in cell or cell in wanted) for cell in cells):
                return table, list(row)
    return None


def check_table_cell(cand: Candidate, paper: PaperRecord) -> tuple[bool | None, str]:
    """Is the transcribed value really in the table row the extractor named?

    Returns `(None, "")` when the candidate claims no table cell, `(None, reason)` when ingestion
    detected no row of that name (imperfect table detection is not evidence against the quote), and
    `(True/False, detail)` when the row was found and the value was or was not in it.
    """
    if not (cand.row_header or cand.col_header):
        return None, ""
    value = cand.value_as_written or ("" if cand.mean is None else f"{cand.mean:g}")
    if not value:
        return None, "no transcribed value to look for"
    if not cand.row_header:
        return None, f"no row header for column {cand.col_header!r}"
    found = _find_row(paper, cand.page, cand.row_header)
    if found is None:
        return None, (f"no ingested table has a row named {cand.row_header!r} "
                      f"(ingestion found {len(paper.tables)} table(s))")
    table, row = found
    wanted = normalize(value)
    index = _column_index(table, cand.col_header)
    if index is not None and index < len(row):
        cell = normalize(str(row[index] or ""))
        if wanted and wanted in cell:
            return True, f"{table.id} row {cand.row_header!r} column {cand.col_header!r}"
    if wanted and wanted in _row_text(row):
        return True, f"{table.id} row {cand.row_header!r}"
    return False, (f"{value!r} is not in {table.id} row {cand.row_header!r} "
                   f"(row reads: {_row_text(row)[:160]!r})")


# ----------------------------------------------------------------------------- candidates
def _note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _pages_to_try(paper: PaperRecord, page: int | None) -> list[int]:
    """The named page, then ±1, then every other page — the order the plan asks for."""
    count = len(paper.pages)
    order: list[int] = []
    for candidate in ([page, (page or 0) - 1, (page or 0) + 1] if page else []):
        if isinstance(candidate, int) and 1 <= candidate <= count and candidate not in order:
            order.append(candidate)
    order += [n for n in range(1, count + 1) if n not in order]
    return order


def ground_candidate(cand: Candidate, paper: PaperRecord) -> Candidate:
    """Set `grounded` / `grounding_similarity` (and repair `page`) on one candidate, in place.

    A quote found on a neighbouring page or elsewhere in the document is still evidence — the page
    is corrected, `page_corrected` records that it moved, and the note says where it came from. A
    quote found nowhere leaves the candidate ungrounded with the best similarity it reached.
    """
    if cand.quote.strip():
        named = cand.page
        best = (0.0, named, "")
        for number in _pages_to_try(paper, named):
            grounded, similarity, _ = ground_quote(cand.quote, paper.page_text(number))
            if grounded:
                cand.grounded = True
                cand.grounding_similarity = similarity
                if number != named:
                    cand.page = number
                    cand.page_corrected = True
                    cand.notes = _note(cand.notes, f"quote found on page {number}"
                                                   f"{f' (extractor said page {named})' if named else ''}")
                break
            if similarity > best[0]:
                best = (similarity, number, "")
        else:
            cand.grounded = False
            cand.grounding_similarity = best[0]
            cand.notes = _note(cand.notes, f"quote not found in the document (best similarity "
                                           f"{best[0]:.2f} on page {best[1]})")

    ok, detail = check_table_cell(cand, paper)
    if ok is False:
        cand.grounded = False
        cand.notes = _note(cand.notes, f"table cell check failed: {detail}")
    elif ok is None and detail:
        cand.notes = _note(cand.notes, f"table cell check skipped: {detail}")
    return cand
