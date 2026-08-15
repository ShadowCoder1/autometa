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

import math
import re
import unicodedata
from difflib import SequenceMatcher

from ..ingest.pdf import PaperRecord, TableRecord
from ..models import Candidate

__all__ = ["ground_quote", "ground_candidate", "check_table_cell", "is_short_quote", "normalize",
           "numbers_in", "THRESHOLD", "SHORT_QUOTE_CHARS", "SHORT_QUOTE_NOTE", "CELL_CONFIRMED",
           "ROW_ONLY", "SIGN_NOTE"]

#: fuzzy similarity at or above which a quote counts as grounded (plan: 0.95)
THRESHOLD = 0.95
#: how far either side of the best anchor the window is re-tried (characters)
_SHIFTS = (0, -2, 2, -5, 5, -10, 10, -20, 20, -40, 40)
#: shortest header text that may match a cell by containment ("SD" must not match "standard")
MIN_HEADER_CHARS = 3
#: a quote shorter than this grounds too easily to be evidence on its own — Task 8 reads the marker
SHORT_QUOTE_CHARS = 20
#: the note a short but grounded quote carries (a stable marker, not prose)
SHORT_QUOTE_NOTE = "short quote"
#: markers in a table cell check's detail: fully confirmed / only elsewhere in the row / sign only
CELL_CONFIRMED = "cell-confirmed"
ROW_ONLY = "row-only"
SIGN_NOTE = "sign not confirmed"

_DASHES = "‐‑‒–—―⁃−－˗"
_DASH_RE = re.compile(f"[{_DASHES}]")
_HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_PLUSMINUS_RE = re.compile(r"\s*±\s*")
_PM_ASCII_RE = re.compile(r"\+\s*/\s*-")
_DIGIT_SPACE_RE = re.compile(r"(?<=\d)[  ](?=\d{3}(?!\d))")
#: a comma between digits is a thousands separator only when exactly three digits follow it
#: ("1,779" -> "1779"). Anything else is left alone, because "27,4" is a European decimal and
#: "F(1,22)" is a pair of degrees of freedom; both sides of every comparison are normalised
#: the same way, and `_decimal_comma_readings` covers the European form where it matters.
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
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
    out = _DIGIT_COMMA_RE.sub("", out)           # "1,779" -> "1779"
    out = _DECIMAL_SPACE_RE.sub(".", out)        # "42. 5" -> "42.5"
    return _WS_RE.sub(" ", out).strip().casefold()


def is_short_quote(quote: str) -> bool:
    """True when a quote is too short to be evidence on its own.

    "42.5" is in a page a dozen times; grounding it proves nothing. Such a quote still grounds (the
    number really is printed), but the candidate carries `SHORT_QUOTE_NOTE` so Task 8 can weigh it.
    """
    return 0 < len(normalize(quote)) < SHORT_QUOTE_CHARS


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
def _names(cell: str, wanted: str) -> bool:
    """Does this cell name that header? Equal, or one contains the other — but a one- or two-letter
    cell never matches by containment, or every row would match every header."""
    if not cell or not wanted:
        return False
    if cell == wanted:
        return True
    return ((len(cell) >= MIN_HEADER_CHARS and cell in wanted)
            or (len(wanted) >= MIN_HEADER_CHARS and wanted in cell))


def _row_text(row: list[str]) -> str:
    return normalize(" | ".join(str(cell or "") for cell in row))


def _column_index(table: TableRecord, col_header: str) -> int | None:
    """The column whose header cell names `col_header` (headers are usually the first row)."""
    wanted = normalize(col_header)
    if not wanted:
        return None
    for row in table.rows[:2]:
        for index, cell in enumerate(row):
            if _names(normalize(str(cell or "")), wanted):
                return index
    return None


def _find_row(paper: PaperRecord, page: int | None, row_header: str
              ) -> tuple[TableRecord, list[str], bool] | None:
    """The ingested table row whose cells name `row_header`, and whether it is where it should be.

    A table more than one page from the one the extractor named is almost certainly a different
    table with a similar row label, so the third element is False and the caller only notes it —
    a cross-page hit must never refute a candidate.
    """
    wanted = normalize(row_header)
    if not wanted:
        return None
    def near(table: TableRecord) -> bool:
        return page is None or abs(table.page - page) <= 1

    for tables in ([t for t in paper.tables if near(t)], [t for t in paper.tables if not near(t)]):
        for table in sorted(tables, key=lambda t: t.id):
            for row in table.rows:
                if any(_names(normalize(str(cell or "")), wanted) for cell in row):
                    return table, list(row), near(table)
    return None


#: A sign belongs to a number only when nothing numeric precedes it: in "0.41-0.83" the dash is a
#: range, in "-0.5" it is a minus. `)` and `]` close a previous value, so they end a number too.
_NUMBER_RE = re.compile(r"(?<![\d)\]])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
#: "95% CI" states a confidence level, not a measurement — it must not have to appear in a table row
CI_LEVEL_TOKENS = frozenset({90.0, 95.0, 99.0})
_PERCENT_RE = re.compile(r"\s*%")
#: a European decimal a table may print ("27,4" is 27.4); read *in addition to* the plain tokens
_DECIMAL_COMMA_RE = re.compile(r"(?<!\d)(\d+),(\d{1,2})(?!\d)")


def numbers_in(text: str, drop_ci_levels: bool = False) -> list[float]:
    """Every number in a piece of text, as floats, after the usual normalisation.

    With `drop_ci_levels`, an integer 90/95/99 written as a percentage is skipped: "0.62 (95% CI
    0.41, 0.83)" is three numbers, not four, and a table cell holding the same interval need not
    repeat the level.
    """
    normalised = normalize(text)
    found: list[float] = []
    for match in _NUMBER_RE.finditer(normalised):
        try:
            value = float(match.group())
        except ValueError:                                   # pragma: no cover - regex is strict
            continue
        if (drop_ci_levels and value in CI_LEVEL_TOKENS
                and _PERCENT_RE.match(normalised, match.end())):
            continue
        found.append(value)
    return found


def _decimal_comma_readings(text: str) -> list[float]:
    """The European reading of any "27,4" in the text — added to a row's numbers, never demanded."""
    return [float(f"{whole}.{part}") for whole, part in _DECIMAL_COMMA_RE.findall(normalize(text))]


def _row_numbers(text: str) -> list[float]:
    """Every number a table row could be offering, read generously (the row is the haystack)."""
    return numbers_in(text, drop_ci_levels=True) + _decimal_comma_readings(text)


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _holds(value: float, present: list[float]) -> bool:
    return any(_same(value, other) for other in present)


def _transcribed_numbers(cand: Candidate) -> list[float]:
    """What this candidate says it read, as numbers: the fields plus anything in the raw string.

    Only the numbers matter. "42.5 ± 6.9 s" and a row holding `42.5 | 6.9` are the same reading;
    demanding the whole string back would refute correct transcriptions on formatting alone.
    """
    wanted: list[float] = []
    for value in (cand.mean, cand.dispersion_value, cand.ci_low, cand.ci_high):
        if value is not None and not any(_same(value, seen) for seen in wanted):
            wanted.append(float(value))
    for value in numbers_in(cand.value_as_written, drop_ci_levels=True):
        if not any(_same(value, seen) for seen in wanted):
            wanted.append(value)
    return wanted


def _confirms(wanted: list[float], present: list[float]) -> tuple[bool, list[float]]:
    """Does `present` hold every wanted number? Also: which ones matched only in absolute value.

    A paper often prints a magnitude where the extractor read a signed value (or the other way
    round). That is a difference worth reporting, not a reason to call the transcription invented.
    """
    unsigned: list[float] = []
    for value in wanted:
        if _holds(value, present):
            continue
        if _holds(abs(value), [abs(other) for other in present]):
            unsigned.append(value)
            continue
        return False, unsigned
    return True, unsigned


def check_table_cell(cand: Candidate, paper: PaperRecord) -> tuple[bool | None, str]:
    """Are the numbers this candidate transcribed really in the table row it named?

    `(None, "")` when the candidate claims no table cell; `(None, reason)` when the check could not
    be made (nothing numeric to look for, no such row anywhere, or the only such row is on a
    different page) — imperfect table detection is not evidence against a quote; `(True/False,
    detail)` otherwise.

    The detail always says at what level the numbers were confirmed: `cell-confirmed` when they are
    in the cell where the named row and column meet, `ROW_ONLY` when they are only somewhere else
    in that row (which is what a group mix-up looks like — still grounded, but Task 8 should know),
    `row-confirmed` when no column was named or found.
    """
    if not (cand.row_header or cand.col_header):
        return None, ""
    wanted = _transcribed_numbers(cand)
    if not wanted:
        return None, "no transcribed number to look for"
    if not cand.row_header:
        return None, f"no row header for column {cand.col_header!r}"
    found = _find_row(paper, cand.page, cand.row_header)
    if found is None:
        return None, (f"no ingested table has a row named {cand.row_header!r} "
                      f"(ingestion found {len(paper.tables)} table(s))")
    table, row, near = found
    if not near:
        return None, (f"the only row named {cand.row_header!r} is in {table.id} on page "
                      f"{table.page}, not near page {cand.page}")

    row_text = _row_text(row)
    where = f"{table.id} row {cand.row_header!r}"
    in_row, unsigned = _confirms(wanted, _row_numbers(row_text))
    if not in_row:
        missing = [v for v in wanted if not _holds(v, _row_numbers(row_text))]
        return False, f"{missing} not in {where} (row reads: {row_text[:160]!r})"
    suffix = f"; {SIGN_NOTE} for {unsigned}" if unsigned else ""

    index = _column_index(table, cand.col_header)
    if index is None or index >= len(row):
        return True, f"row-confirmed ({where}){suffix}"
    in_cell, _ = _confirms(wanted, _row_numbers(str(row[index] or "")))
    if in_cell:
        return True, f"{CELL_CONFIRMED} ({where}, column {cand.col_header!r}){suffix}"
    return True, (f"{ROW_ONLY}: found in {where} but not in column {cand.col_header!r} "
                  f"(that cell reads {str(row[index] or '')!r}){suffix}")


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
        best: tuple[float, int | None] = (0.0, None)
        for number in _pages_to_try(paper, named):
            grounded, similarity, _ = ground_quote(cand.quote, paper.page_text(number))
            if grounded:
                cand.grounded = True
                cand.grounding_similarity = similarity
                if number != named:
                    said = f" (extractor said page {named})" if named else ""
                    cand.page = number
                    cand.page_corrected = True
                    cand.notes = _note(cand.notes, f"quote found on page {number}{said}")
                if is_short_quote(cand.quote):
                    cand.notes = _note(cand.notes, f"{SHORT_QUOTE_NOTE}: "
                                                   f"{len(normalize(cand.quote))} characters "
                                                   f"ground too easily to stand as evidence")
                break
            if similarity > best[0]:
                best = (similarity, number)
        else:
            cand.grounded = False
            cand.grounding_similarity = best[0]
            where = f" on page {best[1]}" if best[1] else ""
            cand.notes = _note(cand.notes, f"quote not found in the document "
                                           f"(best similarity {best[0]:.2f}{where})")

    ok, detail = check_table_cell(cand, paper)
    if ok is False:
        cand.grounded = False
        cand.notes = _note(cand.notes, f"table cell check failed: {detail}")
    elif ok is None and detail:
        cand.notes = _note(cand.notes, f"table cell check skipped: {detail}")
    elif ok is True and (detail.startswith(ROW_ONLY) or SIGN_NOTE in detail):
        cand.notes = _note(cand.notes, f"table cell check: {detail}")
    return cand
