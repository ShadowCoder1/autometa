"""Task 5 (half 1): quote grounding.

A candidate that is not grounded in the page text is a red flag, not a value — so the grounder has
to survive everything a PDF does to a sentence (unicode minus, thin spaces, ± spacing, words
hyphenated across a line break) while still rejecting a quote the paper never printed.
"""
from __future__ import annotations

import dataclasses

from canopy.ingest.pdf import Bbox, PaperRecord, TableRecord
from canopy.models import Candidate
from canopy.verify.grounding import (SHORT_QUOTE_CHARS, SHORT_QUOTE_NOTE, check_table_cell,
                                     ground_candidate, ground_quote, is_short_quote, normalize,
                                     numbers_in)

# `paper` (Bock 2005, ingested once per session) comes from tests/conftest.py.


# ------------------------------------------------------------------ normalisation
def test_unicode_minus_matches_an_ascii_hyphen():
    grounded, similarity, match = ground_quote("the offset was -0.5 deg",
                                               "In the last block the offset was −0.5 deg.")
    assert grounded is True and similarity == 1.0
    assert "-0.5" in match


def test_a_word_hyphenated_across_a_line_break_is_joined():
    page = "the rate of adap-\ntation was slower in the second block"
    grounded, similarity, _ = ground_quote("the rate of adaptation was slower", page)
    assert grounded is True and similarity == 1.0


def test_plus_minus_spacing_and_line_breaks_are_ignored():
    page = "the completion time was 42.5\n±6.9\ns for the second group"
    grounded, _, _ = ground_quote("the completion time was 42.5 ± 6.9 s", page)
    assert grounded is True


def test_a_space_inside_a_number_is_ignored():
    grounded, _, _ = ground_quote("movement time was 1779 ms",
                                  "movement time was 1 779 ms in the last block")
    assert grounded is True


def test_normalisation_is_stable_and_case_insensitive():
    assert normalize("  A B–C  ") == normalize("a b-c")


def test_a_near_miss_still_grounds_above_the_threshold():
    page = "Their errors increased abruptly at the onset of the adaptation phase, and then " \
           "gradually dropped again."
    quote = "Their errors increased abruptly at the onset of the adaptation phase, and then " \
            "gradually dropped again"
    grounded, similarity, _ = ground_quote(quote, page)
    assert grounded is True and 0.95 <= similarity <= 1.0


def test_an_invented_quote_is_rejected_and_the_closest_text_is_returned(paper):
    quote = "the older group averaged 42.0 ± 3.1 deg over the last two episodes"
    grounded, similarity, match = ground_quote(quote, paper.page_text(3))
    assert grounded is False
    assert similarity < 0.95
    assert match                                   # the reviewer sees what the page came closest to


def test_an_empty_quote_never_grounds():
    assert ground_quote("", "any page text") == (False, 0.0, "")


def test_a_real_sentence_from_the_paper_grounds_verbatim(paper):
    quote = ("the completion time for young subjects was 27.4±7.2 s and that for old "
             "subjects was 42.5±6.9 s")
    grounded, similarity, _ = ground_quote(quote, paper.page_text(3))
    assert grounded is True and similarity == 1.0


# ------------------------------------------------------------------ ground_candidate
def _candidate(**over) -> Candidate:
    data = {"kind": "group_stats", "group": "A", "page": 3,
            "quote": "An exponential fit yielded y=9.92+53.51×exp(−x/5.89) for young "
                     "subjects"}
    data.update(over)
    return Candidate(**data)


def test_ground_candidate_confirms_the_page_the_model_named(paper):
    cand = ground_candidate(_candidate(), paper)
    assert cand.grounded is True
    assert cand.grounding_similarity == 1.0
    assert cand.page == 3 and cand.page_corrected is False
    assert not cand.notes


def test_ground_candidate_corrects_the_page_by_one(paper):
    cand = ground_candidate(_candidate(page=2), paper)
    assert cand.grounded is True and cand.page == 3
    assert cand.page_corrected is True
    assert "page 3" in cand.notes and "page 2" in cand.notes


def test_ground_candidate_falls_back_to_the_whole_document(paper):
    """Four pages away: only the whole-document sweep can find it."""
    quote = "Mean tracking errors of young (triangles) and old (squares) subjects"
    cand = ground_candidate(_candidate(page=1, quote=quote), paper)
    assert cand.grounded is True and cand.page == 4
    assert cand.page_corrected is True


def test_ground_candidate_records_a_quote_the_paper_never_printed(paper):
    cand = ground_candidate(_candidate(quote="the old group reached 42.0 deg by episode 20"),
                            paper)
    assert cand.grounded is False
    assert 0.0 < (cand.grounding_similarity or 0.0) < 0.95
    assert cand.page == 3 and cand.page_corrected is False      # nothing was moved
    assert "not found" in cand.notes


def test_a_candidate_without_a_quote_is_not_graded(paper):
    cand = ground_candidate(_candidate(quote="", status="not_on_these_pages"), paper)
    assert cand.grounded is None and cand.grounding_similarity is None
    assert not cand.notes


def test_a_candidate_with_a_page_outside_the_document_still_searches(paper):
    cand = ground_candidate(_candidate(page=99), paper)
    assert cand.grounded is True and cand.page == 3 and cand.page_corrected is True


# ------------------------------------------------------------------ table cells
def _with_table(paper: PaperRecord, rows: list[list[str]], page: int = 3) -> PaperRecord:
    table = TableRecord(id=f"p{page}t1", page=page, bbox=Bbox(0, 0, 1, 1),
                        caption="Table 1 Group means", rows=rows)
    return dataclasses.replace(paper, tables=[table])


ROWS = [["Variable", "Young", "Old"],
        ["Trail making (s)", "27.4 (7.2)", "42.5 (6.9)"],
        ["Reaction time (ms)", "310 (44)", "402 (61)"]]
#: a sentence that really is on page 3, so the *quote* always grounds and the cell check is what
#: the three tests below actually vary
TABLE_QUOTE = "the completion time for young subjects was 27.4±7.2 s"


def test_a_table_cell_is_confirmed_in_its_own_row(paper):
    with_table = _with_table(paper, ROWS)
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="42.5 (6.9)", mean=42.5)
    ok, detail = check_table_cell(cand, with_table)
    assert ok is True, detail


def test_a_value_that_is_not_in_the_named_row_fails_the_cell_check(paper):
    with_table = _with_table(paper, ROWS)
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="402 (61)", mean=402.0)
    ok, detail = check_table_cell(cand, with_table)
    assert ok is False and "Trail making" in detail


def test_a_failed_cell_check_makes_the_candidate_ungrounded(paper):
    with_table = _with_table(paper, ROWS)
    cand = ground_candidate(
        _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                   value_as_written="402 (61)", mean=402.0), with_table)
    assert cand.grounded is False                  # the sentence grounds, the cell does not
    assert "table cell check" in cand.notes


def test_a_row_no_ingested_table_contains_only_leaves_a_note(paper):
    """Table detection is imperfect: an unfindable row is a note, not a refutation of the quote."""
    with_table = _with_table(paper, ROWS)
    cand = ground_candidate(
        _candidate(quote=TABLE_QUOTE, row_header="Grip force (N)", col_header="Old",
                   value_as_written="12.0", mean=12.0), with_table)
    assert cand.grounded is True
    assert "table cell check" in cand.notes and "Grip force" in cand.notes


def test_a_candidate_without_headers_skips_the_cell_check(paper):
    ok, detail = check_table_cell(_candidate(value_as_written="9.92"), paper)
    assert ok is None and detail == ""


def test_a_one_letter_cell_does_not_match_every_header(paper):
    """Containment matching must not turn a stray 'a' cell into the row the extractor named."""
    with_table = _with_table(paper, [["Variable", "Young", "Old"], ["a", "1.0", "2.0"],
                                     ["Trail making (s)", "27.4 (7.2)", "42.5 (6.9)"]])
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="42.5 (6.9)", mean=42.5)
    ok, detail = check_table_cell(cand, with_table)
    assert ok is True and "Trail making" in detail


# ------------------------------------------------------------------ cells match by their numbers
PM_ROWS = [["Variable", "Young", "Old"],
           ["Trail making (s)", "27.4±7.2", "42.5±6.9"]]
SPLIT_ROWS = [["Variable", "Young M", "Young SD", "Old M", "Old SD"],
              ["Trail making (s)", "27.4", "7.2", "42.5", "6.9"]]


def test_formatting_differences_between_the_quote_and_the_cell_do_not_refute(paper):
    """"42.5 ± 6.9 s" and a cell reading `42.5±6.9` are the same reading."""
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="42.5 ± 6.9 s", mean=42.5, dispersion_value=6.9)
    ok, detail = check_table_cell(cand, _with_table(paper, PM_ROWS))
    assert ok is True, detail


def test_a_mean_and_a_dispersion_in_separate_columns_still_confirm_the_row(paper):
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old M",
                      value_as_written="42.5 (6.9)", mean=42.5, dispersion_value=6.9)
    ok, detail = check_table_cell(cand, _with_table(paper, SPLIT_ROWS))
    assert ok is True, detail


def test_a_number_that_is_not_in_the_row_still_fails(paper):
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="41.0 (6.9)", mean=41.0, dispersion_value=6.9)
    ok, detail = check_table_cell(cand, _with_table(paper, PM_ROWS))
    assert ok is False and "41.0" in detail


def test_a_confidence_interval_is_checked_by_its_bounds(paper):
    rows = [["Measure", "Old"], ["Trail making (s)", "42.5 [40.1, 44.9]"]]
    cand = _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                      value_as_written="42.5 [40.1, 44.9]", mean=42.5, ci_low=40.1, ci_high=44.9)
    ok, detail = check_table_cell(cand, _with_table(paper, rows))
    assert ok is True, detail


def test_a_row_of_the_same_name_on_a_far_away_page_is_only_a_note(paper):
    """Two tables can share a row label; the wrong one must never refute a candidate."""
    with_table = _with_table(paper, PM_ROWS, page=1)
    cand = ground_candidate(
        _candidate(quote=TABLE_QUOTE, row_header="Trail making (s)", col_header="Old",
                   value_as_written="99.9 (1.1)", mean=99.9), with_table)
    assert cand.grounded is True
    assert "table cell check skipped" in cand.notes and "not near page 3" in cand.notes


def test_a_row_on_the_neighbouring_page_is_still_checked(paper):
    with_table = _with_table(paper, PM_ROWS, page=2)
    ok, detail = check_table_cell(
        _candidate(row_header="Trail making (s)", col_header="Old", value_as_written="42.5±6.9",
                   mean=42.5, dispersion_value=6.9), with_table)
    assert ok is True, detail


def test_numbers_are_read_out_of_any_formatting():
    assert numbers_in("42.5 ± 6.9 s") == [42.5, 6.9]
    assert numbers_in("42.5\u00b16.9") == [42.5, 6.9]
    assert numbers_in("the offset was \u22120.5 deg") == [-0.5]      # unicode minus keeps its sign
    assert numbers_in("y=9.92+53.51\u00d7exp(-x/5.89)") == [9.92, 53.51, 5.89]


# ------------------------------------------------------------------ short quotes
def test_a_short_quote_grounds_but_is_marked(paper):
    """"42.5±6.9" is on the page — matching it proves much less than a sentence does."""
    cand = ground_candidate(_candidate(quote="42.5±6.9 s"), paper)
    assert cand.grounded is True and cand.grounding_similarity == 1.0
    assert SHORT_QUOTE_NOTE in cand.notes
    assert is_short_quote("42.5±6.9 s") and SHORT_QUOTE_CHARS == 20


def test_a_full_sentence_is_not_marked_short(paper):
    cand = ground_candidate(_candidate(quote=TABLE_QUOTE), paper)
    assert cand.grounded is True and SHORT_QUOTE_NOTE not in cand.notes
    assert not is_short_quote(TABLE_QUOTE)


def test_a_candidate_without_a_page_never_reports_page_none(paper):
    cand = ground_candidate(_candidate(page=None, quote="the old group reached 42.0 deg by "
                                                        "episode 20 of the pointing task"), paper)
    assert cand.grounded is False
    assert "None" not in cand.notes, cand.notes
    assert "best similarity" in cand.notes
