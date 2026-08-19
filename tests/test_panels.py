"""D2's second half: the caption says which panel is whose, so a wrong-panel read is set aside.

A multi-panel figure's caption is the paper's own statement of what each panel shows — "(A) YA DE,
(B) OA DE" binds panel A to the young adults and panel B to the older ones. A digitiser handed
"Fig. 1" reads the cell off BOTH, and the two readings are then two different groups' numbers
wearing one cell's name. Nothing about the numbers can catch that; the caption can.

The check is deliberately narrow. It fires only when the caption binds at least two letters to
BOTH groups with different letters, and it acts only when the group ALSO has a reading from the
panel the caption gives it — otherwise the paper has not said enough, and the check does nothing.
"""
from __future__ import annotations

from canopy.verify.panels import (apply_panel_check, locator_panel, panel_assignments,
                                  panel_groups, panel_mismatch)
from canopy.verify.vote import vote
from tests.helpers import nine

CAPTION = ("Fig. 1. Performance measures for the 30° visuomotor adaptation for both young (YA) "
           "and older (OA) adults. (A) YA DE, (B) OA DE, (C) YA IEE, (D) OA IEE, (E) YA RT and "
           "(F) OA RT.")
VOCAB = {"A": ["Older adults", "older", "elderly", "OA"],
         "B": ["Younger adults", "younger", "young", "YA"]}


def test_caption_assigns_panels_to_groups():
    a = panel_assignments(CAPTION)
    assert len(a) == 6 and a["A"].startswith("YA") and a["B"].startswith("OA")


def test_locator_panel_reads_both_real_forms():
    assert locator_panel("Fig. 1A (YA 30° visuomotor adaptation DE), point at x = A3",
                         "Fig. 1") == "A"
    assert locator_panel("Figure 2, panel a ('adaptive shift'), filled circles = young",
                         "Figure 2") == "A"


def test_langan_wrong_panel_locator_is_a_mismatch():
    assert panel_mismatch("B", "Fig. 1B (OA 30° visuomotor adaptation DE), point at x = A3",
                          CAPTION, VOCAB)
    assert panel_mismatch("B", "Fig. 1A (YA 30° visuomotor adaptation DE), point at x = A3",
                          CAPTION, VOCAB) == ""


def test_no_action_when_caption_does_not_bind_groups():
    cap = "Figure 2. Mean adaptive shifts. (a) Exp 1a; (b) Exp 1b."
    assert panel_mismatch("A", "Figure 2, panel a", cap, VOCAB) == ""


def test_langan_young_late_end_to_end_is_minus_20_5():        # Minor 23: the two halves together
    cell = [c for c in nine.candidates("d1f2946e7e81")
            if c.dataset_id == "d1f2946e7e81:d1" and c.outcome_key == "late_adaptation"
            and c.group == "B"]
    kept, flags = apply_panel_check(cell, CAPTION, VOCAB)
    assert {f.code for f in flags} == {"locator_reads_set_aside"}
    res = vote([c for c in kept if c.extractor_id.endswith("ensemble")])
    assert abs(res.mean + 20.5) < 0.1


def test_a_group_whose_own_panel_has_no_votable_read_keeps_its_number():
    """Fix round 1, MAJOR 1. The second branch exists so a caption regex cannot empty a cell.

    Langan's `aftereffect` young-adult cell has four readings on panel A — the panel the caption
    gives this group — and every one of them is either `ambiguous` or a raw per-route sample that
    `vote_candidates` strips. Counting them as "this group has its own panel" dropped the one
    reading that could carry the cell and left it with no value at all, under a CAPPING flag whose
    message said the opposite. The group has no VOTABLE reading of its own panel, so nothing is
    set aside: the 6.0 stands and a CONTRADICTING flag holds the row for a human.
    """
    cell = [c for c in nine.candidates("d1f2946e7e81")
            if c.dataset_id == "d1f2946e7e81:d1" and c.outcome_key == "aftereffect"
            and c.group == "B"]
    kept, flags = apply_panel_check(cell, CAPTION, VOCAB)
    assert {f.code for f in flags} == {"locator_panel_mismatch"}
    assert len(kept) == len(cell), "a reading was set aside and the cell has nothing left"
    res = vote([c for c in kept if c.extractor_id.endswith("ensemble")])
    assert res.mean == 6.0


def test_a_panel_phrase_that_names_both_groups_binds_neither_letter():
    """Fix round 2, finding 9. `_PANEL_IN_CAPTION` ended a panel's phrase at `\\band\\b`, so
    "(A) Young adults and older adults' reach errors" was read as "(A) Young adults" — the caption
    was made to say a panel belongs to one arm when it says the panel shows both. An older-adult
    reading at Fig 2A was then a mismatch, and set aside if that group had a panel-B ensemble too.
    The phrase ends at a clause boundary, and a letter whose phrase names both groups is left
    unbound — which is what makes the whole check fail closed on a caption like this."""
    caption = ("Figure 2. (A) Young adults and older adults' reach errors; "
               "(B) OA aftereffects during the washout.")
    vocab = {"A": ["young adults"], "B": ["older adults"]}
    assert panel_assignments(caption)["A"] == "Young adults and older adults' reach errors"
    assert panel_groups(caption, vocab) == {}
