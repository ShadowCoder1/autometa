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
                                  panel_mismatch)
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
