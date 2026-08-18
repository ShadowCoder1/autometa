"""Which panel is whose — the caption's own answer, checked against where a reading was taken.

A multi-panel figure caption is the paper stating what each panel shows: "(A) YA DE, (B) OA DE"
binds panel A to one arm of the comparison and panel B to the other. A digitiser pointed at the
figure as a whole reads the same cell off BOTH panels, and the two readings are then two
different groups' numbers wearing one cell's name. Nothing about the NUMBERS can catch that —
they are both plausible, both in the right unit, both measured off a real axis — and averaging
them produces a value neither panel contains. The caption can catch it, and it is the only thing
that can.

The check is deliberately narrow, because a caption that has not said enough must not be made to
say something:

* it fires only when the caption binds at least two letters to BOTH groups, with different
  letters — "(a) Exp 1a; (b) Exp 1b" enumerates panels without naming a group, and nothing
  happens;
* a group's mismatched reading is only SET ASIDE when that group also has a reading from the
  panel the caption gives it. Without one, dropping would leave the cell empty on the strength of
  a regex, so the mismatch is flagged and everything is kept;
* a locator that names no panel at all is not evidence of anything.

`_labels_are_the_same` / `_names_group` live here rather than in the digitiser because both
layers now ask the same question — *do these words name this group?* — and one vocabulary answer
must not drift into two. The digitiser imports them back.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..models import Candidate, CheckFlag
from .checks import CHECK_SEVERITY
from .vote import _ANY_FIGURE, _FIGURE_WORD, LOCATOR_DROPPED, figure_reference, locator_key

__all__ = ["panel_assignments", "locator_panel", "panel_mismatch", "panel_groups",
           "apply_panel_check", "figure_reference", "LOCATOR_DROPPED"]

#: the digitiser's own consensus candidate, and the only digitised reading the vote ever sees.
#: The constant's owner is `canopy.pipeline.rows.ENSEMBLE`; the literal is repeated here rather
#: than imported because `verify` is the layer BELOW `pipeline` and must not depend on it.
_ENSEMBLE = "digitize:ensemble"


def _votable(cand: Candidate) -> bool:
    """A reading the vote could actually weigh.

    Two filters stand between a cell's candidates and the vote, and this check has to know about
    both. `run.vote_candidates` keeps one figure reading per group — the digitiser's ensemble,
    never its per-route samples — and `vote._usable` keeps only a `found` `group_stats` with a
    mean. A reading that fails either can never carry the cell, so counting it as "this group has
    a reading from its own panel" is how the set-aside branch empties a cell it was written to
    protect: on Langan's aftereffect cell every reading on the young adults' own panel was either
    `ambiguous` or a raw route sample, and the group's one usable number was dropped in favour of
    them.
    """
    return (cand.kind == "group_stats" and cand.status == "found" and cand.mean is not None
            and (not cand.extractor_id.startswith("digitize:")
                 or cand.extractor_id == _ENSEMBLE))

_LABEL_JUNK = re.compile(r"[^a-z0-9]+")
#: a label this short matches too much to be evidence of anything ("SD", "n", "A")
_MIN_LABEL_CHARS = 3


def _label_key(text: Any) -> str:
    return _LABEL_JUNK.sub("", str(text or "").lower())


def _labels_are_the_same(a: Any, b: Any) -> bool:
    """Do a plotted x category and a protocol group label name the same thing?

    Equal after stripping case and punctuation, or one contained in the other — a figure axis
    says "Elderly" where the protocol says "Elderly adults", and an axis that says "old" is not
    evidence about a group called "older adults" unless one spells the other.
    """
    left, right = _label_key(a), _label_key(b)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    return len(short) >= _MIN_LABEL_CHARS and short in long


def _names_group(category: Any, names: Sequence[str]) -> bool:
    """Does a plotted x category name this group, in any of the words the protocol gave for it?"""
    return any(_labels_are_the_same(category, name) for name in names if str(name or "").strip())


# ----------------------------------------------------------------------------- the caption
#: "(A) YA DE," / "(b) OA RT." — a bracketed single letter and the phrase it introduces, which
#: ends at the next clause boundary. The 60-character ceiling is what keeps a caption whose
#: letters are NOT an enumeration ("a 30° rotation (b) was applied") from swallowing a sentence.
_PANEL_IN_CAPTION = re.compile(
    r"\(\s*([A-Za-z])\s*\)\s*([^()]{0,60}?)(?=\s*(?:[,;.]|\band\b|\(|$))")

#: the caption's own printed label, so a locator can be anchored to the figure it names.
#: `_FIGURE_WORD` and `_ANY_FIGURE` live in `verify.vote` — the lower layer, which needs them to
#: tell one figure from another before it calls two places a conflict.
_CAPTION_LABEL = re.compile(r"^\W*(" + _FIGURE_WORD + r"\s*\d+)", re.I)
#: "Fig. 1A" — the letter ABUTS the label. A letter after a space is a word ("Fig. 2 a mean of"),
#: not a panel, which is why the bare form is never accepted.
_ABUTTING = re.compile(r"^([A-Za-z])(?![A-Za-z0-9])")
#: "Fig. 1 (a)", "Fig. 1, (b)"
_BRACKETED = re.compile(r"^[\s,:;–—-]*\(\s*([A-Za-z])\s*\)")
#: "panel a", "panels B", "panel (c)" — the word says the letter is a panel, so the space is fine
_PANEL_WORD = re.compile(r"\bpanels?\s*[:\-]?\s*\(?\s*([A-Za-z])\s*\)?(?![A-Za-z0-9])", re.I)
#: a 2–3 letter upper-case token: how a caption abbreviates a group it has already spelled out
_ABBREVIATION = re.compile(r"\b[A-Z]{2,3}\b")
#: how far past a figure reference "panel x" may sit and still be about that figure
_PANEL_WINDOW = 60


def panel_assignments(caption: str) -> dict[str, str]:
    """`{"A": "YA DE", "B": "OA DE", ...}` — what the caption says each panel letter shows."""
    out: dict[str, str] = {}
    for match in _PANEL_IN_CAPTION.finditer(str(caption or "")):
        letter, text = match.group(1).upper(), match.group(2).strip()
        if text and letter not in out:
            out[letter] = text
    return out


def _caption_label(caption: str) -> str:
    """The figure label the caption prints for itself ("Fig. 1"), or ""."""
    match = _CAPTION_LABEL.match(str(caption or "").strip())
    return match.group(1).strip() if match else ""


def _label_pattern(figure_label: str) -> re.Pattern[str] | None:
    """The label as it may appear in a locator: "Fig. 1" also matches "Figure 1"."""
    label = str(figure_label or "").strip()
    if not label:
        return None
    number = re.search(r"\d+", label)
    if number:
        return re.compile(_FIGURE_WORD + r"\s*" + re.escape(number.group(0)) + r"(?!\d)", re.I)
    return re.compile(re.escape(label), re.I)


def locator_panel(locator: str, figure_label: str = "") -> str:
    """The panel letter a locator names, ANCHORED to the figure — "" when it names none.

    Anchored, because a locator is full of letters that are not panels: Langan's own reads say
    "point at x = A3", and an unanchored search for a capital letter answers "A" to every one of
    them. What counts is a letter that abuts the figure reference ("Fig. 1A"), is bracketed just
    after it ("Fig. 1 (a)"), or is introduced by the word `panel`.
    """
    text = str(locator or "")
    if not text.strip():
        return ""
    pattern = _label_pattern(figure_label) or _ANY_FIGURE
    for match in pattern.finditer(text):
        tail = text[match.end():]
        for regex in (_ABUTTING, _BRACKETED):
            hit = regex.match(tail)
            if hit:
                return hit.group(1).upper()
        hit = _PANEL_WORD.search(tail[:_PANEL_WINDOW])
        if hit:
            return hit.group(1).upper()
    hit = _PANEL_WORD.search(text)
    return hit.group(1).upper() if hit else ""


# ----------------------------------------------------------------------------- caption → groups
def _initials(name: str) -> str:
    """`"Younger adults"` → `"YA"`. One word has no initials worth matching a token against."""
    words = [word for word in re.split(r"[^A-Za-z0-9]+", str(name or "")) if word]
    return "".join(word[0] for word in words).upper() if len(words) >= 2 else ""


def _text_names_group(text: str, names: Sequence[str]) -> bool:
    """Does a caption's panel phrase name this group — spelled out, or as its own abbreviation?

    A caption that has already written "both young (YA) and older (OA) adults" then enumerates
    its panels as "YA DE, OA DE". Two letters are far too short for `_labels_are_the_same` to
    treat as evidence, and rightly so — but an upper-case token whose letters are the ORDERED
    initials of a label the protocol gave is the paper's own abbreviation of that label, and it
    is the only form in which many captions ever name a group.
    """
    if _names_group(text, names):
        return True
    initials = {_initials(name) for name in names if str(name or "").strip()} - {""}
    return any(token in initials for token in _ABBREVIATION.findall(str(text or "")))


def panel_groups(caption: str, vocab: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """`{"A": "B", "B": "A", ...}` — the group each panel letter belongs to, when the caption says.

    Empty unless the caption binds at least two letters, to at least two different groups: one
    letter naming one group says nothing about where the OTHER group was plotted, and a check
    that fires on it is guessing.
    """
    bound: dict[str, str] = {}
    for letter, text in panel_assignments(caption).items():
        named = [group for group in sorted(vocab)
                 if _text_names_group(text, vocab.get(group) or ())]
        if len(named) == 1:
            bound[letter] = named[0]
    if len(bound) < 2 or len(set(bound.values())) < 2:
        return {}
    return bound


def panel_mismatch(group: str, locator: str, caption: str,
                   vocab: Mapping[str, Sequence[str]]) -> str:
    """Why this reading's panel is the OTHER group's — "" when it is this group's, or unknown."""
    bound = panel_groups(caption, vocab)
    if not bound:
        return ""
    letter = locator_panel(locator, _caption_label(caption))
    owner = bound.get(letter, "")
    if not letter or not owner or owner == group:
        return ""
    return (f"the caption gives panel {letter} ({panel_assignments(caption)[letter]!r}) to group "
            f"{owner}, but this reading was taken there for group {group}")


# ----------------------------------------------------------------------------- the action
def _panel_of(cand: Candidate, label: str) -> str:
    """The panel a candidate was read off — only figure readings have one."""
    return locator_panel(cand.locator, label) if locator_key(cand) else ""


def apply_panel_check(cell: Sequence[Candidate], caption: str,
                      vocab: Mapping[str, Sequence[str]]) -> tuple[list[Candidate],
                                                                   list[CheckFlag]]:
    """`(the candidates the vote and the verifier may see, the flags this cell earned)`.

    Per group, three outcomes and no fourth. A group with a reading off the panel the caption
    gives it AND a reading off another group's panel keeps the first and sets the second aside —
    the paper has said which is which, and the wrong one is a different group's number. A group
    with only the wrong panel's reading keeps it and is flagged: dropping would leave the cell
    empty on the strength of a caption regex, and a contradiction a human reads is better than a
    hole nobody sees. A group whose readings name no panel is untouched.

    "A reading from its own panel" means one that could CARRY the cell — see `_votable`. Counting
    the rest defeats the second branch entirely, because the readings on a group's own panel are
    routinely the ambiguous ones and the raw route samples: the cell is then emptied under a flag
    whose message says its own panel's readings stand.
    """
    bound = panel_groups(caption, vocab)
    if not bound:
        return list(cell), []
    label = _caption_label(caption)
    panels = [_panel_of(cand, label) for cand in cell]
    flags: list[CheckFlag] = []
    dropped: set[int] = set()
    for group in sorted({c.group for c in cell if c.group}):
        wrong = [i for i, c in enumerate(cell)
                 if c.group == group and panels[i] in bound and bound[panels[i]] != group]
        # `right` is VOTABLE-only and `wrong` is not, deliberately. What decides the branch is
        # whether this group has a reading from its OWN panel that could carry the cell; marking
        # a wrong-panel reading that could never vote anyway costs nothing and keeps the record.
        right = [i for i, c in enumerate(cell)
                 if c.group == group and _votable(c)
                 and panels[i] in bound and bound[panels[i]] == group]
        wrong_votable = [i for i in wrong if _votable(cell[i])]
        if not wrong_votable:
            continue                    # nothing that could carry the vote is at stake
        theirs = ", ".join(sorted({panels[i] for i in wrong}))
        ids = sorted({cell[i].candidate_id for i in wrong})
        if right:
            mine = ", ".join(sorted({panels[i] for i in right}))
            for i in wrong:
                provenance = dict(cell[i].pixel_provenance or {})
                provenance[LOCATOR_DROPPED] = panel_mismatch(group, cell[i].locator, caption,
                                                             vocab)
                cell[i].pixel_provenance = provenance
            dropped.update(wrong)
            flags.append(CheckFlag(
                code="locator_reads_set_aside",
                severity=CHECK_SEVERITY["locator_reads_set_aside"],
                message=(f"the caption gives panel {mine} to group {group} and panel {theirs} to "
                         f"another group; {len(wrong_votable)} reading(s) taken off panel "
                         f"{theirs} were set aside, and the {len(right)} taken off panel {mine} "
                         f"stand"),
                candidate_ids=ids))
        else:
            flags.append(CheckFlag(
                code="locator_panel_mismatch",
                severity=CHECK_SEVERITY["locator_panel_mismatch"],
                message=(f"every reading for group {group} was taken off panel {theirs}, which "
                         f"the caption gives to another group; nothing was set aside because "
                         f"this group has no reading from the panel the caption gives it, so "
                         f"the value may be the other group's"),
                candidate_ids=ids))
    return [c for i, c in enumerate(cell) if i not in dropped], flags
