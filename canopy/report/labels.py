"""What a forest plot's columns and axis are CALLED — read off the protocol, for both renderers.

The R renderer and the matplotlib one draw the same review, so they must not each invent their
own headings: a reader comparing `forest.png` from a run that had R with one from a run that did
not would be comparing two different tables. `forest_columns` is therefore the single place a
column name is decided, and every name in it comes from the protocol — the moderators the
reviewer listed, in the order they listed them; the two group labels; the outcome's own direction
labels; the estimator the settings name. Nothing here knows what is being reviewed.

The moderator headings are the protocol's field names made readable (`perturbation_size_deg` →
`Perturbation size (deg)`), never renamed: a heading that does not match the name in
`extraction_table.csv` is a heading a reviewer cannot trace back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import GroupDef, OutcomeDef, Protocol, StatsSettings
from .theme import estimator_label

__all__ = ["ForestColumns", "forest_columns", "columns_for", "prettify_moderator",
           "group_initial", "elide", "elide_marked", "MAX_LABEL_CH", "MAX_MOD_CH", "N_COLUMN",
           "STUDY_COLUMN",
           "YEAR_COLUMN"]

#: how long a study label and a moderator value may be before the plot elides them. Shared by
#: both renderers so the same row reads the same whichever drew it; the untouched value is always
#: in `extraction_table.csv`, which is where a reviewer checks it.
MAX_LABEL_CH = 17
MAX_MOD_CH = 13

STUDY_COLUMN = "studlab"
YEAR_COLUMN = "year"
N_COLUMN = "n_label"

#: a trailing name part that is a unit of measurement is printed as one — `(deg)`, not ` deg`
_UNITS = {"deg": "deg", "degs": "deg", "degrees": "degrees", "cm": "cm", "mm": "mm", "m": "m",
          "ms": "ms", "sec": "s", "hz": "Hz", "pct": "%", "percent": "%", "kg": "kg", "n": "n"}
#: name parts that are initialisms and stay upper-case wherever they fall
_UPPER = {"n", "sd", "se", "ci", "id", "rt", "iq", "bmi", "emg", "eeg", "fmri", "mri", "tms"}


def elide(text: object, limit: int) -> str:
    """`text`, cut to `limit` characters with an ellipsis when it does not fit."""
    value = str(text)
    return value if len(value) <= limit else value[: max(1, limit - 1)] + "\u2026"


def elide_marked(text: object, limit: int, mark: str = "") -> str:
    """`text` elided to `limit`, with `mark` still on it — the mark goes on AFTER the cut.

    Both forests mark an overridden row's author label with `theme.OVERRIDE_MARK`, and the
    matplotlib one appended it BEFORE eliding: at `MAX_LABEL_CH = 17` every author string of 16
    characters or more lost the marker, silently, on exactly the rows whose labels are longest
    (review finding). A marker that elision can delete is no marker at all — the plot then reads as
    "nothing here was overridden" on a figure where something was.

    So the mark sits outside the character budget: it is a finding about the row rather than part of
    its name, and both renderers size the author column from its own cells, so the two extra
    characters widen that column instead of clipping anything.
    """
    value = elide(text, limit)
    return f"{value} {mark}" if mark else value


def prettify_moderator(name: str) -> str:
    """`perturbation_size_deg` → `Perturbation size (deg)`; `n_targets` → `N targets`.

    Sentence case, because a forest's column headings are labels rather than titles, and the
    protocol's own words in the protocol's own order — only the underscores and a trailing unit
    are touched.
    """
    words = [w for w in str(name).replace("-", "_").replace(" ", "_").split("_") if w]
    if not words:
        return str(name)
    unit = ""
    if len(words) > 1 and words[-1].lower() in _UNITS:
        unit = _UNITS[words[-1].lower()]
        words = words[:-1]
    parts: list[str] = []
    for index, word in enumerate(words):
        if word.lower() in _UPPER:
            parts.append(word.upper())
        elif index == 0:
            parts.append(word[:1].upper() + word[1:])
        else:
            parts.append(word)
    label = " ".join(parts)
    return f"{label} ({unit})" if unit else label


def group_initial(group: GroupDef | None, fallback: str) -> str:
    """`Older adults` → `O` — the letter the N column pairs, so `N (O/Y)` needs no key."""
    label = "" if group is None else str(group.label or "")
    for character in label:
        if character.isalnum():
            return character.upper()
    return fallback


@dataclass(frozen=True)
class ForestColumns:
    """One forest's column spec: what to print, what to head it with, and how the axis reads.

    `leftcols` are the column NAMES the R renderer finds in `rows.csv` (and the matplotlib one
    builds from the same record fields); `leftlabs` the headings a reader sees. The two lists are
    always the same length and always in the same order — `studlab, year, mod_*…, n_label`.
    """

    leftcols: list[str]
    leftlabs: list[str]
    rightcols: list[str]
    rightlabs: list[str]
    label_left: str
    label_right: str
    smlab: str

    @property
    def moderators(self) -> list[str]:
        """The protocol moderator names behind the `mod_*` columns, in column order."""
        return [c[len("mod_"):] for c in self.leftcols if c.startswith("mod_")]


def columns_for(moderators: Sequence[str], outcome: OutcomeDef, settings: StatsSettings, *,
                group_a: GroupDef | None = None,
                group_b: GroupDef | None = None) -> ForestColumns:
    """The column spec from the pieces, for callers that hold no whole `Protocol`."""
    names = [str(m) for m in moderators]
    return ForestColumns(
        leftcols=[STUDY_COLUMN, YEAR_COLUMN, *(f"mod_{m}" for m in names), N_COLUMN],
        leftlabs=["Author", "Year", *(prettify_moderator(m) for m in names),
                  f"N ({group_initial(group_a, 'A')}/{group_initial(group_b, 'B')})"],
        rightcols=["effect", "ci", "w.random"],
        rightlabs=[estimator_label(settings), f"{float(settings.ci_level):.0%} CI", "Weight"],
        label_left=outcome.negative_direction_label,
        label_right=outcome.positive_direction_label,
        smlab=outcome.label or outcome.key,
    )


def forest_columns(protocol: Protocol, outcome: OutcomeDef,
                   settings: StatsSettings) -> ForestColumns:
    """The column spec for one outcome of one protocol — used by BOTH forest renderers."""
    return columns_for(protocol.moderators, outcome, settings,
                       group_a=protocol.group_a, group_b=protocol.group_b)
