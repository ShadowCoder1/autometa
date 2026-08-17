"""Units, normalised far enough to say whether two readings are the same quantity.

Not a unit-conversion library. The one question asked here is "are these two unit strings the
same unit?" — so that a value in degrees and the same bar read off a percentage axis are never
put in one vote, and a mapper's "deg" agrees with a reader's "degrees". Anything unrecognised
keeps its own lower-cased token, and an empty unit is never a mismatch: nothing is decided
against a reading that did not say.
"""
from __future__ import annotations

import re

__all__ = ["unit_key", "same_unit"]

_ALIASES: dict[str, str] = {
    "deg": "deg", "degs": "deg", "degree": "deg", "degrees": "deg", "°": "deg", "º": "deg",
    "%": "%", "percent": "%", "percentage": "%", "pct": "%", "percentofperturbation": "%",
    "mm": "mm", "millimetre": "mm", "millimeter": "mm", "millimetres": "mm", "millimeters": "mm",
    "cm": "cm", "centimetre": "cm", "centimeter": "cm", "centimetres": "cm", "centimeters": "cm",
    "m": "m", "metre": "m", "meter": "m", "metres": "m", "meters": "m",
    "ms": "ms", "msec": "ms", "millisecond": "ms", "milliseconds": "ms",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "n": "N", "newton": "N", "newtons": "N",
    "rad": "rad", "radian": "rad", "radians": "rad",
    "au": "au", "a.u.": "au", "arbitraryunits": "au",
}
_TOKEN = re.compile(r"[a-zA-Z%°º.]+")


def unit_key(unit: str | None) -> str:
    """The canonical token for a unit string, or "" when it names no unit.

    Takes the FIRST unit-like token: "degrees (CCW/left of target); also percentage of the 30°
    distortion" is degrees — the parenthetical and the aside describe it, they do not change it.
    """
    text = str(unit or "").strip().casefold()
    if not text:
        return ""
    head = re.split(r"[;]", text, 1)[0]                        # "…; also percentage…" is an aside
    tokens = [m.group(0).strip(".") for m in _TOKEN.finditer(head)]
    tokens = [t for t in tokens if t]
    # the first KNOWN unit anywhere in the phrase — "RMSE (mm)" is millimetres, "degrees (CCW/left
    # of target)" is degrees — and only when nothing is known does the first token stand for it
    for token in tokens:
        known = _ALIASES.get(token) or _ALIASES.get(token.replace(".", ""))
        if known:
            return known
    return tokens[0] if tokens else ""


def same_unit(a: str | None, b: str | None) -> bool:
    """True unless BOTH name a unit and the units differ."""
    ka, kb = unit_key(a), unit_key(b)
    return not ka or not kb or ka == kb
