"""Rebuild a read-out the way the models returned it, so the post-processing can be tested.

`digitize()` keeps the raw `points` of a read-out only inside the route samples it derives; what
reaches `extract.json` is the `route_sample` and the reader's prose. So a record cannot be
replayed field-for-field into `_samples_from_readout`, and a test that wants to ask *what would
the digitiser have made of THIS reading?* has to state the reading itself.

These builders do exactly that and nothing else: they carry the real strings (`label_read`,
`x_read`, the locator) and the real numbers recorded in the run, and they hand them to the same
two functions `digitize()` calls, in the same order and with the same arguments. Nothing here
decides anything — if a test passes it is because the digitiser's own rules produced the result,
not because a helper arranged one.
"""
from __future__ import annotations

from typing import Any, Sequence

from canopy.digitize import digitizer as dz
from canopy.digitize.vlm import GroupReadOut, PointRead, ReadOut, TargetSpec


def readout(model: str, variant: str, groups: Sequence[dict[str, Any]], *,
            legend_says: str = "", tick_labels: Sequence[float] = ()) -> ReadOut:
    """One model's reading of a figure.

    Each entry of `groups` is `{"group", "label_read", "x_read", "points", "mean",
    "error_half_length"}`, where `points` is a list of `(x_label, mean, error_half_length)` —
    the shape `digitize_readout.md` asks a reader for on a categorical x axis.
    """
    rows: list[GroupReadOut] = []
    for spec in groups:
        rows.append(GroupReadOut(
            group=str(spec.get("group", "A")),
            label_read=str(spec.get("label_read", "")),
            mean=spec.get("mean"),
            error_half_length=spec.get("error_half_length"),
            error_sides=str(spec.get("error_sides", "unknown")),
            x_read=str(spec.get("x_read", "")),
            points=[PointRead(x_label=str(x), mean=m, error_half_length=e)
                    for x, m, e in spec.get("points", ())],
            confidence=float(spec.get("confidence", 0.8)),
            notes=str(spec.get("notes", "")),
        ))
    return ReadOut(status="found", groups=rows, legend_says=legend_says,
                   tick_labels=list(tick_labels), unit="", model=model, variant=variant,
                   confidence=0.8)


def target(*, group_a_label: str, group_b_label: str, collapse_across_x: bool = True,
           x_hint: str = "", panel_hint: str = "", **kw: Any) -> TargetSpec:
    """The mapper's target for a cell, with the collapse mode on by default (a categorical x)."""
    return TargetSpec(group_a_label=group_a_label, group_b_label=group_b_label,
                      collapse_across_x=collapse_across_x, x_hint=x_hint,
                      panel_hint=panel_hint, **kw)


def samples_for(readings: Sequence[ReadOut], target: TargetSpec, *,
                locator: str) -> list[dz.RouteSample]:
    """The route-D samples `digitize()` would build from these readings, under this locator.

    The same two calls `digitize()` makes: the cell-level role over every voting reading (which
    is what decides whether the pixel routes are dropped), then one `_samples_from_readout` per
    reading with the source's locator passed through.
    """
    dz._categorical_role(target, dz.voting(readings), locator=locator)
    out: list[dz.RouteSample] = []
    for reading in readings:
        out.extend(dz._samples_from_readout(reading, collapse=target.collapse_across_x,
                                            target=target, locator=locator))
    return out
