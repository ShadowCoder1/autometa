"""Vision routes of the figure digitizer.

Three independent things a vision model can do with a figure crop, each on its own prompt and
schema, each driven through `LLMClient.tool_loop` so the model can zoom before it answers:

* `read_out`  — path D: read the *values* straight off the chart (several variants vote);
* `coords`    — path C: report *pixel coordinates* of ticks, data points and error-bar caps, which
                `digitizer.py` then calibrates and snaps with the 6a CV core;
* `overlay_verify` — show the model the marks a route resolved, drawn on the figure, and ask
                which ones are in the wrong place.

**Coordinate frames.** Everything a model sees is the image produced by
`ingest.images.prepare_for_claude`, and every coordinate a model reports is in *that* image's
pixels. `FigureView.to_crop()` divides by the prepared scale to get `crop_png` pixels, which is
the frame the whole of task 6a works in (array-index convention: pixel centres are integers).
Vector geometry is normalised to the same convention, so VLM, raster-CV and vector pixels are
directly comparable — never add a half-pixel correction here.
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ..ingest.images import PreparedImage, prepare_for_claude
from ..llm.client import EPHEMERAL, LLMClient, ToolLoopResult
from ..llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from .calibrate import parse_number
from .cv import (Axes, TickLabels, detect_bars, detect_markers, find_axes, find_tick_marks,
                 load_color, load_gray, ocr_tick_labels)
from .overlay import draw_overlay

__all__ = ["PROMPT_VERSION", "TargetSpec", "FigureView", "GroupReadOut", "PointRead",
           "ReadOut", "TickCoord",
           "GroupCoords", "CoordReadout", "Mismatch", "OverlayVerdict", "read_out", "coords",
           "overlay_verify", "load_prompt", "render_prompt", "READOUT_SCHEMA", "COORDS_SCHEMA",
           "OVERLAY_SCHEMA", "READOUT_VARIANTS", "MAX_ZOOM", "MAX_READOUT_TOOL_CALLS"]

#: bump when a prompt or a schema changes (fixtures are content-addressed, so they follow anyway)
PROMPT_VERSION = "digitize/2"
#: read-out variants; each is a `digitize_readout_<name>.md` delta appended to the base prompt.
#: Order matters: `_readout_plan` fills a figure's samples from distinct (model, variant) pairs in
#: this order, and only re-samples an already-used pair once every pair is spent.
READOUT_VARIANTS: tuple[str, ...] = ("direct", "ticks_first", "zoom_first")
MAX_ZOOM = 4.0                      # amendment F: a zoom tool never magnifies more than 4x
MAX_TOOL_CALLS = 8                  # amendment F
#: a read-out that has not answered in six tool calls is not going to (task 15 §A3)
MAX_READOUT_TOOL_CALLS = 6
MAX_TOKENS = 16000                  # enough for adaptive thinking + the submit call, no retry
_MIN_CROP_PX = 8                    # a crop smaller than this is a mis-click, not a zoom

PROMPT_DIR = Path(__file__).resolve().parents[1] / "llm" / "prompts"
_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")


def load_prompt(name: str) -> str:
    """The raw text of `canopy/llm/prompts/<name>.md`."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no prompt {name!r} in {PROMPT_DIR}")
    return path.read_text()


def render_prompt(name: str, **values: str) -> str:
    """Fill `{{PLACEHOLDER}}` markers. An unfilled marker is a bug, so it raises."""
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    left = sorted(set(_PLACEHOLDER.findall(text)))
    if left:
        raise KeyError(f"prompt {name!r} still has unfilled placeholders {left}")
    return text


# ----------------------------------------------------------------------------- target
Quantity = Literal["mean_and_error", "points", "box"]
LateWindowRule = Literal["paper_reported_block", "block_closest_to_end", "mean_of_block_sd"]

#: WHAT a categorical x axis is a set of. Two opposite figure shapes hide behind that one word,
#: and telling them apart is not a detail — it decides whether averaging across the axis is the
#: quantity the review wants or the destruction of it:
#:
#: * `conditions` — target directions, hands, sessions, task blocks with no order. Each series
#:   runs across the whole axis and the outcome is the average ACROSS it, so reading one point is
#:   a different number, not a less precise one.
#: * `groups` — the x categories ARE the comparison arms: the ordinary two-bar group chart, one
#:   bar per group, the commonest effect-size figure there is. Each category IS one group's value.
#:   Averaging across this axis computes `(mean_A + mean_B) / 2` for BOTH arms, which makes
#:   Cohen's d exactly 0.0 with two routes in perfect agreement.
#: * `unknown` — nobody has said. The digitiser decides it from what the readers report about the
#:   x categories, or refuses; it never guesses.
CategoricalX = Literal["conditions", "groups", "unknown"]


@dataclass
class TargetSpec:
    """What the mapper says we are looking for in this figure. Nothing study-specific lives here."""

    outcome_key: str = ""
    group_a_label: str = ""                 # as printed in the paper (legend/axis text)
    group_b_label: str = ""
    series_hint: str = ""                   # legend/marker description from the mapper
    x_hint: str = ""                        # "last adaptation episode", "block 20", ...
    panel_hint: str = ""                    # "Fig 2A", "left panel", ...
    quantity: Quantity = "mean_and_error"
    error_bar_type_hint: str = "UNKNOWN"    # Source.error_bar_type (SD/SE/CI95/...)
    unit_hint: str = ""
    late_window_sd: LateWindowRule = "paper_reported_block"
    #: the protocol PERMITS averaging across a categorical x axis (task 16 P6). Permission is not
    #: the same question as whether averaging is the right thing to do here — see `categorical_x`.
    collapse_across_x: bool = False
    #: what the categorical x axis is a set of. `groups` overrides `collapse_across_x`: when the
    #: categories are the comparison arms there is nothing to average across, and doing it anyway
    #: gives both arms the same mean. Left `unknown` by every caller today, in which case the
    #: digitiser resolves it from the categories the readers name (`_categorical_role`).
    categorical_x: CategoricalX = "unknown"
    #: the protocol's own other names for each group ("elderly", "aged", "young adults", …). A
    #: paper labels its bars in its own words, and `Old adults` is not a substring of
    #: `Older adults`; without this vocabulary the categories-are-the-groups test fails to resolve
    #: and the cell yields no number at all. The protocol already carries these — nothing here is
    #: study-specific, it is whatever vocabulary the review wrote down.
    group_a_synonyms: tuple[str, ...] = ()
    group_b_synonyms: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def describe(self) -> str:
        """The target as prompt text (one fact per line; empty fields are omitted, not faked)."""
        rows = [
            ("outcome", self.outcome_key),
            ("panel", self.panel_hint),
            ("group A", self.group_a_label),
            ("group B", self.group_b_label),
            ("series / legend hint", self.series_hint),
            ("x position to read", self._x_instruction()),
            ("what the x categories are", "" if self.categorical_x == "unknown" else (
                "the two comparison groups themselves — one point per group"
                if self.categorical_x == "groups"
                else "conditions the outcome is averaged across")),
            ("quantity", self.quantity),
            ("error bars are said to be", self.error_bar_type_hint),
            ("expected unit", self.unit_hint),
            ("time-series rule", self.late_window_sd if self.x_hint else ""),
            ("mapper notes", self.notes),
        ]
        lines = [f"- {label}: {value}" for label, value in rows if str(value).strip()]
        return "\n".join(lines) or "- (the mapper gave no details)"

    def _x_instruction(self) -> str:
        """Which x position this group's number is at — the one line the two categorical shapes
        must not share. "Every point on the x axis" is the right instruction for a set of
        conditions and a trap on a group chart, where the points on the x axis ARE the two groups
        and following it literally hands back the same average for both."""
        if self.categorical_x == "groups":
            return ("this group's OWN category on the x axis — the x categories are the two "
                    "groups, so read the one bar or point that belongs to this group")
        if self.collapse_across_x:
            return ("every point of THIS group's series along the x axis (the outcome is their "
                    "average). If the x categories turn out to be the two groups themselves, "
                    "this series has exactly one point: report it and name its category")
        return self.x_hint


# ----------------------------------------------------------------------------- schemas
_STATUS = {"type": "string", "enum": ["found", "not_on_these_pages", "ambiguous"]}
_GROUP = {"type": "string", "enum": ["A", "B", "unknown"]}
_NUM = {"type": ["number", "null"]}


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": sorted(properties), "properties": properties}


READOUT_SCHEMA = _obj({
    "status": _STATUS,
    "panel": {"type": "string"},
    "unit": {"type": "string"},
    #: WHICH value axis these numbers came off. A panel with a left-hand axis in degrees and a
    #: right-hand one in per cent (Cressman 2010 Fig. 3b) gives two readers two different correct
    #: answers, and nothing in the numbers says which ladder each used (critique miss 1).
    "axis_read": {"type": "string"},
    #: the axis' sign convention when the ticks do not carry it — a CW/CCW axis whose direction
    #: lives in the title reads +17.5 in one paper and -21.5 in another for the same outcome
    "axis_direction_note": {"type": "string"},
    "legend_says": {"type": "string"},
    "tick_labels": {"type": "array", "items": {"type": "number"}},
    "pixel_resolution_estimate": _NUM,
    "confidence": {"type": "number"},
    "notes": {"type": "string"},
    "groups": {"type": "array", "items": _obj({
        "group": _GROUP,
        "label_read": {"type": "string"},
        "mean": _NUM,
        "error_half_length": _NUM,
        "error_upper": _NUM,
        "error_lower": _NUM,
        "error_sides": {"type": "string",
                        "enum": ["both", "up", "down", "none", "unknown"]},
        "x_read": {"type": "string"},
        #: every point of this series, when the target asks for the average ACROSS a categorical
        #: x axis (Heuer & Hegele Fig 2a: eight target directions, and the outcome is their mean)
        "points": {"type": "array", "items": _obj({
            "x_label": {"type": "string"},
            "mean": _NUM,
            "error_half_length": _NUM,
        })},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    })},
})

COORDS_SCHEMA = _obj({
    "status": _STATUS,
    "panel": {"type": "string"},
    "unit": {"type": "string"},
    "confidence": {"type": "number"},
    "notes": {"type": "string"},
    "ticks": {"type": "array", "items": _obj({
        "value": {"type": "number"},
        "y_px": {"type": "number"},
    })},
    "groups": {"type": "array", "items": _obj({
        "group": _GROUP,
        "label_read": {"type": "string"},
        "x_px": _NUM,
        "y_px": _NUM,
        "bar_x0_px": _NUM,
        "bar_x1_px": _NUM,
        "cap_top_px": _NUM,
        "cap_bottom_px": _NUM,
        "notes": {"type": "string"},
    })},
})

OVERLAY_SCHEMA = _obj({
    "marks": {"type": "array", "items": _obj({
        "number": {"type": "integer"},
        "verdict": {"type": "string",
                    "enum": ["ok", "not_on_datum", "wrong_series", "wrong_x", "unknown"]},
        "reason": {"type": "string"},
    })},
    "notes": {"type": "string"},
})

for _name, _schema in (("READOUT_SCHEMA", READOUT_SCHEMA), ("COORDS_SCHEMA", COORDS_SCHEMA),
                       ("OVERLAY_SCHEMA", OVERLAY_SCHEMA)):
    assert_valid_output_schema(_schema, _name)
    assert_no_derived_stats(_schema, name=_name)


# ----------------------------------------------------------------------------- results
@dataclass
class PointRead:
    """One point of a series, when the whole series is read across a categorical x axis."""

    x_label: str = ""
    mean: float | None = None
    error_half_length: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GroupReadOut:
    group: str = "unknown"
    label_read: str = ""
    mean: float | None = None
    error_half_length: float | None = None      # datum -> cap, in data units
    error_upper: float | None = None            # absolute value of the upper cap
    error_lower: float | None = None
    error_sides: str = "unknown"                # both | up | down | none (one-armed bars are common)
    x_read: str = ""
    points: list["PointRead"] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["points"] = [p.to_dict() for p in self.points]
        return d


@dataclass
class ReadOut:
    """Path D — the model's own reading of the plotted values."""

    status: str = "found"
    groups: list[GroupReadOut] = field(default_factory=list)
    legend_says: str = ""
    tick_labels: list[float] = field(default_factory=list)
    pixel_resolution_estimate: float | None = None
    unit: str = ""
    axis_read: str = ""                          # which value axis the numbers came off
    axis_direction_note: str = ""                # the axis' sign convention, when the ticks lack one
    panel: str = ""
    confidence: float = 0.0
    notes: str = ""
    model: str = ""
    variant: str = ""
    sample: int = 0                              # >0 = a re-sample of an already-used prompt
    call_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    turns: int = 0

    def group(self, key: str) -> GroupReadOut | None:
        for g in self.groups:
            if g.group == key:
                return g
        return None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["groups"] = [g.to_dict() for g in self.groups]
        return d


@dataclass
class TickCoord:
    value: float
    y_px: float                                  # crop pixels

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GroupCoords:
    group: str = "unknown"
    label_read: str = ""
    x_px: float | None = None                    # crop pixels
    y_px: float | None = None
    bar_x0_px: float | None = None
    bar_x1_px: float | None = None
    cap_top_px: float | None = None
    cap_bottom_px: float | None = None
    notes: str = ""

    @property
    def bar_width(self) -> float | None:
        if self.bar_x0_px is None or self.bar_x1_px is None:
            return None
        return abs(self.bar_x1_px - self.bar_x0_px)

    def to_dict(self) -> dict[str, Any]:
        return dict(group=self.group, label_read=self.label_read, x_px=self.x_px, y_px=self.y_px,
                    bar_x0_px=self.bar_x0_px, bar_x1_px=self.bar_x1_px,
                    cap_top_px=self.cap_top_px, cap_bottom_px=self.cap_bottom_px, notes=self.notes)


@dataclass
class CoordReadout:
    """Path C — pixel coordinates in **crop** pixels (already divided by the prepared scale)."""

    status: str = "found"
    ticks: list[TickCoord] = field(default_factory=list)
    groups: list[GroupCoords] = field(default_factory=list)
    unit: str = ""
    panel: str = ""
    confidence: float = 0.0
    notes: str = ""
    model: str = ""
    scale: float = 1.0                           # sent px per crop px (for provenance)
    call_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    turns: int = 0

    def group(self, key: str) -> GroupCoords | None:
        for g in self.groups:
            if g.group == key:
                return g
        return None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["ticks"] = [t.to_dict() for t in self.ticks]
        d["groups"] = [g.to_dict() for g in self.groups]
        return d


@dataclass
class Mismatch:
    number: int                                  # 1-based mark number as drawn
    verdict: str = "unknown"                     # ok | not_on_datum | wrong_series | wrong_x | unknown
    reason: str = ""

    @property
    def bad(self) -> bool:
        return self.verdict in ("not_on_datum", "wrong_series", "wrong_x")

    def to_dict(self) -> dict[str, Any]:
        return dict(number=self.number, verdict=self.verdict, reason=self.reason)


class OverlayVerdict(list):
    """`list[Mismatch]` that also carries the overlay it judged and the call that judged it.

    (Same shape as `cv.TickLabels`: a plain list for consumers that only want the mismatches, with
    the provenance attached for the ones that need to record it.)
    """

    overlay_path: str = ""
    call_ids: list[str] = ()
    cost_usd: float = 0.0
    model: str = ""
    notes: str = ""

    @property
    def mismatches(self) -> list[Mismatch]:
        return [m for m in self if m.bad]


# ----------------------------------------------------------------------------- figure view
def _pil_image_block(img: Image.Image) -> dict[str, Any]:
    """A base64 PNG content block straight from a PIL image (no temp file, stable bytes)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")}}


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


class FigureView:
    """One figure crop prepared for a vision model, plus the zoom tools that operate on it.

    The model always works in *sent-image* pixels. `to_crop` maps back to `crop_png` pixels, which
    is what every 6a function expects.
    """

    def __init__(self, crop_png: str | Path, work_dir: str | Path | None = None,
                 axes: Axes | None = None, tick_rows: list[float] | None = None,
                 tick_labels: TickLabels | list | None = None,
                 bars: list | None = None, markers: list | None = None):
        self.crop_png = Path(crop_png)
        self.work_dir = Path(work_dir) if work_dir is not None else self.crop_png.parent
        source = Image.open(self.crop_png).convert("RGB")
        self.prepared: PreparedImage = prepare_for_claude(source)
        self.image: Image.Image = self.prepared.image
        self.scale: float = float(self.prepared.scale)
        self.source_size: tuple[int, int] = self.prepared.source_size
        self._axes = axes
        self._tick_rows = tick_rows
        self._tick_labels = tick_labels
        self._bars = bars
        self._markers = markers
        self._regions_text: str | None = None
        self._n_regions: int | None = None
        self._overlays = 0

    # -------------------------------------------------------------- frames
    def to_crop(self, x: float, y: float) -> tuple[float, float]:
        """Sent-image pixels -> `crop_png` pixels."""
        return x / self.scale, y / self.scale

    def to_sent(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def crop_px(self, value: float | None) -> float | None:
        return None if value is None else value / self.scale

    def image_block(self, cache: bool = False) -> dict[str, Any]:
        """The figure as a content block; `cache=True` marks it as the end of the cached prefix."""
        block = _pil_image_block(self.image)
        return {**block, "cache_control": dict(EPHEMERAL)} if cache else block

    def header(self) -> str:
        w, h = self.image.size
        return (f"The figure crop is below, {w} px wide and {h} px tall. Coordinates you report "
                f"must be absolute pixels of THIS image (x right, y down, top-left = 0,0).")

    # -------------------------------------------------------------- regions (6a hint)
    def regions_text(self) -> str:
        if self._regions_text is None:
            self._regions_text = self._build_regions()
        return self._regions_text

    @property
    def has_regions(self) -> bool:
        """True when the CV pass found something worth asking about.

        A `list_regions` tool that can only answer "nothing detected" is a tool call the model
        pays for and learns nothing from, so it is not offered at all (task 15 §A3).
        """
        self.regions_text()
        return bool(self._n_regions)

    def _build_regions(self) -> str:
        s = self.scale
        lines = [f"Computer-vision pass over the same figure. All coordinates are in the pixels of "
                 f"the image you were sent (original crop scaled by x{s:.4f}). These are hints from "
                 f"a non-semantic detector — check them, do not trust them."]
        try:
            gray = load_gray(self.crop_png)
            colour = load_color(self.crop_png)
        except Exception as exc:                              # pragma: no cover - unreadable crop
            return "\n".join(lines + [f"(image could not be analysed: {exc})"])
        axes = self._axes if self._axes is not None else find_axes(gray)
        if axes.y_axis_x is not None:
            lines.append(f"- y axis (vertical line) at x = {axes.y_axis_x * s:.1f}")
        if axes.x_axis_y is not None:
            lines.append(f"- x axis (horizontal line) at y = {axes.x_axis_y * s:.1f}")
        x0, y0, x1, y1 = axes.plot_bbox
        lines.append(f"- plotting area x {x0 * s:.0f}..{x1 * s:.0f}, y {y0 * s:.0f}..{y1 * s:.0f}")
        # what was really DETECTED (a plot box is always guessed, so it does not count)
        found = sum(1 for v in (axes.y_axis_x, axes.x_axis_y) if v is not None)
        rows = self._tick_rows
        if rows is None:
            rows = find_tick_marks(gray, axes).get("left", [])
        if rows:
            lines.append("- y-axis tick marks at y = "
                         + ", ".join(f"{r * s:.1f}" for r in rows[:24]))
        labels = self._tick_labels
        if labels is None:
            labels = ocr_tick_labels(gray, axes, side="left", ticks=rows or None)
        read = [(lb.value, lb.center[1]) for lb in labels if lb.value is not None]
        if read:
            lines.append("- OCR read these y tick labels (value @ y): "
                         + ", ".join(f"{v:g} @ {y * s:.1f}" for v, y in read[:24]))
        status = getattr(labels, "status", "")
        if status and status != "ok":
            lines.append(f"- OCR status: {status} (labels above may be missing)")
        bars = self._bars
        if bars is None:
            try:
                bars = detect_bars(colour, axes)
            except Exception:                                 # pragma: no cover - defensive
                bars = []
        markers = self._markers
        if markers is None and not bars:
            try:
                markers = detect_markers(colour, axes)
            except Exception:                                 # pragma: no cover - defensive
                markers = []
        for b in bars[:16]:
            flag = " (narrow: under 8 px, read-out unreliable)" if b.narrow else ""
            lines.append(f"- bar {b.colour} top at ({b.x_center * s:.1f}, {b.top_y * s:.1f}), "
                         f"x {b.x0 * s:.1f}..{b.x1 * s:.1f}{flag}")
        for m in (markers or [])[:24]:
            lines.append(f"- marker {m.kind} {m.colour} at ({m.x * s:.1f}, {m.y * s:.1f})")
        self._n_regions = found + len(rows or []) + len(read) + len(bars) + len(markers or [])
        if len(lines) == 2:
            lines.append("- (nothing else detected)")
        return "\n".join(lines)

    # -------------------------------------------------------------- tools
    def tools(self) -> list[dict[str, Any]]:
        """The stable zoom-tool list (amendment F). The caller appends its own `submit` tool.

        `list_regions` is offered only when the computer-vision pass actually found something:
        the tool list is part of the cached prefix, so it is decided once per figure and is then
        identical for every call on that figure.
        """
        tools = [
            {"name": "crop_image",
             "description": ("Zoom into a rectangle of the image you were sent. Coordinates are "
                             "absolute pixels of that image. Returns the cropped region as a new "
                             "image plus the mapping back to the original coordinates."),
             "input_schema": _obj({
                 "x0": {"type": "number"}, "y0": {"type": "number"},
                 "x1": {"type": "number"}, "y1": {"type": "number"},
                 "zoom": {"type": ["number", "null"]}})},
            {"name": "overlay_points",
             "description": ("Draw numbered marks at coordinates you supply onto the figure and "
                             "return the result, so you can check whether they land on the data "
                             "you meant. Coordinates are pixels of the image you were sent."),
             "input_schema": _obj({"points": {"type": "array", "items": _obj({
                 "x": {"type": "number"}, "y": {"type": "number"},
                 "label": {"type": "string"}})}})},
        ]
        if self.has_regions:
            tools.insert(1, {
                "name": "list_regions",
                "description": ("List the axis lines, tick marks, OCR'd tick labels and detected "
                                "bars/markers a computer-vision pass found, in the coordinates of "
                                "the image you were sent."),
                "input_schema": {"type": "object", "additionalProperties": False,
                                 "required": [], "properties": {}}})
        return tools

    def handlers(self) -> dict[str, Any]:
        out = {"crop_image": self.handle_crop_image,
               "overlay_points": self.handle_overlay_points}
        if self.has_regions:
            out["list_regions"] = self.handle_list_regions
        return out

    def handle_list_regions(self, _: dict[str, Any]) -> str:
        return self.regions_text()

    def handle_crop_image(self, inp: dict[str, Any]) -> list[dict[str, Any]]:
        w, h = self.image.size
        x0, y0, x1, y1 = (_number(inp.get(k)) for k in ("x0", "y0", "x1", "y1"))
        if None in (x0, y0, x1, y1):
            raise ValueError("crop_image needs numeric x0, y0, x1, y1")
        x0, x1 = sorted((max(0.0, min(float(x0), w)), max(0.0, min(float(x1), w))))
        y0, y1 = sorted((max(0.0, min(float(y0), h)), max(0.0, min(float(y1), h))))
        if x1 - x0 < _MIN_CROP_PX or y1 - y0 < _MIN_CROP_PX:
            raise ValueError(f"crop is {x1 - x0:.0f}x{y1 - y0:.0f} px; ask for at least "
                             f"{_MIN_CROP_PX}x{_MIN_CROP_PX} px inside 0..{w} x 0..{h}")
        box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        patch = self.image.crop(box)
        want = _number(inp.get("zoom")) or MAX_ZOOM
        zoom = max(1.0, min(float(want), MAX_ZOOM))
        if zoom > 1.0:
            patch = patch.resize((max(1, round(patch.width * zoom)),
                                 max(1, round(patch.height * zoom))), Image.LANCZOS)
        # only ever shrink from here: the 4x cap above is the whole magnification budget
        prepared = prepare_for_claude(patch, min_long_edge=0, max_upscale=1.0)
        effective = prepared.image.width / max(1e-9, box[2] - box[0])
        note = (f"This crop is sent-image px [{box[0]}..{box[2]}] x [{box[1]}..{box[3]}] shown at "
                f"zoom {effective:.2f}x. A point at (cx, cy) in this crop is at sent-image "
                f"({box[0]} + cx/{effective:.3f}, {box[1]} + cy/{effective:.3f}).")
        return [_pil_image_block(prepared.image), _text_block(note)]

    def handle_overlay_points(self, inp: dict[str, Any]) -> list[dict[str, Any]]:
        points = inp.get("points") or []
        if not isinstance(points, list) or not points:
            raise ValueError("overlay_points needs a non-empty `points` list of {x, y, label}")
        marks: list[dict[str, Any]] = []
        for p in points[:24]:
            if not isinstance(p, dict):
                continue
            x, y = _number(p.get("x")), _number(p.get("y"))
            if x is None or y is None:
                continue
            cx, cy = self.to_crop(float(x), float(y))
            marks.append({"x": cx, "y": cy, "label": str(p.get("label") or ""), "kind": "point"})
        if not marks:
            raise ValueError("no point had usable numeric x and y")
        self._overlays += 1
        out = self.work_dir / f"{self.crop_png.stem}.model_overlay{self._overlays}.png"
        draw_overlay(self.crop_png, marks, out)
        prepared = prepare_for_claude(Image.open(out).convert("RGB"))
        listing = "; ".join(f"{i}={m['label'] or 'point'}" for i, m in enumerate(marks, start=1))
        return [_pil_image_block(prepared.image),
                _text_block(f"Your points drawn on the figure ({listing}). Same coordinate frame "
                            f"as the original image.")]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_number(str(value))


# ----------------------------------------------------------------------------- path D
def _submit_tool(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {"name": "submit", "description": description, "strict": True, "input_schema": schema}


def _run(client: LLMClient, *, model: str, system: str, view: FigureView, ask: str,
         schema: dict[str, Any], submit_description: str, cell_key: str,
         cache_key_extra: str, task: str = "",
         max_tool_calls: int = MAX_TOOL_CALLS) -> ToolLoopResult:
    """One tool-use pass over `view`, with the figure image as a CACHED prefix.

    Block order is what makes prompt caching work (task 15 §A, measured live): the blocks that do
    not vary between two calls on the same figure come first — the size header, then the image,
    which carries the `cache_control` marker — and everything that varies (the target, the
    caption, this pass's extra instruction) comes after it. `system` must therefore be the same
    text for every variant, or the prefix differs before the image is even reached and each call
    writes its own cache entry instead of reading the previous one.
    """
    content: list[dict[str, Any]] = [_text_block(view.header()), view.image_block(cache=True)]
    if task:
        content.append(_text_block(task))
    content.append(_text_block(ask))
    messages = [{"role": "user", "content": content}]
    tools = view.tools() + [_submit_tool(schema, submit_description)]
    return client.tool_loop(model=model, system=system, messages=messages, tools=tools,
                            handlers=view.handlers(), final_tool="submit",
                            max_tokens=MAX_TOKENS, max_tool_calls=max_tool_calls,
                            prompt_version=PROMPT_VERSION, cell_key=cell_key,
                            cache_key_extra=cache_key_extra)


def read_out(client: LLMClient, crop_png: str | Path, caption: str, target: TargetSpec,
             model: str = "claude-opus-5", variant: str = "direct", *, sample: int = 0,
             view: FigureView | None = None, cell_key: str = "") -> ReadOut:
    """Path D: ask the model to read the values off the figure (one sample of one variant).

    `sample > 0` re-asks an *already used* (model, variant) pair. The prompt is identical, so the
    only thing that differs is the sampling — a weaker vote than a different variant or a different
    model family, and marked as such in provenance. It changes the cache key (so the second sample
    is a real second call rather than a replay of the first), and `sample=0` keeps the original key.
    """
    if variant not in READOUT_VARIANTS:
        raise ValueError(f"unknown read-out variant {variant!r}; have {list(READOUT_VARIANTS)}")
    if sample < 0:
        raise ValueError(f"sample must be >= 0, got {sample}")
    view = view or FigureView(crop_png)
    # the system prompt is IDENTICAL for every variant: what differs (target, caption, the
    # variant's own step 7) travels in the user turn, after the cached image block
    system = load_prompt("digitize_readout")
    task = render_prompt("digitize_task", TARGET=target.describe(),
                         CAPTION=(caption or "").strip() or "(no caption found by ingestion)",
                         VARIANT=load_prompt(f"digitize_readout_{variant}").strip())
    result = _run(client, model=model, system=system, view=view, task=task,
                  ask="Work through the steps above, then call `submit`.",
                  schema=READOUT_SCHEMA,
                  submit_description="Report the values you read off the figure.",
                  cell_key=cell_key, max_tool_calls=MAX_READOUT_TOOL_CALLS,
                  cache_key_extra=f"{PROMPT_VERSION}|readout|{variant}"
                                  + (f"|sample{sample}" if sample else ""))
    return _parse_readout(result, model=model, variant=variant, sample=sample)


def _one_row_per_group(rows: list[GroupReadOut]) -> tuple[list[GroupReadOut], set[str], list[str]]:
    """One reading answers once per group. `(kept, groups_seen, groups_answered_twice)`.

    A turn may hold several `submit` blocks and `LLMClient.tool_loop` merges them, so a model
    that answers twice about the same group arrives here as two rows. Every quorum downstream
    counts rows (`_readout_means`), so leaving both would let ONE reader be its own second
    witness — enough to outvote a correct axis ladder. The first row carrying a value wins; a
    second answer about the same group is a self-contradiction, not corroboration.
    """
    kept: dict[str, GroupReadOut] = {}
    twice: list[str] = []
    for row in rows:
        current = kept.get(row.group)
        if current is None:
            kept[row.group] = row
            continue
        twice.append(row.group)
        if current.mean is None and row.mean is not None:
            kept[row.group] = row
    return list(kept.values()), set(kept), sorted(set(twice))


def _parse_readout(result: ToolLoopResult, model: str, variant: str,
                   sample: int = 0) -> ReadOut:
    data = result.parsed or {}
    groups = []
    for row in data.get("groups") or []:
        if not isinstance(row, dict):
            continue
        groups.append(GroupReadOut(
            group=str(row.get("group") or "unknown"),
            label_read=str(row.get("label_read") or ""),
            mean=_number(row.get("mean")),
            error_half_length=_number(row.get("error_half_length")),
            error_upper=_number(row.get("error_upper")),
            error_lower=_number(row.get("error_lower")),
            error_sides=str(row.get("error_sides") or "unknown"),
            x_read=str(row.get("x_read") or ""),
            points=[PointRead(x_label=str(pt.get("x_label") or ""),
                              mean=_number(pt.get("mean")),
                              error_half_length=_number(pt.get("error_half_length")))
                    for pt in (row.get("points") or []) if isinstance(pt, dict)],
            confidence=float(_number(row.get("confidence")) or 0.0),
            notes=str(row.get("notes") or "")))
    ticks = [float(t) for t in (data.get("tick_labels") or []) if _number(t) is not None]
    groups, seen, twice = _one_row_per_group(groups)

    return ReadOut(
        status=str(data.get("status") or "found"), groups=groups,
        legend_says=str(data.get("legend_says") or ""), tick_labels=ticks,
        pixel_resolution_estimate=_number(data.get("pixel_resolution_estimate")),
        unit=str(data.get("unit") or ""), panel=str(data.get("panel") or ""),
        axis_read=str(data.get("axis_read") or ""),
        axis_direction_note=str(data.get("axis_direction_note") or ""),
        confidence=float(_number(data.get("confidence")) or 0.0),
        notes="; ".join(x for x in (
            str(data.get("notes") or ""),
            (f"this reader answered more than once about group(s) {', '.join(twice)}; the first "
             f"answer carrying a value was kept — one reader is one witness" if twice else "")
        ) if x),
        model=model, variant=variant, sample=sample,
        call_ids=list(result.call_ids), tool_calls=list(result.tool_calls),
        cost_usd=result.cost_usd, turns=result.turns)


# ----------------------------------------------------------------------------- path C
def coords(client: LLMClient, crop_png: str | Path, caption: str, target: TargetSpec,
           model: str = "claude-opus-5", *, view: FigureView | None = None,
           cell_key: str = "") -> CoordReadout:
    """Path C: ask for pixel coordinates, return them mapped into `crop_png` pixels."""
    view = view or FigureView(crop_png)
    system = load_prompt("digitize_coords")
    task = render_prompt("digitize_task", TARGET=target.describe(),
                         CAPTION=(caption or "").strip() or "(no caption found by ingestion)",
                         VARIANT="Report coordinates only; another program converts them.")
    result = _run(client, model=model, system=system, view=view, task=task,
                  ask="Zoom as needed, check yourself with `overlay_points`, then call `submit`.",
                  schema=COORDS_SCHEMA,
                  submit_description="Report the pixel coordinates you located.",
                  cell_key=cell_key, cache_key_extra=f"{PROMPT_VERSION}|coords")
    return _parse_coords(result, view=view, model=model)


def _parse_coords(result: ToolLoopResult, view: FigureView, model: str) -> CoordReadout:
    data = result.parsed or {}
    ticks: list[TickCoord] = []
    for row in data.get("ticks") or []:
        if not isinstance(row, dict):
            continue
        value, y = _number(row.get("value")), _number(row.get("y_px"))
        if value is None or y is None:
            continue
        ticks.append(TickCoord(value=value, y_px=y / view.scale))
    groups: list[GroupCoords] = []
    for row in data.get("groups") or []:
        if not isinstance(row, dict):
            continue
        groups.append(GroupCoords(
            group=str(row.get("group") or "unknown"),
            label_read=str(row.get("label_read") or ""),
            x_px=view.crop_px(_number(row.get("x_px"))),
            y_px=view.crop_px(_number(row.get("y_px"))),
            bar_x0_px=view.crop_px(_number(row.get("bar_x0_px"))),
            bar_x1_px=view.crop_px(_number(row.get("bar_x1_px"))),
            cap_top_px=view.crop_px(_number(row.get("cap_top_px"))),
            cap_bottom_px=view.crop_px(_number(row.get("cap_bottom_px"))),
            notes=str(row.get("notes") or "")))
    return CoordReadout(
        status=str(data.get("status") or "found"), ticks=ticks, groups=groups,
        unit=str(data.get("unit") or ""), panel=str(data.get("panel") or ""),
        confidence=float(_number(data.get("confidence")) or 0.0),
        notes=str(data.get("notes") or ""), model=model, scale=view.scale,
        call_ids=list(result.call_ids), tool_calls=list(result.tool_calls),
        cost_usd=result.cost_usd, turns=result.turns)


# ----------------------------------------------------------------------------- overlay verify
def overlay_verify(client: LLMClient, crop_png: str | Path, marks: list[dict[str, Any]],
                   target: TargetSpec, model: str = "claude-opus-5", *,
                   out_png: str | Path | None = None, cell_key: str = "") -> OverlayVerdict:
    """Draw the resolved marks and ask a *fresh* call which of them are in the wrong place.

    `marks` are `overlay.draw_overlay` dicts in crop pixels, each with a `label` saying what the
    mark claims to be. The returned list is `Mismatch` per numbered mark (verdict `ok` included),
    and carries `overlay_path` / `call_ids` / `cost_usd` for provenance.
    """
    crop_png = Path(crop_png)
    verdict = OverlayVerdict()
    if not marks:
        verdict.notes = "no marks to verify"
        return verdict
    out = Path(out_png) if out_png is not None else crop_png.with_suffix(".overlay.png")
    draw_overlay(crop_png, marks, out)
    verdict.overlay_path = str(out)

    view = FigureView(out, work_dir=out.parent)
    listing = "\n".join(f"{i}. {m.get('label') or 'a datum'}" for i, m in enumerate(marks, start=1))
    system = render_prompt("digitize_overlay_verify", TARGET=target.describe(), MARKS=listing)
    result = _run(client, model=model, system=system, view=view,
                  ask="Judge every numbered mark, then call `submit`.",
                  schema=OVERLAY_SCHEMA,
                  submit_description="Report one verdict per numbered mark.",
                  cell_key=cell_key, cache_key_extra=f"{PROMPT_VERSION}|overlay")
    data = result.parsed or {}
    seen: set[int] = set()
    for row in data.get("marks") or []:
        if not isinstance(row, dict):
            continue
        number = _number(row.get("number"))
        if number is None:
            continue
        n = int(number)
        if n in seen or not (1 <= n <= len(marks)):
            continue
        seen.add(n)
        verdict.append(Mismatch(number=n, verdict=str(row.get("verdict") or "unknown"),
                                reason=str(row.get("reason") or "")))
    for n in range(1, len(marks) + 1):                       # a mark the model skipped is unknown
        if n not in seen:
            verdict.append(Mismatch(number=n, verdict="unknown", reason="not judged by the model"))
    verdict.sort(key=lambda m: m.number)
    verdict.call_ids = list(result.call_ids)
    verdict.cost_usd = result.cost_usd
    verdict.model = model
    verdict.notes = str(data.get("notes") or "")
    return verdict


def summarize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact tool-call log for `Candidate.pixel_provenance` (inputs kept, images hashed)."""
    out: list[dict[str, Any]] = []
    for call in calls:
        out.append({"turn": call.get("turn"), "name": call.get("name"),
                    "input": json.loads(json.dumps(call.get("input", {}), default=str)),
                    "output": str(call.get("output", ""))[:200],
                    "image_hashes": [h[:12] for h in call.get("image_hashes", [])],
                    "is_error": bool(call.get("is_error"))})
    return out
