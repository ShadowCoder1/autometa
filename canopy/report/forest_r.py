"""Draw a forest plot with R's `meta::forest.meta`, and check its numbers against ours.

Why a second renderer at all: `meta` is what the field reads. A forest that came out of
`forest.meta` carries the layout, the diamond, the weights column and the heterogeneity line a
reviewer already knows how to read, and a figure that looks like the ones in the journals is one
fewer thing between the reader and the numbers. Canopy still computes every number itself
(`canopy.stats.meta`) — R is asked to draw, not to decide.

Which makes the cross-check the point of this file rather than a nicety. The plot is drawn from
`TE`/`seTE` by an independent REML implementation, so the estimate under the diamond is R's, not
ours. If the two ever differ by more than rounding, the picture and the tables beside it are two
different analyses, and the only safe outcome is to stop using the picture: `_crosscheck` raises,
the caller falls back to `canopy.report.forest`, the manifest gets a loud warning, `pooled.json`
records every quantity that disagreed and by how much, and `canopy validate` exits non-zero.

Nothing here is allowed to hang a run. `Rscript` is probed once behind an `lru_cache`, every call
has a timeout, and every failure — no R, no `meta`, a device that will not open, a script that
died — comes back as a reason string the report prints, never as an exception a run has to catch.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import EffectSizeRecord, OutcomeDef, Protocol, StatsSettings
from ..stats.meta import MetaResult, prediction_interval
from . import labels as labels_mod
from .theme import BEST_GUESS_CAVEAT, MARK, MUTED, pi_label, study_label

__all__ = ["RInfo", "RenderInfo", "RenderResult", "ForestCrossCheckError", "ForestRenderError",
           "PREDICT", "R_SCRIPT", "RENDERER_R", "RENDERER_MATPLOTLIB", "CONFIRMED", "GUESSED",
           "FORMATS", "r_available", "xlim_for", "write_forest_inputs", "render_forest_r",
           "row_order", "pi_drawn_by"]

R_SCRIPT = Path(__file__).resolve().parent / "forest_meta.R"
RENDERER_R = "R meta::forest.meta"
RENDERER_MATPLOTLIB = "canopy.report.forest (matplotlib)"

#: the two levels of the best-guess forest's subgroup, in the order they are drawn
CONFIRMED = "Confirmed"
GUESSED = "Best guess"

#: our prediction-interval names → `meta`'s `method.predict`. A method NOT in this table is not
#: refused as an analysis — it is refused as something *meta* may draw: `prediction = FALSE`, our
#: own interval printed under the plot instead, and `pi_drawn_by = "canopy"` on the record.
PREDICT: dict[str, str] = {"HTS": "HTS", "V": "V", "z": "S"}

PROBE_SECONDS = 20.0
RENDER_SECONDS = 180.0
FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
FONTFAMILY = "Helvetica"
COLGAP_LEFT = "5mm"
COLGAP_RIGHT = "6mm"

#: cross-check tolerances (DECISION F). `tau2` passes on EITHER the absolute or the relative
#: bound, because a τ² of 1e-9 and one of 2e-9 differ by 100 % and by nothing that matters.
TOL_EFFECT = 1e-3
TOL_TAU2_ABS = 1e-6
TOL_TAU2_REL = 1e-3
TOL_I2 = 1e-3
TOL_PI = 1e-3


class ForestRenderError(RuntimeError):
    """R could not draw the plot — no Rscript, no `meta`, a dead script, a timeout."""


class ForestCrossCheckError(RuntimeError):
    """R's pooled numbers and ours disagree by more than rounding. Names every quantity."""

    def __init__(self, message: str, crosscheck: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.crosscheck: dict[str, Any] = crosscheck or {}


@dataclass(frozen=True)
class RInfo:
    rscript: str
    r_version: str
    meta_version: str
    #: the output formats this machine's R can actually open a device for. Probed once, with the
    #: versions, so a caller can decide it cannot use R at all without paying for a render first
    formats: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderInfo:
    """Who drew the figure, on what, and whether its numbers survived the cross-check."""

    renderer: str
    reason: str = ""
    r_version: str = ""
    meta_version: str = ""
    fontfamily_used: str = ""
    pi_drawn_by: str = "meta"
    crosscheck: dict | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"renderer": self.renderer, "reason": self.reason, "r_version": self.r_version,
                "meta_version": self.meta_version, "fontfamily_used": self.fontfamily_used,
                "pi_drawn_by": self.pi_drawn_by, "crosscheck": self.crosscheck}


@dataclass(frozen=True)
class RenderResult:
    paths: dict[str, Path] = field(default_factory=dict)
    info: RenderInfo = field(default_factory=lambda: RenderInfo(renderer=RENDERER_R))


# ----------------------------------------------------------------------------- availability
@lru_cache(maxsize=1)
def r_available() -> RInfo | None:
    """`RInfo` when an R that can run `forest_meta.R` is on this machine, else `None`.

    Probes once per process (and per `cache_clear()`): the answer cannot change under a run, and
    starting R costs about a second. A probe that hangs, dies or reports a `meta` older than the
    script's contract is the same answer as no R at all — the caller draws the plot itself.
    """
    rscript = shutil.which("Rscript")
    if not rscript:
        return None
    try:
        done = subprocess.run([rscript, "--vanilla", str(R_SCRIPT), "--probe"],
                              capture_output=True, text=True, timeout=PROBE_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        payload = json.loads(_last_json_line(done.stdout))
    except (ValueError, IndexError):
        return None
    if not payload.get("ok"):
        return None
    return RInfo(rscript=rscript, r_version=str(payload.get("r_version", "")),
                 meta_version=str(payload.get("meta_version", "")),
                 formats=tuple(str(f) for f in (payload.get("formats") or ())))


def _last_json_line(text: str) -> str:
    """The script's one JSON object, even if something upstream printed a banner first."""
    for line in reversed([ln.strip() for ln in (text or "").splitlines() if ln.strip()]):
        if line.startswith("{"):
            return line
    raise IndexError("no JSON object on stdout")


# ----------------------------------------------------------------------------- the x axis
def _ceil_to_half(value: float) -> float:
    return math.ceil(float(value) * 2.0 - 1e-9) / 2.0


def xlim_for(bounds: Sequence[tuple[float, float]],
             override: Sequence[float] | None) -> list[float]:
    """The forest's x range: the protocol's if it pinned one, else symmetric around zero.

    Symmetric and rounded to the half, because a forest is read by comparing arrow lengths across
    the null line and an axis that is longer on one side makes an effect look asymmetric when it
    is not. The floor of ±1 keeps a set of tiny effects from being blown up to fill the page.
    """
    if override is not None:
        values = [float(v) for v in override]
        if len(values) != 2 or not (values[0] < values[1]):
            raise ValueError(f"forest_xlim must be [low, high] with low < high, "
                             f"got {list(override)!r}")
        return values
    largest = 0.0
    for pair in bounds:
        for value in pair:
            if value is None:
                continue
            value = float(value)
            if value == value and abs(value) > largest:      # NaN is not a bound
                largest = abs(value)
    limit = max(1.0, _ceil_to_half(largest))
    return [-limit, limit]


def pi_drawn_by(settings: StatsSettings) -> str:
    """`meta` when R can draw this review's prediction interval, `canopy` when only we can."""
    return "meta" if str(settings.pi_method) in PREDICT else "canopy"


def _bounds(rows: Sequence[EffectSizeRecord], pooled: MetaResult,
            settings: StatsSettings) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for row in rows:
        low, high = row.ci_low, row.ci_high
        if (low is None or high is None) and row.es is not None and row.se:
            low, high = row.es - 1.96 * row.se, row.es + 1.96 * row.se
        if low is not None and high is not None:
            out.append((float(low), float(high)))
    out.append((float(pooled.ci_low), float(pooled.ci_high)))
    low, high, _df = _our_pi(pooled, settings)
    if low == low and high == high:
        out.append((low, high))
    return out


def _our_pi(pooled: MetaResult, settings: StatsSettings) -> tuple[float, float, float]:
    if pooled.k < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        return prediction_interval(pooled, str(settings.pi_method))
    except ValueError:
        return float("nan"), float("nan"), float("nan")


# ----------------------------------------------------------------------------- the inputs
ROW_COLUMNS: tuple[str, ...] = ("dataset_id", "studlab", "year", "TE", "seTE", "n_a", "n_b",
                                "n_label", "route", "analysis_line", "best_guess_rule")


def row_order(rows: Sequence[EffectSizeRecord]) -> list[EffectSizeRecord]:
    """Rows in the order the plot draws them — by effect, as `sortvar = TE` asks for.

    Sorted here rather than left to R because `type.study` and `col.square` are per-row VECTORS:
    they are aligned with the data frame, and a data frame R re-orders under them would paint the
    guessed rows' marker onto whichever rows happened to land in those positions.
    """
    return sorted(rows, key=lambda r: (float("inf") if r.es is None else float(r.es),
                                       study_label(r)))


def _n_label(record: EffectSizeRecord) -> str:
    return f"{record.n_a if record.n_a is not None else '—'}/" \
           f"{record.n_b if record.n_b is not None else '—'}"


def _row_cells(record: EffectSizeRecord, moderators: Sequence[str], line: str,
               best_guess_ids: Sequence[str]) -> list[str]:
    se = record.se
    if se is None and record.var:
        se = math.sqrt(float(record.var))
    guessed = line != "strict" and record.dataset_id in set(best_guess_ids)
    cells = [record.dataset_id,
             labels_mod.elide(study_label(record), labels_mod.MAX_LABEL_CH),
             str(record.citation.year or ""),
             repr(float(record.es)) if record.es is not None else "",
             repr(float(se)) if se is not None else "",
             "" if record.n_a is None else str(record.n_a),
             "" if record.n_b is None else str(record.n_b),
             _n_label(record), record.route,
             GUESSED if guessed else CONFIRMED, record.best_guess_rule]
    cells.extend(labels_mod.elide(record.moderators.get(name, "") or "—",
                                  labels_mod.MAX_MOD_CH) for name in moderators)
    return cells


#: inches per character of table text at `pointsize`, and per line of it — Helvetica at 10 pt is
#: about 5.5 pt wide and 12 pt tall, and R sizes the device, not the drawing, so these only have
#: to be generous enough that nothing is clipped and tight enough that the page is not half white
CHAR_IN = 0.082
LINE_IN = 0.20


#: the narrowest table the caveat is allowed to be wrapped into before it starts eliding
MIN_ADDLINE_CH = 48


def _left_ch(columns: labels_mod.ForestColumns, cells: Sequence[Sequence[str]]) -> int:
    """How many characters wide the left-hand table is — headings or content, whichever wins."""
    total = 0
    for index, heading in enumerate(columns.leftlabs):
        column = index + 1                                   # cells[]: dataset_id is column 0
        total += max([len(heading)] + [len(row[column]) for row in cells]) + 2
    return total


def _addlines(is_guess_line: bool, pi_text: str, width_ch: int) -> tuple[str, str]:
    """The two lines meta prints under the plot: the best-guess caveat, our own PI, or neither.

    Wrapped to the width of the left-hand TABLE, not to the figure. `forest.meta` puts an addline
    on the same row as the axis's direction labels whenever a heterogeneity line is printed above
    it, so a line that runs past the table runs straight through "Reduced in Old"; one that stops
    at the table's edge sits harmlessly beside it.
    """
    width_ch = max(MIN_ADDLINE_CH, int(width_ch))
    if not is_guess_line:
        return "", labels_mod.elide(pi_text, width_ch) if pi_text else ""
    if pi_text:                                            # both wanted; the caveat gets one line
        return labels_mod.elide(BEST_GUESS_CAVEAT, width_ch), labels_mod.elide(pi_text, width_ch)
    wrapped = textwrap.wrap(BEST_GUESS_CAVEAT, width_ch) or [""]
    return wrapped[0], labels_mod.elide(" ".join(wrapped[1:]), width_ch)


def _size(columns: labels_mod.ForestColumns, cells: Sequence[Sequence[str]],
          *, k: int, extra_lines: int,
          addlines: Sequence[str] = ()) -> tuple[float, float, float]:
    """`(width_in, height_in, plot_in)` — columns wide, rows tall, and the plot's own share.

    `plot_in` is passed on as `plotwidth`: meta's default 6 cm is a third of what a wide moderator
    table needs beside it, and a forest whose axis is narrower than its Author column is a table
    with a decoration, not a plot.
    """
    left_in = CHAR_IN * max(_left_ch(columns, cells),
                            max((len(line) for line in addlines), default=0))
    right_in = CHAR_IN * (sum(len(head) for head in columns.rightlabs) + 22)
    plot_in = round(min(8.0, max(3.6, 0.42 * (left_in + right_in))), 2)
    width = 0.8 + left_in + plot_in + right_in
    height = 0.55 + LINE_IN * (k + extra_lines)
    return (round(min(30.0, max(6.0, width)), 3), round(min(40.0, max(2.0, height)), 3), plot_in)


def write_forest_inputs(rows: Sequence[EffectSizeRecord], pooled: MetaResult,
                        outcome: OutcomeDef, settings: StatsSettings, protocol: Protocol, *,
                        line: str = "strict", best_guess_ids: Sequence[str] = (),
                        out_dir: str | Path) -> tuple[Path, Path]:
    """Write `rows.csv` and `options.json` for `forest_meta.R`; returns both paths.

    Separate from the call so the exact inputs of a published figure can be pinned by a test and
    re-run by hand in R — the two files ARE the figure's provenance.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = labels_mod.forest_columns(protocol, outcome, settings)
    moderators = columns.moderators
    ordered = row_order(rows)
    guessed = set(best_guess_ids)
    cells = [_row_cells(record, moderators, line, best_guess_ids) for record in ordered]

    csv_path = directory / "rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*ROW_COLUMNS, *(f"mod_{name}" for name in moderators)])
        writer.writerows(cells)

    mapped = str(settings.pi_method) in PREDICT
    our_low, our_high, _df = _our_pi(pooled, settings)
    is_guess_line = line != "strict"
    pi_text = ""
    if not mapped and our_low == our_low:
        pi_text = (f"{settings.ci_level * 100:g}% prediction interval "
                   f"({pi_label(settings, pooled)}), computed by canopy: "
                   f"[{our_low:.2f}, {our_high:.2f}]")
    addline1, addline2 = _addlines(is_guess_line, pi_text, _left_ch(columns, cells))
    # header, blank, the pooled row, heterogeneity, the axis and its labels; a subgroup adds a
    # heading and a summary per level plus the between-subgroup test; an addline pushes the
    # direction labels onto a row of their own
    extra_lines = 6 + (1 if mapped and pooled.k >= 3 else 0) + (5 if is_guess_line else 0) \
        + bool(addline1) + bool(addline2) + bool(addline1 or addline2)
    width_in, height_in, plot_in = _size(columns, cells, k=len(ordered),
                                         extra_lines=extra_lines,
                                         addlines=(addline1, addline2))

    options: dict[str, Any] = {
        "sm": "SMD",
        "method_tau": str(settings.tau2_method),
        "method_random_ci": "HK" if settings.hakn else "classic",
        "method_predict": PREDICT.get(str(settings.pi_method), "HTS"),
        # meta refuses a prediction interval below k = 3 and so do we; an UNMAPPED method is
        # refused at any k, and our own interval is printed under the plot instead
        "prediction": bool(mapped and pooled.k >= 3),
        "level": float(settings.ci_level),
        "print_tau2": False,
        "digits": int(settings.forest_digits),
        "fontfamily": FONTFAMILY,
        "sortvar": "TE",
        "xlim": xlim_for(_bounds(ordered, pooled, settings), settings.forest_xlim),
        "weight_study": "random",
        "label_left": columns.label_left,
        "label_right": columns.label_right,
        "smlab": columns.smlab,
        "title": outcome.label or outcome.key,
        "leftcols": list(columns.leftcols),
        "leftlabs": list(columns.leftlabs),
        "rightcols": list(columns.rightcols),
        "rightlabs": list(columns.rightlabs),
        "plotwidth": f"{plot_in * 2.54:.2f}cm",
        "colgap_forest_left": COLGAP_LEFT,
        "colgap_forest_right": COLGAP_RIGHT,
        "subgroup": "analysis_line" if is_guess_line else None,
        "subgroup_levels": [CONFIRMED, GUESSED] if is_guess_line else None,
        "type_study": ["circle" if record.dataset_id in guessed and is_guess_line else "square"
                       for record in ordered],
        "col_square": [MUTED if record.dataset_id in guessed and is_guess_line else MARK
                       for record in ordered],
        # a guessed row is a CIRCLE, and meta paints circles its own blue unless told otherwise —
        # which would spend the one accent this project reserves for the pooled summary
        "col_circle": MUTED,
        "text_addline1": addline1,
        "text_addline2": addline2,
        "formats": list(FORMATS),
        "width_in": width_in,
        "height_in": height_in,
        "pointsize": 10,
        "res": 300,
    }
    options_path = directory / "options.json"
    options_path.write_text(json.dumps(options, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return csv_path, options_path


# ----------------------------------------------------------------------------- cross-check
def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _check(name: str, ours: Any, theirs: Any, tol: float, *, level: str = "error",
           rel: float | None = None) -> dict[str, Any] | None:
    """One comparison, or `None` when neither engine produced the quantity."""
    a, b = _finite(ours), _finite(theirs)
    if a is None and b is None:
        return None
    if a is None or b is None:
        return {"quantity": name, "ours": a, "r": b, "diff": None, "tol": tol, "level": level,
                "ok": False}
    diff = abs(a - b)
    scale = max(abs(a), abs(b))
    ok = diff <= tol or (rel is not None and scale > 0 and diff / scale <= rel)
    return {"quantity": name, "ours": a, "r": b, "diff": diff, "tol": tol, "level": level,
            "ok": bool(ok)}


def _crosscheck(result: Mapping[str, Any], pooled: MetaResult, settings: StatsSettings, *,
                strict_pooled: MetaResult | None = None) -> dict[str, Any]:
    """Compare every pooled quantity R drew with the one canopy computed. Raises on a real gap.

    `k` must be exact — a plot of a different number of studies is not the analysis. τ² is the
    one quantity allowed a relative tolerance and, at k = 2, only a warning: REML on two studies
    sits on the boundary of what is identifiable and two solvers can stop in different places
    without either being wrong.
    """
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    r_k = result.get("k")
    checks.append({"quantity": "k", "ours": pooled.k, "r": None if r_k is None else int(r_k),
                   "diff": None, "tol": 0, "level": "error",
                   "ok": r_k is not None and int(r_k) == int(pooled.k)})
    tau2_level = "warning" if pooled.k == 2 else "error"
    candidates = [
        _check("estimate", pooled.estimate, result.get("TE_random"), TOL_EFFECT),
        _check("ci_low", pooled.ci_low, result.get("lower_random"), TOL_EFFECT),
        _check("ci_high", pooled.ci_high, result.get("upper_random"), TOL_EFFECT),
        _check("tau2", pooled.tau2, result.get("tau2"), TOL_TAU2_ABS, rel=TOL_TAU2_REL,
               level=tau2_level),
        _check("I2", pooled.I2, result.get("I2"), TOL_I2),
        _check("Q", pooled.Q, result.get("Q"), TOL_EFFECT, rel=TOL_EFFECT, level="warning"),
        _check("Q_df", pooled.Q_df, result.get("Q_df"), 0, level="warning"),
    ]
    if str(settings.pi_method) in PREDICT:
        our_low, our_high, _df = _our_pi(pooled, settings)
        candidates.append(_check("pi_low", our_low, result.get("pi_low"), TOL_PI))
        candidates.append(_check("pi_high", our_high, result.get("pi_high"), TOL_PI))
    checks.extend(check for check in candidates if check is not None)

    # the best-guess forest's Confirmed subgroup IS the strict analysis; a diamond that has moved
    # is a warning, because the two lines are pooled over different row sets and REML on the
    # subgroup is not the same fit as REML on the strict set alone
    subgroup = result.get("subgroup_TE") or {}
    if strict_pooled is not None and isinstance(subgroup, Mapping) and CONFIRMED in subgroup:
        check = _check(f"subgroup[{CONFIRMED}]", strict_pooled.estimate, subgroup[CONFIRMED],
                       TOL_EFFECT, level="warning")
        if check is not None:
            checks.append(check)

    failed = [c for c in checks if not c["ok"] and c["level"] == "error"]
    warned = [c for c in checks if not c["ok"] and c["level"] == "warning"]
    warnings.extend(_phrase(c) for c in warned)
    diffs = [c["diff"] for c in checks if c["diff"] is not None]
    payload = {
        "ok": not failed,
        "k": pooled.k,
        "df_predict": result.get("df_predict"),
        "df_predict_from": result.get("df_predict_from", ""),
        "tolerances": {"effect": TOL_EFFECT, "tau2_abs": TOL_TAU2_ABS, "tau2_rel": TOL_TAU2_REL,
                       "I2": TOL_I2, "prediction_interval": TOL_PI},
        "checks": checks,
        "failed": [c["quantity"] for c in failed],
        "warnings": warnings,
        "max_abs_diff": max(diffs) if diffs else 0.0,
    }
    if failed:
        raise ForestCrossCheckError(
            "R meta and canopy.stats.meta disagree on "
            + ", ".join(c["quantity"] for c in failed) + " — "
            + "; ".join(_phrase(c) for c in failed), payload)
    return payload


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _phrase(check: Mapping[str, Any]) -> str:
    diff = "" if check["diff"] is None else f", |Δ| {check['diff']:.3g} > {check['tol']:g}"
    return (f"{check['quantity']}: canopy {_number(check['ours'])}, "
            f"R {_number(check['r'])}{diff}")


# ----------------------------------------------------------------------------- the call
def render_forest_r(rows: Sequence[EffectSizeRecord], pooled: MetaResult, outcome: OutcomeDef,
                    settings: StatsSettings, protocol: Protocol, out_stem: str | Path, *,
                    line: str = "strict", best_guess_ids: Sequence[str] = (),
                    strict_pooled: MetaResult | None = None) -> RenderResult:
    """Draw one forest with `meta::forest.meta` and cross-check it. Raises on either failure.

    `ForestRenderError` means R could not draw it (no R, a dead script, a timeout, no device);
    `ForestCrossCheckError` means it drew something whose numbers are not ours. Both are the same
    instruction to the caller — draw it here instead and say so — and different findings.
    """
    info = r_available()
    if info is None:
        raise ForestRenderError("Rscript not found" if not shutil.which("Rscript")
                                else "Rscript cannot run canopy/report/forest_meta.R "
                                     "(is the R package `meta` >= 6.0 installed?)")
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canopy-forest-") as scratch:
        csv_path, options_path = write_forest_inputs(
            rows, pooled, outcome, settings, protocol, line=line,
            best_guess_ids=best_guess_ids, out_dir=scratch)
        command = [info.rscript, "--vanilla", str(R_SCRIPT), str(csv_path), str(options_path),
                   str(stem)]
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=RENDER_SECONDS)
        except subprocess.TimeoutExpired:
            raise ForestRenderError(
                f"Rscript did not finish within {RENDER_SECONDS:g}s and was killed") from None
        except OSError as exc:                             # pragma: no cover - probe caught it
            raise ForestRenderError(f"Rscript could not be run: {exc}") from None
    try:
        result = json.loads(_last_json_line(done.stdout))
    except (ValueError, IndexError):
        raise ForestRenderError(
            f"Rscript wrote no result (exit {done.returncode}): "
            f"{_tail(done.stderr) or _tail(done.stdout) or 'no output'}") from None
    if not result.get("ok"):
        raise ForestRenderError(f"R refused to draw the forest: {result.get('error', '?')}")

    paths = {fmt: Path(path) for fmt, path in (result.get("files") or {}).items()
             if Path(path).exists() and Path(path).stat().st_size > 0}
    if not paths:                                          # pragma: no cover - R dies first
        raise ForestRenderError("R reported success but wrote no file")
    missing = {fmt: _short(why) for fmt, why in (result.get("unavailable") or {}).items()}
    reason = "; ".join(f"no {fmt} (R device unavailable: {why})"
                       for fmt, why in sorted(missing.items()))

    crosscheck = _crosscheck(result, pooled, settings, strict_pooled=strict_pooled)
    return RenderResult(paths=paths, info=RenderInfo(
        renderer=RENDERER_R, reason=reason, r_version=str(result.get("r_version", "")),
        meta_version=str(result.get("meta_version", "")),
        fontfamily_used=str(result.get("fontfamily_used", "")),
        pi_drawn_by=pi_drawn_by(settings), crosscheck=crosscheck))


def _short(message: object, limit: int = 90) -> str:
    """An R error's first line, short enough to sit in a figure caption."""
    first = next((line.strip() for line in str(message).splitlines() if line.strip()), "")
    first = " ".join(first.split())
    return first if len(first) <= limit else first[: limit - 1] + "\u2026"


def _tail(text: str, lines: int = 6) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return " / ".join(kept)
