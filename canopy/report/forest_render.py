"""Which renderer draws a forest, and what the report is told about that choice.

One entry point — `render_forest` — so that no caller has to know R exists. It tries
`meta::forest.meta` and falls back to `canopy.report.forest` whenever R is absent, broken, slow
or in disagreement, and it always comes back with files and a `RenderInfo` saying who drew them
and why. A run on a machine without R produces the same artefacts under the same names as a run
on one with it; the difference is recorded in `pooled.json`, printed under the figure and written
into `methods.md`, never left for a reader to notice from the typeface.

The fallback is not an error path that happens to work. A cross-check failure means the picture
and the tables are two different analyses, and the ONLY safe picture is then the one drawn from
the numbers the tables were written from — so the fallback is the correct output, and the loud
warning beside it is the finding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..models import EffectSizeRecord, OutcomeDef, Protocol, StatsSettings
from ..stats.meta import MetaResult
from . import labels as labels_mod
from .forest import forest_plot
from .forest_r import (FORMATS, RENDERER_MATPLOTLIB, RENDERER_R, ForestCrossCheckError,
                       ForestRenderError, RenderInfo, r_available, render_forest_r)
from .theme import BEST_GUESS_CAVEAT

__all__ = ["render_forest", "RenderInfo", "caption_line", "methods_line"]


def render_forest(rows: Sequence[EffectSizeRecord], pooled: MetaResult, outcome: OutcomeDef,
                  settings: StatsSettings, protocol: Protocol | None, out_stem: str | Path, *,
                  line: str = "strict", best_guess_ids: Sequence[str] = (),
                  strict_pooled: MetaResult | None = None,
                  moderators: Sequence[str] | None = None,
                  needs_human_rows: Sequence[EffectSizeRecord] = (),
                  warnings: list[str] | None = None) -> tuple[dict[str, Path], RenderInfo]:
    """Draw one forest with the best renderer available; returns `({format: path}, RenderInfo)`.

    `protocol` is what makes the R renderer possible at all — every column, label and axis on that
    plot is read off it — so a caller with no protocol gets the matplotlib plot and a reason
    saying exactly that, rather than a plot with invented headings.
    """
    stem = Path(out_stem)
    reason = ""
    crosscheck = None
    info = r_available()
    if getattr(settings, "dependency", "independent") == "cluster_robust":
        # not a defect and not warned: meta::forest.meta has no robumeta-style model, so an R
        # render would draw a classic/HK interval under our robust diamond — two analyses in
        # one picture, the exact failure the cross-check exists to prevent. Recorded, not tried.
        reason = ("cluster-robust variance (RVE): meta::forest.meta has no robumeta-style "
                  "model, so the plot is drawn from canopy's own pooled result")
    elif protocol is None:
        reason = "no protocol was passed, so the R renderer's labels cannot be read"
    elif info is not None and info.formats and set(FORMATS) - set(info.formats):
        # asked and answered once per process: this R has no device for one of the formats the
        # report links, so there is no point paying for a render it would have to throw away
        absent = ", ".join(sorted(set(FORMATS) - set(info.formats)))
        reason = (f"{RENDERER_R} cannot write every format this report links ({absent}): "
                  f"this R has no device for it")
        _warn(warnings, f"{outcome.key} ({line}): {reason}. Install the R package `svglite` (or "
                        f"the cairo libraries) to have R draw the forests.")
    else:
        try:
            result = render_forest_r(rows, pooled, outcome, settings, protocol, stem, line=line,
                                     best_guess_ids=best_guess_ids, strict_pooled=strict_pooled)
            missing = [fmt for fmt in FORMATS if fmt not in result.paths]
            if not missing:
                return dict(result.paths), result.info
            # every format the report links has to be the SAME picture. R here can draw the plot
            # but not, on this machine, write one of the files — so the whole figure is redrawn
            # rather than served half from one renderer and half from another.
            for path in result.paths.values():
                path.unlink(missing_ok=True)
            reason = (f"{RENDERER_R} cannot write every format this report links "
                      f"({', '.join(missing)}): {result.info.reason}")
            _warn(warnings, f"{outcome.key} ({line}): {reason}")
        except ForestCrossCheckError as exc:
            reason = str(exc)
            crosscheck = dict(exc.crosscheck)
            crosscheck["ok"] = False
            _warn(warnings, f"{outcome.key} ({line}): the R forest was NOT used — {exc}. The "
                            f"plot was drawn by canopy.report.forest from canopy's own pooled "
                            f"result, and `canopy validate` will exit non-zero.")
        except ForestRenderError as exc:
            reason = str(exc)
            if info is not None:                           # R is here and still could not draw
                _warn(warnings, f"{outcome.key} ({line}): the R forest could not be drawn "
                                f"({exc}); canopy.report.forest drew it instead.")

    columns = None if protocol is None else labels_mod.forest_columns(protocol, outcome, settings)
    if moderators is None and protocol is not None:
        moderators = protocol.moderators
    paths = forest_plot(rows, pooled, outcome, settings, stem,
                        needs_human_rows=needs_human_rows, moderators=moderators,
                        columns=columns, best_guess_ids=best_guess_ids,
                        subtitle=BEST_GUESS_CAVEAT if line != "strict" else "")
    return paths, RenderInfo(renderer=RENDERER_MATPLOTLIB, reason=reason, crosscheck=crosscheck,
                             pi_drawn_by="canopy")


def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(message)


def caption_line(info: RenderInfo | dict | None) -> str:
    """`Drawn by …` — the sentence that goes under every forest in the report."""
    renderer, reason = _renderer_and_reason(info)
    if not renderer:
        return ""
    return f"Drawn by {renderer}{' — ' + reason if reason else ''}"


def methods_line(info: RenderInfo | dict | None) -> str:
    """The Methods sentence naming the renderer, its versions and the cross-check tolerance.

    The cross-check half is claimed only when R actually drew the plot. A run that fell back has
    nothing to cross-check — its plot came from the same numbers as its tables — and a Methods
    section that says otherwise is a false statement about the method (review M4's rule).
    """
    if info is None:
        return ""
    block = info.as_dict() if isinstance(info, RenderInfo) else dict(info)
    renderer = str(block.get("renderer") or "")
    if not renderer:
        return ""
    if renderer != RENDERER_R:
        reason = str(block.get("reason") or "")
        return (f"Forest plots: {renderer}"
                + (f" ({reason})" if reason else "")
                + "; the same pooled result the tables are written from.")
    r_version = str(block.get("r_version") or "")
    meta_version = str(block.get("meta_version") or "")
    where = f" (R {r_version}, meta {meta_version})" if r_version or meta_version else ""
    return (f"Forest plots: {renderer}{where}; REML cross-checked against canopy.stats.meta "
            f"to 1e-3.")


def _renderer_and_reason(info: RenderInfo | dict | None) -> tuple[str, str]:
    if info is None:
        return "", ""
    block = info.as_dict() if isinstance(info, RenderInfo) else dict(info)
    return str(block.get("renderer") or ""), str(block.get("reason") or "")
