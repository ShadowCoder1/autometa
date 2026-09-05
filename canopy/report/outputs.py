"""`write_outcome_outputs` — every artefact one outcome produces, in one call.

Task 10 pools an outcome and hands the result here; this writes the forest plot, that outcome's
extraction table, the leave-one-out table, the sensitivity set, the funnel/Egger pair and a
`pooled.json` that records the estimate together with the conventions that produced it.

A run with nothing poolable still writes `pooled.json` — with `k = 0`, the count of rows held for
review and a note saying why there is no forest. A missing file is a bug report; a file that says
"nothing could be pooled" is a finding.

**Two analysis lines (DECISION A).** The strict line is the primary analysis and owns every
top-level key of `pooled.json`, `extraction_table.*`, `leave_one_out.*` and `forest.*` — nothing
here changes any of them. Beside it, `best_guess` records what the same outcome looks like when
each held row a named rule admits is taken at its own value: its own block in `pooled.json`, its
own leave-one-out table, its own sensitivity entry, and — only when it actually added a row and
has something to pool — its own forest. The line is computed here rather than in the pipeline
because it is a way of READING the analysis: no stage file changes, no row is rebuilt, and a run
that never asks for the second line is the same run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..models import EffectSizeRecord, OutcomeDef, Protocol, StatsSettings, Verdict, Candidate
from ..stats.meta import MetaResult, prediction_interval
from .conclusion import conclusion_payload, outcome_conclusion
from .forest_render import render_forest
from .tables import (dump_json, extraction_table, funnel_plot, leave_one_out_rows,
                     leave_one_out_table, pool_rows, poolable_rows, sensitivity_outputs)

__all__ = ["write_outcome_outputs", "outcome_dir"]


def outcome_dir(run_dir: str | Path, outcome_key: str) -> Path:
    return Path(run_dir) / "results" / outcome_key


def _pooled_payload(pooled: MetaResult | None, rows: Sequence[EffectSizeRecord],
                    needs_human_rows: Sequence[EffectSizeRecord], outcome: OutcomeDef,
                    settings: StatsSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "outcome_key": outcome.key, "outcome_label": outcome.label,
        "k": 0 if pooled is None else pooled.k,
        "k_papers": len({r.cluster_id or r.paper_id or r.dataset_id for r in rows}),
        "n_needs_human": len(needs_human_rows),
        "needs_human_dataset_ids": [r.dataset_id for r in needs_human_rows],
        "settings": settings.model_dump(mode="json"),
        "positive_direction_label": outcome.positive_direction_label,
        "negative_direction_label": outcome.negative_direction_label,
        "note": "",
    }
    if pooled is None:
        payload["note"] = ("nothing could be pooled for this outcome — fewer than two rows "
                           "reached the primary analysis")
        return payload
    try:
        pi_low, pi_high, pi_df = prediction_interval(pooled, str(settings.pi_method))
    except ValueError:                                     # pragma: no cover - guarded upstream
        pi_low = pi_high = pi_df = float("nan")
    payload.update(pooled.as_dict())
    payload.update({"pi_method": settings.pi_method, "pi_low_used": pi_low,
                    "pi_high_used": pi_high, "pi_df_used": pi_df,
                    "estimator": settings.estimator, "variance_method": settings.variance})
    if getattr(pooled, "robust_fallback", ""):
        # a limitation lives in the record itself, not only in a nested field a reader may skip
        payload["note"] = ("cluster-robust pooling was requested and not applied: "
                           f"{pooled.robust_fallback}")
    return payload


def _best_guess_line(primary_pre_agg: Sequence[EffectSizeRecord],
                     needs_human_rows: Sequence[EffectSizeRecord], outcome: OutcomeDef,
                     settings: StatsSettings):
    """`(rows, decisions, added, cells)` for the second line — aggregated once, over its own rows.

    Imported here rather than at module scope: the analysis package must not be pulled in just to
    import the report's writers, and `canopy.pipeline` imports this module.
    """
    from ..pipeline.bestguess import (BEST_GUESS_FLAG, best_guess_cells, best_guess_rows,
                                      mark_composites)

    rows, decisions = best_guess_rows(primary_pre_agg, needs_human_rows, outcome=outcome,
                                      settings=settings)
    if settings.one_row_per_paper and rows:
        from ..pipeline.aggregate import aggregate_one_row_per_paper

        rows = mark_composites(aggregate_one_row_per_paper(rows, settings).rows, decisions)
    return (rows, decisions, [r for r in rows if BEST_GUESS_FLAG in r.flags],
            best_guess_cells(rows, decisions, primary_pre_agg))


def write_outcome_outputs(run_dir: str | Path, outcome: OutcomeDef,
                          rows: Sequence[EffectSizeRecord], pooled: MetaResult | None,
                          settings: StatsSettings, *,
                          needs_human_rows: Sequence[EffectSizeRecord] = (),
                          verdicts: Sequence[Verdict] = (),
                          candidates: Sequence[Candidate] = (),
                          moderators: Sequence[str] | None = None,
                          all_rows: Sequence[EffectSizeRecord] | None = None,
                          primary_pre_agg: Sequence[EffectSizeRecord] | None = None,
                          best_guess_cells: dict[tuple[str, str], dict] | None = None,
                          protocol: Protocol | None = None,
                          warnings: list[str] | None = None) -> dict[str, Path]:
    """Write one outcome's artefacts under `<run_dir>/results/<outcome.key>/`.

    `rows` are the rows that were pooled and `needs_human_rows` the ones held for review.
    `all_rows` is every row the outcome produced, including any a within-paper aggregation
    replaced (amendment A); the extraction table holds all of them and marks which were pooled,
    because a row that was combined away is still a row a reviewer has to be able to check.

    `primary_pre_agg` are the pooled rows BEFORE that aggregation — the rows the best-guess line
    adds to, so that each line aggregates its own members (it defaults to `rows`, which is the
    same list when the protocol does not aggregate). `best_guess_cells` is a sink: the caller
    passes a dict and gets this outcome's per-row best-guess decisions back in it, so the
    run-wide extraction table can show the SAME decision this one does.

    `protocol` (labels and moderator names for the R renderer) and `warnings` (a sink for the
    renderer's cross-check) amend the standing signature; both callers pass them.

    Returns `{artefact_key: path}` — the same keys the HTML report and the manifest link by.
    """
    directory = outcome_dir(run_dir, outcome.key)
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    renderer_info = None
    if pooled is not None and rows:
        forest, renderer_info = render_forest(rows, pooled, outcome, settings, protocol,
                                              directory / "forest",
                                              moderators=moderators,
                                              needs_human_rows=needs_human_rows,
                                              warnings=warnings)
        out.update({f"forest_{k}": v for k, v in forest.items()})

    # --- the second line (DECISION A): strict, plus the held rows a named rule admits
    from ..pipeline.bestguess import best_guess_payload

    bg_rows, decisions, added, cells = _best_guess_line(
        list(primary_pre_agg) if primary_pre_agg is not None else list(rows),
        needs_human_rows, outcome, settings)
    bg_pooled = pool_rows(bg_rows, settings)
    bg_loo = leave_one_out_rows(bg_rows, settings)
    if best_guess_cells is not None:
        best_guess_cells.update(cells)
    # the weights come back from the pooler in the order IT was given the rows, so they are keyed
    # by the pooler's own filter rather than by a second one computed beside it
    bg_weights = {} if bg_pooled is None else {
        row.dataset_id: float(weight)
        for row, weight in zip(poolable_rows(bg_rows), bg_pooled.weights_pct)}
    bg_payload = best_guess_payload(pooled, bg_pooled, decisions, added_rows=added,
                                    loo_bg=bg_loo, rows=bg_rows, weights=bg_weights,
                                    settings=settings)

    table = extraction_table(list(all_rows) if all_rows is not None
                             else [*rows, *needs_human_rows],
                             directory / "extraction_table",
                             verdicts=verdicts, candidates=candidates, primary=rows,
                             best_guess=cells)
    out.update({f"extraction_{k}": v for k, v in table.items()})

    loo_rows = leave_one_out_rows(rows, settings)
    loo = leave_one_out_table(rows, settings, directory / "leave_one_out", table=loo_rows)
    out.update({f"leave_one_out_{k}": v for k, v in loo.items()})

    sensitivity = sensitivity_outputs(rows, settings, directory / "sensitivity",
                                      needs_human_rows=needs_human_rows,
                                      best_guess_rows=bg_rows)
    out.update({f"sensitivity_{k}": v for k, v in sensitivity.items()})

    funnel = funnel_plot(rows, settings, directory / "funnel",
                         center=None if pooled is None else pooled.estimate)
    out.update({f"funnel_{k}": v for k, v in funnel.items()})

    # a forest of a line that added nothing is the strict forest under another name, and one of a
    # single row is not a meta-analysis; either way it would only invite the wrong quotation
    bg_renderer_info = None
    if bg_payload["n_added"] >= 1 and bg_payload["k"] >= 2 and bg_pooled is not None:
        bg_forest, bg_renderer_info = render_forest(
            bg_rows, bg_pooled, outcome, settings, protocol,
            directory / "forest_best_guess", line="best_guess", moderators=moderators,
            best_guess_ids=[r.dataset_id for r in added], strict_pooled=pooled,
            needs_human_rows=[r for r in needs_human_rows
                              if not cells.get((r.dataset_id, r.outcome_key),
                                               {}).get("in_best_guess")],
            warnings=warnings)
        out.update({f"forest_best_guess_{k}": v for k, v in bg_forest.items()})
    # …and no second leave-one-out table unless the line is a different set of rows: the same
    # table under a second name is how two artefacts of one run start being read as two findings.
    # So `leave_one_out_best_guess.*` is absent whenever nothing was added, and below k = 3, where
    # `leave_one_out_rows` returns nothing — the artefact keys are optional for the same reason
    # `forest_best_guess_*` is.
    if added and bg_loo:
        bg_loo_out = leave_one_out_table(bg_rows, settings,
                                         directory / "leave_one_out_best_guess", table=bg_loo)
        out.update({f"leave_one_out_best_guess_{k}": v for k, v in bg_loo_out.items()})

    payload = _pooled_payload(pooled, rows, needs_human_rows, outcome, settings)
    # DECISION F: who drew each forest, on what, and whether R's pooled numbers and ours agreed.
    # The report prints it under the figure, `methods.md` names the versions, and
    # `canopy validate` exits non-zero when a cross-check failed.
    payload["renderer"] = None if renderer_info is None else renderer_info.as_dict()
    payload["renderer_best_guess"] = (None if bg_renderer_info is None
                                      else bg_renderer_info.as_dict())
    payload["analysis_lines"] = ["strict", "best_guess"]
    payload["best_guess"] = bg_payload
    # DECISION B: the paragraph is computed once, here, beside the numbers it is about — the
    # report and the SPA read it out of this file rather than each deriving a sentence of its own
    payload["conclusion"] = conclusion_payload(outcome_conclusion(
        outcome, settings, pooled=pooled, rows=rows, held=needs_human_rows,
        best_guess=bg_payload, loo=loo_rows,
        group_a=None if protocol is None else protocol.group_a,
        group_b=None if protocol is None else protocol.group_b))
    out["pooled_json"] = dump_json(payload, directory / "pooled.json")
    return out
