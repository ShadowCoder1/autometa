"""`write_outcome_outputs` — every artefact one outcome produces, in one call.

Task 10 pools an outcome and hands the result here; this writes the forest plot, that outcome's
extraction table, the leave-one-out table, the sensitivity set, the funnel/Egger pair and a
`pooled.json` that records the estimate together with the conventions that produced it.

A run with nothing poolable still writes `pooled.json` — with `k = 0`, the count of rows held for
review and a note saying why there is no forest. A missing file is a bug report; a file that says
"nothing could be pooled" is a finding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..models import EffectSizeRecord, OutcomeDef, StatsSettings, Verdict, Candidate
from ..stats.meta import MetaResult, prediction_interval
from .forest import forest_plot
from .tables import (dump_json, extraction_table, funnel_plot, leave_one_out_table,
                     sensitivity_outputs)

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
    return payload


def write_outcome_outputs(run_dir: str | Path, outcome: OutcomeDef,
                          rows: Sequence[EffectSizeRecord], pooled: MetaResult | None,
                          settings: StatsSettings, *,
                          needs_human_rows: Sequence[EffectSizeRecord] = (),
                          verdicts: Sequence[Verdict] = (),
                          candidates: Sequence[Candidate] = (),
                          moderators: Sequence[str] | None = None,
                          all_rows: Sequence[EffectSizeRecord] | None = None) -> dict[str, Path]:
    """Write one outcome's artefacts under `<run_dir>/results/<outcome.key>/`.

    `rows` are the rows that were pooled and `needs_human_rows` the ones held for review.
    `all_rows` is every row the outcome produced, including any a within-paper aggregation
    replaced (amendment A); the extraction table holds all of them and marks which were pooled,
    because a row that was combined away is still a row a reviewer has to be able to check.

    Returns `{artefact_key: path}` — the same keys the HTML report and the manifest link by.
    """
    directory = outcome_dir(run_dir, outcome.key)
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    if pooled is not None and rows:
        forest = forest_plot(rows, pooled, outcome, settings, directory / "forest",
                             needs_human_rows=needs_human_rows, moderators=moderators)
        out.update({f"forest_{k}": v for k, v in forest.items()})

    table = extraction_table(list(all_rows) if all_rows is not None
                             else [*rows, *needs_human_rows],
                             directory / "extraction_table",
                             verdicts=verdicts, candidates=candidates, primary=rows)
    out.update({f"extraction_{k}": v for k, v in table.items()})

    loo = leave_one_out_table(rows, settings, directory / "leave_one_out")
    out.update({f"leave_one_out_{k}": v for k, v in loo.items()})

    sensitivity = sensitivity_outputs(rows, settings, directory / "sensitivity",
                                      needs_human_rows=needs_human_rows)
    out.update({f"sensitivity_{k}": v for k, v in sensitivity.items()})

    funnel = funnel_plot(rows, settings, directory / "funnel")
    out.update({f"funnel_{k}": v for k, v in funnel.items()})

    out["pooled_json"] = dump_json(
        _pooled_payload(pooled, rows, needs_human_rows, outcome, settings),
        directory / "pooled.json")
    return out
