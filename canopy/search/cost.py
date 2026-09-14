"""What a search costs, and — the number that dwarfs it — what the review it starts will cost.

Two different bills, and the second used to be invisible (design 03 §10, adversarial BLOCKER-3):

* **screening** is priced per record: `COST_PER_RECORD`, measured on real batches (v1 billed
  $0.00197 a record; the v2 rubric's six answers and a quote add ≈ 70 output tokens → $0.003);
* **the run** is priced per PAPER, in dollars: the review's own mapper and extractors read the
  whole PDF, and the ceiling build measured ≈ $7 on its most expensive paper. A search that fetches
  a hundred `unsure` PDFs is therefore committing the user to several hundred dollars at `begin`,
  and that sentence has to be said before the button is pressed, not discovered on the invoice.

`RUN_COST_PER_PAPER` is the default; `run_cost_per_paper()` prefers what the last completed run
actually cost per paper when that number is a price rather than a cache hit (a run replayed from
the disk cache bills cents, and cents would be the wrong thing to promise).

Nothing here knows a research field, and nothing here is a statistic about any study.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["COST_PER_RECORD", "SHARE_SCREEN", "SHARE_SNOWBALL", "PLAN_CALL", "SEED_REWRITE",
           "AUDIT_SHARE", "OPENALEX_SEARCH", "OPENALEX_SNOWBALL_ROUND", "RUN_COST_PER_PAPER",
           "MIN_REAL_RUN_COST", "run_cost_per_paper", "run_commit", "screening_cap", "predict"]

#: dollars per title/abstract screened. Measured: $0.393834 / 200 records on the v1 prompt
#: (10,193–11,721 input and 1,148–2,386 output tokens a batch) plus ≈ 70 output tokens for the
#: v2 rubric's answers and quote.
COST_PER_RECORD = 0.003
#: how much of `max_usd` the first screening pass may spend; the rest is citation chasing (0.20)
#: and the plan call, the seed rewrite, the exclude audit and overshoot (0.10).
SHARE_SCREEN = 0.70
SHARE_SNOWBALL = 0.20

#: the plan call, the one seed rewrite and the exclude audit's share of the screening bill
#: (design 03 §10; the plan call measured $0.0162–0.0165 live, under the $0.03 planned for it)
PLAN_CALL = 0.03
SEED_REWRITE = 0.03
AUDIT_SHARE = 0.10
#: OpenAlex at $0.001 a search request: 5 + 5 pages × 2 strings + the width checks; and the
#: filter requests of one citation-chasing round at $0.0001
OPENALEX_SEARCH = 0.03
OPENALEX_SNOWBALL_ROUND = 0.03

#: what reading one paper in the review costs, when no completed run has said otherwise. The
#: ceiling build measured the mapper alone at ≈ $7 on its most expensive paper.
RUN_COST_PER_PAPER = float(os.environ.get("CANOPY_RUN_COST_PER_PAPER", "") or 7.0)
#: below this per paper a completed run was served from the disk cache, not from a provider, and
#: its mean is a fact about the cache rather than a price
MIN_REAL_RUN_COST = 0.50


def screening_cap(max_usd: float | None, max_screened: int | None = None) -> int | None:
    """How many records the first pass may screen: an explicit `max_screened`, else what
    `SHARE_SCREEN` of the budget buys at `COST_PER_RECORD`; None when neither bounds it."""
    if max_screened is not None:
        return max(0, int(max_screened))
    if max_usd is None:
        return None
    return max(0, int(float(max_usd) * SHARE_SCREEN / COST_PER_RECORD))


def run_cost_per_paper(runs_dir: str | Path | None = None) -> tuple[float, str]:
    """`(usd per paper, where the number came from)`.

    The last completed run with a real bill wins over the default: a person who has just paid
    $9 a paper should be warned in their own currency. Runs that cost less than
    `MIN_REAL_RUN_COST` a paper were replayed from the cache and are skipped, and a `runs/`
    directory that is missing or unreadable falls back to the default without a word — this is
    a courtesy estimate, never a reason to stop a search.
    """
    default = float(os.environ.get("CANOPY_RUN_COST_PER_PAPER", "") or RUN_COST_PER_PAPER)
    if runs_dir is None:
        return default, "the default (CANOPY_RUN_COST_PER_PAPER)"
    root = Path(runs_dir)
    if not root.is_dir():
        return default, "the default (CANOPY_RUN_COST_PER_PAPER)"
    newest: tuple[str, float, str] | None = None
    for path in root.glob("*/job.json"):
        try:
            job: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(job.get("status")) != "done" or str(job.get("kind") or "run") != "run":
            continue
        try:
            n_files = int(job.get("n_files") or 0)
            cost = float(job.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            continue
        if n_files <= 0 or cost / n_files < MIN_REAL_RUN_COST:
            continue
        stamp = str(job.get("finished_at") or job.get("created_at") or "")
        if newest is None or stamp > newest[0]:
            newest = (stamp, cost / n_files, str(job.get("run_id") or path.parent.name))
    if newest is None:
        return default, "the default (CANOPY_RUN_COST_PER_PAPER)"
    return round(newest[1], 2), f"the last completed run ({newest[2]})"


def predict(max_usd: float | None, *, max_fetch_unsure: int | None = None,
            expected_unsure: int | None = None, per_paper: float | None = None,
            snowball: bool = True, rounds: int = 3) -> dict[str, Any]:
    """What a search with this budget will spend, before it starts (design 03 §10).

    `cap` records the first pass may screen and `screen_usd` their price; `snowball_records`
    and `snowball_usd` the citation-chasing share; `worst_case` the budget plus the plan call,
    the seed rewrite and the OpenAlex requests; `run_commit` what `begin` would then commit —
    `min(max_fetch_unsure, expected_unsure)` papers at `per_paper`. Everything is a constant
    times a count; the bench measures the real number beside it.
    """
    budget = float(max_usd) if max_usd is not None else None
    cap = screening_cap(budget)
    screen_usd = round((cap or 0) * COST_PER_RECORD, 2)
    snowball_records = (int(budget * SHARE_SNOWBALL / COST_PER_RECORD)
                        if budget is not None and snowball else 0)
    snowball_usd = round(snowball_records * COST_PER_RECORD, 2)
    extras = PLAN_CALL + SEED_REWRITE + OPENALEX_SEARCH + (
        OPENALEX_SNOWBALL_ROUND * rounds if snowball else 0.0)
    worst = round((budget or 0.0) + extras, 2)
    unsure = (int(max_fetch_unsure) if max_fetch_unsure is not None else 0)
    if expected_unsure is not None:
        unsure = min(unsure, int(expected_unsure))
    rate = float(per_paper) if per_paper is not None else run_cost_per_paper()[0]
    return {"max_usd": budget, "cap": cap, "screen_usd": screen_usd,
            "audit_usd": round(screen_usd * AUDIT_SHARE, 2),
            "snowball_records": snowball_records, "snowball_rounds": rounds if snowball else 0,
            "snowball_usd": snowball_usd, "plan_usd": PLAN_CALL, "worst_case_usd": worst,
            "run_commit": run_commit(0, unsure, rate)}


def run_commit(n_wanted: int, n_unsure_fetched: int, per_paper: float) -> dict[str, Any]:
    """The sentence `begin` owes the user, as numbers: how many papers a review would read, how
    many of them nobody was sure about, and what that costs at `per_paper`."""
    n_read = int(n_wanted) + int(n_unsure_fetched)
    return {"n_read": n_read, "n_wanted": int(n_wanted), "n_unsure": int(n_unsure_fetched),
            "per_paper_usd": round(float(per_paper), 2),
            "usd": round(n_read * float(per_paper), 2),
            "unsure_usd": round(int(n_unsure_fetched) * float(per_paper), 2)}
