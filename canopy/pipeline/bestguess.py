"""Section A's best-guess line: the primary analysis, PLUS the rows it had to hold back.

A review that can only report what it is sure of reports two rows and calls the question open.
The rows it held back are not noise — most of them carry a number the resolver actually built,
withheld because one corroboration was missing, not because the value is wrong. The best-guess
line says what those rows imply IF each held value is taken at face value, and it says it as a
second analysis standing beside the first, never in place of it.

**Invariant A0.** Best guess = strict, plus rows. Nothing here modifies or removes a strict row:
`best_guess_rows` returns the strict rows it was given, unchanged and identical objects apart from
ordering, and appends copies of the held rows a rule admitted. `k_bg ≥ k_strict`, always.

**A rebuild happens in exactly one place.** DECISION A's rules take a held row's OWN `es`/`var` —
the number the resolver already produced — and nothing else. DECISION B (the answer tier, at the
bottom of this module) is the one deliberate exception: when a NAMED AUTHORITY in the record
points at a held cell's value, the tier enters it and derives the row through the SAME
prepare→resolve path the repool's rebuild uses — never arithmetic of its own — on verdict COPIES,
so every number a guessed row consumes is traceable to a named field of a named record, and no
stage file or live verdict changes. Orientation is untouched either way: a row is signed by a
human answer or by the tiebreak ballot long before this module sees it, and a row that is still
unsigned — or whose sign is contested — is vetoed rather than guessed at, by every tier.

Two things decide each held row, and every held row gets exactly one of them with a reason:

* a **veto** (`VETOES`, first match wins) — a named reason the row's value may not be borrowed at
  all. The two flag-driven vetoes import their vocabulary from `verify.confidence` rather than
  restating it, so a flag added to `ROW_REFUSAL_CODES` or `CONTRADICTING_FLAGS` starts vetoing
  here on the same commit. A flag a human answer retired is no longer on the rebuilt row, so it
  does not veto — which is the whole point of answering.
* a **rule** (`RULES`, first match wins) — the named ground on which the value is admitted, with
  the row's own value and variance carried through untouched.

A verifier's refutation is NOT a veto. A refutation is a dispute about a magnitude or a quantity,
and the best-guess line's job is to say what the reading implies while showing the dispute: the
reason opens with `disputed` and names it. What the line never resolves is a SIGN it does not
have — an unsigned row is vetoed, because guessing a direction would be inventing the finding,
and so is a row whose sign is CONTESTED (`sign_mismatch`: the direction the paper states and the
direction the extracted means imply disagree). A disputed magnitude is a number to show with its
dispute; a disputed direction is a different finding.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from ..models import (BEST_GUESS_RULE_NAMES, DispersionType, EffectSizeRecord, OutcomeDef,
                      StatsSettings)
from ..stats.effect_sizes import sd_from_ci_halfwidth
from ..stats.meta import MetaResult, prediction_interval
from ..verify.checks import severity_of
from .resolve import GROUP_ROUTES
from ..verify.confidence import (CONTRADICTING_FLAGS, INFERRED_PREMISE_FLAGS,
                                 ROW_REFUSAL_CODES, WITHHOLDING_FLAGS)
from .resolve import GROUP_STATISTICS_MISSING, PRECEDENCE_OVERRIDE, _ci_dist

__all__ = ["ANSWER_RULES", "BEST_GUESS_FLAG", "RULES", "VETOES", "VETO_ROW_FLAGS",
           "CONTRADICTED", "BestGuessDecision", "adjudication_index", "best_guess_rows",
           "best_guess_payload", "best_guess_cells", "decision_b_fired", "mark_composites"]

#: on every row the best-guess line added, and on any composite one of them went into
BEST_GUESS_FLAG = "best_guess"
#: the closed set of grounds on which a held row may enter the line, in match order
RULES: tuple[str, ...] = ("inferred_premise", "low_confidence_value", "precedence_override")
#: the closed set of reasons a held row may not, in match order (first wins)
VETOES: tuple[str, ...] = ("row_refusal", "contradicted_value", "orientation_unresolvable",
                           "one_group_only", "no_variance")
#: the resolver owns the refusal vocabulary — imported, never copied (standing ruling)
VETO_ROW_FLAGS = ROW_REFUSAL_CODES
#: "this may be a different quantity" — `verify.confidence`'s own set, likewise imported
CONTRADICTED = CONTRADICTING_FLAGS
#: a contested SIGN, not a contested magnitude: `checks.sign_check` raises it when the direction
#: the paper STATES and the direction the extracted means imply disagree. It vetoes under
#: `contradicted_value` (amendment, whole-branch review MAJOR 2) — the closed enum is unchanged,
#: because what a disputed sign contradicts is the row's direction, and DECISION A's invariant is
#: that this line never resolves a direction it does not have. Named and checked rather than
#: spelled inline, so renaming the code in `verify.checks` fails here on the same commit.
SIGN_DISPUTED = "sign_mismatch"
assert severity_of(SIGN_DISPUTED) == "error"
#: codes that record HOW a decision was made rather than a doubt about a reading. They are not
#: disputes and the reason must not quote them as ones (whole-branch review, MINOR 7).
PROVENANCE_PREFIX = "orientation_"

#: `resolve.available_routes` writes this sentence into `routes_rejected`/`not_convertible_reason`
#: when a group is not ready; from D1 onward the same fact is also a flag
#: (`GROUP_STATISTICS_MISSING`), but records resolved before it carry only the sentence.
#: An alternation, because the sentence changed once: records written before the per-group
#: diagnosis said "have no mean, group size and dispersion" whatever was actually missing, and
#: both spellings must keep matching or old runs' held rows change meaning on reload.
_NO_GROUP_STATS = re.compile(r"group\(s\)[^;]*\bhave no mean\b|group\(s\) not ready —")


@dataclass(frozen=True)
class BestGuessDecision:
    """What the line did with ONE held row, and why — the record a reviewer is owed.

    `rule` is set iff `admitted`, `veto` iff not; `reason` is never empty. `label` is the row's
    citation name, carried so the payload can name the row without the record.
    """

    dataset_id: str
    outcome_key: str
    admitted: bool
    rule: str = ""
    veto: str = ""
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    es: float | None = None
    var: float | None = None
    label: str = ""
    # --- DECISION B (the answer tier). Default-empty on every DECISION A decision, so a legacy
    # decision and a legacy payload are byte-identical to before the tier existed (fire-gating).
    #: per group: the standing findings the rule CROSSED, each `{finding, severity, subject,
    #: basis}` — the findings still stand; the question stays open in unchanged words (§3)
    stepped_past: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: per group: standing instances on a HUMAN-ANSWERED cell, masked from the row veto because
    #: the answer stands in for the judgement (A5's second ground — never conflated with the first)
    masked_on_answered: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: per-slot record of every hold an entered value answers: `{slot, group, hold, rule, value,
    #: source, siblings}` (rules §7, reviewer N1)
    answered_holds: list[dict[str, Any]] = field(default_factory=list)
    #: per group: what was entered, slot by slot, each value with its named source pair
    entered: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: the row the tier BUILT through the ordinary resolver — set only on a cell-rule admission;
    #: `best_guess_rows` appends this instead of a copy of the held row
    row: EffectSizeRecord | None = None


# ----------------------------------------------------------------------------- reading a row
def _label(record: EffectSizeRecord) -> str:
    """`First author Year` — the same name `report.tables` puts on a forest row.

    Restated here rather than imported: `canopy.report` draws with matplotlib at import time and
    the analysis must not pull a renderer in to name a row.
    """
    cite = record.citation
    name = ((cite.first_author or "").strip() or (cite.authors or "").strip()
            or (record.label or record.dataset_id or record.paper_id or "").strip() or "—")
    return f"{name} {cite.year}" if cite.year else name


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _has_value(record: EffectSizeRecord) -> bool:
    """Did the resolver actually build an effect size on this row?"""
    return (record.route != "not_convertible" and _usable(record.es)
            and _usable(record.var) and float(record.var) > 0)


def _group_statistics_missing(record: EffectSizeRecord) -> bool:
    """Are the two groups' statistics NOT both present on this row?

    Three witnesses, in order of how durable they are: D1's flag, the numbers the record kept,
    and the sentence the resolver wrote when it rejected every group route.
    """
    if GROUP_STATISTICS_MISSING in record.flags:
        return True
    inputs = record.inputs or {}
    if "mean_a" in inputs or "mean_b" in inputs:
        return not (_usable(inputs.get("mean_a")) and _usable(inputs.get("mean_b")))
    prose = "; ".join([record.not_convertible_reason, *record.routes_rejected.values()])
    return bool(_NO_GROUP_STATS.search(prose))


def _severity(flag: str) -> str:
    """The severity `verify.checks` DECLARED for a code, or `""` for one it does not declare."""
    try:
        return severity_of(flag)
    except KeyError:
        return ""


def _disputes(record: EffectSizeRecord) -> list[str]:
    """The doubts recorded ON the row — what the reason has to show when it admits it anyway.

    A dispute is what the check layer DECLARED as one: a code it raised at `error` or `warn`, plus
    the withholding sets. Reading the flag's spelling instead (`"disput" in name`) made the mark a
    coincidence — `unit_mismatch` was never quoted because its name says nothing, and
    `calibration_disputed` was quoted only because of four letters in the middle of its name, so
    renaming that code would have silently dropped the whole mark from a row whose two groups may
    be on different scales. Severity is the one place that fact is stated on purpose.

    The `orientation_*` codes are the exception, and for the same reason: they carry a severity
    because a cell settled by a majority must be reviewed, but what they record is HOW the
    direction was decided, not a doubt about a reading. Quoting `orientation_by_majority` under
    "disputed (…)" told a reader the tiebreak itself was contested (MINOR 7).
    """
    flags = set(record.flags)
    return sorted(((flags & WITHHOLDING_FLAGS)
                   | {f for f in flags if _severity(f) in ("error", "warn")})
                  - {f for f in flags if f.startswith(PROVENANCE_PREFIX)})


def _evidence(record: EffectSizeRecord, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"route": record.route, "confidence": str(record.confidence),
                           "flags": list(record.flags)}
    out.update({k: v for k, v in extra.items() if v})
    return out


# ----------------------------------------------------------------------------- the vetoes
def _veto(record: EffectSizeRecord) -> tuple[str, str, dict[str, Any]] | None:
    """`(veto, reason, evidence)` for a row the line may not take, or `None`. First match wins."""
    refused = sorted(set(record.flags) & VETO_ROW_FLAGS)
    if refused:
        return ("row_refusal",
                f"the resolver refused this row ({', '.join(refused)}); a value its own "
                f"conversion refused is not one the best-guess line may borrow",
                _evidence(record, refusal_flags=refused))

    contradicting = sorted(set(record.flags) & CONTRADICTED)
    if contradicting:
        return ("contradicted_value",
                f"{', '.join(contradicting)}: the evidence says this reading may not be the "
                f"quantity the cell asks for, and a best guess about the wrong quantity is not a "
                f"guess about this contrast — a human has to settle it first",
                _evidence(record, contradicting_flags=contradicting))

    if SIGN_DISPUTED in record.flags:
        return ("contradicted_value",
                "sign_mismatch: the paper's stated direction contradicts the extracted means, so "
                "the sign of this row is disputed — the best-guess line never resolves a sign it "
                "does not have",
                _evidence(record, contradicting_flags=[SIGN_DISPUTED]))

    if record.higher_is_better is None:
        return ("orientation_unresolvable",
                "which direction counts as better was never settled for this measure, so this "
                "row has no sign; the best-guess line resolves magnitudes, never a direction it "
                "does not have",
                _evidence(record))

    if not _has_value(record):
        if _group_statistics_missing(record):
            detail = record.not_convertible_reason or "; ".join(record.routes_rejected.values())
            return ("one_group_only",
                    f"the two groups' statistics are not both present, so there is no contrast "
                    f"to take a guess at: {detail[:400] or 'no group statistics were resolved'}",
                    _evidence(record, not_convertible_reason=record.not_convertible_reason))
        detail = record.not_convertible_reason or "; ".join(record.routes_rejected.values())
        return ("no_variance",
                f"this row carries no usable effect size and variance (route {record.route or '—'}"
                f", es {record.es}, var {record.var})"
                + (f": {detail[:400]}" if detail else ""),
                _evidence(record, not_convertible_reason=record.not_convertible_reason))
    return None


# ----------------------------------------------------------------------------- the admit rules
def _precedence_override(record: EffectSizeRecord) -> bool:
    return PRECEDENCE_OVERRIDE in record.flags


def _held_only_by_bucket(record: EffectSizeRecord) -> bool:
    """Held because of its confidence bucket alone — no row-level rule put it here."""
    return not _precedence_override(record)


def _rule(record: EffectSizeRecord) -> tuple[str, str, dict[str, Any]] | None:
    """`(rule, reason, evidence)` for a row the line admits, or `None`. First match wins."""
    value = (f"{record.route} → d = {float(record.es):+.4g} (var {float(record.var):.4g}, "
             f"se {'—' if record.se is None else format(float(record.se), '.4g')})")
    disputed = _disputes(record)
    mark = f"disputed ({', '.join(disputed)}) — " if disputed else ""

    inferred = sorted(set(record.flags) & INFERRED_PREMISE_FLAGS)
    if inferred and record.route not in GROUP_ROUTES:
        inferred = []                     # this conversion consumed no inferred value (see resolve)
    if inferred:
        # FIRST, deliberately: an inferred-premise row also satisfies `low_confidence_value`
        # (the resolver did build a value), and admitting it under that rule would hide WHY it is
        # held — the premise is the tool's inference from the paper's other captions, not the
        # paper's label for this number, and the reviewer deciding whether to trust the line is
        # owed that sentence. The vetoes have already run: an inferred row that is also refuted,
        # refused or unsigned never reaches any rule.
        return ("inferred_premise",
                f"{mark}built on a premise the tool inferred from the paper "
                f"({', '.join(inferred)}): {value}. The paper supplied the evidence; no person "
                f"has confirmed the reading of it, and the question is still open.",
                _evidence(record, disputed=disputed, inferred=inferred))

    if _held_only_by_bucket(record):
        return ("low_confidence_value",
                f"{mark}held for review ({record.confidence}) but the resolver did build a "
                f"value: {value}. The best-guess line takes it at its own value — nothing is "
                f"recomputed, re-signed or rescaled.",
                _evidence(record, disputed=disputed))

    if _precedence_override(record):
        from_route = record.route_overridden_from or "the route the precedence list chose"
        return ("precedence_override",
                f"{mark}the precedence list chose {from_route}, which converts to nothing, so "
                f"the row was rebuilt from a same-locator reading and held: {value}. "
                f"{record.precedence_override_reason}".strip(),
                _evidence(record, disputed=disputed,
                          route_overridden_from=record.route_overridden_from,
                          precedence_override_reason=record.precedence_override_reason))
    return None                                            # pragma: no cover - a rule always fits


def _decide(record: EffectSizeRecord) -> BestGuessDecision:
    common = {"dataset_id": record.dataset_id, "outcome_key": record.outcome_key,
              "label": _label(record)}
    vetoed = _veto(record)
    if vetoed is not None:
        veto, reason, evidence = vetoed
        return BestGuessDecision(admitted=False, veto=veto, reason=reason, evidence=evidence,
                                 es=record.es, var=record.var, **common)
    admitted = _rule(record)
    if admitted is None:                                   # pragma: no cover - unreachable today
        return BestGuessDecision(
            admitted=False, veto="no_variance",
            reason=f"no admit rule fits this row (route {record.route or '—'})",
            evidence=_evidence(record), es=record.es, var=record.var, **common)
    rule, reason, evidence = admitted
    return BestGuessDecision(admitted=True, rule=rule, reason=reason, evidence=evidence,
                             es=record.es, var=record.var, **common)


def best_guess_rows(strict: Sequence[EffectSizeRecord], held: Sequence[EffectSizeRecord], *,
                    outcome: OutcomeDef, settings: StatsSettings,
                    verdicts: Sequence[Any] = (), candidates: Sequence[Any] = (),
                    datasets: Mapping[str, Any] | None = None,
                    retirements: Mapping[tuple[str, str, str], Any] | None = None,
                    adjudications: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
                    cell_rules_enabled: bool = True,
                    cell_guesses: list[dict[str, Any]] | None = None,
                    ) -> tuple[list[EffectSizeRecord], list[BestGuessDecision]]:
    """`(the line's rows, one decision per held row)` — the strict rows first, then the admitted.

    `strict` are the primary analysis's rows BEFORE any within-paper aggregation and `held` the
    rows the confidence filter kept out of it; both come from `run._split_rows`. Aggregation is
    the caller's (`report.outputs`), once per line, so that a composite is built from the line it
    belongs to rather than from a mixture of the two.

    The returned strict rows are the objects passed in — not copies, not reordered among
    themselves, never edited (invariant A0). Each admitted row is a COPY carrying the flag, the
    rule and the reason; the record on disk is untouched, because best guess is a way of reading
    the analysis and not a stage that changes it.

    DECISION B: when `verdicts` (and friends) are threaded and `cell_rules_enabled`, the answer
    tier runs FIRST on each held row — a named authority in the record may enter a held cell's
    value, over standing objections it records (`stepped_past`), always superseded by any human
    answer (`retirements`, G6/G7). With no cell data, an empty `settings.best_guess_rules`, or
    `cell_rules_enabled=False`, every decision and reason is byte-identical to before the tier
    existed. `cell_guesses` is a sink: fires that could not complete a row (a sibling cell with
    no authority) are still first-class and land there.
    """
    rows = list(strict)
    decisions: list[BestGuessDecision] = []
    ctx = _cell_context(outcome=outcome, settings=settings, verdicts=verdicts,
                        candidates=candidates, datasets=datasets, retirements=retirements,
                        adjudications=adjudications,
                        cell_guesses=cell_guesses) if cell_rules_enabled else None
    for record in held:
        decision = _cell_decide(record, ctx) if ctx is not None else None
        if decision is None:
            decision = _decide(record)
            if ctx is not None:
                decision = _rename_agreed_solo(record, decision, ctx)
        decisions.append(decision)
        if decision.admitted:
            base = decision.row if decision.row is not None else record
            rows.append(base.model_copy(update={
                "flags": [*base.flags, BEST_GUESS_FLAG, f"best_guess_rule:{decision.rule}"],
                "best_guess_rule": decision.rule, "best_guess_reason": decision.reason}))
    return rows, decisions


def mark_composites(rows: Sequence[EffectSizeRecord],
                    decisions: Sequence[BestGuessDecision]) -> list[EffectSizeRecord]:
    """After aggregation: a composite with any guessed member is best guess wholesale.

    `aggregate_one_row_per_paper` builds a fresh record whose flags are the union of its members'
    — so `best_guess` arrives on the composite by itself — but its `best_guess_rule`/`reason`
    fields start empty. This fills them in, naming the members that were guessed, because a
    composite half of which is a guess is not a confirmed row and must not be drawn as one.
    """
    admitted = {d.dataset_id: d for d in decisions if d.admitted}
    out: list[EffectSizeRecord] = []
    for row in rows:
        members = [admitted[part] for part in row.dataset_id.split("+") if part in admitted]
        if not members or row.best_guess_rule:
            out.append(row)
            continue
        rules = sorted({d.rule for d in members})
        out.append(row.model_copy(update={
            "flags": sorted(set(row.flags) | {BEST_GUESS_FLAG}),
            "best_guess_rule": rules[0] if len(rules) == 1 else "composite",
            "best_guess_reason": (
                "this paper's combined row includes "
                + ", ".join(f"{d.dataset_id} ({d.rule})" for d in members)
                + " — a composite with a guessed member is a best guess wholesale")}))
    return out


def best_guess_cells(rows: Sequence[EffectSizeRecord],
                     decisions: Sequence[BestGuessDecision],
                     strict: Sequence[EffectSizeRecord] = (),
                     cell_guesses: Sequence[Mapping[str, Any]] = ()) -> dict[tuple[str, str], dict]:
    """`(dataset_id, outcome_key) -> the best-guess columns` for the extraction tables.

    Every row of the line gets an entry (a strict row is in the line, with no rule of its own),
    and so does every held row the line refused — with the veto and its reason, because "why is
    this row in neither analysis" is the question the table has to answer.

    `strict` is the line's PRE-aggregation membership. Without it, a strict row that the
    best-guess aggregation folded into a composite appears in neither the decisions nor the
    post-aggregation rows, and its cell would read `in_best_guess: None` — which the extraction
    table defines as "this run computed no second line", a different and false claim.
    """
    cells: dict[tuple[str, str], dict] = {}
    for record in strict:
        cells[(record.dataset_id, record.outcome_key)] = {
            "in_best_guess": True, "best_guess_rule": "", "best_guess_reason": "",
            "best_guess_es": record.es, "best_guess_se": record.se}
    for decision in decisions:
        entry = (
            {"in_best_guess": True, "best_guess_rule": decision.rule,
             "best_guess_reason": decision.reason, "best_guess_es": decision.es,
             "best_guess_se": (math.sqrt(decision.var)
                               if decision.var and decision.var > 0 else None)}
            if decision.admitted else
            {"in_best_guess": False, "best_guess_rule": "",
             "best_guess_reason": f"{decision.veto}: {decision.reason}",
             "best_guess_es": None, "best_guess_se": None})
        # DECISION B pass-through, consumed only by the queue/questions projection (the pinned
        # extraction-table columns copy a named list, so extra keys are inert there). PER
        # GROUP, because the queue is per-group: decorating the sibling of a guessed cell with
        # the guess's number would count a never-guessed cell as guessed (review F1). Each
        # slot carries ITS group's rule name — an entered number never prints without one.
        if decision.entered:
            rule_of = {h["group"]: h["rule"] for h in decision.answered_holds
                       if h["slot"] == "mean"}
            entry["best_guess_cell_guesses"] = {
                group: {"rule": rule_of.get(group, decision.rule),
                        "entered": _entered_summary({group: slots}),
                        "stepped_past": sorted({e["finding"] for e in
                                                decision.stepped_past.get(group, [])}),
                        "blocked_by": ""}
                for group, slots in decision.entered.items()}
        cells[(decision.dataset_id, decision.outcome_key)] = entry
    # a fire that could not complete a row is still first-class: its cell shows the entered
    # value and its crossings on the card, and `row_blocked_by` says why no row carries it —
    # under the FIRE's own rule name and only on the fired group's row (review F1)
    for guess in cell_guesses:
        key = (str(guess.get("dataset_id")), str(guess.get("outcome_key")))
        entry = cells.get(key)
        if entry is None:
            continue                    # pragma: no cover - a guessed cell always has a decision
        group = str(guess.get("group"))
        entry.setdefault("best_guess_cell_guesses", {}).setdefault(group, {
            "rule": str(guess.get("rule") or ""),
            "entered": _entered_summary({group: dict(guess.get("entered") or {})}),
            "stepped_past": sorted({e["finding"] for e in guess.get("stepped_past") or []}),
            "blocked_by": str(guess.get("row_blocked_by") or "")})
    # the line's own rows last: an admitted row that a within-paper aggregation combined away
    # keeps the entry its decision gave it above, and the composite that replaced it gets its
    # own — carrying forward the DECISION B pass-through keys its decision recorded, which the
    # fresh dict would otherwise silently drop
    for record in rows:
        prior = cells.get((record.dataset_id, record.outcome_key)) or {}
        entry = {
            "in_best_guess": True, "best_guess_rule": record.best_guess_rule,
            "best_guess_reason": record.best_guess_reason,
            "best_guess_es": record.es, "best_guess_se": record.se}
        if "best_guess_cell_guesses" in prior:
            entry["best_guess_cell_guesses"] = prior["best_guess_cell_guesses"]
        cells[(record.dataset_id, record.outcome_key)] = entry
    return cells


# ----------------------------------------------------------------------------- the payload
def _cluster(record: EffectSizeRecord) -> str:
    return record.cluster_id or record.paper_id or record.dataset_id


def _sign_agrees(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return math.copysign(1.0, a) == math.copysign(1.0, b)


def best_guess_payload(strict_pooled: MetaResult | None, bg_pooled: MetaResult | None,
                       decisions: Sequence[BestGuessDecision], *,
                       added_rows: Sequence[EffectSizeRecord],
                       loo_bg: Sequence[Mapping[str, Any]],
                       rows: Sequence[EffectSizeRecord] = (),
                       weights: Mapping[str, float] | None = None,
                       settings: StatsSettings | None = None,
                       cell_guesses: Sequence[Mapping[str, Any]] = (),
                       fired: bool = False) -> dict[str, Any]:
    """The `best_guess` block of `pooled.json` — the second line, next to the first.

    `added_rows` are the line's rows that are guesses (after aggregation, a composite with a
    guessed member is one of them); `rows` is the whole line; `loo_bg` is
    `tables.leave_one_out_rows` over it, so `max_abs_delta_from_one_best_guess_row` can say how
    much of the line's estimate rests on a single guessed row.

    `weights` is `dataset_id -> weight %`, built by the caller from the SAME filter that fed the
    pooler. Re-deriving the poolable subset here and zipping the weights onto it by index would
    put a plausible number beside the wrong row the first time the two filters disagreed, with
    nothing raised — so the map is passed in rather than recomputed.
    """
    line = list(rows) if rows else list(added_rows)
    added_ids = {r.dataset_id for r in added_rows}
    not_added = [d for d in decisions if not d.admitted]
    weight_of = dict(weights or {})

    payload: dict[str, Any] = {
        # the same rule the strict payload uses: `k` is what was POOLED, and is 0 when nothing
        # was. How many rows the line has is a different question, and has its own key.
        "k": bg_pooled.k if bg_pooled is not None else 0,
        "k_rows": len(line),
        "k_papers": len({_cluster(r) for r in line}),
        "estimate": None, "se": None, "ci_low": None, "ci_high": None, "z": None, "p": None,
        "tau2": None, "I2": None, "Q": None, "Q_df": None, "Q_p": None,
        "pi_method": str(settings.pi_method) if settings is not None else "",
        "pi_low": None, "pi_high": None, "pi_df": None,
        "n_added": len(added_rows), "n_still_held": len(not_added),
        "delta_vs_strict": None, "sign_agrees_with_strict": None,
        "max_abs_delta_from_one_best_guess_row": None,
        "added": [{"dataset_id": r.dataset_id, "label": _label(r),
                   "rule": r.best_guess_rule, "reason": r.best_guess_reason,
                   "es": r.es, "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high,
                   "weight_pct": weight_of.get(r.dataset_id),
                   **_answer_tier_extras(r, decisions)}
                  for r in added_rows],
        "not_added": [{"dataset_id": d.dataset_id, "label": d.label, "veto": d.veto,
                       "reason": d.reason} for d in not_added],
        "note": "",
    }

    if bg_pooled is not None:
        payload.update({
            "estimate": bg_pooled.estimate, "se": bg_pooled.se, "ci_low": bg_pooled.ci_low,
            "ci_high": bg_pooled.ci_high, "z": bg_pooled.z, "p": bg_pooled.p,
            "tau2": bg_pooled.tau2, "I2": bg_pooled.I2, "Q": bg_pooled.Q,
            "Q_df": bg_pooled.Q_df, "Q_p": bg_pooled.Q_p})
        pi_low, pi_high, pi_df = bg_pooled.pi_low, bg_pooled.pi_high, bg_pooled.pi_df
        if settings is not None:
            try:
                pi_low, pi_high, pi_df = prediction_interval(bg_pooled, str(settings.pi_method))
            except ValueError:                             # pragma: no cover - guarded upstream
                pi_low = pi_high = float("nan")
                pi_df = 0
        # …as `null` when it is not estimable, never as the bare token `NaN`: at k = 2 under
        # Hartung-Knapp (and under `pi_t` at any k the method cannot serve) the interval has no
        # width, and `NaN` is not JSON — every consumer that is not Python fails on the FILE
        # rather than on the number (review MINOR 21).
        payload.update({"pi_low": pi_low if _usable(pi_low) else None,
                        "pi_high": pi_high if _usable(pi_high) else None, "pi_df": pi_df})
        if strict_pooled is not None:
            payload["delta_vs_strict"] = bg_pooled.estimate - strict_pooled.estimate
            payload["sign_agrees_with_strict"] = _sign_agrees(bg_pooled.estimate,
                                                              strict_pooled.estimate)
        deltas = [abs(float(entry["estimate"]) - bg_pooled.estimate) for entry in loo_bg
                  if entry.get("estimate") is not None
                  and str(entry.get("omitted_dataset_id", "")) in added_ids]
        payload["max_abs_delta_from_one_best_guess_row"] = max(deltas) if deltas else None

    counts: dict[str, int] = {}
    for row in added_rows:
        counts[row.best_guess_rule or "composite"] = counts.get(row.best_guess_rule
                                                                or "composite", 0) + 1
    vetoes: dict[str, int] = {}
    for decision in not_added:
        vetoes[decision.veto] = vetoes.get(decision.veto, 0) + 1
    added_note = (", ".join(f"{rule} ×{n}" for rule, n in sorted(counts.items()))
                  if counts else "no held row could be admitted")
    held_note = (", ".join(f"{veto} ×{n}" for veto, n in sorted(vetoes.items()))
                 if vetoes else "nothing was held back")
    payload["note"] = (
        f"best guess = the primary analysis plus {len(added_rows)} of "
        f"{len(added_rows) + len(not_added)} held row(s), admitted by rule ({added_note}); "
        f"still held: {held_note}. This is not the primary analysis: every added row is one a "
        f"human has not confirmed, taken at its own value.")

    # --- DECISION B, fire-gated (rules §7 / M3): on a run where no answer-tier rule fired —
    # `best_guess_rules: []` included — none of these keys exist and the block is byte-identical
    # to the pre-tier payload. `by_rule` counts each ROW once under its joined name;
    # `by_rule_totals` counts each simple rule name across joined names (reviewer S2 — both,
    # defined). `cell_guesses` are fires with no completable row (`row_blocked_by` says why).
    if fired:
        payload["rules_applied"] = list(settings.best_guess_rules) if settings is not None else []
        by_rule: dict[str, int] = {}
        totals: dict[str, int] = {}
        for row in added_rows:
            joined = row.best_guess_rule or "composite"
            by_rule[joined] = by_rule.get(joined, 0) + 1
            for part in joined.split(";"):
                totals[part] = totals.get(part, 0) + 1
        payload["by_rule"] = dict(sorted(by_rule.items()))
        payload["by_rule_totals"] = dict(sorted(totals.items()))
        payload["cell_guesses"] = [dict(g) for g in cell_guesses]
    return payload


# ============================================================================ DECISION B
# The ANSWER TIER: a held cell's value entered by a NAMED AUTHORITY in the record, over standing
# objections the guess records (`stepped_past`) and never retires — the question stays open in
# unchanged words, and any human answer on the cell disables every rule there permanently (G7).
# Three authorities, in match order: the adjudicator's settled ruling (a), the settings-ordered
# winner of a cross-route disagreement (b), and >=2 independent model families agreeing on the
# standing value (c). An objection whose subject is the authority's own basis is self-undermining
# and uncrossable; a finding with no `candidate_ids` has an UNKNOWABLE subject and is never
# crossable and never maskable (A2). Nothing here writes a stage file, an override, or a question:
# a guess is a pure function of the record plus the override log, recomputed at every evaluation.

#: the closed catalogue, in match order — asserted equal to the models literal so a rename fails
#: on the same commit (the `SIGN_DISPUTED` pattern above)
ANSWER_RULES: tuple[str, ...] = ("adjudicated_value", "route_precedence", "agreed_solo_read")
assert ANSWER_RULES == BEST_GUESS_RULE_NAMES
#: A2's promotion: an orientation dispute joins the F3 absolutes — this line never resolves a
#: direction it does not have, and no subject attribution can cross a checkable claim about
#: which group is higher. Asserted into `CONTRADICTING_FLAGS` so a rename fails here too.
ORIENTATION_CONTRADICTED = "orientation_reader_contradicts_values"
assert ORIENTATION_CONTRADICTED in CONTRADICTING_FLAGS
#: never crossable, never maskable, never in `stepped_past` (F3)
F3_ABSOLUTES: frozenset[str] = frozenset({SIGN_DISPUTED, ORIENTATION_CONTRADICTED})
#: the donor key's label tiebreak: on equal converted SE prefer the label needing the larger SE
#: reading. It survives ONLY as the tiebreak among distinct eligible readings — it never again
#: resolves what one reading's label is (A4).
_TYPE_CONSERVATISM: dict[str, int] = {"SE": 0, "SD": 1, "CI95": 2, "CI90": 3}
_CI_LEVEL: dict[str, float] = {"CI95": 0.95, "CI90": 0.90}


def _dt(value: Any) -> str:
    """A dispersion type as its bare string value (`DispersionType` or already a string)."""
    return getattr(value, "value", None) or str(value or "")


def _same_reading(a: float | None, b: float | None) -> bool:
    """`overrides.within_read_tolerance` — the codebase's own two-numbers-are-one-reading line.

    Imported lazily: `pipeline.overrides` pulls the report package in at import time and the
    analysis must not do that just to compare two floats (A3 pins THIS tolerance, never the
    vote window — Coudière's 1.0 vote window would cross the record's own 3.78-vs-3.94).
    """
    from .overrides import within_read_tolerance

    return within_read_tolerance(a, b)


def adjudication_index(run_dir: str | Path | None,
                       paper_ids: Collection[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """`(dataset_id, outcome_key) -> adjudication METADATA` from the named papers' verify stage.

    Settledness (`needs_human`), the ruling's basis (`chosen_candidate_ids`) and each group's
    locator are not on the `Verdict`, and they are not cell VALUES — the ruling's numbers ARE on
    the live verdict, which is the only place the engine reads them from (supersede rule, G0).
    This loader takes ORDER and provenance from the stage file, never a number: the lazy-read
    pattern integration G2 documents. A run with no `verify.json` yields nothing, and every rule
    that needs an adjudication then simply finds none.
    """
    if run_dir is None:
        return {}
    from .state import paper_dir

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for paper_id in sorted(set(paper_ids)):
        path = paper_dir(Path(run_dir), paper_id) / "verify.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:                                 # pragma: no cover - defensive
            continue
        for adj in payload.get("adjudications") or []:
            key = (str(adj.get("dataset_id") or ""), str(adj.get("outcome_key") or ""))
            entry = {
                "needs_human": bool(adj.get("needs_human")),
                "chosen_candidate_ids": [str(x) for x in adj.get("chosen_candidate_ids") or []],
                "rationale": str(adj.get("rationale") or ""),
                "locators": {str(g.get("group")): str(g.get("locator") or "")
                             for g in adj.get("groups") or []},
            }
            prior = out.get(key)
            if prior is not None:
                # the record can carry two adjudications for one cell (it does, three times);
                # settledness gates rule (a), so a disagreeing pair reads as UNSETTLED —
                # doubt goes against the stronger authority (review F5)
                entry["needs_human"] = entry["needs_human"] or prior["needs_human"]
            out[key] = entry
    return out


@dataclass
class _CellContext:
    """Everything the answer tier reads, threaded once from the evaluation point."""

    outcome: OutcomeDef
    settings: StatsSettings
    rules: tuple[str, ...]
    verdicts: dict[tuple[str, str, str], Any]
    candidates: list[Any]
    datasets: Mapping[str, Any]
    retirements: Mapping[tuple[str, str, str], Any]
    adjudications: Mapping[tuple[str, str], Mapping[str, Any]]
    cell_guesses: list[dict[str, Any]] | None


def _cell_context(*, outcome: OutcomeDef, settings: StatsSettings, verdicts: Sequence[Any],
                  candidates: Sequence[Any], datasets: Mapping[str, Any] | None,
                  retirements: Mapping[tuple[str, str, str], Any] | None,
                  adjudications: Mapping[tuple[str, str], Mapping[str, Any]] | None,
                  cell_guesses: list[dict[str, Any]] | None) -> "_CellContext | None":
    """The tier's context, or None when it cannot run (which is byte-identical DECISION A)."""
    rules = tuple(name for name in ANSWER_RULES if name in (settings.best_guess_rules or []))
    if not rules or not verdicts:
        return None
    return _CellContext(
        outcome=outcome, settings=settings, rules=rules,
        verdicts={(v.dataset_id, v.outcome_key, v.group): v for v in verdicts
                  if v.group in ("A", "B")},
        candidates=list(candidates), datasets=dict(datasets or {}),
        retirements=dict(retirements or {}), adjudications=dict(adjudications or {}),
        cell_guesses=cell_guesses)


# ----------------------------------------------------------- what stands on a cell, post-log
def _retirement(ctx: _CellContext, dataset_id: str, outcome_key: str, group: str) -> Any:
    return ctx.retirements.get((dataset_id, outcome_key, group))


def _retired_codes(ret: Any) -> set[str]:
    return set(getattr(ret, "retired_codes", ()) or ())


def _overruled_names(ret: Any) -> set[str]:
    return set(getattr(ret, "overruled", ()) or ()) | set(getattr(ret, "auto_stale", ()) or ())


def _standing_flags(verdict: Any, ret: Any) -> list[Any]:
    """The verdict's flag INSTANCES that the log has not retired (G6 step 0)."""
    retired = _retired_codes(ret)
    return [flag for flag in verdict.flags if flag.code not in retired]


def _forcing(flag: Any) -> bool:
    """Is this instance one the answer tier must cross or refuse on (H6 errors, H7 codes)?

    Membership, not severity, is what makes a contradicting code H7; the F3 absolutes are not
    crossable AT ALL and are handled before any authority is looked for.
    """
    return (flag.code not in F3_ABSOLUTES
            and (flag.code in CONTRADICTED or flag.severity == "error"))


def _standing_refutations(verdict: Any, ret: Any) -> list[Any]:
    if "verifier_refuted" in _overruled_names(ret):
        return []
    return [v for v in verdict.verifiers if str(v.verdict) == "refuted"]


def _locator_head(ctx: _CellContext, candidate_id: str) -> str:
    for candidate in ctx.candidates:
        if candidate.candidate_id == candidate_id and candidate.locator:
            return str(candidate.locator)[:60]
    return ""


def _subject_pairs(ctx: _CellContext, ids: Sequence[str]) -> list[tuple[str, str]]:
    """`(candidate_id, locator_head)` — ids collide across readings (D2), the pair narrows it."""
    return [(cid, _locator_head(ctx, cid)) for cid in ids]


def _cell_candidates(ctx: _CellContext, dataset_id: str, outcome_key: str,
                     group: str) -> list[Any]:
    return [c for c in ctx.candidates if c.dataset_id == dataset_id
            and c.outcome_key == outcome_key and c.group == group]


def _found(candidates: Sequence[Any]) -> list[Any]:
    return [c for c in candidates if c.status == "found" and c.mean is not None]


def _text_basis_ids(cell: Sequence[Any]) -> set[str]:
    """The printed-text readings of a cell — the basis of a ruling that chose no candidate.

    An adjudication with empty `chosen_candidate_ids` read the paper's own text (Coudière:
    grounded on the Results sentence), so every finding targeting a DIGITIZE reading is disjoint
    from its basis, and every finding targeting a text reading is not.
    """
    return {c.candidate_id for c in cell if str(getattr(c, "route", "")) in ("text", "table")}


# --------------------------------------------------------------------- the crossing calculus
@dataclass
class _CellGuess:
    """What the tier decided about ONE cell: an entered value, a refusal, or F7."""

    group: str
    rule: str = ""                                   # set iff a value was entered
    refusal: str = ""                                # set iff refused; names the first H-hold
    answered: bool = False                           # F7: a human has acted on this cell
    mean: float | None = None
    mean_source: str = ""
    dispersion_value: float | None = None
    dispersion_type: str = ""
    disp_source: tuple[str, str] | None = None       # (candidate_id, locator_head)
    disp_siblings: list[str] = field(default_factory=list)
    disp_note: str = ""
    n: int | None = None
    n_source: str = ""
    stepped_past: list[dict[str, Any]] = field(default_factory=list)


def _cross_flags(ctx: _CellContext, verdict: Any, ret: Any, basis: set[str],
                 rule: str) -> tuple[list[dict[str, Any]], str]:
    """`(stepped_past entries, refusal)` for the cell's standing forcing FLAG instances.

    The subject test (§3): a finding whose `candidate_ids` intersect the authority's basis is
    self-undermining — uncrossable; empty ids are an unknowable subject — uncrossable (A2);
    everything else stands apart from the authority and is crossed, recorded. One failure is the
    whole answer: no guess, and the reason names the finding.
    """
    entries: dict[str, dict[str, Any]] = {}
    for flag in _standing_flags(verdict, ret):
        if not _forcing(flag):
            continue
        ids = [str(x) for x in flag.candidate_ids or []]
        if not ids:
            return [], (f"{flag.code} names no candidate — its subject is unknowable, and "
                        f"doubt goes against crossing")
        if set(ids) & basis:
            return [], (f"{flag.code} targets the authority's own basis — self-undermining, "
                        f"uncrossable")
        entry = entries.setdefault(flag.code, {
            "finding": flag.code, "severity": flag.severity, "subject": [],
            "basis": f"its subject is a reading the {rule.replace('_', ' ')} displaced"})
        for pair in _subject_pairs(ctx, ids):
            if pair not in entry["subject"]:
                entry["subject"].append(pair)
    return list(entries.values()), ""


def _cross_refutations(ctx: _CellContext, verdict: Any, ret: Any, basis: set[str],
                       entering_mean: float) -> tuple[list[dict[str, Any]], str]:
    """The refutation half of the subject test, A3's citation test folded in.

    Test 1: a refutation carries its target's `candidate_id` — intersect with the basis. Test 2,
    only when no target id is usable: the refutation's STRUCTURED alternative (`alt_mean` — the
    engine never parses quote prose) must match the entering value under
    `within_read_tolerance`, or the refutation is treated as disputing — uncrossable. Every
    failure mode of the test is 'disputes' (A3).
    """
    entry: dict[str, Any] | None = None
    for refutation in _standing_refutations(verdict, ret):
        target = str(refutation.candidate_id or "")
        if target:
            if target in basis:
                return [], ("verifier_refuted disputes the authority's own basis — uncrossable")
        elif refutation.alt_mean is None or not _same_reading(refutation.alt_mean,
                                                              entering_mean):
            return [], ("verifier_refuted with no attributable subject and no supporting "
                        "citation — treated as disputing the entering value, uncrossable")
        if entry is None:
            entry = {"finding": "verifier_refuted", "severity": "error", "subject": [],
                     "basis": "its citation is the value the authority entered; its subject "
                              "is the displaced reading"}
        for pair in _subject_pairs(ctx, [target] if target else []):
            if pair not in entry["subject"]:
                entry["subject"].append(pair)
    return ([entry] if entry is not None else []), ""


# ------------------------------------------------------------------------- the donor ladder
def _donor_se(value: float | None, dispersion_type: Any, n: int | None,
              settings: StatsSettings) -> float | None:
    """A donor reading's spread as an SE, or None when it has none (A6: ineligible).

    The conversions are the resolver's own (`sd_from_ci_halfwidth`, `_ci_dist`) — never a
    restatement; a dispersion-less or unlabelled reading returns None and fails the gate by
    construction, which is what keeps Kumar's −360.05 raster read out with no special case.
    """
    if value is None or n is None or n <= 1:
        return None
    kind = _dt(dispersion_type)
    if kind == "SE":
        return float(value)
    if kind == "SD":
        return float(value) / math.sqrt(n)
    if kind in _CI_LEVEL:
        return sd_from_ci_halfwidth(float(value), n, _CI_LEVEL[kind],
                                    dist=_ci_dist(n, settings)) / math.sqrt(n)
    return None


def _donor_key(donor: Any, entering_n: int, settings: StatsSettings) -> tuple:
    """The finalized deterministic tiebreak (§4.5, D2): conservative, then total.

    Converted SE at the ENTERING n, descending (borrowed variance may under-weight, never
    over-weight); then dispersion-type conservatism; then lexicographic `(candidate_id,
    locator)` as the final total order — deterministic on shared-id candidate groups.
    """
    se = _donor_se(donor.dispersion_value, donor.dispersion_type, entering_n, settings) or 0.0
    return (-se, _TYPE_CONSERVATISM.get(_dt(donor.dispersion_type), 9),
            str(donor.candidate_id), str(donor.locator))


def _eligible_donors(pool: Sequence[Any], entering_mean: float, entering_n: int,
                     settings: StatsSettings) -> list[Any]:
    """The z ≤ 1 consistency gate (§4.5), at every ladder step it touches.

    A reading more than one of its own standard errors from the settled value is evidence of a
    different quantity, not corroboration — it excludes Coudière B's wrong-series 3.75±0.2 with
    no prose-parsing of locators or flags. A donor with no convertible spread has no SE to be
    judged by and is ineligible (A6).
    """
    out = []
    for donor in pool:
        se = _donor_se(donor.dispersion_value, donor.dispersion_type,
                       donor.n if donor.n else entering_n, settings)
        if se is None:
            continue
        if abs(float(donor.mean) - entering_mean) <= se:
            out.append(donor)
    return out


def _donor_spread(ctx: _CellContext, verdict: Any, cell: Sequence[Any],
                  adjudication: Mapping[str, Any] | None, entering_mean: float,
                  entering_n: int) -> tuple[Any | None, list[Any], str]:
    """`(winning donor, its eligible set, refusal)` for a mean the authority left spread-less.

    The selection ladder (§4.5): (1) the ruling's own spread is the caller's (it sits on the
    verdict); (2) the reading group the record DESIGNATES — `chosen_candidate_ids`, else the
    verdict's `agreeing_ids` — honoured as designated, reacquire ids included, gated and
    tie-broken; (3) the cell's eligible ensemble-path readings, with derived and second-pass
    instruments fenced out (A6: `vlm_coords`, `raster_cv`, `:reacquire` — a reacquired read
    exists because the first was disputed, and the largest-SE tiebreak would otherwise PREFER
    its wide bars); (4) nothing — F2. A6's sanity ceiling binds the winner to ≤ 2× the median
    eligible converted SE.
    """
    found = _found(cell)
    designated = list((adjudication or {}).get("chosen_candidate_ids") or []) \
        or [str(x) for x in verdict.agreeing_ids or []]
    step2 = [c for c in found if c.candidate_id in set(designated)]
    eligible = _eligible_donors(step2, entering_mean, entering_n, ctx.settings)
    if not eligible:
        step3 = [c for c in found
                 if ":ensemble" in c.candidate_id and ":reacquire" not in c.candidate_id
                 and "vlm_coords" not in c.candidate_id and "raster_cv" not in c.candidate_id]
        eligible = _eligible_donors(step3, entering_mean, entering_n, ctx.settings)
    if not eligible:
        return None, [], ("no eligible same-cell reading carries a spread consistent with the "
                          "entered mean (z ≤ 1) — the record holds no borrowable dispersion")
    ranked = sorted(eligible, key=lambda d: _donor_key(d, entering_n, ctx.settings))
    winner = ranked[0]
    ses = sorted(_donor_se(d.dispersion_value, d.dispersion_type, entering_n, ctx.settings)
                 or 0.0 for d in eligible)
    median = ses[len(ses) // 2] if len(ses) % 2 else (ses[len(ses) // 2 - 1]
                                                      + ses[len(ses) // 2]) / 2
    winner_se = _donor_se(winner.dispersion_value, winner.dispersion_type, entering_n,
                          ctx.settings) or 0.0
    if median and winner_se > 2 * median:
        return None, eligible, (
            f"the widest consistent reading ({winner.dispersion_value:g} "
            f"{_dt(winner.dispersion_type)} at {winner.candidate_id}) breaches the sanity "
            f"ceiling (> 2× the eligible set's median converted SE {median:g}) — the spread "
            f"slot is unanswerable")
    return winner, eligible, ""


def _unique_n(cell: Sequence[Any], verdict: Any) -> tuple[int | None, str]:
    """The n slot: the verdict's own, else the ONE n the cell's found readings agree on.

    No conservative direction exists for n (§4.5), so disagreement is unanswerable, not a pick.
    """
    if verdict.n:
        return int(verdict.n), "the record's own resolved n"
    sizes = {int(c.n) for c in _found(cell) if c.n}
    if len(sizes) == 1:
        return sizes.pop(), "unique across this cell's readings"
    if not sizes:
        return None, "no reading carries a group size"
    return None, f"the cell's readings disagree about n ({sorted(sizes)})"


# ------------------------------------------------------------------------ the three authorities
def _try_adjudicated(ctx: _CellContext, verdict: Any, cell: Sequence[Any], ret: Any,
                     adjudication: Mapping[str, Any]) -> _CellGuess:
    """(a) `adjudicated_value` — the adjudicator's settled ruling enters the cell's mean.

    The ruling's numbers are read off the LIVE verdict (never a stage file — supersede rule);
    the ruling's basis is `chosen_candidate_ids`, or — when it chose none — the printed text
    (Coudière: grounded on the Results sentence), so every digitize-targeted finding is
    disjoint. An entering mean answers H4, H5 and H9 wholesale; every other standing forcing
    finding must pass the subject test or the whole answer refuses.
    """
    group = str(verdict.group)
    entering = float(verdict.mean)
    basis = set(adjudication.get("chosen_candidate_ids") or []) or _text_basis_ids(cell)
    crossed, refusal = _cross_flags(ctx, verdict, ret, basis, "adjudicated_value")
    if refusal:
        return _CellGuess(group=group, refusal=refusal)
    crossed_refs, refusal = _cross_refutations(ctx, verdict, ret, basis, entering)
    if refusal:
        return _CellGuess(group=group, refusal=refusal)
    locator = str((adjudication.get("locators") or {}).get(group) or "")
    guess = _CellGuess(group=group, rule="adjudicated_value", mean=entering,
                       mean_source="adjudicated ruling"
                       + (f", {locator[:60]!r}" if locator else ""),
                       stepped_past=[*crossed, *crossed_refs])
    guess.n, guess.n_source = _unique_n(cell, verdict)
    if guess.n is None:
        guess.refusal, guess.rule = f"the n slot is unanswerable: {guess.n_source}", ""
        return guess
    if verdict.dispersion_value is not None:
        guess.dispersion_value = float(verdict.dispersion_value)
        guess.dispersion_type = _dt(verdict.dispersion_type) or "SE"
        guess.disp_source = ("", "the ruling's own dispersion")
    else:
        donor, eligible, refusal = _donor_spread(ctx, verdict, cell, adjudication,
                                                 entering, guess.n)
        if donor is None:
            guess.refusal, guess.rule = refusal, ""
            return guess
        guess.dispersion_value = float(donor.dispersion_value)
        guess.dispersion_type = _dt(donor.dispersion_type)
        guess.disp_source = (str(donor.candidate_id), str(donor.locator)[:60])
        guess.disp_siblings = [f"{d.dispersion_value:g}" for d in eligible if d is not donor]
        guess.disp_note = ("borrowed from a same-cell reading the record designates — "
                           "borrowing can under-weight this row AND shrink its effect "
                           "toward null")
    return guess


def _mapped_entry(candidate: Any, order: Sequence[str]) -> str:
    """§4.2's vocabulary mapping: candidate `route` literal × dispersion type → precedence entry.

    An unknown literal maps to no entry and can never win; a mean-only printed value's authority
    is its mean (its spread comes from the donor ladder), so it maps to the earliest `text_*`
    entry the protocol lists.
    """
    route = str(getattr(candidate, "route", "") or "")
    if route in ("table", "figure", "test_statistic", "p_value", "reported_d"):
        return route
    if route != "text":
        return ""
    kind = _dt(candidate.dispersion_type)
    if kind == "SD":
        return "text_mean_sd"
    if kind in ("SE", "CI95", "CI90"):
        return "text_mean_se_ci"
    return next((entry for entry in order if entry.startswith("text_")), "")


def _agree(values: Sequence[float], tolerance: float | None) -> bool:
    """Do these means agree among themselves — under the vote's own window, or, when the vote
    recorded none (`vote_tolerance: None`, Kumar's locator-conflict cells), under
    `within_read_tolerance` (A3)?"""
    if len(values) < 2:
        return True
    lo, hi = min(values), max(values)
    if tolerance is not None:
        return (hi - lo) <= float(tolerance)
    return _same_reading(lo, hi)


def _bounds_label(value: float, ci_low: float, ci_high: float, n: int,
                  settings: StatsSettings) -> tuple[str, str]:
    """A4's arithmetic label check: `("SE"|"CI95"|"", detail)` from the record's own bounds.

    The stated value matching the bounds' implied SE means the label is the SEM; matching the
    raw half-width means it is the CI. Both or neither ⇒ the check is inapplicable and decides
    nothing (doubt-against handles what follows). Structured `Candidate.ci_low/ci_high` only —
    prose is never parsed.
    """
    half = (float(ci_high) - float(ci_low)) / 2
    implied_se = sd_from_ci_halfwidth(half, n, 0.95, dist=_ci_dist(n, settings)) / math.sqrt(n)
    mult = half / implied_se if implied_se else float("inf")
    detail = (f"the quoted bounds [{ci_low:g}, {ci_high:g}] have half-width {half:.4g} "
              f"vs {value:g} × {mult:.4g} = {value * mult:.4g}")
    as_se = _same_reading(value, implied_se)
    as_ci = _same_reading(value, half)
    if as_se and not as_ci:
        return "SE", detail
    if as_ci and not as_se:
        return "CI95", detail
    return "", detail


def _try_route_precedence(ctx: _CellContext, verdict: Any, cell: Sequence[Any], ret: Any,
                          adjudication: Mapping[str, Any] | None) -> _CellGuess:
    """(b) `route_precedence` — the settings-ordered winner of a cross-route disagreement.

    Fires only on a real contest: found readings in ≥2 distinct precedence entries after §4.2's
    mapping, with the label-split collapse (N-c: identical mean and spread magnitude across text
    candidates is ONE reading whatever the labels — precedence never settles a label dispute).
    The winner's own candidates must agree on the mean; the label is checked against the
    record's own quoted bounds (A4) — arithmetic adjudicates a conflict and REFUTES a unanimous
    mislabel; an unsettled adjudication that concurs at the same value is crossed as
    `adjudicated_unsettled`, one that disputes it blocks.
    """
    group = str(verdict.group)
    order = list(ctx.settings.route_precedence)
    found = _found(cell)
    # N-c's collapse for the contest count: text readings with one (mean, spread) are one reading
    entries: set[str] = set()
    seen_text: set[tuple[float, float | None]] = set()
    for candidate in found:
        entry = _mapped_entry(candidate, order)
        if not entry:
            continue
        if str(getattr(candidate, "route", "")) == "text":
            key = (float(candidate.mean), None if candidate.dispersion_value is None
                   else float(candidate.dispersion_value))
            if key in seen_text:
                continue
            seen_text.add(key)
        entries.add(entry)
    if len(entries) < 2:
        return _CellGuess(group=group, refusal=(
            "the readings disagree within one route — there is no contest for precedence "
            "to win"))
    winner_entry = next((entry for entry in order if entry in entries), "")
    winners = [c for c in found if _mapped_entry(c, order) == winner_entry]
    winner_ids = {c.candidate_id for c in winners}
    means = [float(c.mean) for c in winners]
    if not _agree(means, verdict.vote_tolerance):
        return _CellGuess(group=group, refusal=(
            f"the winning route's own readings disagree about the mean "
            f"({', '.join(f'{m:g}' for m in sorted(set(means)))}) — a disagreement inside "
            f"the winner is not settled by precedence"))
    if len(set(means)) > 1:
        # within the window but not ONE reading: any pick would be arbitrary and an average
        # is a number the record holds nowhere (G2/F2 — "no authority can point at a value
        # the record does not hold"). Doubt goes against entering either (review F2).
        return _CellGuess(group=group, refusal=(
            f"the winning route's readings agree within the vote window but are not one "
            f"reading ({', '.join(f'{m:g}' for m in sorted(set(means)))}) — no single "
            f"record value is designated, and this line never enters a number the record "
            f"does not hold"))
    entering = means[0]
    crossed, refusal = _cross_flags(ctx, verdict, ret, winner_ids, "route_precedence")
    if refusal:
        return _CellGuess(group=group, refusal=refusal)
    crossed_refs, refusal = _cross_refutations(ctx, verdict, ret, winner_ids, entering)
    if refusal:
        return _CellGuess(group=group, refusal=refusal)
    # an unsettled adjudication either concurs at the same value — crossed, recorded — or it is
    # a standing H4 objection to the entering value and the answer refuses
    if adjudication is not None and adjudication.get("needs_human") \
            and "adjudicated" not in _overruled_names(ret):
        ruling_mean = verdict.mean if str(verdict.route) == "adjudicated" else None
        if ruling_mean is None or not _same_reading(float(ruling_mean), entering):
            return _CellGuess(group=group, refusal=(
                "the adjudicator looked at this cell and did not settle it at this value — "
                "an open adjudication that does not concur blocks the guess"))
        crossed.append({"finding": "adjudicated_unsettled", "severity": "warn",
                        "subject": [], "basis": "ruling concurs at the same value"})
    guess = _CellGuess(group=group, rule="route_precedence", mean=entering,
                       mean_source=(f"the {len(winners)} agreeing {winner_entry} reading(s), "
                                    f"{str(winners[0].locator)[:60]!r}"),
                       stepped_past=[*crossed, *crossed_refs])
    guess.n, guess.n_source = _unique_n(cell, verdict)
    if guess.n is None:
        guess.refusal, guess.rule = f"the n slot is unanswerable: {guess.n_source}", ""
        return guess
    spreads = sorted({float(c.dispersion_value) for c in winners
                      if c.dispersion_value is not None})
    labels = sorted({_dt(c.dispersion_type) for c in winners
                     if c.dispersion_value is not None})
    if len(spreads) > 1:
        return _CellGuess(group=group, refusal=(
            f"the winning route's readings disagree about the spread "
            f"({', '.join(f'{s:g}' for s in spreads)}) — the spread slot is unanswerable"))
    if spreads:
        value = spreads[0]
        bounds = next(((c.ci_low, c.ci_high) for c in winners
                       if c.ci_low is not None and c.ci_high is not None), None)
        if bounds is not None:
            implied, detail = _bounds_label(value, bounds[0], bounds[1], guess.n, ctx.settings)
            conflict_flag = next((f.code for f in _standing_flags(verdict, ret)
                                  if f.code == "dispersion_type_conflict"), "")
            if implied and len(labels) > 1:
                # arithmetic adjudicates the label CONFLICT (Kumar d2 A: CI95-vs-SE ⇒ SEM)
                guess.dispersion_type = implied
                guess.disp_note = (f"label conflict ({'/'.join(labels)}) adjudicated by the "
                                   f"record's own quoted bounds: {detail} ⇒ {implied}")
            elif implied and labels and labels[0] != implied:
                # arithmetic REFUTES a unanimous label: the rule must not enter a label no
                # reading carries — and must not outclaim the human (A4, Kumar d1 B)
                guess.refusal, guess.rule = (
                    f"the winner's unanimous dispersion label {labels[0]} is refuted by the "
                    f"record's own bounds arithmetic ({detail} ⇒ the value is the "
                    f"{'SEM' if implied == 'SE' else implied}), a label no reading carries"
                    + (f"; the standing {conflict_flag} flag was already whispering it"
                       if conflict_flag else ""), "")
                return guess
            elif not implied and len(labels) > 1:
                guess.refusal, guess.rule = (
                    f"the winner's readings conflict about the dispersion label "
                    f"({'/'.join(labels)}) and the quoted bounds cannot adjudicate it "
                    f"({detail}) — doubt goes against entering either", "")
                return guess
            else:
                guess.dispersion_type = labels[0]
        elif len(labels) > 1:
            guess.refusal, guess.rule = (
                f"the winner's readings conflict about the dispersion label "
                f"({'/'.join(labels)}) with no quoted bounds to adjudicate it — doubt goes "
                f"against entering either", "")
            return guess
        else:
            guess.dispersion_type = labels[0]
        guess.dispersion_value = value
        guess.disp_source = (str(winners[0].candidate_id), str(winners[0].locator)[:60])
    else:
        donor, eligible, refusal = _donor_spread(ctx, verdict, cell, adjudication,
                                                 entering, guess.n)
        if donor is None:
            guess.refusal, guess.rule = refusal, ""
            return guess
        guess.dispersion_value = float(donor.dispersion_value)
        guess.dispersion_type = _dt(donor.dispersion_type)
        guess.disp_source = (str(donor.candidate_id), str(donor.locator)[:60])
        guess.disp_siblings = [f"{d.dispersion_value:g}" for d in eligible if d is not donor]
        guess.disp_note = ("borrowed from a same-cell reading the record designates — "
                           "borrowing can under-weight this row AND shrink its effect "
                           "toward null")
    return guess


def _families_in_window(cell: Sequence[Any], verdict: Any) -> set[str]:
    """(c)'s family test (M1, decided): the `model` field of a per-model reading.

    `vlm_coords`, `raster_cv` and `ensemble` candidates are instruments, not families, and an
    instrument's `pixel_provenance` is its own self-report — a score bonus may lean on it; an
    AUTHORITY that crosses objections may not. The window is the verdict's own tolerance
    (`within_read_tolerance` when the vote recorded none), around the standing mean.
    """
    if verdict.mean is None:
        return set()
    families: set[str] = set()
    for candidate in _found(cell):
        cid = str(candidate.candidate_id)
        if any(tag in cid for tag in (":ensemble", "vlm_coords", "raster_cv")):
            continue
        window = verdict.vote_tolerance
        close = (abs(float(candidate.mean) - float(verdict.mean)) <= float(window)
                 if window is not None
                 else _same_reading(float(candidate.mean), float(verdict.mean)))
        if not close:
            continue
        family = str(getattr(candidate, "model", "") or "")
        if not family:
            tail = cid.split(":digitize:readout:", 1)
            family = tail[1].split(":", 1)[0] if len(tail) == 2 else ""
        if family:
            families.add(family)
    return families


def _solo_read_blockers(verdict: Any, ret: Any,
                        adjudication: Mapping[str, Any] | None) -> str:
    """Why (c) may NOT stand behind this cell's own value — the tightest boundary of the three.

    Every CONTRADICTING code is immune to agreement by the record's own sentence ("no amount of
    agreement about a number establishes that it is the right number"); `calibration_disputed`
    attacks the shared instrument both families read against; a refutation is a named objection
    to the designated value itself; and an unsettled adjudication on the cell is treated as
    disputing — every refusal in this corpus disputes the reads, and no structural test can
    tell a reporting quibble from one, so doubt goes against crossing.
    """
    for flag in _standing_flags(verdict, ret):
        if flag.code in CONTRADICTED and flag.code not in F3_ABSOLUTES:
            return f"{flag.code} stands on this cell — agreement never crosses a contradiction"
        if flag.code == "calibration_disputed":
            return ("calibration_disputed stands — the instrument both families read against "
                    "is itself in dispute")
    if _standing_refutations(verdict, ret):
        return "a standing refutation names this very value — never crossable by agreement"
    if adjudication is not None and adjudication.get("needs_human") \
            and "adjudicated" not in _overruled_names(ret):
        return ("an open adjudication stands on this cell — treated as disputing the read "
                "(doubt goes against crossing)")
    return ""


def _try_agreed_solo(ctx: _CellContext, verdict: Any, cell: Sequence[Any], ret: Any,
                     adjudication: Mapping[str, Any] | None) -> _CellGuess:
    """(c) `agreed_solo_read` — ≥2 independent model families behind the standing value (H9)."""
    group = str(verdict.group)
    blocker = _solo_read_blockers(verdict, ret, adjudication)
    if blocker:
        return _CellGuess(group=group, refusal=blocker)
    families = _families_in_window(cell, verdict)
    if len(families) < 2:
        return _CellGuess(group=group, refusal=(
            f"only {len(families)} found model famil{'y' if len(families) == 1 else 'ies'} "
            f"read this value within the vote window — agreement this thin answers nothing"))
    guess = _CellGuess(group=group, rule="agreed_solo_read", mean=float(verdict.mean),
                       mean_source=(f"the standing value, corroborated by "
                                    f"{len(families)} model families "
                                    f"({', '.join(sorted(families))})"))
    guess.dispersion_value = (None if verdict.dispersion_value is None
                              else float(verdict.dispersion_value))
    guess.dispersion_type = _dt(verdict.dispersion_type)
    guess.n = verdict.n
    guess.n_source = "the record's own resolved n"
    return guess


# ------------------------------------------------------------- the per-cell stacking (§4 v2)
def _no_authority_reason(verdict: Any, ret: Any, adjudication: Mapping[str, Any] | None) -> str:
    """The refusal for a cell no authority answers — the FIRST standing hold in H-table order.

    H1 (a refused row) and H3 (an unsigned one) never reach here: the row-level pre-checks fall
    straight through to DECISION A's own vetoes. The order below is the H-table's, so the tests
    can assert reasons without knowing which check happened to run first (reviewer S4).
    """
    if verdict.mean is None:
        return ("no resolved value stands on this cell and no authority in the record "
                "supplies one")
    if adjudication is not None and adjudication.get("needs_human") \
            and "adjudicated" not in _overruled_names(ret):
        return ("the adjudicator was convened on this cell and refused to settle it — "
                "an open adjudication is not an authority")
    if verdict.agreement == "disagree":
        return "the readings disagree and no cross-route precedence winner is clean"
    for severity_first in (True, False):
        for flag in _standing_flags(verdict, ret):
            if flag.code in F3_ABSOLUTES or not _forcing(flag):
                continue
            if (flag.severity == "error") is severity_first:
                return f"{flag.code} stands on this cell and no rule may cross it"
    if _standing_refutations(verdict, ret):
        return "a standing refutation names this cell's value"
    return "held for corroboration no rule supplies"


def _answer_cell(ctx: _CellContext, verdict: Any) -> _CellGuess:
    """One cell through stacking v2: F7, F3, then the authorities in match order."""
    group = str(verdict.group)
    ret = _retirement(ctx, verdict.dataset_id, verdict.outcome_key, group)
    if ret is not None and getattr(ret, "answered", False):
        return _CellGuess(group=group, answered=True)       # G7: permanent, every rule
    if not verdict.needs_human:
        # G1: only open questions are ever guessed at. A cell the verification layer
        # RELEASED has no question, and a guess about it would be an answer to nothing —
        # its values are consumed as they stand (Coudière d2 aftereffect's shape).
        return _CellGuess(group=group, refusal="the cell is not held — nothing is open here")
    cell = _cell_candidates(ctx, verdict.dataset_id, verdict.outcome_key, group)
    if verdict.higher_is_better is None:
        return _CellGuess(group=group, refusal=(
            "which direction counts as better was never settled — this line never resolves "
            "a direction it does not have"))
    for flag in _standing_flags(verdict, ret):
        if flag.code in F3_ABSOLUTES:
            return _CellGuess(group=group, refusal=(
                f"{flag.code} stands on this cell — a disputed direction is beyond every "
                f"rule (F3)"))
    adjudication = ctx.adjudications.get((verdict.dataset_id, verdict.outcome_key))
    settled = (adjudication is not None and not adjudication.get("needs_human")
               and "adjudicated" not in _overruled_names(ret))
    if settled and str(verdict.route) == "adjudicated" and verdict.mean is not None:
        if "adjudicated_value" in ctx.rules:
            return _try_adjudicated(ctx, verdict, cell, ret, adjudication)
        return _CellGuess(group=group, refusal=(
            "a settled adjudication owns this cell and adjudicated_value is not enabled"))
    if verdict.agreement == "disagree" and not settled and "route_precedence" in ctx.rules:
        return _try_route_precedence(ctx, verdict, cell, ret, adjudication)
    if verdict.agreement in ("agree", "single") and verdict.mean is not None \
            and not settled and "agreed_solo_read" in ctx.rules:
        return _try_agreed_solo(ctx, verdict, cell, ret, adjudication)
    return _CellGuess(group=group,
                      refusal=_no_authority_reason(verdict, ret, adjudication))


def _slot_entries(guess: _CellGuess) -> dict[str, Any]:
    """The per-slot `entered` record for one guessed cell — every value with its named source."""
    out: dict[str, Any] = {"mean": {"value": guess.mean, "source": guess.mean_source}}
    if guess.dispersion_value is not None:
        out["dispersion"] = {"value": guess.dispersion_value, "type": guess.dispersion_type,
                             "source": None if guess.disp_source is None else
                             {"candidate_id": guess.disp_source[0],
                              "locator": guess.disp_source[1]}}
        if guess.disp_siblings:
            out["dispersion"]["passed_over"] = list(guess.disp_siblings)
        if guess.disp_note:
            out["dispersion"]["note"] = guess.disp_note
    if guess.n is not None:
        out["n"] = {"value": guess.n, "source": guess.n_source}
    return out


_MEAN_HOLD = {"adjudicated_value": "adjudicated", "route_precedence": "disagree",
              "agreed_solo_read": "low_score"}


def _holds_answered(guess: _CellGuess) -> list[dict[str, Any]]:
    entries = [{"slot": "mean", "group": guess.group, "hold": _MEAN_HOLD[guess.rule],
                "rule": guess.rule, "value": guess.mean, "source": guess.mean_source,
                "siblings": []}]
    if guess.dispersion_value is not None and guess.disp_source is not None:
        entries.append({"slot": "dispersion", "group": guess.group,
                        "hold": "dispersion_missing", "rule": guess.rule,
                        "value": guess.dispersion_value,
                        "source": {"candidate_id": guess.disp_source[0],
                                   "locator": guess.disp_source[1]},
                        "siblings": list(guess.disp_siblings)})
    if guess.n is not None:
        entries.append({"slot": "n", "group": guess.group, "hold": "n_missing",
                        "rule": guess.rule, "value": guess.n, "source": guess.n_source,
                        "siblings": []})
    return entries


def _entered_summary(entered: Mapping[str, Mapping[str, Any]]) -> str:
    """One line per guessed cell for the queue's `best_guess_entered` column (A7)."""
    parts = []
    for group in sorted(entered):
        slots = entered[group]
        mean = slots.get("mean") or {}
        piece = f"{group} {mean.get('value'):g}" if mean.get("value") is not None else group
        spread = slots.get("dispersion") or {}
        if spread.get("value") is not None:
            piece += f" ± {spread['value']:g} {spread.get('type') or ''}".rstrip()
        n = slots.get("n") or {}
        if n.get("value") is not None:
            piece += f" n {n['value']}"
        parts.append(piece)
    return " · ".join(parts)


def _guess_clause(guess: _CellGuess) -> str:
    """One cell's slice of the row reason: every entered slot with its named source (§7)."""
    clause = f"{guess.group}: mean {guess.mean:g} ({guess.mean_source})"
    if guess.dispersion_value is not None:
        clause += f" · spread {guess.dispersion_value:g} {guess.dispersion_type}"
        if guess.disp_source is not None and guess.disp_source[0]:
            clause += f" from ({guess.disp_source[0]}, {guess.disp_source[1]!r})"
        if guess.disp_siblings:
            clause += f" passing over {'/'.join(guess.disp_siblings)}"
        if guess.disp_note:
            clause += f" — {guess.disp_note}"
    if guess.n is not None:
        clause += f" · n {guess.n} ({guess.n_source})"
    return clause


def _landed_clause(group: str, ret: Any) -> str:
    landed = getattr(ret, "landed", None) or {}
    piece = f"{group}: the human's own answer stands"
    if landed.get("mean") is not None:
        piece += f" (mean {landed['mean']:g}"
        if landed.get("dispersion_value") is not None:
            piece += f" ± {landed['dispersion_value']:g} {landed.get('dispersion_type') or ''}".rstrip()
        piece += f", override seq {landed.get('seq', '?')})"
    return piece


def _record_cell_guesses(ctx: _CellContext, record: EffectSizeRecord,
                         guesses: Mapping[str, _CellGuess], blocked_by: str) -> None:
    """A fire with no completable row is still first-class (§4 stacking): it reaches the
    payload's `cell_guesses` and the question card, and nowhere else."""
    if ctx.cell_guesses is None:
        return
    for group in sorted(guesses):
        guess = guesses[group]
        ctx.cell_guesses.append({
            "dataset_id": record.dataset_id, "outcome_key": record.outcome_key,
            "group": group, "rule": guess.rule, "label": _label(record),
            "entered": _slot_entries(guess), "stepped_past": list(guess.stepped_past),
            "answered_holds": _holds_answered(guess), "row_blocked_by": blocked_by})


def _mask_pair(ctx: _CellContext, record: EffectSizeRecord, copies: Mapping[str, Any],
               guesses: Mapping[str, _CellGuess]) -> dict[str, list[dict[str, Any]]]:
    """A5's instance-scoped masking, applied at synthesis time on the verdict COPIES.

    A finding INSTANCE is dropped iff every cell it attaches to (through its `candidate_ids`'
    reading groups) either CROSSED it — it is in that cell's `stepped_past` — or is
    HUMAN-ANSWERED (the answer stands in for the judgement), or has retired its code through
    the log. Never code-string masking across cells: an instance of the same code attached to
    a cell that neither crossed it nor was answered survives onto the built row and vetoes it
    exactly as today (the reviewer's leak fixture). An instance with empty `candidate_ids`
    attaches to no cell and is neither crossable (A2) nor maskable — it survives too. The
    drops happen on engine-internal copies only; live verdicts, stage files, the queue, the
    cards and the score all still carry every instance (G5).
    """
    group_of = {c.candidate_id: c.group for c in ctx.candidates
                if c.dataset_id == record.dataset_id
                and c.outcome_key == record.outcome_key and c.group in ("A", "B")}
    crossed = {group: {entry["finding"] for entry in guess.stepped_past}
               for group, guess in guesses.items()}
    answered = {group for group, guess in guesses.items() if guess.answered}
    retired = {group: _retired_codes(_retirement(ctx, record.dataset_id, record.outcome_key,
                                                 group)) for group in ("A", "B")}
    masked: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for copy_ in copies.values():
        kept = []
        for flag in copy_.flags:
            ids = tuple(str(x) for x in flag.candidate_ids or [])
            cells = {group_of.get(cid) for cid in ids} - {None}
            if not _forcing(flag) or not ids or not cells:
                kept.append(flag)                          # A2: no subject, no mask
                continue
            if not all(cell in answered or flag.code in crossed.get(cell, set())
                       or flag.code in retired.get(cell, set()) for cell in cells):
                kept.append(flag)                          # the leak fixture's veto survives
                continue
            for cell in cells:
                if cell in answered and flag.code not in retired.get(cell, set()) \
                        and (flag.code, ids, cell) not in seen:
                    seen.add((flag.code, ids, cell))
                    masked.setdefault(cell, []).append({
                        "finding": flag.code, "severity": flag.severity,
                        "subject": _subject_pairs(ctx, list(ids)), "cell": cell,
                        "note": "standing on a human-answered cell — the answer stands in "
                                "for the judgement"})
        copy_.flags = kept
    return masked


def _cell_decide(record: EffectSizeRecord, ctx: _CellContext) -> BestGuessDecision | None:
    """The answer tier's whole attempt on one held row, or None for DECISION A's own path.

    None is byte-identity: every reason the classic `_decide` writes is written by it, exactly
    as today. The tier only speaks when it ADMITS a row it built — and a fire that cannot
    complete a row says so in `cell_guesses` and then steps aside.
    """
    if _has_value(record):
        return None                                  # rename territory — handled after _decide
    if set(record.flags) & VETO_ROW_FLAGS or SIGN_DISPUTED in record.flags \
            or record.higher_is_better is None:
        return None                                  # H1/H3/F3: DECISION A's own vetoes stand
    verdict_a = ctx.verdicts.get((record.dataset_id, record.outcome_key, "A"))
    verdict_b = ctx.verdicts.get((record.dataset_id, record.outcome_key, "B"))
    if verdict_a is None or verdict_b is None:
        return None
    answers = {"A": _answer_cell(ctx, verdict_a), "B": _answer_cell(ctx, verdict_b)}
    guessed = {group: guess for group, guess in answers.items() if guess.rule}
    if not guessed:
        return None

    def consumable(verdict: Any, guess: _CellGuess) -> bool:
        # healthy, human-landed, or guessed (§4 stacking). HEALTHY means the verification
        # layer RELEASED the cell — a needs_human cell whose guess was refused is a cell with
        # a standing objection nobody crossed, and consuming its values "as they stand" would
        # smuggle a refused cell into the line under the sibling's rule name.
        return bool(guess.rule) or guess.answered or (
            not verdict.needs_human and verdict.mean is not None
            and verdict.dispersion_value is not None and bool(verdict.n))

    blockers = [f"cell {group} has no authority ({answers[group].refusal})"
                for group, verdict in (("A", verdict_a), ("B", verdict_b))
                if not consumable(verdict, answers[group])]
    dataset = ctx.datasets.get(record.dataset_id)
    if not blockers and dataset is None:
        blockers = ["no dataset spec was threaded for this row"]
    if not blockers and getattr(dataset, "shared_control", False):
        # the guessed row would be prepared alone and silently un-split the shared control
        # arm (the H1/H2 trap `_rebuild_row`'s docstring documents) — refused, never risked
        blockers = ["this row shares a control arm; a guess prepared outside its cluster "
                    "would un-split it"]
    if blockers:
        _record_cell_guesses(ctx, record, guessed, "; ".join(blockers))
        return None

    from .resolve import resolve_effect_with_fallback
    from .rows import prepare_rows

    copies = {"A": verdict_a.model_copy(deep=True), "B": verdict_b.model_copy(deep=True)}
    for group, guess in guessed.items():
        copy_ = copies[group]
        copy_.mean = guess.mean
        copy_.dispersion_value = guess.dispersion_value
        try:
            copy_.dispersion_type = DispersionType(guess.dispersion_type or "UNKNOWN")
        except ValueError:                                 # pragma: no cover - defensive
            copy_.dispersion_type = DispersionType.UNKNOWN
        if guess.n is not None:
            copy_.n = guess.n
        copy_.ci_low = copy_.ci_high = None
    masked = _mask_pair(ctx, record, copies, answers)
    prepared = prepare_rows([(dataset, record.outcome_key, copies["A"], copies["B"])],
                            ctx.candidates, ctx.settings)
    # alternatives deliberately EMPTY: the authority chose these values, and a same-locator
    # fallback pair would enter numbers no rule named (the `keep_printed` precedent)
    built = resolve_effect_with_fallback(dataset, ctx.outcome, prepared[0].values, [],
                                         ctx.settings)
    for name in ("paper_id", "cluster_id", "sample_id", "citation", "label", "moderators",
                 "analysis_metric", "notes"):
        setattr(built, name, getattr(record, name))
    built.orientation_source = prepared[0].orientation_source or record.orientation_source
    vetoed = _veto(built)
    if vetoed is not None:
        veto, reason, _ = vetoed
        _record_cell_guesses(ctx, record, guessed, f"the built row is vetoed — {veto}: "
                                                   f"{reason[:200]}")
        return None
    if not _has_value(built):
        detail = built.not_convertible_reason or "; ".join(built.routes_rejected.values())
        _record_cell_guesses(ctx, record, guessed,
                             f"the resolver built no effect size from the entered values"
                             + (f": {detail[:200]}" if detail else ""))
        return None

    rule = ";".join(name for name in ANSWER_RULES
                    if any(g.rule == name for g in guessed.values()))
    clauses = []
    for group in ("A", "B"):
        guess = answers[group]
        if guess.rule:
            clauses.append(_guess_clause(guess))
        elif guess.answered:
            clauses.append(_landed_clause(
                group, _retirement(ctx, record.dataset_id, record.outcome_key, group)))
    stepped = {group: list(guess.stepped_past) for group, guess in guessed.items()
               if guess.stepped_past}
    codes = sorted({entry["finding"] for entries in stepped.values() for entry in entries})
    reason = "answers this row's open questions by rule — " + " | ".join(clauses)
    if codes:
        reason += (f" — stepped past: {', '.join(codes)} — these findings still stand")
    for group, entries in sorted(masked.items()):
        names = sorted({entry["finding"] for entry in entries})
        if names:
            reason += (f" — masked on the answered cell {group}: {', '.join(names)} — the "
                       f"answer stands in for the judgement")
    reason += ". The question is still open; a human answer replaces this guess on the next repool."
    evidence = _evidence(record, entered={g: _slot_entries(answers[g]) for g in guessed},
                         route_built=built.route)
    return BestGuessDecision(
        dataset_id=record.dataset_id, outcome_key=record.outcome_key, label=_label(record),
        admitted=True, rule=rule, reason=reason, evidence=evidence,
        es=built.es, var=built.var, stepped_past=stepped, masked_on_answered=masked,
        answered_holds=[e for g in sorted(guessed) for e in _holds_answered(answers[g])],
        entered={g: _slot_entries(answers[g]) for g in sorted(guessed)}, row=built)


def _rename_agreed_solo(record: EffectSizeRecord, decision: BestGuessDecision,
                        ctx: _CellContext) -> BestGuessDecision:
    """(c) as row-admission RENAME: a `low_confidence_value` row whose two cells' values are
    each corroborated by ≥2 independent model families enters under the corroboration's own
    name. The value and weight are untouched; only the stated ground changes — and the D1
    boundary holds it to cells with NO contradicting code, refutation, calibration dispute or
    open adjudication ("agreement about a number never establishes that it is the right
    number" — the record's own sentence)."""
    if not decision.admitted or decision.rule != "low_confidence_value" \
            or "agreed_solo_read" not in ctx.rules:
        return decision
    families: list[str] = []
    for group in ("A", "B"):
        verdict = ctx.verdicts.get((record.dataset_id, record.outcome_key, group))
        if verdict is None or verdict.mean is None \
                or verdict.agreement not in ("agree", "single"):
            return decision
        ret = _retirement(ctx, record.dataset_id, record.outcome_key, group)
        if ret is not None and getattr(ret, "answered", False):
            return decision                                # G7: nothing is guessed there
        adjudication = ctx.adjudications.get((record.dataset_id, record.outcome_key))
        if _solo_read_blockers(verdict, ret, adjudication):
            return decision
        cell = _cell_candidates(ctx, verdict.dataset_id, verdict.outcome_key, group)
        window = _families_in_window(cell, verdict)
        if len(window) < 2:
            return decision
        families.append(f"{group}: {', '.join(sorted(window))}")
    from dataclasses import replace

    return replace(decision, rule="agreed_solo_read",
                   reason=(f"held for review ({record.confidence}) but ≥2 independent model "
                           f"families stand behind each cell's own value "
                           f"({'; '.join(families)}) — entered at the row's own value, "
                           f"nothing recomputed. The question is still open; a human answer "
                           f"replaces this guess on the next repool."))


def _answer_tier_extras(row: EffectSizeRecord,
                        decisions: Sequence[BestGuessDecision]) -> dict[str, Any]:
    """The DECISION B keys of one `added[]` payload entry — absent on every other row."""
    for decision in decisions:
        if decision.dataset_id == row.dataset_id and decision.admitted and decision.entered:
            return {"stepped_past": decision.stepped_past,
                    "masked_on_answered": decision.masked_on_answered,
                    "answered_holds": decision.answered_holds,
                    "entered": decision.entered}
    return {}


def decision_b_fired(decisions: Sequence[BestGuessDecision],
                     cell_guesses: Sequence[Mapping[str, Any]] = ()) -> bool:
    """Did the answer tier fire on this outcome — a rule-admitted row (rename included) or a
    cell-level fire? DECISION A's three rules do NOT trip it: they fire today and their
    surfaces are today's surfaces (integration §G)."""
    if cell_guesses:
        return True
    return any(decision.admitted
               and any(part in ANSWER_RULES for part in decision.rule.split(";"))
               for decision in decisions)
