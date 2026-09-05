"""DECISION B — the conclusion a run writes about itself, from its own numbers and no others.

This module is a template, not a generation. It makes no model call and imports no model client:
every sentence it returns is assembled from a `MetaResult`, the rows behind it, the best-guess
block and the protocol's OWN direction labels, and a test asserts the import graph as well as the
prose. A review's conclusion is the one paragraph a reader quotes, so it is the last place a
sentence should be produced by something that cannot be re-derived from the artefacts.

The rules (panel A B1, R1–R12) are hard and testable, and they are all refusals:

* never the word "significant" — an interval either excludes zero or includes it, and a p-value
  is only ever printed in the same sentence as the interval it belongs to;
* no magnitude adjective, because the protocol supplies no thresholds and the tool has no
  standing to invent one — the number is the finding;
* direction words come only from `positive_direction_label`/`negative_direction_label`, falling
  back to the two group labels when the protocol leaves them empty, and never from anywhere else;
* below k = 2 the outcome is "not estimable" and that is all that is said — no direction, no
  heterogeneity, no caveats, and nothing from the best-guess line beyond the row counts;
* heterogeneity is always paired with the prediction interval or with the sentence that says the
  interval is not estimable, and no `nan`/`inf` ever reaches the prose;
* the held-rows sentence is mandatory whenever a row is held, and it is quantitative;
* **every number in the prose is in `facts`** — the ledger the report and the SPA read, and the
  invariant a test enforces by tokenising the rendered text. Numbers inside a NAME the prose
  quotes (a method label, a study label) are registered from the name itself: a name is not a
  statistic, but it still puts digits on the page and the ledger has to account for them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..models import EffectSizeRecord, GroupDef, OutcomeDef, Protocol, StatsSettings
from ..stats.meta import MetaResult, prediction_interval
from .theme import ROUTE_GLYPHS, estimator_label, pi_label, route_glyph

__all__ = ["Conclusion", "VETO_PHRASES", "outcome_conclusion", "overall_conclusion",
           "render_text", "conclusion_payload", "conclusion_from_payload"]

#: what each best-guess veto is called in prose. The names come from `pipeline.bestguess.VETOES`;
#: an unknown one is spelled out rather than dropped, so a veto added there is still readable here
#: on the day it is added (and reads as its own name until someone writes it a phrase).
VETO_PHRASES: dict[str, str] = {
    "row_refusal": "the resolver refused the row",
    "contradicted_value": "the reading may be a different quantity",
    "orientation_unresolvable": "direction unresolved",
    "one_group_only": "only one group's statistics",
    "no_variance": "no usable value and variance",
}

_DIGITS = re.compile(r"[-+]?\d+\.\d+|\d+")


@dataclass(frozen=True)
class Conclusion:
    """One outcome's paragraph: the sentences, and the ledger every number in them came from."""

    outcome_key: str
    sentences: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    direction_word: str = ""
    estimable: bool = False
    #: the one sentence carrying the STRICT line's pooled number, and the sentences carrying the
    #: best-guess line's. Both are members of `sentences` — the paragraph is printed whole and
    #: this changes nothing about it — and they are named because a viewer that shows one line at
    #: a time cannot otherwise tell which of the numbers in the paragraph belongs to which line
    #: (review finding 15: the conclusion card claimed no state showed both, and showed both).
    headline: str = ""
    best_guess_sentences: list[str] = field(default_factory=list)


def render_text(c: Conclusion) -> str:
    return " ".join(c.sentences)


# ----------------------------------------------------------------------------- the ledger
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _number(value: float, digits: int = 2, sign: str = "") -> str:
    """A number as the prose prints it, with no `-0.00` in it.

    A value that rounds to zero at the printed precision IS zero on the page, and a minus sign in
    front of it reads as a direction — the one thing a reader takes away from a pooled estimate at
    a glance. `runs/nine` wrote "gives -0.00 (95% CI -0.33 to 0.33)" for an estimate of -0.0025
    (whole-branch review, MINOR 6). Only the printed form is normalised; the ledger keeps the
    number that was actually pooled.
    """
    text = f"{float(value):{sign}.{digits}f}"
    return f"{0.0:{sign}.{digits}f}" if float(text) == 0.0 else text


def _fact(facts: dict[str, Any], key: str, value: float, digits: int = 2,
          sign: str = "") -> str:
    """Record a number under its name and return it formatted exactly as the prose prints it.

    When the printed form is not what the stored number formats to — the negative zero above — the
    form itself is recorded under `<key>_text`, because the invariant this module is checked
    against is that every number in the prose is in `facts`.
    """
    facts[key] = float(value)
    text = _number(value, digits, sign)
    if text != f"{float(value):{sign}.{digits}f}":
        facts[f"{key}_text"] = text
    return text


def _register(facts: dict[str, Any], key: str, text: str) -> str:
    """Account for the digits inside a name (`t(k−2)`, `Heuer 2011`) the prose is about to quote."""
    for i, token in enumerate(_DIGITS.findall(str(text))):
        facts[f"{key}_digits_{i}"] = token
    return str(text)


def _name_list(names: Sequence[str]) -> str:
    names = [n for n in names if n]
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} or {names[-1]}"


# ----------------------------------------------------------------------------- the pieces
def _label(group: GroupDef | None, fallback: str) -> str:
    return ((group.label if group is not None else "") or fallback).strip()


def _direction_phrase(outcome: OutcomeDef, estimate: float, group_a: GroupDef | None,
                      group_b: GroupDef | None) -> str:
    """R3. The protocol's own word, or the two group labels — never an invented one."""
    a, b = _label(group_a, "group A"), _label(group_b, "group B")
    if estimate == 0:
        return f"no difference between {a} and {b}"
    label = (outcome.positive_direction_label if estimate > 0
             else outcome.negative_direction_label).strip()
    if label:
        return label.lower()                      # the protocol's label, mid-sentence
    # …and with no label, the CONSTRUCT, never the raw scores. "A scored higher than B" is a claim
    # about the numbers the paper printed, and the estimate is not one: it is orientation-applied,
    # so on a measure where a larger raw value means less of the construct (`higher_is_better =
    # false` — every error-type outcome) the sign has already been flipped, and the raw-score
    # sentence states the opposite of the finding (review finding 10). What a positive estimate
    # says is that group A shows MORE of the outcome, whatever direction its raw numbers run in.
    return (f"{'more' if estimate > 0 else 'less'} {outcome.label or outcome.key} "
            f"in {a} than in {b}")


def _veto_reasons(best_guess: Mapping[str, Any]) -> str:
    """The two commonest reasons a held row could not be given a value, most common first."""
    counts: dict[str, int] = {}
    for entry in best_guess.get("not_added") or ():
        name = str((entry or {}).get("veto") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    return "; ".join(VETO_PHRASES.get(name, name.replace("_", " ")) for name, _ in top)


def _crosses_zero(pooled: MetaResult, entry: Mapping[str, Any]) -> bool:
    """Did dropping this one dataset move the interval across zero (either way)?"""
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if not (_finite(low) and _finite(high)):
        return False
    return (pooled.ci_low <= 0 <= pooled.ci_high) is not (float(low) <= 0 <= float(high))


def _figure_rows(rows: Sequence[EffectSizeRecord]) -> int:
    return sum(1 for r in rows if route_glyph(r.route) == ROUTE_GLYPHS["figure"])


# ----------------------------------------------------------------------------- one outcome
def outcome_conclusion(outcome: OutcomeDef, settings: StatsSettings, *,
                       pooled: MetaResult | None = None,
                       rows: Sequence[EffectSizeRecord] = (),
                       held: Sequence[EffectSizeRecord] = (),
                       best_guess: Mapping[str, Any] | None = None,
                       loo: Sequence[Mapping[str, Any]] = (),
                       group_a: GroupDef | None = None,
                       group_b: GroupDef | None = None) -> Conclusion:
    """The paragraph for one outcome (template B2), and the ledger behind every number in it.

    `pooled` is the strict line — the primary analysis — and `rows` the rows it pooled; `held`
    are the rows kept out of it, `best_guess` the `best_guess` block of `pooled.json` and `loo`
    the leave-one-out table as data (`report.tables.leave_one_out_rows`). `group_a`/`group_b`
    only ever supply the fallback direction phrase, so a caller without a protocol still gets a
    sentence that says which way the estimate points, in the groups' own names.
    """
    rows, held, loo = list(rows), list(held), list(loo)
    bg: Mapping[str, Any] = dict(best_guess or {})
    facts: dict[str, Any] = {}
    sentences: list[str] = []
    caveats: list[str] = []

    label = (outcome.label or outcome.key).strip()
    facts["outcome_label"] = label
    _register(facts, "outcome_label", label)
    sentences.append(f"{label}.")

    k = int(pooled.k) if pooled is not None else len(rows)
    facts["k"] = k
    estimable = pooled is not None and k >= 2
    direction = ""
    headline, second_line = "", []                # named below when there is a line to name

    if not estimable:
        # R2: below two datasets there is no estimate, and an estimate is all the rest of the
        # paragraph is about. Everything after this point would be a statement about nothing.
        sentences.append(f"k = {k}: with fewer than two datasets no pooled estimate is defined, "
                         f"so this outcome is not estimable.")
    else:
        level_pct = float(settings.ci_level) * 100.0
        pct = _fact(facts, "ci_level_pct", level_pct, 0) + "%"
        k_papers = len({r.cluster_id or r.paper_id or r.dataset_id for r in rows}) or k
        facts["k_papers"] = k_papers
        direction = _direction_phrase(outcome, pooled.estimate, group_a, group_b)
        facts["direction"] = direction
        _register(facts, "direction", direction)
        estimator = _register(facts, "estimator", estimator_label(settings))
        headline = (
            f"Across k = {k} datasets from {k_papers} papers, the pooled {estimator} is "
            f"{_fact(facts, 'estimate', pooled.estimate)} "
            f"({pct} CI {_fact(facts, 'ci_low', pooled.ci_low)} to "
            f"{_fact(facts, 'ci_high', pooled.ci_high)}, "
            f"p = {_fact(facts, 'p', pooled.p, 3)}), i.e. {direction}; the interval "
            f"{'excludes' if not (pooled.ci_low <= 0 <= pooled.ci_high) else 'includes'} zero.")
        sentences.append(headline)

        # --- heterogeneity, always followed by the prediction interval or by its absence (R5)
        if _finite(pooled.tau2):
            inner = []
            if _finite(pooled.I2):
                inner.append(f"I² = {_fact(facts, 'i2', 100 * pooled.I2, 0)}%")
            if _finite(pooled.Q):
                if getattr(pooled, "robust", False):
                    # robumeta's df_Q is non-integer; truncating it would state a wrong df as fact
                    inner.append(f"Q({_fact(facts, 'q_df', pooled.Q_df, 2)}) = "
                                 f"{_fact(facts, 'q', pooled.Q)}")
                else:
                    facts["q_df"] = int(pooled.Q_df)
                    inner.append(f"Q({int(pooled.Q_df)}) = {_fact(facts, 'q', pooled.Q)}")
            if _finite(pooled.Q_p):
                inner.append(f"p = {_fact(facts, 'q_p', pooled.Q_p, 3)}")
            sentences.append(
                f"Between-dataset heterogeneity is τ² = {_fact(facts, 'tau2', pooled.tau2, 3)}"
                + (f" ({', '.join(inner)})." if inner else "."))

        try:
            pi_low, pi_high, pi_df = prediction_interval(pooled, str(settings.pi_method))
        except ValueError:                                 # pragma: no cover - guarded upstream
            pi_low = pi_high = float("nan")
            pi_df = 0
        method = _register(facts, "pi_label", pi_label(settings, pooled))
        if _finite(pi_low) and _finite(pi_high):
            # the z convention returns df = inf BY DESIGN (`prediction_interval`'s own contract):
            # finite bounds whose reference distribution has no df. int(inf) raised OverflowError
            # here and one unguarded conversion at the very end killed a whole 22-paper run after
            # every paper had resolved. An infinite df is recorded by its absence.
            if _finite(pi_df):
                facts["pi_df"] = int(pi_df)
            sentences.append(
                f"A dataset drawn from the same population would be expected to fall between "
                f"{_fact(facts, 'pi_low', pi_low)} and {_fact(facts, 'pi_high', pi_high)} "
                f"({method}).")
        else:
            # R12: `pi_low_used` really is nan at k = 2 under HTS. It says so in words.
            sentences.append(f"A prediction interval is not estimable at k = {k} under {method}.")

        # --- how the interval was built, when it was not the ordinary one (RVE)
        if getattr(pooled, "robust", False):
            sentence = (
                f"Standard errors are cluster-robust over m = "
                f"{_fact(facts, 'n_clusters', pooled.n_clusters, 0)} clusters (Satterthwaite "
                f"df = {_fact(facts, 'df_robust', pooled.df_robust, 2)}); rows from one paper "
                f"are treated as dependent (ρ = {_fact(facts, 'rho', pooled.rho, 2)}).")
            if getattr(pooled, "robust_small_sample", False):
                sentence += (f" With Satterthwaite df below "
                             f"{_fact(facts, 'df_trust_threshold', 4, 0)}, robumeta's own "
                             f"guidance is not to trust the result.")
            sentences.append(sentence)
        elif getattr(pooled, "robust_fallback", ""):
            fallback = _register(facts, "robust_fallback", pooled.robust_fallback)
            sentences.append(f"Cluster-robust standard errors were requested but could not be "
                             f"applied ({fallback}); the interval shown is the ordinary "
                             f"random-effects one.")

        # --- leave-one-out (R8)
        estimates = [float(e["estimate"]) for e in loo if _finite(e.get("estimate"))]
        if estimates:
            sentences.append(
                f"Leaving out one dataset at a time, the pooled estimate ranges from "
                f"{_fact(facts, 'loo_min', min(estimates))} to "
                f"{_fact(facts, 'loo_max', max(estimates))}.")
            crossing = [str(e.get("omitted_label") or e.get("omitted_dataset_id") or "")
                        for e in loo if _crosses_zero(pooled, e)]
            if any(crossing):
                names = _name_list(crossing)
                facts["loo_crossing"] = names
                _register(facts, "loo_crossing", names)
                sentences.append(f"Omitting {names} alone moves the interval across zero.")

    # --- the held rows (R7): mandatory, quantitative, and the same sentence either way
    n_held, n_rows = len(held), len(rows) + len(held)
    facts["n_held"], facts["n_rows"] = n_held, n_rows
    if n_held:
        sentences.append(f"{n_held} of {n_rows} rows for this outcome are held for human review.")
        second_line = _best_guess_sentences(facts, bg, estimable=estimable)
        sentences.extend(second_line)

    # --- caveats (R6, plus the figure share) — never at k < 2, where there is nothing to caveat
    if estimable:
        if k < 5:
            caveats.append(
                f"With k = {k}, τ² is estimated very imprecisely and both the interval and I² "
                f"should be read as indicative"
                + (", and the heterogeneity statistics are essentially uninterpretable."
                   if k < 3 else "."))
        n_figure = _figure_rows(rows)
        if n_figure * 2 > k:
            facts["n_figure_rows"] = n_figure
            caveats.append(f"{n_figure} of {k} pooled rows were read off a figure rather than "
                           f"from printed text.")
        sentences.extend(caveats)

    return Conclusion(outcome_key=outcome.key, sentences=sentences, facts=facts, caveats=caveats,
                      direction_word=direction, estimable=estimable, headline=headline,
                      best_guess_sentences=second_line)


def _best_guess_sentences(facts: dict[str, Any], bg: Mapping[str, Any], *,
                          estimable: bool) -> list[str]:
    """What the second line did with the held rows — counts always, numbers only if there is one.

    At k < 2 this is deliberately counts-only (R2): a delta from an estimate that does not exist
    is not a number, and the one thing a reader must not take away from an outcome that could not
    be pooled is a pooled-looking figure.
    """
    n_added = int(bg.get("n_added") or 0)
    n_still = int(bg.get("n_still_held") or 0)
    reasons = _veto_reasons(bg)
    facts["best_guess_n_added"] = n_added
    facts["best_guess_n_still_held"] = n_still
    if n_added == 0:
        return ["No held row could be given a value by any rule"
                + (f" ({reasons})." if reasons else ".")]

    tail = (f"; {n_still} row(s) could not be given a value by any rule"
            + (f" ({reasons})" if reasons else "")) if n_still else ""
    if not (estimable and _finite(bg.get("estimate")) and _finite(bg.get("ci_low"))
            and _finite(bg.get("ci_high"))):
        return [f"The best-guess line adds {n_added} of them{tail}."]

    facts["best_guess_k"] = int(bg.get("k") or 0)
    sentence = (
        f"The best-guess line adds {n_added} of them and gives "
        f"{_fact(facts, 'best_guess_estimate', bg['estimate'])} "
        f"({facts['ci_level_pct']:.0f}% CI {_fact(facts, 'best_guess_ci_low', bg['ci_low'])} to "
        f"{_fact(facts, 'best_guess_ci_high', bg['ci_high'])}, k = {facts['best_guess_k']})")
    if _finite(bg.get("delta_vs_strict")):
        delta = _number(bg["delta_vs_strict"], sign="+")
        facts["best_guess_delta"] = float(bg["delta_vs_strict"])
        facts["best_guess_delta_text"] = delta
        sentence += f", a change of {delta} from the primary estimate"
    sentences = [sentence + tail + "."]
    agrees = bg.get("sign_agrees_with_strict")
    if agrees is not None:
        facts["best_guess_sign_agrees_with_strict"] = "yes" if agrees else "no"
    if agrees is False:
        sentences.append("The best-guess line points the other way; read both.")
    return sentences


# ----------------------------------------------------------------------------- the whole run
def overall_conclusion(protocol: Protocol, per_outcome: Mapping[str, Conclusion]) -> str:
    """The paragraph above the per-outcome ones: what was poolable, and what is still held.

    Every number comes out of the per-outcome ledgers rather than being recomputed, so the
    overall paragraph cannot disagree with the paragraph directly beneath it.
    """
    items = list(per_outcome.items())
    lines = [f"{protocol.title}: {sum(1 for _, c in items if c.estimable)} of {len(items)} "
             f"outcomes could be pooled."]
    n_held = n_rows = n_flips = 0
    deltas: list[float] = []
    for key, c in items:
        facts = c.facts
        label = str(facts.get("outcome_label") or key)
        n_held += int(facts.get("n_held") or 0)
        n_rows += int(facts.get("n_rows") or 0)
        if _finite(facts.get("best_guess_delta")):
            deltas.append(float(facts["best_guess_delta"]))
        if str(facts.get("best_guess_sign_agrees_with_strict") or "") == "no":
            n_flips += 1
        if c.estimable:
            lines.append(f"{label}: {_number(facts['estimate'])} "
                         f"({float(facts['ci_level_pct']):.0f}% CI "
                         f"{_number(facts['ci_low'])} to {_number(facts['ci_high'])}), "
                         f"k = {facts['k']} — {c.direction_word}.")
        else:
            lines.append(f"{label}: not estimable (k = {facts.get('k', 0)}).")
    max_delta = max(deltas, key=abs) if deltas else 0.0
    lines.append(f"Across all outcomes, {n_held} of {n_rows} rows are held for human review; the "
                 f"best-guess line changes a pooled estimate by at most "
                 f"{_number(max_delta, sign='+')} and "
                 f"changes the direction of {n_flips} outcome(s). Every row above is traceable "
                 f"to a page or a figure in the extraction table, and nothing in this section "
                 f"goes beyond the numbers in it.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- pooled.json
def conclusion_payload(c: Conclusion) -> dict[str, Any]:
    """The `conclusion` block of `pooled.json` — sentences, ledger and the rendered text.

    `text` is written out so that a reader (or a viewer that only shows text nodes) never has to
    re-join the sentences and get a different string from the one the report printed.
    """
    return {"outcome_key": c.outcome_key, "sentences": list(c.sentences), "facts": dict(c.facts),
            "caveats": list(c.caveats), "direction_word": c.direction_word,
            "estimable": c.estimable, "headline": c.headline,
            "best_guess_sentences": list(c.best_guess_sentences), "text": render_text(c)}


def conclusion_from_payload(payload: Mapping[str, Any]) -> Conclusion:
    """The inverse, for the report: read what was written rather than deriving it a second time."""
    return Conclusion(outcome_key=str(payload.get("outcome_key") or ""),
                      sentences=[str(s) for s in payload.get("sentences") or ()],
                      facts=dict(payload.get("facts") or {}),
                      caveats=[str(s) for s in payload.get("caveats") or ()],
                      direction_word=str(payload.get("direction_word") or ""),
                      estimable=bool(payload.get("estimable")),
                      headline=str(payload.get("headline") or ""),
                      best_guess_sentences=[str(s) for s
                                            in payload.get("best_guess_sentences") or ()])
