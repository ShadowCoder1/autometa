"""Canopy data model (pydantic 2).

Everything here is plain-JSON serialisable: enums are string enums, no numpy/pathlib objects.
Rules that shaped these classes:
  * agents *locate and label*, they never compute — no `Candidate`/agent-facing model carries a
    derived statistic (d, g, pooled SD, ...). Only `EffectSizeRecord`, produced in code by
    `canopy.stats`, holds effect sizes. See `canopy.llm.schemas.assert_no_derived_stats`.
  * every scalar carries provenance (page, quote or crop+pixels, model, prompt version, call id).
  * every enum has an `unknown` / `UNKNOWN` member, because a value may simply be absent in a paper.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CanopyModel(BaseModel):
    """Strict base: unknown keys are an error (a typo in an agent's JSON must not pass silently)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), use_enum_values=False)


# ----------------------------------------------------------------------------- enums
class SourceKind(str, Enum):
    text_mean_sd = "text_mean_sd"
    text_mean_se = "text_mean_se"
    text_mean_ci = "text_mean_ci"
    table = "table"
    figure_bar = "figure_bar"
    figure_line = "figure_line"
    figure_points = "figure_points"
    figure_box = "figure_box"
    test_statistic = "test_statistic"
    reported_effect_size = "reported_effect_size"
    author_data = "author_data"
    unknown = "unknown"                 # a numeric source of a kind not listed above,
    #                                     e.g. fitted-model parameters — a human routes it


class DispersionType(str, Enum):
    SD = "SD"
    SE = "SE"
    CI95 = "CI95"
    CI90 = "CI90"
    IQR = "IQR"
    RANGE = "RANGE"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


#: `categorical` — the x axis is a set of conditions/targets/directions with no order that makes
#: "the last one" meaningful; `time` — trials, blocks, episodes; `other` — a continuous covariate.
XAxisKind = Literal["categorical", "time", "other", "unknown"]
GroupKey = Literal["A", "B"]
CandidateKind = Literal["group_stats", "test_statistic", "reported_d"]
CandidateStatus = Literal["found", "not_on_these_pages", "ambiguous"]
RawValueSemantics = Literal["higher_more_construct", "higher_more_error", "signed_direction", "unknown"]
AnalysisMetric = Literal["endpoint", "change_from_baseline", "baseline_corrected",
                         "percent_of_perturbation", "unknown"]
ErrorBarScope = Literal["between_subject", "within_subject_normalized", "unknown"]
ErrorBarAgreement = Literal["agreed", "conflict", "unconfirmed"]
#: what a source IS for the outcome it is listed under. `value` — the outcome's own number for
#: the groups can be read here; `baseline` — a pre-manipulation or control series that could
#: correct the value but is not the value (an aligned-cursor curve beside the rotated one, a
#: pre-test beside a post-test); `context` — a location that defines the window, names the
#: blocks, or reports a test with no group values; `alternate` — a location that measures the
#: SAME outcome by a second operationalization that lost the map-stage `which_measure` decision
#: (C6: one outcome carries one measure) — as does a location carrying NO metric at all when a
#: measure settlement withheld it, since "this location states no metric" is not evidence that it
#: states the winning one. An `alternate` is kept on the record with the quote that demoted it and
#: is never read for a value — it is not a `context` location, and nothing may treat "not `value`"
#: as "`context`". Only `value` sources are read for the number.
SourceRole = Literal["value", "baseline", "context", "alternate", "unknown"]
WhiskerDefinition = Literal["min_max", "iqr_1_5", "percentile_5_95", "sd", "se", "ci", "unknown"]
TestDesign = Literal["independent_t", "one_way_between", "mixed_main_effect", "interaction",
                     "ancova", "paired", "welch", "unknown"]
PKind = Literal["exact", "less_than", "greater_than", "ns", "unknown"]
#: WHAT a printed test statistic contrasts (P-B). Only `groups` — exactly group A versus group B
#: on this outcome — can stand in for the two group means. `against_constant` is a test of one
#: group against zero or any other fixed value (its df can equal n_a + n_b - 2 and it still is not
#: the contrast); `interaction` is a product term; `within` is a within-subject effect. `unknown`
#: means nobody recorded it, which is refused in code rather than assumed to be `groups`.
ContrastKind = Literal["groups", "against_constant", "interaction", "within", "unknown"]
Standardizer = Literal["pooled_sd_between", "dz_paired", "glass_delta", "partial_eta", "unknown"]
ReportedScale = Literal["cohens_d", "hedges_g", "glass_delta", "partial_eta_squared", "unknown"]
Direction = Literal["a_greater", "b_greater", "unknown"]
ExposureOrder = Literal["first", "repeated", "counterbalanced_collapsed", "unknown"]
ConfidenceBucket = Literal["auto_accept", "accept_with_note", "needs_human"]
Estimator = Literal["cohen", "hedges"]
VarianceMethod = Literal["borenstein", "hedges_olkin_df", "meta_exact", "meta_exact_g", "meta_hedges_approx"]
Tau2Method = Literal["REML", "DL", "PM"]
PIMethod = Literal["V", "HTS", "z"]


# ----------------------------------------------------------------------------- protocol
class GroupDef(CanopyModel):
    """One arm of the contrast, as the *user* defines it (nothing study-specific)."""

    key: GroupKey
    label: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)


class OutcomeDef(CanopyModel):
    key: str
    label: str
    definition: str
    measurement_window: str = ""
    higher_is_better_hint: str = ""
    positive_direction_label: str = ""      # e.g. "Enhanced in Old"
    negative_direction_label: str = ""      # e.g. "Reduced in Old"
    units_hint: str = ""


class StatsSettings(CanopyModel):
    """Statistical conventions. Field defaults follow the plan amendments (section A)."""

    profile: str = "metafor"
    estimator: Estimator = "cohen"
    variance: VarianceMethod = "borenstein"
    tau2_method: Tau2Method = "REML"
    hakn: bool = False
    pi_method: PIMethod = "V"
    route_precedence: list[str] = Field(default_factory=lambda: [
        "text_mean_sd", "table", "text_mean_se_ci", "figure", "test_statistic", "p_value", "reported_d"])
    late_window_sd: Literal["paper_reported_block", "block_closest_to_end", "mean_of_block_sd"] = "paper_reported_block"
    ci_to_sd_dist: Literal["auto", "z", "t"] = "auto"          # t(n-1) when n < 100
    primary_analysis_includes: list[ConfidenceBucket] = Field(
        default_factory=lambda: ["auto_accept", "accept_with_note"])
    shared_control_strategy: Literal["split_n", "combine_arms", "keep_first", "keep_all_flagged"] = "split_n"
    multi_group_policy: Literal["extremes", "combine_matching", "closest_to_definition", "needs_human"] = \
        "closest_to_definition"
    digitization_variance: Literal["off", "sensitivity", "primary"] = "sensitivity"
    one_row_per_paper: bool = True
    #: correlation assumed between two rows of one paper that came from the SAME participants,
    #: used by the within-paper composite (`canopy.pipeline.aggregate`, Borenstein ch. 24)
    within_paper_r: float = 0.5
    ci_level: float = 0.95
    #: forest-plot x axis, as `[low, high]`. None means the renderer picks a symmetric data-driven
    #: range; a profile pins it so two runs of the same review are comparable by eye — a reader
    #: who compares forests across runs is comparing arrow lengths, and an axis that rescales
    #: itself to whatever survived the run makes a shrunken effect look unchanged.
    forest_xlim: list[float] | None = None
    #: decimals the forest prints for the effect and its interval. The tables keep full precision:
    #: this is how the plot READS, not how anything was computed.
    forest_digits: int = 1


class DigitizeSettings(CanopyModel):
    """How hard the figure digitizer works — the knob that decides most of a run's cost.

    A figure is read by `readouts_min` vision passes; the further passes up to `readouts_max` are
    bought only when those disagree about a mean, because a third vote that confirms two agreeing
    ones changes nothing and a figure read-out is the most expensive call in the pipeline
    (task 15 §A3: the read-outs were ~60 % of the first live run's spend).
    """

    readouts_min: int = 2
    readouts_max: int = 3
    #: when to spend the overlay-verification call: on every figure, only when the routes disagree
    #: or one was already dropped, or never
    overlay_verify: Literal["always", "on_disagreement", "never"] = "on_disagreement"
    #: zoom/inspect tool calls one read-out may make before it must answer
    max_tool_calls: int = 6
    #: read EVERY point of a series whose x axis the mapper called `categorical` and average them
    #: in code (task 16 P6). Off by default: the dispersion it produces is an approximation the
    #: paper never stated, so a review turns it on deliberately or gets `not_convertible` instead
    #: of a number read at one arbitrary point of the axis.
    collapse_across_categorical_x: bool = False


class Protocol(CanopyModel):
    title: str
    research_question: str = ""
    group_a: GroupDef
    group_b: GroupDef
    outcomes: list[OutcomeDef]
    eligibility: list[str] = Field(default_factory=list)
    dataset_rules: list[str] = Field(default_factory=list)
    moderators: list[str] = Field(default_factory=list)
    stats: StatsSettings = Field(default_factory=StatsSettings)
    digitize: DigitizeSettings = Field(default_factory=DigitizeSettings)
    notes: str = ""

    def outcome(self, key: str) -> OutcomeDef:
        for o in self.outcomes:
            if o.key == key:
                return o
        raise KeyError(f"no outcome {key!r} in protocol (have {[o.key for o in self.outcomes]})")

    def group(self, key: str) -> GroupDef:
        if key == "A":
            return self.group_a
        if key == "B":
            return self.group_b
        raise KeyError(f"no group {key!r}")

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    def hash(self) -> str:
        """sha256 of the canonical JSON — identifies the protocol in run manifests."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- study map
class Citation(CanopyModel):
    authors: str = ""
    year: int | None = None
    title: str = ""
    journal: str = ""
    doi: str = ""
    first_author: str = ""


#: WHOSE numbers a location reports — the people the value at it describes. `both_groups` is this
#: contrast's own two groups, reported separately; `one_group` is one arm's own analysis;
#: `pooled` is the two combined, or a wider/narrower sample that is not the pair (an age-collapsed
#: analysis, a subgroup); `other` is a different set of people again; `unknown` is the honest
#: absence, which reads. Only `both_groups`/`unknown` are read for a cell's value: Bock's
#: adaptation magnitude `A=(I-F)/I` is computed over the POOLED seniors and sat in the map as a
#: `value` location for a two-group cell, with no field in which to say so.
SourceSample = Literal["both_groups", "one_group", "pooled", "other", "unknown"]


#: What a `Source.notes` says when a `which_measure` decision (C6) set that location aside, and
#: when it was set aside for naming no measure at all. Here rather than beside the code that writes
#: them because they are the RECORD of who demoted a location, and both the agent that writes the
#: record and the review page that has to offer that location back read them. Deliberately notes
#: rather than a field on `Source`: the whole study map is dumped verbatim into the map-adjudicator's
#: prompt, so a field added here changes that prompt and re-buys the call for every paper of every
#: run already on disk — the same rule `run` cites for the exclusions table's decider.
C6_DEMOTION_NOTE = "demoted to alternate"
C6_WITHHELD_NOTE = ("set aside by the which_measure decision: this location cannot say which "
                    "measure it reads, so it cannot be read as the winner's number")


#: `Source.sample` answers that are NOT the two groups a contrast compares, so a location carrying
#: one may never be read for the cell's value. Here beside `SourceSample` itself because the review
#: page has to refuse to OFFER such a location and may not import the agent package: a card that
#: offers a reading the pipeline will not read is a click that empties the cell.
UNREADABLE_SAMPLES: frozenset[str] = frozenset({"pooled", "other"})

#: how the record names who took a map-stage decision. Here, beside the notes those decisions
#: write, because the review page has to print the name and may not import the agent package.
HUMAN_DECIDER_NAME = "a human reviewer"
MAP_ADJUDICATOR_NAME = "map-adjudicator"


def c6_demoted_note(notes: str) -> bool:
    """Do these notes say a `which_measure` decision is what set this location aside?

    The test on the note alone; a caller that has the location itself must also check that its role
    is still `alternate`. An `alternate` the MAPPER wrote (a paper's own "DE (primary); IEE
    (alternative)") carries neither note, is nobody's decision about this review, and is never
    reopened by an answer to a C6 question.
    """
    text = notes or ""
    return (text.startswith(C6_DEMOTION_NOTE) or f"; {C6_DEMOTION_NOTE}" in text
            or C6_WITHHELD_NOTE in text)


class Source(CanopyModel):
    """Where a number lives in the paper (the mapper finds these; extractors read them)."""

    kind: SourceKind
    page: int                                   # 1-based
    locator: str = ""                           # "Fig 2A", "Table 1 row 3", "Results ¶2"
    quote: str = ""
    figure_id: str | None = None                # ingestion FigureRegion.id
    table_id: str | None = None
    error_bar_type: DispersionType = DispersionType.UNKNOWN
    #: what the x axis of a figure source IS. A `categorical` x (target directions, conditions,
    #: hands) means the outcome is an average ACROSS the axis, not a value at one point on it —
    #: reading one point there is a wrong number, not an imprecise one (task 16 P6).
    x_axis_kind: XAxisKind = "unknown"
    error_bar_scope: ErrorBarScope = "unknown"
    error_bar_evidence: str = ""
    analysis_metric: AnalysisMetric = "unknown"
    values_in_text: str = ""                    # verbatim values if printed
    #: set in code: did a second agent independently confirm `error_bar_type` for this location?
    error_bar_agreement: ErrorBarAgreement = "unconfirmed"
    #: `value` unless the mapper says otherwise. A baseline series listed "for baseline
    #: correction" was read as the outcome itself once (Cressman's aligned-cursor curves, 3.9°,
    #: beside the misaligned curves' 31.4°) — the map knew, and had no field to say it in.
    role: SourceRole = "value"
    #: whose numbers these are (see `SourceSample`). A `value` location whose sample is not this
    #: contrast's two groups is kept on the record and NOT read for the cell's value.
    sample: SourceSample = "unknown"
    sample_note: str = ""                       # the words that say whose numbers these are
    notes: str = ""


class GroupSpec(CanopyModel):
    label: str = ""                             # as written in the paper
    n: int | None = None                        # analysed n (post-exclusion)
    n_evidence: str = ""
    age_mean: float | None = None
    age_sd: float | None = None
    age_range: str = ""
    notes: str = ""


class OutcomeSources(CanopyModel):
    outcome_key: str
    measure_name: str = ""
    units: str = ""
    higher_is_better: bool | None = None
    higher_is_better_evidence: str = ""
    operationalization: str = ""
    analysis_metric: AnalysisMetric = "unknown"
    sources: list[Source] = Field(default_factory=list)
    #: C6: when the map named two measures for one outcome, the map-stage ruling that chose one —
    #: the winning metric and the verbatim quotes for winner and loser. Empty when nothing was
    #: disputed; a dispute nobody could settle becomes a `which_measure` question instead.
    measure_ruling: str = ""


class RosterDecision(CanopyModel):
    """The mapper's verdict on one figure/table that ingestion detected.

    Ingestion supplies the roster (id, page, printed label); the mapper only says whether the item
    carries numbers this review needs, so nothing in the paper is silently skipped.
    """

    kind: Literal["figure", "table"]
    id: str                                     # ingestion id, e.g. "fig03" / "p1t1"
    page: int
    label: str = ""
    relevant: bool
    reason: str = ""
    outcome_keys: list[str] = Field(default_factory=list)


class DatasetSpec(CanopyModel):
    """One independent A-vs-B contrast (sample × condition) inside a paper."""

    dataset_id: str
    label: str = ""
    experiment: str = ""
    condition: str = ""
    cluster_id: str = ""                        # paper sha12 — rows sharing it are dependent
    shared_control: bool = False
    exposure_order: ExposureOrder = "unknown"
    group_a: GroupSpec = Field(default_factory=GroupSpec)
    group_b: GroupSpec = Field(default_factory=GroupSpec)
    all_groups_listed: list[GroupSpec] = Field(default_factory=list)
    chosen_pair_rationale: str = ""
    moderators: dict[str, str] = Field(default_factory=dict)
    outcomes: list[OutcomeSources] = Field(default_factory=list)
    #: C7: a dataset only one of the two mapping agents proposed is an INCLUSION question, settled
    #: at the map stage. `included=False` means the adjudicator rejected it on a named protocol
    #: rule (quoted below) — the dataset stays on the record and is never extracted.
    included: bool = True
    exclusion_rule: str = ""                    # the protocol rule the rejection cited
    exclusion_quote: str = ""                   # the paper's own words the rejection rests on
    notes: str = ""

    def outcome(self, key: str) -> OutcomeSources:
        for o in self.outcomes:
            if o.outcome_key == key:
                return o
        raise KeyError(f"dataset {self.dataset_id} has no outcome {key!r}")


class MapQuestion(CanopyModel):
    """A question the MAP could not settle, which no extraction may be bought against.

    Two kinds today: `include_dataset` (C7 — only one mapping agent proposed this dataset and the
    adjudicator cited no protocol rule to reject it) and `which_measure` (C6 — one outcome, two
    measures, and the protocol's window fits both). Both are answered by a person; until then the
    cell they name is not read, because buying an extraction against an unsettled question is how
    a rejected dataset produced a fully signed effect size.
    """

    kind: Literal["include_dataset", "which_measure"]
    dataset_id: str = ""
    outcome_key: str = ""                       # empty for a whole-dataset question
    question: str = ""
    options: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)


class StudyMap(CanopyModel):
    paper_id: str                               # paper sha256
    citation: Citation = Field(default_factory=Citation)
    eligible: bool | None = None
    eligibility_rationale: str = ""
    exclusion_reason: str = ""
    design_notes: str = ""
    datasets: list[DatasetSpec] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)     # supplements referenced by the paper
    roster: list[RosterDecision] = Field(default_factory=list)  # one per ingested figure/table
    disagreements: list[str] = Field(default_factory=list)     # cross-check diffs
    needs_human: list[str] = Field(default_factory=list)       # cells no two agents agreed on
    #: map-stage questions that BLOCK extraction of the cell they name (C6, C7)
    open_questions: list[MapQuestion] = Field(default_factory=list)
    notes: str = ""
    model: str = ""
    prompt_version: str = ""
    llm_call_ids: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------- candidates
class Candidate(CanopyModel):
    """One extractor's answer for one (dataset, outcome, group) quantity set.

    NEVER contains a computed effect size; `reported_*` fields hold values transcribed verbatim
    from the paper.
    """

    candidate_id: str = ""
    paper_id: str = ""
    dataset_id: str = ""
    outcome_key: str = ""
    kind: CandidateKind
    group: GroupKey | None = None
    status: CandidateStatus = "found"
    source_kind: SourceKind | None = None

    # --- group statistics
    n: int | None = None
    n_quote: str = ""
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    ci_low: float | None = None
    ci_high: float | None = None
    unit: str = ""
    value_as_written: str = ""
    raw_value_semantics: RawValueSemantics = "unknown"
    analysis_metric: AnalysisMetric = "unknown"
    error_bar_scope: ErrorBarScope = "unknown"
    whisker_definition: WhiskerDefinition = "unknown"        # box plots
    points: list[float] = Field(default_factory=list)        # scatter read-outs

    # --- test statistics
    stat_type: Literal["t", "F", "p", "chi2", "unknown"] | None = None
    stat_value: float | None = None
    df: float | None = None
    df1: float | None = None
    df2: float | None = None
    tails: int | None = None
    p_kind: PKind | None = None
    p_value: float | None = None
    design: TestDesign = "unknown"
    direction: Direction = "unknown"
    #: what the statistic contrasts (P-B). Refused in `canopy.stats` unless `groups`.
    contrast_kind: ContrastKind = "unknown"
    #: the within-subject factors of the model this statistic came from, each with its number of
    #: levels as the paper states it ("target direction (8 levels)"). A main effect from a model
    #: containing them estimates the group contrast AVERAGED OVER every one of their levels, so
    #: an empty list is not "there were none" — it is "nobody recorded any" (C5 fails closed).
    within_factors: list[str] = Field(default_factory=list)
    #: the factors the OUTCOME's own measurement window averages over, read off the protocol.
    #: A within-subject factor that is not in this list is a factor the outcome does not average
    #: over, so the statistic answers a different question than the cell asks.
    outcome_averages_over: list[str] = Field(default_factory=list)
    #: the quantity the model was fitted to, in the paper's words ("per-subject mean of the eight
    #: target directions"), so a reader can check the estimand without re-reading the paper
    model_fitted_to: str = ""
    admissible: bool = True
    admissible_reason: str = ""

    # --- reported effect sizes (verbatim, not computed here)
    reported_value: float | None = None
    reported_scale: ReportedScale = "unknown"
    standardizer: Standardizer = "unknown"
    reported_ci_low: float | None = None
    reported_ci_high: float | None = None
    positive_means: Direction = "unknown"

    # --- provenance
    page: int | None = None
    page_corrected: bool = False
    quote: str = ""
    locator: str = ""
    row_header: str = ""
    col_header: str = ""
    crop_path: str = ""
    overlay_path: str = ""
    pixel_provenance: dict[str, Any] = Field(default_factory=dict)
    sigma: float | None = None                   # digitization uncertainty of the MEAN (data units)
    #: digitization uncertainty of `dispersion_value`, in the same units. Carries a disagreement
    #: about the error-bar half-length that the routes' agreement about the mean survives
    #: (amendment F applied per quantity — see `canopy.digitize.digitizer`).
    dispersion_sigma: float | None = None
    grounded: bool | None = None
    grounding_similarity: float | None = None
    route: str = ""
    model: str = ""
    prompt_version: str = ""
    llm_call_id: str = ""
    extractor_id: str = ""
    notes: str = ""


# ----------------------------------------------------------------------------- verification
class CheckFlag(CanopyModel):
    code: str
    severity: Literal["info", "warn", "error"] = "warn"
    message: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    #: what the check MEASURED, for the question a reviewer will be asked to answer with a number.
    #: D4-lite's `n_before_exclusions` puts `{"recruited", "excluded", "quote"}` here, so the card
    #: can offer `recruited − excluded` as an option instead of making the reviewer re-read the
    #: paper for two counts the check has already parsed. Free-form on purpose: the keys belong to
    #: the check that raised the flag and are documented there, and nothing in `confidence` or the
    #: resolver branches on them — a finding is weighed by its CODE, never by its detail.
    detail: dict[str, Any] = Field(default_factory=dict)


class VerifierVerdict(CanopyModel):
    """One adversarial reader's attempt to refute one candidate (Task 8)."""

    candidate_id: str = ""
    #: `not_run` and `no_value_printed` are ABSENCES, not doubts (ceiling C8): the first says the
    #: call never produced a verdict (truncated output, transport failure), the second that the
    #: paper prints no independent value for this cell. `canopy.verify.confidence` prices both at
    #: zero rather than as evidence against the reading, so an infrastructure failure can no longer
    #: score a cell BELOW one that was never scheduled.
    verdict: Literal["confirmed", "refuted", "ambiguous", "not_run",
                     "no_value_printed"] = "ambiguous"
    reason: str = ""
    alt_mean: float | None = None
    alt_dispersion_value: float | None = None
    alt_dispersion_type: DispersionType = DispersionType.UNKNOWN
    alt_n: int | None = None
    alt_page: int | None = None
    alt_quote: str = ""
    better_source: str = ""
    #: the named traps the verifier says it checked (wrong group, SE-vs-SD, baseline vs post, ...)
    checked: list[str] = Field(default_factory=list)
    #: how many times this cell had already been re-opened when this verdict was produced
    reopen: int = 0
    model: str = ""
    prompt_version: str = ""
    llm_call_id: str = ""
    notes: str = ""


class AdjudicatedGroup(CanopyModel):
    """The adjudicator's final answer for one group of one cell.

    An adjudicated value is a value like any other: it must point at a place in the paper. `quote`,
    `page` and `locator` carry that, `grounded` records whether the quote was found in the
    deterministic page text, and a value that matches no candidate AND cannot be grounded sets
    `needs_human` — an LLM's unverifiable number never enters a pooled estimate.
    """

    group: GroupKey
    n: int | None = None
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    unit: str = ""
    quote: str = ""
    page: int | None = None
    locator: str = ""
    grounded: bool | None = None
    grounding_similarity: float | None = None
    chosen_candidate_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    needs_human: bool = False


class Adjudication(CanopyModel):
    """The strongest model's ruling on a cell whose vote failed or whose verifier refuted."""

    dataset_id: str = ""
    outcome_key: str = ""
    groups: list[AdjudicatedGroup] = Field(default_factory=list)
    rationale: str = ""
    needs_human: bool = False
    chosen_candidate_ids: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    llm_call_id: str = ""
    notes: str = ""

    def group_values(self, key: str) -> AdjudicatedGroup | None:
        for group in self.groups:
            if group.group == key:
                return group
        return None


class OrientationRun(CanopyModel):
    """What one model answered about the direction of one measure."""

    #: C12: this reply was not a reply (a stub, raw serialisation debris, a decoding loop). The
    #: detector in `canopy.llm.client.degenerate_reply` decides it; recording it here means a
    #: replayed `verify.json` says so on the record instead of being re-derived from the prose.
    not_run: bool = False
    #: C3 row 1: the deterministic check found this reader's own stated direction contradicted by
    #: the resolved raw means, so its vote on the polarity was removed. Recorded per READER for the
    #: same reason `not_run` is: the summary field on the verdict collapses two readers into one
    #: answer, so the sentence the reviewer has to arbitrate — "the reader said group A came out
    #: higher" — is not recoverable from it once the two readers disagreed (whole-diff L3).
    discarded: bool = False
    higher_is_better: bool | None = None
    raw_value_semantics: RawValueSemantics = "unknown"
    direction_stated_in_text: Direction = "unknown"
    quotes: list[str] = Field(default_factory=list)
    reason: str = ""
    model: str = ""
    prompt_version: str = ""
    llm_call_id: str = ""


class OrientationVerdict(CanopyModel):
    """Whether a larger raw value on this measure means *more of the construct* (spec §3.3(5)).

    Decided once per (outcome, measure) by two independent agents; they must agree, or a human
    decides. `direction_stated_in_text` is what the paper itself says about which group came out
    higher — code compares it with the sign it computed (`sign_mismatch`).
    """

    outcome_key: str = ""
    measure_name: str = ""
    #: which dataset the readers were actually asked about.
    #:
    #: C3's contradiction check — a reader's own stated direction against the resolved raw group
    #: means — runs **ONCE**, on this dataset and no other. Its outcome (a discard, and therefore
    #: possibly an abstention) is a property of the **MEASURE**, not of the cell that happened to
    #: expose it: a reader whose words contradict the numbers it was reading is not a reader whose
    #: ballot can be trusted about that measure anywhere. So later datasets carrying the same
    #: measure **inherit the checked verdict** and are **not** re-checked against their own means —
    #: another dataset's means are a different comparison, and running the filter against them
    #: would throw out whichever reader read THIS paper correctly.
    dataset_id: str = ""
    higher_is_better: bool | None = None
    raw_value_semantics: RawValueSemantics = "unknown"
    #: C3 rule 5: the summary field above cannot hold two answers, so a disagreement collapses it to
    #: `"unknown"`. This keeps each reader's own answer, which is what a reviewer needs.
    raw_value_semantics_by_model: dict[str, str] = Field(default_factory=dict)
    direction_stated_in_text: Direction = "unknown"
    quotes: list[str] = Field(default_factory=list)
    reason: str = ""
    agreed: bool = False
    needs_human: bool = True
    #: P-A: a third read was bought for this measure, so a recombination of `runs` must settle by
    #: majority exactly as the original call did. Without it a re-run of `combine_orientation` over
    #: the same three ballots would silently downgrade a settled measure to a question.
    third_read: bool = False
    #: HOW the direction above was settled: `agreed` (the two readers said the same thing),
    #: `single_witness` (one ballot survived C3's contradiction check and stood alone),
    #: `tiebreak_ballot` (a third read broke a disagreement) or `human` (a review answered the
    #: card). It is copied onto every row the verdict signs, because "older adults adapted less"
    #: read off a majority of three machines is a different claim from one a person made, and a
    #: reader of the extraction table cannot tell them apart from the sign alone.
    orientation_source: str = ""
    runs: list[OrientationRun] = Field(default_factory=list)
    llm_call_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Verdict(CanopyModel):
    """Verification outcome for one resolved quantity set (dataset × outcome × group)."""

    dataset_id: str = ""
    outcome_key: str = ""
    group: GroupKey | None = None
    agreement: Literal["agree", "disagree", "single", "none"] = "none"
    agreeing_ids: list[str] = Field(default_factory=list)
    disagreeing_ids: list[str] = Field(default_factory=list)
    vote_method: str = ""
    vote_tolerance: float | None = None
    #: the two text extractors disagreed — the orchestrator owes this cell a third cheap candidate
    needs_third_candidate: bool = False
    #: every state `VerifierVerdict.verdict` can report, because `_verifier_summary` copies it
    #: here verbatim. `CanopyModel` does not validate on assignment, so a member missing from this
    #: literal is written and dumped without complaint and then raises on the first `--resume` —
    #: after the whole run has been paid for. Pinned by `test_models.py`.
    verifier_verdict: Literal["confirmed", "refuted", "ambiguous", "not_run",
                              "no_value_printed"] = "not_run"
    verifier_reason: str = ""
    verifiers: list[VerifierVerdict] = Field(default_factory=list)
    reopens: int = 0
    adjudicated: bool = False
    adjudication_rationale: str = ""
    flags: list[CheckFlag] = Field(default_factory=list)
    confidence: ConfidenceBucket = "needs_human"
    confidence_score: float | None = None
    #: C11: how far this score is from the bucket boundary nearest it, and which boundary that is.
    #: A cell 0.0000 from the line that decided it was not decided by the evidence, and a reviewer
    #: reading the queue cannot see that from the score alone.
    confidence_margin: float | None = None
    nearest_boundary: str = ""
    confidence_reasons: list[str] = Field(default_factory=list)
    needs_human: bool = False
    # resolved values
    n: int | None = None
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    ci_low: float | None = None
    ci_high: float | None = None
    points: list[float] = Field(default_factory=list)
    unit: str = ""
    sigma: float | None = None
    mad: float | None = None
    analysis_metric: AnalysisMetric = "unknown"
    route: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    # orientation (decided once per outcome/measure, copied onto every cell that used it)
    higher_is_better: bool | None = None
    orientation_evidence: str = ""
    #: `OrientationVerdict.orientation_source`, copied here by `confidence.resolve_cell` and from
    #: here onto the ROW (`pipeline.rows.prepare_rows`). It travels through the cell because the
    #: row is built from the two cells and from nothing else — the same path `higher_is_better`
    #: takes — so a direction and the account of how it was settled can never come apart.
    orientation_source: str = ""
    overridden_by_human: bool = False
    override_justification: str = ""


# ----------------------------------------------------------------------------- results
class EffectSizeRecord(CanopyModel):
    """Computed in code (canopy.stats) from resolved values — the only place effect sizes live."""

    paper_id: str = ""
    cluster_id: str = ""
    #: which PARTICIPANT sample this row came from, when the paper identifies one (the
    #: orchestrator stamps it from the mapper's experiment label for a first exposure). Empty means
    #: "no separate sample can be claimed", so `one_row_per_paper` treats such rows as dependent.
    sample_id: str = ""
    dataset_id: str = ""
    outcome_key: str = ""
    label: str = ""
    route: str = ""
    n_a: int | None = None
    n_b: int | None = None
    d: float | None = None
    g: float | None = None
    es: float | None = None                      # the estimator actually used
    var: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: Estimator = "cohen"
    variance_method: VarianceMethod = "borenstein"
    level: float = 0.95
    higher_is_better: bool | None = None
    orientation_applied: bool = False
    #: how the orientation above was settled — `OrientationVerdict.orientation_source`, copied
    #: onto the row so the extraction table can say who signed it
    orientation_source: str = ""
    conversion_chain: str = ""
    conversion_steps: list[str] = Field(default_factory=list)
    #: routes the resolved values could have supported, and why the ones ahead were not taken
    routes_available: list[str] = Field(default_factory=list)
    routes_rejected: dict[str, str] = Field(default_factory=dict)
    #: D1's precedence override: the route the precedence list actually chose, which could not be
    #: converted (a printed mean with no dispersion), and the reason a same-locator alternative
    #: was built instead. Set together with the `precedence_override` flag; a row carrying them is
    #: HELD (`confidence = "needs_human"`), because it is not the number the precedence list asked
    #: for. Empty on every row the precedence list could satisfy.
    route_overridden_from: str = ""
    precedence_override_reason: str = ""
    #: the numbers actually fed to the formula (after every conversion), for the extraction table
    inputs: dict[str, float | None] = Field(default_factory=dict)
    #: set when no route could produce an effect size (amendment C gates, missing values, ...)
    not_convertible_reason: str = ""
    digitization_var: float | None = None
    var_with_digitization: float | None = None
    digitization_var_share: float | None = None
    confidence: ConfidenceBucket = "needs_human"
    #: why the best-guess line admitted this held row, and on what evidence. Written by the
    #: report-side best-guess model onto a COPY of the row (the strict row is never touched), so a
    #: record read back from `resolve.json` always has them empty — best guess is a way of reading
    #: the primary analysis, never a stage that changes it.
    best_guess_rule: str = ""
    best_guess_reason: str = ""
    #: what the two groups' numbers measured (endpoint, change from baseline, ...) — copied from
    #: the verified cell by the orchestrator so the sensitivity set can split rows by metric
    analysis_metric: AnalysisMetric = "unknown"
    flags: list[str] = Field(default_factory=list)
    #: `flags`, kept per ARM instead of unioned — `{"A": [...], "B": [...]}`, and
    #: `{"<member>|A": [...], ...}` on a composite. A rule about ONE number (an untyped spread
    #: beside a guessed group size, say) must not be satisfiable by two of them, and `flags` alone
    #: cannot tell the two cases apart: it is a union, so it reads the same whether both codes
    #: describe one arm or one each. Unioning ARM SETS is safe; unioning code sets is not. Empty on
    #: a row nothing recorded arms for, which is a row no per-arm rule may fire on.
    arm_flags: dict[str, list[str]] = Field(default_factory=dict)
    moderators: dict[str, str] = Field(default_factory=dict)
    citation: Citation = Field(default_factory=Citation)
    notes: str = ""


class PaperStatus(CanopyModel):
    paper_id: str
    filename: str = ""
    status: Literal["pending", "ingested", "mapped", "extracted", "verified", "resolved",
                    "excluded", "error", "cancelled"] = "pending"
    eligible: bool | None = None
    stages: dict[str, str] = Field(default_factory=dict)     # stage -> "done"/"skipped"/error text
    cost_usd: float = 0.0
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class RunManifest(CanopyModel):
    run_id: str
    created_at: str
    protocol_hash: str
    protocol_path: str = ""
    canopy_version: str = ""
    git_commit: str = ""
    papers: list[PaperStatus] = Field(default_factory=list)
    settings: StatsSettings = Field(default_factory=StatsSettings)
    models: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    cost_usd: float = 0.0
    cache_hits: int = 0                        # replies the on-disk cache served (no API call)
    #: what each stage spent, and how much of the input the API's prompt cache served
    cost_by_stage: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cache: dict[str, Any] = Field(default_factory=dict)
    n_llm_calls: int = 0
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    human_review_queue: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
