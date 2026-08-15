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


class DispersionType(str, Enum):
    SD = "SD"
    SE = "SE"
    CI95 = "CI95"
    CI90 = "CI90"
    IQR = "IQR"
    RANGE = "RANGE"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


GroupKey = Literal["A", "B"]
CandidateKind = Literal["group_stats", "test_statistic", "reported_d"]
CandidateStatus = Literal["found", "not_on_these_pages", "ambiguous"]
RawValueSemantics = Literal["higher_more_construct", "higher_more_error", "signed_direction", "unknown"]
AnalysisMetric = Literal["endpoint", "change_from_baseline", "baseline_corrected",
                         "percent_of_perturbation", "unknown"]
ErrorBarScope = Literal["between_subject", "within_subject_normalized", "unknown"]
WhiskerDefinition = Literal["min_max", "iqr_1_5", "percentile_5_95", "sd", "se", "ci", "unknown"]
TestDesign = Literal["independent_t", "one_way_between", "mixed_main_effect", "interaction",
                     "ancova", "paired", "welch", "unknown"]
PKind = Literal["exact", "less_than", "greater_than", "ns", "unknown"]
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
    ci_level: float = 0.95


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


class Source(CanopyModel):
    """Where a number lives in the paper (the mapper finds these; extractors read them)."""

    kind: SourceKind
    page: int                                   # 1-based
    locator: str = ""                           # "Fig 2A", "Table 1 row 3", "Results ¶2"
    quote: str = ""
    figure_id: str | None = None                # ingestion FigureRegion.id
    table_id: str | None = None
    error_bar_type: DispersionType = DispersionType.UNKNOWN
    error_bar_scope: ErrorBarScope = "unknown"
    error_bar_evidence: str = ""
    analysis_metric: AnalysisMetric = "unknown"
    values_in_text: str = ""                    # verbatim values if printed
    relevant: bool = True                       # mapper decides for every figure/table
    relevance_reason: str = ""
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
    notes: str = ""

    def outcome(self, key: str) -> OutcomeSources:
        for o in self.outcomes:
            if o.outcome_key == key:
                return o
        raise KeyError(f"dataset {self.dataset_id} has no outcome {key!r}")


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
    sigma: float | None = None                   # digitization uncertainty (data units)
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


class VerifierVerdict(CanopyModel):
    verdict: Literal["confirmed", "refuted", "ambiguous"] = "ambiguous"
    reason: str = ""
    alt_mean: float | None = None
    alt_dispersion_value: float | None = None
    alt_n: int | None = None
    better_source: str = ""
    model: str = ""
    llm_call_id: str = ""


class Verdict(CanopyModel):
    """Verification outcome for one resolved quantity set (dataset × outcome × group)."""

    dataset_id: str = ""
    outcome_key: str = ""
    group: GroupKey | None = None
    agreement: Literal["agree", "disagree", "single", "none"] = "none"
    agreeing_ids: list[str] = Field(default_factory=list)
    disagreeing_ids: list[str] = Field(default_factory=list)
    verifier_verdict: Literal["confirmed", "refuted", "ambiguous", "not_run"] = "not_run"
    verifier_reason: str = ""
    verifiers: list[VerifierVerdict] = Field(default_factory=list)
    adjudicated: bool = False
    adjudication_rationale: str = ""
    flags: list[CheckFlag] = Field(default_factory=list)
    confidence: ConfidenceBucket = "needs_human"
    confidence_score: float | None = None
    confidence_reasons: list[str] = Field(default_factory=list)
    needs_human: bool = False
    # resolved values
    n: int | None = None
    mean: float | None = None
    dispersion_value: float | None = None
    dispersion_type: DispersionType = DispersionType.UNKNOWN
    unit: str = ""
    sigma: float | None = None
    route: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    overridden_by_human: bool = False
    override_justification: str = ""


# ----------------------------------------------------------------------------- results
class EffectSizeRecord(CanopyModel):
    """Computed in code (canopy.stats) from resolved values — the only place effect sizes live."""

    paper_id: str = ""
    cluster_id: str = ""
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
    higher_is_better: bool | None = None
    orientation_applied: bool = False
    conversion_chain: str = ""
    digitization_var: float | None = None
    var_with_digitization: float | None = None
    confidence: ConfidenceBucket = "needs_human"
    flags: list[str] = Field(default_factory=list)
    moderators: dict[str, str] = Field(default_factory=dict)
    citation: Citation = Field(default_factory=Citation)
    notes: str = ""


class PaperStatus(CanopyModel):
    paper_id: str
    filename: str = ""
    status: Literal["pending", "ingested", "mapped", "extracted", "verified", "resolved",
                    "excluded", "error"] = "pending"
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
    cache_hits: int = 0
    n_llm_calls: int = 0
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    human_review_queue: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
