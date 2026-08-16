"""Shared helpers for the Cisneros-2024 validation scripts.

Three jobs:

* **read a finished run** back into the objects it was built from (`load_run`), with no model
  call — a run directory is self-describing, and every script here works from one;
* **read the human's gold spreadsheets** (`load_gold`) — the per-study Cohen's d, CI and seTE
  that Elizabeth Cisneros computed by hand.  These files are validation-only: nothing under
  `canopy/` may read them, and nothing here feeds them back into the pipeline;
* **the agreement statistics** (`lins_ccc`, `mae`, `sign_agreement`, `bland_altman`) and the
  join between the two, which is the only genuinely fiddly part (see `join_rows`).

No statistic is computed twice: pooling always goes through `canopy.stats.meta`.
"""
from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION = REPO_ROOT / "validation"
GOLD_DIR = VALIDATION / "reference" / "cisneros2024"
OUT_DIR = VALIDATION / "out"
SPLITS = VALIDATION / "splits.json"
PROTOCOL = REPO_ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"

#: which gold spreadsheet holds which protocol outcome
GOLD_FOR_OUTCOME = {"late_adaptation": "late_gsheet.csv", "aftereffect": "aft_gsheet.csv"}

#: how close an auto d has to be to the manual d to count as "reproduced".  0.1 is the same
#: threshold amendment F uses for the digitiser's own route agreement (`|Δd| < 0.1`), so the
#: whole project has one definition of "these two readings are the same number".
TOLERANCE_D = 0.1


# ============================================================================ text normalisation
def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def first_author(authors: str) -> str:
    """The first author's surname, from either spreadsheet style or a citation string.

    `"Fernández-Ruiz et al."` -> `fernandezruiz`; `"Heuer & Hegele"` -> `heuer`;
    `"Bock, O."` -> `bock`; `"J. Bock and R. Smith"` -> `bock`.
    """
    text = strip_accents(str(authors or "")).strip()
    text = re.sub(r"\b(et al\.?|and others)\b", " ", text, flags=re.I)
    text = re.split(r"\s*(?:&|,| and )\s*", text)[0]
    words = [w for w in re.split(r"[^A-Za-z-]+", text) if w]
    # initials ("J.", "O.") are single letters; the surname is the first word longer than one
    surname = next((w for w in words if len(w) > 1), words[0] if words else "")
    return re.sub(r"[^a-z]", "", surname.lower())


def norm_measure(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", strip_accents(str(text or "")).lower()).strip("_")


def norm_experiment(text: str) -> str:
    """`"1a"`, `"Exp. 1a"`, `"1"` -> `"1a"` / `"1"`; blank stays blank."""
    t = re.sub(r"(?i)\b(exp(eriment)?\.?|study)\b", " ", str(text or ""))
    t = re.sub(r"[^a-z0-9]+", "", t.lower())
    return t


def as_float(value: Any) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def as_int(value: Any) -> int | None:
    out = as_float(value)
    return None if out is None else int(round(out))


# ============================================================================ the gold data
@dataclass
class GoldRow:
    """One human-extracted dataset from the reference review's spreadsheet."""

    index: int                       # 1-based row number in the CSV, so a person can find it
    outcome_key: str
    author: str
    author_key: str
    year: int | None
    title: str
    experiment: str
    figure: str
    measure: str
    phase: str
    n_young: int | None
    n_old: int | None
    te: float | None                 # the human's Cohen's d
    ci_low: float | None
    ci_high: float | None
    se: float | None
    task: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        year = self.year if self.year is not None else "?"
        exp = f" exp {self.experiment}" if self.experiment else ""
        return f"{self.author_key.title()} {year}{exp}"


def load_gold(outcome_key: str, gold_dir: str | Path = GOLD_DIR) -> list[GoldRow]:
    """Every usable row of the gold spreadsheet for one outcome (rows with no TE are dropped)."""
    name = GOLD_FOR_OUTCOME.get(outcome_key)
    if name is None:
        raise KeyError(f"no gold spreadsheet is defined for outcome {outcome_key!r} "
                       f"(known: {sorted(GOLD_FOR_OUTCOME)})")
    path = Path(gold_dir) / name
    rows: list[GoldRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, raw in enumerate(csv.DictReader(handle), start=1):
            te = as_float(raw.get("TE"))
            if te is None:
                continue                                   # blank / "Unpublished" rows
            author = str(raw.get("Author", "")).strip()
            rows.append(GoldRow(
                index=index, outcome_key=outcome_key, author=author,
                author_key=first_author(author), year=as_int(raw.get("Year")),
                title=str(raw.get("Title", "")).strip(),
                experiment=norm_experiment(raw.get("Experiment")),
                figure=str(raw.get("Figure", "")).strip(),
                measure=norm_measure(raw.get("Dependent_Measure")),
                phase=norm_measure(raw.get("Phase")),
                n_young=as_int(raw.get("N_young")), n_old=as_int(raw.get("N_old")),
                te=te, ci_low=as_float(raw.get("CI_low")), ci_high=as_float(raw.get("CI_high")),
                se=as_float(raw.get("seTE")), task=str(raw.get("Task", "")).strip(), raw=dict(raw)))
    return rows


def gold_records(rows: Sequence[GoldRow], settings: Any) -> list[Any]:
    """The gold rows as `EffectSizeRecord`s, so `canopy.report.forest` can draw them.

    Nothing is recomputed: TE, seTE and the CI are the human's own numbers, carried across
    unchanged.  `var` is `seTE²` because that is what `random_effects` needs.
    """
    from canopy.models import Citation, EffectSizeRecord

    out: list[EffectSizeRecord] = []
    for row in rows:
        var = None if row.se is None else row.se ** 2
        out.append(EffectSizeRecord(
            paper_id=f"gold:{row.author_key}:{row.year}", cluster_id=f"gold:{row.author_key}",
            dataset_id=f"g{row.index}", outcome_key=row.outcome_key, label=row.label,
            route="manual", n_a=row.n_old, n_b=row.n_young, d=row.te, es=row.te, var=var,
            se=row.se, ci_low=row.ci_low, ci_high=row.ci_high,
            estimator=getattr(settings, "estimator", "cohen"),
            variance_method=getattr(settings, "variance", "hedges_olkin_df"),
            confidence="auto_accept", conversion_chain="human extraction (WebPlotDigitizer -> d)",
            citation=Citation(authors=row.author, year=row.year, first_author=row.author_key),
            moderators={"task": row.task, "measure": row.measure}))
    return out


def pool_gold(rows: Sequence[GoldRow], settings: Any) -> Any:
    """Pool the human's own TE/seTE with `canopy.stats.meta.random_effects` under `settings`."""
    from canopy.stats.meta import random_effects

    usable = [r for r in rows if r.te is not None and r.se not in (None, 0.0)]
    yi = [r.te for r in usable]
    vi = [r.se ** 2 for r in usable]                       # type: ignore[union-attr]
    if len(yi) < 2:
        return None
    return random_effects(yi, vi, method=settings.tau2_method, hakn=bool(settings.hakn),
                          level=float(settings.ci_level))


# ============================================================================ reading a run
@dataclass
class RunData:
    """A finished run, read back from disk with no model call."""

    run_dir: Path
    manifest: Any
    protocol: Any
    records: list[Any] = field(default_factory=list)         # EffectSizeRecord
    candidates: list[Any] = field(default_factory=list)      # Candidate
    verdicts: list[Any] = field(default_factory=list)        # Verdict
    studies: dict[str, Any] = field(default_factory=dict)    # sha256 -> StudyMap
    papers: dict[str, Any] = field(default_factory=dict)     # sha256 -> PaperRecord
    pooled: dict[str, dict[str, Any]] = field(default_factory=dict)   # outcome -> pooled.json

    @property
    def settings(self) -> Any:
        return self.protocol.stats

    def rows_for(self, outcome_key: str) -> list[Any]:
        return [r for r in self.records if r.outcome_key == outcome_key]

    def candidates_for(self, dataset_id: str, outcome_key: str) -> list[Any]:
        return [c for c in self.candidates
                if c.dataset_id == dataset_id and c.outcome_key == outcome_key]

    def verdicts_for(self, dataset_id: str, outcome_key: str) -> list[Any]:
        return [v for v in self.verdicts
                if v.dataset_id == dataset_id and v.outcome_key == outcome_key]

    def paper_of(self, record: Any) -> Any | None:
        return self.papers.get(record.paper_id)

    def study_of(self, record: Any) -> Any | None:
        return self.studies.get(record.paper_id)


def load_run(run_dir: str | Path, *, papers: bool = True) -> RunData:
    """Read a run directory back into the objects that produced it.

    `papers=False` skips re-reading the ingested `PaperRecord`s (they are only needed by the
    scripts that draw page crops), which makes loading a big run noticeably faster.
    """
    from canopy.models import Candidate, EffectSizeRecord, StudyMap, Verdict
    from canopy.pipeline.state import load_manifest, read_stage
    from canopy.protocol import load_protocol

    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(f"no such run directory: {run}")
    manifest = load_manifest(run)
    local = run / "protocol.yaml"
    protocol = load_protocol(local if local.exists() else manifest.protocol_path)
    data = RunData(run_dir=run, manifest=manifest, protocol=protocol)

    for status in manifest.papers:
        sha = status.paper_id
        for stage, sink in (("map", "map"), ("extract", "extract"), ("verify", "verify"),
                            ("resolve", "resolve")):
            try:
                payload = read_stage(run, sha, stage)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if sink == "map":
                data.studies[sha] = StudyMap.model_validate(payload["study"])
            elif sink == "extract":
                data.candidates += [Candidate.model_validate(c) for c in payload["candidates"]]
            elif sink == "verify":
                data.verdicts += [Verdict.model_validate(v) for v in payload["verdicts"]]
                data.candidates += [Candidate.model_validate(c)
                                    for c in payload.get("extra_candidates", [])]
            else:
                data.records += [EffectSizeRecord.model_validate(r) for r in payload["records"]]
        if papers:
            paper = load_paper(run, sha)
            if paper is not None:
                data.papers[sha] = paper

    for outcome in protocol.outcomes:
        path = run / "results" / outcome.key / "pooled.json"
        if path.exists():
            data.pooled[outcome.key] = json.loads(path.read_text())
    return data


def load_paper(run_dir: str | Path, sha256: str) -> Any | None:
    """The ingested `PaperRecord` for one paper of a run (`<ingest.out_dir>/paper.json`)."""
    from canopy.ingest.pdf import PaperRecord
    from canopy.pipeline.state import read_stage

    try:
        ingest = read_stage(run_dir, sha256, "ingest")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    directory = Path(ingest.get("out_dir", ""))
    if not (directory / "paper.json").exists():
        return None
    return PaperRecord.load(directory)


# ============================================================================ agreement metrics
def lins_ccc(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Lin's concordance correlation coefficient — agreement WITH the identity line.

    Pearson's r would be 1.0 for a tool that read every value at exactly half the human's; CCC
    penalises that, which is the whole point of a backward-validation plot.
    """
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((a - mx) ** 2 for a in xs) / n                # biased (n) — Lin's own definition
    vy = sum((b - my) ** 2 for b in ys) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    denom = vx + vy + (mx - my) ** 2
    return None if denom == 0 else 2 * cov / denom


def mae(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    return None if not pairs else sum(abs(a - b) for a, b in pairs) / len(pairs)


def rmse(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


def sign_agreement(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Fraction of pairs whose effects point the same way (an exact zero counts as agreeing)."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pairs:
        return None
    same = sum(1 for a, b in pairs if (a > 0) == (b > 0) or a == 0 or b == 0)
    return same / len(pairs)


def bland_altman(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    """Mean difference (bias) and the 95 % limits of agreement, auto − manual."""
    diffs = [b - a for a, b in zip(x, y) if a is not None and b is not None]
    if len(diffs) < 2:
        return {"bias": diffs[0] if diffs else None, "sd": None, "loa_low": None, "loa_high": None}
    bias = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - bias) ** 2 for d in diffs) / (len(diffs) - 1))
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def ci_of(record: Any) -> tuple[float | None, float | None]:
    """A record's CI, falling back to es ± 1.96·se when only the SE survived."""
    low, high = record.ci_low, record.ci_high
    if low is not None and high is not None:
        return low, high
    se = record.se if record.se is not None else (
        math.sqrt(record.var) if record.var not in (None, 0) else None)
    if record.es is None or se is None:
        return None, None
    return record.es - 1.96 * se, record.es + 1.96 * se


# ============================================================================ the join
@dataclass
class Pair:
    """One auto row matched (or not) to one gold row."""

    outcome_key: str
    auto: Any | None                    # EffectSizeRecord
    gold: GoldRow | None
    how: str                            # exact | n_pair | author_year | override | unmatched_*
    note: str = ""

    @property
    def auto_d(self) -> float | None:
        return None if self.auto is None else self.auto.es

    @property
    def manual_d(self) -> float | None:
        return None if self.gold is None else self.gold.te

    @property
    def delta(self) -> float | None:
        if self.auto_d is None or self.manual_d is None:
            return None
        return self.auto_d - self.manual_d

    @property
    def within_tolerance(self) -> bool:
        return self.delta is not None and abs(self.delta) <= TOLERANCE_D

    @property
    def label(self) -> str:
        if self.gold is not None:
            return self.gold.label
        from canopy.report.theme import study_label

        return study_label(self.auto) if self.auto is not None else "?"


def load_overrides(path: str | Path) -> dict[tuple[str, str], int]:
    """`(outcome_key, dataset_id) -> gold row index`, from the hand-maintained override CSV.

    The file exists so a genuinely hard join (a paper whose spreadsheet row carries a different
    experiment label, say) can be fixed by a person **once**, in a file a reviewer can read,
    instead of by loosening the matcher for everyone.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple[str, str], int] = {}
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 2:
        return {}
    for raw in csv.DictReader(lines):
        key = (str(raw.get("outcome_key", "")).strip(), str(raw.get("dataset_id", "")).strip())
        index = as_int(raw.get("gold_row_index"))
        if key[0] and key[1] and index:
            out[key] = index
    return out


def _auto_key(record: Any, study: Any | None) -> tuple[str, int | None]:
    author = record.citation.first_author or first_author(record.citation.authors)
    return first_author(author), record.citation.year


def _n_distance(record: Any, gold: GoldRow) -> float | None:
    """How far apart the two group sizes are (auto n_a = OLD, n_b = YOUNG)."""
    if record.n_a is None or record.n_b is None or gold.n_old is None or gold.n_young is None:
        return None
    return abs(record.n_a - gold.n_old) + abs(record.n_b - gold.n_young)


def join_rows(records: Sequence[Any], gold: Sequence[GoldRow], outcome_key: str, *,
              overrides: dict[tuple[str, str], int] | None = None,
              studies: dict[str, Any] | None = None) -> list[Pair]:
    """Match auto rows to gold rows, hardest-evidence-first, and report HOW each pair was made.

    The order matters: a wrong pairing is worse than an unmatched row, because it invents a
    disagreement (or hides one).  So the exact key is tried for every row before any fuzzier
    rule is allowed to consume a gold row, and every match records the rule that made it.

    1. `override`   — the hand-written CSV wins over everything.
    2. `exact`      — same first author, same year, same N pair, same experiment label.
    3. `n_pair`     — same author, same year, same N pair (the experiment label differs or the
                      spreadsheet left it blank).
    4. `author_year`— same author, year within ±1 (spreadsheets and Crossref disagree about a
                      couple of the 2001/2002 papers), and only when exactly one gold row and
                      one auto row are left for that author.
    """
    overrides = overrides or {}
    studies = studies or {}
    by_index = {row.index: row for row in gold}
    unclaimed = {row.index: row for row in gold}
    pairs: list[Pair] = []
    left: list[Any] = []

    for record in records:                                            # 1. overrides
        index = overrides.get((outcome_key, record.dataset_id))
        if index is not None and index in unclaimed:
            pairs.append(Pair(outcome_key, record, unclaimed.pop(index), "override",
                              "matched by validation/reference/cisneros2024/join_overrides.csv"))
        elif index is not None and index in by_index:
            pairs.append(Pair(outcome_key, record, by_index[index], "override",
                              "override points at a gold row another auto row already claimed"))
        else:
            left.append(record)

    for rule in ("exact", "n_pair"):                                  # 2, 3
        still: list[Any] = []
        for record in left:
            author, year = _auto_key(record, studies.get(record.paper_id))
            best: GoldRow | None = None
            for row in unclaimed.values():
                if row.author_key != author or row.year != year:
                    continue
                if _n_distance(record, row) != 0:
                    continue
                if rule == "exact":
                    exp = norm_experiment(getattr(record, "label", ""))
                    if row.experiment and row.experiment not in (exp, ""):
                        # the dataset label rarely carries the experiment; only reject when the
                        # label names a DIFFERENT experiment
                        if exp and exp != row.experiment:
                            continue
                best = row
                break
            if best is None:
                still.append(record)
            else:
                pairs.append(Pair(outcome_key, record, unclaimed.pop(best.index), rule))
        left = still

    remaining_by_author: dict[str, list[Any]] = {}                    # 4. author (+/- 1 year)
    for record in left:
        remaining_by_author.setdefault(_auto_key(record, None)[0], []).append(record)
    still = []
    for author, group in remaining_by_author.items():
        rows = [r for r in unclaimed.values() if r.author_key == author]
        if len(group) == 1 and len(rows) == 1:
            record, row = group[0], rows[0]
            year = _auto_key(record, None)[1]
            close = year is None or row.year is None or abs(year - row.year) <= 1
            distance = _n_distance(record, row)
            if close:
                note = "" if distance in (0, None) else f"group sizes differ by {distance}"
                pairs.append(Pair(outcome_key, record, unclaimed.pop(row.index), "author_year",
                                  note))
                continue
        still.extend(group)
    left = still

    for record in left:
        pairs.append(Pair(outcome_key, record, None, "unmatched_auto",
                          "no gold row for this dataset — a paper, experiment or outcome the "
                          "human review did not include, or a join that needs an override"))
    for row in unclaimed.values():
        pairs.append(Pair(outcome_key, None, row, "unmatched_gold",
                          "the human extracted this dataset and the run did not produce it"))
    pairs.sort(key=lambda p: (p.gold.index if p.gold else 10_000,
                              p.auto.dataset_id if p.auto else ""))
    return pairs


# ============================================================================ small utilities
def write_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path,
              columns: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(columns) if columns else []
    if not names:
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in names})
    return path


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def banner(title: str, width: int = 78) -> str:
    return f"\n{title}\n{'─' * min(width, max(len(title), 8))}"


def iter_records(run: RunData, outcome_key: str) -> Iterable[Any]:
    return (r for r in run.records if r.outcome_key == outcome_key)
