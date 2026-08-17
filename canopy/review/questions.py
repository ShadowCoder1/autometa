"""Turn every held cell into a question a person can answer from a screenshot.

The review queue says *that* a cell is held and lists its candidates. A reviewer does not want a
list; they want to be shown the picture and asked the one thing the tool could not settle:
"which of these two numbers is the older group's bar?", "is this the left or the right axis?",
"what do the error bars show?". Every question carries the screenshot the tool itself read, the
answers it is choosing between, and the override its answer becomes — so answering it is a
recorded decision, not a note in the margin.

Nothing here calls a model. It reads the run's own stage files and writes `questions.json` and
`questions.md` beside them; the server serves the same list and accepts answers.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..pipeline.state import paper_dir, read_json

__all__ = ["Question", "questions_for_run", "write_questions", "answer_to_override",
           "QUESTION_KINDS"]

#: what a question is about; the UI groups and phrases by kind
QUESTION_KINDS: tuple[str, ...] = (
    "which_value",        # candidates disagree — pick the right number, or "none of these"
    "confirm_value",      # one uncorroborated number — is it right?
    "which_axis",         # the readers and the ladder disagree about which axis the value is on
    "which_series",       # the marker/series for a group is disputed or may be swapped
    "error_bar_type",     # nothing settled what the error bars show
    "group_mapping",      # which printed group is A and which is B
    "verifier_refuted",   # a verifier says the value is wrong; the tool could not settle it
    "no_value",           # nothing usable was found — where is it, if anywhere?
    "other",              # a held cell that fits none of the above: show the reason
)

#: flag codes → the question they raise (first match wins, in this order)
_FLAG_TO_KIND: tuple[tuple[str, str], ...] = (
    ("axis_conflict", "which_axis"),
    ("calibration_disputed", "which_axis"),
    ("calibration_refuted", "which_axis"),
    ("series_transposed", "which_series"),
    ("series_identity_conflict", "which_series"),
    ("series_marker_mismatch", "which_series"),
    ("dispersion_type_from_legend", "error_bar_type"),
    ("figure_error_bar_unknown", "error_bar_type"),
    ("dispersion_type_conflict", "error_bar_type"),
    ("group_label_swapped", "group_mapping"),
    ("multi_group_closest_to_definition", "group_mapping"),
)

_MAX_OPTIONS = 6


class Question(dict):
    """A plain dict with a stable shape; subclassed only so the intent is visible in signatures."""


# ----------------------------------------------------------------------------- building
def questions_for_run(run_dir: str | Path) -> list[Question]:
    """Every held cell of a finished run, as questions, worst first (biggest |Δ pooled| on top)."""
    run = Path(run_dir)
    manifest = _json_if_present(run / "manifest.json") or {}
    queue = list(manifest.get("human_review_queue") or [])
    if not queue:                       # a run that held nothing writes no queue file at all
        queue = list(_json_if_present(run / "human_review_queue.json") or [])
    overrides = _overrides(run)
    provenance = _json_if_present(run / "provenance" / "provenance.json") or {}
    out: list[Question] = []
    for entry in queue:
        paper_id = str(entry.get("paper_id") or "")
        dataset_id = str(entry.get("dataset_id") or "")
        outcome_key = str(entry.get("outcome_key") or "")
        group = entry.get("group") if entry.get("group") in ("A", "B") else None
        verdict = _verdict(run, paper_id, dataset_id, outcome_key, group)
        candidates = _candidates(run, paper_id, dataset_id, outcome_key, group)
        record = _record(run, outcome_key, dataset_id)
        study, dataset = _dataset(run, paper_id, dataset_id)
        already = [o for o in overrides if o.get("dataset_id") == dataset_id
                   and o.get("outcome_key") in ("", outcome_key)
                   and (o.get("group") in (None, group) or o.get("kind") != "value")]
        out.append(_question(entry, verdict, candidates, record, study, dataset, provenance,
                             run, already))
    out.sort(key=lambda q: (q.get("answered", False),
                            -(q.get("impact") if isinstance(q.get("impact"), (int, float))
                              else -1.0)))
    for i, q in enumerate(out, 1):
        q["number"] = i
    return out


def _question(entry: Mapping[str, Any], verdict: Mapping[str, Any],
              candidates: Sequence[Mapping[str, Any]], record: Mapping[str, Any],
              study: Mapping[str, Any], dataset: Mapping[str, Any],
              provenance: Mapping[str, Any], run: Path,
              already: Sequence[Mapping[str, Any]]) -> Question:
    group = entry.get("group") if entry.get("group") in ("A", "B") else None
    outcome_key = str(entry.get("outcome_key") or "")
    flags = [str(f.get("code") or "") for f in (verdict.get("flags") or [])]
    valued = [c for c in candidates if c.get("mean") is not None]
    ensembles = [c for c in valued if str(c.get("extractor_id") or "").endswith("ensemble")]
    kind = _kind(entry, verdict, flags, valued)
    label = _group_label(dataset, group)
    citation = study.get("citation") or {}
    paper = f"{citation.get('first_author') or citation.get('authors') or '?'} {citation.get('year') or ''}".strip()
    where = _where(candidates, verdict)
    unit = _unit(candidates, dataset, outcome_key)
    x_hint = _x_hint(candidates)
    options = _options(kind, valued, ensembles, verdict, flags, unit)
    image = _image(run, candidates, provenance, verdict)
    prompt = _prompt(kind, label, outcome_key, where, unit, x_hint, options, verdict, entry)
    return Question(OrderedDict([
        ("id", "|".join([str(entry.get("dataset_id") or ""), outcome_key, group or "", kind])),
        ("kind", kind),
        ("prompt", prompt),
        ("paper", paper),
        ("paper_id", str(entry.get("paper_id") or "")),
        ("dataset_id", str(entry.get("dataset_id") or "")),
        ("dataset_label", str(dataset.get("label") or dataset.get("experiment") or "")),
        ("outcome_key", outcome_key),
        ("group", group),
        ("group_label", label),
        ("where", where),
        ("unit", unit),
        ("image", image),
        ("options", options),
        ("free_text", True),                       # a reviewer may always type the answer
        ("answer_writes", _answer_kind(kind)),
        ("why", _why(entry, verdict)),
        ("confidence", entry.get("confidence")),
        ("route", entry.get("route")),
        ("impact", entry.get("impact_abs_delta_pooled")),
        ("answered", bool(already)),
        ("answers", [{"kind": o.get("kind"), "justification": o.get("justification"),
                      "mean": o.get("mean"), "at": o.get("at") or o.get("timestamp")}
                     for o in already]),
    ]))


def _kind(entry: Mapping[str, Any], verdict: Mapping[str, Any], flags: Sequence[str],
          valued: Sequence[Mapping[str, Any]]) -> str:
    if str(entry.get("route") or "") == "not_convertible" or not valued:
        return "no_value"
    if verdict.get("verifier_verdict") == "refuted":
        return "verifier_refuted"
    for code, kind in _FLAG_TO_KIND:
        if code in flags:
            return kind
    distinct = _distinct_values(valued)
    if len(distinct) >= 2:
        return "which_value"
    if len(distinct) == 1:
        return "confirm_value"
    return "other"


def _options(kind: str, valued: Sequence[Mapping[str, Any]],
             ensembles: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
             flags: Sequence[str], unit: str) -> list[dict[str, Any]]:
    """The answers on offer. Values are the candidates' own numbers; never invented."""
    if kind == "error_bar_type":
        return [{"key": k, "label": lbl, "dispersion_type": k}
                for k, lbl in (("SD", "standard deviation"), ("SE", "standard error"),
                               ("CI95", "95% confidence interval"), ("IQR", "interquartile range"),
                               ("RANGE", "range"))]
    if kind == "group_mapping":
        return [{"key": "as_mapped", "label": "the mapping is right"},
                {"key": "swapped", "label": "the two groups are swapped"}]
    if kind == "which_axis":
        seen: list[dict[str, Any]] = []
        for c in valued:
            pp = c.get("pixel_provenance") or {}
            for axis in (pp.get("axis_reads") or ([pp.get("axis_read")] if pp.get("axis_read") else [])):
                axis = str(axis or "").strip()
                if axis and all(o["label"] != axis for o in seen):
                    seen.append({"key": f"axis_{len(seen) + 1}", "label": axis})
        return seen[:_MAX_OPTIONS]
    if kind == "which_series":
        return [{"key": "as_read", "label": "the value belongs to this group as read"},
                {"key": "other_series", "label": "the value belongs to the other group"}]
    if kind in ("which_value", "confirm_value", "verifier_refuted"):
        out: list[dict[str, Any]] = []
        for value, backers in _distinct_values(valued).items():
            best = backers[0]
            out.append({
                "key": f"v{len(out) + 1}",
                "label": f"{_fmt(value)}{(' ' + unit) if unit else ''}",
                "mean": value,
                "dispersion_value": best.get("dispersion_value"),
                "dispersion_type": _enum(best.get("dispersion_type")),
                "n": best.get("n"),
                "unit": unit,
                "backed_by": sorted({f"{_route_name(b)}" for b in backers}),
                "n_backers": len(backers),
                "quote": next((b.get("quote") for b in backers if b.get("quote")), ""),
                "page": best.get("page"),
            })
        out.sort(key=lambda o: -o["n_backers"])
        return out[:_MAX_OPTIONS]
    return []


def _distinct_values(valued: Sequence[Mapping[str, Any]]) -> "OrderedDict[float, list]":
    """Candidate values grouped by rounded mean, ensembles first, most-backed first."""
    order = sorted(valued, key=lambda c: (not str(c.get("extractor_id") or "").endswith("ensemble"),
                                          str(c.get("candidate_id") or "")))
    groups: "OrderedDict[float, list]" = OrderedDict()
    for c in order:
        value = float(c["mean"])
        key = next((k for k in groups if abs(k - value) <= max(abs(k) * 0.005, 1e-9)), None)
        groups.setdefault(value if key is None else key, []).append(c)
    return groups


def _prompt(kind: str, label: str, outcome_key: str, where: str, unit: str, x_hint: str,
            options: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
            entry: Mapping[str, Any]) -> str:
    who = f"the {label} group" if label else "this group"
    at = f" at {x_hint}" if x_hint else ""
    u = f" ({unit})" if unit else ""
    src = f" in {where}" if where else ""
    outcome = outcome_key.replace("_", " ")
    if kind == "which_value":
        return (f"Which of these is {who}'s {outcome}{at}{src}{u}? The routes that read it "
                f"disagree.")
    if kind == "confirm_value":
        return (f"Is {options[0]['label'] if options else 'this value'} {who}'s {outcome}{at}"
                f"{src}? Only one route produced it, so nothing independent confirms it.")
    if kind == "which_axis":
        return (f"Which value axis are {who}'s numbers{src} read from? The readers and the axis "
                f"ladder disagree about the scale.")
    if kind == "which_series":
        return (f"In {where or 'this figure'}, which plotted series is {who}? The marker the "
                f"reader described and the one the pixel pass found do not match.")
    if kind == "error_bar_type":
        return (f"What do the error bars{src} show? Nothing in the paper's text settled it, so "
                f"the spread cannot be converted with confidence.")
    if kind == "group_mapping":
        return (f"Is the group mapping right for this dataset — is {label or 'group A'} the "
                f"group the protocol calls A?")
    if kind == "verifier_refuted":
        return (f"A verifier reading the whole paper says this {outcome} value for {who} is "
                f"wrong: “{str(verdict.get('verifier_reason') or '')[:300]}”. Is it right "
                f"anyway, or should the cell be excluded?")
    if kind == "no_value":
        return (f"No usable {outcome} value was found for {who}{src}. Where in the paper is it "
                f"— page, figure or table — or is it genuinely not reported?")
    return f"This {outcome} cell for {who} was held for review. Why is given below."


# ----------------------------------------------------------------------------- answers
def _answer_kind(kind: str) -> str:
    """Which override an answer to this kind of question becomes."""
    return {"which_value": "value", "confirm_value": "mark_reviewed", "verifier_refuted": "value",
            "which_axis": "value", "which_series": "value", "error_bar_type": "value",
            "group_mapping": "mark_reviewed", "no_value": "re_extract"}.get(kind, "mark_reviewed")


def answer_to_override(question: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an answer into the override payload `append_override` validates.

    `answer` carries either `option` (a key from the question's options) or the free-text
    fields (`mean`, `dispersion_value`, `dispersion_type`, `n`, `hint`, `exclude`), plus an
    optional `note`. Whatever the reviewer chose, the justification names the question so the
    log reads as a decision about a stated uncertainty, not a bare number.
    """
    kind = str(question.get("kind") or "other")
    base = {"paper_id": question.get("paper_id", ""), "dataset_id": question.get("dataset_id", ""),
            "outcome_key": question.get("outcome_key", ""), "group": question.get("group")}
    note = str(answer.get("note") or "").strip()
    stem = f"answered question #{question.get('number', '?')} ({kind})"
    just = f"{stem}: {note}" if note else stem
    option = next((o for o in question.get("options") or []
                   if o.get("key") == answer.get("option")), None)

    if answer.get("exclude"):
        return {**base, "kind": "exclude_dataset", "justification": f"{just} — excluded"}
    if kind == "no_value" or answer.get("hint"):
        hint = str(answer.get("hint") or note or "").strip()
        if hint:
            return {**base, "kind": "re_extract", "hint": hint, "justification": just}
        return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                "justification": f"{just} — not reported"}
    if kind == "error_bar_type" and option is not None:
        return {**base, "kind": "value", "dispersion_type": option["dispersion_type"],
                "mean": None, "dispersion_value": None, "n": None,
                "justification": f"{just} — error bars are {option['label']}"}
    if kind == "group_mapping":
        if option is not None and option.get("key") == "swapped":
            return {**base, "kind": "exclude_dataset",
                    "justification": f"{just} — groups swapped; excluded pending re-mapping"}
        return {**base, "kind": "mark_reviewed", "confidence": "accept_with_note",
                "justification": f"{just} — mapping confirmed"}
    if kind == "which_series" and option is not None and option.get("key") == "other_series":
        return {**base, "kind": "exclude_dataset",
                "justification": f"{just} — value belongs to the other series; excluded "
                                 f"pending re-extraction"}
    if kind == "which_axis" and option is not None:
        return {**base, "kind": "mark_reviewed", "confidence": "needs_human",
                "justification": f"{just} — axis is {option['label'][:120]}"}
    # value-shaped answers: a chosen option's number, or a typed one
    mean = option.get("mean") if option is not None else answer.get("mean")
    if mean is not None or answer.get("dispersion_value") is not None or answer.get("n"):
        payload = {**base, "kind": "value",
                   "mean": mean,
                   "dispersion_value": (option or {}).get("dispersion_value")
                   if option is not None else answer.get("dispersion_value"),
                   "dispersion_type": (option or {}).get("dispersion_type")
                   if option is not None else answer.get("dispersion_type"),
                   "n": (option or {}).get("n") if option is not None else answer.get("n"),
                   "unit": question.get("unit") or "",
                   "justification": f"{just} — {(option or {}).get('label') or 'typed value'}"}
        return payload
    if kind == "confirm_value" and answer.get("option") == "yes":
        return {**base, "kind": "mark_reviewed", "confidence": "accept_with_note",
                "justification": f"{just} — confirmed"}
    return {**base, "kind": "mark_reviewed", "confidence": "needs_human", "justification": just}


# ----------------------------------------------------------------------------- output
def write_questions(run_dir: str | Path, questions: Sequence[Mapping[str, Any]] | None = None
                    ) -> dict[str, Path]:
    """`questions.json` and a readable `questions.md` in the run directory."""
    run = Path(run_dir)
    qs = list(questions) if questions is not None else questions_for_run(run)
    json_path = run / "questions.json"
    json_path.write_text(json.dumps(qs, ensure_ascii=False, indent=1, default=str),
                         encoding="utf-8")
    md = ["# Questions for the reviewer", "",
          f"{len(qs)} cell(s) need a decision. Each shows the picture the tool read, the answers "
          f"it is choosing between, and why it could not decide. Answer in the review tab, or by "
          f"appending to `overrides.jsonl` (`canopy validate` re-pools).", ""]
    for q in qs:
        head = f"## {q['number']}. {q['paper']} — {q['dataset_label'] or q['dataset_id']}"
        if q.get("group_label"):
            head += f" — {q['group_label']}"
        md += [head, "", f"**{q['prompt']}**", ""]
        if q.get("image", {}).get("path"):
            md += [f"![{q['kind']}]({q['image']['path']})", ""]
        for o in q.get("options") or []:
            extra = f" — backed by {', '.join(o['backed_by'])}" if o.get("backed_by") else ""
            md.append(f"- **{o['label']}**{extra}")
        if q.get("free_text"):
            md.append("- _(or type the value / where to find it)_")
        md += ["", f"<details><summary>why the tool could not decide</summary>", "",
               f"{q['why']}", "", "</details>", ""]
        if q.get("answered"):
            md += [f"_Answered: {q['answers'][-1].get('justification', '')}_", ""]
    (run / "questions.md").write_text("\n".join(md), encoding="utf-8")
    return {"json": json_path, "md": run / "questions.md"}


# ----------------------------------------------------------------------------- readers
def _overrides(run: Path) -> list[dict[str, Any]]:
    path = run / "overrides.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _json_if_present(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _stage(run: Path, paper_id: str, stage: str) -> dict[str, Any]:
    if not paper_id:
        return {}
    return _json_if_present(paper_dir(run, paper_id) / f"{stage}.json") or {}


def _verdict(run: Path, paper_id: str, dataset_id: str, outcome_key: str,
             group: str | None) -> dict[str, Any]:
    for v in _stage(run, paper_id, "verify").get("verdicts") or []:
        if (v.get("dataset_id") == dataset_id and v.get("outcome_key") == outcome_key
                and (group is None or v.get("group") in (group, None))):
            return v
    return {}


def _candidates(run: Path, paper_id: str, dataset_id: str, outcome_key: str,
                group: str | None) -> list[dict[str, Any]]:
    extract = _stage(run, paper_id, "extract")
    verify = _stage(run, paper_id, "verify")
    every = [*(extract.get("candidates") or []), *(verify.get("extra_candidates") or [])]
    return [c for c in every if c.get("dataset_id") == dataset_id
            and c.get("outcome_key") == outcome_key
            and (group is None or c.get("group") == group)]


def _record(run: Path, outcome_key: str, dataset_id: str) -> dict[str, Any]:
    rows = _json_if_present(run / "results" / outcome_key / "extraction_table.json") or []
    return next((r for r in rows if r.get("dataset_id") == dataset_id), {})


def _dataset(run: Path, paper_id: str, dataset_id: str
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    study = (_stage(run, paper_id, "map").get("study") or {})
    dataset = next((d for d in study.get("datasets") or [] if d.get("dataset_id") == dataset_id),
                   {})
    return study, dataset


def _group_label(dataset: Mapping[str, Any], group: str | None) -> str:
    if group is None:
        return ""
    return str(((dataset.get("group_a" if group == "A" else "group_b") or {}).get("label")) or "")


def _where(candidates: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any]) -> str:
    for c in candidates:
        loc = c.get("locator") or (c.get("pixel_provenance") or {}).get("figure_id")
        if loc:
            page = c.get("page")
            loc = str(loc)
            if len(loc) > 70:
                loc = loc[:67].rsplit(" ", 1)[0] + "…"
            return f"{loc}" + (f" (p. {page})" if page else "")
    return ""


def _unit(candidates: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any],
          outcome_key: str) -> str:
    def short(unit: Any) -> str:
        text = str(unit or "").strip()
        for sep in (";", " (", ","):                     # "degrees (CCW…); also percentage…"
            text = text.split(sep, 1)[0].strip()
        return text[:24]

    for c in candidates:
        if c.get("unit"):
            return short(c["unit"])
    for o in dataset.get("outcomes") or []:
        if o.get("outcome_key") == outcome_key and o.get("units"):
            return short(o["units"])
    return ""


def _x_hint(candidates: Sequence[Mapping[str, Any]]) -> str:
    def first(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = next((v for v in value if str(v or "").strip()), "")
        return str(value or "").strip()[:60]

    for c in candidates:
        pp = c.get("pixel_provenance") or {}
        for key in ("x_read", "late_window_x_read"):
            if first(pp.get(key)):
                return first(pp[key])
        rs = (pp.get("route_sample") or {}).get("extra") or {}
        if first(rs.get("x_read")):
            return first(rs["x_read"])
    return ""


def _image(run: Path, candidates: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any],
           verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The picture the reviewer needs: the overlay if there is one (its marks show where every
    route landed), else the figure crop, else the page crop behind a text quote. Paths in the
    records are written relative to the working directory of the run, or absolute; the result is
    made relative to the run directory so the server can serve it and the markdown can link it."""
    best: dict[str, Any] = {}

    def offer(raw: Any, rank: int, kind: str) -> None:
        nonlocal best
        if not raw or (best and best["rank"] <= rank):
            return
        path = Path(str(raw))
        for candidate_path in (path, run / path, Path.cwd() / path):
            if candidate_path.exists():
                best = {"rank": rank, "path": _rel(candidate_path, run), "kind": kind}
                return

    for c in candidates:
        offer(c.get("overlay_path"), 0, "overlay")
        offer((c.get("pixel_provenance") or {}).get("overlay_path"), 0, "overlay")
        offer(c.get("crop_path"), 1, "crop")
        fig_id = (c.get("pixel_provenance") or {}).get("figure_id")
        if fig_id:
            offer(paper_dir(run, str(c.get("paper_id") or "")) / "ingest" / "figures"
                  / f"{fig_id}.png", 2, "figure")
        entry = provenance.get(c.get("candidate_id") or "", {}) or {}
        offer(entry.get("crop"), 3, "page")
    if best:
        best.pop("rank", None)
    return best


def _rel(path: Path, run: Path) -> str:
    try:
        return str(path.resolve().relative_to(run.resolve()))
    except ValueError:
        return str(path)


def _why(entry: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    reason = str(entry.get("reason") or "").strip()
    return reason or "; ".join(verdict.get("confidence_reasons") or []) or "held for review"


def _route_name(c: Mapping[str, Any]) -> str:
    cid = str(c.get("candidate_id") or "")
    if ":digitize:" in cid:
        tail = cid.split(":digitize:", 1)[1]
        return {"ensemble": "figure (ensemble)", "raster_cv": "figure (pixels)",
                "vlm_coords": "figure (model coordinates)"}.get(
            tail.split(":")[0], f"figure ({tail.split(':')[0]}"
                                f"{', ' + tail.split(':')[1] if ':' in tail else ''})")
    return str(c.get("route") or "text") + (f" ({c.get('model')})" if c.get("model") else "")


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


def _fmt(value: float) -> str:
    return f"{value:.4g}" if abs(value) < 1000 else f"{value:.0f}"
