"""The search's own vocabulary: one candidate, one record, and the projection the page reads.

THIS MODULE IS THE CONTRACT. The design review found the backend and the frontend had drifted
into two different APIs — five of ten count names differed, `phases` was a list on one side and a
map on the other, and the object the page rendered did not exist — so the shapes live here, in
code, and both sides import or mirror them rather than each describing them separately.

Two shapes on purpose:

* `Candidate` is the RECORD — everything the search learned about one paper, including how it was
  found, what the screener said and why, and every URL that was tried. It is what
  `search.json` holds and what is copied into the run, so a reader can reconstruct the whole
  search afterwards. It grows freely; nothing on the page depends on its internals.
* `project(candidate)` is the VIEW — the flat object the page renders, with display strings
  instead of structures. It exists so that adding a field to the record can never change what a
  browser receives, and so the page never has to know that `pdf.pages` lives at
  `fetch.probe.n_pages`.

Nothing here computes a statistic and nothing here knows anything about any research field
(`server/app.py`'s rule): a search chooses which papers to read, never what they mean.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = ["Candidate", "CandidateState", "SearchRecord", "PHASES", "COUNT_KEYS", "KEY_RE",
           "ABSTRACT_EXCERPT", "project", "counts_of", "new_key"]

#: where a candidate ended up, in the words the page shows. One field, because "is it fetched"
#: and "did the screener want it" and "did a human attach a PDF" are the same question asked at
#: different moments, and a page that reads three booleans renders a contradiction the first time
#: two of them disagree.
CandidateState = Literal[
    "fetched",       # an open-access PDF is on disk and readable
    "uploaded",      # a human supplied the PDF for this candidate
    "wanted",        # the screener wants it and the fetch stage has not resolved it yet
    "paywalled",     # wanted, and no open-access copy could be fetched — the user gets links
    "unsure",        # the screener could not tell from the title and abstract
    "excluded",      # the screener ruled it out, with its reason
    "not_screened",  # nobody read it: no model, or the cap stopped the search first
    "extra",         # a PDF the user added that no index proposed
]

#: the ladder the page draws, in order. The names are shared verbatim: two of these rungs used to
#: differ between the two sides ("query"/"queries", "oa"/"fetch") and never lit.
PHASES: tuple[str, ...] = ("queries", "index", "dedupe", "screen", "snowball", "fetch")

#: the PRISMA-ish ladder, and the ONLY count vocabulary. The page renders these keys directly, so
#: a name changed here is a name changed on screen — which is the point: one word per thing.
COUNT_KEYS: tuple[str, ...] = (
    "records",         # rows the indexes returned, before anything was removed
    "after_dedupe",    # distinct papers among them
    "screened",        # abstracts actually read
    "included",        # the screener wanted these
    "unsure",          # …could not tell about these
    "excluded",        # …ruled these out
    "excluded_by_user",      # a PERSON forbade these before anything read them — never a model's
    "not_screened",    # nobody read these (no model, or the cap stopped first)
    "snowballed",      # found by citation chasing, not by a search string
    "fetched",         # open-access PDFs on disk
    "unsure_fetched",  # …of which the screener could not decide about: read in the run, at a
                       # price the record states (`predicted.run_commit`)
    "wanted",          # the screener wants these; the fetch stage has not answered yet
    "paywalled",       # wanted, no OA copy — listed with links
    "uploaded",        # PDFs a human supplied for a paywalled candidate
    "extra",           # PDFs a human added that no index proposed
    "possible_duplicates",   # pairs the tool will NOT merge on its own (see dedupe)
)

#: an opaque, path-safe key. Deliberately not a DOI (it contains `/` and can never be path
#: material) and not an index id (it names one index, and a paper found twice has two).
KEY_RE = re.compile(r"^[cu][0-9a-f]{12}$")

#: characters of abstract `project()` sends to the page. See its docstring: the whole abstract is
#: a megabyte of JSON per poll, and none of it is a list of titles nobody can judge.
ABSTRACT_EXCERPT = 400


def new_key(prefix: str, seed: str) -> str:
    """A stable key for a candidate: `c…` for something an index proposed, `u…` for a user's PDF.

    Stable because it is derived from the paper's own identity (its DOI when it has one, else
    its normalised title and year), so the same paper found twice in one search — or found again
    in a re-search — keeps the key the page already has in its DOM.
    """
    import hashlib

    if prefix not in ("c", "u"):
        raise ValueError(f"a candidate key is 'c' or 'u', not {prefix!r}")
    return prefix + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


@dataclass
class Candidate:
    """One paper the search considered, and everything it learned about it.

    Every field that records a DECISION carries the reason beside it. That is Canopy's rule
    everywhere else (a flag names why it fired, an override carries its justification) and it is
    the whole difference between a search a reviewer can defend and a list of papers that
    appeared. `screen_reason` is the screener's own sentence, verbatim, never a paraphrase.
    """

    key: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    #: which index(es) proposed it — a paper found by two indexes lists both, and the record
    #: keeps that because "how many indexes agreed" is evidence about the search, not noise
    found_by: list[str] = field(default_factory=list)
    #: how many rows the indexes actually returned for this paper, across every index and every
    #: query. NOT `len(found_by)`: that list is unique, so a paper one index returned for five
    #: queries counted once and the top-of-funnel number a reviewer publishes under-counted (a
    #: measured 12 rows were reported as 4). `run.py` fills it in from the raw list the dedupe
    #: consumed. `0` means nobody counted — an earlier `search.json`, or a candidate built by hand —
    #: and `counts_of` falls back to `found_by` for those rather than shrinking the funnel.
    n_rows: int = 0
    #: ids the indexes gave it, for the audit trail and for re-finding the record later
    ids: dict[str, str] = field(default_factory=dict)
    #: the keys of rows this candidate absorbed when two indexes proposed the same paper. A
    #: merge is the one operation here that makes a row disappear, so it leaves a receipt: a
    #: reader of the record can account for every row an index returned.
    merged_from: list[str] = field(default_factory=list)
    #: the position at which each index form returned it — `{"pubmed:Q1:": 19,
    #: "openalex:Q1:ta": 89, …}`, best position kept across a merge. This is the number the
    #: bench grades an index on, and one of the things `relevance` is made of.
    ranks: dict[str, int] = field(default_factory=dict)
    #: where it stands before any cap (`rank.py`), and the one line saying why. None until the
    #: search ranked it; the screening cap cuts this order, never the arrival order.
    relevance: float | None = None
    relevance_why: str = ""
    #: citation chasing (design 03 §6): which round found it (0 = the first pass), how many seeds
    #: cite or are cited by it, whether it was itself expanded as a seed. Recorded now, filled
    #: by the snowball step.
    found_in_round: int = 0
    cited_by_seeds: int = 0
    seed: bool = False
    expanded_as_seed: bool = False
    #: the screener's per-question answers (`q1`…`q6`, `quote`) once the rubric screener writes
    #: them; empty under the first-generation screener
    screen_answers: dict[str, Any] = field(default_factory=dict)

    state: CandidateState = "not_screened"
    screen_decision: str = ""          # include | exclude | unknown | "" (never read)
    screen_reason: str = ""            # the screener's own words
    screened_on_title_only: bool = False   # no abstract was available to read

    #: the exclusion entry a USER typed that caught this paper, verbatim — `""` for every paper a
    #: user did not forbid. A structured field and not just a sentence in `screen_reason`, because
    #: "the tool judged this" and "the person forbade this" are different claims about a search and
    #: a reader of `search.json` must be able to tell them apart without parsing prose. It is also
    #: what keeps the arithmetic honest: `screen_decision` stays EMPTY on these rows, so a paper
    #: nobody read is never counted among the abstracts that were read.
    excluded_by_user: str = ""

    #: whether this paper goes into the run. Starts as the screener's verdict and is a human's
    #: to change; `state` never changes when this does, so a dropped paper keeps showing why it
    #: was found in the first place.
    keep: bool = False

    #: every URL tried, with what happened — `{"url", "outcome", "status", "host"}`. Kept even
    #: when one succeeded: "we tried the publisher, it refused, Europe PMC served it" is the
    #: sentence a user needs when a fetch is missing.
    fetch_attempts: list[dict[str, Any]] = field(default_factory=list)
    fetch_outcome: str = ""            # fetched | rate_limited | http_error | no_oa_location | …
    license: str = ""                  # per-article, as the index stated it — never assumed
    #: links a HUMAN may follow (doi.org first, publisher second and only when it differs). The
    #: page cannot build these: its own test forbids the literal `https://` in the SPA, so every
    #: outbound URL is one the server supplied.
    links: list[dict[str, str]] = field(default_factory=list)

    pdf_path: str = ""                 # relative to the search directory
    pdf_pages: int | None = None
    pdf_bytes: int | None = None
    upload_filename: str = ""          # what the human called the file they attached

    def study_label(self) -> str:
        """`Bock 2005` — the same shape the run's own tables use, so one paper reads the same
        on the search screen and in the review it becomes."""
        first = (self.authors[0] if self.authors else "").split(",")[0].strip()
        year = str(self.year) if self.year else ""
        return " ".join(x for x in (first or self.title[:28], year) if x).strip()


def project(candidate: Candidate) -> dict[str, Any]:
    """The candidate as the PAGE reads it — display strings, no nesting, no surprises.

    The abstract travels as an EXCERPT, not in full. In full it is up to 4 kB of publisher prose
    per row and 200 of them is a megabyte of JSON on every poll; absent altogether — which is what
    this used to send — it left the keyless user a list of bare titles they could not judge, while
    `run.py` promised "a list of candidates with their abstracts". `ABSTRACT_EXCERPT` characters is
    a sentence or three: enough to decide whether to open the link, bounded enough that the
    row-size arithmetic above still holds.

    `fetch` is the fetch stage's own outcome word, because "there is no PDF here" and "why there
    is no PDF here" are the same question, and the record already knew the answer.
    """
    return {
        "key": candidate.key,
        "title": candidate.title,
        "authors": _authors_line(candidate.authors),
        "study_label": candidate.study_label(),
        "year": candidate.year,
        "venue": candidate.venue,
        "doi": candidate.doi,
        "links": [dict(link) for link in candidate.links],
        "abstract_excerpt": (candidate.abstract or "")[:ABSTRACT_EXCERPT],
        "state": candidate.state,
        "fetch": candidate.fetch_outcome,
        "reason": candidate.screen_reason,
        # the page shows an excluded row the same way whoever excluded it — so it has to be told
        # WHO did. A boolean and not the entry itself: the entry is already inside `reason`, and
        # one fact does not travel twice.
        "excluded_by_you": bool(candidate.excluded_by_user),
        "title_only": candidate.screened_on_title_only,
        "source": ", ".join(candidate.found_by),
        "keep": bool(candidate.keep),
        "pdf": ({"pages": candidate.pdf_pages, "bytes": candidate.pdf_bytes}
                if candidate.pdf_path else None),
        "upload": {"filename": candidate.upload_filename},
    }


def _authors_line(authors: list[str]) -> str:
    """Three names, then `et al.` — a citation line, not a full author list."""
    if not authors:
        return ""
    if len(authors) <= 3:
        return "; ".join(authors)
    return "; ".join(authors[:3]) + " et al."


@dataclass
class SearchRecord:
    """What a search did, in the order it did it — the audit trail, copied into the run.

    A reviewer must be able to answer "why is this paper in my meta-analysis, and what did you
    not show me?" from this file alone, without the server running.
    """

    search_id: str = ""
    question: str = ""
    created_at: str = ""
    #: how the queries were built: `model` (one call) or `template` (from the protocol's own
    #: words, when no model was available). Shown on the page, because it changes what a
    #: reader should expect of the recall.
    query_source: str = "template"
    #: one row per (query, index, form): `{query_id, index, form, text, why, chars}`, plus
    #: `skipped: True` for a form the string was too long for
    queries: list[dict[str, Any]] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    #: the concept blocks the strings were built from, with every variant, every pruned term and
    #: its hit count, the width measurements and the rubric (`blocks.py`). The reason a word is
    #: or is not in a string is readable here without re-running anything.
    plan: dict[str, Any] = field(default_factory=dict)
    #: citation-chasing rounds (design 03 §6), one row each; empty until that step exists
    rounds: list[dict[str, Any]] = field(default_factory=list)
    #: rows asked of each index form per query (Europe PMC three times that)
    depth: int = 0
    #: per index: what it was asked, what it returned, what it uniquely contributed, and its
    #: error if it had one. `unique_contributed` is why the ordering is a measured fact here
    #: rather than an assumption about which index is best.
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: pairs the fuzzy pass thinks MIGHT be the same paper. Never merged automatically: a false
    #: merge deletes a study from a meta-analysis and nobody ever sees it, while a false split
    #: costs one click. Measured, not argued — see tests/test_search_dedupe.py.
    possible_duplicates: list[dict[str, Any]] = field(default_factory=list)
    #: what the USER forbade this search to propose, one row per entry they typed:
    #: `{"entry", "kind", "matched", "note"}`. `kind` is how the entry was read — `doi`, `title`,
    #: or `refused` for one too short to be safe as a title fragment — and `matched` is how many
    #: papers it actually caught.
    #:
    #: Every entry is listed, including the ones that caught nothing. A mistyped DOI that silently
    #: matched no paper would leave a user certain they had excluded something they had not, and
    #: a recall measured against a known review is only a number if the search says what it was
    #: forbidden to find. Empty for every search nobody excluded anything from.
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    #: one row per screening batch — `{index, keys, sent, estimated_usd, cost_usd, n_verdicts,
    #: error}`. It is what lets a reader price screening per batch and see which twenty records a
    #: failed call cost. `screen.py` used to say this was re-read on resume; there is no resume,
    #: and saying so was worse than not writing it down at all (review §M10).
    batches: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    #: "" | budget | deadline | cancelled — why the search stopped early, if it did. The page
    #: keys its banner on this, NOT on the job status: a capped search still finishes `done`,
    #: and a user told only "done" would never learn the cap cut their search short.
    stopped_because: str = ""
    cost_usd: float = 0.0
    #: what this search was predicted to cost, and what beginning it commits the user to:
    #: `run_commit = {n_read, n_wanted, n_unsure, per_paper_usd, usd, unsure_usd,
    #: per_paper_source}`. Predicted, never billed — the bills are `cost_usd` and the run's own.
    predicted: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["counts"] = counts_of(self.candidates, self.possible_duplicates)
        return out


def counts_of(candidates: list[Candidate],
              possible_duplicates: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """The PRISMA ladder, derived from the candidates themselves.

    Derived rather than incremented, because a counter maintained beside the thing it counts
    drifts from it the first time a code path forgets one — and the count a reviewer reads must
    be a fact about the list they are looking at.
    """
    states = [c.state for c in candidates]
    screened = [c for c in candidates if c.screen_decision]
    return {
        # `n_rows` where it was measured, `found_by` where it was not. `found_by` is unique, so
        # one index answering five queries with the same paper counted ONE record while the dedupe
        # rung four lines above said "2 distinct paper(s) from 12 record(s)" — two numbers on one
        # screen, disagreeing, and the wrong one was the top of the funnel. Both quantities are
        # LOWER bounds on the rows the indexes returned, so the larger is the honest one, and a
        # record written before `n_rows` existed still counts what it can account for.
        "records": sum(max(len(c.found_by) or 1, int(c.n_rows or 0)) for c in candidates),
        "after_dedupe": len(candidates),
        "screened": len(screened),
        "included": sum(1 for c in screened if c.screen_decision == "include"),
        "unsure": sum(1 for c in screened if c.screen_decision == "unknown"),
        "excluded": sum(1 for c in screened if c.screen_decision == "exclude"),
        # a separate rung from `excluded`, and it can never overlap with it: a user's exclusion is
        # applied before screening and leaves `screen_decision` empty, so these papers are not
        # among the abstracts that were read. Reported even when it is the same list a screener
        # would have thrown out anyway — a recall a reviewer publishes means nothing unless the
        # search states what it was forbidden to look at.
        "excluded_by_user": sum(1 for c in candidates if c.excluded_by_user),
        "not_screened": states.count("not_screened"),
        "snowballed": sum(1 for c in candidates if int(c.found_in_round or 0) > 0),
        "fetched": states.count("fetched"),
        # a fetched unsure paper is in BOTH `unsure` and `fetched`: the screener's verdict is one
        # fact and the file on disk another. This count says how many are both, because the run
        # cost that `begin` commits to is dollars per fetched paper and half of those may be
        # papers nobody was sure about (design 03 §5, m4).
        "unsure_fetched": sum(1 for c in screened
                              if c.screen_decision == "unknown" and c.state == "fetched"),
        # `wanted` and `paywalled` are deliberately separate: telling a user a paper is behind
        # a paywall when nothing ever tried to fetch it is a claim about a publisher that
        # nobody checked. Before the fetch stage answers, the honest word is "wanted".
        "wanted": states.count("wanted"),
        "paywalled": states.count("paywalled"),
        "uploaded": states.count("uploaded"),
        "extra": states.count("extra"),
        "possible_duplicates": len(possible_duplicates or []),
    }
