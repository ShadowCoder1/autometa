"""Which of these rows are the same paper — and, far more importantly, which ones we refuse to guess.

Three indexes return the same study three times, so something has to collapse them. The design's
first rule was: fold the titles, score them with `SequenceMatcher`, and merge anything at or above
0.93. The review implemented exactly that and ran it over this repo's own corpus (99 titles from
`runs/*/papers/*/ingest/paper.json`) plus realistic trial pairs, and the measurement came out the
opposite way round to the argument:

* it MERGED papers that are different — `…Part I. Kinematics` vs `…Part II.` at 0.990, and one
  trial's tremor paper vs its gait paper at 0.949, because a character-level ratio is dominated by
  the ~95 % of characters two sibling papers share and barely notices the one word that separates
  them;
* it SPLIT papers that are the same — the PMC copy stamped `Author Manuscript …` at 0.889, and a
  `Correction:` record at 0.927 — because those systematic prefixes are long enough to move the
  ratio further than a discriminating word does.

So the rule here is not the design's rule. It is the review's amendment, and its shape follows from
one asymmetry that the design itself stated and then legislated against:

    a false merge silently deletes a study from a meta-analysis and nobody ever sees it,
    while a false split shows the user two rows and costs one click.

Therefore:

1. **Automatic merges are identity-only.** An equal DOI, or an equal (normalised title, year). Both
   are equalities, not similarities — there is no threshold to be wrong about.
2. **Fuzz never merges.** A close pair becomes a recorded `possible_duplicate`: both rows stay in
   the list, linked, with the score and a sentence saying why we think so. The only destructive
   operation this module could perform is the one it does not have.
3. **A discriminator block.** Even at ratio 1.0 − ε, two titles that differ on a numeral, a roman
   numeral, or a word from a small vocabulary (`part`, `trial`, `low`, `high`, `dose`, …) are not
   proposed at all. Those are the words that carry the difference between sibling papers, and a
   character ratio is structurally blind to them.
4. **The author guard is affirmative.** Both first-author surnames present and equal. The design's
   "…or one of them is missing" made the guard vacuous on exactly the DOI-less, metadata-poor
   population the fuzzy pass exists to serve.

Normalisation does the rest of the work, and it earns its keep twice over: NFKD folds the `ﬁ`
ligature that publishers print into `force-ﬁeld`, and stripping `Author Manuscript` /
`Correction:` / `[…]` turns two of the review's three false splits into plain equalities that merge
with no judgement call at all. The orthography fold (`behavioural`→`behavioral`,
`randomised`→`randomized`) is there for the same reason: without it those pairs would only ever be
*proposed*, and the review measured them as papers that should simply be one row.

Nothing here computes a statistic and nothing here knows anything about any research field
(`server/app.py`'s rule). This module decides that two rows are one paper; it never decides what
the paper means.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any, Iterator, Sequence
from urllib.parse import unquote

from .models import Candidate, new_key

__all__ = ["FUZZ_THRESHOLD", "YEAR_WINDOW", "DISCRIMINATOR_WORDS", "ROMAN_NUMERALS",
           "normalise_title", "normalise_doi", "first_author_surname", "identity", "key_for",
           "title_ratio", "discriminating_tokens", "dedupe"]

#: the review kept the design's 0.93 verbatim, but it now gates a *proposal* rather than a merge —
#: which is why it is safe to leave high: the cost of missing a pair is that the user reads two
#: rows, not that a study vanishes.
FUZZ_THRESHOLD = 0.93

#: a paper and its correction (or its preprint) are usually a year apart; two years apart is
#: normally a different study. Only used to decide whether a pair is worth *scoring*.
YEAR_WINDOW = 1

#: a DOI is `10.<registrant>/<suffix>`. Anything else in a `doi` field — `n/a`, `PMC123456`, a bare
#: URL, an empty string an index filled with a dash — normalises to "" and merges nothing. Without
#: this, two records that both say `doi: "n/a"` would be automatically merged into one paper.
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


# ---------------------------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------------------------

#: the editorial furniture publishers and archives print *into* the title field. Each of these was
#: measured as a false split, not imagined: PMC stamps `Author Manuscript` onto the green-OA copy
#: of a paper whose publisher record has the clean title, and correction/erratum notices repeat the
#: original title verbatim behind a label.
#:
#: The correction family REQUIRES a delimiter (`Correction: X`, `Erratum to: X`) because the bare
#: word is a real title word — "Correction of gait asymmetry…" is a paper, not an erratum — while
#: `Author Manuscript` and a leading `[…]` never are, so those match without one.
#:
#: The character class is plain ASCII on purpose: this runs AFTER the ascii fold, by which point an
#: en-dash is already a space.
_WRAPPER_RE = re.compile(
    r"""^\s*(?:
          \[[^\]]{0,160}\](?=\s*\S)                            # [Article in German] <title…>
        | author\s+manuscripts?                                # PMC's stamp on green-OA copies
        | reply\s+to
        | (?:correction|corrigendum|erratum|retraction
            |retracted(?:\s+article)?|editorial)
          (?:\s+(?:to|on|of|for|in))?\s*[:.-]                  # the delimiter is mandatory here
    )\s*[:.-]?\s*""",
    re.VERBOSE,
)

#: `[Effect of levodopa on gait]` — PubMed brackets a title translated from another language. The
#: brackets are packaging, the title is inside them, and a rule that treated this like a prefix
#: would delete the entire title.
_BRACKETED_WHOLE_RE = re.compile(r"^\s*\[(.+)\]\s*$")

#: British → American orthography, as suffix rules rather than a word list, because a word list
#: would silently stop working on the first title that used a verb we had not thought of.
#:
#: These rules are applied to BOTH sides of every comparison, so a rule that mangles an innocent
#: word (`precise` → `preciz e`-ish stems) cannot invent a match between two different words: it
#: would have to map two genuinely different tokens onto one string, and `…ise`/`…ize` and
#: `…our`/`…or` pairs *are* the variant spellings we are trying to fold. The stem-length guards are
#: there for the one family that could collide for real — `four`→`for`, `hour`→`hor`, `tour`→`tor`.
_SPELLING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(.{3,})our(s|al|ally|able|ed|ing|ist|ite)?$"), r"\1or\2"),   # behavioural
    (re.compile(r"^(.{4,})is(e|ed|es|er|ers|ing|ation|ations)$"), r"\1iz\2"),   # randomised
    (re.compile(r"^(.{3,})lys(e|ed|es|ing)$"), r"\1lyz\2"),                     # analyse (not -lysis)
)


def _fold_ascii(text: str) -> str:
    """NFKD, then drop what will not fit in ASCII.

    NFKD is doing real work, not tidiness: it decomposes the `ﬁ` ligature that a publisher's
    typesetter left in `force-ﬁeld`, which is the difference between one row and two. The same
    pass turns `Müller` into `Muller` so a title indexed with and without diacritics is one title.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _strip_wrappers(text: str) -> str:
    """Peel editorial packaging off the front, repeatedly — `Correction: Author Manuscript X` happens."""
    previous = None
    while previous != text:
        previous = text
        text = _BRACKETED_WHOLE_RE.sub(r"\1", text)
        text = _WRAPPER_RE.sub("", text, count=1)
    return text


def _fold_spelling(token: str) -> str:
    for pattern, replacement in _SPELLING_RULES:
        folded = pattern.sub(replacement, token)
        if folded != token:
            return folded
    return token


def normalise_title(title: str) -> str:
    """The comparable form of a title: ascii-folded, unwrapped, lowercase, `[a-z0-9 ]`, one space.

    Order matters. The wrappers are stripped while the punctuation is still there (a `Correction:`
    is recognisable by its colon and a `[…]` by its brackets — after the punctuation strip they are
    just words), and the orthography fold runs last, on whole tokens, when there is nothing left in
    the string but letters, digits and single spaces.
    """
    text = _fold_ascii(title).lower()
    text = _strip_wrappers(text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [_fold_spelling(tok) for tok in text.split()]
    return " ".join(tokens)


def normalise_doi(doi: str) -> str:
    """`https://doi.org/10.1000/AB`, `DOI: 10.1000/ab`, `10.1000/ab.` → `10.1000/ab`; else `""`.

    Percent-decoding happens ONCE. A DOI arrives url-encoded from some indexes (`10.1%2Fab`) and
    decoding it makes those records merge with the plain ones; decoding it repeatedly would let a
    doubly-encoded string decide what it wants to be, which is how a normaliser becomes an attack
    surface rather than a normaliser.
    """
    text = unquote(str(doi or "").strip()).lower()
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^(?:info:)?doi:\s*", "", text)
    text = text.strip().rstrip(".,;:)]}’'\"")
    # a DOI is the one identifier here that is authoritative enough to merge two rows without a
    # second opinion, so it has to actually BE one.
    return text if DOI_RE.match(text) else ""


def first_author_surname(candidate: Candidate) -> str:
    """`Bock, Otmar` → `bock`; `Otmar Bock` → `bock`; no authors → `""`.

    The comma form is the contract's own assumption (`Candidate.study_label` splits on it to build
    `Bock 2005`), and the no-comma fallback takes the last token because that is what every index
    that omits the comma means. A surname we cannot read comes back as `""`, and the caller treats
    that as "do not propose" rather than "cannot object" — see `dedupe`.
    """
    if not candidate.authors:
        return ""
    raw = candidate.authors[0] or ""
    head = raw.split(",")[0].strip() if "," in raw else raw.strip().split(" ")[-1]
    return re.sub(r"[^a-z]", "", _fold_ascii(head).lower())


def identity(candidate: Candidate) -> str:
    """`doi:10.1/ab` or `t:<normalised title>|<year>` — the thing two rows must share to be one row.

    Exported because the candidate key is derived from it (§2.5) and the two must never drift: if
    the key said one thing about a paper's identity and the dedupe said another, the same paper
    found in a re-search would arrive under a key the page had never seen.
    """
    doi = normalise_doi(candidate.doi)
    if doi:
        return f"doi:{doi}"
    return f"t:{normalise_title(candidate.title)}|{candidate.year if candidate.year else ''}"


def key_for(candidate: Candidate, prefix: str = "c") -> str:
    """The stable key for a candidate, from the same identity the dedupe groups on."""
    return new_key(prefix, identity(candidate))


# ---------------------------------------------------------------------------------------------
# the fuzzy pass — which never merges anything
# ---------------------------------------------------------------------------------------------

#: words whose presence on one side and absence on the other IS the difference between two papers.
#: Every one of these was taken from a measured false merge or near-miss: `part` (Part I / Part II,
#: 0.990), `trial`/`protocol` (a protocol paper vs its results paper, 0.895), `low`/`high`/`dose`
#: (`Low-dose levodopa…` vs `High-dose…`, 0.929 — one thousandth from a silent deletion),
#: `follow`/`up`/`extension` (a follow-up report of the same trial), `pilot` (the pilot before the
#: full study).
DISCRIMINATOR_WORDS = frozenset({
    "part", "trial", "protocol", "low", "high", "dose", "follow", "up", "pilot", "extension",
})

#: `i` vs `ii` is two characters in a hundred and a completely different paper. Blocking is the
#: safe direction, so the set runs past the four the review named: a spurious block costs a
#: proposal the user never sees, a missing one costs a study.
ROMAN_NUMERALS = frozenset({"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"})


def title_ratio(left: str, right: str) -> float:
    """`SequenceMatcher` on the NORMALISED titles — the same number the review measured."""
    return SequenceMatcher(None, normalise_title(left), normalise_title(right)).ratio()


def discriminating_tokens(left_key: str, right_key: str) -> list[str]:
    """The tokens that make these two normalised titles different papers, if any.

    Returns the offending tokens rather than a bool so the record can say *which* word stopped the
    proposal — a decision in this repo carries its reason or it is not a decision.
    """
    difference = set(left_key.split()) ^ set(right_key.split())
    return sorted(tok for tok in difference
                  if tok.isdigit() or tok in ROMAN_NUMERALS or tok in DISCRIMINATOR_WORDS)


def _years_are_close(left: int | None, right: int | None) -> bool:
    """Within ±1 — or unknown on one side, which cannot rule a pair out and must not pretend to."""
    if left is None or right is None:
        return True
    return abs(left - right) <= YEAR_WINDOW


def _neighbour_pairs(candidates: Sequence[Candidate]) -> Iterator[tuple[int, int]]:
    """Index pairs worth scoring: same year, adjacent year, or an unknown year on either side.

    Bucketed rather than all-against-all because the pass is O(n²) otherwise and a search can carry
    a few hundred candidates. There is deliberately NO pair cap here: with year bucketing the real
    comparison count on a capped search is a few thousand, and a cap would need somewhere to record
    that it fired — the contract's return shape has no channel for that, and a pass that quietly
    degrades is worse than one that runs.
    """
    buckets: dict[int | None, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        buckets[candidate.year].append(index)

    emitted: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        year = candidate.year
        neighbourhood = [None] if year is None else [year - 1, year, year + 1, None]
        for neighbour in neighbourhood:
            for other in buckets.get(neighbour, ()):
                pair = (index, other) if index < other else (other, index)
                if pair[0] != pair[1] and pair not in emitted:
                    emitted.add(pair)
                    yield pair


def _possible_duplicate(left: Candidate, right: Candidate, ratio: float) -> dict[str, Any]:
    """One proposal, with the sentence a human needs to accept or reject it in one click."""
    return {
        "keys": [left.key, right.key],
        "ratio": round(ratio, 3),
        "why": (f"titles are {round(ratio * 100)}% alike after normalisation and the first author "
                f"is the same ({first_author_surname(left)}); "
                f"{_year_phrase(left.year, right.year)}. "
                f"Not merged: only an equal DOI or an equal title and year merge on their own."),
    }


def _year_phrase(left: int | None, right: int | None) -> str:
    if left is None or right is None:
        return "one of them has no year"
    if left == right:
        return f"both are {left}"
    return f"they are one year apart ({min(left, right)} and {max(left, right)})"


# ---------------------------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------------------------

def _richness(candidate: Candidate) -> tuple[int, int, int, int, int]:
    """How much this record knows, most authoritative first.

    A DOI outranks everything because it is the identity; then an abstract, because a record with
    one can be screened and a record without one can only be screened on its title; then the author
    list, then the length of the abstract, then how many indexes vouched for it.
    """
    return (
        1 if normalise_doi(candidate.doi) else 0,
        1 if candidate.abstract.strip() else 0,
        len(candidate.authors),
        len(candidate.abstract),
        len(candidate.found_by),
    )


def _is_wrapped(title: str) -> bool:
    """True when the raw title carries editorial furniture we had to strip to compare it."""
    return _strip_wrappers(_fold_ascii(title).lower()) != _fold_ascii(title).lower()


def _union(values: Sequence[str], *more: Sequence[str]) -> list[str]:
    """Order-preserving union — the page prints `found_by` and the search order is meaningful."""
    seen: dict[str, None] = {}
    for group in (values, *more):
        for value in group:
            seen.setdefault(value, None)
    return list(seen)


def _merge(group: Sequence[Candidate]) -> Candidate:
    """One paper out of several records of it, keeping the richest of everything.

    Field by field rather than "pick the best record wholesale", because the richest record is
    routinely the one missing the venue, and a merge that throws away a field some index did supply
    makes the search worse than not deduping at all.
    """
    if len(group) == 1:
        return group[0]

    # `sorted` is stable, so equally-rich records keep the order the indexes returned them in.
    ordered = sorted(group, key=_richness, reverse=True)
    primary = ordered[0]

    # The clean publisher title beats the archive's stamped one: after a merge the user should see
    # "Title of a paper", not "Author Manuscript Title of a paper".
    title = next((c.title for c in ordered if c.title.strip() and not _is_wrapped(c.title)),
                 primary.title)

    # A PDF already on disk decides the pdf/upload/state fields wholesale: those four describe one
    # file, and taking them from different records would describe a file that does not exist.
    with_pdf = next((c for c in ordered if c.pdf_path), None)
    screened = next((c for c in ordered if c.screen_decision), None)

    return replace(
        primary,
        # ONE key survives. A `u…` key wins when there is one, because a human's uploaded PDF is
        # bound to it on disk and re-keying would orphan the file.
        key=next((c.key for c in ordered if c.key.startswith("u")), primary.key),
        title=title,
        authors=list(max((c.authors for c in ordered), key=len)),
        year=next((c.year for c in ordered if c.year is not None), None),
        venue=next((c.venue for c in ordered if c.venue.strip()), ""),
        # the normalised form, because the page builds a doi.org link out of this string and a
        # `doi:` prefix or a trailing full stop would make that link 404.
        doi=next((normalise_doi(c.doi) for c in ordered if normalise_doi(c.doi)), ""),
        abstract=max((c.abstract for c in ordered), key=len),
        found_by=_union(*[c.found_by for c in group]),
        ids={k: v for c in reversed(ordered) for k, v in c.ids.items()},
        # the best position each index form gave any of the rows: a paper returned at 12 by one
        # query and 400 by another entered at 12, and that is the number the ranking reads
        ranks=_best_ranks(group),
        # every key this row absorbed, including keys those rows had absorbed before: a merge
        # is the only thing here that removes a row, and it may not do so without a receipt
        merged_from=_union(*[[c.key] + c.merged_from for c in ordered])[1:],
        license=next((c.license for c in ordered if c.license.strip()), ""),
        links=_merge_links(ordered),
        fetch_attempts=[a for c in group for a in c.fetch_attempts],
        fetch_outcome=next((c.fetch_outcome for c in ordered if c.fetch_outcome), ""),
        # Dedupe runs before screening, but it must not be the thing that erases a decision if it
        # is ever re-run on a search that has one — and `keep` is a human's answer, so any yes wins.
        state=(with_pdf or screened or primary).state,
        screen_decision=screened.screen_decision if screened else "",
        screen_reason=screened.screen_reason if screened else "",
        screened_on_title_only=screened.screened_on_title_only if screened else False,
        keep=any(c.keep for c in group),
        pdf_path=with_pdf.pdf_path if with_pdf else "",
        pdf_pages=with_pdf.pdf_pages if with_pdf else None,
        pdf_bytes=with_pdf.pdf_bytes if with_pdf else None,
        upload_filename=with_pdf.upload_filename if with_pdf else "",
    )


def _best_ranks(group: Sequence[Candidate]) -> dict[str, int]:
    best: dict[str, int] = {}
    for candidate in group:
        for label, position in (candidate.ranks or {}).items():
            try:
                value = int(position)
            except (TypeError, ValueError):
                continue
            if label not in best or value < best[label]:
                best[label] = value
    return best


def _merge_links(ordered: Sequence[Candidate]) -> list[dict[str, str]]:
    """Every outbound link any index gave us, once each — the page can only offer what we supply."""
    seen: dict[str, dict[str, str]] = {}
    for candidate in ordered:
        for link in candidate.links:
            seen.setdefault(link.get("url", ""), dict(link))
    return list(seen.values())


def _dois_conflict(left: str, right: str) -> bool:
    """Two different registered DOIs are two different registered records — full stop.

    This is what keeps a `Correction: X` (its own DOI) from being swallowed by `X` when the two
    normalise to the same title in the same year. The correction is still *proposed*, so the user
    can drop it; it is never silently folded into the paper it corrects.
    """
    return bool(left) and bool(right) and left != right


# ---------------------------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------------------------

def dedupe(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Collapse the records that are provably one paper; propose the rest, merge none of them.

    Returns `(merged, possible_duplicates)` where every pair in the second list names two keys that
    are BOTH still present in the first — that is the whole point. `SearchRecord.possible_duplicates`
    takes the second list verbatim and `counts_of` counts it.
    """
    groups: list[list[Candidate]] = []
    by_doi: dict[str, int] = {}
    by_title: dict[tuple[str, int | None], int] = {}
    group_dois: list[str] = []

    for candidate in candidates:
        doi = normalise_doi(candidate.doi)
        title_key = normalise_title(candidate.title)
        target: int | None = by_doi.get(doi) if doi else None

        if target is None and title_key:
            # rule 2: an equal (title, year) is one paper — unless the two carry different DOIs,
            # in which case the registries disagree with the titles and the registries win.
            sibling = by_title.get((title_key, candidate.year))
            if sibling is not None and not _dois_conflict(doi, group_dois[sibling]):
                target = sibling

        if target is None:
            groups.append([candidate])
            group_dois.append(doi)
            target = len(groups) - 1
        else:
            groups[target].append(candidate)
            if doi and not group_dois[target]:
                group_dois[target] = doi

        if doi:
            by_doi.setdefault(doi, target)
        if title_key:
            by_title.setdefault((title_key, candidate.year), target)

    merged = [_merge(group) for group in groups]
    title_keys = [normalise_title(c.title) for c in merged]

    pairs: list[dict[str, Any]] = []
    for left_index, right_index in _neighbour_pairs(merged):
        left, right = merged[left_index], merged[right_index]
        if not _years_are_close(left.year, right.year):
            continue

        left_key, right_key = title_keys[left_index], title_keys[right_index]
        if not left_key or not right_key:
            continue

        # affirmative only. "One of them is missing" was the design's rule and the review showed it
        # makes the guard vacuous on exactly the metadata-poor records the pass exists to serve.
        surname = first_author_surname(left)
        if not surname or surname != first_author_surname(right):
            continue

        # the words that separate sibling papers, checked BEFORE the ratio: a character ratio
        # cannot see them and no threshold will ever make it able to.
        if discriminating_tokens(left_key, right_key):
            continue

        # `real_quick_ratio` and `quick_ratio` are documented upper bounds on `ratio`, so skipping
        # on them cannot change which pairs are found — only how long the pass takes.
        matcher = SequenceMatcher(None, left_key, right_key)
        if (matcher.real_quick_ratio() < FUZZ_THRESHOLD
                or matcher.quick_ratio() < FUZZ_THRESHOLD):
            continue
        ratio = matcher.ratio()
        if ratio < FUZZ_THRESHOLD:
            continue

        pairs.append(_possible_duplicate(left, right, ratio))

    pairs.sort(key=lambda pair: (-pair["ratio"], pair["keys"]))
    return merged, pairs
