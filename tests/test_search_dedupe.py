"""The dedupe rule, pinned against the twelve pairs the review actually measured.

The design proposed "merge anything scoring >= 0.93". The review implemented that rule and ran it
over this repo's own corpus plus realistic trial pairs, and every row below is one of its results.
Four of them were wrong in a way that matters — two false splits, two false merges — and the
amendment (identity-only automatic merges, fuzz becomes a *proposal*, a discriminator block, an
affirmative author guard) exists to fix exactly those.

So the table is the test. It is here so that the next person who wants to "simplify" this module
back to one threshold has to delete a measurement rather than win an argument.

`reviewer_ratio` is the number the review measured under the ORIGINAL rule. It is documentation,
not an assertion, except where a row's whole point is the score (see the low/high-dose test) —
because three of these rows are now decided before a ratio is ever computed.
"""
from __future__ import annotations

import pytest

from canopy.search.dedupe import (DISCRIMINATOR_WORDS, FUZZ_THRESHOLD, dedupe,
                                  discriminating_tokens, first_author_surname, identity, key_for,
                                  normalise_doi, normalise_title, title_ratio)
from canopy.search.models import KEY_RE, Candidate, SearchRecord, counts_of, new_key

BOCK = "Bock, Otmar"


def cand(title: str, *, year: int | None = 2020, authors: tuple[str, ...] = (BOCK,),
         doi: str = "", found_by: tuple[str, ...] = ("epmc",), **extra) -> Candidate:
    """A candidate with a key derived the way the pipeline derives it, so keys are never the reason
    two rows do or do not merge."""
    seed = f"{title}|{year}|{doi}|{found_by}"
    return Candidate(key=new_key("c", seed), title=title, year=year, authors=list(authors),
                     doi=doi, found_by=list(found_by), **extra)


def verdict(candidates: list[Candidate]) -> str:
    """`merge` | `possible_duplicate` | `split` — what the pass did with two records."""
    merged, pairs = dedupe(candidates)
    if len(merged) == 1:
        assert not pairs, "a merged pair cannot also be a proposal"
        return "merge"
    if pairs:
        return "possible_duplicate"
    return "split"


# ---------------------------------------------------------------------------------------------
# the review's twelve rows
# ---------------------------------------------------------------------------------------------

#: (id, reviewer_ratio, title_a, title_b, year_a, year_b, expected verdict under the AMENDED rule)
#:
#: The titles for rows 1, 2, 5 and 6 are verbatim from this repo's own corpus
#: (`runs/*/papers/*/ingest/paper.json`), which is where the review found them. The rest are the
#: review's "realistic PD/exercise pairs", reconstructed to reproduce its measured ratios to within
#: a thousandth.
TABLE = [
    # --- the four false splits the original rule produced -------------------------------------
    pytest.param(
        0.889,
        "Author Manuscript Aging reduces asymmetries in interlimb transfer of visuomotor adaptation",
        "Aging reduces asymmetries in interlimb transfer of visuomotor adaptation",
        2011, 2011, "merge",
        id="pmc_author_manuscript_prefix",
    ),
    pytest.param(
        0.877,
        "Author Manuscript Mechanisms underlying interlimb transfer of visuomotor rotations",
        "Mechanisms underlying interlimb transfer of visuomotor rotations",
        2010, 2010, "merge",
        id="pmc_author_manuscript_prefix_2",
    ),
    pytest.param(
        # a correction carries its own DOI and its own year, so it is never folded into the paper
        # it corrects — but it is put in front of the user instead of being lost.
        0.927,
        "Correction: Resistance training and tremor in Parkinson disease: a randomised trial",
        "Resistance training and tremor in Parkinson disease: a randomised trial",
        2021, 2020, "possible_duplicate",
        id="correction_record",
    ),
    pytest.param(
        # THE ONE ROW THE AMENDMENT DOES NOT FIX. A subtitle that only one index carries scores
        # 0.86, below the 0.93 the review deliberately kept, and no rule in the amendment addresses
        # subtitles. Pinned as a split on purpose: the honest record of a known residual beats a
        # threshold lowered until the table goes green.
        0.861,
        "Visuomotor adaptation and proprioceptive recalibration in older adults",
        "Visuomotor adaptation and proprioceptive recalibration in older adults: evidence from reaching",
        2018, 2018, "split",
        id="subtitle_in_one_index_only_UNRESOLVED",
    ),
    # --- the rows the original rule got right, which must stay right ---------------------------
    pytest.param(
        0.712,
        "Interlimb transfer of visuomotor rotations depends on handedness",
        "Interlimb transfer of visuomotor rotations: independence of direction and final position information",
        2013, 2013, "split",
        id="two_genuinely_different_interlimb_papers",
    ),
    pytest.param(
        1.000,
        "Enhanced crosslimb transfer of force-ﬁeld learning for dynamics that are identical in "
        "extrinsic and joint-based coordinates for both limbs",
        "Enhanced crosslimb transfer of force-field learning for dynamics that are identical in "
        "extrinsic and joint-based coordinates for both limbs",
        2016, 2016, "merge",
        id="force_field_ligature",
    ),
    pytest.param(
        0.989,
        "Behavioural correlates of tremor in Parkinson disease",
        "Behavioral correlates of tremor in Parkinson disease",
        2019, 2019, "merge",
        id="behavioural_vs_behavioral",
    ),
    pytest.param(
        0.986,
        "A randomised controlled trial of resistance training in Parkinson disease",
        "A randomized controlled trial of resistance training in Parkinson disease",
        2015, 2015, "merge",
        id="randomised_vs_randomized",
    ),
    # --- the two false merges, which are what made the original rule a blocker ------------------
    pytest.param(
        0.990,
        "Gait in Parkinson disease. Part I. Kinematic analysis",
        "Gait in Parkinson disease. Part II. Kinematic analysis",
        2012, 2012, "split",
        id="part_i_vs_part_ii",
    ),
    pytest.param(
        # Not merged — which was the blocker — but the two do share an author, a year and 95 % of
        # their characters, so they are put to the user rather than silently kept apart.
        0.949,
        "Effects of resistance training on tremor in Parkinson disease: a randomised trial",
        "Effects of resistance training on gait in Parkinson disease: a randomised trial",
        2017, 2017, "possible_duplicate",
        id="one_trials_tremor_paper_vs_its_gait_paper",
    ),
    pytest.param(
        0.929,
        "Low-dose levodopa for tremor in Parkinson disease",
        "High-dose levodopa for tremor in Parkinson disease",
        2014, 2014, "split",
        id="low_dose_vs_high_dose",
    ),
    pytest.param(
        0.895,
        "Resistance training for tremor in Parkinson disease: study protocol for a randomised controlled trial",
        "Resistance training for tremor in Parkinson disease: a randomised controlled trial",
        2016, 2017, "split",
        id="study_protocol_vs_results_paper",
    ),
]


@pytest.mark.parametrize("reviewer_ratio,title_a,title_b,year_a,year_b,expected", TABLE)
def test_every_row_the_review_measured(reviewer_ratio, title_a, title_b, year_a, year_b, expected):
    """Same first author on both sides throughout — that is the population the review tested, and
    it is what makes the author guard useless as an excuse for any of these verdicts."""
    assert verdict([cand(title_a, year=year_a, found_by=("epmc",)),
                    cand(title_b, year=year_b, found_by=("openalex",))]) == expected


@pytest.mark.parametrize("reviewer_ratio,title_a,title_b,year_a,year_b,expected", TABLE)
def test_no_row_is_decided_by_the_order_the_indexes_returned_it(
        reviewer_ratio, title_a, title_b, year_a, year_b, expected):
    """The same two records the other way round. A dedupe whose answer depends on arrival order
    gives two users different reviews of the same literature."""
    assert verdict([cand(title_b, year=year_b, found_by=("openalex",)),
                    cand(title_a, year=year_a, found_by=("epmc",))]) == expected


@pytest.mark.parametrize("reviewer_ratio,title_a,title_b,year_a,year_b,expected", TABLE)
def test_no_row_is_ever_merged_by_the_fuzzy_pass(
        reviewer_ratio, title_a, title_b, year_a, year_b, expected):
    """The amendment's first clause, stated as an invariant over the whole table: anything that
    merged did so because two normalised titles (or two DOIs) were EQUAL, never because they were
    close. A merge of unequal identities is the operation that deletes a study."""
    merged, _ = dedupe([cand(title_a, year=year_a), cand(title_b, year=year_b)])
    if len(merged) == 1:
        assert normalise_title(title_a) == normalise_title(title_b)
        assert year_a == year_b


# ---------------------------------------------------------------------------------------------
# the rows whose whole point is the number
# ---------------------------------------------------------------------------------------------

def test_low_dose_and_high_dose_never_merge_even_at_a_ratio_of_0_929():
    """0.929 is one thousandth from 0.93. Under the original rule this pair was a coin toss decided
    by the length of the journal's title style; under this one the ratio never gets consulted."""
    low = "Low-dose levodopa for tremor in Parkinson disease"
    high = "High-dose levodopa for tremor in Parkinson disease"
    assert title_ratio(low, high) == pytest.approx(0.929, abs=0.001)

    merged, pairs = dedupe([cand(low, year=2014), cand(high, year=2014)])
    assert len(merged) == 2
    assert pairs == []
    assert discriminating_tokens(normalise_title(low), normalise_title(high)) == ["high", "low"]


def test_part_i_and_part_ii_are_blocked_by_the_discriminator_not_by_the_threshold():
    """The ratio is 0.99 — far above anything a threshold could save us from. Only the roman
    numeral tells these two apart, which is the case a character ratio cannot ever see."""
    one = "Gait in Parkinson disease. Part I. Kinematic analysis"
    two = "Gait in Parkinson disease. Part II. Kinematic analysis"
    assert title_ratio(one, two) > FUZZ_THRESHOLD
    assert discriminating_tokens(normalise_title(one), normalise_title(two)) == ["i", "ii"]
    merged, pairs = dedupe([cand(one, year=2012), cand(two, year=2012)])
    assert len(merged) == 2 and pairs == []


def test_the_tremor_and_gait_papers_are_proposed_rather_than_merged():
    """The review's second false merge. It still scores above the threshold and it still shares an
    author — so the answer is "ask", not "guess"."""
    tremor = "Effects of resistance training on tremor in Parkinson disease: a randomised trial"
    gait = "Effects of resistance training on gait in Parkinson disease: a randomised trial"
    assert title_ratio(tremor, gait) >= FUZZ_THRESHOLD

    merged, pairs = dedupe([cand(tremor, year=2017), cand(gait, year=2017)])
    assert [c.title for c in merged] == [tremor, gait]
    assert len(pairs) == 1
    assert set(pairs[0]["keys"]) == {c.key for c in merged}


# ---------------------------------------------------------------------------------------------
# possible duplicates: recorded, never acted on
# ---------------------------------------------------------------------------------------------

def test_a_correction_record_is_proposed_not_merged_and_not_forgotten():
    """Both halves of the review's finding at once: the `Correction:` prefix is normalised away (so
    the pair is *seen*, which at 0.927 it was not), and the pair is proposed rather than merged (so
    the erratum cannot swallow the paper, or the paper the erratum)."""
    paper = cand("Resistance training and tremor in Parkinson disease: a randomised trial",
                 year=2020, doi="10.1000/pd.2020.114", found_by=("epmc",))
    correction = cand("Correction: Resistance training and tremor in Parkinson disease: "
                      "a randomised trial",
                      year=2021, doi="10.1000/pd.2021.998", found_by=("crossref",))

    merged, pairs = dedupe([paper, correction])
    assert len(merged) == 2, "a correction is its own registered record and keeps its own row"
    assert len(pairs) == 1
    assert pairs[0]["ratio"] == 1.0
    assert set(pairs[0]["keys"]) == {paper.key, correction.key}
    assert "Not merged" in pairs[0]["why"]


def test_two_different_dois_are_never_merged_even_with_an_identical_title_and_year():
    """The registries disagreeing with the titles is the registries being right. This is what keeps
    the same-year correction — and a reprint, and a translated republication — out of the paper."""
    merged, pairs = dedupe([
        cand("Aerobic exercise and gait in Parkinson disease", year=2019, doi="10.1000/a"),
        cand("Aerobic exercise and gait in Parkinson disease", year=2019, doi="10.1000/b"),
    ])
    assert len(merged) == 2
    assert len(pairs) == 1, "still surfaced to the user, just not decided for them"


def test_every_proposed_pair_names_two_rows_that_are_both_still_there():
    """The point of a proposal is that the user can act on it. A pair naming a key that the merge
    pass removed is a dead link on the page and an undeletable row in the record."""
    _, pairs = dedupe(_MIXED_CORPUS())
    merged, _ = dedupe(_MIXED_CORPUS())
    live = {c.key for c in merged}
    for pair in pairs:
        assert len(pair["keys"]) == 2
        assert set(pair["keys"]) <= live


def test_a_proposal_carries_its_reason_and_a_score_the_record_can_hold():
    _, pairs = dedupe([
        cand("Effects of resistance training on tremor in Parkinson disease: a randomised trial"),
        cand("Effects of resistance training on gait in Parkinson disease: a randomised trial"),
    ])
    assert len(pairs) == 1
    pair = pairs[0]
    assert set(pair) == {"keys", "ratio", "why"}
    assert isinstance(pair["ratio"], float) and FUZZ_THRESHOLD <= pair["ratio"] <= 1.0
    assert "bock" in pair["why"], "the reason names the evidence, not just the verdict"


def test_the_pairs_go_into_the_record_and_are_counted():
    """The contract's own field. `possible_duplicates` is the one count on the PRISMA ladder that
    exists because of this amendment, so it has to add up."""
    merged, pairs = dedupe([
        cand("Effects of resistance training on tremor in Parkinson disease: a randomised trial"),
        cand("Effects of resistance training on gait in Parkinson disease: a randomised trial"),
    ])
    record = SearchRecord(candidates=merged, possible_duplicates=pairs)
    assert record.to_json()["counts"]["possible_duplicates"] == 1
    assert counts_of(merged, pairs)["after_dedupe"] == 2


# ---------------------------------------------------------------------------------------------
# the author guard — affirmative only
# ---------------------------------------------------------------------------------------------

def test_a_close_pair_with_different_first_authors_is_not_proposed():
    merged, pairs = dedupe([
        cand("Effects of resistance training on tremor in Parkinson disease: a randomised trial",
             authors=("Bock, Otmar",)),
        cand("Effects of resistance training on gait in Parkinson disease: a randomised trial",
             authors=("Sainburg, Robert",)),
    ])
    assert len(merged) == 2 and pairs == []


def test_a_missing_first_author_does_not_pass_the_guard():
    """The design said "…or one of them is missing". The review showed that makes the guard vacuous
    on precisely the DOI-less, author-less records the fuzzy pass exists for — every one of them
    would pass a guard that treats silence as agreement."""
    merged, pairs = dedupe([
        cand("Effects of resistance training on tremor in Parkinson disease: a randomised trial",
             authors=("Bock, Otmar",)),
        cand("Effects of resistance training on gait in Parkinson disease: a randomised trial",
             authors=()),
    ])
    assert len(merged) == 2 and pairs == []


@pytest.mark.parametrize("written,expected", [
    ("Bock, Otmar", "bock"),
    ("Otmar Bock", "bock"),
    ("BOCK, O.", "bock"),
    ("Müller, Anna", "muller"),
    ("", ""),
])
def test_a_surname_is_read_the_same_way_however_the_index_wrote_it(written, expected):
    assert first_author_surname(cand("x", authors=(written,) if written else ())) == expected


# ---------------------------------------------------------------------------------------------
# automatic merges: DOI, then title+year
# ---------------------------------------------------------------------------------------------

def test_a_doi_merges_two_indexes_and_unions_found_by():
    """"How many indexes agreed" is evidence about the search (models.py), so a merge that kept one
    name and dropped the other would be deleting a measurement, not tidying a list."""
    merged, pairs = dedupe([
        cand("Older adults learn less, but still reduce metabolic cost, during motor adaptation",
             year=2021, doi="10.1152/jn.00105.2021", found_by=("epmc",)),
        cand("Older adults learn less but still reduce metabolic cost during motor adaptation",
             year=2021, doi="https://doi.org/10.1152/JN.00105.2021", found_by=("openalex",)),
    ])
    assert len(merged) == 1 and pairs == []
    assert merged[0].found_by == ["epmc", "openalex"]
    assert merged[0].doi == "10.1152/jn.00105.2021"


def test_a_doi_merge_beats_a_disagreeing_title_and_year():
    """DOIs are authoritative; nothing overrides them. Two indexes that disagree about the year of
    the same DOI are one paper with one year, not two papers."""
    merged, _ = dedupe([
        cand("Motor adaptation does not differ when a perturbation is introduced abruptly",
             year=2015, doi="10.1000/x"),
        cand("Motor adaptation does not differ when a perturbation is introduced gradually",
             year=2016, doi="doi:10.1000/X"),
    ])
    assert len(merged) == 1


@pytest.mark.parametrize("title_a,title_b", [
    # NFKD earns its keep: the ligature a typesetter left in the title
    ("Enhanced crosslimb transfer of force-ﬁeld learning",
     "Enhanced crosslimb transfer of force-field learning"),
    # …and the diacritics one index folds and another does not
    ("Prism adaptation in Parkinson's disease — a naïve cohort",
     "Prism adaptation in Parkinson's disease - a naive cohort"),
    ("Behavioural effects of aerobic exercise training",
     "Behavioral effects of aerobic exercise training"),
    ("A randomised controlled trial of treadmill training",
     "A randomized controlled trial of treadmill training"),
    ("Organisation of interlimb transfer after visuomotor adaptation",
     "Organization of interlimb transfer after visuomotor adaptation"),
])
def test_ligature_and_spelling_variants_are_one_paper(title_a, title_b):
    """These are not similar titles, they are the SAME title written by two typesetters. They merge
    on an equality after normalisation, which is why no threshold is involved and none can drift."""
    assert normalise_title(title_a) == normalise_title(title_b)
    merged, pairs = dedupe([cand(title_a, found_by=("epmc",)), cand(title_b, found_by=("openalex",))])
    assert len(merged) == 1 and pairs == []
    assert merged[0].found_by == ["epmc", "openalex"]


def test_the_same_title_in_a_different_year_is_not_merged():
    """A trial reported twice a decade apart is two reports. Year is half the identity."""
    merged, _ = dedupe([cand("Visuomotor adaptation in normal aging", year=2005),
                        cand("Visuomotor adaptation in normal aging", year=2015)])
    assert len(merged) == 2


def test_the_merged_title_is_the_publishers_not_the_archives_stamp():
    merged, _ = dedupe([
        cand("Author Manuscript Cerebellar direct current stimulation enhances motor learning "
             "in older adults", year=2016, found_by=("epmc",)),
        cand("Cerebellar direct current stimulation enhances motor learning in older adults",
             year=2016, found_by=("crossref",)),
    ])
    assert len(merged) == 1
    assert merged[0].title.startswith("Cerebellar")


# ---------------------------------------------------------------------------------------------
# normalisers
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("10.1000/ab", "10.1000/ab"),
    ("https://doi.org/10.1152/jn.00105.2021", "10.1152/jn.00105.2021"),
    ("http://dx.doi.org/10.1152/JN.00105.2021", "10.1152/jn.00105.2021"),
    ("DOI: 10.1152/jn.00105.2021", "10.1152/jn.00105.2021"),
    ("doi:10.1152/jn.00105.2021.", "10.1152/jn.00105.2021"),
    ("  10.1152/jn.00105.2021  ", "10.1152/jn.00105.2021"),
    ("10.1152%2Fjn.00105.2021", "10.1152/jn.00105.2021"),
    # not DOIs, and a normaliser that returned them would merge every record that said "n/a"
    ("n/a", ""),
    ("-", ""),
    ("PMC3141348", ""),
    ("", ""),
    ("https://europepmc.org/article/MED/21734104", ""),
])
def test_a_doi_is_normalised_or_it_is_not_a_doi(raw, expected):
    assert normalise_doi(raw) == expected


def test_a_junk_doi_field_merges_nothing():
    """Two unrelated papers whose index wrote `n/a` into the DOI field. If `normalise_doi` returned
    that string, this test would be one paper and one of these studies would be gone."""
    merged, _ = dedupe([cand("Prism adaptation changes proprioceptive localization", doi="n/a"),
                        cand("Contributions of spatial working memory to visuomotor learning",
                             doi="n/a")])
    assert len(merged) == 2


@pytest.mark.parametrize("wrapped,clean", [
    ("Author Manuscript Mechanisms underlying interlimb transfer of visuomotor rotations",
     "Mechanisms underlying interlimb transfer of visuomotor rotations"),
    ("Author manuscript How do age and nature of the motor task influence visuomotor adaptation?",
     "How do age and nature of the motor task influence visuomotor adaptation?"),
    ("Correction: Aerobic exercise in Parkinson disease", "Aerobic exercise in Parkinson disease"),
    ("Corrigendum: Aerobic exercise in Parkinson disease", "Aerobic exercise in Parkinson disease"),
    ("Erratum to: Aerobic exercise in Parkinson disease", "Aerobic exercise in Parkinson disease"),
    ("RETRACTED ARTICLE: Aerobic exercise in Parkinson disease",
     "Aerobic exercise in Parkinson disease"),
    ("Editorial: Aerobic exercise in Parkinson disease", "Aerobic exercise in Parkinson disease"),
    ("Reply to: Aerobic exercise in Parkinson disease", "Aerobic exercise in Parkinson disease"),
    ("[Aerobic exercise in Parkinson disease]", "Aerobic exercise in Parkinson disease"),
    ("[Article in German] Aerobic exercise in Parkinson disease",
     "Aerobic exercise in Parkinson disease"),
    ("Correction: Author Manuscript Aerobic exercise in Parkinson disease",
     "Aerobic exercise in Parkinson disease"),
])
def test_the_editorial_furniture_comes_off_the_front_of_a_title(wrapped, clean):
    assert normalise_title(wrapped) == normalise_title(clean)


def test_a_title_that_merely_starts_with_the_word_correction_is_left_alone():
    """`Correction of gait asymmetry…` is a paper. The prefix rule wants a delimiter for exactly
    this reason: a stripper that eats real title words is a false-merge machine."""
    assert normalise_title("Correction of gait asymmetry in Parkinson disease") == \
        "correction of gait asymmetry in parkinson disease"


@pytest.mark.parametrize("word", ["four", "hour", "tour", "your"])
def test_the_orthography_fold_leaves_short_our_words_alone(word):
    """`four` → `for` would be a fold that invents a match. The stem-length guard is what stops it."""
    assert normalise_title(f"A {word} armed study") == f"a {word} armed study"


def test_normalisation_collapses_punctuation_case_and_runs_of_space():
    assert normalise_title("  Gait,   Balance & Falls:  a  Review!  ") == "gait balance falls a review"


# ---------------------------------------------------------------------------------------------
# what a merge keeps
# ---------------------------------------------------------------------------------------------

def test_a_merge_keeps_the_richest_of_everything():
    """The richest record is routinely the one missing the venue. Merging field by field is why a
    dedupe makes the metadata better rather than merely shorter."""
    poor = cand("Adaptation to visuomotor rotations in younger and older adults", year=2019,
                found_by=("crossref",), venue="Journal of Neurophysiology",
                authors=("Bock, Otmar",))
    rich = Candidate(key=new_key("c", "rich"), title="Adaptation to Visuomotor Rotations in "
                     "Younger and Older Adults", year=2019, found_by=["openalex"],
                     authors=["Bock, Otmar", "Girgenrath, Michaela"],
                     doi="10.1152/jn.00105.2019", abstract="Older adults adapt more slowly.",
                     ids={"openalex": "W123"}, license="cc-by")
    poor.ids = {"crossref": "C9"}

    merged, _ = dedupe([poor, rich])
    assert len(merged) == 1
    one = merged[0]
    assert one.abstract == "Older adults adapt more slowly."
    assert one.authors == ["Bock, Otmar", "Girgenrath, Michaela"]
    assert one.doi == "10.1152/jn.00105.2019"
    assert one.venue == "Journal of Neurophysiology", "kept from the record that had it"
    assert one.license == "cc-by"
    assert one.ids == {"crossref": "C9", "openalex": "W123"}
    assert one.found_by == ["crossref", "openalex"]


def test_a_merge_keeps_one_key_and_the_list_never_repeats_one():
    """The key is what the page's DOM, the upload endpoint and the run all address a paper by. Two
    rows with one key is a PDF attached to the wrong study."""
    merged, _ = dedupe(_MIXED_CORPUS())
    keys = [c.key for c in merged]
    assert len(keys) == len(set(keys))
    assert all(KEY_RE.match(k) for k in keys)


def test_a_merge_never_drops_a_humans_answer_or_their_pdf():
    """Dedupe runs before screening — but if it is ever re-run on a search that has been worked on,
    it must not be the thing that throws a human's decision away."""
    plain = cand("Prism adaptation changes the subjective proprioceptive localization of the hands",
                 year=2008, found_by=("epmc",))
    worked = cand("Prism adaptation changes the subjective proprioceptive localization of the hands",
                  year=2008, found_by=("openalex",))
    worked.keep = True
    worked.state = "uploaded"
    worked.pdf_path = "pdfs/prism.pdf"
    worked.pdf_pages = 12
    worked.upload_filename = "prism.pdf"
    worked.screen_decision = "include"
    worked.screen_reason = "a prism-adaptation study in healthy adults"

    merged, _ = dedupe([plain, worked])
    assert len(merged) == 1
    one = merged[0]
    assert one.keep is True
    assert (one.state, one.pdf_path, one.pdf_pages) == ("uploaded", "pdfs/prism.pdf", 12)
    assert one.screen_decision == "include"
    assert one.screen_reason == "a prism-adaptation study in healthy adults"


def test_dedupe_does_not_mutate_what_it_was_given():
    """The caller keeps the raw index rows for the audit trail; a pass that edited them in place
    would leave `search.json` describing a search that never happened."""
    inputs = _MIXED_CORPUS()
    before = [(c.key, c.title, list(c.found_by)) for c in inputs]
    dedupe(inputs)
    assert [(c.key, c.title, list(c.found_by)) for c in inputs] == before


# ---------------------------------------------------------------------------------------------
# identity and the key derived from it
# ---------------------------------------------------------------------------------------------

def test_identity_prefers_the_doi_and_falls_back_to_title_and_year():
    assert identity(cand("Anything", year=2020, doi="DOI: 10.1000/ab")) == "doi:10.1000/ab"
    assert identity(cand("Visuomotor Adaptation in Normal Aging", year=2005)) == \
        "t:visuomotor adaptation in normal aging|2005"


def test_the_key_is_stable_across_two_searches_of_the_same_question():
    """The key exists so a PDF a user attached in one search binds to the same paper in the next.
    That only holds if it is derived from the paper's identity, not from the row's arrival."""
    first = cand("Author Manuscript Visuomotor Adaptation in Normal Aging", year=2005,
                 found_by=("epmc",))
    later = cand("Visuomotor adaptation in normal aging", year=2005, found_by=("openalex",))
    assert key_for(first) == key_for(later)
    assert KEY_RE.match(key_for(first))


# ---------------------------------------------------------------------------------------------
# a whole small corpus at once
# ---------------------------------------------------------------------------------------------

def _MIXED_CORPUS() -> list[Candidate]:
    """Nine rows from three indexes: two DOI duplicates, two prefix duplicates, one proposable
    pair, and four papers that are simply themselves."""
    return [
        cand("Older adults learn less, but still reduce metabolic cost, during motor adaptation",
             year=2021, doi="10.1152/jn.00105.2021", found_by=("epmc",)),
        cand("Older adults learn less but still reduce metabolic cost during motor adaptation",
             year=2021, doi="https://doi.org/10.1152/JN.00105.2021", found_by=("openalex",)),
        cand("Author Manuscript Mechanisms underlying interlimb transfer of visuomotor rotations",
             year=2010, found_by=("epmc",)),
        cand("Mechanisms underlying interlimb transfer of visuomotor rotations", year=2010,
             found_by=("crossref",)),
        cand("Effects of resistance training on tremor in Parkinson disease: a randomised trial",
             year=2017, found_by=("epmc",)),
        cand("Effects of resistance training on gait in Parkinson disease: a randomised trial",
             year=2017, found_by=("openalex",)),
        cand("Gait in Parkinson disease. Part I. Kinematic analysis", year=2012,
             found_by=("epmc",)),
        cand("Gait in Parkinson disease. Part II. Kinematic analysis", year=2012,
             found_by=("epmc",)),
        cand("Low-dose levodopa for tremor in Parkinson disease", year=2014, found_by=("epmc",)),
    ]


def test_the_whole_corpus_lands_where_the_review_said_it_should():
    merged, pairs = dedupe(_MIXED_CORPUS())
    assert len(merged) == 7, "two DOI duplicates and two prefix duplicates collapsed, nothing else"
    assert len(pairs) == 1, "only the tremor/gait pair is close enough to ask about"
    titles = {c.title for c in merged}
    assert "Gait in Parkinson disease. Part I. Kinematic analysis" in titles
    assert "Gait in Parkinson disease. Part II. Kinematic analysis" in titles
    assert "Low-dose levodopa for tremor in Parkinson disease" in titles


def test_the_prisma_records_count_still_adds_up_after_a_merge():
    """`counts_of` derives `records` from `found_by`, so a merge that lost an index name would
    under-report how many rows the indexes actually returned."""
    corpus = _MIXED_CORPUS()
    merged, pairs = dedupe(corpus)
    counts = counts_of(merged, pairs)
    assert counts["records"] == len(corpus)
    assert counts["after_dedupe"] == len(merged)


def test_an_empty_search_is_not_a_crash():
    assert dedupe([]) == ([], [])


def test_the_discriminator_vocabulary_is_the_one_the_review_named():
    """A pin, not a tautology: these ten words are the amendment, and quietly dropping one of them
    is how `Low-dose` and `High-dose` become one study again."""
    assert {"part", "trial", "protocol", "low", "high", "dose", "follow", "up", "pilot",
            "extension"} <= DISCRIMINATOR_WORDS
