"""Task 5 (half 2): the text/table extractors.

Two halves, like the mapper's tests:
  * offline `FakeProvider` unit tests for everything the extractor decides in *code* — context
    building, parsing, the null-numbers envelope, grounding, page correction, the group-label and
    dispersion-type cross-checks;
  * replayed calls on Bock 2005 (fixtures in `tests/fixtures/llm`), which record what the two
    variants really answer for a paper that prints its outcome only in a figure.

Bock 2005 prints no group mean for either protocol outcome of this contrast: the pointing errors
live only in Fig. 1, and the text gives a fitted curve and ANOVAs instead. (Its other "M ± SD"
values are participant ages, the screening test, a speed-normalised screening score, and one index
from a pooled 54-participant sample built out of two further experiments — none of them this
contrast's outcome.) So the honest answer is `not_on_these_pages`, and that is what the replay
asserts. The screening test *is* printed as "M ± SD" for both groups, so a second replay points the
same extractor at it with a protocol that asks for it: same code path, a real "found" answer, real
numbers, a real grounded quote.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from canopy.agents import load_prompt
from canopy.agents.extract_text import (EXTRACT_TEXT_SCHEMA, PROMPT_VERSION, VARIANTS,
                                        extract_group_stats)
from canopy.agents.mapper import map_study
from canopy.config import MODELS, live_enabled, load_env, record_enabled
from canopy.ingest.pdf import PaperRecord, ingest_pdf
from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import (Candidate, DatasetSpec, DispersionType, GroupSpec, OutcomeDef,
                           Protocol, Source, SourceKind, StudyMap)
from canopy.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf"
REPLAY = ROOT / "tests" / "fixtures" / "llm"
PROTOCOL_PATH = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"

replayed = pytest.mark.replay


@pytest.fixture(scope="session")
def paper(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF, tmp_path_factory.mktemp("bock2005-extract"))


@pytest.fixture(scope="session")
def protocol() -> Protocol:
    return load_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="session")
def client() -> LLMClient:
    live, record = live_enabled(), record_enabled()          # read here, not at import
    if live:
        load_env()
    return LLMClient(replay_dir=REPLAY, record_dir=REPLAY if record else None, allow_live=live,
                     cache_dir=None)


@pytest.fixture(scope="session")
def bock_map(client, paper, protocol) -> StudyMap:
    """The mapper's real map of Bock 2005 — the extractors read the locations it found."""
    return map_study(client, paper, protocol)


@pytest.fixture(scope="session")
def dataset(bock_map):
    """d1: the pointing experiment, twelve older vs twelve younger participants."""
    return bock_map.datasets[0]


#: a protocol whose outcome is the one quantity Bock prints as "M ± SD" for both groups. Domain
#: knowledge belongs in a protocol, which is exactly what this is.
SCREENING = OutcomeDef(
    key="screening_time",
    label="Screening test completion time",
    definition="Time to complete the paper-and-pencil screening test in which participants join "
               "numbered dots in ascending order.",
    measurement_window="The single administration of the screening test.",
    higher_is_better_hint="A longer completion time is worse performance.",
    units_hint="seconds",
)


@pytest.fixture(scope="session")
def screening_protocol(protocol) -> Protocol:
    return protocol.model_copy(update={"outcomes": [SCREENING]})


SCREENING_SOURCE = Source(
    kind=SourceKind.text_mean_sd, page=3, locator="Results, screening-test paragraph",
    quote="In the trail-making test, the completion time for young subjects was",
    analysis_metric="endpoint")


# ------------------------------------------------------------------ schema and prompt hygiene
def test_schema_is_valid_and_carries_no_derived_stats():
    assert_valid_output_schema(EXTRACT_TEXT_SCHEMA, "EXTRACT_TEXT_SCHEMA")
    assert_no_derived_stats(EXTRACT_TEXT_SCHEMA, name="EXTRACT_TEXT_SCHEMA")


def test_the_schema_stays_far_under_the_grammar_limit():
    """Task 4 found the API rejects a schema over ~55 leaf properties."""
    leaves = EXTRACT_TEXT_SCHEMA["properties"]["groups"]["items"]["properties"]
    assert len(leaves) + 1 <= 35


def test_schema_fields_exist_on_the_candidate_model():
    leaves = set(EXTRACT_TEXT_SCHEMA["properties"]["groups"]["items"]["properties"])
    extra = {"group_label_as_written"}                    # checked in code, not stored verbatim
    assert leaves - extra <= set(Candidate.model_fields) | {"kind"}


def test_prompts_carry_no_domain_knowledge():
    """The protocol supplies the domain; the prompts must work for any review."""
    banned = ["older adult", "younger adult", "visuomotor", "adaptation", "aftereffect",
              "ageing", "aging", "elderly", "perturbation", "sensorimotor"]
    for settings in VARIANTS.values():
        text = load_prompt(settings["prompt"]).lower()
        assert not [word for word in banned if word in text], settings["prompt"]


def test_both_variants_answer_the_same_contract():
    """Two voters may differ in search order, never in what the answer means."""
    first, second = (load_prompt(v["prompt"]) for v in VARIANTS.values())
    rules = lambda text: text.split("# Rules")[1].split("# The outcome")[0]      # noqa: E731
    assert rules(first) == rules(second)
    assert first.split("# Where to look")[1] != second.split("# Where to look")[1]


def test_prompt_version_tracks_the_prompt_files(monkeypatch):
    from canopy.agents import extract_common

    assert PROMPT_VERSION.startswith("extract_text/1@")
    assert len(PROMPT_VERSION) == len("extract_text/1@") + 8
    original = extract_common.load_prompt
    monkeypatch.setattr(extract_common, "load_prompt", lambda name: original(name) + "\nedited")
    assert extract_common.prompt_fingerprint(("extract_table_first",)) != PROMPT_VERSION[-8:]


# ------------------------------------------------------------------ offline unit tests
#: the offline tests build their own contrast: they must never pull in the replayed mapper
#: fixtures, because a session client created by a non-replay test cannot record.
@pytest.fixture
def fake_dataset() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="b511dbb76fa6:d1", label="pointing", cluster_id="b511dbb76fa6",
        group_a=GroupSpec(label="old (older) subjects", n=12, n_evidence="twelve old subjects"),
        group_b=GroupSpec(label="young subjects", n=12, n_evidence="twelve young subjects"))


def _row(**over):
    row = {"group": "A", "group_label_as_written": "old (older) subjects", "status": "found",
           "page": 3, "kind": "text_mean_sd",
           "quote": "the completion time for young subjects was 27.4±7.2 s and that for old "
                    "subjects was 42.5±6.9 s",
           "row_header": "", "col_header": "", "value_as_written": "42.5±6.9 s", "mean": 42.5,
           "dispersion_value": 6.9, "dispersion_type": "SD", "unit": "s", "n": 12,
           "n_quote": "twelve old subjects", "raw_value_semantics": "higher_more_error",
           "analysis_metric": "endpoint", "error_bar_scope": "between_subject", "notes": ""}
    row.update(over)
    return row


def _payload(rows=None, notes=""):
    rows = [_row(), _row(group="B", group_label_as_written="young subjects",
                         value_as_written="27.4±7.2 s", mean=27.4, dispersion_value=7.2,
                         n_quote="twelve young subjects")] if rows is None else rows
    return {"groups": [dict(r) for r in rows], "notes": notes}


TEXT_SOURCE = Source(kind=SourceKind.text_mean_sd, page=3, locator="Results, second paragraph",
                     quote="the completion time for young subjects was")


def _extract(paper, protocol, fake_dataset, payload, *, sources=(TEXT_SOURCE,),
             variant="table_first", outcome_key="late_adaptation", model=None):
    provider = FakeProvider([payload])
    client = LLMClient(provider=provider, cache_dir=None)
    candidates = extract_group_stats(client, paper, protocol, fake_dataset, outcome_key,
                                     list(sources), variant=variant, model=model)
    return candidates, provider


def test_two_rows_become_two_grounded_candidates(paper, protocol, fake_dataset):
    cands, provider = _extract(paper, protocol, fake_dataset, _payload())
    assert len(provider.requests) == 1
    assert [c.group for c in cands] == ["A", "B"]
    a, b = cands
    assert (a.mean, a.dispersion_value, a.unit) == (42.5, 6.9, "s")
    assert (b.mean, b.dispersion_value) == (27.4, 7.2)
    assert a.dispersion_type is DispersionType.SD and a.status == "found"
    assert a.grounded is True and a.grounding_similarity == 1.0 and a.page == 3
    assert a.n == 12 and a.n_quote == "twelve old subjects"
    assert a.analysis_metric == "endpoint" and a.raw_value_semantics == "higher_more_error"
    assert a.error_bar_scope == "between_subject"


def test_code_fills_the_bookkeeping(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, _payload())
    a = cands[0]
    assert a.paper_id == paper.sha256 and a.dataset_id == fake_dataset.dataset_id
    assert a.outcome_key == "late_adaptation" and a.kind == "group_stats"
    assert a.route == "text" and a.model == MODELS["primary"]
    assert a.extractor_id == f"text:table_first:{MODELS['primary']}"
    assert a.prompt_version == PROMPT_VERSION and a.llm_call_id
    assert a.source_kind is SourceKind.text_mean_sd
    assert a.locator == "Results, second paragraph"           # the location it was sent to
    assert len({c.candidate_id for c in cands}) == len(cands)
    assert fake_dataset.dataset_id in a.candidate_id and "late_adaptation" in a.candidate_id


def test_each_variant_has_its_own_prompt_model_and_effort(paper, protocol, fake_dataset):
    _, first = _extract(paper, protocol, fake_dataset, _payload(), variant="table_first")
    _, second = _extract(paper, protocol, fake_dataset, _payload(), variant="narrative_first")
    assert first.requests[0].model == MODELS["primary"] and first.requests[0].effort == "high"
    assert second.requests[0].model == MODELS["secondary"] and second.requests[0].effort == "medium"
    prompts = [r.messages[0]["content"][-1]["text"]
               for r in (first.requests[0], second.requests[0])]
    assert prompts[0] != prompts[1]
    assert prompts[0].startswith(load_prompt("extract_table_first")[:60])
    assert prompts[1].startswith(load_prompt("extract_narrative_first")[:60])
    assert first.requests[0].key != second.requests[0].key


def test_an_unknown_variant_is_a_programming_error(paper, protocol, fake_dataset):
    with pytest.raises(ValueError, match="unknown variant"):
        _extract(paper, protocol, fake_dataset, _payload(), variant="figure_first")


def test_a_caller_can_override_the_model(paper, protocol, fake_dataset):
    cands, provider = _extract(paper, protocol, fake_dataset, _payload(), model=MODELS["secondary"])
    assert provider.requests[0].model == MODELS["secondary"]
    assert cands[0].extractor_id == f"text:table_first:{MODELS['secondary']}"


def test_a_text_source_gets_text_only_context(paper, protocol, fake_dataset):
    """Amendment E: a sentence is already in the text layer; a page raster only adds cost."""
    _, provider = _extract(paper, protocol, fake_dataset, _payload())
    blocks = provider.requests[0].messages[0]["content"]
    assert not [b for b in blocks if b.get("type") == "image"]
    assert "[page 3 text]" in blocks[0]["text"]
    prompt = blocks[-1]["text"]
    assert "Results, second paragraph" in prompt                      # the mapper's location
    assert "old (older) subjects" in prompt and "analysed n = 12" in prompt
    assert "late_adaptation" in prompt and "{{" not in prompt


def test_only_the_pages_the_sources_name_are_sent(paper, protocol, fake_dataset):
    sources = (TEXT_SOURCE, Source(kind=SourceKind.text_mean_sd, page=1, locator="Abstract"))
    _, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=sources)
    texts = [b.get("text", "") for b in provider.requests[0].messages[0]["content"]]
    assert [t for t in texts if t.startswith("[page 1 text]")]
    assert [t for t in texts if t.startswith("[page 3 text]")]
    assert not [t for t in texts if t.startswith("[page 4 text]")]


def test_a_table_source_also_gets_the_page_image_and_the_cells(paper, protocol, fake_dataset):
    table = Source(kind=SourceKind.table, page=1, locator="Table 1, row 'old'", table_id="p1t1")
    _, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=(table,))
    blocks = provider.requests[0].messages[0]["content"]
    assert [b for b in blocks if b.get("type") == "image"]
    assert [b for b in blocks if b.get("type") == "text" and "[table p1t1 on page 1]" in b["text"]]


def test_a_source_kind_this_extractor_cannot_read_is_left_alone(paper, protocol, fake_dataset):
    figure = Source(kind=SourceKind.figure_line, page=3, locator="Fig 1", figure_id="fig01")
    cands, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=(figure,))
    assert cands == [] and provider.requests == []            # no call, no cost


def test_a_source_pointing_outside_the_document_is_dropped(paper, protocol, fake_dataset):
    stray = Source(kind=SourceKind.text_mean_sd, page=99, locator="page that does not exist")
    cands, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=(stray,))
    assert cands == [] and provider.requests == []


def test_a_number_of_an_unlisted_kind_is_still_attempted(paper, protocol, fake_dataset):
    """`SourceKind.unknown` (a fitted parameter, say) is read as text, not skipped."""
    unknown = Source(kind=SourceKind.unknown, page=3, locator="Results, exponential fit",
                     quote="An exponential fit yielded y=9.92+53.51×exp(−x/5.89)")
    cands, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=(unknown,))
    assert len(provider.requests) == 1 and len(cands) == 2


def test_a_status_that_is_not_found_carries_no_numbers(paper, protocol, fake_dataset):
    rows = [_row(status="not_on_these_pages", quote="", mean=None, dispersion_value=None,
                 value_as_written="", notes="the page shows this only in a figure"),
            _row(group="B", status="ambiguous", mean=27.4, dispersion_value=7.2)]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    a, b = cands
    assert a.status == "not_on_these_pages" and a.mean is None and a.grounded is None
    assert "only in a figure" in a.notes
    assert b.status == "ambiguous" and b.mean is None and b.dispersion_value is None
    assert "dropped because status is ambiguous" in b.notes and "27.4" in b.notes


def test_a_quote_the_paper_never_printed_is_marked_ungrounded(paper, protocol, fake_dataset):
    rows = [_row(quote="the old group averaged 42.0 ± 3.1 deg over the last two episodes")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].grounded is False
    assert (cands[0].grounding_similarity or 0) < 0.95
    assert "not found" in cands[0].notes
    assert cands[0].mean == 42.5                     # kept: Task 8 decides, the extractor reports


def test_the_page_is_corrected_when_the_quote_is_one_page_away(paper, protocol, fake_dataset):
    rows = [_row(page=2)]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].grounded is True and cands[0].page == 3 and cands[0].page_corrected is True


def test_a_table_cell_candidate_keeps_its_row_and_column_headers(paper, protocol, fake_dataset):
    rows = [_row(kind="table", row_header="Trail making (s)", col_header="Old")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].row_header == "Trail making (s)" and cands[0].col_header == "Old"
    assert cands[0].source_kind is SourceKind.table
    assert "table cell check skipped" in cands[0].notes       # Bock has no such table row


def test_the_extractors_own_dispersion_reading_governs_and_the_conflict_is_noted(
        paper, protocol, fake_dataset):
    """Text sources are exempt from the mapper's error-bar agreement rule (Task 4)."""
    hinted = Source(kind=SourceKind.text_mean_sd, page=3, locator="Results, second paragraph",
                    error_bar_type=DispersionType.SE)
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(), sources=(hinted,))
    assert cands[0].dispersion_type is DispersionType.SD              # our reading survives
    assert "dispersion type disagreement" in cands[0].notes
    assert "SD" in cands[0].notes and "SE" in cands[0].notes


def test_a_group_label_that_belongs_to_the_other_group_is_flagged(paper, protocol, fake_dataset):
    rows = [_row(group="A", group_label_as_written=fake_dataset.group_b.label)]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].group == "A"                       # never silently swapped
    assert "group label mismatch" in cands[0].notes and "group B" in cands[0].notes


def test_a_shorter_wording_of_the_same_group_is_not_flagged(paper, protocol, fake_dataset):
    """The first live run flagged 'old subjects' against 'old (older) subjects': only the words
    that tell the two groups apart may decide, never whole-string similarity."""
    rows = [_row(group="A", group_label_as_written="old subjects"),
            _row(group="B", group_label_as_written="the young group")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert not [c for c in cands if "group label mismatch" in c.notes], [c.notes for c in cands]
    assert "group label as written: 'old subjects'" in cands[0].notes


def test_a_label_that_names_neither_group_is_not_flagged(paper, protocol, fake_dataset):
    rows = [_row(group="A", group_label_as_written="all participants")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert "group label mismatch" not in cands[0].notes


def test_a_row_without_a_group_is_kept_and_flagged(paper, protocol, fake_dataset):
    rows = [_row(group="unknown")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    unassigned = [c for c in cands if c.group is None]
    assert len(unassigned) == 1 and "did not say which group" in unassigned[0].notes
    assert {c.group for c in cands} == {None, "A", "B"}          # both groups still answered


def test_a_group_the_extractor_skipped_becomes_an_explicit_empty_answer(paper, protocol,
                                                                        fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=[_row()]))
    assert [c.group for c in cands] == ["A", "B"]
    assert cands[1].status == "not_on_these_pages" and cands[1].mean is None
    assert "returned no row for this group" in cands[1].notes
    assert cands[1].llm_call_id == cands[0].llm_call_id


def test_overall_notes_travel_with_every_candidate(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(notes="the table gives no units"))
    assert all("the table gives no units" in c.notes for c in cands)


def test_an_empty_answer_still_reports_both_groups(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, {"groups": [], "notes": "nothing here"})
    assert [(c.group, c.status) for c in cands] == [("A", "not_on_these_pages"),
                                                    ("B", "not_on_these_pages")]


# ------------------------------------------------------------------ replayed Bock 2005
def _text_sources(dataset, outcome_key):
    from canopy.agents.extract_common import TEXT_SOURCE_KINDS

    return [s for s in dataset.outcome(outcome_key).sources if s.kind in TEXT_SOURCE_KINDS]


@replayed
@pytest.mark.parametrize("variant", ["table_first", "narrative_first"])
def test_bock_prints_no_group_means_for_its_outcome_and_the_extractor_says_so(
        client, paper, protocol, dataset, variant):
    """Bock reports the pointing error only in Fig. 1; the text has a fitted curve and ANOVAs.

    An extractor that answered anything else here would be inventing the meta-analysis's inputs.
    """
    sources = _text_sources(dataset, "late_adaptation")
    assert sources, "the mapper found no text-like source for late_adaptation"
    cands = extract_group_stats(client, paper, protocol, dataset, "late_adaptation", sources,
                                variant=variant)
    assert [c.group for c in cands] == ["A", "B"]
    for cand in cands:
        assert cand.status == "not_on_these_pages", (cand.group, cand.status, cand.notes)
        assert cand.mean is None and cand.dispersion_value is None
        assert not cand.value_as_written                        # not even the fitted asymptote
        assert cand.notes.strip()                               # it said what the pages do carry
        assert cand.grounded in (None, True), cand.quote        # no quote, or a real one
        assert cand.route == "text" and cand.prompt_version == PROMPT_VERSION


@replayed
@pytest.mark.parametrize("variant", ["table_first", "narrative_first"])
def test_bock_screening_times_are_transcribed_with_a_grounded_quote(
        client, paper, screening_protocol, dataset, variant):
    """The one "M ± SD" for both groups in Bock's text: 27.4±7.2 s young, 42.5±6.9 s old."""
    cands = extract_group_stats(client, paper, screening_protocol, dataset, "screening_time",
                                [SCREENING_SOURCE], variant=variant)
    by_group = {c.group: c for c in cands}
    assert set(by_group) == {"A", "B"}
    older, younger = by_group["A"], by_group["B"]
    assert older.status == "found" and younger.status == "found"
    assert (older.mean, older.dispersion_value) == (42.5, 6.9)
    assert (younger.mean, younger.dispersion_value) == (27.4, 7.2)
    assert older.unit.startswith("s") and younger.unit.startswith("s")
    for cand in cands:
        assert cand.grounded is True and (cand.grounding_similarity or 0) >= 0.95
        assert cand.page == 3 and cand.page_corrected is False
        assert "completion time" in cand.quote
        # the paper never says whether its ± is a deviation or an error, so an honest reader
        # answers UNKNOWN (or SD, from the figure legend on the same page) — never SE
        assert cand.dispersion_type in (DispersionType.UNKNOWN, DispersionType.SD), cand.notes
