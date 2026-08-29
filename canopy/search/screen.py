"""Title/abstract screening: which of the papers an index proposed are worth reading in full.

This is the one stage of a search that spends money per record, and the one whose mistakes are
invisible afterwards. A paper wrongly EXCLUDED here is never seen again — it does not appear in the
run, in the forest plot, or in the funnel; nothing downstream can audit it back. A paper wrongly
INCLUDED costs one PDF and one row a human unticks. The whole module is built around that
asymmetry: it is deliberately over-inclusive, it never defaults to `exclude`, and every decision it
writes carries the screener's own sentence beside it (`Candidate.screen_reason`, models.py:92).

Four rules, each of which is a test:

* **The reason is the model's sentence, verbatim.** Never paraphrased, never summarised, never
  dropped. It is the audit trail a reviewer defends in front of an editor. Where this module has
  something of its own to say — the abstract was missing, the word the model used was not one of
  ours — it *appends* a parenthetical and leaves the model's words untouched. Annotate, never
  subtract.
* **`unknown`, not `unsure`.** `assert_valid_output_schema` requires every string enum to offer an
  escape hatch from `UNKNOWN_ENUM_MEMBERS = {"unknown", "not_reported", "none", "ambiguous"}`
  (llm/schemas.py:88, enforced at :117-122) and `client.structured` runs that audit before a byte is
  sent (llm/client.py:368-370). The original design's `["include", "exclude", "unsure"]` therefore
  raised `AssertionError` on every call — screening could not run at all (review §B-BLK1). The
  schema says `unknown`; the page shows the word "unsure"; `counts_of` already maps
  `screen_decision == "unknown"` to the `unsure` count (models.py:233).
* **Nobody is dropped in silence.** A record the model returned no verdict for, a record the cost
  cap was reached before, a record screened with no model at all — each keeps `state
  "not_screened"`, an empty `screen_decision` (so `counts_of` does not count it as screened) and a
  reason saying, in words, that nobody read it.
* **The batch composition is pinned.** Candidates are sorted by `key` and cut into fixed batches, so
  the same paper set always builds byte-identical prompts and hits the disk cache at $0
  (client.py:511-513) rather than re-batching in a different order and paying a second full bill
  (review §D3). Two honest limits on that, because this docstring used to claim more than the code
  did: there is **no resume endpoint** — a search is screened once, in `_start` — and the cache
  lives inside the search's own directory (`jobs.py:75`), so pressing **"Search again" pays the
  full screening bill again** even for byte-identical prompts. The batch map is written into
  `search.json` (`SearchRecord.batches`) as an audit trail, not as a resume key.

**Model role.** The design asks for `MODELS["screener"] = "claude-sonnet-5"`; `canopy/config.py` has
no such role and this task may not add one, so callers pass `MODELS["secondary"]` — which *is*
`claude-sonnet-5`, the same model the design chose. Adding the named role is a follow-up, not a
behaviour change. Not haiku: it saves ~$0.12 on a 200-record search and it is the one model that
rejects `output_config.effort` (llm/providers.py:19-24), so the cheap model is also the one that
cannot be dialled down — a poor trade against a permanent, unauditable `exclude`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.costs import estimate_request_cost
from ..llm.errors import BudgetExceeded, LLMError
from .models import Candidate

__all__ = ["SCREEN_SCHEMA", "SCREEN_DECISIONS", "PROMPT_VERSION", "SYSTEM", "BATCH",
           "MAX_ABSTRACT_CHARS", "MAX_TOKENS", "EFFORT", "ScreenBatch", "ScreenOutcome",
           "screen_candidates", "NO_MODEL_REASON", "NO_VERDICT_REASON", "TITLE_ONLY_NOTE",
           "NO_ABSTRACT_MARKER", "TRUNCATION_MARKER", "cap_reason", "batches_of"]

PROMPT_VERSION = "search-screen-1"

#: the three answers, and the only three. `unknown` is the schema's word for what the page calls
#: "unsure" — see the module docstring and review §B-BLK1.
SCREEN_DECISIONS: tuple[str, ...] = ("include", "exclude", "unknown")

#: records per call. Big enough that the criteria preamble (~700 tokens) is amortised twenty ways;
#: small enough that one refusal or one truncation costs twenty decisions and not two hundred, and
#: that the cost cap is re-checked ten times over a 200-record search.
BATCH = 20

#: characters of abstract sent per record. Publisher abstracts routinely run past this (the Europe
#: PMC record sampled while designing this had 2,140 characters), and the tail of an abstract is
#: methods detail that a title/abstract screen does not turn on. This limit is *load-bearing for the
#: cost estimate* — see the cost note on `screen_candidates`.
MAX_ABSTRACT_CHARS = 1800

#: output budget per batch. Twenty verdicts of one sentence each is ~1,200 tokens; the headroom is
#: for a model that writes longer reasons rather than for one that writes more of them.
MAX_TOKENS = 3000

#: reading a title and an abstract against written criteria is a reading task, not a reasoning task.
EFFORT = "low"

#: what this module says when it, and not the model, is the author of a reason.
NO_MODEL_REASON = "no model was available to screen this record"
NO_VERDICT_REASON = ("the screener returned no verdict for this record — nobody read it, so it is "
                     "listed unscreened rather than excluded")
TITLE_ONLY_NOTE = "(screened on the title alone: this record had no abstract.)"
NO_ABSTRACT_MARKER = "[no abstract was available for this record — screen it on its title alone]"
TRUNCATION_MARKER = " […abstract truncated]"

#: `state` after a verdict, when no PDF is on disk yet. An `include` becomes `wanted`, NOT
#: `paywalled`: the fetch stage has not run, so nothing has established that a paywall exists,
#: and telling a user their paper sits behind one is a claim about a publisher nobody contacted.
#: `fetch` overwrites this with `fetched`, or with the `paywalled` it actually measured — and a
#: search that stops before fetching leaves rows reading "wanted", which is what is true.
_STATE_FOR: dict[str, str] = {"include": "wanted", "exclude": "excluded", "unknown": "unsure"}

#: `include` and `unknown` default to ON, `exclude` to OFF (design §2.6). Screening is
#: over-inclusive by construction: the run's own mapper decides eligibility per paper and records
#: it, so an `unknown` that reaches the run is corrected there, while an `exclude` that should not
#: have been is corrected nowhere.
_KEEP_FOR: dict[str, bool] = {"include": True, "unknown": True, "exclude": False}


# --------------------------------------------------------------------------------- the schema
#: One batch verdict. Strict objects, every property required, no derived statistic anywhere — the
#: house rules `assert_valid_output_schema` / `assert_no_derived_stats` enforce (llm/schemas.py).
#: `tests/test_search_screen.py` runs both audits against this constant, because a schema that the
#: client rejects is a stage that cannot run at all.
SCREEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "description": "One entry per record you were shown, in any order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "decision", "reason"],
                "properties": {
                    "ref": {"type": "string",
                            "description": "The record's number, exactly as it was given to you."},
                    "decision": {
                        "type": "string",
                        "enum": list(SCREEN_DECISIONS),
                        "description": "include = read this paper in full; exclude = it fails a "
                                       "named criterion; unknown = the title and abstract cannot "
                                       "tell you.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence naming the criterion this record meets or "
                                       "fails, in the words of its own title and abstract. Never "
                                       "empty: this sentence is shown to the reviewer as the "
                                       "justification for the decision.",
                    },
                },
            },
        },
    },
}

SYSTEM = (
    "You screen titles and abstracts for a systematic review: for each numbered record you decide "
    "whether it should be read in full. You never invent a fact about a record you were not shown. "
    "If a record has no abstract, say so in your reason and answer `unknown` unless the title alone "
    "settles it. `include` means it meets the criteria; `exclude` means it fails a criterion you "
    "can name; `unknown` means the title and abstract cannot tell you. Screen over-inclusively: a "
    "paper you wrongly exclude here is never looked at again, while a paper you wrongly include "
    "costs one download and one click. Every decision carries one sentence naming the criterion it "
    "meets or fails, in the words of the record itself."
)

PROMPT = """The review asks:

    {QUESTION}

A paper is eligible when it meets these criteria:

{CRITERIA}

Screen the {N} records below. Answer once per record, using its number as `ref`, and give every
answer its one-sentence reason.

{RECORDS}"""


# --------------------------------------------------------------------------------- outcome shapes
@dataclass
class ScreenBatch:
    """One batch, sent or not — the pinned composition plus what it cost.

    `keys` is the audit trail: `run.py` writes these rows into `search.json` so a reader can price
    screening per batch and see which twenty records one failed call cost. It is NOT a resume key —
    there is no resume, and the module docstring says what "Search again" actually costs.
    """

    index: int
    keys: list[str]
    sent: bool = False
    #: what the cap was checked against, before the call — the WORST case, which is this call plus
    #: the one `client.structured` makes if the model runs out of output room. See
    #: `screen_candidates`; reserving only the first call let one batch bill 2.6× its reservation.
    estimated_usd: float = 0.0
    cost_usd: float = 0.0            # what it actually cost
    n_verdicts: int = 0              # verdicts that landed on a candidate in this batch
    error: str = ""                  # "" unless the call failed; the failure's own words


@dataclass
class ScreenOutcome:
    """What screening did. The candidates themselves carry every decision; this is the summary.

    `stopped_because` mirrors `SearchRecord.stopped_because` (models.py:207): `""` when screening
    ran to the end of the list, `"budget"` when the cap stopped it. A capped search still finishes
    `done`, so a page keying its banner on the job status alone would never tell the user their
    search was cut short.
    """

    batches: list[ScreenBatch] = field(default_factory=list)
    n_screened: int = 0              # candidates a model actually returned a verdict for
    cost_usd: float = 0.0
    stopped_because: str = ""        # "" | "budget"
    notes: list[str] = field(default_factory=list)


def _money(usd: float) -> str:
    """`$1.00`, but `$0.0010` for a cap small enough that two decimal places would print `$0.00`."""
    return f"${usd:.2f}" if usd >= 0.01 else f"${usd:.4f}"


def cap_reason(budget_usd: float | None) -> str:
    """The sentence a record gets when the cap stopped the search before anyone read it.

    It names the cap that actually fired — the caller's `budget_usd` when this module's own
    between-batch check stopped things, the client's when its reservation refused the call — because
    a reader asking "why was this paper never screened?" needs the number they can change.
    """
    if budget_usd is None:
        return "the search's cost cap was reached before this record was screened"
    return f"the search's {_money(budget_usd)} cost cap was reached before this record was screened"


# --------------------------------------------------------------------------------- small helpers
def batches_of(candidates: Sequence[Candidate], size: int = BATCH) -> list[list[Candidate]]:
    """The pinned composition: sorted by `key`, then cut into fixed runs of `size`.

    Sorted rather than left in arrival order because arrival order depends on which index answered
    first, which is a network fact, not a property of the paper set — and a resumed search that
    re-batches in a different order misses every cache entry and bills the user a second time with
    no warning (review §D3).
    """
    if size < 1:
        raise ValueError(f"a screening batch holds at least one record, not {size}")
    ordered = sorted(candidates, key=lambda c: c.key)
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def _truncate_abstract(abstract: str, limit: int = MAX_ABSTRACT_CHARS) -> str:
    """Whitespace-collapsed abstract, cut to `limit` characters with a visible marker.

    The marker matters: a model shown a sentence that stops mid-clause with no explanation may treat
    the missing half as absent from the paper, which is exactly the invented fact the system prompt
    forbids.
    """
    text = " ".join((abstract or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_MARKER


def _record_block(index: int, candidate: Candidate) -> str:
    """One numbered record, exactly as the model sees it."""
    head = f"{index}. {candidate.title.strip() or '[no title]'}"
    meta = ", ".join(x for x in (str(candidate.year) if candidate.year else "",
                                 candidate.venue.strip()) if x)
    if meta:
        head = f"{head} ({meta})"
    body = _truncate_abstract(candidate.abstract) or NO_ABSTRACT_MARKER
    return f"{head}\n{body}"


def _prompt_for(batch: Sequence[Candidate], *, question: str,
                criteria: Sequence[str]) -> str:
    """The user message for one batch. Deterministic, because the disk cache keys on it."""
    bullets = "\n".join(f"* {str(c).strip()}" for c in criteria if str(c).strip())
    if not bullets:
        bullets = "* (no criteria were written down; judge each record against the question itself.)"
    records = "\n\n".join(_record_block(i + 1, c) for i, c in enumerate(batch))
    return PROMPT.format(QUESTION=question.strip() or "(no question was given)",
                         CRITERIA=bullets, N=len(batch), RECORDS=records)


def _normalise(decision: Any) -> tuple[str, str]:
    """`(decision, note)` — one of `SCREEN_DECISIONS`, plus what to append when it was not.

    A strict schema should make this dead code, but a model that answers "unsure" (the word the page
    shows) or a provider that relaxes the enum must not silently become an `exclude`, and must not
    land a word `counts_of` cannot count. Anything unrecognised becomes `unknown` — the answer that
    keeps the paper visible — and says so.
    """
    word = str(decision or "").strip().lower()
    if word in SCREEN_DECISIONS:
        return word, ""
    return "unknown", (f'(the screener answered "{decision}", which is not one of '
                       f'{"/".join(SCREEN_DECISIONS)}; it is recorded as unknown.)')


def _reason_for(model_reason: Any, *, title_only: bool, note: str) -> str:
    """The model's sentence verbatim, plus our own parentheticals — never instead of it."""
    parts = [str(model_reason or "").strip()]
    if not parts[0]:
        parts = ["the screener gave no reason for this decision"]
    if title_only:
        parts.append(TITLE_ONLY_NOTE)
    if note:
        parts.append(note)
    return " ".join(parts)


def _write_verdict(candidate: Candidate, *, decision: str, reason: str, title_only: bool) -> None:
    """Put one verdict on one candidate."""
    candidate.screen_decision = decision
    candidate.screen_reason = reason
    candidate.screened_on_title_only = title_only
    candidate.keep = _KEEP_FOR[decision]
    # A candidate that already has a PDF on disk keeps the state the file gave it: `fetched`,
    # `uploaded` and `extra` are facts about the filesystem, and a screener's opinion does not
    # un-fetch a paper. The decision, the reason and `keep` are still recorded, so an excluded
    # paper whose PDF a human attached still shows why the screener did not want it.
    if not candidate.pdf_path:
        candidate.state = _STATE_FOR[decision]


def _write_unscreened(candidate: Candidate, reason: str) -> None:
    """Mark a candidate nobody read — with the reason nobody read it.

    `screen_decision` stays `""` on purpose: `counts_of` counts a candidate as *screened* exactly
    when that field is non-empty (models.py:226), so an unread record must never carry a decision,
    and `keep` is False because the tool has no grounds to put an unread paper into a review.
    """
    candidate.screen_decision = ""
    candidate.screen_reason = reason
    candidate.keep = False
    if not candidate.pdf_path:
        candidate.state = "not_screened"


def _client_cost(client: Any) -> float:
    """What the client has spent in total, or 0.0 for a client that does not keep score."""
    total = getattr(client, "total_cost", None)
    try:
        return float(total()) if callable(total) else 0.0
    except Exception:                                # pragma: no cover - a stand-in without a ledger
        return 0.0


def _client_budget(client: Any, default: float | None = None) -> float | None:
    """The client's own USD budget, when it has one — the cap that raised `BudgetExceeded`."""
    budget = getattr(client, "budget_usd", None)
    return float(budget) if isinstance(budget, (int, float)) else default


def _emit(on_progress: Callable[[Mapping[str, Any]], None] | None, **event: Any) -> None:
    """Progress, straight to the caller's callback.

    Not through `pipeline.state.emit`: that builds a fixed five-key dict and takes no extra keys
    (state.py:256-261), so `n`/`total` — the two the bar needs — cannot travel through it.
    """
    if on_progress is None:
        return
    on_progress(dict(event))


# --------------------------------------------------------------------------------- the stage
def screen_candidates(client: Any | None, candidates: Sequence[Candidate], *,
                      question: str, criteria: Sequence[str], model: str,
                      budget_usd: float | None, cell_key_prefix: str,
                      batch_size: int = BATCH, max_tokens: int = MAX_TOKENS,
                      on_progress: Callable[[Mapping[str, Any]], None] | None = None,
                      ) -> ScreenOutcome:
    """Screen `candidates` on title + abstract, writing every verdict onto the candidate itself.

    Mutates each `Candidate` in place — `screen_decision`, `screen_reason`,
    `screened_on_title_only`, `state`, `keep` — and returns the summary. Sequential by design: the
    cost cap is checked *between* batches, and a batch already paid for is never thrown away.

    `client is None` means no model is available (no API key). That is not an error: every record
    comes back `not_screened` with the reason "no model was available to screen this record",
    `keep=False` and `stopped_because=""`, and the user reads the abstracts themselves. Degrade
    honestly, never die.

    **The cost cap is hard, and its worst case is one batch — INCLUDING that batch's retry.**
    Before each batch the next call's reservation price (`estimate_request_cost`, costs.py:217-221)
    is added to what screening has already spent; if that exceeds `budget_usd` the stage stops
    cleanly, every remaining record is `not_screened` with a reason naming the cap, and
    `stopped_because == "budget"`. The reservation is **two** calls, not one: when the model runs
    out of output room `client.structured` silently retries at double `max_tokens`
    (client.py:539-546), and that retry is a second billed call. Reserving only the first one let a
    measured batch of 20 bill **$0.134 against a $0.051 reservation** — the retry, not tokeniser
    variance, was the dominant term, and it is deterministic rather than occasional (review §M8).
    `LLMClient` reserves its own money before each call (client.py:525) and raises `BudgetExceeded`
    if its budget would not take it — that is caught here, charged to the ledger and turned into the
    same clean stop, never a lost batch. The residual overshoot is now only one batch's *actual*
    cost above its *reservation*, which is positive when the real tokeniser beats the `chars/3.5`
    estimate (costs.py:143): **≈ $0.05 on ordinary English abstracts and ≈ $0.15 on a batch of dense
    non-Latin abstracts**, where CJK text runs nearer one character per token. The client logs real
    cost, so the next check sees it and the cap self-corrects; the overshoot cannot compound across
    batches (review §D2).

    **What it costs.** With abstracts truncated at `MAX_ABSTRACT_CHARS` = 1,800 characters, one
    record is ~1,800/3.5 ≈ 514 tokens of abstract plus ~30 of title/year/venue ≈ **545**, so a batch
    of 20 is ~700 (system + criteria) + 20 × 545 ≈ **11,600 input tokens** ≈ $0.023 on
    `claude-sonnet-5`, plus ~1,200 output tokens ≈ $0.012 — **≈ $0.035 per batch**, ≈ **$0.35 per
    200 records**. The original design said $0.25 because it costed a 290-token record while its own
    §2.6 truncated at 1,800 characters: ~35 % low (review §D1). The *reservation* is much larger
    than that actual — it reserves `max_tokens` of output for two calls, ≈ $0.13 a batch — so under
    a $1.00 cap the effective ceiling is ~7 batches ≈ 150 records rather than the arithmetic 28.
    That is the price of a cap that cannot be walked past: a cap the retry could overshoot is not a
    cap, and a user who wants the other 200 records raises a number they can see.

    `cell_key_prefix` is the call log's handle on this stage — pass `f"screen:{search_id}"`; each
    batch is logged as `<prefix>:<index>` so a reader can price screening apart from the rest.
    """
    outcome = ScreenOutcome()
    records = list(candidates)
    total = len(records)
    if not records:
        return outcome

    # ---- no model: say so on every record, and change nothing else about the search.
    if client is None:
        for candidate in records:
            _write_unscreened(candidate, NO_MODEL_REASON)
        outcome.notes.append(
            f"no model was configured, so none of the {total} records were screened — every one of "
            f"them is listed for you to read")
        _emit(on_progress, stage="screen", status="skipped", n=0, total=total, cost_so_far=0.0,
              message=NO_MODEL_REASON)
        return outcome

    spent = 0.0
    done = 0
    #: set once the cap fires — the exact sentence every remaining record gets, so the whole tail of
    #: the search names one cap rather than each batch inventing its own wording.
    stopped_reason = ""
    for index, batch in enumerate(batches_of(records, batch_size)):
        record = ScreenBatch(index=index, keys=[c.key for c in batch])
        outcome.batches.append(record)
        if stopped_reason:                           # the cap already fired; nobody reads these
            for candidate in batch:
                _write_unscreened(candidate, stopped_reason)
            continue

        messages = [{"role": "user", "content": _prompt_for(batch, question=question,
                                                            criteria=criteria)}]
        # this call AND the retry it may provoke — see the money paragraph in the docstring. Two
        # separate estimates rather than one at `2 * max_tokens`, because the retry re-sends the
        # whole prompt too: the worst case is 2× input, not 2× output.
        record.estimated_usd = (estimate_request_cost(model, SYSTEM, messages, max_tokens)
                                + estimate_request_cost(model, SYSTEM, messages, max_tokens * 2))

        # The cap, checked BETWEEN batches: the work already paid for is kept, and the batch that
        # would not fit is never sent.
        if budget_usd is not None and spent + record.estimated_usd > budget_usd:
            stopped_reason = cap_reason(budget_usd)
            outcome.stopped_because = "budget"
            outcome.notes.append(
                f"the {_money(budget_usd)} cost cap stopped screening after {done} of {total} "
                f"records; the rest are listed unscreened")
            for candidate in batch:
                _write_unscreened(candidate, stopped_reason)
            _emit(on_progress, stage="screen", status="skipped", n=done, total=total,
                  cost_so_far=round(spent, 6), message=outcome.notes[-1])
            continue

        billed_before = _client_cost(client)
        try:
            result = client.structured(
                model=model, system=SYSTEM, schema=SCREEN_SCHEMA, effort=EFFORT,
                max_tokens=max_tokens, prompt_version=PROMPT_VERSION,
                cell_key=f"{cell_key_prefix}:{index}", messages=messages)
        except BudgetExceeded as exc:
            # The client's own reservation refused it (client.py:331-340). Same clean stop, same
            # wording: the user is told about the cap, not about an exception. The cap NAMED is the
            # client's, because that is the one that fired — it can be lower than `budget_usd`, and
            # a reason quoting the wrong number sends the user to change the wrong setting.
            stopped_reason = cap_reason(_client_budget(client, default=budget_usd))
            outcome.stopped_because = "budget"
            record.error = str(exc)
            # A `BudgetExceeded` raised AFTER the provider answered — a retry the reservation
            # would not take — was still billed. This branch used to leave `record.cost_usd` at
            # zero, so a batch that really cost $0.052 was reported to the user as $0.00 and the
            # cap's own arithmetic never saw it (review §M9). Same ledger, same rule as `LLMError`.
            record.cost_usd = max(0.0, _client_cost(client) - billed_before)
            spent += record.cost_usd
            outcome.notes.append(
                f"the cost cap stopped screening after {done} of {total} records; the rest are "
                f"listed unscreened")
            for candidate in batch:
                _write_unscreened(candidate, stopped_reason)
            _emit(on_progress, stage="screen", status="skipped", n=done, total=total,
                  cost_so_far=round(spent, 6), message=outcome.notes[-1])
            continue
        except LLMError as exc:
            # A refusal, a truncation or unparseable JSON costs THIS batch and no other — the
            # reason batches are twenty records and not two hundred. The failure is recorded on
            # every record it touched, in the failure's own words.
            record.error = f"{type(exc).__name__}: {exc}"
            # A call that failed AFTER the provider answered was still billed. Charging it to the
            # cap keeps a run of failing batches from spending past the number the user set.
            record.cost_usd = max(0.0, _client_cost(client) - billed_before)
            spent += record.cost_usd
            outcome.notes.append(f"batch {index + 1} failed ({record.error}); its "
                                 f"{len(batch)} records are listed unscreened")
            for candidate in batch:
                _write_unscreened(candidate, f"the screener failed on this batch and nobody read "
                                             f"this record ({record.error})")
            _emit(on_progress, stage="screen", status="error", n=done, total=total,
                  cost_so_far=round(spent, 6), message=outcome.notes[-1])
            continue

        record.sent = True
        record.cost_usd = float(getattr(result, "cost_usd", 0.0) or 0.0)
        spent += record.cost_usd
        n_verdicts = _apply(batch, getattr(result, "parsed", None))
        record.n_verdicts = n_verdicts
        done += n_verdicts
        outcome.n_screened += n_verdicts
        _emit(on_progress, stage="screen", status="progress", n=done, total=total,
              cost_so_far=round(spent, 6),
              message=f"screened {done} of {total} records")

    outcome.cost_usd = round(spent, 6)
    return outcome


def _apply(batch: Sequence[Candidate], parsed: Any) -> int:
    """Write one batch's verdicts onto its candidates; return how many landed.

    A record the model did not answer for keeps `not_screened` and is told so. It is never defaulted
    to `exclude`: "nobody read it" and "someone read it and said no" are different facts, and the
    page shows them in different buckets (models.py:41).
    """
    by_ref: dict[str, Candidate] = {}
    for position, candidate in enumerate(batch, start=1):
        by_ref[str(position)] = candidate
        by_ref[candidate.key.lower()] = candidate     # a model that echoes the key instead

    answered: set[str] = set()
    for entry in (parsed or {}).get("decisions", []) or []:
        if not isinstance(entry, Mapping):
            continue
        ref = str(entry.get("ref", "")).strip().strip(".)").lower()
        candidate = by_ref.get(ref)
        if candidate is None or candidate.key in answered:
            continue                                  # a ref for nobody, or a second answer
        decision, note = _normalise(entry.get("decision"))
        title_only = not (candidate.abstract or "").strip()
        _write_verdict(candidate, decision=decision,
                       reason=_reason_for(entry.get("reason"), title_only=title_only, note=note),
                       title_only=title_only)
        answered.add(candidate.key)

    for candidate in batch:
        if candidate.key not in answered:
            _write_unscreened(candidate, NO_VERDICT_REASON)
    return len(answered)
