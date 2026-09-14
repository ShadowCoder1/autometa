"""Citation chasing: the papers the included and unsure papers cite, are cited by, and share
authors with — the round a review team runs after the database search.

Why it exists (design 03 §6, 01): eight of the first answer key's twenty-three papers are one
laboratory's, and one of them (a 2020 paper that says none of the block words in its record) is
reached by no string at all — only by a citation. A person chasing a cluster reads the reference
lists of what they have and the papers that cite them; this stage does that by request.

One round, from the SEEDS — the includes and unknowns not yet expanded, best relevance first,
at most `MAX_SEEDS`:

    resolve   seeds without an OpenAlex id → `filter=doi:a|b|…` (≤ 50 a request)
    backward  every seed's `referenced_works`
    forward   per seed `filter=cites:<W>`, ≤ `FORWARD_PAGES` pages of 200
    author    authors on ≥ 2 seeds (≤ `MAX_AUTHORS`) → `authorships.author.id:<A>` AND the task
              block, title and abstract
    fallback  a seed with a PMID and no OpenAlex answer → Europe PMC `/MED/<pmid>/citations` and
              `/references`, hydrated through `SRC:MED AND (EXT_ID:…)`
    hydrate   EVERY unseen id first — `filter=openalex_id:W1|…|W50`, full SELECT — because the
              ranking needs titles and abstracts and a raw id has neither
    rank      `cited_by_seeds` = how many seeds cite or are cited by it; then `rank.py`
    screen    the first `round_cap` by rank, `screen_candidates(round_index=r)`

Stop on a round with no new include, on `MAX_ROUNDS`, or when the round's screening would pass
the snowball's share of the budget. Every seed is expanded ONCE (`expanded_as_seed`) — a round
that re-expanded the same sixty best papers spent two hundred requests to learn nothing (design
04 MAJOR-2). Bound ≈ 2 resolve + 120 forward + 84 hydrate + 10 author ≈ 250 requests a round, and
`record.rounds[]` says what each round asked and found. Unknowns are seeds (they are the
cluster); the stop rule counts includes only.

Nothing here knows a research field: the seeds are whatever the screener kept, the task block is
whatever the model wrote, and every id came from an index.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from .dedupe import dedupe, normalise_doi
from .indices import EuropePmc, OpenAlex, source_record
from .models import Candidate
from .rank import rank
from .transport import SearchTransport

__all__ = ["MAX_SEEDS", "MAX_ROUNDS", "MAX_AUTHORS", "FORWARD_PAGES", "RESOLVE_BATCH",
           "HYDRATE_BATCH", "MAX_REQUESTS_PER_ROUND", "SnowballRound", "snowball",
           "round_cap"]

MAX_SEEDS = 60
MAX_ROUNDS = 3
MAX_AUTHORS = 10
FORWARD_PAGES = 2
RESOLVE_BATCH = 50
HYDRATE_BATCH = 50
#: Europe PMC's citations/references pages go to 1,000 in one request; the first run asked for
#: 100 and the one paper the design expected the chase to find sat at position 300-odd of a
#: seed's 457 citations
EPMC_LINKS_PAGE = 1000
#: the request bound the design states per round; the stage stops asking when it is reached
MAX_REQUESTS_PER_ROUND = 250
#: raw ids a round may collect before hydration — 60 seeds × (40 back + 30 forward) ≈ 4,200
MAX_RAW_IDS = 4500


def round_cap(max_usd: float | None, *, share: float, cost_per_record: float,
              rounds: int = MAX_ROUNDS) -> int | None:
    """Records a round may screen: the snowball's share of the budget, spread over the rounds.
    None when there is no budget (nothing is screened anyway without a model)."""
    if max_usd is None:
        return None
    return max(0, int(float(max_usd) * share / cost_per_record / max(1, rounds)))


class SnowballRound(dict):
    """One row of `record.rounds` — a dict so it serialises as it is."""


def _w_id(candidate: Candidate) -> str:
    return str(candidate.ids.get("openalex") or "").strip()


def _bare(value: Any) -> str:
    text = str(value or "").strip()
    return text.rstrip("/").rsplit("/", 1)[-1] if text else ""


def snowball(candidates: Sequence[Candidate], *, transport: SearchTransport, plan: Mapping[str, Any],
             screen: Callable[[Sequence[Candidate], int], Any], cap: int | None,
             max_rounds: int = MAX_ROUNDS, budget_usd: float | None = None,
             cost_so_far: Callable[[], float] | None = None,
             cancelled: Callable[[], bool] | None = None,
             now: Callable[[], float] = time.monotonic) -> tuple[list[Candidate], list[dict]]:
    """Run the rounds. Returns `(all candidates, rounds)`; the candidate list grows with what
    the rounds found (deduped against what was there), and every new candidate carries
    `found_in_round`, `cited_by_seeds` and a `snowball` entry in `found_by`.

    `screen(chosen, round_index)` is the orchestrator's screening call (it prices, logs and
    writes verdicts); it returns the `ScreenOutcome`. `cap` is `round_cap(...)`; `budget_usd`
    with `cost_so_far` stops a round whose screening could not fit the snowball's share.
    """
    everything = list(candidates)
    rounds: list[dict[str, Any]] = []
    openalex, epmc = OpenAlex(), EuropePmc()
    for r in range(1, max(0, int(max_rounds)) + 1):
        if cancelled and cancelled():
            break
        # includes before unknowns, each in relevance order: a screener's include is stronger
        # evidence of the cluster than its unknown, and the first measured run expanded 180
        # seeds without reaching a key paper cited by only two of the includes, because unknowns
        # with a higher string score took the seats first
        eligible = [c for c in everything
                    if c.screen_decision in ("include", "unknown")
                    and not c.expanded_as_seed and not c.excluded_by_user]
        ranked = rank(eligible, plan)
        seeds = ([c for c in ranked if c.screen_decision == "include"]
                 + [c for c in ranked if c.screen_decision == "unknown"])[:MAX_SEEDS]
        if not seeds:
            rounds.append(SnowballRound(round=r, n_seeds=0, stopped="no unexpanded seed"))
            break
        started = now()
        row = SnowballRound(round=r, n_seeds=len(seeds), requests={}, n_raw_ids=0, n_hydrated=0,
                            n_new=0, n_screened=0, n_included=0, n_unknown=0, cost_usd=0.0,
                            notes=[], stopped="")
        for seed in seeds:
            seed.expanded_as_seed = True
            seed.seed = True
        budget = _RequestBudget(MAX_REQUESTS_PER_ROUND, row["requests"])
        known_ids = {_w_id(c) for c in everything if _w_id(c)}
        known_dois = {normalise_doi(c.doi) for c in everything if normalise_doi(c.doi)}

        # ---- resolve: seeds without an OpenAlex id, by DOI, fifty a request
        by_w: dict[str, Candidate] = {_w_id(s): s for s in seeds if _w_id(s)}
        unresolved = [s for s in seeds if not _w_id(s) and normalise_doi(s.doi)]
        seed_rows: dict[str, dict[str, Any]] = {}
        for start in range(0, len(unresolved), RESOLVE_BATCH):
            batch = unresolved[start:start + RESOLVE_BATCH]
            if not budget.take("resolve"):
                break
            expr = "doi:" + "|".join(normalise_doi(s.doi) for s in batch)
            rows, info = openalex.filter_page(
                transport, expr, limit=RESOLVE_BATCH,
                select="id,doi,referenced_works,authorships,cited_by_count",
                context={"index": "openalex", "form": "resolve", "query_id": f"r{r}",
                         "page": start // RESOLVE_BATCH + 1, "query_text": expr, "exact": True})
            if info.get("error"):
                row["notes"].append(f"OpenAlex could not resolve {len(batch)} seed DOI(s): "
                                    f"{info.get('note') or info['error']}")
                break
            by_doi = {normalise_doi(s.doi): s for s in batch}
            for work in rows:
                doi = normalise_doi(work.get("doi") or "")
                seed = by_doi.get(doi)
                w = _bare(work.get("id"))
                if seed is not None and w:
                    seed.ids["openalex"] = w
                    by_w[w] = seed
                    seed_rows[w] = work
        # the seeds we already had an id for still need their reference lists
        need_refs = [w for w in by_w if w not in seed_rows]
        for start in range(0, len(need_refs), HYDRATE_BATCH):
            batch = need_refs[start:start + HYDRATE_BATCH]
            if not budget.take("resolve"):
                break
            expr = "openalex_id:" + "|".join(batch)
            rows, info = openalex.filter_page(
                transport, expr, limit=HYDRATE_BATCH,
                select="id,doi,referenced_works,authorships,cited_by_count",
                context={"index": "openalex", "form": "resolve", "query_id": f"r{r}",
                         "page": 100 + start // HYDRATE_BATCH, "query_text": expr,
                         "exact": True})
            if info.get("error"):
                row["notes"].append(f"OpenAlex could not read {len(batch)} seed record(s): "
                                    f"{info.get('note') or info['error']}")
                break
            for work in rows:
                seed_rows[_bare(work.get("id"))] = work

        # ---- backward + forward + author
        raw: dict[str, set[str]] = {}          # new W id → the seeds it is linked to
        authors: dict[str, int] = {}

        def link(w: str, seed_w: str) -> None:
            if w and w not in known_ids and len(raw) < MAX_RAW_IDS:
                raw.setdefault(w, set()).add(seed_w)

        for w, work in seed_rows.items():
            for ref in work.get("referenced_works") or []:
                link(_bare(ref), w)
            for authorship in work.get("authorships") or []:
                a = _bare(((authorship or {}).get("author") or {}).get("id"))
                if a:
                    authors[a] = authors.get(a, 0) + 1
        oa_dead = False
        for w in list(by_w):
            cursor = "*"
            for page in range(1, FORWARD_PAGES + 1):
                if oa_dead or not budget.take("forward"):
                    break
                rows, info = openalex.filter_page(
                    transport, f"cites:{w}", limit=200, cursor=cursor,
                    select="id,doi", context={"index": "openalex", "form": "forward",
                                              "query_id": f"r{r}:{w}", "page": page,
                                              "query_text": f"cites:{w}", "exact": True})
                if info.get("error"):
                    if info.get("outcome") == "rate_limited":
                        oa_dead = True
                        row["notes"].append(f"OpenAlex stopped answering forward citations "
                                            f"({info.get('note') or info['error']}); the rest "
                                            f"of this round's forward links came from Europe "
                                            f"PMC where a seed had a PMID")
                    break
                for work in rows:
                    link(_bare(work.get("id")), w)
                cursor = str(info.get("next_cursor") or "")
                if not cursor:
                    break
        shared = sorted((a for a, n in authors.items() if n >= 2), key=lambda a: -authors[a])
        task_terms = _task_terms(plan)
        for a in shared[:MAX_AUTHORS]:
            if oa_dead or not budget.take("author") or not task_terms:
                break
            expr = f"authorships.author.id:{a},title_and_abstract.search:{task_terms}"
            rows, info = openalex.filter_page(
                transport, expr, limit=200, select="id,doi",
                context={"index": "openalex", "form": "author", "query_id": f"r{r}:{a}",
                         "page": 1, "query_text": expr, "exact": True})
            if info.get("error"):
                if info.get("outcome") == "rate_limited":
                    oa_dead = True
                    row["notes"].append("OpenAlex would not answer the author expansion "
                                        f"({info.get('note') or info['error']})")
                break
            for work in rows:
                link(_bare(work.get("id")), f"author:{a}")

        # ---- fallback: Europe PMC citations / references for seeds with a PMID that OpenAlex
        # did not serve (no id resolved, or the forward pass died)
        epmc_ids: dict[str, set[str]] = {}
        failures: dict[str, list[str]] = {}
        for seed in seeds:
            pmid = str(seed.ids.get("pmid") or "").strip()
            needs = pmid and (not _w_id(seed) or oa_dead)
            if not needs:
                continue
            for kind in ("citations", "references"):
                if not budget.take("epmc_fallback"):
                    break
                url, params = epmc.citations_request(pmid, kind, page=1, limit=EPMC_LINKS_PAGE)
                response = transport.get_json(
                    url, params=params, context={"index": "europepmc", "form": kind,
                                                 "query_id": f"r{r}:{pmid}", "page": 1,
                                                 "query_text": url, "exact": True})
                payload = response.json() if response.ok else None
                if not isinstance(payload, dict):
                    failures.setdefault(kind, []).append(
                        f"PMID {pmid}: {response.error or response.outcome}")
                    continue
                key = "citation" if kind == "citations" else "reference"
                for entry in ((payload.get(f"{key}List") or {}).get(key) or []):
                    if isinstance(entry, dict) and str(entry.get("source") or "MED") == "MED":
                        cited = str(entry.get("id") or "").strip()
                        if cited:
                            epmc_ids.setdefault(cited, set()).add(pmid)
        for kind, failed in failures.items():
            # one note per endpoint, not one per seed: the references endpoint answered 503
            # ("temporarily unavailable due to maintenance") for every seed of a whole run
            row["notes"].append(f"Europe PMC {kind} did not answer for {len(failed)} seed(s) "
                                f"(first: {failed[0]})")
        row["n_raw_ids"] = len(raw) + len(epmc_ids)

        # ---- hydrate everything unseen, then dedupe against what we have
        found: list[Candidate] = []
        ids = list(raw)
        for start in range(0, len(ids), HYDRATE_BATCH):
            batch = ids[start:start + HYDRATE_BATCH]
            if oa_dead or not budget.take("backward_hydrate"):
                break
            expr = "openalex_id:" + "|".join(batch)
            rows, info = openalex.filter_page(
                transport, expr, limit=HYDRATE_BATCH,
                context={"index": "openalex", "form": "hydrate", "query_id": f"r{r}",
                         "page": start // HYDRATE_BATCH + 1, "query_text": expr,
                         "exact": True})
            if info.get("error"):
                if info.get("outcome") == "rate_limited":
                    oa_dead = True
                row["notes"].append(f"OpenAlex could not hydrate {len(batch)} id(s): "
                                    f"{info.get('note') or info['error']}")
                break
            for work in rows:
                candidate = openalex.parse(work)
                w = _bare(work.get("id"))
                candidate.found_by = ["snowball"]
                candidate.found_in_round = r
                candidate.cited_by_seeds = len(raw.get(w, ()))
                found.append(candidate)
        pmids = [p for p in epmc_ids if p not in {str(c.ids.get("pmid") or "") for c in everything}]
        for start in range(0, len(pmids), 100):
            batch = pmids[start:start + 100]
            if not budget.take("epmc_hydrate"):
                break
            url, params = epmc.hydrate_request(batch, limit=100)
            response = transport.get_json(
                url, params=params, context={"index": "europepmc", "form": "hydrate",
                                             "query_id": f"r{r}:snowball", "page": start // 100 + 1,
                                             "query_text": params["query"], "exact": True})
            payload = response.json() if response.ok else None
            rows = ((payload.get("resultList") or {}).get("result") or []) \
                if isinstance(payload, dict) else []
            for work in rows:
                if not isinstance(work, dict):
                    continue
                candidate = epmc.parse(work)
                candidate.found_by = ["snowball"]
                candidate.found_in_round = r
                candidate.cited_by_seeds = len(epmc_ids.get(str(candidate.ids.get("pmid") or ""),
                                                            ()))
                found.append(candidate)
        row["n_hydrated"] = len(found)
        before = {c.key for c in everything}
        merged, _pairs = dedupe(everything + found)
        # keep every existing candidate object as it is: the dedupe returns new objects for a
        # merge, and the screener's verdicts live on the old ones. Only genuinely new keys join.
        new = [c for c in merged if c.key not in before
               and normalise_doi(c.doi) not in known_dois]
        for candidate in new:
            candidate.found_in_round = r
        everything.extend(new)
        row["n_new"] = len(new)

        # ---- rank, cap, screen
        chosen = rank(new, plan)[:cap] if cap is not None else rank(new, plan)
        if budget_usd is not None and cost_so_far is not None and chosen:
            # the snowball's share of the budget, in records at the design's constant
            from .cost import COST_PER_RECORD, SHARE_SNOWBALL, SHARE_SCREEN

            ceiling = float(budget_usd) * (SHARE_SCREEN + SHARE_SNOWBALL)
            room = max(0, int((ceiling - float(cost_so_far())) / COST_PER_RECORD))
            if room < len(chosen):
                row["notes"].append(f"the budget left room to screen {room} of the {len(chosen)} "
                                    f"chosen this round")
                chosen = chosen[:room]
        outcome = screen(chosen, r) if chosen else None
        row["n_screened"] = int(getattr(outcome, "n_screened", 0) or 0) if outcome else 0
        row["cost_usd"] = round(float(getattr(outcome, "cost_usd", 0.0) or 0.0), 6)
        row["n_included"] = sum(1 for c in chosen if c.screen_decision == "include")
        row["n_unknown"] = sum(1 for c in chosen if c.screen_decision == "unknown")
        row["seconds"] = round(now() - started, 1)
        rounds.append(row)
        if getattr(outcome, "stopped_because", "") == "budget":
            row["stopped"] = "budget"
            break
        if row["n_included"] == 0:
            row["stopped"] = "no new include"
            break
    return everything, rounds


def _task_terms(plan: Mapping[str, Any]) -> str:
    """The task block as OpenAlex's title-and-abstract filter for the author expansion."""
    from .blocks import block_terms
    from .expand import render_block

    for block in plan.get("blocks") or []:
        if block.get("name") == "task" and block_terms(block):
            return render_block(block_terms(block)[:12])
    return ""


class _RequestBudget:
    """`MAX_REQUESTS_PER_ROUND`, counted per kind onto the round's row."""

    def __init__(self, limit: int, counts: dict[str, int]) -> None:
        self.limit = int(limit)
        self.counts = counts
        self.n = 0

    def take(self, kind: str) -> bool:
        if self.n >= self.limit:
            return False
        self.n += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return True
