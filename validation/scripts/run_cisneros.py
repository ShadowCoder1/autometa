#!/usr/bin/env python
"""Run Canopy over the Cisneros-2024 corpus — the one command the whole validation needs.

    .venv/bin/python validation/scripts/run_cisneros.py --split dev \
        --out validation/out/run_cisneros_dev --budget-usd 25

What it adds on top of `canopy run`:

* **one corpus from several folders** — the user's `~/Downloads/Systematic Review/papers`
  (79 files, 9 unique) plus `validation/papers_oa` (10 open-access PDFs).  `dedupe_pdfs` collapses
  them to 19 unique papers before anything is spent on them;
* **the dev / held-out split** (amendment J).  `validation/splits.json` assigns every unique paper
  to `dev` or `heldout` by sha256 — deterministic, path-independent and frozen.  Prompts are tuned
  on `dev`; `heldout` is run **once**, at the git tag `validation-heldout-v1`, and its numbers are
  the ones that mean anything;
* **offline modes** so every downstream script can be exercised with no credit at all:
  `--demo` (the pipeline's own `FakeProvider`, real PDFs, fake model answers) and
  `--replay DIR` (recorded cassettes).

Re-running is free: `--resume` (the default) skips every stage that already has a file, and the
disk cache under `<out>/cache` means even a fresh run directory does not re-ask a question the
account has already paid for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scripts._common import PROTOCOL, SPLITS, VALIDATION  # noqa: E402

DEFAULT_PAPER_DIRS = (Path.home() / "Downloads" / "Systematic Review" / "papers",
                      VALIDATION / "papers_oa")
PROFILE = "cisneros2024"
HELDOUT_TAG = "validation-heldout-v1"


# ----------------------------------------------------------------------------- the corpus
def discover(dirs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in dirs:
        directory = Path(directory).expanduser()
        if not directory.exists():
            print(f"! paper folder not found, skipping: {directory}")
            continue
        paths.extend(sorted(p for p in directory.rglob("*.pdf") if p.is_file()))
    return paths


def unique_papers(dirs: Sequence[Path]) -> list[Any]:
    """The unique papers of the corpus, as `PaperGroup`s (representative + duplicates)."""
    from canopy.ingest.dedupe import dedupe_pdfs

    return dedupe_pdfs(list(discover(dirs)))


def build_split(groups: Sequence[Any]) -> dict[str, Any]:
    """Assign every unique paper to `dev` or `heldout`, deterministically and reproducibly.

    Sorted by sha256 and alternated, so the assignment depends only on the papers' bytes — not on
    file names, not on the order a folder happens to list, and not on when this ran.  Adding a
    paper later re-shuffles the alternation, which is exactly why the result is FROZEN in
    `validation/splits.json` and read from there afterwards.
    """
    ordered = sorted(groups, key=lambda g: g.sha256)
    papers = []
    for index, group in enumerate(ordered):
        papers.append({
            "sha256": group.sha256,
            "sha12": group.sha256[:12],
            "filename": group.representative.name,
            # a label, not a path: splits.json is read on machines where the corpus lives
            # somewhere else, and an absolute home directory has no business in a committed file
            "source": ("validation/papers_oa"
                       if VALIDATION / "papers_oa" == group.representative.parent
                       else "papers_folder"),
            "title": group.title,
            "doi": group.doi,
            "n_duplicate_files": len(group.duplicates),
            "split": "dev" if index % 2 == 0 else "heldout",
        })
    return {
        "rule": ("unique papers sorted by sha256, alternating dev/heldout starting with dev; "
                 "frozen once written"),
        "heldout_tag": HELDOUT_TAG,
        "n_papers": len(papers),
        "n_dev": sum(1 for p in papers if p["split"] == "dev"),
        "n_heldout": sum(1 for p in papers if p["split"] == "heldout"),
        "papers": papers,
    }


def load_split(path: str | Path = SPLITS) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — create it once with:\n"
            f"    .venv/bin/python validation/scripts/run_cisneros.py --write-split")
    return json.loads(path.read_text())


def select(groups: Sequence[Any], split: str, splits: dict[str, Any]) -> list[Any]:
    """The `PaperGroup`s belonging to one side of the split (`all` keeps everything)."""
    if split == "all":
        return list(groups)
    wanted = {p["sha256"] for p in splits["papers"] if p["split"] == split}
    unknown = [g for g in groups if g.sha256 not in {p["sha256"] for p in splits["papers"]}]
    if unknown:
        names = ", ".join(g.representative.name for g in unknown)
        print(f"! {len(unknown)} paper(s) are not in {SPLITS.name} and were NOT run: {names}\n"
              f"  (the split is frozen on purpose — re-write it deliberately with --write-split)")
    return [g for g in groups if g.sha256 in wanted]


def stage_papers(groups: Sequence[Any], directory: Path) -> Path:
    """Link the selected papers into one folder for the pipeline (symlinks, never copies)."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.pdf"):
        old.unlink()
    for group in groups:
        target = directory / f"{group.sha256[:12]}_{group.representative.name}"
        try:
            target.symlink_to(group.representative.resolve())
        except OSError:                                    # pragma: no cover - no symlink support
            import shutil

            shutil.copyfile(group.representative, target)
    return directory


# ----------------------------------------------------------------------------- the client
def offline_client(mode: str, replay_dir: Path | None, out: Path) -> Any:
    """A client that makes no live call: the pipeline's own fake, or recorded cassettes."""
    from canopy.llm.client import LLMClient

    if mode == "replay":
        return LLMClient(replay_dir=replay_dir, allow_live=False, cache_dir=out / "cache")
    # `--demo`: reuse the pipeline's OWN offline fake rather than keeping a second one alive here.
    # It answers every agent with model-shaped JSON built from the real fixture PDFs, so the run
    # is real in every respect except the model. `_fake` is the single import seam (see its
    # docstring for why the fake still lives in the test package).
    from canopy.llm.providers import FakeProvider

    from validation.scripts._fake import demo_router

    return LLMClient(provider=FakeProvider([demo_router(out / "demo_ingest")]),
                     allow_live=True, cache_dir=None)


# ----------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", action="append", type=Path, default=None,
                        help="paper folder (repeatable; defaults to the two corpus folders)")
    parser.add_argument("--out", type=Path, default=VALIDATION / "out" / "run_cisneros",
                        help="run directory to write")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
    parser.add_argument("--write-split", action="store_true",
                        help="(re)write validation/splits.json from the corpus and exit")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--max-usd-per-paper", type=float, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="offline: the pipeline's own FakeProvider over the fixture PDFs")
    parser.add_argument("--replay", type=Path, default=None,
                        help="offline: replay recorded cassettes from this directory")
    args = parser.parse_args(argv)

    dirs = args.papers or list(DEFAULT_PAPER_DIRS)
    if args.write_split:
        groups = unique_papers(dirs)
        payload = build_split(groups)
        SPLITS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {SPLITS} — {payload['n_papers']} unique papers "
              f"({payload['n_dev']} dev, {payload['n_heldout']} held out)")
        return 0

    if args.demo:                                          # the demo corpus is the fixture PDFs
        from validation.scripts._fake import demo_pdfs

        dirs = sorted({p.parent for p in demo_pdfs()})

    groups = unique_papers(dirs)
    if args.demo:
        chosen = list(groups)
    else:
        chosen = select(groups, args.split, load_split())
    if args.max_papers is not None:
        chosen = chosen[:args.max_papers]

    print(f"corpus: {len(groups)} unique paper(s) from {len(dirs)} folder(s); "
          f"split={args.split} -> {len(chosen)} to run")
    for group in chosen:
        print(f"  {group.sha256[:12]}  {group.representative.name:<34} "
              f"{(group.title or '')[:58]}")
    if args.list:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    staged = stage_papers(chosen, out / "papers_in")

    # The run keeps the protocol it actually used, with the profile resolved into it. `PROFILE`
    # is a floor, not a bulldozer: any `stats:` field the protocol sets EXPLICITLY wins, which is
    # `apply_profile`'s own contract. (Reading the raw YAML is the only way to tell: `load_protocol`
    # rebuilds settings from a full dump, after which every field looks explicitly set.)
    import yaml

    from canopy.models import StatsSettings
    from canopy.protocol import apply_profile, dump_protocol, load_protocol

    protocol = load_protocol(args.protocol)
    raw = yaml.safe_load(Path(args.protocol).read_text()) or {}
    explicit = {k: v for k, v in (raw.get("stats") or {}).items() if k != "profile"}
    protocol.stats = apply_profile(StatsSettings(profile=PROFILE, **explicit))
    if explicit:
        print(f"profile {PROFILE}; kept the protocol's own {', '.join(sorted(explicit))}")
    protocol_path = dump_protocol(protocol, out / "protocol.yaml")

    client = None
    if args.demo or args.replay is not None:
        client = offline_client("replay" if args.replay is not None else "demo",
                                args.replay, out)
    else:
        from canopy.config import load_env

        load_env()

    def progress(event: dict[str, Any]) -> None:
        if event.get("status") == "started":
            return
        print(f"{event['stage']:<9} {(event.get('paper') or '—'):<14} {event['status']:<9} "
              f"${event['cost_so_far']:.2f} {event.get('message', '')}", flush=True)

    from canopy.pipeline.run import run_pipeline

    manifest = run_pipeline(staged, protocol_path, out, budget_usd=args.budget_usd,
                            max_usd_per_paper=args.max_usd_per_paper,
                            concurrency=args.concurrency, resume=not args.no_resume,
                            progress=progress, client=client,
                            allow_live=not (args.demo or args.replay is not None))

    print(f"\n{len(manifest.papers)} paper(s) · ${manifest.cost_usd:.2f} · "
          f"{manifest.n_llm_calls} model call(s) · {manifest.cache_hits} cache hit(s)")
    for status in manifest.papers:
        print(f"  {status.paper_id[:12]} {status.filename:<34} {status.status:<9} "
              f"${status.cost_usd:.2f} {status.error[:60]}")
    for outcome in protocol.outcomes:
        pooled = out / "results" / outcome.key / "pooled.json"
        if pooled.exists():
            payload = json.loads(pooled.read_text())
            print(f"  {outcome.key}: k={payload.get('k', 0)} "
                  f"d={payload.get('estimate')} held={payload.get('n_needs_human')}")
    print(f"\nreport: {out / 'report.html'}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
