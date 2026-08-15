"""`canopy` — the command line.

Four commands, and one rule they all obey: **the API key is loaded here and never printed.**
`load_env()` runs once at startup so library code only ever reads `os.environ`; nothing in this
module echoes a value, and `canopy run` reports only whether a key is configured.

    canopy run --papers PDFS --protocol protocol.yaml --out runs/name
    canopy validate runs/name          # re-pool from the stage files, no model calls
    canopy protocol init my-review     # a commented skeleton to fill in
    canopy serve                       # the review UI (Task 12)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import api_key, load_env
from .protocol import available_profiles

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Agent-verified meta-analysis from a folder of paper PDFs.")
protocol_app = typer.Typer(no_args_is_help=True, help="Create and inspect protocols.")
app.add_typer(protocol_app, name="protocol")
console = Console()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "protocols"

SKELETON = '''# Canopy protocol — the ONLY place your review's domain knowledge lives.
# Everything below is yours: Canopy itself knows nothing about your field.
# Fill it in, then run:  canopy run --papers PDFS --protocol {name}.yaml --out runs/{name}

title: {title}

# One sentence. Agents see this, so write it the way you would explain it to a colleague.
research_question: >-
  Do <group A> differ from <group B> in <outcome>?

# The two arms of every comparison. `synonyms` are the words a paper might use instead;
# an extractor echoes the label it actually read so a swapped pair is caught.
group_a:
  key: A
  label: <name of group A>
  definition: >-
    How a paper's own wording identifies this group.
  synonyms: [<other words a paper may use>]

group_b:
  key: B
  label: <name of group B>
  definition: >-
    How a paper's own wording identifies this group.
  synonyms: [<other words a paper may use>]

# One entry per quantity you want pooled. The direction labels become the forest plot's x axis.
outcomes:
  - key: primary_outcome
    label: <human-readable name>
    definition: >-
      What counts as this outcome, in the words a reader of the papers would use.
    measurement_window: >-
      Which timepoint or block to read, when a paper reports several.
    higher_is_better_hint: >-
      Say how to decide whether a larger raw number means more or less of the construct.
      An agent must find the answer in each paper and quote it; this only tells it how to look.
    positive_direction_label: <what a positive effect means, e.g. "Higher in A">
    negative_direction_label: <what a negative effect means, e.g. "Lower in A">
    units_hint: <the units you expect a paper to print>

# Screening rules. A paper failing any of these is excluded, with the quote that decided it.
eligibility:
  - <the design a study must have used>
  - <the comparison it must report>
  - The study is written in English.

# How to treat a paper that contributes more than one comparison.
dataset_rules:
  - <e.g. when the same participants did two experiments, include only the first>

# Recorded per dataset and shown as columns on the forest plot.
moderators:
  - <moderator_name>

# Statistical conventions. `profile` sets them all at once; anything you write here wins.
# Available profiles: {profiles}
stats:
  profile: metafor
  # estimator: cohen        # cohen | hedges
  # tau2_method: REML       # REML | DL | PM
  # hakn: false             # Hartung-Knapp adjustment
  # pi_method: V            # V (t, k-1) | HTS (t, k-2) | z
  # one_row_per_paper: true

notes: >-
  Anything else an agent should know about how to read these papers.
'''


def _startup() -> None:
    load_env()                                             # never printed, never logged


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


# ----------------------------------------------------------------------------- run
@app.command()
def run(
    papers: Path = typer.Option(..., "--papers", help="Directory of PDFs to review.",
                                exists=True, file_okay=False),
    protocol: Path = typer.Option(..., "--protocol", help="Protocol YAML.", exists=True,
                                  dir_okay=False),
    out: Path = typer.Option(..., "--out", help="Run directory to write."),
    budget_usd: Optional[float] = typer.Option(None, "--budget-usd",
                                               help="Stop the run at this total spend."),
    max_usd_per_paper: Optional[float] = typer.Option(
        None, "--max-usd-per-paper",
        help="Stop ONE paper at this spend and carry on with the others."),
    max_papers: Optional[int] = typer.Option(None, "--max-papers",
                                             help="Only the first N unique papers."),
    concurrency: int = typer.Option(4, "--concurrency", min=1,
                                    help="Papers processed at the same time."),
    resume: bool = typer.Option(True, "--resume/--no-resume",
                                help="Skip stages that already have a file (default)."),
    profile: Optional[str] = typer.Option(None, "--profile",
                                          help=f"Statistics profile: {available_profiles()}"),
    quiet: bool = typer.Option(False, "--quiet", help="Only print the summary."),
) -> None:
    """Review every PDF in a folder and write the results to a run directory."""
    _startup()
    from .protocol import dump_protocol, load_protocol
    from .pipeline.run import run_pipeline

    out.mkdir(parents=True, exist_ok=True)
    protocol_path = protocol
    loaded = load_protocol(protocol)
    if profile:
        if profile not in available_profiles():
            _fail(f"unknown profile {profile!r} (available: {available_profiles()})")
        from .models import StatsSettings
        from .protocol import apply_profile
        loaded.stats = apply_profile(StatsSettings(profile=profile))
        protocol_path = dump_protocol(loaded, out / "protocol.yaml")
    else:
        shutil.copyfile(protocol, out / "protocol.yaml")   # the run keeps the protocol it used

    console.print(f"[bold]{loaded.title}[/bold]")
    console.print(f"protocol [dim]{protocol}[/dim] · profile [dim]{loaded.stats.profile}[/dim] · "
                  f"API key {'configured' if api_key() else 'NOT configured'}")

    def progress(event: dict[str, Any]) -> None:
        if quiet or event.get("status") == "started":
            return
        paper = event.get("paper") or "—"
        console.print(f"[dim]{event['stage']:<9}[/dim] {paper:<14} {event['status']:<9} "
                      f"[dim]${event['cost_so_far']:.2f}[/dim] {event.get('message', '')}")

    manifest = run_pipeline(papers, protocol_path, out, budget_usd=budget_usd,
                            max_usd_per_paper=max_usd_per_paper, max_papers=max_papers,
                            concurrency=concurrency, resume=resume, progress=progress,
                            allow_live=True)

    table = Table(show_header=True, header_style="dim")
    for column in ("paper", "file", "status", "eligible", "cost", "s"):
        table.add_column(column)
    for paper_status in manifest.papers:
        table.add_row(paper_status.paper_id[:12], paper_status.filename, paper_status.status,
                      "—" if paper_status.eligible is None else str(paper_status.eligible),
                      f"${paper_status.cost_usd:.2f}", f"{paper_status.seconds:.0f}")
    console.print(table)
    console.print(f"[bold]${manifest.cost_usd:.2f}[/bold] over {manifest.n_llm_calls} calls "
                  f"({manifest.cache_hits} served from cache) in {manifest.seconds:.0f}s")
    if manifest.human_review_queue:
        console.print(f"[yellow]{len(manifest.human_review_queue)} cell(s) need a human[/yellow] "
                      f"— see human_review_queue.csv")
    report = manifest.outputs.get("report.html")
    if report:
        console.print(f"report: {out / report}")
    if any(p.status == "error" for p in manifest.papers):
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------------- validate
@app.command()
def validate(
    run_dir: Path = typer.Argument(..., help="A run directory written by `canopy run`.",
                                   exists=True, file_okay=False),
    protocol: Optional[Path] = typer.Option(None, "--protocol",
                                            help="Re-pool against a different protocol."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Re-pool a finished run from its stage files and check every promised artefact exists.

    Makes no model calls, so it is safe to run on any machine at any time.
    """
    _startup()
    from .pipeline.run import revalidate

    try:
        report = revalidate(run_dir, protocol)
    except FileNotFoundError as exc:
        _fail(f"{exc}")
        return
    if as_json:
        console.print_json(json.dumps(report))
    else:
        console.print(f"[bold]{report['run_dir']}[/bold] — {report['records']} effect sizes "
                      f"from the stage files")
        table = Table(show_header=True, header_style="dim")
        for column in ("outcome", "k", "estimate", "95% CI", "needs human"):
            table.add_column(column)
        for key, row in report["outcomes"].items():
            estimate = "—" if row["estimate"] is None else f"{row['estimate']:.3f}"
            interval = ("—" if row["ci_low"] is None
                        else f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
            table.add_row(key, str(row["k"]), estimate, interval, str(row["n_needs_human"]))
        console.print(table)
        if not report["protocol_matches_manifest"]:
            console.print("[yellow]the protocol has changed since this run[/yellow]")
        for missing in report["missing_outputs"]:
            console.print(f"[red]missing output:[/red] {missing}")
        for missing in report["missing_stages"]:
            console.print(f"[red]missing stage file:[/red] {missing}")
    if not report["ok"]:
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------------- protocol
@protocol_app.command("init")
def protocol_init(
    name: str = typer.Argument(..., help="Name of the review (used for the file name)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write (default ./NAME.yaml)."),
    from_example: Optional[str] = typer.Option(
        None, "--from", help=f"Start from an example instead: {sorted(p.stem for p in EXAMPLES.glob('*.yaml'))}"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a commented protocol skeleton to fill in."""
    target = Path(out) if out is not None else Path(f"{name}.yaml")
    if target.exists() and not force:
        _fail(f"{target} already exists (use --force to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    if from_example:
        source = EXAMPLES / f"{from_example}.yaml"
        if not source.exists():
            _fail(f"no example protocol {from_example!r} in {EXAMPLES}")
        shutil.copyfile(source, target)
    else:
        target.write_text(SKELETON.format(name=name, title=name.replace("_", " ").replace("-", " "),
                                          profiles=available_profiles()), encoding="utf-8")
    console.print(f"wrote [bold]{target}[/bold]")
    console.print("edit it, then: [dim]canopy run --papers PDFS --protocol "
                  f"{target} --out runs/{name}[/dim]")


@protocol_app.command("check")
def protocol_check(
    path: Path = typer.Argument(..., help="Protocol YAML to load.", exists=True, dir_okay=False),
) -> None:
    """Load a protocol, resolve its profile and print what Canopy will use."""
    _startup()
    from .protocol import load_protocol

    try:
        loaded = load_protocol(path)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return
    console.print(f"[bold]{loaded.title}[/bold]  sha256 {loaded.hash()[:12]}")
    console.print(f"groups: A={loaded.group_a.label!r}  B={loaded.group_b.label!r}")
    console.print(f"outcomes: {[o.key for o in loaded.outcomes]}")
    console.print(f"moderators: {loaded.moderators}")
    console.print(f"stats: {loaded.stats.model_dump()}")


# ----------------------------------------------------------------------------- serve
@app.command()
def serve(
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind (loopback default)."),
    runs: Path = typer.Option(Path("runs"), "--runs", help="Directory holding run directories."),
) -> None:
    """Start the review UI."""
    _startup()
    try:
        from .server.app import serve as _serve            # Task 12 supplies this
    except ImportError:
        console.print("[yellow]the Canopy web UI is not installed in this build[/yellow]")
        console.print("`canopy serve` needs `canopy/server/` (Task 12). Until then, open the "
                      "static report a run already wrote:")
        console.print("  [dim]open runs/<name>/report.html[/dim]")
        raise typer.Exit(code=1)
    _serve(host=host, port=port, runs_dir=runs)


@app.command()
def version() -> None:
    """Print the installed Canopy version."""
    from .pipeline.run import _git_commit, _version

    console.print(f"canopy {_version() or 'dev'} ({_git_commit() or 'no git'})")


def main() -> None:                                        # pragma: no cover - console script
    app()


if __name__ == "__main__":                                 # pragma: no cover
    main()
