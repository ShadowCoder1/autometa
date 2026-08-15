"""Test defaults: offline, no API key needed.

Every test runs with `CANOPY_LIVE`/`CANOPY_RECORD` cleared unless it is marked `live` (real calls)
or `replay` (runs from fixtures, but must see the flags during a recording run), so a missing
fixture fails loudly instead of quietly spending money.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:                               # imports stay lazy: most tests need none of these
    from canopy.ingest.pdf import PaperRecord
    from canopy.llm.client import LLMClient
    from canopy.models import DatasetSpec, Protocol, StudyMap


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: performs real API calls (needs CANOPY_LIVE=1)")
    config.addinivalue_line(
        "markers",
        "replay: replays recorded LLM fixtures, and re-records them under CANOPY_LIVE=1 "
        "CANOPY_RECORD=1 (so it must see those env vars, unlike every other test)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("CANOPY_LIVE", "") not in ("", "0", "false", "False"):
        return
    skip_live = pytest.mark.skip(reason="live test: set CANOPY_LIVE=1 to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def offline_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    # `live` tests need the env to reach the API; `replay` tests need it to know whether this run
    # is a recording run. Everything else runs offline whatever the shell says.
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("replay"):
        return
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    monkeypatch.delenv("CANOPY_RECORD", raising=False)


# --------------------------------------------------------------------------- shared fixtures
#: Bock 2005 is the corpus every agent test runs against: ingested once per session (rasterising
#: five pages is not free), mapped once from the recorded fixtures, and shared by every module.
ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf"
REPLAY = ROOT / "tests" / "fixtures" / "llm"
PROTOCOL_PATH = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"


@pytest.fixture(scope="session")
def paper(tmp_path_factory) -> "PaperRecord":
    from canopy.ingest.pdf import ingest_pdf

    return ingest_pdf(PDF, tmp_path_factory.mktemp("bock2005"))


@pytest.fixture(scope="session")
def protocol() -> "Protocol":
    from canopy.protocol import load_protocol

    return load_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="session")
def client() -> "LLMClient":
    """A replaying client (and a recording one under CANOPY_LIVE=1 CANOPY_RECORD=1).

    Only `replay`/`live` tests may depend on this: `offline_by_default` clears the flags for every
    other test, and a session fixture built by one of those could never record.
    """
    from canopy.config import live_enabled, load_env, record_enabled
    from canopy.llm.client import LLMClient

    live, record = live_enabled(), record_enabled()          # read here, not at import
    if live:
        load_env()
    return LLMClient(replay_dir=REPLAY, record_dir=REPLAY if record else None, allow_live=live,
                     cache_dir=None)


@pytest.fixture(scope="session")
def bock_map(client, paper, protocol) -> "StudyMap":
    """The mapper's real map of Bock 2005 — the extractors read the locations it found."""
    from canopy.agents.mapper import map_study

    study = map_study(client, paper, protocol)
    if client.live:                             # recording run: report what it cost
        print(f"\n[mapper] ${client.total_cost():.4f} over {len(client.calls())} calls: "
              f"{[c['model'] for c in client.calls()]}")
    return study


@pytest.fixture(scope="session")
def dataset(bock_map) -> "DatasetSpec":
    """d1: the pointing experiment, twelve older vs twelve younger participants."""
    return bock_map.datasets[0]
