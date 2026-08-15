"""Test defaults: offline, no API key needed.

Every test runs with `CANOPY_LIVE`/`CANOPY_RECORD` cleared unless it is marked `@pytest.mark.live`,
so a missing fixture fails loudly instead of quietly spending money.
"""
from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: performs real API calls (needs CANOPY_LIVE=1)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("CANOPY_LIVE", "") not in ("", "0", "false", "False"):
        return
    skip_live = pytest.mark.skip(reason="live test: set CANOPY_LIVE=1 to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def offline_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.delenv("CANOPY_LIVE", raising=False)
    monkeypatch.delenv("CANOPY_RECORD", raising=False)
