"""Protocol loading + statistics profiles.

A protocol is the *only* place domain knowledge lives: eligibility, group definitions, outcome
definitions, dataset rules and the statistical conventions. Profiles are named bundles of those
conventions (e.g. reproduce `metafor` defaults, or a published review's choices).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import Protocol, StatsSettings

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
_META_KEYS = {"name", "description", "reference"}


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict[str, Any]:
    """Raw profile mapping (statistics settings only). Raises KeyError for an unknown name."""
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown stats profile {name!r} (available: {available_profiles()})")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"profile {name!r} must be a mapping")
    unknown = set(data) - _META_KEYS - set(StatsSettings.model_fields)
    if unknown:
        raise ValueError(f"profile {name!r} has unknown settings: {sorted(unknown)}")
    return data


def apply_profile(settings: StatsSettings) -> StatsSettings:
    """Fill settings from `settings.profile`; values the caller set explicitly always win.

    Idempotent: applying to an already-resolved settings object returns the same values.
    """
    profile = load_profile(settings.profile)
    explicit = set(settings.model_fields_set) - {"profile"}
    data = settings.model_dump()
    for key, value in profile.items():
        if key in _META_KEYS or key in explicit:
            continue
        data[key] = value
    return StatsSettings(**data)


def load_protocol(path: str | Path) -> Protocol:
    """Load a protocol YAML/JSON file and resolve its statistics profile."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"protocol {path} must be a mapping")
    protocol = Protocol.model_validate(raw)
    protocol.stats = apply_profile(protocol.stats)
    return protocol


def dump_protocol(protocol: Protocol, path: str | Path) -> Path:
    """Write a protocol back to YAML (used by the CLI/UI protocol editor)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False,
                                   allow_unicode=True, width=100))
    return path
