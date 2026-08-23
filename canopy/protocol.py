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


class _StrictLoader(yaml.SafeLoader):
    """A YAML loader that refuses duplicate mapping keys instead of silently keeping the last.

    PyYAML follows the spec's "last one wins" for a repeated key, without a word. A protocol is
    the one file where every line is load-bearing, and the usual way to break it is to append a
    section that already exists further up. A live run of this tool lost a whole `digitize:`
    block that way; the setting's absence then looked exactly like the feature being broken
    rather than like it never having been switched on. Refuse to load such a file at all.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:                # an unhashable key is legal YAML and not our problem
                continue
            if duplicate:
                raise ValueError(
                    f"duplicate key {key!r} on line {key_node.start_mark.line + 1}: YAML keeps "
                    f"only the last one, so everything under the first is dropped in silence. "
                    f"Merge the two blocks into one.")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml_strict(text: str, what: str) -> Any:
    """`yaml.safe_load` that will not silently drop a repeated key. See `_StrictLoader`."""
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict[str, Any]:
    """Raw profile mapping (statistics settings only). Raises KeyError for an unknown name."""
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown stats profile {name!r} (available: {available_profiles()})")
    data = load_yaml_strict(path.read_text(), f"profile {name!r}") or {}
    if not isinstance(data, dict):
        raise ValueError(f"profile {name!r} must be a mapping")
    unknown = set(data) - _META_KEYS - set(StatsSettings.model_fields)
    if unknown:
        raise ValueError(f"profile {name!r} has unknown settings: {sorted(unknown)}")
    return data


def apply_profile(settings: StatsSettings) -> StatsSettings:
    """Fill settings from `settings.profile`; values the caller set explicitly always win.

    "Explicit" means present in `model_fields_set` — the fields the caller actually passed (or
    that were present in the protocol YAML). Because the returned object is rebuilt from a full
    dump, *every* field counts as explicit afterwards, which makes this idempotent but also means
    a resolved `StatsSettings` will not pick up a different profile if you mutate `.profile` in
    place. Change the profile by re-loading the protocol (`load_protocol`) or by constructing a
    fresh `StatsSettings(profile=...)`, which is what the CLI, the UI and the tests all do.
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
    raw = load_yaml_strict(path.read_text(), f"protocol {path}") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"protocol {path} must be a mapping")
    protocol = Protocol.model_validate(raw)
    protocol.stats = apply_profile(protocol.stats)
    return protocol


def dump_protocol(protocol: Protocol, path: str | Path) -> Path:
    """Write a protocol back to YAML (used by the CLI/UI protocol editor).

    A dumped protocol is a RESOLVED protocol. `model_dump` writes every field, so a later
    `load_protocol` finds them all in the YAML, counts them all as explicit, and `apply_profile`
    becomes a no-op — the dump manufactures explicitness. A dump taken before the profile was
    applied therefore freezes the class defaults as if somebody chose them: the web UI's YAML
    path did exactly that, and a run whose protocol asked for Hedges' g via `profile: metafor`
    would have computed Cohen's d and labelled it Hedges' g. Resolving here makes the only full
    dump anyone can write one on which `apply_profile` is already a fixed point; callers that
    resolved already (the CLI, `load_protocol` round-trips) are unchanged.
    """
    resolved = protocol.model_copy(deep=True)
    resolved.stats = apply_profile(resolved.stats)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False,
                                   allow_unicode=True, width=100))
    return path
