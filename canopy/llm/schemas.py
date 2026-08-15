"""Static guard on agent output schemas (plan amendment A).

Agents locate and label; they never compute. So no schema an agent fills in may contain a
derived-statistic field (`d`, `g`, `smd`, `cohen*`, `hedges*`, `effect_size`, `pooled_sd`) —
all of those are produced in code by `canopy.stats` from raw extracted values.

Numbers *printed in the paper* are extractions, not derivations: name those fields with a
`reported_` prefix (`reported_value`, `reported_ci_low`, ...) and they are allowed.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from typing import Any, Iterable

#: whole-token names that mark a computed statistic
DERIVED_STAT_TOKENS: frozenset[str] = frozenset({
    "d", "g", "es", "smd", "cohen", "cohens", "hedges", "hedge", "delta_d", "gav", "grm",
})
#: substrings that mark a computed statistic wherever they appear in a field name
DERIVED_STAT_SUBSTRINGS: tuple[str, ...] = (
    "effect_size", "effectsize", "pooled_sd", "pooledsd", "std_mean_diff", "smd_",
)
#: prefixes for values transcribed verbatim from the paper (allowed)
VERBATIM_PREFIXES: tuple[str, ...] = ("reported_",)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: packages scanned by `agent_schemas()` for module-level `*_SCHEMA` constants
AGENT_SCHEMA_PACKAGES: tuple[str, ...] = ("canopy.agents", "canopy.digitize", "canopy.verify")


def _normalize(name: str) -> str:
    return _CAMEL.sub("_", name).lower()


def is_derived_stat_field(name: str, allow: Iterable[str] = ()) -> bool:
    """True when a *property name* looks like a statistic we must compute ourselves."""
    if name in set(allow):
        return False
    norm = _normalize(name)
    if norm.startswith(VERBATIM_PREFIXES):
        return False
    if any(sub in norm for sub in DERIVED_STAT_SUBSTRINGS):
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", norm) if t]
    return any(t in DERIVED_STAT_TOKENS for t in tokens)


def find_derived_stat_fields(schema: Any, allow: Iterable[str] = ()) -> list[str]:
    """Every offending property path in a JSON schema (recurses into $defs, items, anyOf, ...)."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for name in props:
                    if is_derived_stat_field(str(name), allow):
                        found.append(f"{path}.{name}" if path else str(name))
            for key, value in node.items():
                if key in ("enum", "const", "examples", "description", "title"):
                    continue
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(schema, "")
    return found


def assert_no_derived_stats(schema: Any, allow: Iterable[str] = (), name: str = "schema") -> None:
    """Raise AssertionError when `schema` asks a model for a statistic we must compute."""
    bad = find_derived_stat_fields(schema, allow)
    if bad:
        raise AssertionError(
            f"{name} contains derived-statistic field(s): {bad}. Agents must return raw extracted "
            f"values only; effect sizes are computed in canopy.stats. (Values printed in the paper "
            f"belong in `reported_*` fields.)")


def agent_schemas() -> dict[str, Any]:
    """All registered agent output schemas: module-level `SCHEMA` / `*_SCHEMA` dicts."""
    out: dict[str, Any] = {}
    for pkg_name in AGENT_SCHEMA_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        for mod_info in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
            try:
                mod = importlib.import_module(f"{pkg_name}.{mod_info.name}")
            except ImportError:                              # pragma: no cover - defensive
                continue
            for attr in dir(mod):
                if attr == "SCHEMA" or attr.endswith("_SCHEMA"):
                    value = getattr(mod, attr)
                    if isinstance(value, dict):
                        out[f"{pkg_name}.{mod_info.name}.{attr}"] = value
    return out
