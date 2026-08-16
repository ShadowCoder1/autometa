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


#: JSON-schema keywords the structured-output API rejects
UNSUPPORTED_KEYWORDS: tuple[str, ...] = ("minimum", "maximum", "minLength", "maxLength",
                                         "pattern", "format")
#: an enum of strings must offer one of these, so a model can say "the paper does not say"
UNKNOWN_ENUM_MEMBERS: frozenset[str] = frozenset({"unknown", "not_reported", "none", "ambiguous"})
_BRANCH_KEYS = ("items", "prefixItems", "anyOf", "oneOf", "allOf", "additionalItems", "contains")
#: keys whose value is a *mapping of name -> schema*, not a schema
_MAP_KEYS = ("$defs", "definitions", "patternProperties")


def _is_object_schema(node: dict[str, Any]) -> bool:
    types = node.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    return "object" in types or isinstance(node.get("properties"), dict)


def find_schema_problems(schema: Any) -> list[str]:
    """Structured-output rules: strict objects, no unsupported keywords, escapable enums."""
    problems: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return

        for keyword in UNSUPPORTED_KEYWORDS:
            if keyword in node:
                problems.append(f"{path or '<root>'}: unsupported keyword {keyword!r} "
                                f"(structured outputs reject it)")

        enum = node.get("enum")
        if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
            if not any(str(v).strip().lower() in UNKNOWN_ENUM_MEMBERS for v in enum):
                problems.append(
                    f"{path or '<root>'}: enum {enum} has no unknown-like member — add one of "
                    f"{sorted(UNKNOWN_ENUM_MEMBERS)} so a model can report an absent value")

        if _is_object_schema(node):
            props = node.get("properties")
            if node.get("additionalProperties") is not False:
                problems.append(f"{path or '<root>'}: object needs additionalProperties: false")
            if isinstance(props, dict):
                required = node.get("required")
                if not isinstance(required, list):
                    problems.append(f"{path or '<root>'}: object needs a required list")
                elif set(required) != set(props):
                    missing = sorted(set(props) - set(required))
                    extra = sorted(set(required) - set(props))
                    problems.append(
                        f"{path or '<root>'}: required must list every property "
                        f"(missing {missing}, unknown {extra})")
                for name, sub in props.items():
                    walk(sub, f"{path}.{name}" if path else str(name))

        for key in _BRANCH_KEYS:
            if key in node:
                walk(node[key], f"{path}.{key}" if path else key)
        for key in _MAP_KEYS:
            sub_map = node.get(key)
            if isinstance(sub_map, dict):
                for name, sub in sub_map.items():
                    walk(sub, f"{path}.{key}.{name}" if path else f"{key}.{name}")

    walk(schema, "")
    return problems


def assert_valid_output_schema(schema: Any, name: str = "schema") -> None:
    """Raise AssertionError when a structured-output schema breaks the API/protocol rules."""
    problems = find_schema_problems(schema)
    if problems:
        raise AssertionError(f"invalid output {name}:\n  - " + "\n  - ".join(problems))


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
