"""Amendment A static guard: no agent output schema may contain derived-statistic fields."""
from __future__ import annotations

import pytest

from canopy.llm.schemas import (DERIVED_STAT_TOKENS, agent_schemas,
                                assert_no_derived_stats, is_derived_stat_field)
from canopy.models import Candidate, DatasetSpec, StudyMap

CLEAN = {
    "type": "object",
    "additionalProperties": False,
    "required": ["groups"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["group", "mean", "dispersion_value"],
                "properties": {
                    "group": {"type": "string", "enum": ["A", "B"]},
                    "mean": {"type": ["number", "null"]},
                    "dispersion_value": {"type": ["number", "null"]},
                    "dispersion_type": {"type": "string",
                                        "enum": ["SD", "SE", "CI95", "UNKNOWN"]},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}


def _with_property(name: str) -> dict:
    s = {"type": "object", "additionalProperties": False, "required": [],
         "properties": dict(CLEAN["properties"])}
    s["properties"][name] = {"type": "number"}
    return s


def test_clean_schema_passes():
    assert_no_derived_stats(CLEAN)          # must not raise


@pytest.mark.parametrize("bad", ["d", "g", "es", "smd", "cohens_d", "hedges_g", "effect_size",
                                 "pooled_sd", "cohen", "hedges", "effectSize", "pooledSD"])
def test_derived_fields_are_rejected(bad):
    with pytest.raises(AssertionError) as e:
        assert_no_derived_stats(_with_property(bad))
    assert bad in str(e.value)


def test_rejects_nested_and_defs():
    nested = {"type": "object", "properties": {"x": {"type": "array", "items": _with_property("g")}}}
    with pytest.raises(AssertionError):
        assert_no_derived_stats(nested)
    defs = {"type": "object", "$defs": {"Row": _with_property("pooled_sd")},
            "properties": {"row": {"$ref": "#/$defs/Row"}}}
    with pytest.raises(AssertionError):
        assert_no_derived_stats(defs)


def test_reported_values_are_allowed():
    """Values transcribed verbatim from the paper are extractions, not derivations."""
    ok = _with_property("reported_value")
    ok["properties"]["reported_ci_low"] = {"type": ["number", "null"]}
    ok["properties"]["standardizer"] = {"type": "string"}
    assert_no_derived_stats(ok)


def test_enum_values_are_not_field_names():
    """`kind: "reported_d"` is a route label, not a computed statistic."""
    s = {"type": "object", "properties": {"kind": {"type": "string",
                                                   "enum": ["group_stats", "reported_d"]}}}
    assert_no_derived_stats(s)


def test_explicit_allow_list():
    assert_no_derived_stats(_with_property("d"), allow={"d"})


def test_tokens_cover_the_amendment_list():
    for token in ("d", "g", "hedges", "cohen", "smd", "effect_size", "pooled_sd"):
        assert is_derived_stat_field(token), token
    assert DERIVED_STAT_TOKENS >= {"d", "g", "smd", "cohen", "hedges"}
    assert not is_derived_stat_field("mean") and not is_derived_stat_field("dispersion_value")


def test_pipeline_models_are_clean():
    """Candidate/StudyMap carry raw extractions only (EffectSizeRecord is the only exception)."""
    for model in (Candidate, StudyMap, DatasetSpec):
        assert_no_derived_stats(model.model_json_schema())


def test_registered_agent_schemas_are_clean():
    schemas = agent_schemas()
    assert isinstance(schemas, dict)
    for name, schema in schemas.items():
        assert_no_derived_stats(schema), name
