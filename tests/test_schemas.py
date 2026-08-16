"""Amendment A static guard: no agent output schema may contain derived-statistic fields."""
from __future__ import annotations

import json

import pytest

from canopy.llm.schemas import (DERIVED_STAT_TOKENS, agent_schemas,
                                assert_no_derived_stats, assert_valid_output_schema,
                                is_derived_stat_field)
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


# ------------------------------------------------------------------ output-schema validity
GOOD = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "rows"],
    "properties": {
        "status": {"type": "string", "enum": ["found", "not_on_these_pages", "unknown"]},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mean", "dispersion_type"],
                "properties": {
                    "mean": {"type": ["number", "null"]},
                    "dispersion_type": {"type": "string", "enum": ["SD", "SE", "UNKNOWN"]},
                },
            },
        },
    },
}


def test_valid_schema_passes():
    assert_valid_output_schema(GOOD)


def test_missing_additional_properties_false():
    bad = json.loads(json.dumps(GOOD))
    del bad["additionalProperties"]
    with pytest.raises(AssertionError, match="additionalProperties"):
        assert_valid_output_schema(bad)
    bad2 = json.loads(json.dumps(GOOD))
    bad2["additionalProperties"] = True
    with pytest.raises(AssertionError, match="additionalProperties"):
        assert_valid_output_schema(bad2)


def test_required_must_list_every_property():
    bad = json.loads(json.dumps(GOOD))
    bad["required"] = ["status"]
    with pytest.raises(AssertionError, match="required"):
        assert_valid_output_schema(bad)
    missing = json.loads(json.dumps(GOOD))
    del missing["required"]
    with pytest.raises(AssertionError, match="required"):
        assert_valid_output_schema(missing)


@pytest.mark.parametrize("keyword,value", [("minimum", 0), ("maximum", 10), ("minLength", 1),
                                           ("maxLength", 5), ("pattern", "^a"), ("format", "date")])
def test_unsupported_keywords_are_rejected(keyword, value):
    bad = json.loads(json.dumps(GOOD))
    bad["properties"]["status"][keyword] = value
    with pytest.raises(AssertionError, match=keyword):
        assert_valid_output_schema(bad)


def test_nested_objects_are_checked():
    bad = json.loads(json.dumps(GOOD))
    del bad["properties"]["rows"]["items"]["additionalProperties"]
    with pytest.raises(AssertionError, match="rows"):
        assert_valid_output_schema(bad)


def test_string_enums_need_an_unknown_member():
    bad = json.loads(json.dumps(GOOD))
    bad["properties"]["status"]["enum"] = ["found", "not_on_these_pages"]
    with pytest.raises(AssertionError, match="enum"):
        assert_valid_output_schema(bad)
    for member in ("unknown", "UNKNOWN", "not_reported", "none", "ambiguous"):
        ok = json.loads(json.dumps(GOOD))
        ok["properties"]["status"]["enum"] = ["found", member]
        assert_valid_output_schema(ok)


def test_non_string_enums_are_exempt():
    ok = json.loads(json.dumps(GOOD))
    ok["properties"]["status"] = {"type": "integer", "enum": [1, 2]}
    assert_valid_output_schema(ok)


def test_defs_and_anyof_branches_are_checked():
    schema = {"type": "object", "additionalProperties": False, "required": ["row"],
              "properties": {"row": {"$ref": "#/$defs/Row"}},
              "$defs": {"Row": {"type": "object", "required": ["a"],
                                "properties": {"a": {"type": "string"}}}}}
    with pytest.raises(AssertionError, match="additionalProperties"):
        assert_valid_output_schema(schema)
    branch = {"type": "object", "additionalProperties": False, "required": ["x"],
              "properties": {"x": {"anyOf": [{"type": "null"},
                                             {"type": "object", "properties": {"y": {"type": "string"}},
                                              "required": ["y"]}]}}}
    with pytest.raises(AssertionError):
        assert_valid_output_schema(branch)
