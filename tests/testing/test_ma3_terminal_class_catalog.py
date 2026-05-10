from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_dev_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "MA3" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_property_probe_output_extracts_full_metadata() -> None:
    module = _load_dev_module("ma3_terminal_class_catalog")
    output = "\n".join(
        [
            'OK:Lua "..."',
            "Admin[Fixture]>__PROPERTY_COUNT__\t2",
            "__PROPERTY__\t0\tNAME\tString\tfalse\tfalse\tfalse\t",
            "__PROPERTY__\t1\tENUMPROP\tUInt32\ttrue\tfalse\ttrue\tYesNo",
        ]
    )

    parsed = module.parse_property_probe_output(output)

    assert parsed["property_count"] == 2
    assert parsed["properties"][0]["name"] == "NAME"
    assert parsed["properties"][0]["type"] == "String"
    assert parsed["properties"][0]["read_only"] is False
    assert parsed["properties"][1]["name"] == "ENUMPROP"
    assert parsed["properties"][1]["type"] == "UInt32"
    assert parsed["properties"][1]["read_only"] is True
    assert parsed["properties"][1]["import_ignore"] is True
    assert parsed["properties"][1]["enum_collection"] == "YesNo"


def test_build_class_catalog_merges_children_and_property_variants() -> None:
    module = _load_dev_module("ma3_terminal_class_catalog")
    node_payloads = [
        {
            "expression": "DataPool()[6][1]",
            "class": "Sequence",
            "object_path": "ShowData.DataPools.Default.Sequences.Sequence 1",
            "children": [
                {"index": 0, "name": "OffCue", "class": "Cue"},
                {"index": 1, "name": "Intro", "class": "Cue"},
            ],
            "properties": [
                {
                    "index": 0,
                    "name": "NAME",
                    "type": "String",
                    "read_only": False,
                    "export_ignore": False,
                    "import_ignore": False,
                    "enum_collection": "",
                },
                {
                    "index": 1,
                    "name": "FADERENABLED",
                    "type": "UInt32",
                    "read_only": True,
                    "export_ignore": False,
                    "import_ignore": False,
                    "enum_collection": "YesNo",
                },
            ],
        },
        {
            "expression": "DataPool()[6][2]",
            "class": "Sequence",
            "object_path": "ShowData.DataPools.Default.Sequences.Sequence 2",
            "children": [
                {"index": 0, "name": "OffCue", "class": "Cue"},
                {"index": 1, "name": "Verse", "class": "Cue"},
            ],
            "properties": [
                {
                    "index": 0,
                    "name": "NAME",
                    "type": "String",
                    "read_only": False,
                    "export_ignore": False,
                    "import_ignore": False,
                    "enum_collection": "",
                },
                {
                    "index": 1,
                    "name": "FADERENABLED",
                    "type": "UInt32",
                    "read_only": True,
                    "export_ignore": False,
                    "import_ignore": False,
                    "enum_collection": "YesNo",
                },
            ],
        },
    ]

    catalog = module.build_class_catalog(node_payloads)

    sequence = catalog["Sequence"]
    assert sequence["node_count"] == 2
    assert sequence["child_classes"]["0"] == ["Cue"]
    assert sequence["child_classes"]["1"] == ["Cue"]
    assert sequence["properties"]["NAME"]["types"] == ["String"]
    assert sequence["properties"]["NAME"]["read_only"] == [False]
    assert sequence["properties"]["FADERENABLED"]["enum_collections"] == ["YesNo"]


def test_build_preset_behavior_summary_tracks_modes_and_references() -> None:
    module = _load_dev_module("ma3_terminal_class_catalog")
    node_payloads = [
        {
            "class": "Preset",
            "dump_properties": [
                {"name": "PRESETMODE", "value": "Universal"},
                {"name": "PRESETMODEINTERNAL", "value": "Universal"},
                {"name": "RECIPETEMPLATE", "value": "No"},
                {"name": "PHASERTRANSFORM", "value": "None"},
                {"name": "STOREDDATA", "value": "Universal"},
                {"name": "REFERENCES", "value": "abc"},
            ],
        },
        {
            "class": "Preset",
            "dump_properties": [
                {"name": "PRESETMODE", "value": "Selective"},
                {"name": "PRESETMODEINTERNAL", "value": "Selective"},
                {"name": "RECIPETEMPLATE", "value": "Yes"},
                {"name": "PHASERTRANSFORM", "value": "None"},
                {"name": "STOREDDATA", "value": "Selective/Global"},
                {"name": "REFERENCES", "value": ""},
            ],
        },
    ]

    summary = module.build_preset_behavior_summary(node_payloads)

    assert summary["sampled_preset_count"] == 2
    assert summary["references_non_empty_count"] == 1
    assert summary["recipe_template_yes_count"] == 1
    assert summary["observed_values"]["PRESETMODE"] == ["Selective", "Universal"]
