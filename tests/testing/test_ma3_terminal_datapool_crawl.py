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


def test_parse_children_probe_output_extracts_ordered_children() -> None:
    module = _load_dev_module("ma3_terminal_datapool_crawl")
    output = "\n".join(
        [
            'OK:Lua "..."',
            "Admin[Fixture]>__CHILDREN_COUNT__\t2",
            "1|MarkerTrack|Marker",
            "2|Track|1 'AUTOMATOR'",
        ]
    )

    children = module.parse_children_probe_output(output)

    assert children == [
        {"ordinal": 1, "class": "MarkerTrack", "name": "Marker"},
        {"ordinal": 2, "class": "Track", "name": "1 'AUTOMATOR'"},
    ]
