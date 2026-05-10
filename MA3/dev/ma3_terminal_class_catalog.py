#!/usr/bin/env python3
"""ma3-terminal-class-catalog: Build a native MA3 class/property catalog from sampled DataPool objects.
Exists because sampled dump trees need a second native pass to recover property types and editability metadata.
Connects sample-crawl evidence to live terminal property-method probing and writes class-centric artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_terminal_datapool_crawl import MA3TerminalSession

PROMPT_PREFIX_RE = r"^[^>\n]*>"
PROPERTY_COUNT_SENTINEL = "__PROPERTY_COUNT__\t"
PROPERTY_SENTINEL = "__PROPERTY__\t"


def load_sample_nodes(sample_json_path: Path) -> list[dict[str, object]]:
    """Load root, top-level, and sampled nodes from one sample crawl artifact."""

    payload = json.loads(sample_json_path.read_text(encoding="utf-8"))
    nodes = [payload["root"], *payload["top_level_nodes"]]
    for subtree_nodes in payload["sampled_subtrees"].values():
        nodes.extend(subtree_nodes)
    return nodes


def build_property_probe_command(expression: str) -> str:
    """Build one Lua command that prints all property metadata for one object."""

    return (
        'Lua "'
        "local h="
        f"{expression}; "
        "local function esc(v) "
        "if v == nil then return '' end; "
        "local s=tostring(v); "
        "s=s:gsub('\\\\','\\\\\\\\'); "
        "s=s:gsub('\\t','\\\\t'); "
        "s=s:gsub('\\n','\\\\n'); "
        "s=s:gsub('\\r','\\\\r'); "
        "return s; "
        "end; "
        "local count=h:PropertyCount(); "
        f"Printf('{PROPERTY_COUNT_SENTINEL}'..count); "
        "for i=0,count-1 do "
        "local info=h:PropertyInfo(i) or {}; "
        f"Printf('{PROPERTY_SENTINEL}'"
        "..i..'\\t'..esc(h:PropertyName(i))"
        "..'\\t'..esc(h:PropertyType(i))"
        "..'\\t'..esc(info.ReadOnly)"
        "..'\\t'..esc(info.ExportIgnore)"
        "..'\\t'..esc(info.ImportIgnore)"
        "..'\\t'..esc(info.EnumCollection)); "
        'end"'
    )


def _strip_prompt_prefix(line: str) -> str:
    line = line.rstrip()
    if ">" not in line:
        return line
    prefix, suffix = line.split(">", 1)
    if "\n" in prefix:
        return line
    return suffix


def _unescape_field(value: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\" and i + 1 < len(value):
            next_char = value[i + 1]
            if next_char == "t":
                chars.append("\t")
                i += 2
                continue
            if next_char == "n":
                chars.append("\n")
                i += 2
                continue
            if next_char == "r":
                chars.append("\r")
                i += 2
                continue
            chars.append(next_char)
            i += 2
            continue
        chars.append(char)
        i += 1
    return "".join(chars)


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def parse_property_probe_output(output_text: str) -> dict[str, object]:
    """Parse one property probe response into structured property metadata."""

    property_count: int | None = None
    properties: list[dict[str, object]] = []
    for raw_line in output_text.splitlines():
        line = _strip_prompt_prefix(raw_line).strip()
        if line.startswith(PROPERTY_COUNT_SENTINEL):
            property_count = int(line.removeprefix(PROPERTY_COUNT_SENTINEL))
            continue
        if not line.startswith(PROPERTY_SENTINEL):
            continue
        fields = line.removeprefix(PROPERTY_SENTINEL).split("\t")
        while len(fields) < 7:
            fields.append("")
        properties.append(
            {
                "index": int(fields[0]),
                "name": _unescape_field(fields[1]),
                "type": _unescape_field(fields[2]),
                "read_only": _parse_bool(fields[3]),
                "export_ignore": _parse_bool(fields[4]),
                "import_ignore": _parse_bool(fields[5]),
                "enum_collection": _unescape_field(fields[6]),
            }
        )
    if property_count is None:
        raise ValueError(f"Property probe output missing count sentinel:\n{output_text}")
    return {"property_count": property_count, "properties": properties}


def collect_node_properties(
    session: MA3TerminalSession,
    *,
    nodes: list[dict[str, object]],
    progress: bool = False,
) -> list[dict[str, object]]:
    """Probe one live property inventory for every sampled node."""

    collected: list[dict[str, object]] = []
    for index, node in enumerate(nodes, start=1):
        if not str(node.get("class", "")).strip():
            continue
        expression = str(node["expression"])
        if progress:
            print(f"PROBE {index}/{len(nodes)}: {expression}", flush=True)
        command = build_property_probe_command(expression)
        output = session.send_command(command)
        parsed = parse_property_probe_output(output)
        collected.append(
            {
                "expression": expression,
                "class": node["class"],
                "name": node["name"],
                "object_path": node["object_path"],
                "path": node["path"],
                "children": node["children"],
                "dump_properties": node["properties"],
                "property_count": parsed["property_count"],
                "properties": parsed["properties"],
                "probe_output": output,
            }
        )
    return collected


def build_class_catalog(node_payloads: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Merge per-node property payloads into a class-centric schema summary."""

    catalog: dict[str, dict[str, object]] = {}
    for node in node_payloads:
        class_name = str(node["class"] or "").strip()
        if not class_name:
            continue
        entry = catalog.setdefault(
            class_name,
            {
                "class": class_name,
                "node_count": 0,
                "expressions": [],
                "object_paths": [],
                "child_classes": defaultdict(set),
                "property_names": defaultdict(
                    lambda: {
                        "indices": set(),
                        "types": set(),
                        "read_only": set(),
                        "export_ignore": set(),
                        "import_ignore": set(),
                        "enum_collections": set(),
                    }
                ),
            },
        )
        entry["node_count"] += 1
        entry["expressions"].append(node["expression"])
        entry["object_paths"].append(node["object_path"])
        for child in node["children"]:
            entry["child_classes"][int(child["index"])].add(str(child["class"]))
        for prop in node["properties"]:
            prop_entry = entry["property_names"][str(prop["name"])]
            prop_entry["indices"].add(int(prop["index"]))
            prop_entry["types"].add(str(prop["type"]))
            prop_entry["read_only"].add(prop["read_only"])
            prop_entry["export_ignore"].add(prop["export_ignore"])
            prop_entry["import_ignore"].add(prop["import_ignore"])
            if prop["enum_collection"]:
                prop_entry["enum_collections"].add(str(prop["enum_collection"]))

    normalized: dict[str, dict[str, object]] = {}
    for class_name, entry in catalog.items():
        normalized[class_name] = {
            "class": class_name,
            "node_count": entry["node_count"],
            "sample_expressions": sorted(entry["expressions"])[:5],
            "sample_object_paths": sorted({path for path in entry["object_paths"] if path})[:5],
            "child_classes": {
                str(index): sorted(values)
                for index, values in sorted(entry["child_classes"].items())
            },
            "properties": {
                name: {
                    "indices": sorted(details["indices"]),
                    "types": sorted(details["types"]),
                    "read_only": sorted(
                        value for value in details["read_only"] if value is not None
                    ),
                    "export_ignore": sorted(
                        value for value in details["export_ignore"] if value is not None
                    ),
                    "import_ignore": sorted(
                        value for value in details["import_ignore"] if value is not None
                    ),
                    "enum_collections": sorted(details["enum_collections"]),
                }
                for name, details in sorted(entry["property_names"].items())
            },
        }
    return normalized


def build_preset_behavior_summary(node_payloads: list[dict[str, object]]) -> dict[str, object]:
    """Summarize observed preset behavior fields from sampled Preset objects."""

    values_by_key: dict[str, set[str]] = defaultdict(set)
    references_non_empty = 0
    recipe_template_yes = 0
    sampled_preset_count = 0
    for node in node_payloads:
        if node["class"] != "Preset":
            continue
        sampled_preset_count += 1
        dump_property_map = {
            str(prop["name"]): str(prop["value"]) for prop in node["dump_properties"]
        }
        for key in (
            "PRESETMODE",
            "PRESETMODEINTERNAL",
            "RECIPETEMPLATE",
            "PHASERTRANSFORM",
            "RELATIVEPHASE",
            "RELATIVESPEED",
            "STOREDDATA",
            "OWNDATAPRESENT",
            "OWNNONCOOKEDDATAPRESENT",
            "ACTION",
        ):
            value = dump_property_map.get(key, "")
            if value:
                values_by_key[key].add(value)
        references_value = dump_property_map.get("REFERENCES", "")
        if references_value:
            references_non_empty += 1
        if dump_property_map.get("RECIPETEMPLATE") == "Yes":
            recipe_template_yes += 1
    return {
        "sampled_preset_count": sampled_preset_count,
        "references_non_empty_count": references_non_empty,
        "recipe_template_yes_count": recipe_template_yes,
        "observed_values": {key: sorted(values) for key, values in sorted(values_by_key.items())},
    }


def write_artifacts(
    output_dir: Path,
    *,
    sample_json_path: Path,
    node_payloads: list[dict[str, object]],
    class_catalog: dict[str, dict[str, object]],
    transcript: list[str],
) -> dict[str, Path]:
    """Write node inventory and class catalog artifacts to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    node_json_path = output_dir / "node_property_inventory.json"
    class_json_path = output_dir / "class_catalog.json"
    class_markdown_path = output_dir / "class_catalog.md"
    preset_markdown_path = output_dir / "preset_pool_notes.md"
    preset_behavior_path = output_dir / "preset_behavior_notes.md"
    transcript_path = output_dir / "terminal_transcript.txt"
    preset_behavior = build_preset_behavior_summary(node_payloads)

    node_json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "source_sample": str(sample_json_path),
                "node_count": len(node_payloads),
                "nodes": node_payloads,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    class_json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "class_count": len(class_catalog),
                "classes": class_catalog,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    class_lines = [
        "# MA3 Class Catalog",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source sample: `{sample_json_path}`",
        f"- Sampled nodes probed: `{len(node_payloads)}`",
        f"- Classes observed: `{len(class_catalog)}`",
        "",
    ]
    for class_name, entry in sorted(class_catalog.items()):
        class_lines.append(f"## {class_name}")
        class_lines.append("")
        class_lines.append(f"- Sampled nodes: `{entry['node_count']}`")
        if entry["sample_expressions"]:
            class_lines.append(f"- Sample expressions: `{', '.join(entry['sample_expressions'])}`")
        if entry["child_classes"]:
            child_parts = [
                f"{index}: {', '.join(classes)}"
                for index, classes in entry["child_classes"].items()
            ]
            class_lines.append(f"- Child classes by index: `{'; '.join(child_parts)}`")
        class_lines.append("- Properties:")
        for property_name, property_entry in entry["properties"].items():
            read_only = (
                ",".join("true" if value else "false" for value in property_entry["read_only"])
                or "unknown"
            )
            property_type = ",".join(property_entry["types"]) or "unknown"
            enum_collection = ",".join(property_entry["enum_collections"]) or "-"
            class_lines.append(
                f"  - `{property_name}` indices={property_entry['indices']} "
                f"type={property_type} read_only={read_only} enum={enum_collection}"
            )
        class_lines.append("")
    class_markdown_path.write_text("\n".join(class_lines), encoding="utf-8")

    preset_lines = [
        "# Preset Pool Notes",
        "",
        "This note isolates the sampled preset-related classes for deeper follow-up.",
        "",
    ]
    for class_name in ("PresetPools", "Presets", "Preset"):
        entry = class_catalog.get(class_name)
        if entry is None:
            continue
        preset_lines.append(f"## {class_name}")
        preset_lines.append("")
        preset_lines.append(f"- Sampled nodes: `{entry['node_count']}`")
        preset_lines.append(f"- Property count entries: `{len(entry['properties'])}`")
        for property_name, property_entry in list(entry["properties"].items())[:25]:
            preset_lines.append(
                f"- `{property_name}`: types={property_entry['types']} "
                f"read_only={property_entry['read_only']} enum={property_entry['enum_collections']}"
            )
        preset_lines.append("")
    preset_markdown_path.write_text("\n".join(preset_lines), encoding="utf-8")

    preset_behavior_lines = [
        "# Preset Behavior Notes",
        "",
        "Observed behavior-oriented preset values from the sampled native Preset objects.",
        "",
        f"- Sampled presets: `{preset_behavior['sampled_preset_count']}`",
        f"- Non-empty `REFERENCES`: `{preset_behavior['references_non_empty_count']}`",
        f"- `RECIPETEMPLATE = Yes`: `{preset_behavior['recipe_template_yes_count']}`",
        "",
    ]
    for key, values in preset_behavior["observed_values"].items():
        preset_behavior_lines.append(f"- `{key}`: `{', '.join(values)}`")
    preset_behavior_lines.extend(
        [
            "",
            "Interpretation from the sampled show state:",
            "",
            "- Non-empty `REFERENCES` already appear in multiple presets, so cross-object preset relationships are normal.",
            "- `PRESETMODE` is not fixed to one value; the sample includes `Universal`, `Selective`, and `Global` variants.",
            "- `STOREDDATA` also varies, including mixed forms like `Selective/Global`.",
            "- The sampled pool does not yet prove a live `RECIPETEMPLATE = Yes` object, so recipe-template behavior still needs a targeted show example.",
            "- Phaser-capable fields are present in the class schema even when the sampled objects keep many of them at `None`.",
            "",
        ]
    )
    preset_behavior_path.write_text("\n".join(preset_behavior_lines), encoding="utf-8")
    transcript_path.write_text("\n\n".join(transcript), encoding="utf-8")
    return {
        "node_inventory": node_json_path,
        "class_catalog": class_json_path,
        "class_markdown": class_markdown_path,
        "preset_notes": preset_markdown_path,
        "preset_behavior": preset_behavior_path,
        "transcript": transcript_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe MA3 sampled objects for native property metadata and build a class catalog."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=Path("artifacts") / "ma3-terminal-crawl" / "sample" / "datapool_sample_crawl.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-terminal-crawl" / "class-catalog",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the native MA3 class catalog from one sample crawl artifact."""

    args = _build_parser().parse_args(argv)
    sample_nodes = load_sample_nodes(args.sample_json)
    probe_nodes = [node for node in sample_nodes if str(node.get("class", "")).strip()]
    files: dict[str, Path]
    with MA3TerminalSession(host=args.host, timeout_seconds=args.timeout_seconds) as session:
        node_payloads = collect_node_properties(session, nodes=probe_nodes, progress=True)
        class_catalog = build_class_catalog(node_payloads)
        files = write_artifacts(
            args.output_dir,
            sample_json_path=args.sample_json,
            node_payloads=node_payloads,
            class_catalog=class_catalog,
            transcript=session.transcript,
        )
    print(
        json.dumps(
            {
                "host": args.host,
                "sample_json": str(args.sample_json),
                "output_dir": str(args.output_dir),
                "sample_node_count": len(sample_nodes),
                "probed_node_count": len(probe_nodes),
                "class_count": len(class_catalog),
                "files": {key: str(value) for key, value in files.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
