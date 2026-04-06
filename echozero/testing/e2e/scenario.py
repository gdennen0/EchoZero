"""Scenario model and file loading helpers for E2E runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

StepKind = Literal["act", "assert", "capture", "wait"]


@dataclass(slots=True)
class ActStep:
    kind: Literal["act"] = "act"
    name: str = ""
    action: str = ""
    target: str | None = None
    value: Any = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssertStep:
    kind: Literal["assert"] = "assert"
    name: str = ""
    query: str = ""
    expected: Any = None
    comparator: str = "equals"


@dataclass(slots=True)
class CaptureStep:
    kind: Literal["capture"] = "capture"
    name: str = ""
    artifact: str = "screenshot"
    path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WaitStep:
    kind: Literal["wait"] = "wait"
    name: str = ""
    duration_ms: int = 0
    until_query: str | None = None
    expected: Any = None
    timeout_ms: int = 1000
    poll_interval_ms: int = 50


ScenarioStep = ActStep | AssertStep | CaptureStep | WaitStep


@dataclass(slots=True)
class Scenario:
    name: str
    steps: list[ScenarioStep]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def load_scenario(path: str | Path) -> Scenario:
    source_path = Path(path)
    raw = source_path.read_text(encoding="utf-8")
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(raw, source_path)
    else:
        raise ValueError(f"Unsupported scenario file extension: {source_path.suffix}")
    scenario = scenario_from_mapping(payload)
    scenario.source_path = source_path
    return scenario


def scenario_from_mapping(payload: dict[str, Any]) -> Scenario:
    if not isinstance(payload, dict):
        raise TypeError("Scenario payload must be a mapping.")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("Scenario requires a non-empty name.")
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("Scenario requires a non-empty steps list.")
    steps = [_step_from_mapping(index, item) for index, item in enumerate(steps_payload)]
    description = str(payload.get("description", ""))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Scenario metadata must be a mapping.")
    return Scenario(name=name, description=description, metadata=dict(metadata), steps=steps)


def _step_from_mapping(index: int, payload: Any) -> ScenarioStep:
    if not isinstance(payload, dict):
        raise TypeError(f"Step {index} must be a mapping.")
    kind = payload.get("kind")
    if kind == "act":
        action = str(payload.get("action", "")).strip()
        if not action:
            raise ValueError(f"Step {index} act requires an action.")
        return ActStep(
            name=str(payload.get("name", f"act-{index}")),
            action=action,
            target=_optional_str(payload.get("target")),
            value=payload.get("value"),
            args=_mapping(payload.get("args"), f"Step {index} act args"),
        )
    if kind == "assert":
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError(f"Step {index} assert requires a query.")
        return AssertStep(
            name=str(payload.get("name", f"assert-{index}")),
            query=query,
            expected=payload.get("expected"),
            comparator=str(payload.get("comparator", "equals")),
        )
    if kind == "capture":
        artifact = str(payload.get("artifact", "screenshot")).strip()
        if not artifact:
            raise ValueError(f"Step {index} capture requires an artifact kind.")
        return CaptureStep(
            name=str(payload.get("name", f"capture-{index}")),
            artifact=artifact,
            path=_optional_str(payload.get("path")),
            options=_mapping(payload.get("options"), f"Step {index} capture options"),
        )
    if kind == "wait":
        return WaitStep(
            name=str(payload.get("name", f"wait-{index}")),
            duration_ms=int(payload.get("duration_ms", 0)),
            until_query=_optional_str(payload.get("until_query")),
            expected=payload.get("expected"),
            timeout_ms=int(payload.get("timeout_ms", 1000)),
            poll_interval_ms=int(payload.get("poll_interval_ms", 50)),
        )
    raise ValueError(f"Unsupported step kind at index {index}: {kind!r}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping.")
    return dict(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_yaml(raw: str, source_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        payload = _load_minimal_yaml(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise TypeError("YAML scenario payload must be a mapping.")
    return payload


def _load_minimal_yaml(raw: str) -> dict[str, Any]:
    lines = [
        _YamlLine(indent=len(line) - len(line.lstrip(" ")), text=line.strip())
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("YAML scenario payload is empty.")
    value, index = _parse_yaml_block(lines, 0, lines[0].indent)
    if index != len(lines):
        raise ValueError("Unexpected trailing YAML content.")
    if not isinstance(value, dict):
        raise TypeError("YAML scenario payload must be a mapping.")
    return value


@dataclass(slots=True)
class _YamlLine:
    indent: int
    text: str


def _parse_yaml_block(lines: list[_YamlLine], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        raise ValueError("Unexpected end of YAML content.")
    if lines[start].indent != indent:
        raise ValueError("Invalid YAML indentation.")
    if lines[start].text.startswith("- "):
        return _parse_yaml_list(lines, start, indent)
    return _parse_yaml_mapping(lines, start, indent)


def _parse_yaml_mapping(lines: list[_YamlLine], start: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent != indent or line.text.startswith("- "):
            raise ValueError("Invalid YAML mapping structure.")
        key, has_value, value_text = _split_yaml_key_value(line.text)
        if not has_value:
            value, index = _parse_yaml_block(lines, index + 1, indent + 2)
        elif value_text == "":
            value = ""
            index += 1
        else:
            value = _parse_yaml_scalar(value_text)
            index += 1
        mapping[key] = value
    return mapping, index


def _parse_yaml_list(lines: list[_YamlLine], start: int, indent: int) -> tuple[list[Any], int]:
    values: list[Any] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent != indent or not line.text.startswith("- "):
            raise ValueError("Invalid YAML list structure.")
        remainder = line.text[2:].strip()
        if not remainder:
            value, index = _parse_yaml_block(lines, index + 1, indent + 2)
            values.append(value)
            continue
        if ":" in remainder:
            key, has_value, value_text = _split_yaml_key_value(remainder)
            item: dict[str, Any] = {}
            if not has_value:
                nested, index = _parse_yaml_block(lines, index + 1, indent + 4)
                item[key] = nested
            else:
                item[key] = _parse_yaml_scalar(value_text) if value_text else ""
                index += 1
            while index < len(lines) and lines[index].indent >= indent + 2:
                nested_line = lines[index]
                if nested_line.indent == indent and nested_line.text.startswith("- "):
                    break
                nested_map, index = _parse_yaml_mapping(lines, index, indent + 2)
                item.update(nested_map)
            values.append(item)
            continue
        values.append(_parse_yaml_scalar(remainder))
        index += 1
    return values, index


def _split_yaml_key_value(text: str) -> tuple[str, bool, str]:
    if ":" not in text:
        raise ValueError(f"Invalid YAML mapping entry: {text!r}")
    key, value = text.split(":", 1)
    stripped = value.strip()
    return key.strip(), stripped != "", stripped


def _parse_yaml_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text
