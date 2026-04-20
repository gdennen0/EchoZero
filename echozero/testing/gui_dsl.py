from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_ACTIONS = {
    "song.add",
    "timeline.classify_drum_events",
    "timeline.duplicate_selection",
    "timeline.extract_drum_events",
    "timeline.extract_stems",
    "timeline.nudge_selection",
    "selection.first_event",
    "transfer.workspace_open",
    "transfer.plan_apply",
    "sync.enable",
    "sync.disable",
    "capture.screenshot",
}


@dataclass(slots=True, frozen=True)
class ScenarioStep:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass(slots=True, frozen=True)
class GuiScenario:
    name: str
    seed: int
    steps: tuple[ScenarioStep, ...]
    source_path: Path


def load_scenario(path: str | Path) -> GuiScenario:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scenario payload must be a JSON object")

    name = _require_string(payload, "name")
    seed = int(payload.get("seed", 0))
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Scenario requires a non-empty 'steps' list")

    steps: list[ScenarioStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {index} must be a JSON object")
        action = _require_string(raw_step, "action", step_index=index)
        if action not in SUPPORTED_ACTIONS:
            supported = ", ".join(sorted(SUPPORTED_ACTIONS))
            raise ValueError(
                f"Step {index} action '{action}' is unsupported; supported actions: {supported}"
            )
        params = raw_step.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Step {index} params must be an object")
        label = raw_step.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"Step {index} label must be a string when provided")
        _validate_step(action=action, params=params, step_index=index)
        steps.append(ScenarioStep(action=action, params=dict(params), label=label))

    return GuiScenario(name=name, seed=seed, steps=tuple(steps), source_path=source_path)


def _require_string(payload: dict[str, Any], key: str, *, step_index: int | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        prefix = f"Step {step_index} " if step_index is not None else ""
        raise ValueError(f"{prefix}requires non-empty string field '{key}'")
    return value.strip()


def _validate_step(*, action: str, params: dict[str, Any], step_index: int) -> None:
    if action == "song.add":
        _require_string(params, "title", step_index=step_index)
        _require_string(params, "audio_path", step_index=step_index)
    elif action in {
        "timeline.extract_stems",
        "timeline.extract_drum_events",
        "timeline.classify_drum_events",
        "selection.first_event",
        "transfer.workspace_open",
    }:
        _require_layer_target(params, step_index=step_index, action=action)
        if action == "timeline.classify_drum_events":
            _require_string(params, "model_path", step_index=step_index)
        if action == "transfer.workspace_open" and params.get("direction") not in {"push", "pull"}:
            raise ValueError(
                f"Step {step_index} transfer.workspace_open requires params.direction in push/pull"
            )
    elif action == "timeline.nudge_selection":
        if params.get("direction") not in {"left", "right"}:
            raise ValueError(f"Step {step_index} {action} requires params.direction in left/right")
        if "steps" in params and (
            not isinstance(params["steps"], int) or int(params["steps"]) < 1
        ):
            raise ValueError(f"Step {step_index} {action} params.steps must be an integer >= 1")
    elif action == "timeline.duplicate_selection":
        if "steps" in params and (
            not isinstance(params["steps"], int) or int(params["steps"]) < 1
        ):
            raise ValueError(f"Step {step_index} {action} params.steps must be an integer >= 1")
    elif action == "capture.screenshot":
        _require_string(params, "filename", step_index=step_index)


def _require_layer_target(params: dict[str, Any], *, step_index: int, action: str) -> None:
    layer_id = params.get("layer_id")
    layer_title = params.get("layer_title")
    has_layer_id = isinstance(layer_id, str) and bool(layer_id.strip())
    has_layer_title = isinstance(layer_title, str) and bool(layer_title.strip())
    if not has_layer_id and not has_layer_title:
        raise ValueError(
            f"Step {step_index} {action} requires params.layer_id or params.layer_title"
        )
    if layer_id is not None and not isinstance(layer_id, str):
        raise ValueError(f"Step {step_index} {action} params.layer_id must be a string")
    if layer_title is not None and not isinstance(layer_title, str):
        raise ValueError(f"Step {step_index} {action} params.layer_title must be a string")
