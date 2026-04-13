from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_ACTIONS = {
    "add_song_from_path",
    "classify_drum_events",
    "extract_drum_events",
    "extract_stems",
    "trigger_action",
    "select_first_event",
    "nudge_selected_events",
    "duplicate_selected_events",
    "open_push_surface",
    "open_pull_surface",
    "enable_sync",
    "disable_sync",
    "screenshot",
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
            raise ValueError(f"Step {index} action '{action}' is unsupported; supported actions: {supported}")
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
    if action == "add_song_from_path":
        _require_string(params, "title", step_index=step_index)
        _require_string(params, "audio_path", step_index=step_index)
    elif action in {"extract_stems", "extract_drum_events", "classify_drum_events"}:
        _require_string(params, "layer_id", step_index=step_index)
    elif action == "trigger_action":
        value = params.get("action_id")
        if value not in {"new", "open", "save", "save_as"}:
            raise ValueError(f"Step {step_index} trigger_action requires params.action_id in new/open/save/save_as")
    elif action == "select_first_event":
        if "layer_id" in params and not isinstance(params["layer_id"], str):
            raise ValueError(f"Step {step_index} select_first_event params.layer_id must be a string")
    elif action == "nudge_selected_events":
        if params.get("direction") not in {"left", "right"}:
            raise ValueError(f"Step {step_index} nudge_selected_events requires params.direction in left/right")
        if "steps" in params and (not isinstance(params["steps"], int) or int(params["steps"]) < 1):
            raise ValueError(f"Step {step_index} nudge_selected_events params.steps must be an integer >= 1")
    elif action == "duplicate_selected_events":
        if "steps" in params and (not isinstance(params["steps"], int) or int(params["steps"]) < 1):
            raise ValueError(f"Step {step_index} duplicate_selected_events params.steps must be an integer >= 1")
    elif action in {"open_push_surface", "open_pull_surface"}:
        if "layer_id" in params and not isinstance(params["layer_id"], str):
            raise ValueError(f"Step {step_index} {action} params.layer_id must be a string")
    elif action == "screenshot":
        _require_string(params, "filename", step_index=step_index)
