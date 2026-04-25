"""Runtime-model dropdown option helpers for object-action settings.
Exists to keep model-file discovery and option labeling out of the settings runtime mixin.
Connects pipeline knob metadata to installed-model picker choices for Qt settings forms.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from echozero.pipelines.params import Knob, KnobWidget

if TYPE_CHECKING:
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption


def build_runtime_model_picker_options(
    *,
    knob: Knob,
    value: object,
) -> tuple["ObjectActionSettingOption", ...]:
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption

    if not supports_runtime_model_picker(knob):
        return ()
    models_root = resolve_installed_models_root()
    candidate_paths = discover_runtime_model_paths(models_root=models_root, knob=knob)
    if not candidate_paths:
        return ()
    options = [ObjectActionSettingOption(value="", label="Select Model")]
    option_values = {""}
    for path in candidate_paths:
        resolved = str(path)
        if resolved in option_values:
            continue
        options.append(
            ObjectActionSettingOption(
                value=resolved,
                label=runtime_model_option_label(path=path, models_root=models_root),
            )
        )
        option_values.add(resolved)
    current_value = str(value or "").strip()
    if current_value and current_value not in option_values:
        current_path = Path(current_value)
        release_date = runtime_model_release_date(path=current_path)
        current_label = Path(current_value).name or current_value
        if release_date is not None:
            current_label = f"{current_label} · Released {release_date}"
        options.append(
            ObjectActionSettingOption(
                value=current_value,
                label=f"Current: {current_label}",
            )
        )
    return tuple(options)


def supports_runtime_model_picker(knob: Knob) -> bool:
    if knob.widget is KnobWidget.MODEL_PICKER:
        return True
    if knob.widget is not KnobWidget.FILE_PICKER:
        return False
    file_types = {str(file_type).strip().lower() for file_type in (knob.file_types or ())}
    return ".manifest.json" in file_types or ".pth" in file_types


def resolve_installed_models_root() -> Path:
    from echozero.application.timeline.object_action_settings_service import (
        ensure_installed_models_dir,
    )

    return ensure_installed_models_dir().resolve()


def discover_runtime_model_paths(*, models_root: Path, knob: Knob) -> tuple[Path, ...]:
    patterns = runtime_model_glob_patterns(knob)
    if not patterns:
        return ()
    discovered: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(models_root.rglob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            normalized = str(resolved)
            if normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(resolved)
    return tuple(discovered)


def runtime_model_glob_patterns(knob: Knob) -> tuple[str, ...]:
    file_types = {str(file_type).strip().lower() for file_type in (knob.file_types or ())}
    patterns: list[str] = []
    if ".manifest.json" in file_types:
        patterns.append("*.manifest.json")
    if ".pth" in file_types:
        patterns.append("*.pth")
    if not patterns and knob.widget is KnobWidget.MODEL_PICKER:
        patterns.extend(["*.manifest.json", "*.pth"])
    return tuple(patterns)


def runtime_model_option_label(*, path: Path, models_root: Path) -> str:
    label = _runtime_model_relative_label(path=path, models_root=models_root)
    release_date = runtime_model_release_date(path=path)
    if release_date is None:
        return label
    return f"{label} · Released {release_date}"


def runtime_model_release_date(*, path: Path) -> str | None:
    if not path.name.endswith(".manifest.json"):
        return None
    manifest = _load_manifest_payload(path)
    if manifest is None:
        return None
    for key in ("releasedAt", "released_at", "releaseDate", "release_date", "createdAt", "created_at"):
        normalized = _normalize_manifest_date(manifest.get(key))
        if normalized is not None:
            return normalized
    return None


def _runtime_model_relative_label(*, path: Path, models_root: Path) -> str:
    try:
        relative = path.relative_to(models_root)
        return str(relative)
    except ValueError:
        return str(path)


def _load_manifest_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_manifest_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        pass
    if len(text) >= 10:
        candidate = text[:10]
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            return None
    return None
