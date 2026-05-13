"""Runtime-model dropdown option helpers for object-action settings.
Exists to keep model-file discovery and option labeling out of the settings runtime mixin.
Connects pipeline knob metadata to installed-model picker choices for Qt settings forms.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from echozero.models.classifier_model_catalog import (
    ClassifierModelCandidate,
    build_runtime_classifier_model_catalog,
    describe_binary_drum_manifest_candidate,
)
from echozero.pipelines.params import Knob, KnobWidget

if TYPE_CHECKING:
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption


def build_runtime_model_picker_options(
    *,
    knob: Knob,
    value: object,
    key: str = "",
    action_id: str = "",
) -> tuple["ObjectActionSettingOption", ...]:
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption

    if not supports_runtime_model_picker(knob):
        return ()
    models_root = resolve_installed_models_root()
    drum_label = _binary_drum_label_for_model_key(key)
    if drum_label is not None and _uses_binary_drum_model_catalog(action_id):
        return _build_binary_drum_model_picker_options(
            label=drum_label,
            value=value,
            models_root=models_root,
        )
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


def _build_binary_drum_model_picker_options(
    *,
    label: str,
    value: object,
    models_root: Path,
) -> tuple["ObjectActionSettingOption", ...]:
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption

    catalog = build_runtime_classifier_model_catalog(models_dir=models_root)
    candidates = catalog.candidates_for_label(label)
    option_label = f"Select {_display_label(label)} Model"
    if not candidates:
        option_label = f"No compatible {_display_label(label)} model installed"
    options = [
        ObjectActionSettingOption(
            value="",
            label=option_label,
            metadata={"label": label, "status": "missing" if not candidates else "select"},
        )
    ]
    option_values = {""}
    for candidate in candidates:
        options.append(_option_for_binary_candidate(candidate))
        option_values.add(str(candidate.manifest_path))

    current_value = str(value or "").strip()
    if current_value and current_value not in option_values:
        current_candidate = describe_binary_drum_manifest_candidate(
            Path(current_value),
            label=label,
        )
        if current_candidate is not None and current_candidate.is_compatible:
            options.append(_option_for_binary_candidate(current_candidate, prefix="Current"))
    return tuple(options)


def _option_for_binary_candidate(
    candidate: ClassifierModelCandidate,
    *,
    prefix: str = "",
) -> "ObjectActionSettingOption":
    from echozero.application.timeline.object_actions.settings import ObjectActionSettingOption

    label_parts: list[str] = []
    if prefix:
        label_parts.append(prefix)
    elif candidate.is_current_default:
        label_parts.append("Current Default")
    label_parts.append(candidate.display_name)
    readiness = (
        "Ready"
        if candidate.is_compatible
        else candidate.compatibility_status.replace("_", " ").title()
    )
    label_parts.append(readiness)
    if candidate.eval_summary:
        label_parts.append(candidate.eval_summary)
    release_date = candidate.released_at or candidate.created_at
    if release_date:
        label_parts.append(f"Released {release_date}")
    return ObjectActionSettingOption(
        value=str(candidate.manifest_path),
        label=" · ".join(label_parts),
        metadata={
            "family": candidate.family,
            "label": candidate.label,
            "model_instance": candidate.model_instance,
            "status": candidate.compatibility_status,
            "is_current_default": candidate.is_current_default,
            "eval_score": candidate.eval_score,
        },
    )


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
    for key in (
        "releasedAt",
        "released_at",
        "releaseDate",
        "release_date",
        "createdAt",
        "created_at",
    ):
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


def _uses_binary_drum_model_catalog(action_id: str) -> bool:
    return action_id in {
        "timeline.extract_classified_drums",
        "timeline.extract_song_drum_events",
    }


def _binary_drum_label_for_model_key(key: str) -> str | None:
    text = str(key or "").strip().lower()
    if not text.endswith("_model_path"):
        return None
    label = text[: -len("_model_path")]
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    if label in {"kick", "snare", "clap", "cymbal"}:
        return label
    return None


def _display_label(label: str) -> str:
    return str(label).strip().replace("_", " ").title()
