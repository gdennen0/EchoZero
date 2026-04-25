"""Application-side provenance and freshness helpers.

This module intentionally lives outside the engine.
It provides small, explicit helpers for editor/application semantics:
- generation provenance
- freshness / stale markers
- manual modification markers
- future song-version remap hooks
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from echozero.persistence.entities import LayerRecord

GENERATED_ANALYSIS_PROVENANCE_SCHEMA = "echozero.generated-analysis.v1"
ANALYSIS_BUILD_SCHEMA = "echozero.analysis-build.v1"
MODEL_ARTIFACT_SCHEMA = "echozero.model-artifact.v1"


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_timestamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return _normalize_optional_text(value)


def build_model_artifact(
    *,
    role: str,
    kind: str,
    locator: str,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Return one canonical artifact reference for generated analysis provenance."""

    return {
        "schema": MODEL_ARTIFACT_SCHEMA,
        "role": role,
        "kind": kind,
        "locator": locator,
        "content_type": _normalize_optional_text(content_type),
    }


def build_analysis_build(
    *,
    pipeline_id: str,
    pipeline_config_id: str | None,
    block_id: str,
    block_type: str | None,
    output_name: str,
    data_type: str,
    execution_id: str,
    build_id: str | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Return the canonical build identity for one generated analysis result."""

    resolved_execution_id = _normalize_optional_text(execution_id)
    resolved_build_id = _normalize_optional_text(build_id) or resolved_execution_id
    return {
        "schema": ANALYSIS_BUILD_SCHEMA,
        "build_id": resolved_build_id,
        "execution_id": resolved_execution_id,
        "pipeline_id": _normalize_optional_text(pipeline_id),
        "pipeline_config_id": _normalize_optional_text(pipeline_config_id),
        "block_id": _normalize_optional_text(block_id),
        "block_type": _normalize_optional_text(block_type),
        "output_name": _normalize_optional_text(output_name),
        "data_type": _normalize_optional_text(data_type),
        "generated_at": _normalize_timestamp(generated_at),
    }


def build_legacy_source_pipeline(
    analysis_build: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Mirror the canonical build into the legacy layer source_pipeline field."""

    if not analysis_build:
        return None
    return {
        "pipeline_id": analysis_build.get("pipeline_id"),
        "pipeline_config_id": analysis_build.get("pipeline_config_id"),
        "block_id": analysis_build.get("block_id"),
        "block_type": analysis_build.get("block_type"),
        "output_name": analysis_build.get("output_name"),
        "data_type": analysis_build.get("data_type"),
        "analysis_build_id": analysis_build.get("build_id"),
        "execution_id": analysis_build.get("execution_id"),
        "generated_at": analysis_build.get("generated_at"),
    }


def build_generated_layer_provenance(
    *,
    source_song_version_id: str,
    analysis_build: dict[str, Any],
    source_layer_id: str | None = None,
    source_run_id: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the canonical persisted provenance payload for one generated layer."""

    resolved_source_run_id = (
        _normalize_optional_text(source_run_id)
        or _normalize_optional_text(analysis_build.get("execution_id"))
        or _normalize_optional_text(analysis_build.get("build_id"))
    )
    return {
        "schema": GENERATED_ANALYSIS_PROVENANCE_SCHEMA,
        "source_layer_id": _normalize_optional_text(source_layer_id),
        "source_song_version_id": _normalize_optional_text(source_song_version_id),
        "source_run_id": resolved_source_run_id,
        "analysis_build": dict(analysis_build),
        "artifacts": list(artifacts or []),
        # Compatibility aliases for existing app/UI readers.
        "pipeline_id": analysis_build.get("pipeline_id"),
        "output_name": analysis_build.get("output_name"),
        "block_id": analysis_build.get("block_id"),
        "data_type": analysis_build.get("data_type"),
    }


def initialize_generated_layer_state(
    layer: LayerRecord,
    *,
    pipeline_id: str,
    pipeline_config_id: str | None = None,
    output_name: str,
    block_id: str,
    block_type: str | None = None,
    data_type: str,
    analysis_build_id: str | None = None,
    source_song_version_id: str,
    source_layer_id: str | None = None,
    source_run_id: str | None = None,
    generated_at: datetime | str | None = None,
    source_audio_path: str | None = None,
) -> LayerRecord:
    """Attach baseline provenance/state to a newly generated layer.

    This is intentionally minimal for the current pass.
    It establishes explicit slots for concepts we already know we need.
    """
    state_flags = {
        "derived": True,
        "stale": False,
        "manually_modified": False,
        "source_main_changed": False,
    }
    artifacts: list[dict[str, Any]] = []
    normalized_source_audio_path = _normalize_optional_text(source_audio_path)
    if normalized_source_audio_path is not None:
        artifacts.append(
            build_model_artifact(
                role="source_audio",
                kind="audio_file",
                locator=normalized_source_audio_path,
                content_type="audio/*",
            )
        )
    analysis_build = build_analysis_build(
        pipeline_id=pipeline_id,
        pipeline_config_id=pipeline_config_id,
        block_id=block_id,
        block_type=block_type,
        output_name=output_name,
        data_type=data_type,
        execution_id=source_run_id or "",
        build_id=analysis_build_id,
        generated_at=generated_at or layer.created_at,
    )
    provenance = build_generated_layer_provenance(
        source_song_version_id=source_song_version_id,
        source_layer_id=source_layer_id,
        source_run_id=source_run_id,
        analysis_build=analysis_build,
        artifacts=artifacts,
    )
    return replace(
        layer,
        state_flags=state_flags,
        source_pipeline=build_legacy_source_pipeline(analysis_build),
        provenance=provenance,
    )


def mark_layer_stale(layer: LayerRecord, *, reason: str, upstream_layer_id: str | None = None) -> LayerRecord:
    state_flags = dict(layer.state_flags)
    state_flags.update({
        "stale": True,
        "source_main_changed": True,
        "stale_reason": reason,
        "stale_upstream_layer_id": upstream_layer_id,
    })
    return replace(layer, state_flags=state_flags)


def clear_layer_stale(layer: LayerRecord) -> LayerRecord:
    state_flags = dict(layer.state_flags)
    state_flags["stale"] = False
    state_flags["source_main_changed"] = False
    state_flags.pop("stale_reason", None)
    state_flags.pop("stale_upstream_layer_id", None)
    return replace(layer, state_flags=state_flags)


def mark_layer_manually_modified(layer: LayerRecord) -> LayerRecord:
    state_flags = dict(layer.state_flags)
    state_flags["manually_modified"] = True
    return replace(layer, state_flags=state_flags)


def build_song_version_rebuild_plan(*, previous_version_id: str, new_version_id: str, pipeline_config_ids: list[str]) -> dict[str, Any]:
    """Return a minimal stub plan for rebuilding a new song version from copied configs.

    Current policy:
    - new SongVersion starts as blank slate
    - copied configs are the rebuild substrate
    - future remap/alignment tools can hang off this plan structure
    """
    return {
        "mode": "blank_slate_with_rerun",
        "previous_version_id": previous_version_id,
        "new_version_id": new_version_id,
        "pipeline_config_ids": list(pipeline_config_ids),
        "remap": {
            "status": "deferred",
            "strategy": None,
            "notes": "Future hook for beat/section/alignment-assisted remap.",
        },
    }
