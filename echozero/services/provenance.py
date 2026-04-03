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
from typing import Any

from echozero.persistence.entities import LayerRecord


def initialize_generated_layer_state(
    layer: LayerRecord,
    *,
    pipeline_id: str,
    output_name: str,
    block_id: str,
    data_type: str,
    source_song_version_id: str,
    source_layer_id: str | None = None,
    source_run_id: str | None = None,
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
    provenance = {
        "pipeline_id": pipeline_id,
        "output_name": output_name,
        "block_id": block_id,
        "data_type": data_type,
        "source_song_version_id": source_song_version_id,
        "source_layer_id": source_layer_id,
        "source_run_id": source_run_id,
    }
    return replace(layer, state_flags=state_flags, provenance=provenance)


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
