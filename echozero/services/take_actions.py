"""Application-side take actions.

This module owns take/main transitions that matter to editor truth.
It is intentionally application-layer code, not engine code.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.persistence.entities import LayerRecord
from echozero.persistence.session import ProjectStorage
from echozero.services.dependencies import (
    capture_main_lineage,
    mark_dependents_stale_on_upstream_main_change,
)
from echozero.takes import Take


def promote_take_to_main(session: ProjectStorage, *, layer_id: str, take_id: str) -> tuple[LayerRecord, list[LayerRecord]]:
    """Promote a take to main and mark direct dependents stale if the main changed.

    Returns:
        (updated_parent_layer, updated_layers)
    where updated_layers includes any stale-marked dependents.
    """
    layer = session.layers.get(layer_id)
    if layer is None:
        raise ValueError(f'Layer not found: {layer_id}')

    takes = session.takes.list_by_layer(layer_id)
    previous_main = next((t for t in takes if t.is_main), None)
    target = next((t for t in takes if t.id == take_id), None)
    if target is None:
        raise ValueError(f'Take not found: {take_id}')

    if previous_main is not None and previous_main.id == target.id:
        return layer, session.layers.list_by_version(layer.song_version_id)

    for take in takes:
        if take.id == take_id:
            session.takes.update(replace(take, is_main=True))
        elif take.is_main:
            session.takes.update(replace(take, is_main=False))

    new_main = session.takes.get(take_id)
    assert new_main is not None

    updated_parent = capture_main_lineage(layer, new_main)
    session.layers.update(updated_parent)

    all_layers = session.layers.list_by_version(layer.song_version_id)
    stale_updated = mark_dependents_stale_on_upstream_main_change(
        layers=all_layers,
        upstream_layer_id=layer_id,
        previous_main_take=previous_main,
        new_main_take=new_main,
    )

    by_id = {existing.id: existing for existing in all_layers}
    result_layers: list[LayerRecord] = []
    for candidate in stale_updated:
        if candidate.id == updated_parent.id:
            result_layers.append(updated_parent)
            continue
        if candidate.state_flags != by_id[candidate.id].state_flags:
            session.layers.update(candidate)
        result_layers.append(candidate)

    session.commit()
    return updated_parent, result_layers
