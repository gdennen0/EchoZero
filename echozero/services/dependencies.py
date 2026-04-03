"""Application-side dependency and stale propagation helpers.

Ground rules:
- downstream becomes stale only when upstream MAIN changes
- a new non-main take upstream does not make anything stale
- provenance/source references are the backbone for future UI inspection
"""

from __future__ import annotations

from dataclasses import replace

from echozero.persistence.entities import LayerRecord
from echozero.services.provenance import mark_layer_stale
from echozero.takes import Take


def capture_main_lineage(layer: LayerRecord, main_take: Take | None) -> LayerRecord:
    """Snapshot the currently active main lineage for a generated layer.

    This gives us something concrete to compare against later when upstream main changes.
    """
    provenance = dict(layer.provenance)
    provenance['current_main_take_id'] = main_take.id if main_take else None
    provenance['current_main_run_id'] = main_take.source.run_id if (main_take and main_take.source) else None
    return replace(layer, provenance=provenance)


def upstream_main_change_requires_stale(*, previous_main_take: Take | None, new_main_take: Take | None) -> bool:
    """Return True only when upstream main actually changed.

    Presence of a new non-main take is irrelevant; compare the actual main take identity.
    """
    previous_id = previous_main_take.id if previous_main_take else None
    new_id = new_main_take.id if new_main_take else None
    return previous_id != new_id


def mark_dependents_stale_on_upstream_main_change(
    *,
    layers: list[LayerRecord],
    upstream_layer_id: str,
    previous_main_take: Take | None,
    new_main_take: Take | None,
) -> list[LayerRecord]:
    """Mark direct dependents stale only if upstream main changed."""
    if not upstream_main_change_requires_stale(
        previous_main_take=previous_main_take,
        new_main_take=new_main_take,
    ):
        return layers

    updated: list[LayerRecord] = []
    for layer in layers:
        if layer.provenance.get('source_layer_id') == upstream_layer_id:
            updated.append(
                mark_layer_stale(
                    layer,
                    reason='Upstream main changed',
                    upstream_layer_id=upstream_layer_id,
                )
            )
        else:
            updated.append(layer)
    return updated
