"""Application-side manual edit actions.

These actions intentionally mark layer state as manually modified.
They do not try to be a full editor command system yet.
"""

from __future__ import annotations

from echozero.persistence.session import ProjectStorage
from echozero.services.provenance import mark_layer_manually_modified


def mark_layer_as_manually_modified(session: ProjectStorage, *, layer_id: str):
    layer = session.layers.get(layer_id)
    if layer is None:
        raise ValueError(f'Layer not found: {layer_id}')
    updated = mark_layer_manually_modified(layer)
    session.layers.update(updated)
    session.commit()
    return updated
