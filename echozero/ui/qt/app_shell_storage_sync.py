"""Storage-backed timeline sync helpers for the Qt app shell.
Exists to isolate manual-layer persistence and runtime-to-storage reconciliation.
Connects app-shell timeline edits to ProjectStorage layer and take records.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from echozero.application.session.models import Session
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import LayerId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.models import Layer
from echozero.domain.types import EventData
from echozero.persistence.entities import TimelineRegionRecord
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take as PersistedTake
from echozero.ui.qt.app_shell_layer_storage import (
    build_manual_layer_record,
    manual_layer_take_data,
    next_persisted_manual_layer_order,
    persisted_take_from_runtime_take,
    runtime_layer_record,
)


class StorageSyncShell(Protocol):
    _app: TimelineApplication
    _draft_layers: list[Layer]
    project_storage: ProjectStorage

    @property
    def session(self) -> Session: ...


def store_manual_layer(shell: StorageSyncShell, layer: Layer) -> None:
    song_version_id = shell.session.active_song_version_id
    if song_version_id is None:
        shell._draft_layers.append(layer)
        return
    persisted_song_version_id = str(song_version_id)
    with shell.project_storage.transaction():
        persist_manual_layer(shell, layer, song_version_id=persisted_song_version_id)
    shell.project_storage.dirty_tracker.mark_dirty(persisted_song_version_id)


def persist_manual_layer(
    shell: StorageSyncShell,
    layer: Layer,
    *,
    song_version_id: str,
    order: int | None = None,
) -> None:
    persisted_order = (
        next_persisted_manual_layer_order(shell.project_storage.layers.list_by_version(song_version_id))
        if order is None
        else int(order)
    )
    record = build_manual_layer_record(
        layer,
        song_version_id=song_version_id,
        persisted_order=persisted_order,
    )
    take = PersistedTake.create(
        data=manual_layer_take_data(layer),
        label="Take 1",
        origin="user",
        is_main=True,
    )
    shell.project_storage.layers.create(record)
    shell.project_storage.takes.create(record.id, take)


def materialize_draft_layers(shell: StorageSyncShell, *, song_version_id: str) -> None:
    if not shell._draft_layers:
        return
    starting_order = -(len(shell._draft_layers) + 1)
    with shell.project_storage.transaction():
        for offset, layer in enumerate(shell._draft_layers):
            persist_manual_layer(
                shell,
                layer,
                song_version_id=song_version_id,
                order=starting_order + offset,
            )
    shell.project_storage.dirty_tracker.mark_dirty(song_version_id)
    shell._draft_layers = []


def sync_runtime_take_records(shell: StorageSyncShell, layer: Layer) -> None:
    existing_takes = {
        take.id: take for take in shell.project_storage.takes.list_by_layer(str(layer.id))
    }
    runtime_takes = list(layer.takes)
    if not runtime_takes:
        _sync_empty_main_take(shell, layer, existing_takes)
        return

    runtime_take_ids = {str(take.id) for take in runtime_takes}
    for take_id in existing_takes:
        if take_id not in runtime_take_ids:
            shell.project_storage.takes.delete(take_id)

    for index, runtime_take in enumerate(runtime_takes):
        persisted = persisted_take_from_runtime_take(
            layer,
            runtime_take,
            existing=existing_takes.get(str(runtime_take.id)),
            is_main=index == 0,
        )
        if str(runtime_take.id) in existing_takes:
            shell.project_storage.takes.update(persisted)
        else:
            shell.project_storage.takes.create(str(layer.id), persisted)


def sync_storage_backed_timeline(shell: StorageSyncShell) -> None:
    song_version_id = shell.session.active_song_version_id
    if song_version_id is None:
        return
    persisted_song_version_id = str(song_version_id)
    existing_records = {
        record.id: record
        for record in shell.project_storage.layers.list_by_version(persisted_song_version_id)
    }
    runtime_layers = [
        layer
        for layer in shell._app.timeline.layers
        if layer.id != LayerId("source_audio")
        and (is_event_like_layer_kind(layer.kind) or str(layer.id) in existing_records)
    ]
    runtime_layer_ids = {str(layer.id) for layer in runtime_layers}

    with shell.project_storage.transaction():
        for record in existing_records.values():
            if record.layer_type == "manual" and record.id not in runtime_layer_ids:
                shell.project_storage.layers.delete(record.id)

        for layer in runtime_layers:
            existing = existing_records.get(str(layer.id))
            if existing is None:
                persist_manual_layer(
                    shell,
                    layer,
                    song_version_id=persisted_song_version_id,
                    order=max(0, int(layer.order_index) - 1),
                )
            else:
                shell.project_storage.layers.update(runtime_layer_record(layer, existing=existing))
            sync_runtime_take_records(shell, layer)
        _sync_runtime_region_records(shell, song_version_id=persisted_song_version_id)

    shell.project_storage.dirty_tracker.mark_dirty(persisted_song_version_id)


def _sync_runtime_region_records(
    shell: StorageSyncShell,
    *,
    song_version_id: str,
) -> None:
    existing_regions = {
        record.id: record
        for record in shell.project_storage.timeline_regions.list_by_version(song_version_id)
    }
    runtime_regions = sorted(
        shell._app.timeline.regions,
        key=lambda region: (
            float(region.start),
            float(region.end),
            int(region.order_index),
            str(region.id),
        ),
    )
    runtime_region_ids = {str(region.id) for region in runtime_regions}
    for region_id in existing_regions:
        if region_id not in runtime_region_ids:
            shell.project_storage.timeline_regions.delete(region_id)

    for index, region in enumerate(runtime_regions):
        existing = existing_regions.get(str(region.id))
        record = TimelineRegionRecord(
            id=str(region.id),
            song_version_id=song_version_id,
            label=region.label,
            start_seconds=float(region.start),
            end_seconds=float(region.end),
            color=region.color,
            order_index=index,
            kind=region.kind,
            created_at=(
                existing.created_at if existing is not None else datetime.now(timezone.utc)
            ),
        )
        if existing is None:
            shell.project_storage.timeline_regions.create(record)
        else:
            shell.project_storage.timeline_regions.update(record)


def _sync_empty_main_take(
    shell: StorageSyncShell,
    layer: Layer,
    existing_takes: dict[str, PersistedTake],
) -> None:
    existing_main = next((take for take in existing_takes.values() if take.is_main), None)
    if existing_main is None:
        if not is_event_like_layer_kind(layer.kind):
            return
        empty_main = PersistedTake.create(
            data=EventData(layers=()),
            label="Take 1",
            origin="user",
            source=None,
            is_main=True,
        )
        shell.project_storage.takes.create(str(layer.id), empty_main)
    else:
        shell.project_storage.takes.update(
            replace(
                existing_main,
                data=EventData(layers=())
                if is_event_like_layer_kind(layer.kind)
                else existing_main.data,
                is_main=True,
            )
        )
    for take_id, take in existing_takes.items():
        if not take.is_main:
            shell.project_storage.takes.delete(take_id)
