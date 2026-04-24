"""Preset and live-sync helpers for the timeline orchestrator.
Exists to isolate transfer preset persistence and live-sync guardrail resets from transfer-plan execution.
Connects session preset state and live-sync safety behavior to the canonical timeline orchestrator.
"""

from __future__ import annotations

import re
from typing import Protocol, cast

from echozero.application.session.models import Session, TransferPresetState
from echozero.application.shared.ids import LayerId
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.models import Timeline

_RECONNECT_REARM_REQUIRED_REASON = "Live sync reconnected; explicit re-arm required"


class _SyncPresetHost(Protocol):
    @staticmethod
    def _selected_layer_scope(timeline: Timeline) -> list[LayerId]: ...

    def _rebuild_push_transfer_plan(self, timeline: Timeline, session: Session) -> None: ...

    def _rebuild_pull_transfer_plan(self, timeline: Timeline, session: Session) -> None: ...


class TimelineOrchestratorSyncPresetMixin:
    def _save_transfer_preset(self, timeline: Timeline, session: Session, name: str) -> None:
        push_mapping = self._capture_push_preset_mapping(timeline, session)
        pull_mapping = self._capture_pull_preset_mapping(session)
        preset_id = self._next_transfer_preset_id(session, name)
        session.transfer_presets.append(
            TransferPresetState(
                preset_id=preset_id,
                name=name,
                push_target_mapping_by_layer_id=push_mapping,
                pull_target_mapping_by_source_track=pull_mapping,
            )
        )

    def _apply_transfer_preset(self, timeline: Timeline, session: Session, preset_id: str) -> None:
        host = cast(_SyncPresetHost, self)
        preset = self._require_transfer_preset(
            session, preset_id, action_name="ApplyTransferPreset"
        )

        if session.manual_push_flow.push_mode_active:
            for layer in timeline.layers:
                target_track_coord = preset.push_target_mapping_by_layer_id.get(layer.id)
                if target_track_coord:
                    layer.sync.ma3_track_coord = target_track_coord
            selected_layer_scope = host._selected_layer_scope(timeline)
            if selected_layer_scope:
                session.manual_push_flow.target_track_coord = (
                    preset.push_target_mapping_by_layer_id.get(
                        timeline.selection.selected_layer_id or selected_layer_scope[0]
                    )
                )
            host._rebuild_push_transfer_plan(timeline, session)

        if session.manual_pull_flow.workspace_active:
            selected_coords = {
                coord for coord in session.manual_pull_flow.selected_source_track_coords
            }
            available_target_ids = {
                target.layer_id for target in session.manual_pull_flow.available_target_layers
            }
            next_mapping: dict[str, LayerId] = {}
            for (
                source_track_coord,
                target_layer_id,
            ) in preset.pull_target_mapping_by_source_track.items():
                if (
                    source_track_coord in selected_coords
                    and target_layer_id in available_target_ids
                ):
                    next_mapping[source_track_coord] = target_layer_id
            session.manual_pull_flow.target_layer_id_by_source_track = next_mapping
            active_coord = (
                session.manual_pull_flow.active_source_track_coord
                or session.manual_pull_flow.source_track_coord
            )
            session.manual_pull_flow.target_layer_id = (
                next_mapping.get(active_coord) if active_coord is not None else None
            )
            host._rebuild_pull_transfer_plan(timeline, session)

    @staticmethod
    def _delete_transfer_preset(session: Session, preset_id: str) -> None:
        preset = TimelineOrchestratorSyncPresetMixin._require_transfer_preset(
            session,
            preset_id,
            action_name="DeleteTransferPreset",
        )
        session.transfer_presets = [
            candidate
            for candidate in session.transfer_presets
            if candidate.preset_id != preset.preset_id
        ]

    @staticmethod
    def _capture_push_preset_mapping(timeline: Timeline, session: Session) -> dict[LayerId, str]:
        plan = session.batch_transfer_plan
        if plan is not None and plan.operation_type in {"push", "mixed"}:
            return {
                row.source_layer_id: row.target_track_coord
                for row in plan.rows
                if row.direction == "push"
                and row.source_layer_id is not None
                and row.target_track_coord
            }
        return {
            layer.id: layer.sync.ma3_track_coord
            for layer in timeline.layers
            if layer.sync.ma3_track_coord
        }

    @staticmethod
    def _capture_pull_preset_mapping(session: Session) -> dict[str, LayerId]:
        plan = session.batch_transfer_plan
        if plan is not None and plan.operation_type in {"pull", "mixed"}:
            return {
                row.source_track_coord: row.target_layer_id
                for row in plan.rows
                if row.direction == "pull"
                and row.source_track_coord
                and row.target_layer_id is not None
            }
        return dict(session.manual_pull_flow.target_layer_id_by_source_track)

    @staticmethod
    def _slugify_transfer_preset_name(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        return slug or "preset"

    @classmethod
    def _next_transfer_preset_id(cls, session: Session, name: str) -> str:
        base_slug = cls._slugify_transfer_preset_name(name)
        existing_ids = {preset.preset_id for preset in session.transfer_presets}
        if base_slug not in existing_ids:
            return base_slug
        counter = 2
        while True:
            candidate = f"{base_slug}-{counter}"
            if candidate not in existing_ids:
                return candidate
            counter += 1

    @staticmethod
    def _require_transfer_preset(
        session: Session, preset_id: str, *, action_name: str
    ) -> TransferPresetState:
        for preset in session.transfer_presets:
            if preset.preset_id == preset_id:
                return preset
        raise ValueError(f"{action_name} preset_id not found: {preset_id}")

    @staticmethod
    def _reset_live_sync_guardrails(timeline: Timeline) -> None:
        for layer in timeline.layers:
            layer.sync.live_sync_state = LiveSyncState.OFF
            layer.sync.live_sync_pause_reason = None
            layer.sync.live_sync_divergent = False

    @staticmethod
    def _pause_armed_write_layers_on_reconnect(timeline: Timeline) -> None:
        for layer in timeline.layers:
            if layer.sync.live_sync_state is LiveSyncState.ARMED_WRITE:
                layer.sync.live_sync_state = LiveSyncState.PAUSED
                layer.sync.live_sync_pause_reason = _RECONNECT_REARM_REQUIRED_REASON
