"""Editing facade mixin for the Qt app shell runtime.
Exists to isolate manual-layer creation and intent dispatch on the public shell surface.
Connects AppShellRuntime to undo/history helpers and the timeline edit contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import LayerId
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    CommitBoundaryCorrectedEventReview,
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitRelabeledEventReview,
    CommitVerifiedEventsReview,
    CommitVerifiedEventReview,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    CreateRegion,
    DeleteRegion,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ReplaceSectionCues,
    ReorderLayer,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateEventLabel,
    UpdateRegion,
)
from echozero.application.timeline.ma3_push_intents import SetLayerMA3Route
from echozero.application.timeline.models import Layer, derive_section_cues_from_layers
from echozero.domain.types import AudioData
from echozero.persistence.audio import (
    AudioImportOptions,
    cleanup_prepared_audio,
    compute_audio_hash,
    import_audio,
    prepare_audio_for_import,
    scan_audio_metadata,
)
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take as PersistedTake
from echozero.ui.qt.app_shell_history import (
    history_label_for_intent,
    is_history_barrier_intent,
    is_storage_backed_undoable_intent,
    is_undoable_intent,
)
from echozero.ui.qt.app_shell_layer_storage import build_manual_layer
from echozero.ui.qt.app_shell_timeline_review import (
    commit_boundary_corrected_review,
    commit_missed_events_review,
    commit_missed_event_review,
    commit_rejected_events_review,
    commit_rejected_review,
    commit_relabel_review,
    commit_verified_events_review,
    commit_verified_review,
)
from echozero.ui.qt.app_shell_timeline_state import clear_selected_events

_DIRTYING_INTENT_TYPES = (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    CommitMissedEventReview,
    CommitMissedEventsReview,
    CommitVerifiedEventReview,
    CommitVerifiedEventsReview,
    CommitRejectedEventReview,
    CommitRejectedEventsReview,
    CommitRelabeledEventReview,
    CommitBoundaryCorrectedEventReview,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    CreateRegion,
    DeleteRegion,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ReplaceSectionCues,
    ReorderLayer,
    SetGain,
    SetLayerMA3Route,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateEventLabel,
    UpdateRegion,
)


class AppShellEditingShell(Protocol):
    _app: TimelineApplication
    _is_dirty: bool

    @property
    def session(self) -> Session: ...
    project_storage: ProjectStorage

    def presentation(self) -> TimelinePresentation: ...

    def _clear_history(self) -> None: ...

    def _run_undoable_operation(
        self,
        *,
        label: str,
        storage_backed: bool,
        mark_dirty: bool,
        operation: Callable[[], TimelinePresentation],
    ) -> TimelinePresentation: ...

    def _store_manual_layer(self, layer: Layer) -> None: ...

    def _sync_storage_backed_timeline(self) -> None: ...

    def _sync_storage_backed_layers(self, layer_ids: list[LayerId]) -> None: ...

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None: ...

    def _refresh_from_storage(
        self,
        *,
        active_song_id: object | None = None,
        active_song_version_id: object | None = None,
    ) -> None: ...


class AppShellEditingMixin:
    def add_smpte_layer_from_import_split(
        self: AppShellEditingShell,
    ) -> TimelinePresentation:
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is None:
            raise ValueError("Select a song version before creating a SMPTE layer from import split.")
        ltc_artifact_path = self._resolve_import_split_ltc_artifact_path(str(active_song_version_id))
        if ltc_artifact_path is None:
            raise ValueError(
                "No retained split LTC artifact was found for the active song version."
            )

        added = self.add_layer(LayerKind.AUDIO, "SMPTE Layer")
        layer_id = added.selected_layer_id
        if layer_id is None:
            raise ValueError("Failed to create SMPTE layer.")
        return self.import_smpte_audio_to_layer(str(layer_id), str(ltc_artifact_path))

    def add_layer(
        self: AppShellEditingShell,
        kind: LayerKind,
        title: str | None = None,
    ) -> TimelinePresentation:
        def _perform_add_layer() -> TimelinePresentation:
            layer_kind = kind
            if not isinstance(layer_kind, LayerKind):
                try:
                    layer_kind = LayerKind(str(layer_kind))
                except ValueError as exc:
                    raise ValueError(f"Unsupported layer kind '{kind}'.") from exc

            layer_title = (title or "").strip()
            if not layer_title:
                layer_title = f"{layer_kind.value.title()} Layer"

            timeline = self._app.timeline
            new_layer = build_manual_layer(
                timeline=timeline,
                layer_kind=layer_kind,
                layer_title=layer_title,
            )
            timeline.layers.append(new_layer)
            timeline.section_cues = derive_section_cues_from_layers(timeline.layers)
            self._store_manual_layer(new_layer)
            timeline.selection.selected_layer_id = new_layer.id
            timeline.selection.selected_layer_ids = [new_layer.id]
            timeline.selection.selected_take_id = None
            clear_selected_events(timeline)
            self._sync_runtime_audio_from_presentation(self.presentation())
            self._is_dirty = True
            return self.presentation()

        return self._run_undoable_operation(
            label="Add Layer",
            storage_backed=self.session.active_song_version_id is not None,
            mark_dirty=True,
            operation=_perform_add_layer,
        )

    def import_smpte_audio_to_layer(
        self: AppShellEditingShell,
        layer_id: str,
        audio_path: str | Path,
        *,
        strip_ltc_timecode: bool = True,
        ltc_channel_override: str | None = None,
    ) -> TimelinePresentation:
        target_layer_id = LayerId(str(layer_id).strip())
        source_path = Path(audio_path).expanduser()
        if not source_path.exists():
            raise ValueError(f"Audio file not found: {source_path}")
        resolved_source_path = source_path.resolve()
        normalized_override: Literal["left", "right"] | None = None
        if isinstance(ltc_channel_override, str):
            candidate_override = ltc_channel_override.strip().lower()
            if candidate_override in {"left", "right"}:
                normalized_override = candidate_override
            elif candidate_override:
                raise ValueError(
                    f"Unsupported LTC channel override '{ltc_channel_override}'."
                )

        def _perform_import_smpte_audio() -> TimelinePresentation:
            active_song_id = self.session.active_song_id
            active_song_version_id = self.session.active_song_version_id
            if active_song_id is None or active_song_version_id is None:
                raise ValueError("Import SMPTE audio requires an active song version.")

            target_layer = next(
                (layer for layer in self._app.timeline.layers if layer.id == target_layer_id),
                None,
            )
            if target_layer is None:
                raise ValueError(f"Layer not found: {target_layer_id}")
            if target_layer.kind is not LayerKind.AUDIO:
                raise ValueError("SMPTE import is only available for audio layers.")

            layer_record = self.project_storage.layers.get(str(target_layer_id))
            if layer_record is None:
                raise ValueError("SMPTE layer is not persisted in the active song version.")
            if layer_record.song_version_id != str(active_song_version_id):
                raise ValueError("SMPTE layer does not belong to the active song version.")

            prepared_source = prepare_audio_for_import(
                resolved_source_path,
                self.project_storage.working_dir,
                options=AudioImportOptions(
                    strip_ltc_timecode=bool(strip_ltc_timecode),
                    ltc_detection_mode="aggressive",
                    ltc_channel_override=normalized_override,
                ),
            )
            try:
                import_source = (
                    prepared_source.ltc_artifact_path
                    if prepared_source.ltc_artifact_path is not None
                    else prepared_source.source_path
                )
                metadata = scan_audio_metadata(import_source)
                imported_relative_path, _audio_hash = import_audio(
                    import_source,
                    self.project_storage.working_dir,
                )
            finally:
                cleanup_prepared_audio(prepared_source)

            imported_audio = AudioData(
                sample_rate=int(metadata.sample_rate),
                duration=float(metadata.duration_seconds),
                file_path=imported_relative_path,
                channel_count=max(1, int(metadata.channel_count)),
            )

            with self.project_storage.transaction():
                takes = self.project_storage.takes.list_by_layer(str(target_layer_id))
                main_take = next((take for take in takes if take.is_main), takes[0] if takes else None)
                if main_take is None:
                    self.project_storage.takes.create(
                        str(target_layer_id),
                        PersistedTake.create(
                            data=imported_audio,
                            label="Take 1",
                            origin="user",
                            source=None,
                            is_main=True,
                        ),
                    )
                else:
                    self.project_storage.takes.update(
                        replace(
                            main_take,
                            data=imported_audio,
                            origin="user",
                            source=None,
                            is_main=True,
                        )
                    )
                self.project_storage.dirty_tracker.mark_dirty(str(active_song_version_id))

            self._refresh_from_storage(
                active_song_id=active_song_id,
                active_song_version_id=active_song_version_id,
            )
            self._is_dirty = True
            return self.presentation()

        return self._run_undoable_operation(
            label="Import SMPTE Audio",
            storage_backed=True,
            mark_dirty=True,
            operation=_perform_import_smpte_audio,
        )

    def _resolve_import_split_ltc_artifact_path(
        self: AppShellEditingShell,
        song_version_id: str,
    ) -> Path | None:
        version_record = self.project_storage.song_versions.get(song_version_id)
        if version_record is None:
            return None

        split_dir = self.project_storage.working_dir / "audio" / "split_channels"
        if not split_dir.exists():
            return None

        for program_path in sorted(split_dir.glob("*_program_*.wav")):
            if not program_path.is_file():
                continue
            try:
                candidate_hash = compute_audio_hash(program_path)
            except OSError:
                continue
            if candidate_hash != version_record.audio_hash:
                continue
            base_prefix, _sep, _rest = program_path.name.partition("_program_")
            for ltc_path in sorted(split_dir.glob(f"{base_prefix}_ltc_*.wav")):
                if ltc_path.is_file():
                    return ltc_path
        return None

    def delete_layer(
        self: AppShellEditingShell,
        layer_id: str,
    ) -> TimelinePresentation:
        target_layer_id = LayerId(layer_id)

        def _perform_delete_layer() -> TimelinePresentation:
            if target_layer_id == LayerId("source_audio"):
                raise ValueError("Cannot delete source audio layer.")

            timeline = self._app.timeline
            target_layer = next(
                (layer for layer in timeline.layers if layer.id == target_layer_id),
                None,
            )
            if target_layer is None:
                raise ValueError(f"Layer not found: {layer_id}")

            timeline.layers = [layer for layer in timeline.layers if layer.id != target_layer_id]
            timeline.section_cues = derive_section_cues_from_layers(timeline.layers)
            previous_selected_layer_id = timeline.selection.selected_layer_id
            self._draft_layers = [
                layer for layer in self._draft_layers if layer.id != target_layer_id
            ]

            selected_layer_ids = [
                layer_id for layer_id in dict.fromkeys(timeline.selection.selected_layer_ids)
                if layer_id != target_layer_id
            ]
            selected_layer_id = timeline.selection.selected_layer_id
            if selected_layer_id == target_layer_id:
                selected_layer_id = selected_layer_ids[0] if selected_layer_ids else (
                    timeline.layers[0].id if timeline.layers else None
                )
            timeline.selection.selected_layer_id = selected_layer_id
            if selected_layer_id is not None and selected_layer_id not in selected_layer_ids:
                selected_layer_ids = [selected_layer_id]
            timeline.selection.selected_layer_ids = selected_layer_ids

            if previous_selected_layer_id == target_layer_id:
                timeline.selection.selected_take_id = None

            clear_selected_events(timeline)

            active_version_id = self.session.active_song_version_id
            if active_version_id is not None:
                with self.project_storage.transaction():
                    for take in target_layer.takes:
                        self.project_storage.takes.delete(str(take.id))
                    self.project_storage.layers.delete(str(target_layer_id))
                    self.project_storage.dirty_tracker.mark_dirty(str(active_version_id))

            self._sync_runtime_audio_from_presentation(self.presentation())
            return self.presentation()

        return self._run_undoable_operation(
            label="Delete Layer",
            storage_backed=self.session.active_song_version_id is not None,
            mark_dirty=True,
            operation=_perform_delete_layer,
        )

    def dispatch(
        self: AppShellEditingShell,
        intent: TimelineIntent,
    ) -> TimelinePresentation:
        if isinstance(intent, CommitMissedEventReview):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or "Add Missed Event",
                storage_backed=True,
                mark_dirty=True,
                operation=lambda: commit_missed_event_review(self, intent),
            )
        if isinstance(intent, CommitMissedEventsReview):
            presentation = commit_missed_events_review(self, intent)
            self._is_dirty = True
            return presentation
        if isinstance(intent, CommitVerifiedEventReview):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or "Verify Event",
                storage_backed=True,
                mark_dirty=True,
                operation=lambda: commit_verified_review(self, intent),
            )
        if isinstance(intent, CommitVerifiedEventsReview):
            presentation = commit_verified_events_review(self, intent)
            self._is_dirty = True
            return presentation
        if isinstance(intent, CommitRejectedEventReview):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or "Reject Event",
                storage_backed=True,
                mark_dirty=True,
                operation=lambda: commit_rejected_review(self, intent),
            )
        if isinstance(intent, CommitRejectedEventsReview):
            presentation = commit_rejected_events_review(self, intent)
            self._is_dirty = True
            return presentation
        if isinstance(intent, CommitRelabeledEventReview):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or "Relabel Event",
                storage_backed=True,
                mark_dirty=True,
                operation=lambda: commit_relabel_review(self, intent),
            )
        if isinstance(intent, CommitBoundaryCorrectedEventReview):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or "Correct Boundary",
                storage_backed=True,
                mark_dirty=True,
                operation=lambda: commit_boundary_corrected_review(self, intent),
            )
        if is_undoable_intent(intent):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or intent.__class__.__name__,
                storage_backed=is_storage_backed_undoable_intent(intent),
                mark_dirty=isinstance(intent, _DIRTYING_INTENT_TYPES),
                operation=lambda: self._app.dispatch(intent),
            )

        presentation = self._app.dispatch(intent)
        if isinstance(intent, ToggleLayerExpanded):
            self._sync_storage_backed_timeline()
        if is_history_barrier_intent(intent):
            self._clear_history()
        if isinstance(intent, _DIRTYING_INTENT_TYPES):
            self._is_dirty = True
        return presentation
