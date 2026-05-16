"""General contract-action helpers for the timeline widget.
Exists to isolate non-transfer inspector action routing from transfer workspace and dialog orchestration.
Connects inspector actions to app intents and runtime shell callbacks on the canonical timeline widget surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Protocol, cast

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QWidget

from echozero.application.song.title_extraction import resolve_import_song_titles
from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import (
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TimelinePresentation,
)
from echozero.application.settings.models import SongImportNameMode
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.event_batch_scope import event_batch_scope_from_params
from echozero.application.timeline.object_content import is_imported_song_layer
from echozero.application.timeline.intents import (
    ClearLayerLiveSyncPauseReason,
    CommitRejectedEventsReview,
    CommitVerifiedEventsReview,
    CreateEvent,
    DuplicateSelectedEvents,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    RenumberEventCueNumbers,
    Seek,
    SelectEveryOtherEvents,
    SetSelectedEvents,
    SetGain,
    SetLayerMute,
    SetLayerOutputBus,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetLayerSolo,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.application.timeline.object_actions import resolve_action_id
from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
)
from echozero.persistence.audio import detect_ltc_channel, scan_audio_metadata
from echozero.ui.qt.timeline.find_similar_dialog import EventComparisonDialog
from echozero.ui.qt.timeline.layer_routing_dialog import LayerRoutingSettingsDialog
from echozero.output_routing import canonical_layer_output_bus

_AUDIO_FILE_DIALOG_FILTER = "Audio Files (*.wav *.mp3 *.flac *.aiff *.aif *.ogg);;All Files (*)"
_IMPORT_SMPTE_AS_IS_LABEL = "Import As-Is (No LTC Extraction)"


@dataclass(slots=True)
class _MoveSelectionDestinationOption:
    label: str
    layer_id: LayerId | None = None
    take_id: TakeId | None = None
    create_layer_kind: LayerKind | None = None
    default_layer_title: str | None = None


class _TimelineRuntimeShell(Protocol):
    def presentation(self) -> TimelinePresentation: ...


class _ImproveModelRuntimeShell(_TimelineRuntimeShell, Protocol):
    def summarize_improve_model_selection(self, event_refs: list[object]) -> object: ...

    def train_improved_model_from_selection(
        self,
        request: ImproveModelTrainingRequest,
    ) -> object: ...


class _AddSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_song_from_path(
        self,
        title: str,
        audio_path: str,
        *,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation | None: ...


class _SelectSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def select_song(self, song_id: str) -> TimelinePresentation | None: ...


class _RenameSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def rename_song(self, song_id: str, title: str) -> TimelinePresentation | None: ...


class _SwitchSongVersionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def switch_song_version(self, song_version_id: str) -> TimelinePresentation | None: ...


class _AddSongVersionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_song_version(
        self,
        song_id: str,
        audio_path: str,
        *,
        label: str | None = None,
        transfer_layers: bool = False,
        transfer_layer_ids: list[str] | None = None,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation | None: ...


class _MoveSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def move_song(self, song_id: str, *, steps: int) -> TimelinePresentation | None: ...


class _ReorderSongsRuntimeShell(_TimelineRuntimeShell, Protocol):
    def reorder_songs(self, song_ids: list[str]) -> TimelinePresentation | None: ...


class _SongVersionTransferLookupRuntimeShell(_TimelineRuntimeShell, Protocol):
    def list_song_version_transfer_layers(
        self,
        song_id: str,
    ) -> list[tuple[str, str]]: ...


class _DeleteSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def delete_song(self, song_id: str) -> TimelinePresentation | None: ...


class _DeleteSongVersionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def delete_song_version(
        self,
        song_version_id: str,
    ) -> TimelinePresentation | None: ...


class _MA3TimecodeRuntimeShell(_TimelineRuntimeShell, Protocol):
    def list_ma3_timecode_pools(self) -> list[tuple[int, str | None]]: ...

    def set_song_version_ma3_timecode_pool(
        self,
        song_version_id: str,
        timecode_pool_no: int | None,
    ) -> TimelinePresentation | None: ...


class _SongVersionBeatGridRuntimeShell(_TimelineRuntimeShell, Protocol):
    def set_song_version_beat_anchor_seconds(
        self,
        song_version_id: str,
        beat_anchor_seconds: float,
    ) -> TimelinePresentation | None: ...


class _ProjectMA3PushOffsetRuntimeShell(_TimelineRuntimeShell, Protocol):
    def get_project_ma3_push_offset_seconds(self) -> float: ...

    def set_project_ma3_push_offset_seconds(
        self,
        offset_seconds: float,
    ) -> TimelinePresentation | None: ...


class _AddLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_layer(
        self,
        kind: LayerKind,
        title: str | None = None,
    ) -> TimelinePresentation | None: ...


class _DeleteLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def delete_layer(self, layer_id: str) -> TimelinePresentation | None: ...


class _ImportSmpteAudioLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def import_smpte_audio_to_layer(
        self,
        layer_id: str,
        audio_path: str,
        *,
        strip_ltc_timecode: bool = True,
        ltc_channel_override: str | None = None,
    ) -> TimelinePresentation | None: ...


class _AddImportSplitSmpteLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_smpte_layer_from_import_split(self) -> TimelinePresentation | None: ...


class _PreviewEventRuntimeShell(_TimelineRuntimeShell, Protocol):
    def preview_event_clip(
        self,
        *,
        layer_id: LayerId,
        take_id: TakeId | None,
        event_id: EventId,
    ) -> None: ...


def _coerce_layer_id(value: object) -> LayerId | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return LayerId(stripped)
    return None


def _coerce_take_id(value: object) -> TakeId | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return TakeId(stripped)
    return None


def _coerce_event_id(value: object) -> EventId | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return EventId(stripped)
    return None


def _coerce_step_count(value: object, *, default: int = 1) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return default
    return default


class _ContractActionHost(Protocol):
    _widget: QWidget
    _dispatch: Callable[[object], None]
    _get_presentation: Callable[[], TimelinePresentation]
    _set_presentation: Callable[[TimelinePresentation], None]
    _resolve_runtime_shell: Callable[[], _TimelineRuntimeShell | None]
    _input_dialog: type[QInputDialog]
    _file_dialog: type[QFileDialog]
    _message_box: type[QMessageBox]

    def handle_transfer_action(self, action_id: str, params: dict[str, object]) -> bool: ...


class TimelineWidgetContractActionMixin:
    _event_comparison_dialog_class = EventComparisonDialog
    _find_similar_dialog_class = EventComparisonDialog

    def _default_song_title_from_audio_path(self, audio_path: str) -> str:
        stem = Path(audio_path).stem.strip()
        return stem or "Imported Song"

    def _resolved_import_song_title(
        self,
        runtime: _TimelineRuntimeShell,
        audio_path: str,
        *,
        batch_audio_paths: tuple[str, ...] = (),
    ) -> str:
        paths = batch_audio_paths or (audio_path,)
        return resolve_import_song_titles(
            paths,
            name_mode=self._song_import_name_mode(runtime),
        ).get(audio_path, self._default_song_title_from_audio_path(audio_path))

    @staticmethod
    def _song_import_name_mode(runtime: _TimelineRuntimeShell) -> SongImportNameMode:
        service = getattr(runtime, "app_settings_service", None)
        if service is None or not callable(getattr(service, "preferences", None)):
            return SongImportNameMode.FILENAME
        try:
            preferences = service.preferences()
        except Exception:
            return SongImportNameMode.FILENAME
        configured = getattr(preferences, "song_import", None)
        value = getattr(configured, "name_mode", SongImportNameMode.FILENAME)
        try:
            return SongImportNameMode(str(getattr(value, "value", value)).strip())
        except ValueError:
            return SongImportNameMode.FILENAME

    def _resolve_audio_picker_start_directory(self) -> str:
        configured_value = getattr(self, "_last_audio_picker_directory", "")
        if isinstance(configured_value, str) and configured_value.strip():
            return configured_value
        return ""

    def _remember_audio_picker_directory(self, audio_path: str) -> None:
        selected_parent = Path(audio_path).expanduser().parent
        setattr(self, "_last_audio_picker_directory", str(selected_parent))

    def _prompt_for_audio_path(self, *, title: str) -> str | None:
        host = cast(_ContractActionHost, self)
        audio_path, _ = host._file_dialog.getOpenFileName(
            host._widget,
            title,
            self._resolve_audio_picker_start_directory(),
            _AUDIO_FILE_DIALOG_FILTER,
        )
        if not audio_path:
            return None
        self._remember_audio_picker_directory(audio_path)
        return audio_path

    @staticmethod
    def _require_native_import_pipeline_control(
        *,
        runtime_name: str,
        action_name: str,
        configured_action_ids: tuple[str, ...],
        supports_native_pipeline_control: bool,
    ) -> None:
        if supports_native_pipeline_control or not configured_action_ids:
            return
        raise RuntimeError(
            f"{runtime_name}.{action_name} must accept "
            "'run_import_pipeline' and 'import_pipeline_action_ids' when import pipeline "
            "actions are configured"
        )

    def _configured_import_pipeline_action_ids(
        self,
        runtime: _TimelineRuntimeShell,
    ) -> tuple[str, ...] | None:
        resolver = getattr(self, "_configured_import_pipeline_actions", None)
        if not callable(resolver):
            return None
        try:
            action_ids = resolver(runtime)
        except Exception:
            return ()
        if not action_ids:
            return ()
        return tuple(
            action_id.strip()
            for action_id in action_ids
            if isinstance(action_id, str) and action_id.strip()
        )

    @staticmethod
    def _method_supports_any_kwargs(
        method: Callable[..., object],
        *kwargs: str,
    ) -> bool:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return True
        parameters = signature.parameters
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return True
        return any(keyword in parameters for keyword in kwargs)

    @staticmethod
    def _invoke_with_supported_kwargs(
        method: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        parameters = signature.parameters
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return method(*args, **kwargs)
        supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return method(*args, **supported_kwargs)

    def trigger_contract_action(self, action: InspectorAction) -> None:
        """Execute one inspector contract action against the widget/runtime surface."""
        host = cast(_ContractActionHost, self)
        params = action.params
        action_id = resolve_action_id(action.action_id, warn_on_alias=True) or action.action_id
        if action_id == "seek_here":
            time_seconds = params.get("time_seconds")
            if isinstance(time_seconds, (int, float)):
                host._dispatch(Seek(float(time_seconds)))
            return
        if action_id in {
            "video.import",
            "video.replace",
            "video.remove",
            "video.reset_offset",
            "video.open_window",
        }:
            self._run_video_reference_action(action_id)
            return
        if action_id == "timeline.nudge_selection":
            raw_direction = params.get("direction", "left")
            direction = 1 if raw_direction in {1, "1", "right"} else -1
            host._dispatch(
                NudgeSelectedEvents(
                    direction=direction, steps=_coerce_step_count(params.get("steps", 1))
                )
            )
            return
        if action_id == "timeline.duplicate_selection":
            host._dispatch(
                DuplicateSelectedEvents(steps=_coerce_step_count(params.get("steps", 1)))
            )
            return
        if action_id == "selection.move_to_destination":
            self._run_move_selected_events_destination_action()
            return
        if action_id == "selection.select_every_other":
            scope = event_batch_scope_from_params(params)
            if scope is not None:
                host._dispatch(SelectEveryOtherEvents(scope=scope))
            return
        if action_id == "selection.improve_model_from_selection":
            runtime = cast(_ImproveModelRuntimeShell | None, host._resolve_runtime_shell())
            if runtime is None:
                host._message_box.warning(
                    host._widget,
                    "Improve Model From Selection",
                    "Open a project before training a candidate model from selected reviewed events.",
                )
                return
            selected_event_refs = list(host._get_presentation().resolved_selected_event_refs())
            if not selected_event_refs:
                host._message_box.warning(
                    host._widget,
                    "Improve Model From Selection",
                    "Select one or more reviewed events first.",
                )
                return
            try:
                summary = runtime.summarize_improve_model_selection(selected_event_refs)
            except Exception as exc:
                host._message_box.warning(
                    host._widget,
                    "Improve Model From Selection",
                    str(exc),
                )
                return
            from echozero.ui.qt.improve_model_dialog import ImproveModelDialog

            dialog = ImproveModelDialog(summary, parent=host._widget)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            try:
                result = runtime.train_improved_model_from_selection(dialog.result_payload().request)
            except Exception as exc:
                host._message_box.warning(
                    host._widget,
                    "Improve Model From Selection",
                    str(exc),
                )
                return
            comparison_note = (
                "Compared against the selected base model."
                if getattr(result, "compared_to_base_model", False)
                else "No base-model comparison was recorded for this V1 run."
            )
            host._message_box.information(
                host._widget,
                "Improve Model From Selection",
                (
                    f"Candidate run complete for '{result.target_label}'.\n\n"
                    f"Run: {result.run_id}\n"
                    f"Artifact: {result.artifact_id}\n"
                    f"Anchors: {result.anchor_sample_count}\n"
                    f"Related: {result.related_sample_count}\n\n"
                    f"{comparison_note}"
                ),
            )
            return
        if action_id in {"selection.compare_events", "selection.find_similar_sounding"}:
            layer_id = _coerce_layer_id(params.get("layer_id"))
            take_id = _coerce_take_id(params.get("take_id"))
            event_id = _coerce_event_id(params.get("event_id"))
            if layer_id is None or take_id is None or event_id is None:
                return
            payload = self._run_event_comparison_dialog(
                layer_id=layer_id,
                take_id=take_id,
                event_id=event_id,
                default_scope_mode="take",
            )
            if payload is None:
                return
            event_refs = list(payload["event_refs"])
            outcome_action = str(payload.get("outcome_action", "select"))
            host._dispatch(
                SetSelectedEvents(
                    event_ids=list(payload["event_ids"]),
                    event_refs=event_refs,
                    anchor_layer_id=payload["anchor_layer_id"],
                    anchor_take_id=payload["anchor_take_id"],
                    selected_layer_ids=list(payload["selected_layer_ids"]),
                )
            )
            if outcome_action == "promote":
                host._dispatch(
                    CommitVerifiedEventsReview(
                        event_refs=event_refs,
                        review_note="Find Similar matched events",
                    )
                )
            elif outcome_action == "demote":
                host._dispatch(
                    CommitRejectedEventsReview(
                        event_refs=event_refs,
                        review_note="Find Similar matched events",
                    )
                )
            elif outcome_action == "create_layer":
                self._create_layer_from_matched_events(
                    event_refs=event_refs,
                    title=str(payload.get("new_layer_title") or "Similar Events"),
                )
            return
        if action_id == "selection.renumber_cues_from_one":
            scope = event_batch_scope_from_params(params)
            if scope is not None:
                host._dispatch(RenumberEventCueNumbers(scope=scope, start_at=1, step=1))
            return
        if action_id == "layer.set_expanded":
            layer_id = _coerce_layer_id(params.get("layer_id"))
            expanded = params.get("expanded")
            if layer_id is not None and isinstance(expanded, bool):
                self._set_layer_expanded_state(layer_id=layer_id, expanded=expanded)
            return
        if action_id == "timeline.expand_all_layers":
            self._set_all_layers_expanded_state(expanded=True)
            return
        if action_id == "timeline.collapse_all_layers":
            self._set_all_layers_expanded_state(expanded=False)
            return
        if action_id == "song.add":
            self._run_add_song_from_path_action(params)
            return
        if action_id == "song.select":
            self._run_select_song_action(params)
            return
        if action_id == "song.rename":
            self._run_rename_song_action(params)
            return
        if action_id == "song.version.switch":
            self._run_switch_song_version_action(params)
            return
        if action_id == "song.version.add":
            self._run_add_song_version_action(params)
            return
        if action_id == "song.delete":
            self._run_delete_song_action(params)
            return
        if action_id == "song.version.delete":
            self._run_delete_song_version_action(params)
            return
        if action_id == "song.version.set_ma3_timecode_pool":
            self._run_set_song_version_ma3_timecode_pool_action(params)
            return
        if action_id in {
            "song.version.set_first_beat_here",
            "song.version.set_first_beat_to_playhead",
        }:
            self._run_set_song_version_beat_anchor_action(action_id, params)
            return
        if action_id == "project.settings.set_ma3_push_offset":
            self._run_set_project_ma3_push_offset_action(params)
            return
        if action_id == "add_event_layer":
            self._run_add_layer_action(LayerKind.EVENT)
            return
        if action_id == "add_section_layer":
            self._run_add_layer_action(LayerKind.SECTION)
            return
        if action_id == "add_smpte_layer":
            self._run_add_layer_action(LayerKind.AUDIO, title="SMPTE Layer")
            return
        if action_id == "add_smpte_layer_from_import_split":
            self._run_add_smpte_layer_from_import_split_action()
            return
        if action_id == "delete_layer":
            self._run_delete_layer_action(params)
            return
        if action_id == "import_smpte_audio_to_layer":
            self._run_import_smpte_audio_to_layer_action(params)
            return
        if action_id == "preview_event_clip":
            self._handle_preview_event_clip(params)
            return
        if action_id == "layer.routing_settings":
            self._open_layer_routing_settings(params)
            return
        if action_id in {"gain_down", "gain_unity", "gain_up", "set_gain_custom"}:
            gain_db = params.get("gain_db")
            if isinstance(gain_db, (int, float)):
                for layer_id in self._coerce_target_layer_ids(params):
                    host._dispatch(SetGain(layer_id=layer_id, gain_db=float(gain_db)))
            return
        if action_id in {"set_layer_mute_on", "set_layer_mute_off"}:
            target_layer_ids = self._coerce_target_layer_ids(params)
            if not target_layer_ids:
                return
            muted = bool(params.get("muted", action_id == "set_layer_mute_on"))
            for layer_id in target_layer_ids:
                host._dispatch(SetLayerMute(layer_id=layer_id, muted=muted))
            return
        if action_id in {"set_layer_solo_on", "set_layer_solo_off"}:
            target_layer_ids = self._coerce_target_layer_ids(params)
            if not target_layer_ids:
                return
            soloed = bool(params.get("soloed", action_id == "set_layer_solo_on"))
            for layer_id in target_layer_ids:
                host._dispatch(SetLayerSolo(layer_id=layer_id, soloed=soloed))
            return
        if action_id == "set_layer_output_bus_auto" or action_id.startswith(
            "set_layer_output_bus_"
        ):
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is None:
                return
            raw_output_bus = params.get("output_bus")
            output_bus = canonical_layer_output_bus(raw_output_bus, reject_invalid=True)
            host._dispatch(SetLayerOutputBus(layer_id=layer_id, output_bus=output_bus))
            return
        if action_id in {
            "live_sync_set_off",
            "live_sync_set_observe",
            "live_sync_set_armed_write",
        }:
            self._handle_live_sync_action(action_id, params)
            return
        if action_id == "live_sync_set_pause_reason":
            layer_id = _coerce_layer_id(params.get("layer_id"))
            pause_reason = params.get("pause_reason")
            if layer_id is not None and isinstance(pause_reason, str) and pause_reason.strip():
                host._dispatch(
                    SetLayerLiveSyncPauseReason(
                        layer_id=layer_id,
                        pause_reason=pause_reason,
                    )
                )
            return
        if action_id == "live_sync_clear_pause_reason":
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is not None:
                host._dispatch(ClearLayerLiveSyncPauseReason(layer_id=layer_id))
            return
        if host.handle_transfer_action(action_id, params):
            return
        if action_id:
            layer_id = _coerce_layer_id(params.get("layer_id"))
            take_id = _coerce_take_id(params.get("take_id"))
            if layer_id is not None and take_id is not None:
                host._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    def _set_layer_expanded_state(self, *, layer_id: LayerId, expanded: bool) -> None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        layer = next((item for item in presentation.layers if item.layer_id == layer_id), None)
        if layer is None or not layer.takes:
            return
        if bool(layer.is_expanded) == bool(expanded):
            return
        host._dispatch(ToggleLayerExpanded(layer_id=layer_id))

    def _set_all_layers_expanded_state(self, *, expanded: bool) -> None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        for layer in presentation.layers:
            if not layer.takes:
                continue
            if bool(layer.is_expanded) == bool(expanded):
                continue
            host._dispatch(ToggleLayerExpanded(layer_id=layer.layer_id))

    def _open_layer_routing_settings(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return
        presentation = host._get_presentation()
        layer = next((item for item in presentation.layers if item.layer_id == layer_id), None)
        if layer is None:
            return
        if layer.kind is not LayerKind.AUDIO:
            host._message_box.information(
                host._widget,
                "Layer Routing Settings",
                "Audio output routing is available for audio layers only.",
            )
            return

        output_channels = max(1, min(16, int(presentation.playback_output_channels or 2)))
        dialog = LayerRoutingSettingsDialog(
            layer_title=str(layer.title),
            playback_output_channels=output_channels,
            current_output_bus=layer.output_bus,
            parent=host._widget,
        )
        if not bool(dialog.exec()):
            return
        host._dispatch(
            SetLayerOutputBus(
                layer_id=layer_id,
                output_bus=dialog.selected_output_bus(),
            )
        )

    def _handle_live_sync_action(self, action_id: str, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return
        if action_id == "live_sync_set_armed_write":
            reply = host._message_box.question(
                host._widget,
                "Arm Live Sync Write",
                "Arm live sync write for this layer? MA3 changes may be written immediately.",
                host._message_box.StandardButton.Yes | host._message_box.StandardButton.No,
                host._message_box.StandardButton.No,
            )
            if reply != host._message_box.StandardButton.Yes:
                return
            state = LiveSyncState.ARMED_WRITE
        elif action_id == "live_sync_set_observe":
            state = LiveSyncState.OBSERVE
        else:
            state = LiveSyncState.OFF
        host._dispatch(SetLayerLiveSyncState(layer_id=layer_id, live_sync_state=state))

    def _run_move_selected_events_destination_action(self) -> None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        selected_refs = list(presentation.resolved_selected_event_refs())
        if not selected_refs:
            host._message_box.warning(
                host._widget,
                "Move Selected Events",
                "Select one or more events before choosing a destination.",
            )
            return

        layers_by_id = {layer.layer_id: layer for layer in presentation.layers}
        selected_layers = [
            layers_by_id[event_ref.layer_id]
            for event_ref in selected_refs
            if event_ref.layer_id in layers_by_id
        ]
        if not selected_layers:
            host._message_box.warning(
                host._widget,
                "Move Selected Events",
                "The selected events could not be resolved to timeline layers.",
            )
            return

        layer_kinds = {layer.kind for layer in selected_layers}
        if len(layer_kinds) != 1:
            host._message_box.warning(
                host._widget,
                "Move Selected Events",
                "Move To works with selections from one layer type at a time.",
            )
            return

        source_kind = next(iter(layer_kinds))
        if not is_event_like_layer_kind(source_kind):
            host._message_box.warning(
                host._widget,
                "Move Selected Events",
                "Move To is available for event and section layers only.",
            )
            return

        destination_options = self._move_selection_destination_options(
            presentation,
            source_kind=source_kind,
        )
        if not destination_options:
            host._message_box.warning(
                host._widget,
                "Move Selected Events",
                "No compatible destinations are available.",
            )
            return

        selected_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Move Selected Events",
            "Destination",
            [option.label for option in destination_options],
            0,
            False,
        )
        if not accepted:
            return

        selected_option = next(
            (option for option in destination_options if option.label == selected_label),
            None,
        )
        if selected_option is None:
            return

        if selected_option.create_layer_kind is not None:
            default_title = (
                selected_option.default_layer_title
                or (
                    "New Section Layer"
                    if selected_option.create_layer_kind is LayerKind.SECTION
                    else "New Event Layer"
                )
            )
            entered_title, accepted = host._input_dialog.getText(
                host._widget,
                "Create Destination Layer",
                "Layer name",
                text=default_title,
            )
            if not accepted:
                return
            layer_title = entered_title.strip() or default_title
            host._dispatch(
                MoveSelectedEvents(
                    delta_seconds=0.0,
                    create_layer_title=layer_title,
                )
            )
            return

        host._dispatch(
            MoveSelectedEvents(
                delta_seconds=0.0,
                target_layer_id=selected_option.layer_id,
                target_take_id=selected_option.take_id,
            )
        )

    def _move_selection_destination_options(
        self,
        presentation: TimelinePresentation,
        *,
        source_kind: LayerKind,
    ) -> list[_MoveSelectionDestinationOption]:
        options: list[_MoveSelectionDestinationOption] = []
        for layer in presentation.layers:
            if layer.kind is not source_kind:
                continue
            if not layer.visible or layer.locked:
                continue
            if layer.main_take_id is not None:
                options.append(
                    _MoveSelectionDestinationOption(
                        label=f"{layer.title} -> Main",
                        layer_id=layer.layer_id,
                        take_id=layer.main_take_id,
                    )
                )
            for take in layer.takes:
                options.append(
                    _MoveSelectionDestinationOption(
                        label=f"{layer.title} -> {take.name}",
                        layer_id=layer.layer_id,
                        take_id=take.take_id,
                    )
                )

        default_title = (
            "New Section Layer" if source_kind is LayerKind.SECTION else "New Event Layer"
        )
        options.append(
            _MoveSelectionDestinationOption(
                label=(
                    "Create New Section Layer..."
                    if source_kind is LayerKind.SECTION
                    else "Create New Event Layer..."
                ),
                create_layer_kind=source_kind,
                default_layer_title=default_title,
            )
        )
        return options

    def _run_video_reference_action(self, action_id: str) -> None:
        host = cast(_ContractActionHost, self)
        runtime = host._resolve_runtime_shell()
        if runtime is None:
            return
        try:
            if action_id in {"video.import", "video.replace"}:
                selected, _ = host._file_dialog.getOpenFileName(
                    host._widget,
                    "Select Video File",
                    "",
                    "Video Files (*.mov *.mp4 *.m4v *.avi *.mkv);;All Files (*)",
                )
                if not selected:
                    return
                importer = getattr(runtime, "import_or_replace_song_video", None)
                if callable(importer):
                    updated = importer(selected)
                    if updated is not None:
                        host._set_presentation(updated)
                return
            if action_id == "video.remove":
                remover = getattr(runtime, "remove_active_song_video", None)
                if callable(remover):
                    updated = remover()
                    if updated is not None:
                        host._set_presentation(updated)
                return
            if action_id == "video.reset_offset":
                setter = getattr(runtime, "set_active_song_video_start_seconds", None)
                if callable(setter):
                    updated = setter(0.0)
                    if updated is not None:
                        host._set_presentation(updated)
                return
            if action_id == "video.open_window":
                opener = getattr(runtime, "open_video_window", None)
                if callable(opener):
                    opener()
                return
        except Exception as exc:
            host._message_box.warning(host._widget, "Video Reference", str(exc))

    def _run_add_song_from_path_action(self, params: dict[str, object] | None = None) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddSongRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "add_song_from_path", None)):
            host._message_box.warning(
                host._widget,
                "Add Song",
                "This runtime does not support adding songs from a path.",
            )
            return
        payload = params or {}
        raw_audio_path = payload.get("audio_path")
        audio_path = str(raw_audio_path).strip() if isinstance(raw_audio_path, str) else ""
        if audio_path:
            self._remember_audio_picker_directory(audio_path)
        else:
            audio_path = self._prompt_for_audio_path(title="Select Audio File") or ""
        if not audio_path:
            return
        requested_title = payload.get("title")
        if isinstance(requested_title, str) and requested_title.strip():
            title = requested_title.strip()
        else:
            title = self._resolved_import_song_title(runtime, audio_path)
        configured_action_ids = self._configured_import_pipeline_action_ids(runtime)
        canonical_import = getattr(self, "_invoke_add_song_from_path", None)
        if callable(canonical_import) and configured_action_ids is not None:
            handled = canonical_import(
                runtime,
                title.strip(),
                audio_path,
                run_import_pipeline=bool(configured_action_ids),
                pipeline_action_ids=configured_action_ids or None,
            )
            if not bool(handled):
                return
            return
        supports_native_pipeline_control = self._method_supports_any_kwargs(
            runtime.add_song_from_path,
            "run_import_pipeline",
            "import_pipeline_action_ids",
        )
        call_kwargs: dict[str, object] = {}
        if configured_action_ids is not None:
            call_kwargs["run_import_pipeline"] = bool(configured_action_ids)
            call_kwargs["import_pipeline_action_ids"] = configured_action_ids or None
        self._require_native_import_pipeline_control(
            runtime_name=type(runtime).__name__,
            action_name="add_song_from_path",
            configured_action_ids=configured_action_ids or (),
            supports_native_pipeline_control=supports_native_pipeline_control,
        )
        try:
            updated = self._invoke_with_supported_kwargs(
                runtime.add_song_from_path,
                title.strip(),
                audio_path,
                **call_kwargs,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Add Song", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def add_song_from_dialog(self) -> None:
        self._run_add_song_from_path_action()

    def _run_select_song_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_SelectSongRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "select_song", None)):
            host._message_box.warning(
                host._widget,
                "Select Song",
                "This runtime does not support switching songs.",
            )
            return
        song_id = params.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            presentation = host._get_presentation()
            if not presentation.available_songs:
                host._message_box.warning(
                    host._widget,
                    "Select Song",
                    "No songs are available in this project.",
                )
                return
            labels = [self._song_option_label(song) for song in presentation.available_songs]
            default_index = next(
                (
                    index
                    for index, song in enumerate(presentation.available_songs)
                    if song.is_active
                ),
                0,
            )
            selected_label, accepted = host._input_dialog.getItem(
                host._widget,
                "Select Song",
                "Song",
                labels,
                default_index,
                False,
            )
            if not accepted:
                return
            selected_song = next(
                (
                    song
                    for song, label in zip(presentation.available_songs, labels)
                    if label == selected_label
                ),
                None,
            )
            if selected_song is None:
                return
            song_id = selected_song.song_id
        try:
            updated = runtime.select_song(song_id.strip())
        except Exception as exc:
            host._message_box.warning(host._widget, "Select Song", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def select_song(self, song_id: str) -> None:
        self._run_select_song_action({"song_id": song_id})

    def _run_rename_song_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_RenameSongRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "rename_song", None)):
            host._message_box.warning(
                host._widget,
                "Rename Song",
                "This runtime does not support renaming songs.",
            )
            return
        song_id = params.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            song_id = host._get_presentation().active_song_id
        if not isinstance(song_id, str) or not song_id.strip():
            host._message_box.warning(
                host._widget,
                "Rename Song",
                "Select a song before renaming it.",
            )
            return
        current_title = self._resolve_song_title(song_id)
        requested_title = params.get("title")
        if isinstance(requested_title, str) and requested_title.strip():
            next_title = requested_title.strip()
        else:
            next_value, accepted = host._input_dialog.getText(
                host._widget,
                "Rename Song",
                "Song name",
                text=current_title,
            )
            if not accepted:
                return
            next_title = str(next_value or "").strip()
        if not next_title or next_title == current_title:
            return
        try:
            updated = runtime.rename_song(song_id.strip(), next_title)
        except Exception as exc:
            host._message_box.warning(host._widget, "Rename Song", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def rename_song(self, song_id: str) -> None:
        self._run_rename_song_action({"song_id": song_id})

    def _run_switch_song_version_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_SwitchSongVersionRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "switch_song_version", None)):
            host._message_box.warning(
                host._widget,
                "Switch Version",
                "This runtime does not support switching song versions.",
            )
            return
        song_version_id = params.get("song_version_id")
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            presentation = host._get_presentation()
            if not presentation.available_song_versions:
                host._message_box.warning(
                    host._widget,
                    "Switch Version",
                    "No song versions are available for the current song.",
                )
                return
            labels = [
                self._song_version_option_label(version)
                for version in presentation.available_song_versions
            ]
            default_index = next(
                (
                    index
                    for index, version in enumerate(presentation.available_song_versions)
                    if version.is_active
                ),
                0,
            )
            selected_label, accepted = host._input_dialog.getItem(
                host._widget,
                "Switch Version",
                "Version",
                labels,
                default_index,
                False,
            )
            if not accepted:
                return
            selected_version = next(
                (
                    version
                    for version, label in zip(presentation.available_song_versions, labels)
                    if label == selected_label
                ),
                None,
            )
            if selected_version is None:
                return
            song_version_id = selected_version.song_version_id
        try:
            updated = runtime.switch_song_version(song_version_id.strip())
        except Exception as exc:
            host._message_box.warning(host._widget, "Switch Version", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def switch_song_version(self, song_version_id: str) -> None:
        self._run_switch_song_version_action({"song_version_id": song_version_id})

    def _run_add_song_version_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddSongVersionRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "add_song_version", None)):
            host._message_box.warning(
                host._widget,
                "Add Version",
                "This runtime does not support adding song versions.",
            )
            return
        song_id = params.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            song_id = self._resolve_song_id_for_new_version()
            if song_id is None:
                return
        label = params.get("label")
        resolved_label = label.strip() if isinstance(label, str) and label.strip() else None
        audio_path = params.get("audio_path")
        prompt_for_transfer = False
        if not isinstance(audio_path, str) or not audio_path.strip():
            audio_path = self._prompt_for_audio_path(title="Select Audio File")
            if not audio_path:
                return
            text_value, accepted = host._input_dialog.getText(
                host._widget,
                "Add Version",
                "Version label (optional)",
            )
            if not accepted:
                return
            resolved_label = text_value.strip() or None
            prompt_for_transfer = True
        transfer_options = self._resolve_add_song_version_transfer_options(
            runtime=runtime,
            song_id=song_id,
            params=params,
            prompt_user=prompt_for_transfer,
        )
        if transfer_options is None:
            return
        transfer_layers, transfer_layer_ids = transfer_options
        configured_action_ids = self._configured_import_pipeline_action_ids(runtime)
        canonical_import = getattr(self, "_invoke_add_song_version", None)
        if callable(canonical_import) and configured_action_ids is not None:
            handled = canonical_import(
                runtime,
                song_id,
                audio_path,
                label=resolved_label,
                transfer_layers=transfer_layers,
                transfer_layer_ids=transfer_layer_ids,
                run_import_pipeline=bool(configured_action_ids),
                pipeline_action_ids=configured_action_ids or None,
            )
            if not bool(handled):
                return
            return
        call_kwargs: dict[str, object] = {"label": resolved_label}
        if transfer_layers:
            call_kwargs["transfer_layers"] = True
            if transfer_layer_ids is not None:
                call_kwargs["transfer_layer_ids"] = transfer_layer_ids
        supports_native_pipeline_control = self._method_supports_any_kwargs(
            runtime.add_song_version,
            "run_import_pipeline",
            "import_pipeline_action_ids",
        )
        if configured_action_ids is not None:
            call_kwargs["run_import_pipeline"] = bool(configured_action_ids)
            call_kwargs["import_pipeline_action_ids"] = configured_action_ids or None
        self._require_native_import_pipeline_control(
            runtime_name=type(runtime).__name__,
            action_name="add_song_version",
            configured_action_ids=configured_action_ids or (),
            supports_native_pipeline_control=supports_native_pipeline_control,
        )
        try:
            updated = self._invoke_with_supported_kwargs(
                runtime.add_song_version,
                song_id,
                audio_path,
                **call_kwargs,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Add Version", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def add_song_version(self, song_id: str) -> None:
        self._run_add_song_version_action({"song_id": song_id})

    def _run_delete_song_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_DeleteSongRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "delete_song", None)):
            host._message_box.warning(
                host._widget,
                "Delete Song",
                "This runtime does not support deleting songs.",
            )
            return

        song_id = params.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            song_id = host._get_presentation().active_song_id
        if not isinstance(song_id, str) or not song_id.strip():
            host._message_box.warning(
                host._widget,
                "Delete Song",
                "Select a song before deleting it.",
            )
            return

        title = self._resolve_song_title(song_id)
        reply = host._message_box.question(
            host._widget,
            "Delete Song",
            (
                f'Delete "{title}" and all of its versions, layers, and settings?\n\n'
                "This cannot be undone."
            ),
            host._message_box.StandardButton.Yes | host._message_box.StandardButton.No,
            host._message_box.StandardButton.No,
        )
        if reply != host._message_box.StandardButton.Yes:
            return
        try:
            updated = runtime.delete_song(song_id.strip())
        except Exception as exc:
            host._message_box.warning(host._widget, "Delete Song", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def delete_song(self, song_id: str) -> None:
        self._run_delete_song_action({"song_id": song_id})

    def _run_delete_song_version_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_DeleteSongVersionRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "delete_song_version", None)):
            host._message_box.warning(
                host._widget,
                "Delete Version",
                "This runtime does not support deleting song versions.",
            )
            return

        song_version_id = params.get("song_version_id")
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            song_version_id = host._get_presentation().active_song_version_id
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            host._message_box.warning(
                host._widget,
                "Delete Version",
                "Select a song version before deleting it.",
            )
            return

        version_label = self._resolve_song_version_label(song_version_id)
        reply = host._message_box.question(
            host._widget,
            "Delete Version",
            (
                f'Delete version "{version_label}"?\n\n'
                "If this is the last version, the song will also be deleted."
            ),
            host._message_box.StandardButton.Yes | host._message_box.StandardButton.No,
            host._message_box.StandardButton.No,
        )
        if reply != host._message_box.StandardButton.Yes:
            return
        try:
            updated = runtime.delete_song_version(song_version_id.strip())
        except Exception as exc:
            host._message_box.warning(host._widget, "Delete Version", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def delete_song_version(self, song_version_id: str) -> None:
        self._run_delete_song_version_action({"song_version_id": song_version_id})

    def _run_add_layer_action(self, kind: LayerKind, *, title: str | None = None) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddLayerRuntimeShell | None, host._resolve_runtime_shell())
        layer_title = (title or "").strip()
        label = f"Add {layer_title}" if layer_title else f"Add {kind.value.title()} Layer"
        if runtime is None or not callable(getattr(runtime, "add_layer", None)):
            host._message_box.warning(
                host._widget,
                label,
                f"This runtime does not support adding {kind.value} layers.",
            )
            return
        try:
            if layer_title:
                try:
                    updated = runtime.add_layer(kind, title=layer_title)
                except TypeError:
                    updated = runtime.add_layer(kind)
            else:
                updated = runtime.add_layer(kind)
        except Exception as exc:
            host._message_box.warning(host._widget, label, str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _run_delete_layer_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_DeleteLayerRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "delete_layer", None)):
            host._message_box.warning(
                host._widget,
                "Delete Layer",
                "This runtime does not support deleting layers.",
            )
            return
        layer_id = params.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            layer_id = self._resolve_selected_layer_id()
        if not isinstance(layer_id, str) or not layer_id.strip():
            host._message_box.warning(
                host._widget,
                "Delete Layer",
                "Select a layer before deleting it.",
            )
            return
        label = self._resolve_layer_title(layer_id)
        reply = host._message_box.question(
            host._widget,
            "Delete Layer",
            (f'Delete layer "{label}"?\n\n' "This action cannot be undone."),
            host._message_box.StandardButton.Yes | host._message_box.StandardButton.No,
            host._message_box.StandardButton.No,
        )
        if reply != host._message_box.StandardButton.Yes:
            return
        try:
            updated = runtime.delete_layer(layer_id.strip())
        except Exception as exc:
            host._message_box.warning(host._widget, "Delete Layer", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _run_import_smpte_audio_to_layer_action(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(
            _ImportSmpteAudioLayerRuntimeShell | None,
            host._resolve_runtime_shell(),
        )
        if runtime is None or not callable(getattr(runtime, "import_smpte_audio_to_layer", None)):
            host._message_box.warning(
                host._widget,
                "Import SMPTE Audio",
                "This runtime does not support importing SMPTE audio.",
            )
            return

        layer_id = params.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id.strip():
            layer_id = self._resolve_selected_layer_id()
        if not isinstance(layer_id, str) or not layer_id.strip():
            host._message_box.warning(
                host._widget,
                "Import SMPTE Audio",
                "Select a SMPTE layer before importing audio.",
            )
            return

        raw_audio_path = params.get("audio_path")
        if isinstance(raw_audio_path, str) and raw_audio_path.strip():
            audio_path = raw_audio_path.strip()
        else:
            audio_path = self._prompt_for_audio_path(title="Import SMPTE Audio")
        if not audio_path:
            return

        accepted, ltc_channel_override, strip_ltc_timecode = (
            self._resolve_smpte_import_ltc_strategy(audio_path)
        )
        if not accepted:
            return

        try:
            updated = self._invoke_with_supported_kwargs(
                runtime.import_smpte_audio_to_layer,
                layer_id.strip(),
                audio_path,
                strip_ltc_timecode=strip_ltc_timecode,
                ltc_channel_override=ltc_channel_override,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Import SMPTE Audio", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _resolve_smpte_import_ltc_strategy(
        self,
        audio_path: str,
    ) -> tuple[bool, str | None, bool]:
        source_path = Path(audio_path).expanduser()
        if not source_path.exists():
            return True, None, True

        try:
            metadata = scan_audio_metadata(source_path)
        except Exception:
            return True, None, True
        if int(metadata.channel_count) < 2:
            return True, None, False

        strict_channel = detect_ltc_channel(source_path, mode="strict")
        if strict_channel in {"left", "right"}:
            return True, strict_channel, True

        aggressive_channel = detect_ltc_channel(source_path, mode="aggressive")
        return self._prompt_smpte_ltc_channel_choice(aggressive_channel)

    def _prompt_smpte_ltc_channel_choice(
        self,
        aggressive_channel: str | None,
    ) -> tuple[bool, str | None, bool]:
        host = cast(_ContractActionHost, self)
        option_labels: list[str] = []
        option_values: dict[str, tuple[str | None, bool]] = {}
        prompt = "LTC detection was not confident. Choose how to import this stereo file."

        if aggressive_channel in {"left", "right"}:
            recommended_label = f"Use {aggressive_channel.title()} Channel as LTC (Recommended)"
            alternate_channel = "right" if aggressive_channel == "left" else "left"
            alternate_label = f"Use {alternate_channel.title()} Channel as LTC"
            option_labels = [
                recommended_label,
                alternate_label,
                _IMPORT_SMPTE_AS_IS_LABEL,
            ]
            option_values = {
                recommended_label: (aggressive_channel, True),
                alternate_label: (alternate_channel, True),
                _IMPORT_SMPTE_AS_IS_LABEL: (None, False),
            }
            prompt = (
                "LTC detection is low confidence for this stereo file. "
                "Choose which channel should be treated as LTC."
            )
        else:
            option_labels = [
                "Use Left Channel as LTC",
                "Use Right Channel as LTC",
                _IMPORT_SMPTE_AS_IS_LABEL,
            ]
            option_values = {
                "Use Left Channel as LTC": ("left", True),
                "Use Right Channel as LTC": ("right", True),
                _IMPORT_SMPTE_AS_IS_LABEL: (None, False),
            }

        selected_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Import SMPTE Audio",
            prompt,
            option_labels,
            0,
            False,
        )
        if not accepted:
            return False, None, False
        selected_value = option_values.get(str(selected_label))
        if selected_value is None:
            return False, None, False
        override, strip_ltc = selected_value
        return True, override, strip_ltc

    def _run_add_smpte_layer_from_import_split_action(self) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(
            _AddImportSplitSmpteLayerRuntimeShell | None,
            host._resolve_runtime_shell(),
        )
        if runtime is None or not callable(
            getattr(runtime, "add_smpte_layer_from_import_split", None)
        ):
            host._message_box.warning(
                host._widget,
                "Add SMPTE Layer from Import Split",
                "This runtime does not support adding SMPTE layers from import splits.",
            )
            return

        try:
            updated = runtime.add_smpte_layer_from_import_split()
        except Exception as exc:
            host._message_box.warning(host._widget, "Add SMPTE Layer from Import Split", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _resolve_layer_title(self, layer_id: str) -> str:
        host = cast(_ContractActionHost, self)
        for layer in host._get_presentation().layers:
            if str(layer.layer_id) == layer_id:
                return layer.title
        return "Selected Layer"

    def _resolve_selected_layer_id(self) -> str | None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        return (
            str(presentation.selected_layer_id)
            if presentation.selected_layer_id is not None
            else None
        )

    def _handle_preview_event_clip(self, params: dict[str, object]) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_PreviewEventRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "preview_event_clip", None)):
            host._message_box.warning(
                host._widget,
                "Event Clip Preview",
                "This runtime does not support event clip preview.",
            )
            return
        layer_id = _coerce_layer_id(params.get("layer_id"))
        take_id = _coerce_take_id(params.get("take_id"))
        event_id = _coerce_event_id(params.get("event_id"))
        if layer_id is None or event_id is None:
            host._message_box.warning(
                host._widget,
                "Event Clip Preview",
                "The selected event is missing clip preview metadata.",
            )
            return
        try:
            runtime.preview_event_clip(
                layer_id=layer_id,
                take_id=take_id,
                event_id=event_id,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Event Clip Preview", str(exc))

    def _run_set_song_version_ma3_timecode_pool_action(
        self,
        params: dict[str, object],
    ) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_MA3TimecodeRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not all(
            callable(getattr(runtime, method_name, None))
            for method_name in ("list_ma3_timecode_pools", "set_song_version_ma3_timecode_pool")
        ):
            host._message_box.warning(
                host._widget,
                "Set MA3 TC Pool",
                "This runtime does not support MA3 timecode pool configuration.",
            )
            return

        presentation = host._get_presentation()
        song_version_id = params.get("song_version_id")
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            song_version_id = presentation.active_song_version_id
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            host._message_box.warning(
                host._widget,
                "Set MA3 TC Pool",
                "Select a song version before configuring the MA3 timecode pool.",
            )
            return

        if "timecode_pool_no" in params:
            try:
                selected_pool_no = self._parse_ma3_timecode_pool_input(
                    params.get("timecode_pool_no")
                )
            except ValueError:
                host._message_box.warning(
                    host._widget,
                    "Set MA3 TC Pool",
                    "Enter a numeric MA3 timecode pool (for example: 113 or TC113).",
                )
                return
        else:
            timecodes = runtime.list_ma3_timecode_pools()
            options: list[tuple[str, int | None]] = [("None (Unconfigured)", None)]
            options.extend(
                (
                    f"TC{timecode_no} · {name}" if name else f"TC{timecode_no}",
                    timecode_no,
                )
                for timecode_no, name in timecodes
            )
            option_lookup = {label: value for label, value in options}
            current_pool_no = presentation.active_song_version_ma3_timecode_pool_no
            default_index = next(
                (
                    index
                    for index, (_label, value) in enumerate(options)
                    if value == current_pool_no
                ),
                0,
            )
            chosen_label, accepted = host._input_dialog.getItem(
                host._widget,
                "Set MA3 TC Pool",
                (
                    "Song version MA3 timecode pool\n"
                    "Select a discovered pool or type one manually (for example: 113)."
                ),
                [label for label, _value in options],
                default_index,
                True,
            )
            if not accepted:
                return
            if chosen_label in option_lookup:
                selected_pool_no = option_lookup[chosen_label]
            else:
                try:
                    selected_pool_no = self._parse_ma3_timecode_pool_input(chosen_label)
                except ValueError:
                    host._message_box.warning(
                        host._widget,
                        "Set MA3 TC Pool",
                        "Enter a numeric MA3 timecode pool (for example: 113 or TC113).",
                    )
                    return
        try:
            updated = runtime.set_song_version_ma3_timecode_pool(
                song_version_id.strip(),
                selected_pool_no,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Set MA3 TC Pool", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _run_set_song_version_beat_anchor_action(
        self,
        action_id: str,
        params: dict[str, object],
    ) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_SongVersionBeatGridRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(
            getattr(runtime, "set_song_version_beat_anchor_seconds", None)
        ):
            host._message_box.warning(
                host._widget,
                "Set First Beat",
                "This runtime does not support first-beat alignment.",
            )
            return

        presentation = host._get_presentation()
        song_version_id = params.get("song_version_id")
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            song_version_id = presentation.active_song_version_id
        if not isinstance(song_version_id, str) or not song_version_id.strip():
            host._message_box.warning(
                host._widget,
                "Set First Beat",
                "Select a song version before aligning the first beat.",
            )
            return

        if action_id == "song.version.set_first_beat_to_playhead":
            beat_anchor_seconds = float(presentation.playhead)
        else:
            raw_anchor = params.get("beat_anchor_seconds")
            if not isinstance(raw_anchor, (int, float)):
                host._message_box.warning(
                    host._widget,
                    "Set First Beat",
                    "Choose a timeline position before aligning the first beat.",
                )
                return
            beat_anchor_seconds = float(raw_anchor)
        if beat_anchor_seconds < 0.0:
            beat_anchor_seconds = 0.0

        try:
            updated = runtime.set_song_version_beat_anchor_seconds(
                song_version_id.strip(),
                beat_anchor_seconds,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Set First Beat", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _run_set_project_ma3_push_offset_action(
        self,
        params: dict[str, object],
    ) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(
            _ProjectMA3PushOffsetRuntimeShell | None,
            host._resolve_runtime_shell(),
        )
        if runtime is None or not all(
            callable(getattr(runtime, method_name, None))
            for method_name in (
                "get_project_ma3_push_offset_seconds",
                "set_project_ma3_push_offset_seconds",
            )
        ):
            host._message_box.warning(
                host._widget,
                "Set Global MA3 Push Offset",
                "This runtime does not support project MA3 push offset settings.",
            )
            return

        current_offset = float(runtime.get_project_ma3_push_offset_seconds())
        if "offset_seconds" in params:
            try:
                resolved_offset = self._parse_project_ma3_push_offset_input(
                    params.get("offset_seconds")
                )
            except ValueError:
                host._message_box.warning(
                    host._widget,
                    "Set Global MA3 Push Offset",
                    "Enter a numeric offset in seconds (for example: -1, -0.35, 0.5).",
                )
                return
        else:
            entered_value, accepted = host._input_dialog.getText(
                host._widget,
                "Set Global MA3 Push Offset",
                (
                    "MA3 push offset in seconds.\n"
                    "Negative values move events earlier on the clock.\n"
                    "Positive values move events later."
                ),
                text=f"{current_offset:.3f}",
            )
            if not accepted:
                return
            try:
                resolved_offset = self._parse_project_ma3_push_offset_input(entered_value)
            except ValueError:
                host._message_box.warning(
                    host._widget,
                    "Set Global MA3 Push Offset",
                    "Enter a numeric offset in seconds (for example: -1, -0.35, 0.5).",
                )
                return

        try:
            updated = runtime.set_project_ma3_push_offset_seconds(resolved_offset)
        except Exception as exc:
            host._message_box.warning(host._widget, "Set Global MA3 Push Offset", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

    def _resolve_add_song_version_transfer_options(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        song_id: str,
        params: dict[str, object],
        prompt_user: bool,
    ) -> tuple[bool, list[str] | None] | None:
        host = cast(_ContractActionHost, self)
        explicit_transfer_layers = params.get("transfer_layers")
        explicit_layer_ids = self._coerce_transfer_layer_ids(params.get("transfer_layer_ids"))
        if isinstance(explicit_transfer_layers, bool) or explicit_layer_ids is not None:
            if explicit_transfer_layers is False:
                return False, None
            if explicit_layer_ids is not None:
                return True, explicit_layer_ids
            return bool(explicit_transfer_layers), None
        if not prompt_user:
            return False, None

        available_layers = self._resolve_transfer_layers_for_song_version(
            runtime=runtime,
            song_id=song_id,
        )
        if not available_layers:
            return False, None
        transfer_mode_options = [
            "Do not transfer layers",
            "Transfer all layers",
            "Choose layers to transfer",
        ]
        selected_mode, accepted = host._input_dialog.getItem(
            host._widget,
            "Add Version",
            ("Layer transfer options\n" "(the source song layer is excluded automatically)"),
            transfer_mode_options,
            1,
            False,
        )
        if not accepted:
            return None
        if selected_mode == transfer_mode_options[0]:
            return False, None
        if selected_mode == transfer_mode_options[1]:
            return True, None

        selected_layer_ids = self._prompt_selected_transfer_layer_ids(available_layers)
        if selected_layer_ids is None:
            return None
        if not selected_layer_ids:
            return False, None
        return True, selected_layer_ids

    def _resolve_transfer_layers_for_song_version(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        song_id: str,
    ) -> list[tuple[str, str]]:
        host = cast(_ContractActionHost, self)
        runtime_lookup = cast(_SongVersionTransferLookupRuntimeShell | None, runtime)
        if runtime_lookup is not None and callable(
            getattr(runtime_lookup, "list_song_version_transfer_layers", None)
        ):
            try:
                raw_layers = runtime_lookup.list_song_version_transfer_layers(song_id)
            except Exception:
                raw_layers = []
            resolved_layers = [
                (layer_id.strip(), layer_label.strip() or f"Layer {index + 1}")
                for index, (layer_id, layer_label) in enumerate(raw_layers)
                if isinstance(layer_id, str) and layer_id.strip()
            ]
            if resolved_layers:
                return resolved_layers

        presentation = host._get_presentation()
        if presentation.active_song_id != song_id:
            return []
        return [
            (str(layer.layer_id), layer.title.strip() or f"Layer {index + 1}")
            for index, layer in enumerate(presentation.layers)
            if not _is_imported_song_layer(layer)
        ]

    def _prompt_selected_transfer_layer_ids(
        self,
        available_layers: list[tuple[str, str]],
    ) -> list[str] | None:
        host = cast(_ContractActionHost, self)
        layer_lines = "\n".join(
            f"{index + 1}. {label}" for index, (_layer_id, label) in enumerate(available_layers)
        )
        while True:
            selected_text, accepted = host._input_dialog.getText(
                host._widget,
                "Select Layers",
                (
                    "Choose layers to transfer by number (comma-separated).\n"
                    'Use "all" for every layer.\n\n'
                    f"{layer_lines}"
                ),
            )
            if not accepted:
                return None
            selected_ids = self._parse_selected_transfer_layer_ids(
                selected_text,
                available_layers=available_layers,
            )
            if selected_ids is not None:
                return selected_ids
            host._message_box.warning(
                host._widget,
                "Select Layers",
                "Use comma-separated numbers like 1,2,4 or ranges like 1-3.",
            )

    @staticmethod
    def _parse_selected_transfer_layer_ids(
        selection_text: str,
        *,
        available_layers: list[tuple[str, str]],
    ) -> list[str] | None:
        normalized = selection_text.strip()
        if not normalized:
            return []
        if normalized.lower() == "all":
            return [layer_id for layer_id, _label in available_layers]

        selected_numbers: list[int] = []
        max_index = len(available_layers)
        for raw_token in normalized.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if "-" in token:
                start_token, end_token = token.split("-", 1)
                try:
                    start = int(start_token.strip())
                    end = int(end_token.strip())
                except ValueError:
                    return None
                if start < 1 or end < 1 or start > max_index or end > max_index:
                    return None
                step = 1 if start <= end else -1
                selected_numbers.extend(range(start, end + step, step))
                continue
            try:
                value = int(token)
            except ValueError:
                return None
            if value < 1 or value > max_index:
                return None
            selected_numbers.append(value)

        ordered_unique_numbers: list[int] = []
        seen_numbers: set[int] = set()
        for value in selected_numbers:
            if value in seen_numbers:
                continue
            ordered_unique_numbers.append(value)
            seen_numbers.add(value)
        return [available_layers[value - 1][0] for value in ordered_unique_numbers]

    @staticmethod
    def _coerce_transfer_layer_ids(value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        if isinstance(value, (list, tuple)):
            normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            return normalized
        return None

    def _coerce_target_layer_ids(self, params: dict[str, object]) -> list[LayerId]:
        selected_layer_ids = self._coerce_transfer_layer_ids(params.get("selected_layer_ids"))
        if selected_layer_ids:
            return [LayerId(layer_id) for layer_id in selected_layer_ids]
        layer_id = _coerce_layer_id(params.get("layer_id"))
        return [] if layer_id is None else [layer_id]

    def _resolve_song_id_for_new_version(self) -> str | None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        if presentation.active_song_id:
            return presentation.active_song_id
        if not presentation.available_songs:
            host._message_box.warning(
                host._widget,
                "Add Version",
                "Add a song before creating a song version.",
            )
            return None
        labels = [self._song_option_label(song) for song in presentation.available_songs]
        selected_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Add Version",
            "Song",
            labels,
            0,
            False,
        )
        if not accepted:
            return None
        selected_song = next(
            (
                song
                for song, label in zip(presentation.available_songs, labels)
                if label == selected_label
            ),
            None,
        )
        return None if selected_song is None else selected_song.song_id

    def _run_event_comparison_dialog(
        self,
        *,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        default_scope_mode: str,
    ) -> dict[str, object] | None:
        host = cast(_ContractActionHost, self)
        legacy_dialog_override = self.__dict__.get("_find_similar_dialog_class")
        if legacy_dialog_override is not None:
            dialog_class = legacy_dialog_override
        else:
            dialog_class = getattr(
                self,
                "_event_comparison_dialog_class",
                getattr(self, "_find_similar_dialog_class", EventComparisonDialog),
            )
        dialog = dialog_class(
            presentation=host._get_presentation(),
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
            default_scope_mode=default_scope_mode,
            parent=host._widget,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return None
        return dialog.selected_payload()

    def _run_find_similar_sounding_dialog(
        self,
        *,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        default_scope_mode: str,
    ) -> dict[str, object] | None:
        return self._run_event_comparison_dialog(
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
            default_scope_mode=default_scope_mode,
        )


    def _create_layer_from_matched_events(
        self,
        *,
        event_refs: list[object],
        title: str,
    ) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddLayerRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not event_refs:
            return
        presentation = runtime.add_layer(LayerKind.EVENT, title.strip() or "Similar Events")
        if presentation is None or presentation.selected_layer_id is None:
            return
        target_layer_id = presentation.selected_layer_id
        source_presentation = host._get_presentation()
        created_refs = []
        for index, event_ref in enumerate(event_refs, start=1):
            event = _find_presentation_event(source_presentation, event_ref)
            if event is None:
                continue
            host._dispatch(
                CreateEvent(
                    layer_id=target_layer_id,
                    time_range=TimeRange(float(event.start), float(event.end)),
                    label=event.label or "Matched Event",
                    cue_number=index,
                    source_event_id=str(getattr(event_ref, "event_id", "") or "") or None,
                    color=getattr(event, "color", None),
                )
            )
            created_refs.append(event_ref)

    def _resolve_song_title(self, song_id: str) -> str:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        if presentation.active_song_id == song_id and presentation.active_song_title:
            return presentation.active_song_title
        for song in presentation.available_songs:
            if song.song_id == song_id:
                return song.title
        return "Selected Song"

    def _resolve_song_version_label(self, song_version_id: str) -> str:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        if (
            presentation.active_song_version_id == song_version_id
            and presentation.active_song_version_label
        ):
            return presentation.active_song_version_label
        for version in presentation.available_song_versions:
            if version.song_version_id == song_version_id:
                return version.label
        for song in presentation.available_songs:
            for version in song.versions:
                if version.song_version_id == song_version_id:
                    return version.label
        return "Selected Version"

    @staticmethod
    def _song_option_label(song: SongOptionPresentation) -> str:
        version_suffix = f" · {song.active_version_label}" if song.active_version_label else ""
        return f"{song.title}{version_suffix}"

    @staticmethod
    def _song_version_option_label(version: SongVersionOptionPresentation) -> str:
        if version.ma3_timecode_pool_no is None:
            return version.label
        return f"{version.label} · TC{version.ma3_timecode_pool_no}"

    @staticmethod
    def _parse_ma3_timecode_pool_input(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("MA3 timecode pool must be numeric.")
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("MA3 timecode pool must be a whole number.")
            parsed_float = int(value)
            return parsed_float if parsed_float > 0 else None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized.upper().startswith("TC"):
                normalized = normalized[2:].strip()
            if normalized.startswith("#"):
                normalized = normalized[1:].strip()
            if not normalized:
                return None
            try:
                parsed_text = int(normalized)
            except ValueError as exc:
                raise ValueError("MA3 timecode pool must be numeric.") from exc
            return parsed_text if parsed_text > 0 else None
        raise ValueError("MA3 timecode pool must be numeric.")

    @staticmethod
    def _parse_project_ma3_push_offset_input(value: object) -> float:
        if value is None:
            raise ValueError("MA3 push offset is required.")
        if isinstance(value, bool):
            raise ValueError("MA3 push offset must be numeric.")
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if not text:
            raise ValueError("MA3 push offset is required.")
        if text.endswith("s"):
            text = text[:-1].strip()
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError("MA3 push offset must be numeric.") from exc


def _is_imported_song_layer(layer: object) -> bool:
    return is_imported_song_layer(layer)


def _find_presentation_event(presentation: TimelinePresentation, event_ref: object):
    layer_id = getattr(event_ref, "layer_id", None)
    take_id = getattr(event_ref, "take_id", None)
    event_id = getattr(event_ref, "event_id", None)
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            continue
        for event in layer.events:
            if event.event_id == event_id:
                return event
        for take in layer.takes:
            if take_id is not None and take.take_id != take_id:
                continue
            for event in take.events:
                if event.event_id == event_id:
                    return event
    return None
