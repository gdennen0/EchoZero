"""General contract-action helpers for the timeline widget.
Exists to isolate non-transfer inspector action routing from transfer workspace and dialog orchestration.
Connects inspector actions to app intents and runtime shell callbacks on the canonical timeline widget surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import (
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.event_batch_scope import event_batch_scope_from_params
from echozero.application.timeline.intents import (
    ClearLayerLiveSyncPauseReason,
    DuplicateSelectedEvents,
    NudgeSelectedEvents,
    RenumberEventCueNumbers,
    Seek,
    SelectEveryOtherEvents,
    SetActivePlaybackTarget,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    TriggerTakeAction,
)
from echozero.application.timeline.object_actions import canonical_action_id


class _TimelineRuntimeShell(Protocol):
    def presentation(self) -> TimelinePresentation: ...


class _AddSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_song_from_path(self, title: str, audio_path: str) -> TimelinePresentation | None: ...


class _SelectSongRuntimeShell(_TimelineRuntimeShell, Protocol):
    def select_song(self, song_id: str) -> TimelinePresentation | None: ...


class _SwitchSongVersionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def switch_song_version(self, song_version_id: str) -> TimelinePresentation | None: ...


class _AddSongVersionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_song_version(
        self,
        song_id: str,
        audio_path: str,
        *,
        label: str | None = None,
    ) -> TimelinePresentation | None: ...


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


class _AddLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def add_layer(self, kind: LayerKind) -> TimelinePresentation | None: ...


class _DeleteLayerRuntimeShell(_TimelineRuntimeShell, Protocol):
    def delete_layer(self, layer_id: str) -> TimelinePresentation | None: ...


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
    def trigger_contract_action(self, action: InspectorAction) -> None:
        """Execute one inspector contract action against the widget/runtime surface."""
        host = cast(_ContractActionHost, self)
        params = action.params
        action_id = canonical_action_id(action.action_id) or action.action_id
        if action_id == "seek_here":
            time_seconds = params.get("time_seconds")
            if isinstance(time_seconds, (int, float)):
                host._dispatch(Seek(float(time_seconds)))
            return
        if action_id == "nudge_left":
            host._dispatch(
                NudgeSelectedEvents(direction=-1, steps=_coerce_step_count(params.get("steps", 1)))
            )
            return
        if action_id == "nudge_right":
            host._dispatch(
                NudgeSelectedEvents(direction=1, steps=_coerce_step_count(params.get("steps", 1)))
            )
            return
        if action_id == "timeline.duplicate_selection":
            host._dispatch(
                DuplicateSelectedEvents(steps=_coerce_step_count(params.get("steps", 1)))
            )
            return
        if action_id == "selection.select_every_other":
            scope = event_batch_scope_from_params(params)
            if scope is not None:
                host._dispatch(SelectEveryOtherEvents(scope=scope))
            return
        if action_id == "selection.renumber_cues_from_one":
            scope = event_batch_scope_from_params(params)
            if scope is not None:
                host._dispatch(RenumberEventCueNumbers(scope=scope, start_at=1, step=1))
            return
        if action_id == "song.add":
            self._run_add_song_from_path_action()
            return
        if action_id == "song.select":
            self._run_select_song_action(params)
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
        if action_id == "add_event_layer":
            self._run_add_layer_action(LayerKind.EVENT)
            return
        if action_id == "delete_layer":
            self._run_delete_layer_action(params)
            return
        if action_id == "preview_event_clip":
            self._handle_preview_event_clip(params)
            return
        if action_id == "set_active_playback_target":
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is not None:
                host._dispatch(SetActivePlaybackTarget(layer_id=layer_id, take_id=None))
            return
        if action_id in {"gain_down", "gain_unity", "gain_up", "set_gain_custom"}:
            layer_id = _coerce_layer_id(params.get("layer_id"))
            gain_db = params.get("gain_db")
            if layer_id is not None and isinstance(gain_db, (int, float)):
                host._dispatch(SetGain(layer_id=layer_id, gain_db=float(gain_db)))
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

    def _run_add_song_from_path_action(self) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddSongRuntimeShell | None, host._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "add_song_from_path", None)):
            host._message_box.warning(
                host._widget,
                "Add Song",
                "This runtime does not support adding songs from a path.",
            )
            return
        title, accepted = host._input_dialog.getText(host._widget, "Add Song", "Song title")
        if not accepted or not title.strip():
            return
        audio_path, _ = host._file_dialog.getOpenFileName(
            host._widget,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aiff *.aif *.ogg);;All Files (*)",
        )
        if not audio_path:
            return
        try:
            updated = runtime.add_song_from_path(title.strip(), audio_path)
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
        if not isinstance(audio_path, str) or not audio_path.strip():
            audio_path, _ = host._file_dialog.getOpenFileName(
                host._widget,
                "Select Audio File",
                "",
                "Audio Files (*.wav *.mp3 *.flac *.aiff *.aif *.ogg);;All Files (*)",
            )
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
        try:
            updated = runtime.add_song_version(song_id, audio_path, label=resolved_label)
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

    def _run_add_layer_action(self, kind: LayerKind) -> None:
        host = cast(_ContractActionHost, self)
        runtime = cast(_AddLayerRuntimeShell | None, host._resolve_runtime_shell())
        label = f"Add {kind.value.title()} Layer"
        if runtime is None or not callable(getattr(runtime, "add_layer", None)):
            host._message_box.warning(
                host._widget,
                label,
                f"This runtime does not support adding {kind.value} layers.",
            )
            return
        try:
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
            (
                f'Delete layer "{label}"?\n\n'
                "This action cannot be undone."
            ),
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

    def _resolve_layer_title(self, layer_id: str) -> str:
        host = cast(_ContractActionHost, self)
        for layer in host._get_presentation().layers:
            if str(layer.layer_id) == layer_id:
                return layer.title
        return "Selected Layer"

    def _resolve_selected_layer_id(self) -> str | None:
        host = cast(_ContractActionHost, self)
        presentation = host._get_presentation()
        return str(presentation.selected_layer_id) if presentation.selected_layer_id is not None else None

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
            raw_selected_pool_no = params.get("timecode_pool_no")
            selected_pool_no = (
                None
                if raw_selected_pool_no in {None, "", 0}
                else int(raw_selected_pool_no)
            )
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
                "Song version MA3 timecode pool",
                [label for label, _value in options],
                default_index,
                False,
            )
            if not accepted:
                return
            selected_pool_no = next(
                (
                    value
                    for label, value in options
                    if label == chosen_label
                ),
                current_pool_no,
            )
        try:
            updated = runtime.set_song_version_ma3_timecode_pool(
                song_version_id.strip(),
                selected_pool_no,
            )
        except Exception as exc:
            host._message_box.warning(host._widget, "Set MA3 TC Pool", str(exc))
            return
        host._set_presentation(updated if updated is not None else runtime.presentation())

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
