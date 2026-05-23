"""Operator-first MA3 push helpers for the timeline widget.
Exists to keep saved-route pickers and push dispatch out of the legacy batch transfer path.
Connects layer-local send actions to the typed MA3 push application intents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from echozero.application.settings import AppSettingsService
from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    ManualPushFlowPresentation,
    ManualPushSequenceOptionPresentation,
    ManualPushTrackOptionPresentation,
    TimelinePresentation,
)
from echozero.ui.qt.ma3_connection_hud import MA3ConnectionHUD
from echozero.ui.qt.timeline.ma3_cue_matcher import (
    EventCueMatchRow,
    MA3CueMatcherDialog,
    MA3CueOption,
)
from echozero.ui.qt.timeline.manual_push_route import ManualPushRouteDialog
from echozero.application.shared.cue_numbers import (
    cue_number_from_ref_text,
    cue_number_text,
    normalize_ma_scaled_cue_number,
    parse_positive_cue_number,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import (
    EventCueMappingEdit,
    SetPushTransferMode,
    UpdateEventCueMappings,
)
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Timecode,
    CreateMA3Track,
    CreateMA3TrackGroup,
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3PushScope,
    MA3PushTargetMode,
    MA3SequenceCreationMode,
    MA3TrackSequenceAction,
    PushLayerToMA3,
    RefreshMA3Sequences,
    RefreshMA3PushTracks,
    SetLayerMA3Route,
)
from echozero.ui.qt.timeline.widget_action_contract_mixin import _coerce_layer_id
from echozero.ui.qt.timeline.widget_action_transfer_workspace_mixin import (
    TimelineWidgetTransferWorkspaceMixin,
    _TransferActionHost,
)


@dataclass(slots=True)
class _ManualPushRoutePopupResult:
    target_track_coord: str | None
    ma3_channel_no: int | None = None
    sequence_action: MA3TrackSequenceAction | None = None
    apply_mode: MA3PushApplyMode | None = None
    skip_cue_1: bool = False


@dataclass(slots=True)
class _SavedRoutePushDecision:
    apply_mode: MA3PushApplyMode | str
    skip_cue_1: bool = False


_SAVED_ROUTE_REROUTE = "reroute"


class TimelineWidgetMA3PushActionMixin(TimelineWidgetTransferWorkspaceMixin):
    """Routes layer-local route/send actions into the typed MA3 push contract."""

    def _handle_route_layer_to_ma3_track(self, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        presentation = host._get_presentation()
        layer = self._find_layer_presentation(presentation, layer_id)
        if layer is None:
            return True
        popup_result = self._coerce_route_popup_result(
            self._open_manual_push_route_popup(
                title="Route Layer to MA3 Track",
                prompt="MA3 track",
                reference_track_coord=layer.sync_target_label,
                reference_channel_no=layer.sync_target_channel_no,
                include_sequence_picker=True,
                include_apply_mode_picker=False,
                layer=layer,
            ),
        )
        if popup_result is None or popup_result.target_track_coord is None:
            return True
        selected_track = self._resolve_popup_track(
            target_track_coord=popup_result.target_track_coord,
            refresh_target_track_coord=layer.sync_target_label,
        )
        if selected_track is None:
            return True
        sequence_action = popup_result.sequence_action
        if sequence_action is None:
            sequence_action, accepted = self._resolve_ma3_sequence_action(
                title="Route Layer to MA3 Track",
                layer=layer,
                target_track=selected_track,
            )
            if not accepted:
                return True
        host._dispatch(
            SetLayerMA3Route(
                layer_id=layer_id,
                target_track_coord=selected_track.coord,
                ma3_channel_no=popup_result.ma3_channel_no,
                sequence_action=sequence_action,
            )
        )
        if self._should_match_section_layer_to_existing_sequence(
            layer=layer,
            target_track=selected_track,
            sequence_action=sequence_action,
        ):
            self._match_events_to_ma3_cues(
                layer_id=layer_id,
                selected_event_ids=[event.event_id for event in layer.events],
                refresh_track=False,
                abort_on_cancel=False,
                title="Match Section Cues to MA3",
                show_info_messages=False,
            )
        return True

    def _handle_send_layer_to_ma3(self, params: dict[str, object]) -> bool:
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        return self._dispatch_ma3_push(
            action_title="Send Layer to MA3",
            layer_id=layer_id,
            scope=MA3PushScope.LAYER_MAIN,
            require_saved_route=True,
        )

    def _handle_send_selected_events_to_ma3(self, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        selected_event_ids = self._coerce_explicit_event_ids(params.get("event_ids"))
        if not selected_event_ids:
            selected_event_ids = self._selected_main_event_ids_for_layer(
                host._get_presentation(), layer_id
            )
        if not selected_event_ids:
            host._message_box.information(
                host._widget,
                "Send Selected Events to MA3",
                "Select one or more main-take events on this layer first.",
            )
            return True
        return self._dispatch_ma3_push(
            action_title="Send Selected Events to MA3",
            layer_id=layer_id,
            scope=MA3PushScope.SELECTED_EVENTS,
            selected_event_ids=selected_event_ids,
            require_saved_route=True,
            prompt_for_saved_route=True,
        )

    def _handle_match_events_to_ma3_cues(self, params: dict[str, object]) -> bool:
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        event_ids = self._coerce_explicit_event_ids(params.get("event_ids"))
        self._match_events_to_ma3_cues(
            layer_id=layer_id,
            selected_event_ids=event_ids,
            refresh_track=True,
            abort_on_cancel=False,
            title="Match EZ Events to MA3 Cues",
            show_info_messages=True,
        )
        return True

    def _match_events_to_ma3_cues(
        self,
        *,
        layer_id: LayerId,
        selected_event_ids: list[EventId],
        refresh_track: bool,
        abort_on_cancel: bool,
        title: str,
        show_info_messages: bool,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        layer = self._find_layer_presentation(presentation, layer_id)
        if layer is None:
            return True
        if not layer.sync_target_label:
            if show_info_messages:
                host._message_box.information(
                    host._widget,
                    title,
                    "Route this layer to an MA3 track first.",
                )
            return True

        selected_track = self._find_ma3_track_by_coord(
            layer.sync_target_label,
            refresh=refresh_track,
        )
        if selected_track is None:
            return True
        if selected_track.sequence_no is None:
            if show_info_messages:
                host._message_box.information(
                    host._widget,
                    title,
                    (
                        f"{selected_track.name} ({selected_track.coord}) has no assigned MA3 sequence.\n"
                        "Assign a sequence first, then match cues."
                    ),
                )
            return True

        if not selected_event_ids:
            selected_event_ids = self._selected_main_event_ids_for_layer(presentation, layer_id)
        event_id_lookup = set(selected_event_ids)
        target_events = [
            event
            for event in layer.events
            if not event_id_lookup or event.event_id in event_id_lookup
        ]
        if not target_events:
            if show_info_messages:
                host._message_box.information(
                    host._widget,
                    title,
                    "No timeline events are available to match.",
                )
            return True

        cue_options = self._ma3_cue_options_for_track(selected_track.coord)
        if not cue_options and layer.kind is LayerKind.SECTION:
            cue_options = self._section_layer_cue_options_from_events(target_events)
        if not cue_options:
            if show_info_messages:
                host._message_box.information(
                    host._widget,
                    title,
                    (
                        f"No MA3 cues were found for {selected_track.name} ({selected_track.coord}).\n"
                        "Add cue events in MA3 first, then retry matching."
                    ),
                )
            return True

        rows = [
            EventCueMatchRow(
                event_id=str(event.event_id),
                start=float(event.start),
                label=str(event.label or "").strip() or "Event",
                current_cue_number=event.cue_number,
                current_cue_ref=event.cue_ref,
            )
            for event in target_events
        ]
        dialog = MA3CueMatcherDialog(
            title=title,
            subtitle=(
                f"TC/TG/TR {selected_track.coord}  |  Sequence {selected_track.sequence_no}\n"
                "Map timeline events to existing MA3 cues before sending."
            ),
            rows=rows,
            cue_options=cue_options,
            parent=host._widget,
        )
        if not bool(dialog.exec()):
            return not abort_on_cancel

        mappings = dialog.selected_mappings()
        if not mappings:
            return True
        host._dispatch(
            UpdateEventCueMappings(
                layer_id=layer.layer_id,
                take_id=layer.main_take_id,
                edits=[
                    EventCueMappingEdit(
                        event_id=EventId(mapping.event_id),
                        cue_number=mapping.cue_number,
                        cue_ref=mapping.cue_ref,
                    )
                    for mapping in mappings
                ],
            )
        )
        return True

    @staticmethod
    def _should_match_section_layer_to_existing_sequence(
        *,
        layer: LayerPresentation,
        target_track: ManualPushTrackOptionPresentation,
        sequence_action: MA3TrackSequenceAction | None,
    ) -> bool:
        if layer.kind is not LayerKind.SECTION:
            return False
        if isinstance(sequence_action, AssignMA3TrackSequence):
            return True
        return target_track.sequence_no is not None

    @staticmethod
    def _section_layer_cue_options_from_events(
        events: list[EventPresentation],
    ) -> list[MA3CueOption]:
        cue_options: list[MA3CueOption] = []
        seen: set[str] = set()
        for index, event in enumerate(sorted(events, key=lambda item: float(item.start)), start=1):
            cue_number = event.cue_number
            if cue_number is None:
                cue_number = cue_number_from_ref_text(event.cue_ref)
            if cue_number is None:
                cue_number = index
            cue_no_text = cue_number_text(cue_number)
            if cue_no_text is None or cue_no_text in seen:
                continue
            label = str(event.label or "").strip() or f"Cue {cue_no_text}"
            cue_options.append(MA3CueOption(cue_number=cue_number, name=label))
            seen.add(cue_no_text)
        return cue_options

    def _handle_send_to_different_track_once(self, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        layer = self._find_layer_presentation(host._get_presentation(), layer_id)
        if layer is None:
            return True
        selected_event_ids = self._selected_main_event_ids_for_layer(
            host._get_presentation(), layer_id
        )
        scope = MA3PushScope.SELECTED_EVENTS if selected_event_ids else MA3PushScope.LAYER_MAIN
        popup_result = self._coerce_route_popup_result(
            self._open_manual_push_route_popup(
                title="Send to Different Track Once",
                prompt="Temporary MA3 track",
                reference_track_coord=layer.sync_target_label,
                reference_channel_no=layer.sync_target_channel_no,
                include_sequence_picker=True,
                include_apply_mode_picker=True,
                layer=layer,
                default_apply_mode=self._current_ma3_apply_mode(),
            ),
            fallback_apply_mode=self._current_ma3_apply_mode(),
        )
        if popup_result is None or popup_result.target_track_coord is None:
            return True
        selected_track = self._resolve_popup_track(
            target_track_coord=popup_result.target_track_coord,
            refresh_target_track_coord=layer.sync_target_label,
        )
        if selected_track is None:
            return True
        return self._dispatch_ma3_push(
            action_title="Send to Different Track Once",
            layer_id=layer_id,
            scope=scope,
            selected_event_ids=selected_event_ids,
            require_saved_route=False,
            explicit_target=selected_track,
            explicit_channel_no=popup_result.ma3_channel_no,
            explicit_sequence_action=popup_result.sequence_action,
            explicit_apply_mode=popup_result.apply_mode,
            explicit_skip_cue_1=popup_result.skip_cue_1,
        )

    def _dispatch_ma3_push(
        self,
        *,
        action_title: str,
        layer_id: LayerId,
        scope: MA3PushScope,
        selected_event_ids: list[EventId] | None = None,
        require_saved_route: bool,
        prompt_for_saved_route: bool = False,
        explicit_target: ManualPushTrackOptionPresentation | None = None,
        explicit_channel_no: int | None = None,
        explicit_sequence_action: MA3TrackSequenceAction | None = None,
        explicit_apply_mode: MA3PushApplyMode | None = None,
        explicit_skip_cue_1: bool = False,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        layer = self._find_layer_presentation(presentation, layer_id)
        if layer is None:
            return True

        selected_ids = list(selected_event_ids or [])
        if scope is MA3PushScope.SELECTED_EVENTS and not selected_ids:
            host._message_box.information(
                host._widget,
                "Send Selected Events to MA3",
                "Select one or more main-take events on this layer first.",
            )
            return True

        target_mode = MA3PushTargetMode.SAVED_ROUTE
        target_track = explicit_target
        ma3_channel_no = explicit_channel_no
        sequence_action: MA3TrackSequenceAction | None = explicit_sequence_action
        apply_mode: MA3PushApplyMode | None = explicit_apply_mode
        skip_cue_1 = bool(explicit_skip_cue_1)
        if require_saved_route and (prompt_for_saved_route or not layer.sync_target_label):
            routed = self._choose_and_save_ma3_route(
                action_title=action_title,
                layer_id=layer_id,
                layer=layer,
                apply_mode=apply_mode,
            )
            if routed is None:
                return True
            target_track, ma3_channel_no, sequence_action, apply_mode, layer, skip_cue_1 = routed
        elif require_saved_route:
            target_track = self._find_ma3_track_by_coord(
                layer.sync_target_label,
                refresh=True,
            )
            if target_track is None:
                return True
            ma3_channel_no = layer.sync_target_channel_no
            selected_apply_mode = self._confirm_send_to_saved_ma3_route(
                layer=layer,
                target_track=target_track,
            )
            if selected_apply_mode is None:
                return True
            selected_skip_cue_1 = False
            if isinstance(selected_apply_mode, _SavedRoutePushDecision):
                selected_skip_cue_1 = selected_apply_mode.skip_cue_1
                selected_apply_mode = selected_apply_mode.apply_mode
            if selected_apply_mode == _SAVED_ROUTE_REROUTE:
                routed = self._choose_and_save_ma3_route(
                    action_title=action_title,
                    layer_id=layer_id,
                    layer=layer,
                    apply_mode=apply_mode,
                )
                if routed is None:
                    return True
                target_track, ma3_channel_no, sequence_action, apply_mode, layer, skip_cue_1 = routed
            else:
                apply_mode = selected_apply_mode
                skip_cue_1 = selected_skip_cue_1
        elif not require_saved_route:
            target_mode = MA3PushTargetMode.DIFFERENT_TRACK_ONCE
            if target_track is None:
                popup_result = self._coerce_route_popup_result(
                    self._open_manual_push_route_popup(
                        title=action_title,
                        prompt="MA3 track",
                        reference_track_coord=layer.sync_target_label,
                        reference_channel_no=layer.sync_target_channel_no,
                        include_sequence_picker=True,
                        include_apply_mode_picker=True,
                        layer=layer,
                        default_apply_mode=self._current_ma3_apply_mode(),
                    ),
                    fallback_apply_mode=self._current_ma3_apply_mode(),
                )
                if popup_result is None or popup_result.target_track_coord is None:
                    return True
                target_track = self._resolve_popup_track(
                    target_track_coord=popup_result.target_track_coord,
                    refresh_target_track_coord=layer.sync_target_label,
                )
                if target_track is None:
                    return True
                ma3_channel_no = popup_result.ma3_channel_no
                sequence_action = popup_result.sequence_action
                apply_mode = popup_result.apply_mode or apply_mode
                skip_cue_1 = popup_result.skip_cue_1

        if target_mode is MA3PushTargetMode.SAVED_ROUTE and target_track is None:
            target_track = self._find_ma3_track_by_coord(
                layer.sync_target_label,
                refresh=True,
            )
        if (
            sequence_action is None
            and target_track is not None
            and target_track.sequence_no is None
        ):
            sequence_action, accepted = self._resolve_ma3_sequence_action(
                title=action_title,
                layer=layer,
                target_track=target_track,
            )
            if not accepted:
                return True

        if target_track is not None and self._track_requires_auto_merge(target_track):
            apply_mode = MA3PushApplyMode.MERGE

        if apply_mode is None:
            apply_mode = self._current_ma3_apply_mode()

        selected_count = (
            len(selected_ids) if scope is MA3PushScope.SELECTED_EVENTS else len(layer.events)
        )
        if (
            target_track is not None
            and target_track.sequence_no is not None
            and selected_count > 0
            and self._section_layer_needs_cue_matching(
                layer=layer,
                selected_event_ids=selected_ids,
                target_track=target_track,
            )
        ):
            mapping_event_ids = (
                selected_ids if selected_ids else [event.event_id for event in layer.events]
            )
            if not self._match_events_to_ma3_cues(
                layer_id=layer_id,
                selected_event_ids=mapping_event_ids,
                refresh_track=False,
                abort_on_cancel=True,
                title="Match Section Cues to MA3",
                show_info_messages=False,
            ):
                return True
            presentation = host._get_presentation()
            refreshed_layer = self._find_layer_presentation(presentation, layer_id)
            if refreshed_layer is None:
                return True
            layer = refreshed_layer

        host._dispatch(SetPushTransferMode(mode=apply_mode.value))

        if (
            apply_mode is MA3PushApplyMode.OVERWRITE
            and target_track is not None
            and target_track.sequence_no is not None
            and selected_count > 0
            and layer.kind is not LayerKind.SECTION
        ):
            mapping_event_ids = (
                selected_ids if selected_ids else [event.event_id for event in layer.events]
            )
            if not self._match_events_to_ma3_cues(
                layer_id=layer_id,
                selected_event_ids=mapping_event_ids,
                refresh_track=False,
                abort_on_cancel=True,
                title="Match EZ Events to MA3 Cues",
                show_info_messages=False,
            ):
                return True

        if apply_mode is MA3PushApplyMode.OVERWRITE and not self._confirm_overwrite_push(
            selected_count=selected_count,
            target_track=target_track,
            fallback_target_coord=(
                target_track.coord if target_track is not None else layer.sync_target_label
            ),
        ):
            return True

        host._dispatch(
            PushLayerToMA3(
                layer_id=layer_id,
                scope=scope,
                target_mode=target_mode,
                apply_mode=apply_mode,
                ma3_channel_no=ma3_channel_no,
                target_track_coord=(
                    (None if target_track is None else target_track.coord)
                    if target_mode is MA3PushTargetMode.DIFFERENT_TRACK_ONCE
                    else None
                ),
                selected_event_ids=selected_ids,
                sequence_action=sequence_action,
                skip_cue_1=skip_cue_1,
            )
        )
        return True

    def _section_layer_needs_cue_matching(
        self,
        *,
        layer: LayerPresentation,
        selected_event_ids: list[EventId],
        target_track: ManualPushTrackOptionPresentation,
    ) -> bool:
        if layer.kind is not LayerKind.SECTION or target_track.sequence_no is None:
            return False
        event_id_lookup = set(selected_event_ids)
        target_events = [
            event
            for event in layer.events
            if not event_id_lookup or event.event_id in event_id_lookup
        ]
        if not target_events:
            return False

        cue_options = self._ma3_cue_options_for_track(target_track.coord)
        if not cue_options:
            return True
        available_cue_numbers = {
            cue_number_text(option.cue_number)
            for option in cue_options
            if cue_number_text(option.cue_number) is not None
        }
        ordered_layer_events = sorted(
            layer.events,
            key=lambda item: (float(item.start), str(item.event_id)),
        )
        internal_numbers = {
            event.event_id: index for index, event in enumerate(ordered_layer_events, start=1)
        }
        desired_cue_numbers: list[str] = []
        for event in sorted(
            target_events,
            key=lambda item: (float(item.start), str(item.event_id)),
        ):
            cue_number = cue_number_from_ref_text(event.cue_ref)
            if cue_number is None:
                cue_number = internal_numbers.get(event.event_id)
            cue_no_text = cue_number_text(cue_number)
            if cue_no_text is None or cue_no_text not in available_cue_numbers:
                return True
            desired_cue_numbers.append(cue_no_text)
        return len(set(desired_cue_numbers)) != len(desired_cue_numbers)

    def _choose_and_save_ma3_route(
        self,
        *,
        action_title: str,
        layer_id: LayerId,
        layer: LayerPresentation,
        apply_mode: MA3PushApplyMode | None,
    ) -> tuple[
        ManualPushTrackOptionPresentation,
        int | None,
        MA3TrackSequenceAction | None,
        MA3PushApplyMode | None,
        LayerPresentation,
        bool,
    ] | None:
        host = cast(_TransferActionHost, self)
        popup_result = self._coerce_route_popup_result(
            self._open_manual_push_route_popup(
                title=action_title,
                prompt="Saved MA3 track",
                reference_track_coord=layer.sync_target_label,
                reference_channel_no=layer.sync_target_channel_no,
                include_sequence_picker=True,
                include_apply_mode_picker=True,
                layer=layer,
                default_apply_mode=self._current_ma3_apply_mode(),
            ),
            fallback_apply_mode=self._current_ma3_apply_mode(),
        )
        if popup_result is None or popup_result.target_track_coord is None:
            return None
        target_track = self._resolve_popup_track(
            target_track_coord=popup_result.target_track_coord,
            refresh_target_track_coord=layer.sync_target_label,
        )
        if target_track is None:
            return None
        ma3_channel_no = popup_result.ma3_channel_no
        sequence_action = popup_result.sequence_action
        resolved_apply_mode = popup_result.apply_mode or apply_mode
        host._dispatch(
            SetLayerMA3Route(
                layer_id=layer_id,
                target_track_coord=target_track.coord,
                ma3_channel_no=ma3_channel_no,
                sequence_action=sequence_action,
            )
        )
        refreshed_layer = self._find_layer_presentation(host._get_presentation(), layer_id)
        if refreshed_layer is None:
            return None
        return (
            target_track,
            ma3_channel_no,
            sequence_action,
            resolved_apply_mode,
            refreshed_layer,
            popup_result.skip_cue_1,
        )

    def _confirm_send_to_saved_ma3_route(
        self,
        *,
        layer: LayerPresentation,
        target_track: ManualPushTrackOptionPresentation | None,
    ) -> MA3PushApplyMode | str | _SavedRoutePushDecision | None:
        host = cast(_TransferActionHost, self)
        target_label = (
            f"{target_track.name} ({target_track.coord})"
            if target_track is not None
            else layer.sync_target_label
        )
        if not target_label:
            target_label = "a saved MA3 target"
        layer_label = layer.title or str(layer.layer_id)
        dialog = QDialog(host._widget)
        dialog.setWindowTitle("Send to MA3")
        layout = QVBoxLayout(dialog)
        label = QLabel(
            (
                f"'{layer_label}' is already routed to {target_label}.\n\n"
                "Choose how to send this layer."
            ),
            dialog,
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        skip_cue_1_check = QCheckBox("Skip cue 1 for timecode", dialog)
        skip_cue_1_check.setVisible(layer.kind is LayerKind.SECTION)
        layout.addWidget(skip_cue_1_check)
        buttons = QDialogButtonBox(dialog)
        merge_button = buttons.addButton("Merge", QDialogButtonBox.ButtonRole.AcceptRole)
        overwrite_button = buttons.addButton(
            "Overwrite",
            QDialogButtonBox.ButtonRole.DestructiveRole,
        )
        reroute_button = buttons.addButton("Reroute", QDialogButtonBox.ButtonRole.ActionRole)
        cancel_button = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        chosen: dict[str, object] = {"value": None}

        def _choose(value: object) -> None:
            chosen["value"] = value
            dialog.accept()

        merge_button.clicked.connect(lambda: _choose(MA3PushApplyMode.MERGE))
        overwrite_button.clicked.connect(lambda: _choose(MA3PushApplyMode.OVERWRITE))
        reroute_button.clicked.connect(lambda: _choose(_SAVED_ROUTE_REROUTE))
        cancel_button.clicked.connect(dialog.reject)
        default_button = merge_button
        if self._current_ma3_apply_mode() is MA3PushApplyMode.OVERWRITE:
            default_button = overwrite_button
        set_default = getattr(default_button, "setDefault", None)
        if callable(set_default):
            set_default(True)
        if not bool(dialog.exec()):
            return None
        value = chosen["value"]
        if value in {MA3PushApplyMode.MERGE, MA3PushApplyMode.OVERWRITE, _SAVED_ROUTE_REROUTE}:
            if layer.kind is LayerKind.SECTION and value != _SAVED_ROUTE_REROUTE:
                return _SavedRoutePushDecision(
                    apply_mode=cast(MA3PushApplyMode | str, value),
                    skip_cue_1=bool(skip_cue_1_check.isChecked()),
                )
            return cast(MA3PushApplyMode | str, value)
        return None

    def _resolve_ma3_sequence_action(
        self,
        *,
        title: str,
        layer: LayerPresentation,
        target_track: ManualPushTrackOptionPresentation,
    ) -> tuple[MA3TrackSequenceAction | None, bool]:
        host = cast(_TransferActionHost, self)
        if target_track.sequence_no is not None:
            return None, True

        host._dispatch(RefreshMA3Sequences())
        flow = host._get_presentation().manual_push_flow
        prep_labels = [
            "Use existing sequence",
            "Create next available sequence",
            "Create sequence in current song range",
        ]
        chosen_label, accepted = host._input_dialog.getItem(
            host._widget,
            title,
            self._manual_push_sequence_prep_prompt(target_track=target_track, flow=flow),
            prep_labels,
            0,
            False,
        )
        if not accepted:
            return None, False
        if chosen_label == prep_labels[0]:
            sequence_action = self._choose_existing_ma3_sequence(
                title=title,
                target_track=target_track,
                flow=flow,
            )
            return (
                sequence_action,
                sequence_action is not None,
            )
        preferred_name = self._default_ma3_sequence_name(
            presentation=host._get_presentation(),
            layer=layer,
        )
        if chosen_label == prep_labels[1]:
            return (
                CreateMA3Sequence(
                    creation_mode=MA3SequenceCreationMode.NEXT_AVAILABLE,
                    preferred_name=preferred_name,
                ),
                True,
            )
        if flow.current_song_sequence_range is None:
            host._message_box.warning(
                host._widget,
                title,
                "Current song MA3 sequence range is not available right now.",
            )
            return None, False
        return (
            CreateMA3Sequence(
                creation_mode=MA3SequenceCreationMode.CURRENT_SONG_RANGE,
                preferred_name=preferred_name,
            ),
            True,
        )

    def _choose_existing_ma3_sequence(
        self,
        *,
        title: str,
        target_track: ManualPushTrackOptionPresentation,
        flow: ManualPushFlowPresentation,
    ) -> AssignMA3TrackSequence | None:
        host = cast(_TransferActionHost, self)
        if not flow.available_sequences:
            host._message_box.warning(
                host._widget,
                title,
                "No MA3 sequences are available right now.",
            )
            return None
        labels = [
            self._manual_push_sequence_label(sequence) for sequence in flow.available_sequences
        ]
        chosen_label, accepted = host._input_dialog.getItem(
            host._widget,
            title,
            self._manual_push_existing_sequence_prompt(
                target_track=target_track,
                flow=flow,
            ),
            labels,
            0,
            False,
        )
        if not accepted:
            return None
        selected_sequence = next(
            (
                sequence
                for sequence, label in zip(flow.available_sequences, labels)
                if label == chosen_label
            ),
            None,
        )
        if selected_sequence is None:
            return None
        return AssignMA3TrackSequence(
            target_track_coord=target_track.coord,
            sequence_no=selected_sequence.number,
        )

    def _choose_ma3_target_track(
        self,
        *,
        title: str,
        prompt: str,
        reference_track_coord: str | None = None,
    ) -> ManualPushTrackOptionPresentation | None:
        host = cast(_TransferActionHost, self)
        if not self._ensure_ma3_timecode_pool_ready(
            title=title,
            reference_track_coord=reference_track_coord,
        ):
            return None
        if not self._refresh_and_validate_ma3_push_tracks(
            target_track_coord=reference_track_coord,
        ):
            return None
        flow = host._get_presentation().manual_push_flow
        if not flow.available_tracks:
            host._message_box.warning(
                host._widget,
                title,
                "No MA3 tracks are available right now.",
            )
            return None
        popup_result = self._coerce_route_popup_result(
            self._open_manual_push_route_popup(
                title=title,
                prompt=prompt,
                reference_track_coord=reference_track_coord,
            )
        )
        if popup_result is None or popup_result.target_track_coord is None:
            return None
        refreshed_flow = host._get_presentation().manual_push_flow
        return next(
            (
                track
                for track in refreshed_flow.available_tracks
                if track.coord == popup_result.target_track_coord
            ),
            None,
        )

    def _open_manual_push_route_popup(
        self,
        *,
        title: str,
        prompt: str,
        reference_track_coord: str | None,
        reference_channel_no: int | None = None,
        include_sequence_picker: bool = False,
        include_apply_mode_picker: bool = False,
        layer: LayerPresentation | None = None,
        default_apply_mode: MA3PushApplyMode | None = None,
    ) -> str | _ManualPushRoutePopupResult | None:
        host = cast(_TransferActionHost, self)
        dialog = ManualPushRouteDialog(
            title=title,
            prompt=prompt,
            parent=host._widget,
        )
        resolved_default_apply_mode = (
            default_apply_mode
            if default_apply_mode is not None
            else self._current_ma3_apply_mode()
        )
        dialog.configure_sheet(
            show_sequence_controls=include_sequence_picker,
            show_apply_mode_controls=include_apply_mode_picker,
            show_channel_controls=False,
            show_skip_cue_1_controls=layer is not None and layer.kind is LayerKind.SECTION,
            default_channel_no=reference_channel_no,
            default_apply_mode=resolved_default_apply_mode.value,
        )

        def _refresh_dialog_flow(
            *,
            target_track_coord: str | None = None,
            timecode_no: int | None = None,
            track_group_no: int | None = None,
            channel_no: int | None = None,
        ) -> bool:
            if not self._refresh_and_validate_ma3_push_tracks(
                target_track_coord=target_track_coord,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            ):
                return False
            if include_sequence_picker:
                try:
                    host._dispatch(RefreshMA3Sequences())
                except Exception as exc:
                    host._message_box.warning(
                        host._widget,
                        "Refresh MA3 Sequences",
                        f"Unable to refresh MA3 sequences: {exc}",
                    )
                    return False
            dialog.set_flow(
                host._get_presentation().manual_push_flow,
                preferred_track_coord=target_track_coord,
                preferred_channel_no=(
                    channel_no if channel_no is not None else reference_channel_no
                ),
            )
            return True

        def _dispatch_create_action(*, action_title: str, intent) -> bool:
            try:
                host._dispatch(intent)
            except Exception as exc:
                host._message_box.warning(
                    host._widget,
                    action_title,
                    f"Unable to create MA3 target: {exc}",
                )
                return False
            return True

        if not _refresh_dialog_flow(target_track_coord=reference_track_coord):
            return None

        def _handle_timecode_selected(timecode_no: int) -> None:
            _refresh_dialog_flow(
                timecode_no=int(timecode_no),
                target_track_coord=dialog.selected_track_coord(),
                channel_no=dialog.selected_channel_no(),
            )

        def _handle_track_group_selected(track_group_no: int) -> None:
            _refresh_dialog_flow(
                timecode_no=dialog.selected_timecode_no(),
                track_group_no=int(track_group_no),
                target_track_coord=dialog.selected_track_coord(),
                channel_no=dialog.selected_channel_no(),
            )

        def _handle_refresh_requested() -> None:
            refreshed_flow = host._get_presentation().manual_push_flow
            selected_timecode_no = (
                dialog.selected_timecode_no() or refreshed_flow.selected_timecode_no
            )
            selected_track_group_no = (
                dialog.selected_track_group_no() or refreshed_flow.selected_track_group_no
            )
            if selected_timecode_no is None:
                selected_track_group_no = None
            _refresh_dialog_flow(
                timecode_no=selected_timecode_no,
                track_group_no=selected_track_group_no,
                target_track_coord=dialog.selected_track_coord(),
                channel_no=dialog.selected_channel_no(),
            )

        def _handle_create_timecode_requested() -> None:
            if not _dispatch_create_action(
                action_title="Create MA3 Timecode",
                intent=CreateMA3Timecode(),
            ):
                _refresh_dialog_flow(
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            refreshed_flow = host._get_presentation().manual_push_flow
            _refresh_dialog_flow(
                timecode_no=refreshed_flow.selected_timecode_no,
                target_track_coord=None,
                channel_no=dialog.selected_channel_no(),
            )

        def _handle_create_track_group_requested() -> None:
            refreshed_flow = host._get_presentation().manual_push_flow
            selected_timecode_no = (
                dialog.selected_timecode_no() or refreshed_flow.selected_timecode_no
            )
            if selected_timecode_no is None:
                host._message_box.warning(
                    host._widget,
                    "Create MA3 Track Group",
                    "Select or create a timecode pool first.",
                )
                _refresh_dialog_flow(
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            if not _dispatch_create_action(
                action_title="Create MA3 Track Group",
                intent=CreateMA3TrackGroup(timecode_no=int(selected_timecode_no)),
            ):
                _refresh_dialog_flow(
                    timecode_no=int(selected_timecode_no),
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            refreshed_flow = host._get_presentation().manual_push_flow
            _refresh_dialog_flow(
                timecode_no=int(selected_timecode_no),
                track_group_no=refreshed_flow.selected_track_group_no,
                target_track_coord=None,
                channel_no=dialog.selected_channel_no(),
            )

        def _handle_create_track_requested() -> None:
            refreshed_flow = host._get_presentation().manual_push_flow
            selected_timecode_no = (
                dialog.selected_timecode_no() or refreshed_flow.selected_timecode_no
            )
            selected_track_group_no = (
                dialog.selected_track_group_no() or refreshed_flow.selected_track_group_no
            )
            if selected_timecode_no is None or selected_track_group_no is None:
                host._message_box.warning(
                    host._widget,
                    "Create MA3 Track",
                    "Select or create a timecode pool and track group first.",
                )
                _refresh_dialog_flow(
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            accepted_name, preferred_track_name = self._prompt_ma3_track_name(
                title="Create MA3 Track",
                layer=layer,
            )
            if not accepted_name:
                _refresh_dialog_flow(
                    timecode_no=int(selected_timecode_no),
                    track_group_no=int(selected_track_group_no),
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            if not _dispatch_create_action(
                action_title="Create MA3 Track",
                intent=CreateMA3Track(
                    timecode_no=int(selected_timecode_no),
                    track_group_no=int(selected_track_group_no),
                    preferred_name=preferred_track_name,
                ),
            ):
                _refresh_dialog_flow(
                    timecode_no=int(selected_timecode_no),
                    track_group_no=int(selected_track_group_no),
                    target_track_coord=dialog.selected_track_coord(),
                    channel_no=dialog.selected_channel_no(),
                )
                return
            refreshed_flow = host._get_presentation().manual_push_flow
            _refresh_dialog_flow(
                timecode_no=int(selected_timecode_no),
                track_group_no=int(selected_track_group_no),
                target_track_coord=refreshed_flow.target_track_coord,
                channel_no=dialog.selected_channel_no(),
            )

        dialog.timecode_selected.connect(_handle_timecode_selected)
        dialog.track_group_selected.connect(_handle_track_group_selected)
        dialog.refresh_requested.connect(_handle_refresh_requested)
        dialog.create_timecode_requested.connect(_handle_create_timecode_requested)
        dialog.create_track_group_requested.connect(_handle_create_track_group_requested)
        dialog.create_track_requested.connect(_handle_create_track_requested)
        if not bool(dialog.exec()):
            return None
        selected_track_coord = dialog.selected_track_coord()
        if not include_sequence_picker and not include_apply_mode_picker:
            return selected_track_coord

        selected_track = (
            None
            if selected_track_coord is None
            else self._find_ma3_track_by_coord(selected_track_coord, refresh=False)
        )
        flow = host._get_presentation().manual_push_flow
        sequence_action: MA3TrackSequenceAction | None = None
        if include_sequence_picker and selected_track is not None:
            sequence_action = self._sequence_action_from_route_popup(
                dialog=dialog,
                target_track=selected_track,
                flow=flow,
                layer=layer,
                title=title,
            )
        apply_mode: MA3PushApplyMode | None = None
        if include_apply_mode_picker:
            apply_mode_text = str(dialog.selected_apply_mode() or "merge").strip().lower()
            apply_mode = (
                MA3PushApplyMode.OVERWRITE
                if apply_mode_text == MA3PushApplyMode.OVERWRITE.value
                else MA3PushApplyMode.MERGE
            )
        return _ManualPushRoutePopupResult(
            target_track_coord=selected_track_coord,
            ma3_channel_no=dialog.selected_channel_no(),
            sequence_action=sequence_action,
            apply_mode=apply_mode,
            skip_cue_1=dialog.skip_cue_1(),
        )

    @staticmethod
    def _coerce_route_popup_result(
        raw_result: str | _ManualPushRoutePopupResult | None,
        *,
        fallback_apply_mode: MA3PushApplyMode | None = None,
    ) -> _ManualPushRoutePopupResult | None:
        if raw_result is None:
            return None
        if isinstance(raw_result, _ManualPushRoutePopupResult):
            return raw_result
        track_coord = str(raw_result or "").strip()
        if not track_coord:
            return None
        return _ManualPushRoutePopupResult(
            target_track_coord=track_coord,
            ma3_channel_no=None,
            apply_mode=fallback_apply_mode,
        )

    def _sequence_action_from_route_popup(
        self,
        *,
        dialog: ManualPushRouteDialog,
        target_track: ManualPushTrackOptionPresentation,
        flow: ManualPushFlowPresentation,
        layer: LayerPresentation | None,
        title: str,
    ) -> MA3TrackSequenceAction | None:
        host = cast(_TransferActionHost, self)

        mode = dialog.selected_sequence_mode()
        if mode in {
            ManualPushRouteDialog.SEQUENCE_MODE_NONE,
            ManualPushRouteDialog.SEQUENCE_MODE_KEEP_ASSIGNED,
        }:
            return None
        if mode == ManualPushRouteDialog.SEQUENCE_MODE_ASSIGN_EXISTING:
            sequence_no = dialog.selected_sequence_no()
            if sequence_no is None:
                return None
            return AssignMA3TrackSequence(
                target_track_coord=target_track.coord,
                sequence_no=int(sequence_no),
            )

        preferred_name = self._preferred_ma3_sequence_name_for_track(
            track=target_track,
            presentation=host._get_presentation(),
            layer=layer,
        )
        if mode == ManualPushRouteDialog.SEQUENCE_MODE_CREATE_CURRENT_SONG:
            if flow.current_song_sequence_range is None:
                host._message_box.warning(
                    host._widget,
                    title,
                    "Current song MA3 sequence range is not available right now.",
                )
                return None
            return CreateMA3Sequence(
                creation_mode=MA3SequenceCreationMode.CURRENT_SONG_RANGE,
                preferred_name=preferred_name,
            )

        return CreateMA3Sequence(
            creation_mode=MA3SequenceCreationMode.NEXT_AVAILABLE,
            preferred_name=preferred_name,
        )

    def _choose_ma3_apply_mode(self) -> MA3PushApplyMode | None:
        host = cast(_TransferActionHost, self)
        current_mode = self._current_ma3_apply_mode().value
        mode_labels = ["Merge", "Overwrite"]
        default_index = 0 if current_mode == "merge" else 1
        chosen_mode, accepted = host._input_dialog.getItem(
            host._widget,
            "Send to MA3",
            "Apply mode",
            mode_labels,
            default_index,
            False,
        )
        if not accepted:
            return None
        return MA3PushApplyMode(chosen_mode.strip().lower())

    def _current_ma3_apply_mode(self) -> MA3PushApplyMode:
        host = cast(_TransferActionHost, self)
        current_mode = str(host._get_presentation().manual_push_flow.transfer_mode or "merge")
        normalized = current_mode.strip().lower()
        if normalized == MA3PushApplyMode.OVERWRITE.value:
            return MA3PushApplyMode.OVERWRITE
        return MA3PushApplyMode.MERGE

    def _confirm_overwrite_push(
        self,
        *,
        selected_count: int,
        target_track: ManualPushTrackOptionPresentation | None,
        fallback_target_coord: str,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        target_label = (
            f"{target_track.name} ({target_track.coord})"
            if target_track is not None
            else fallback_target_coord or "Unknown target"
        )
        target_count = (
            "Unknown"
            if target_track is None or target_track.event_count is None
            else str(target_track.event_count)
        )
        noun = "event" if selected_count == 1 else "events"
        reply = host._message_box.question(
            host._widget,
            "Overwrite MA3 Track",
            (
                f"Overwrite {target_label}?\n\n"
                f"Selected EZ events: {selected_count} {noun}\n"
                f"Current MA3 target events: {target_count}"
            ),
            host._message_box.StandardButton.Yes | host._message_box.StandardButton.No,
            host._message_box.StandardButton.No,
        )
        return reply == host._message_box.StandardButton.Yes

    @staticmethod
    def _find_layer_presentation(
        presentation: TimelinePresentation,
        layer_id: LayerId,
    ) -> LayerPresentation | None:
        for layer in presentation.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    @staticmethod
    def _selected_main_event_ids_for_layer(
        presentation: TimelinePresentation,
        layer_id: LayerId,
    ) -> list[EventId]:
        selected_ids: list[EventId] = []
        selected_ref_lookup = {
            (event_ref.layer_id, event_ref.take_id, event_ref.event_id)
            for event_ref in presentation.selected_event_refs
        }
        layer = TimelineWidgetMA3PushActionMixin._find_layer_presentation(presentation, layer_id)
        if layer is None or layer.main_take_id is None:
            return selected_ids

        for event in layer.events:
            if (
                event.is_selected
                or event.event_id in presentation.selected_event_ids
                or (layer.layer_id, layer.main_take_id, event.event_id) in selected_ref_lookup
            ):
                selected_ids.append(event.event_id)
        return list(dict.fromkeys(selected_ids))

    def _find_ma3_track_by_coord(
        self,
        coord: str,
        *,
        refresh: bool,
    ) -> ManualPushTrackOptionPresentation | None:
        host = cast(_TransferActionHost, self)
        if refresh:
            if not self._refresh_and_validate_ma3_push_tracks(target_track_coord=coord):
                return None
        for track in host._get_presentation().manual_push_flow.available_tracks:
            if track.coord == coord:
                return track
        return None

    def _resolve_popup_track(
        self,
        *,
        target_track_coord: str,
        refresh_target_track_coord: str | None,
    ) -> ManualPushTrackOptionPresentation | None:
        selected_track = self._find_ma3_track_by_coord(
            target_track_coord,
            refresh=False,
        )
        if selected_track is not None:
            return selected_track
        if not self._refresh_and_validate_ma3_push_tracks(
            target_track_coord=refresh_target_track_coord,
        ):
            return None
        return self._find_ma3_track_by_coord(
            target_track_coord,
            refresh=False,
        )

    @staticmethod
    def _track_requires_auto_merge(track: ManualPushTrackOptionPresentation) -> bool:
        event_count = getattr(track, "event_count", None)
        if event_count in {None, ""}:
            return True
        try:
            return int(event_count) <= 0
        except (TypeError, ValueError):
            return True

    def _refresh_and_validate_ma3_push_tracks(
        self,
        *,
        target_track_coord: str | None,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
        _retried_after_hud: bool = False,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        try:
            host._dispatch(
                RefreshMA3PushTracks(
                    target_track_coord=target_track_coord,
                    timecode_no=timecode_no,
                    track_group_no=track_group_no,
                )
            )
        except Exception as exc:
            host._message_box.warning(
                host._widget,
                "Refresh MA3 Tracks",
                f"Unable to refresh MA3 tracks: {exc}",
            )
            if not _retried_after_hud and self._open_ma3_connection_hud():
                return self._refresh_and_validate_ma3_push_tracks(
                    target_track_coord=target_track_coord,
                    timecode_no=timecode_no,
                    track_group_no=track_group_no,
                    _retried_after_hud=True,
                )
            return False
        return True

    def _open_ma3_connection_hud(self) -> bool:
        host = cast(_TransferActionHost, self)
        settings_service = self._resolve_runtime_app_settings_service(host)
        if settings_service is None:
            host._message_box.warning(
                host._widget,
                "MA3 OSC Connection",
                "The MA3 OSC connection overlay is unavailable in this shell.",
            )
            return False
        if not isinstance(settings_service, AppSettingsService):
            host._message_box.warning(
                host._widget,
                "MA3 OSC Connection",
                "MA3 OSC settings are not available in this shell.",
            )
            return False
        dialog = MA3ConnectionHUD(
            settings_service,
            on_saved=lambda _result: self._require_apply_ma3_osc_runtime_config(host),
            parent=host._widget,
        )
        return bool(dialog.exec())

    def _require_apply_ma3_osc_runtime_config(self, host: _TransferActionHost) -> None:
        if self._apply_ma3_osc_runtime_config(host):
            return
        raise RuntimeError("Unable to reconfigure the live MA3 connection for this session.")

    def _apply_ma3_osc_runtime_config(self, host: _TransferActionHost) -> bool:
        resolve_runtime_shell = getattr(host, "_resolve_runtime_shell", None)
        if callable(resolve_runtime_shell):
            runtime = resolve_runtime_shell()
            apply_runtime = getattr(runtime, "apply_ma3_osc_runtime_config", None)
            if callable(apply_runtime):
                return bool(apply_runtime())
        return False

    @staticmethod
    def _resolve_runtime_app_settings_service(
        host: _TransferActionHost,
    ) -> AppSettingsService | None:
        direct_service = getattr(host, "_app_settings_service", None)
        if isinstance(direct_service, AppSettingsService):
            return direct_service

        widget = getattr(host, "_widget", None)
        if isinstance(widget, object):
            widget_service = getattr(widget, "_app_settings_service", None)
            if isinstance(widget_service, AppSettingsService):
                return widget_service

        resolve_runtime_shell = getattr(host, "_resolve_runtime_shell", None)
        if not callable(resolve_runtime_shell):
            return None
        runtime = resolve_runtime_shell()
        if runtime is None:
            return None
        service = getattr(runtime, "app_settings_service", None)
        if isinstance(service, AppSettingsService):
            return service
        service = getattr(runtime, "_app_settings_service", None)
        if isinstance(service, AppSettingsService):
            return service
        return None

    def _ensure_ma3_timecode_pool_ready(
        self,
        *,
        title: str,
        reference_track_coord: str | None,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        if str(reference_track_coord or "").strip():
            return True
        presentation = host._get_presentation()
        if presentation.active_song_version_ma3_timecode_pool_no is not None:
            return True
        if not presentation.active_song_version_id:
            host._message_box.warning(
                host._widget,
                title,
                "Select a song version before choosing an MA3 target track.",
            )
            return False
        self.trigger_contract_action(
            InspectorAction(
                action_id="song.version.set_ma3_timecode_pool",
                label="Set MA3 TC Pool",
                params={"song_version_id": presentation.active_song_version_id},
            )
        )
        refreshed = host._get_presentation()
        return refreshed.active_song_version_ma3_timecode_pool_no is not None

    @staticmethod
    def _coerce_explicit_event_ids(raw_event_ids: object) -> list[EventId]:
        if not isinstance(raw_event_ids, list):
            return []
        event_ids: list[EventId] = []
        for raw_event_id in raw_event_ids:
            if not isinstance(raw_event_id, str):
                continue
            stripped = raw_event_id.strip()
            if stripped:
                event_ids.append(EventId(stripped))
        return list(dict.fromkeys(event_ids))

    def _ma3_cue_options_for_track(self, track_coord: str) -> list[MA3CueOption]:
        host = cast(_TransferActionHost, self)
        runtime = getattr(host, "_resolve_runtime_shell", lambda: None)()
        if runtime is None:
            return []
        sync_service = getattr(runtime, "sync_service", None)
        if sync_service is None:
            app = getattr(runtime, "_app", None)
            sync_service = None if app is None else getattr(app, "sync_service", None)
        if sync_service is None:
            return []

        list_sequence_cues = getattr(sync_service, "list_sequence_cues", None)
        raw_cues: list[object] = []
        selected_track = self._find_ma3_track_by_coord(track_coord, refresh=False)
        if (
            callable(list_sequence_cues)
            and selected_track is not None
            and selected_track.sequence_no is not None
        ):
            try:
                raw_sequence_cues = list_sequence_cues(sequence_no=int(selected_track.sequence_no))
                if isinstance(raw_sequence_cues, list):
                    raw_cues = list(raw_sequence_cues)
            except Exception:
                raw_cues = []

        if not raw_cues:
            list_pull_source_events = getattr(sync_service, "list_pull_source_events", None)
            if callable(list_pull_source_events):
                try:
                    raw_events = list_pull_source_events(track_coord)
                    if isinstance(raw_events, list):
                        raw_cues = list(raw_events)
                except Exception:
                    raw_cues = []

        return self._cue_options_from_raw_rows(raw_cues)

    @staticmethod
    def _cue_options_from_raw_rows(raw_rows: list[object]) -> list[MA3CueOption]:
        deduped: dict[str, MA3CueOption] = {}
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            cue_number = normalize_ma_scaled_cue_number(
                raw.get("cue_number")
                or raw.get("cue_no")
                or raw.get("no")
                or raw.get("cueNo")
                or raw.get("cue_ref")
                or raw.get("cueRef")
            )
            if cue_number is None:
                continue
            cue_no_text = cue_number_text(cue_number)
            if cue_no_text is None:
                continue
            name = str(
                raw.get("cue_name")
                or raw.get("cue_label")
                or raw.get("name")
                or raw.get("label")
                or ""
            ).strip()
            deduped[cue_no_text] = MA3CueOption(cue_number=cue_number, name=name)
        return sorted(
            deduped.values(),
            key=lambda option: float(option.cue_number),
        )

    @staticmethod
    def _manual_push_sequence_label(sequence: ManualPushSequenceOptionPresentation) -> str:
        return f"{sequence.number} - {sequence.name}"

    @staticmethod
    def _manual_push_sequence_prep_prompt(
        *,
        target_track: ManualPushTrackOptionPresentation,
        flow: ManualPushFlowPresentation,
    ) -> str:
        return (
            f"{target_track.name} ({target_track.coord}) has no assigned MA3 sequence.\n\n"
            "Choose how to prepare this target before routing or sending.\n"
            f"{TimelineWidgetMA3PushActionMixin._manual_push_sequence_range_summary(flow)}"
        )

    @staticmethod
    def _manual_push_existing_sequence_prompt(
        *,
        target_track: ManualPushTrackOptionPresentation,
        flow: ManualPushFlowPresentation,
    ) -> str:
        return (
            f"Assign an existing MA3 sequence to {target_track.name} ({target_track.coord}).\n\n"
            f"{TimelineWidgetMA3PushActionMixin._manual_push_sequence_range_summary(flow)}"
        )

    @staticmethod
    def _manual_push_sequence_range_summary(flow: ManualPushFlowPresentation) -> str:
        sequence_range = flow.current_song_sequence_range
        if sequence_range is None:
            return "Current song range: unavailable"
        if sequence_range.song_label:
            return (
                f"Current song range: {sequence_range.song_label} "
                f"({sequence_range.start}-{sequence_range.end})"
            )
        return f"Current song range: {sequence_range.start}-{sequence_range.end}"

    @staticmethod
    def _default_ma3_sequence_name(
        *,
        presentation: TimelinePresentation,
        layer: LayerPresentation,
    ) -> str | None:
        song_title = (presentation.active_song_title or "").strip()
        layer_title = (layer.title or "").strip()
        if song_title and layer_title:
            return f"{song_title} - {layer_title}"
        if song_title:
            return song_title
        if layer_title:
            return layer_title
        return None

    @staticmethod
    def _preferred_ma3_sequence_name_for_track(
        *,
        track: ManualPushTrackOptionPresentation | None,
        presentation: TimelinePresentation,
        layer: LayerPresentation | None,
    ) -> str | None:
        if track is not None:
            track_name = str(track.name or "").strip()
            if track_name:
                return track_name
        if layer is None:
            return None
        return TimelineWidgetMA3PushActionMixin._default_ma3_sequence_name(
            presentation=presentation,
            layer=layer,
        )

    def _prompt_ma3_track_name(
        self,
        *,
        title: str,
        layer: LayerPresentation | None,
    ) -> tuple[bool, str | None]:
        host = cast(_TransferActionHost, self)
        default_name = str(layer.title or "").strip() if layer is not None else ""
        entered_name, accepted = host._input_dialog.getText(
            host._widget,
            title,
            "Track name (optional)",
            text=default_name,
        )
        if not accepted:
            return False, None
        resolved = str(entered_name or "").strip()
        return True, (resolved or None)
