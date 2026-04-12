"""Stage Zero timeline shell composed from reusable blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QContextMenuEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox, QPushButton, QScrollArea, QScrollBar, QSplitter, QToolTip, QVBoxLayout, QWidget

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContract,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
    render_inspector_contract_text,
)
from echozero.application.presentation.models import TimelinePresentation, LayerPresentation, TakeLanePresentation
from echozero.application.shared.enums import FollowMode
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPreset,
    ApplyTransferPlan,
    CancelTransferPlan,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    DeleteTransferPreset,
    DuplicateSelectedEvents,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    PreviewTransferPlan,
    SaveTransferPreset,
    Seek,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPushTransferMode,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTracks,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    SelectTake,
    Stop,
    ToggleMute,
    ToggleSolo,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    GRID_LINE_ALPHA,
    GRID_LINE_COLOR,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    RULER_HEIGHT_PX,
    TAKE_ROW_HEIGHT_PX,
    TIMELINE_ZOOM_MAX_PPS,
    TIMELINE_ZOOM_MIN_PPS,
    TIMELINE_ZOOM_STEP_FACTOR,
    TIMELINE_RIGHT_PADDING_PX,
)
from echozero.ui.style.tokens import SHELL_TOKENS
from echozero.ui.style.qt.qss import build_object_info_panel_qss
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout, TakeRowLayout
from echozero.ui.qt.timeline.blocks.ruler import (
    RulerBlock,
    RulerLayout,
    playhead_head_polygon,
    seek_time_for_x,
    timeline_x_for_time,
    visible_ruler_seconds,
)
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.style import (
    TIMELINE_STYLE,
    TimelineShellStyle,
    build_object_palette_stylesheet,
    build_timeline_scroll_area_stylesheet,
)
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock, WaveformLanePresentation


_SPAN_CACHE: dict[tuple, float] = {}
_MAX_SPAN_CACHE_ENTRIES = 24


def _span_signature(presentation: TimelinePresentation) -> tuple:
    layer_sig: list[tuple[int, tuple[int, ...]]] = []
    for layer in presentation.layers:
        take_sig = tuple(id(take.events) for take in layer.takes)
        layer_sig.append((id(layer.events), take_sig))
    return (id(presentation.layers), tuple(layer_sig), presentation.end_time_label)


def estimate_timeline_span_seconds(presentation: TimelinePresentation) -> float:
    """Best-effort duration estimate for viewport/scroll math (memoized by lane identity)."""
    key = _span_signature(presentation)
    cached = _SPAN_CACHE.get(key)
    if cached is not None:
        return cached

    span = max(0.0, presentation.playhead)
    for layer in presentation.layers:
        for event in layer.events:
            span = max(span, event.end)
        for take in layer.takes:
            for event in take.events:
                span = max(span, event.end)

    span = max(span, _parse_time_label_seconds(presentation.end_time_label))
    resolved = max(0.0, span)

    _SPAN_CACHE[key] = resolved
    if len(_SPAN_CACHE) > _MAX_SPAN_CACHE_ENTRIES:
        oldest = next(iter(_SPAN_CACHE.keys()))
        _SPAN_CACHE.pop(oldest, None)
    return resolved


def compute_scroll_bounds(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    right_padding_px: int = TIMELINE_RIGHT_PADDING_PX,
) -> tuple[int, int]:
    """Return (content_width, max_scroll_x) for horizontal timeline navigation."""
    viewport = max(1, int(viewport_width))
    span = estimate_timeline_span_seconds(presentation)
    content_width = max(viewport, int(header_width + (span * presentation.pixels_per_second) + right_padding_px))
    max_scroll = max(0, content_width - viewport)
    return content_width, max_scroll


def compute_follow_scroll_x(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    content_padding_px: int = 24,
) -> float:
    """Compute follow-mode adjusted scroll target for the current playhead."""
    if presentation.follow_mode == FollowMode.OFF or not presentation.is_playing:
        return presentation.scroll_x

    viewport = max(1, int(viewport_width))
    content_width = max(1.0, viewport - header_width)
    pps = max(1.0, presentation.pixels_per_second)
    timeline_x = presentation.playhead * pps
    current = presentation.scroll_x
    left_bound = current + content_padding_px
    right_bound = current + max(content_padding_px + 1.0, content_width - content_padding_px)

    target = current
    if presentation.follow_mode == FollowMode.PAGE:
        if timeline_x < left_bound:
            target = max(0.0, timeline_x - content_padding_px)
        elif timeline_x > right_bound:
            target = max(0.0, timeline_x - content_padding_px)
    elif presentation.follow_mode == FollowMode.CENTER:
        target = max(0.0, timeline_x - (content_width * 0.5))
    elif presentation.follow_mode == FollowMode.SMOOTH:
        # Match EZ1 semantics: keep playhead around 75% of the viewport
        # rather than centered, reducing forward-jump feel.
        target = max(0.0, timeline_x - (content_width * 0.75))

    _, max_scroll = compute_scroll_bounds(presentation, viewport, header_width=header_width)
    return float(max(0.0, min(target, max_scroll)))


def badge_tooltip_labels(badges: list[str]) -> list[str]:
    mapping = {
        "main": "Main take",
        "stem": "Stem output",
        "audio": "Audio lane",
        "event": "Event lane",
        "classifier-preview": "Classifier preview",
        "real-data": "Real data",
    }
    labels: list[str] = []
    for badge in badges:
        key = str(badge).strip().lower()
        if not key:
            continue
        labels.append(mapping.get(key, key.replace("-", " ").title()))
    return labels


def _parse_time_label_seconds(label: str | None) -> float:
    if not label:
        return 0.0
    text = label.strip()
    if not text:
        return 0.0
    try:
        if ':' in text:
            mins_txt, secs_txt = text.split(':', 1)
            return max(0.0, int(mins_txt) * 60 + float(secs_txt))
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


@dataclass(slots=True, frozen=True)
class ManualPullTimelineSelectionResult:
    selected_event_ids: list[str]
    target_layer_id: object


class ManualPullTimelineCanvas(QWidget):
    selection_changed = pyqtSignal(object)

    def __init__(self, events, selected_event_ids: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._events = list(events)
        self._selected_event_ids = list(selected_event_ids or [])
        self._anchor_index: int | None = self._selected_index() if self._selected_event_ids else None
        self._rects: list[QRectF] = []
        self.setMinimumHeight(150)

    def selected_event_ids(self) -> list[str]:
        ordered_ids = [event.event_id for event in self._events if event.event_id in self._selected_event_ids]
        return ordered_ids

    def set_selected_event_ids(self, event_ids: list[str]) -> None:
        self._selected_event_ids = list(dict.fromkeys(event_ids))
        self._anchor_index = self._selected_index()
        self.selection_changed.emit(self.selected_event_ids())
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        index = self._index_at(event.position())
        if index is None:
            super().mousePressEvent(event)
            return

        modifiers = event.modifiers()
        has_toggle = bool(
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        event_id = self._events[index].event_id

        if has_shift and self._anchor_index is not None:
            start_index = min(self._anchor_index, index)
            end_index = max(self._anchor_index, index)
            self._selected_event_ids = [candidate.event_id for candidate in self._events[start_index : end_index + 1]]
        elif has_toggle:
            selected = list(self._selected_event_ids)
            if event_id in selected:
                selected.remove(event_id)
            else:
                selected.append(event_id)
            self._selected_event_ids = selected
            self._anchor_index = index
        else:
            self._selected_event_ids = [event_id]
            self._anchor_index = index

        self.selection_changed.emit(self.selected_event_ids())
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        background = QColor("#11161b")
        painter.fillRect(self.rect(), background)

        track_pen = QPen(QColor("#2b3642"))
        track_pen.setWidth(1)
        painter.setPen(track_pen)
        baseline_y = self.height() * 0.5
        painter.drawLine(16, int(baseline_y), max(16, self.width() - 16), int(baseline_y))

        self._rects = self._compute_event_rects()
        for index, (event_model, rect) in enumerate(zip(self._events, self._rects)):
            is_selected = event_model.event_id in self._selected_event_ids
            fill = QColor("#5cb2ff" if is_selected else "#475569")
            stroke = QColor("#d7ebff" if is_selected else "#90a2b5")
            painter.setPen(QPen(stroke, 1.5))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 6.0, 6.0)

            label_rect = QRectF(rect.left() + 8.0, rect.top(), max(0.0, rect.width() - 16.0), rect.height())
            painter.setPen(QColor("#08111a" if is_selected else "#eef4ff"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, event_model.label)

            if event_model.start is not None and event_model.end is not None:
                painter.setPen(QColor("#c9d6e2"))
                footer = f"{_format_seconds(event_model.start)}-{_format_seconds(event_model.end)}"
                painter.drawText(
                    QRectF(rect.left(), rect.bottom() + 4.0, rect.width(), 14.0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    footer,
                )

    def _selected_index(self) -> int | None:
        if not self._selected_event_ids:
            return None
        first_id = self._selected_event_ids[-1]
        for index, event in enumerate(self._events):
            if event.event_id == first_id:
                return index
        return None

    def _index_at(self, pos) -> int | None:
        for index, rect in enumerate(self._rects or self._compute_event_rects()):
            if rect.contains(pos):
                return index
        return None

    def _compute_event_rects(self) -> list[QRectF]:
        if not self._events:
            return []
        left_padding = 16.0
        right_padding = 16.0
        top_padding = 18.0
        lane_height = 34.0
        bar_height = 24.0
        content_width = max(120.0, self.width() - left_padding - right_padding)
        timed = [
            event for event in self._events
            if event.start is not None and event.end is not None
        ]
        rects: list[QRectF] = []
        if timed:
            min_start = min(event.start for event in timed)
            max_end = max(max(event.end, event.start + 0.25) for event in timed)
            span = max(0.25, max_end - min_start)
            for index, event in enumerate(self._events):
                lane_index = index % 3
                y = top_padding + (lane_index * lane_height)
                if event.start is None or event.end is None:
                    slot_width = content_width / max(1, len(self._events))
                    x = left_padding + (index * slot_width)
                    width = max(48.0, slot_width - 10.0)
                else:
                    start_ratio = (event.start - min_start) / span
                    end_ratio = (max(event.end, event.start + 0.25) - min_start) / span
                    x = left_padding + (start_ratio * content_width)
                    width = max(44.0, (end_ratio - start_ratio) * content_width)
                rects.append(QRectF(x, y, width, bar_height))
            return rects

        slot_width = content_width / max(1, len(self._events))
        for index, _event in enumerate(self._events):
            lane_index = index % 3
            x = left_padding + (index * slot_width) + 4.0
            y = top_padding + (lane_index * lane_height)
            rects.append(QRectF(x, y, max(48.0, slot_width - 8.0), bar_height))
        return rects


class ManualPullTimelineDialog(QDialog):
    def __init__(
        self,
        *,
        source_track_label: str,
        events,
        selected_event_ids: list[str] | None,
        available_targets,
        selected_target_layer_id,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import from MA3")
        self.resize(760, 320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        context_label = QLabel(f"Source track: {source_track_label}", self)
        layout.addWidget(context_label)

        help_label = QLabel(
            "Select source events on the timeline and choose the destination EZ layer below. "
            "Click to select. Ctrl/Cmd toggles. Shift selects a range.",
            self,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self._canvas = ManualPullTimelineCanvas(events, selected_event_ids=selected_event_ids, parent=self)
        layout.addWidget(self._canvas)

        self._selection_label = QLabel(self)
        layout.addWidget(self._selection_label)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target EZ layer", self))
        self._target_combo = QComboBox(self)
        for target in available_targets:
            self._target_combo.addItem(target.name, target.layer_id)
        if selected_target_layer_id is not None:
            for index in range(self._target_combo.count()):
                if self._target_combo.itemData(index) == selected_target_layer_id:
                    self._target_combo.setCurrentIndex(index)
                    break
        target_row.addWidget(self._target_combo, 1)
        layout.addLayout(target_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._canvas.selection_changed.connect(self._refresh_state)
        self._target_combo.currentIndexChanged.connect(self._refresh_state)
        self._refresh_state()

    def selected_event_ids(self) -> list[str]:
        return self._canvas.selected_event_ids()

    def selected_target_layer_id(self):
        return self._target_combo.currentData()

    def accept(self) -> None:
        if not self.selected_event_ids():
            QMessageBox.warning(self, "Import from MA3", "Select at least one source event.")
            return
        if self.selected_target_layer_id() is None:
            QMessageBox.warning(self, "Import from MA3", "Select a target EZ layer.")
            return
        super().accept()

    def _refresh_state(self, *_args) -> None:
        selected_count = len(self.selected_event_ids())
        noun = "event" if selected_count == 1 else "events"
        self._selection_label.setText(f"Selected: {selected_count} {noun}")
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(self.selected_event_ids()) and self.selected_target_layer_id() is not None)


class ObjectInfoPanel(QFrame):
    action_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        style = TIMELINE_STYLE.object_palette
        self.setObjectName(style.frame_object_name)
        self.setStyleSheet(build_object_info_panel_qss())

        self.setMinimumWidth(style.min_width_px)
        self.setMaximumWidth(style.max_width_px)

        self._selected_layer_id = None
        self._selected_event_start: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            style.content_padding.left,
            style.content_padding.top,
            style.content_padding.right,
            style.content_padding.bottom,
        )
        layout.setSpacing(style.section_spacing_px)

        title = QLabel("Object Palette", self)
        title.setObjectName(style.title_object_name)
        layout.addWidget(title)

        selection_section = QLabel("SELECTION", self)
        selection_section.setObjectName("timeline_object_info_section")
        layout.addWidget(selection_section)

        self._kind = QLabel("None", self)
        self._kind.setObjectName("timeline_object_info_kind")
        layout.addWidget(self._kind, 0, Qt.AlignmentFlag.AlignLeft)

        self._body = QLabel(self)
        self._body.setObjectName(style.body_object_name)
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setMinimumHeight(72)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._body)
        self._contract = InspectorContract(title="No timeline object selected.")

        event_section = QLabel("EVENT ACTIONS", self)
        event_section.setObjectName("timeline_object_info_section")
        layout.addWidget(event_section)

        event_actions = QGridLayout()
        event_actions.setHorizontalSpacing(6)
        event_actions.setVerticalSpacing(6)
        self._seek_btn = QPushButton("Seek", self)
        self._nudge_left_btn = QPushButton("Nudge -", self)
        self._nudge_right_btn = QPushButton("Nudge +", self)
        self._duplicate_btn = QPushButton("Duplicate", self)
        event_actions.addWidget(self._seek_btn, 0, 0)
        event_actions.addWidget(self._nudge_left_btn, 0, 1)
        event_actions.addWidget(self._nudge_right_btn, 1, 0)
        event_actions.addWidget(self._duplicate_btn, 1, 1)
        layout.addLayout(event_actions)

        layer_section = QLabel("LAYER ACTIONS", self)
        layer_section.setObjectName("timeline_object_info_section")
        layout.addWidget(layer_section)

        layer_actions = QGridLayout()
        layer_actions.setHorizontalSpacing(6)
        layer_actions.setVerticalSpacing(6)
        self._mute_btn = QPushButton("Mute", self)
        self._solo_btn = QPushButton("Solo", self)
        self._gain_spin = QDoubleSpinBox(self)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply Gain", self)
        layer_actions.addWidget(self._mute_btn, 0, 0)
        layer_actions.addWidget(self._solo_btn, 0, 1)
        layer_actions.addWidget(self._gain_spin, 1, 0)
        layer_actions.addWidget(self._gain_apply_btn, 1, 1)
        layout.addLayout(layer_actions)

        layout.addStretch(1)

        self._seek_btn.clicked.connect(self._emit_seek_selected_event)
        self._nudge_left_btn.clicked.connect(lambda: self._emit_contract_action("nudge_left"))
        self._nudge_right_btn.clicked.connect(lambda: self._emit_contract_action("nudge_right"))
        self._duplicate_btn.clicked.connect(lambda: self._emit_contract_action("duplicate"))
        self._mute_btn.clicked.connect(self._emit_toggle_mute)
        self._solo_btn.clicked.connect(self._emit_toggle_solo)
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._set_controls_enabled(has_layer=False, has_event=False)

    def _set_controls_enabled(self, *, has_layer: bool, has_event: bool) -> None:
        self._seek_btn.setEnabled(has_event)
        self._nudge_left_btn.setEnabled(has_event)
        self._nudge_right_btn.setEnabled(has_event)
        self._duplicate_btn.setEnabled(has_event)

        self._mute_btn.setEnabled(has_layer)
        self._solo_btn.setEnabled(has_layer)
        self._gain_spin.setEnabled(has_layer)
        self._gain_apply_btn.setEnabled(has_layer)

    def set_context(self, presentation: TimelinePresentation, text: str) -> None:
        self._body.setText(text)

    def _iter_contract_actions(self):
        for section in self._contract.context_sections:
            for action in section.actions:
                yield action

    def _find_contract_action(self, action_id: str) -> InspectorAction | None:
        return next((action for action in self._iter_contract_actions() if action.action_id == action_id), None)

    def _emit_contract_action(self, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_seek_selected_event(self) -> None:
        self._emit_contract_action("seek_here")

    def _emit_toggle_mute(self) -> None:
        self._emit_contract_action("toggle_mute")

    def _emit_toggle_solo(self) -> None:
        self._emit_contract_action("toggle_solo")

    def _emit_apply_gain(self) -> None:
        layer_action = self._find_contract_action("toggle_mute")
        layer_id = layer_action.params.get("layer_id") if layer_action is not None else None
        if layer_id is None:
            return
        self.action_requested.emit(
            InspectorAction(
                action_id="set_gain_custom",
                label="Set Gain",
                group="gain",
                params={"layer_id": layer_id, "gain_db": float(self._gain_spin.value())},
            )
        )

    def set_contract(self, contract: InspectorContract) -> None:
        self._contract = contract
        self._body.setText(render_inspector_contract_text(contract))

        object_type = contract.identity.object_type if contract.identity is not None else "none"
        self._kind.setText(object_type.capitalize())

        has_event = self._find_contract_action("seek_here") is not None
        has_layer = self._find_contract_action("toggle_mute") is not None
        self._set_controls_enabled(has_layer=has_layer, has_event=has_event)

        mute_action = self._find_contract_action("toggle_mute")
        solo_action = self._find_contract_action("toggle_solo")
        self._mute_btn.setText(mute_action.label if mute_action is not None else "Mute")
        self._solo_btn.setText(solo_action.label if solo_action is not None else "Solo")

    def contract(self) -> InspectorContract:
        return self._contract

    def text(self) -> str:
        return self._body.text()


@dataclass(slots=True)
class _ContextMenuEntry:
    label: str
    enabled: bool
    callback: Callable[[], None] | None = None


class TimelineCanvas(QWidget):
    layer_clicked = pyqtSignal(object)
    mute_clicked = pyqtSignal(object)
    solo_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object, object, str)
    move_selected_events_requested = pyqtSignal(float, object)
    take_action_selected = pyqtSignal(object, object, str)
    contract_action_selected = pyqtSignal(object)
    horizontal_scroll_requested = pyqtSignal(int)
    zoom_requested = pyqtSignal(int, float)
    playhead_drag_requested = pyqtSignal(float)
    clear_selection_requested = pyqtSignal()
    select_all_requested = pyqtSignal()
    nudge_requested = pyqtSignal(int, int)
    duplicate_requested = pyqtSignal(int)
    preview_transfer_plan_requested = pyqtSignal()
    apply_transfer_plan_requested = pyqtSignal()
    cancel_transfer_plan_requested = pyqtSignal()

    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._style = TIMELINE_STYLE
        self._header_width = LAYER_HEADER_WIDTH_PX
        self._top_padding = LAYER_HEADER_TOP_PADDING_PX
        self._main_row_height = LAYER_ROW_HEIGHT_PX
        self._take_row_height = TAKE_ROW_HEIGHT_PX
        self._event_height = EVENT_BAR_HEIGHT_PX
        self._take_rects: list[tuple[QRectF, object, object]] = []
        self._take_option_rects: list[tuple[QRectF, object, object]] = []
        self._take_action_rects: list[tuple[QRectF, object, object, str]] = []
        self._open_take_options: set[tuple[object, object]] = set()
        self._toggle_rects: list[tuple[QRectF, object]] = []
        self._mute_rects: list[tuple[QRectF, object]] = []
        self._solo_rects: list[tuple[QRectF, object]] = []
        self._event_rects: list[tuple[QRectF, object, object | None, object]] = []
        self._header_select_rects: list[tuple[QRectF, object]] = []
        self._row_body_select_rects: list[tuple[QRectF, object]] = []
        self._header_hover_rects: list[tuple[QRectF, LayerPresentation]] = []
        self._event_drop_rects: list[tuple[QRectF, object]] = []
        self._hovered_layer_id: object | None = None
        self._dragging_playhead = False
        self._drag_candidate: dict[str, object] | None = None
        self._dragging_events = False
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(1440)
        self.setMouseTracking(True)
        self._recompute_height()

    def _recompute_height(self) -> None:
        height = self._top_padding
        for layer in self.presentation.layers:
            height += self._main_row_height
            if layer.is_expanded:
                height += len(layer.takes) * self._take_row_height
        self.setMinimumHeight(max(320, height + 12))

    def _any_solo(self) -> bool:
        return any(layer.soloed for layer in self.presentation.layers)

    def _layer_dimmed(self, layer: LayerPresentation) -> bool:
        if layer.muted:
            return True
        if self._any_solo() and not layer.soloed:
            return True
        return False

    def _push_outline_active_for_layer(self, layer: LayerPresentation) -> bool:
        if not self.presentation.manual_push_flow.push_mode_active:
            return False
        plan = self.presentation.batch_transfer_plan
        if plan is None or plan.operation_type not in {"push", "mixed"}:
            return False
        return any(
            row.direction == "push" and row.source_layer_id == layer.layer_id
            for row in plan.rows
        )

    def set_presentation(self, presentation: TimelinePresentation, *, recompute_layout: bool = True) -> None:
        self.presentation = presentation
        if recompute_layout:
            self._recompute_height()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self._style.canvas.background_hex))
        self._take_rects.clear()
        self._take_option_rects.clear()
        self._take_action_rects.clear()
        self._toggle_rects.clear()
        self._mute_rects.clear()
        self._solo_rects.clear()
        self._event_rects.clear()
        self._header_select_rects.clear()
        self._row_body_select_rects.clear()
        self._header_hover_rects.clear()
        self._event_drop_rects.clear()
        with timed("timeline.paint.layers"):
            self._draw_layers(painter)
        with timed("timeline.paint.playhead"):
            self._draw_playhead(painter)

    def _draw_time_grid_band(self, painter: QPainter, *, top: int, row_height: int) -> None:
        content_left = float(self._header_width)
        content_width = max(1.0, float(self.width()) - content_left)
        marks = visible_ruler_seconds(
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_width=content_width,
            content_start_x=content_left,
        )

        grid_color = QColor(GRID_LINE_COLOR)
        grid_color.setAlpha(max(0, min(255, GRID_LINE_ALPHA)))
        painter.setPen(QPen(grid_color, 1))

        band_top = int(top)
        band_bottom = int(top + max(1, row_height) - 1)
        for _, x in marks:
            if x < content_left:
                continue
            painter.drawLine(int(x), band_top, int(x), band_bottom)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_playhead and event.buttons() & Qt.MouseButton.LeftButton:
            self.playhead_drag_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return

        if self._drag_candidate is not None and event.buttons() & Qt.MouseButton.LeftButton:
            dx = abs(event.position().x() - float(self._drag_candidate["anchor_x"]))
            dy = abs(event.position().y() - float(self._drag_candidate["anchor_y"]))
            if max(dx, dy) >= 4.0:
                self._dragging_events = True
                event.accept()
                return

        pos = event.position()
        hovered: LayerPresentation | None = None
        hovered_rect: QRectF | None = None
        for rect, layer in self._header_hover_rects:
            if rect.contains(pos):
                hovered = layer
                hovered_rect = rect
                break

        next_id = hovered.layer_id if hovered is not None else None
        if next_id != self._hovered_layer_id:
            self._hovered_layer_id = next_id
            if hovered is not None and hovered_rect is not None:
                tip = self._header_tooltip_text(hovered)
                if tip:
                    QToolTip.showText(
                        self.mapToGlobal(pos.toPoint()),
                        tip,
                        self,
                        hovered_rect.toRect(),
                        6000,
                    )
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered_layer_id = None
        self._dragging_playhead = False
        self._drag_candidate = None
        self._dragging_events = False
        QToolTip.hideText()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()

        if event.button() == Qt.MouseButton.RightButton:
            if self._show_context_menu(pos):
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self._playhead_head_contains(pos):
            self._dragging_playhead = True
            self.playhead_drag_requested.emit(self._seek_time_at_x(pos.x()))
            event.accept()
            return
        for rect, layer_id in self._mute_rects:
            if rect.contains(pos):
                self.mute_clicked.emit(layer_id)
                return
        for rect, layer_id in self._solo_rects:
            if rect.contains(pos):
                self.solo_clicked.emit(layer_id)
                return
        for rect, layer_id, take_id, action_id in self._take_action_rects:
            if rect.contains(pos):
                self.take_action_selected.emit(layer_id, take_id, action_id)
                return
        for rect, layer_id, take_id in self._take_option_rects:
            if rect.contains(pos):
                key = (layer_id, take_id)
                if key in self._open_take_options:
                    self._open_take_options.remove(key)
                else:
                    self._open_take_options.add(key)
                self.update()
                return
        for rect, layer_id, take_id in self._take_rects:
            if rect.contains(pos):
                self.take_selected.emit(layer_id, take_id)
                return
        for rect, layer_id, take_id, event_id in self._event_rects:
            if rect.contains(pos):
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._can_start_event_drag(event.modifiers(), event_id)
                ):
                    self._drag_candidate = {
                        "anchor_x": pos.x(),
                        "anchor_y": pos.y(),
                        "source_layer_id": layer_id,
                    }
                    self._dragging_events = False
                    event.accept()
                    return
                self.event_selected.emit(layer_id, take_id, event_id, self._selection_mode_for_modifiers(event.modifiers()))
                return
        for rect, layer_id in self._toggle_rects:
            if rect.contains(pos):
                self.take_toggle_clicked.emit(layer_id)
                return
        for rect, layer_id in self._header_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(layer_id)
                return
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(layer_id)
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        hit_target = self._hit_target_for_position(event.position())
        contract = build_timeline_inspector_contract(self.presentation, hit_target=hit_target)
        menu = self._build_context_menu(contract)
        if menu.isEmpty():
            event.ignore()
            return
        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return
        payload = chosen.data()
        if isinstance(payload, InspectorAction):
            self.contract_action_selected.emit(payload)
            event.accept()

    def _build_context_menu(self, contract: InspectorContract) -> QMenu:
        menu = QMenu(self)
        first_section = True
        for section in contract.context_sections:
            if not first_section:
                menu.addSeparator()
            first_section = False
            for action in section.actions:
                qt_action = menu.addAction(action.label)
                qt_action.setEnabled(action.enabled)
                qt_action.setData(action)
        return menu

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
            if self._drag_candidate is not None:
                if self._dragging_events:
                    delta_seconds = (event.position().x() - float(self._drag_candidate["anchor_x"])) / max(1.0, self.presentation.pixels_per_second)
                    target_layer_id = self._event_drop_target_layer_id(event.position())
                    source_layer_id = self._drag_candidate["source_layer_id"]
                    if target_layer_id == source_layer_id:
                        target_layer_id = None
                    if abs(delta_seconds) >= 0.0001 or target_layer_id is not None:
                        self.move_selected_events_requested.emit(float(delta_seconds), target_layer_id)
                    event.accept()
                self._drag_candidate = None
                self._dragging_events = False
                if event.isAccepted():
                    return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta:
                self.zoom_requested.emit(int(delta), float(event.position().x()))
                event.accept()
                return

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta:
                self.horizontal_scroll_requested.emit(-delta)
                event.accept()
                return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        modifiers = event.modifiers()
        has_primary = bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        steps = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() == Qt.Key.Key_P:
            self.preview_transfer_plan_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.apply_transfer_plan_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() == Qt.Key.Key_Backspace:
            self.cancel_transfer_plan_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_A and has_primary:
            self.select_all_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_D and has_primary:
            self.duplicate_requested.emit(steps)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self.nudge_requested.emit(-1, steps)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.nudge_requested.emit(1, steps)
            event.accept()
            return

        if has_primary and event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_requested.emit(120, float(self.width() * 0.5))
            event.accept()
            return
        if has_primary and event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_requested.emit(-120, float(self.width() * 0.5))
            event.accept()
            return

        super().keyPressEvent(event)

    @staticmethod
    def _selection_mode_for_modifiers(modifiers: Qt.KeyboardModifier) -> str:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "additive"
        if modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier:
            return "toggle"
        return "replace"

    def _draw_layers(self, painter: QPainter) -> None:
        y = self._top_padding
        for layer in self.presentation.layers:
            self._draw_main_row(painter, layer, y)
            y += self._main_row_height
            if layer.is_expanded:
                for take in layer.takes:
                    self._draw_take_row(painter, layer, take, y)
                    y += self._take_row_height
    @staticmethod
    def _header_tooltip_text(layer: LayerPresentation) -> str:
        labels = badge_tooltip_labels(layer.badges)
        parts: list[str] = []
        if labels:
            parts.append(" | ".join(labels))
        if layer.status.stale:
            stale_text = "Status: Stale"
            stale_reason = getattr(layer.status, "stale_reason", "")
            if stale_reason:
                stale_text = f"{stale_text} ({stale_reason})"
            parts.append(stale_text)
        if layer.status.manually_modified:
            parts.append("Status: Manually modified")
        if layer.status.source_label:
            parts.append(layer.status.source_label)
        source_layer_id = getattr(layer.status, "source_layer_id", "")
        if source_layer_id:
            parts.append(f"Source layer: {source_layer_id}")
        source_song_version_id = getattr(layer.status, "source_song_version_id", "")
        if source_song_version_id:
            parts.append(f"Source song version: {source_song_version_id}")
        pipeline_id = getattr(layer.status, "pipeline_id", "")
        if pipeline_id:
            parts.append(f"Pipeline: {pipeline_id}")
        output_name = getattr(layer.status, "output_name", "")
        if output_name:
            parts.append(f"Output: {output_name}")
        source_run_id = getattr(layer.status, "source_run_id", "")
        if source_run_id:
            parts.append(f"Run: {source_run_id}")
        if layer.status.sync_label and layer.status.sync_label.lower() != "no sync":
            parts.append(f"Sync: {layer.status.sync_label}")
        return "\n".join(parts)

    def _draw_main_row(self, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = MainRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._main_row_height)
        row_bg = QColor(self._style.canvas.selected_row_fill_hex if layer.is_selected else self._style.canvas.row_fill_hex)
        if dimmed:
            row_bg = QColor(self._style.canvas.dimmed_row_fill_hex)
        painter.fillRect(layout.row_rect, row_bg)
        if self._push_outline_active_for_layer(layer):
            outline_rect = layout.row_rect.adjusted(1.0, 1.0, -1.0, -1.0)
            outline_color = QColor("#8fd0ff")
            painter.save()
            painter.setPen(QPen(outline_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outline_rect, 8.0, 8.0)
            painter.restore()
        self._draw_time_grid_band(painter, top=top, row_height=self._main_row_height)
        painter.fillRect(0, top + self._main_row_height - 1, self.width(), 1, QColor(self._style.canvas.row_divider_hex))

        slots = HeaderSlots(
            rect=layout.header_rect,
            title_rect=layout.title_rect,
            subtitle_rect=layout.subtitle_rect,
            status_rect=layout.status_rect,
            controls_rect=layout.controls_rect,
            toggle_rect=layout.toggle_rect,
            metadata_rect=layout.metadata_rect,
        )
        if layer.takes:
            self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_select_rects.append((layout.header_rect, layer.layer_id))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        self._header_hover_rects.append((layout.header_rect, layer))
        if layer.kind.name == 'EVENT':
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
        hit_targets = self._header_block.paint(painter, slots, layer, dimmed=dimmed)
        self._mute_rects.append((hit_targets.mute_rect, layer.layer_id))
        self._solo_rects.append((hit_targets.solo_rect, layer.layer_id))

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if layer.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                        row_height=self._main_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=dimmed,
                        waveform_key=layer.waveform_key,
                    ),
                )
            else:
                self._event_rects.extend(
                    self._event_lane_block.paint(
                        painter,
                        top + 24,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            take_id=layer.main_take_id,
                            events=layer.events,
                            default_fill_hex=layer.color,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=dimmed,
                            viewport_width=self.width(),
                        ),
                    )
                )

            if not layer.takes:
                hint_color = self._style.canvas.no_takes_hint_dimmed_hex if dimmed else self._style.canvas.no_takes_hint_hex
                painter.setPen(QColor(hint_color))
                painter.drawText(
                    layout.content_rect.adjusted(10, 0, -10, 0),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    "No takes yet",
                )
        finally:
            painter.restore()

    def _is_take_options_open(self, layer_id: object, take_id: object) -> bool:
        return (layer_id, take_id) in self._open_take_options

    def _draw_take_row(self, painter: QPainter, layer: LayerPresentation, take: TakeLanePresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = TakeRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._take_row_height)
        options_open = self._is_take_options_open(layer.layer_id, take.take_id)
        hit_targets = self._take_row_block.paint_header(
            painter,
            layout,
            layer,
            take,
            options_open=options_open,
            dimmed=dimmed,
        )
        self._draw_time_grid_band(painter, top=top, row_height=self._take_row_height)
        self._take_rects.append(hit_targets.take_rect)
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        if take.kind.name == 'EVENT':
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
        if hit_targets.options_toggle_rect is not None:
            self._take_option_rects.append(hit_targets.options_toggle_rect)
        self._take_action_rects.extend(hit_targets.action_rects)

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if take.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                        row_height=self._take_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=True or dimmed,
                        waveform_key=take.waveform_key,
                    ),
                )
            else:
                self._event_rects.extend(
                    self._event_lane_block.paint(
                        painter,
                        top + 10,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            take_id=take.take_id,
                            events=take.events,
                            default_fill_hex=layer.color,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=True or dimmed,
                            viewport_width=self.width(),
                        ),
                    )
                )
        finally:
            painter.restore()

    def _draw_playhead(self, painter: QPainter) -> None:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        # Hide playhead when it is outside timeline content band.
        if x < self._header_width or x > self.width():
            return

        painter.setPen(QPen(QColor(self._style.playhead.color_hex), self._style.playhead.line_width_px))
        painter.drawLine(int(x), 0, int(x), self.height())
        painter.setBrush(QColor(self._style.playhead.color_hex))
        painter.setPen(QPen(QColor(self._style.playhead.color_hex), self._style.playhead.head_outline_width_px))
        painter.drawPolygon(playhead_head_polygon(x, float(self._top_padding)))

    def _playhead_head_contains(self, pos: QPointF) -> bool:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        return playhead_head_polygon(x, float(self._top_padding)).boundingRect().contains(pos)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )

    def _can_start_event_drag(self, modifiers: Qt.KeyboardModifier, event_id: object) -> bool:
        if modifiers & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        return event_id in set(self.presentation.selected_event_ids)

    def _event_drop_target_layer_id(self, pos: QPointF):
        for rect, layer_id in self._event_drop_rects:
            if rect.contains(pos):
                return layer_id
        return None

    def _hit_target_for_position(self, pos: QPointF) -> TimelineInspectorHitTarget:
        for rect, layer_id, take_id, event_id in self._event_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="event",
                    layer_id=layer_id,
                    take_id=take_id,
                    event_id=event_id,
                    time_seconds=self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None,
                )
        for rect, layer_id, take_id in self._take_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="take",
                    layer_id=layer_id,
                    take_id=take_id,
                    time_seconds=self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None,
                )
        for rect, layer_id in self._header_select_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(kind="layer", layer_id=layer_id)
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="layer",
                    layer_id=layer_id,
                    time_seconds=self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None,
                )
        return TimelineInspectorHitTarget(
            kind="timeline",
            time_seconds=self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None,
        )


class TransportBar(QWidget):
    def __init__(self, presentation: TimelinePresentation, on_intent: Callable[[object], object | None] | None = None, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._block = TransportBarBlock()
        self._control_rects: dict[str, QRectF] = {}
        self.setMinimumHeight(44)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._control_rects = self._block.paint(
            painter,
            TransportLayout.create(width=self.width(), height=self.height()),
            self.presentation,
        )

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if (play_rect := self._control_rects.get('play')) is not None and play_rect.contains(pos):
            self._dispatch(Pause() if self.presentation.is_playing else Play())
            return
        if (stop_rect := self._control_rects.get('stop')) is not None and stop_rect.contains(pos):
            self._dispatch(Stop())
            return
        super().mousePressEvent(event)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)


class TimelineRuler(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, presentation: TimelinePresentation, *, header_width: float = float(LAYER_HEADER_WIDTH_PX), parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = header_width
        self._block = RulerBlock()
        self._dragging = False
        self.setMinimumHeight(RULER_HEIGHT_PX)
        self.setMaximumHeight(RULER_HEIGHT_PX)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._block.paint(
            painter,
            RulerLayout(QRectF(0, 0, self.width(), self.height()), self._header_width),
            self.presentation,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= self._header_width:
            self._dragging = True
            self.seek_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )


class TimelineWidget(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[object], TimelinePresentation] | None = None,
        *,
        runtime_audio: TimelineRuntimeAudioController | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._style: TimelineShellStyle = TIMELINE_STYLE
        self.presentation = presentation
        self._on_intent = on_intent
        self._runtime_audio = runtime_audio
        self._runtime_source_signature: tuple[tuple[str, str], ...] | None = None
        self._runtime_playhead_floor: float | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle(self._style.window_title)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        left_pane = QWidget(self)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._transport = TransportBar(self.presentation, on_intent=self._dispatch)
        left_layout.addWidget(self._transport)

        self._canvas = TimelineCanvas(self.presentation)
        self._ruler = TimelineRuler(self.presentation, header_width=self._canvas._header_width)
        left_layout.addWidget(self._ruler)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"background: {SHELL_TOKENS.canvas_bg}; border: none;")
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.mute_clicked.connect(self._toggle_mute)
        self._canvas.solo_clicked.connect(self._toggle_solo)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.event_selected.connect(self._select_event)
        self._canvas.move_selected_events_requested.connect(self._move_selected_events)
        self._canvas.take_action_selected.connect(self._trigger_take_action)
        self._canvas.contract_action_selected.connect(self._trigger_contract_action)
        self._canvas.playhead_drag_requested.connect(self._seek)
        self._canvas.horizontal_scroll_requested.connect(self._scroll_horizontally_by_steps)
        self._canvas.zoom_requested.connect(self._zoom_from_input)
        self._canvas.clear_selection_requested.connect(self._clear_selection)
        self._canvas.select_all_requested.connect(self._select_all_events)
        self._canvas.nudge_requested.connect(self._nudge_selected_events)
        self._canvas.duplicate_requested.connect(self._duplicate_selected_events)
        self._canvas.preview_transfer_plan_requested.connect(self._preview_active_transfer_plan)
        self._canvas.apply_transfer_plan_requested.connect(self._apply_active_transfer_plan)
        self._canvas.cancel_transfer_plan_requested.connect(self._cancel_active_transfer_plan)
        self._ruler.seek_requested.connect(self._seek)
        self._scroll.setWidget(self._canvas)
        self.setFocusProxy(self._canvas)
        left_layout.addWidget(self._scroll)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.setSingleStep(24)
        self._hscroll.setPageStep(200)
        self._hscroll.valueChanged.connect(self._on_horizontal_scroll_changed)
        left_layout.addWidget(self._hscroll)

        self._object_info = ObjectInfoPanel(self)
        self._object_info.action_requested.connect(self._trigger_contract_action)
        self._object_info_panel = self._object_info
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(left_pane)
        self._main_splitter.addWidget(self._object_info)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setSizes([1080, 320])
        root_layout.addWidget(self._main_splitter)

        self._runtime_timer = QTimer(self)
        self._runtime_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._runtime_timer.setInterval(8)
        self._runtime_timer.timeout.connect(self._on_runtime_tick)
        self._runtime_timer.start()

        self.set_presentation(self.presentation)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        viewport = max(1, self._scroll.viewport().width())
        followed = compute_follow_scroll_x(
            presentation,
            viewport,
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._object_info.set_contract(build_timeline_inspector_contract(self.presentation))
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation)
        if self._runtime_audio is not None:
            runtime_signature = tuple(
                (str(layer.layer_id), layer.source_audio_path or "")
                for layer in self.presentation.layers
                if layer.source_audio_path
            )
            if runtime_signature != self._runtime_source_signature:
                self._runtime_audio.build_for_presentation(self.presentation)
                self._runtime_source_signature = runtime_signature
            else:
                self._runtime_audio.apply_mix_state(self.presentation)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_horizontal_scroll_bounds(sync_bar_value=False)

    def _update_horizontal_scroll_bounds(self, *, sync_bar_value: bool) -> None:
        viewport = max(1, self._scroll.viewport().width())
        _, max_scroll = compute_scroll_bounds(self.presentation, viewport)

        current = int(round(self.presentation.scroll_x))
        clamped = max(0, min(current, max_scroll))

        self._hscroll.blockSignals(True)
        self._hscroll.setRange(0, max_scroll)
        self._hscroll.setPageStep(viewport)
        if sync_bar_value or self._hscroll.value() != clamped:
            self._hscroll.setValue(clamped)
        self._hscroll.blockSignals(False)

        if clamped != current:
            self.presentation = replace(self.presentation, scroll_x=float(clamped))

    def _on_horizontal_scroll_changed(self, value: int) -> None:
        next_scroll = float(max(0, value))
        if abs(next_scroll - self.presentation.scroll_x) < 0.5:
            return
        self.presentation = replace(self.presentation, scroll_x=next_scroll)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _scroll_horizontally_by_steps(self, delta: int) -> None:
        if delta == 0:
            return
        notches = max(-6, min(6, int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)))
        next_value = self._hscroll.value() + (notches * self._hscroll.singleStep())
        self._hscroll.setValue(max(self._hscroll.minimum(), min(self._hscroll.maximum(), next_value)))

    def _zoom_from_input(self, delta: int, anchor_x: float) -> None:
        if delta == 0:
            return
        factor = TIMELINE_ZOOM_STEP_FACTOR if delta > 0 else (1.0 / TIMELINE_ZOOM_STEP_FACTOR)
        self._apply_zoom_factor(factor, anchor_x=anchor_x)

    def _apply_zoom_factor(self, factor: float, *, anchor_x: float) -> None:
        current_pps = max(1.0, float(self.presentation.pixels_per_second))
        target_pps = max(TIMELINE_ZOOM_MIN_PPS, min(TIMELINE_ZOOM_MAX_PPS, current_pps * float(factor)))
        if abs(target_pps - current_pps) < 0.001:
            return

        content_start_x = float(self._canvas._header_width)
        anchor_view_x = max(content_start_x, float(anchor_x))
        anchor_time = seek_time_for_x(
            anchor_view_x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=current_pps,
            content_start_x=content_start_x,
        )
        new_scroll = max(0.0, (anchor_time * target_pps) - (anchor_view_x - content_start_x))

        self.presentation = replace(
            self.presentation,
            pixels_per_second=target_pps,
            scroll_x=new_scroll,
        )
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._object_info.set_contract(build_timeline_inspector_contract(self.presentation))
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            # Viewport state (scroll/zoom) is owned by the widget, not the demo/app
            # transport intent responses. Preserve local viewport through dispatch.
            updated = replace(
                updated,
                scroll_x=self.presentation.scroll_x,
                scroll_y=self.presentation.scroll_y,
                pixels_per_second=self.presentation.pixels_per_second,
            )
            if self._runtime_audio is not None:
                runtime_time = self._runtime_audio.current_time_seconds()
                runtime_playing = self._runtime_audio.is_playing()
                if isinstance(intent, Seek):
                    runtime_time = max(0.0, float(intent.position))
                    self._runtime_playhead_floor = runtime_time if runtime_playing else None
                elif isinstance(intent, Stop):
                    runtime_time = 0.0
                    self._runtime_playhead_floor = None
                else:
                    runtime_time = self._stabilize_runtime_playhead(runtime_time, playing=runtime_playing)
                updated = replace(
                    updated,
                    playhead=runtime_time,
                    is_playing=runtime_playing,
                    current_time_label=_format_time_label(runtime_time),
                )
            self.set_presentation(updated)

    def _on_runtime_tick(self) -> None:
        if self._runtime_audio is None:
            return

        playing = self._runtime_audio.is_playing()
        current_time = self._stabilize_runtime_playhead(
            self._runtime_audio.current_time_seconds(),
            playing=playing,
        )
        current_label = _format_time_label(current_time)
        if (
            abs(current_time - self.presentation.playhead) < 0.001
            and playing == self.presentation.is_playing
            and current_label == self.presentation.current_time_label
        ):
            return

        next_presentation = replace(
            self.presentation,
            playhead=current_time,
            is_playing=playing,
            current_time_label=current_label,
        )
        followed = compute_follow_scroll_x(
            next_presentation,
            max(1, self._scroll.viewport().width()),
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(next_presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _stabilize_runtime_playhead(self, runtime_time: float, *, playing: bool) -> float:
        next_time = max(0.0, float(runtime_time))
        if not playing:
            self._runtime_playhead_floor = None
            return next_time

        if self._runtime_playhead_floor is None:
            self._runtime_playhead_floor = next_time
            return next_time

        # Guard against stale backward clock samples during churn.
        if next_time + 1e-6 < self._runtime_playhead_floor:
            return self._runtime_playhead_floor

        self._runtime_playhead_floor = next_time
        return next_time

    def _seek(self, position: float) -> None:
        self._dispatch(Seek(position))

    def _select_layer(self, layer_id) -> None:
        self._dispatch(SelectLayer(layer_id))

    def _toggle_take_selector(self, layer_id) -> None:
        self._dispatch(ToggleLayerExpanded(layer_id))

    def _select_take(self, layer_id, take_id) -> None:
        if take_id is None:
            return
        self._dispatch(SelectTake(layer_id, take_id))

    def _select_event(self, layer_id, take_id, event_id, mode: str) -> None:
        if event_id is None:
            return
        self._dispatch(SelectEvent(layer_id, take_id, event_id, mode=mode))

    def _clear_selection(self) -> None:
        self._dispatch(ClearSelection())

    def _select_all_events(self) -> None:
        self._dispatch(SelectAllEvents())

    def _nudge_selected_events(self, direction: int, steps: int) -> None:
        self._dispatch(NudgeSelectedEvents(direction=direction, steps=steps))

    def _duplicate_selected_events(self, steps: int) -> None:
        self._dispatch(DuplicateSelectedEvents(steps=steps))

    def _trigger_take_action(self, layer_id, take_id, action_id: str) -> None:
        if take_id is None or not action_id:
            return
        self._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    def _move_selected_events(self, delta_seconds: float, target_layer_id) -> None:
        self._dispatch(MoveSelectedEvents(delta_seconds=delta_seconds, target_layer_id=target_layer_id))

    def _toggle_mute(self, layer_id) -> None:
        self._dispatch(ToggleMute(layer_id))

    def _toggle_solo(self, layer_id) -> None:
        self._dispatch(ToggleSolo(layer_id))

    def _preview_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._trigger_contract_action(
            InspectorAction(
                action_id="preview_transfer_plan",
                label=_preview_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _apply_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._trigger_contract_action(
            InspectorAction(
                action_id="apply_transfer_plan",
                label=_apply_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _cancel_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._trigger_contract_action(
            InspectorAction(
                action_id="cancel_transfer_plan",
                label="Cancel Transfer Plan",
                params={"plan_id": plan.plan_id},
            )
        )

    def _trigger_contract_action(self, action: InspectorAction) -> None:
        params = action.params
        action_id = action.action_id
        if action_id == "seek_here":
            time_seconds = params.get("time_seconds")
            if isinstance(time_seconds, (int, float)):
                self._dispatch(Seek(float(time_seconds)))
            return
        if action_id == "nudge_left":
            self._dispatch(NudgeSelectedEvents(direction=-1, steps=int(params.get("steps", 1))))
            return
        if action_id == "nudge_right":
            self._dispatch(NudgeSelectedEvents(direction=1, steps=int(params.get("steps", 1))))
            return
        if action_id == "duplicate":
            self._dispatch(DuplicateSelectedEvents(steps=int(params.get("steps", 1))))
            return
        if action_id == "toggle_mute":
            layer_id = params.get("layer_id")
            if layer_id is not None:
                self._dispatch(ToggleMute(layer_id))
            return
        if action_id == "toggle_solo":
            layer_id = params.get("layer_id")
            if layer_id is not None:
                self._dispatch(ToggleSolo(layer_id))
            return
        if action_id in {"gain_down", "gain_unity", "gain_up", "set_gain_custom"}:
            layer_id = params.get("layer_id")
            gain_db = params.get("gain_db")
            if layer_id is not None and isinstance(gain_db, (int, float)):
                self._dispatch(SetGain(layer_id=layer_id, gain_db=float(gain_db)))
            return
        if action_id in {"live_sync_set_off", "live_sync_set_observe", "live_sync_set_armed_write"}:
            layer_id = params.get("layer_id")
            if layer_id is None:
                return
            if action_id == "live_sync_set_armed_write":
                reply = QMessageBox.question(
                    self,
                    "Arm Live Sync Write",
                    "Arm live sync write for this layer? MA3 changes may be written immediately.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                state = LiveSyncState.ARMED_WRITE
            elif action_id == "live_sync_set_observe":
                state = LiveSyncState.OBSERVE
            else:
                state = LiveSyncState.OFF
            self._dispatch(SetLayerLiveSyncState(layer_id=layer_id, live_sync_state=state))
            return
        if action_id == "live_sync_set_pause_reason":
            layer_id = params.get("layer_id")
            pause_reason = params.get("pause_reason")
            if layer_id is not None and isinstance(pause_reason, str) and pause_reason.strip():
                self._dispatch(
                    SetLayerLiveSyncPauseReason(
                        layer_id=layer_id,
                        pause_reason=pause_reason,
                    )
                )
            return
        if action_id == "live_sync_clear_pause_reason":
            layer_id = params.get("layer_id")
            if layer_id is not None:
                self._dispatch(ClearLayerLiveSyncPauseReason(layer_id=layer_id))
            return
        if action_id == "push_to_ma3":
            selected_event_ids = list(self.presentation.selected_event_ids)
            self._dispatch(OpenPushToMA3Dialog(selection_event_ids=selected_event_ids))
            return
        if action_id == "push_select_all_events":
            self._dispatch(SelectAllEvents())
            return
        if action_id == "push_unselect_all_events":
            self._dispatch(ClearSelection())
            return
        if action_id == "set_push_transfer_mode":
            current_mode = (self.presentation.manual_push_flow.transfer_mode or "merge").strip().lower()
            mode_labels = ["Merge", "Overwrite"]
            default_index = 0 if current_mode == "merge" else 1
            chosen_mode, accepted = QInputDialog.getItem(
                self,
                "Push Transfer Mode",
                "Transfer mode",
                mode_labels,
                default_index,
                False,
            )
            if not accepted:
                return
            selected_mode = chosen_mode.strip().lower()
            if selected_mode:
                self._dispatch(SetPushTransferMode(mode=selected_mode))
            return
        if action_id == "save_transfer_preset":
            preset_name, accepted = QInputDialog.getText(
                self,
                "Save Transfer Preset",
                "Preset name",
            )
            if not accepted or not preset_name.strip():
                return
            self._dispatch(SaveTransferPreset(name=preset_name))
            return
        if action_id in {"apply_transfer_preset", "delete_transfer_preset"}:
            if not self.presentation.transfer_presets:
                return
            labels = [self._transfer_preset_label(preset) for preset in self.presentation.transfer_presets]
            title = "Apply Transfer Preset" if action_id == "apply_transfer_preset" else "Delete Transfer Preset"
            chosen_label, accepted = QInputDialog.getItem(
                self,
                title,
                "Preset",
                labels,
                0,
                False,
            )
            if not accepted:
                return
            selected_preset = next(
                (preset for preset, label in zip(self.presentation.transfer_presets, labels) if label == chosen_label),
                None,
            )
            if selected_preset is None:
                return
            if action_id == "apply_transfer_preset":
                self._dispatch(ApplyTransferPreset(preset_id=selected_preset.preset_id))
            else:
                self._dispatch(DeleteTransferPreset(preset_id=selected_preset.preset_id))
            return
        if action_id == "preview_transfer_plan":
            plan_id = params.get("plan_id")
            if not isinstance(plan_id, str):
                return
            if not self._resolve_blocked_push_rows_for_plan_action(plan_id):
                return
            self._dispatch(PreviewTransferPlan(plan_id=plan_id))
            plan = self.presentation.batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                QMessageBox.information(
                    self,
                    "Transfer Plan Preview",
                    self._transfer_plan_preview_summary(
                        operation_type=plan.operation_type,
                        total_rows=len(plan.rows),
                        ready_count=plan.ready_count,
                        blocked_count=plan.blocked_count,
                        applied_count=plan.applied_count,
                        failed_count=plan.failed_count,
                    ),
                )
            return
        if action_id == "apply_transfer_plan":
            plan_id = params.get("plan_id")
            if not isinstance(plan_id, str):
                return
            if not self._resolve_blocked_push_rows_for_plan_action(plan_id):
                return
            self._dispatch(ApplyTransferPlan(plan_id=plan_id))
            plan = self.presentation.batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                QMessageBox.information(
                    self,
                    "Transfer Plan Results",
                    self._transfer_plan_apply_summary(
                        operation_type=plan.operation_type,
                        total_rows=len(plan.rows),
                        applied_count=plan.applied_count,
                        failed_count=plan.failed_count,
                        blocked_count=plan.blocked_count,
                    ),
                )
            return
        if action_id == "cancel_transfer_plan":
            plan_id = params.get("plan_id")
            if isinstance(plan_id, str):
                self._dispatch(CancelTransferPlan(plan_id=plan_id))
            return
        if action_id == "select_push_target_track":
            flow = self.presentation.manual_push_flow
            layer_id = params.get("layer_id")
            if layer_id is None or not flow.available_tracks:
                return
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
            chosen_label, accepted = QInputDialog.getItem(
                self,
                "Select Push Target Track",
                "Target track",
                labels,
                0,
                False,
            )
            if not accepted:
                return

            selected_track = next(
                (track for track, label in zip(flow.available_tracks, labels) if label == chosen_label),
                None,
            )
            if selected_track is None:
                return

            self._dispatch(SelectPushTargetTrack(target_track_coord=selected_track.coord, layer_id=layer_id))
            return
        if action_id == "preview_push_diff":
            layer_id = params.get("layer_id")
            if layer_id is None:
                return
            row = next(
                (
                    candidate
                    for candidate in (self.presentation.batch_transfer_plan.rows if self.presentation.batch_transfer_plan else [])
                    if candidate.direction == "push" and candidate.source_layer_id == layer_id
                ),
                None,
            )
            if row is None or not row.target_track_coord or not row.selected_event_ids:
                return
            self._dispatch(
                ConfirmPushToMA3(
                    target_track_coord=row.target_track_coord,
                    selected_event_ids=list(row.selected_event_ids),
                )
            )
            flow = self.presentation.manual_push_flow
            preview = flow.diff_preview
            if flow.diff_gate_open and preview is not None:
                QMessageBox.information(
                    self,
                    "Push Diff Preview",
                    self._manual_push_diff_preview_summary(preview.selected_count, preview.target_track_name, preview.target_track_coord),
                )
            return
        if action_id == "exit_push_mode":
            self._dispatch(ExitPushToMA3Mode())
            return
        if action_id == "pull_from_ma3":
            self._dispatch(OpenPullFromMA3Dialog())
            return
        if action_id == "select_pull_source_tracks":
            flow = self.presentation.manual_pull_flow
            if not flow.available_tracks:
                return

            track_labels = [self._manual_pull_track_label(track) for track in flow.available_tracks]
            chosen_track_label, accepted = QInputDialog.getItem(
                self,
                "Import from MA3",
                "Source track",
                track_labels,
                0,
                False,
            )
            if not accepted:
                return

            selected_track = next(
                (track for track, label in zip(flow.available_tracks, track_labels) if label == chosen_track_label),
                None,
            )
            if selected_track is None:
                return

            next_selected = list(flow.selected_source_track_coords)
            if selected_track.coord not in next_selected:
                next_selected.append(selected_track.coord)
            self._dispatch(SelectPullSourceTracks(source_track_coords=next_selected))
            self._dispatch(SelectPullSourceTrack(source_track_coord=selected_track.coord))
            return
        if action_id == "select_pull_source_events":
            flow = self.presentation.manual_pull_flow
            if not flow.active_source_track_coord or not flow.available_events or not flow.available_target_layers:
                return
            selection = self._open_manual_pull_timeline_popup(flow)
            if selection is None:
                return
            self._dispatch(SelectPullSourceEvents(selected_ma3_event_ids=selection.selected_event_ids))
            self._dispatch(SelectPullTargetLayer(target_layer_id=selection.target_layer_id))
            return
        if action_id == "set_pull_target_layer_mapping":
            flow = self.presentation.manual_pull_flow
            if not flow.available_target_layers:
                return

            target_labels = [self._manual_pull_target_label(target) for target in flow.available_target_layers]
            chosen_target_label, accepted = QInputDialog.getItem(
                self,
                "Import from MA3",
                "Destination EZ layer",
                target_labels,
                0,
                False,
            )
            if not accepted:
                return

            selected_target = next(
                (target for target, label in zip(flow.available_target_layers, target_labels) if label == chosen_target_label),
                None,
            )
            if selected_target is None:
                return

            self._dispatch(SelectPullTargetLayer(target_layer_id=selected_target.layer_id))
            return
        if action_id == "preview_pull_diff":
            layer_id = params.get("layer_id")
            if layer_id is None:
                return
            row = next(
                (
                    candidate
                    for candidate in (self.presentation.batch_transfer_plan.rows if self.presentation.batch_transfer_plan else [])
                    if candidate.direction == "pull" and candidate.target_layer_id == layer_id
                ),
                None,
            )
            if row is None or not row.source_track_coord or not row.target_layer_id or not row.selected_ma3_event_ids:
                return
            self._dispatch(SelectPullSourceTrack(source_track_coord=row.source_track_coord))
            self._dispatch(
                ConfirmPullFromMA3(
                    source_track_coord=row.source_track_coord,
                    selected_ma3_event_ids=list(row.selected_ma3_event_ids),
                    target_layer_id=row.target_layer_id,
                )
            )
            flow = self.presentation.manual_pull_flow
            preview = flow.diff_preview
            if flow.diff_gate_open and preview is not None:
                QMessageBox.information(
                    self,
                    "Pull Diff Preview",
                    self._manual_pull_diff_preview_summary(
                        preview.selected_count,
                        preview.source_track_name,
                        preview.source_track_coord,
                        preview.target_layer_name,
                    ),
                )
            return
        if action_id == "exit_pull_workspace":
            self._dispatch(ExitPullFromMA3Workspace())
            return
        if action_id:
            layer_id = params.get("layer_id")
            take_id = params.get("take_id")
            if layer_id is not None and take_id is not None:
                self._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    @staticmethod
    def _manual_push_track_label(track) -> str:
        parts = [track.name, f"({track.coord})"]
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} existing]")
        return " ".join(parts)

    def _resolve_blocked_push_rows_for_plan_action(self, plan_id: str) -> bool:
        plan = self.presentation.batch_transfer_plan
        if plan is None or plan.plan_id != plan_id or plan.operation_type not in {"push", "mixed"}:
            return True
        blocked_rows = [
            row
            for row in plan.rows
            if row.direction == "push" and not row.target_track_coord
        ]
        if not blocked_rows:
            return True
        flow = self.presentation.manual_push_flow
        if not flow.available_tracks:
            return False
        labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        for row in blocked_rows:
            chosen_label, accepted = QInputDialog.getItem(
                self,
                "Map Push Layer",
                f"Target MA3 track for {row.source_label}",
                labels,
                0,
                False,
            )
            if not accepted:
                return False
            selected_track = next(
                (track for track, label in zip(flow.available_tracks, labels) if label == chosen_label),
                None,
            )
            if selected_track is None:
                return False
            self._dispatch(
                SelectPushTargetTrack(
                    target_track_coord=selected_track.coord,
                    layer_id=row.source_layer_id,
                )
            )
            flow = self.presentation.manual_push_flow
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        return True

    @staticmethod
    def _manual_push_diff_preview_summary(selected_count: int, target_track_name: str, target_track_coord: str) -> str:
        noun = "event" if selected_count == 1 else "events"
        return (
            f"Prepared diff preview for {selected_count} selected {noun}.\n\n"
            f"Target track: {target_track_name} ({target_track_coord})\n"
            f"No MA3 transfer has been started in this step."
        )

    @staticmethod
    def _manual_pull_track_label(track) -> str:
        parts = [track.name, f"({track.coord})"]
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} events]")
        return " ".join(parts)

    @staticmethod
    def _manual_pull_event_label(event) -> str:
        parts = [event.label]
        if event.start is not None and event.end is not None:
            parts.append(f"({_format_seconds(event.start)}-{_format_seconds(event.end)})")
        return " ".join(parts)

    def _open_manual_pull_timeline_popup(self, flow) -> ManualPullTimelineSelectionResult | None:
        source_track = next(
            (track for track in flow.available_tracks if track.coord == flow.active_source_track_coord),
            None,
        )
        source_track_label = (
            self._manual_pull_track_label(source_track)
            if source_track is not None
            else str(flow.active_source_track_coord)
        )
        selected_event_ids = list(
            flow.selected_ma3_event_ids_by_track.get(flow.active_source_track_coord, flow.selected_ma3_event_ids)
        )
        selected_target_layer_id = flow.target_layer_id_by_source_track.get(
            flow.active_source_track_coord,
            flow.target_layer_id,
        )
        dialog = ManualPullTimelineDialog(
            source_track_label=source_track_label,
            events=flow.available_events,
            selected_event_ids=selected_event_ids,
            available_targets=flow.available_target_layers,
            selected_target_layer_id=selected_target_layer_id,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return ManualPullTimelineSelectionResult(
            selected_event_ids=dialog.selected_event_ids(),
            target_layer_id=dialog.selected_target_layer_id(),
        )

    @staticmethod
    def _manual_pull_target_label(target) -> str:
        return target.name

    @staticmethod
    def _manual_pull_diff_preview_summary(
        selected_count: int,
        source_track_name: str,
        source_track_coord: str,
        target_layer_name: str,
    ) -> str:
        noun = "event" if selected_count == 1 else "events"
        return (
            f"Prepared diff preview for {selected_count} selected {noun}.\n\n"
            f"Source track: {source_track_name} ({source_track_coord})\n"
            f"Target layer: {target_layer_name}\n"
            f"No MA3 import has been started in this step."
        )

    @staticmethod
    def _transfer_plan_preview_summary(
        *,
        operation_type: str,
        total_rows: int,
        ready_count: int,
        blocked_count: int,
        applied_count: int,
        failed_count: int,
    ) -> str:
        return (
            f"{_transfer_plan_operation_label(operation_type)} plan preview complete.\n\n"
            f"Rows: {total_rows}\n"
            f"Ready: {ready_count}\n"
            f"Blocked: {blocked_count}\n"
            f"Applied: {applied_count}\n"
            f"Failed: {failed_count}"
        )

    @staticmethod
    def _transfer_plan_apply_summary(
        *,
        operation_type: str,
        total_rows: int,
        applied_count: int,
        failed_count: int,
        blocked_count: int,
    ) -> str:
        return (
            f"{_transfer_plan_operation_label(operation_type)} plan apply complete.\n\n"
            f"Rows: {total_rows}\n"
            f"Applied: {applied_count}\n"
            f"Failed: {failed_count}\n"
            f"Blocked: {blocked_count}"
        )

    @staticmethod
    def _transfer_preset_label(preset) -> str:
        return f"{preset.name} ({preset.preset_id})"


def _format_time_label(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def _transfer_plan_operation_label(operation_type: str) -> str:
    return (operation_type or "mixed").strip().capitalize()


def _ready_count_label(count: int) -> str:
    noun = "ready row" if count == 1 else "ready rows"
    return f"{count} {noun}"


def _preview_transfer_plan_label(plan) -> str:
    return f"Preview Transfer Plan ({_ready_count_label(plan.ready_count)})"


def _apply_transfer_plan_label(plan) -> str:
    return f"Apply Transfer Plan ({_ready_count_label(plan.ready_count)})"


    DeleteTransferPreset,
