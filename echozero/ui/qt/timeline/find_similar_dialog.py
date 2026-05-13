"""
Event-comparison dialog for timeline event selection.
Exists to make one comparison strategy visible, tunable, and reviewable for operators.
Connects anchor preview, comparison controls, live candidate scoring, and final selection review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PyQt6.QtCore import QPointF, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract_lookup import find_event
from echozero.application.presentation.inspector_contract_preview import event_preview_params
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TimelinePresentation,
)
from echozero.application.timeline.event_comparison_service import (
    TimbreFingerprintSettings,
    build_timbre_fingerprint_preview,
    EventComparisonCandidateRecord,
    EventComparisonRequest,
    EventComparisonScoredCandidate,
    EventComparisonService,
)
from echozero.application.timeline.event_similarity_audio import (
    ShapeNormalizationSettings,
    load_event_shape_bundle,
)
from echozero.application.timeline.models import EventRef
from echozero.ui.qt.timeline.object_info_panel_preview import (
    EventPreviewState,
    event_preview_meta_text,
)

_DEFAULT_TOLERANCE_PERCENTAGE = 78.0
_DEFAULT_COMPARISON_MODE = "shape_envelope"
_COMPARISON_MODE_OPTIONS = (
    ("Shape Envelope", "shape_envelope"),
    ("Timbre Fingerprint", "timbre_fingerprint"),
)


@dataclass(slots=True)
class _ScopeTakeRecord:
    """Presentation-side take record normalized for event comparison."""

    take_id: object
    name: str
    events: list[EventPresentation]
    source_content_ref: object | None = None
    waveform_key: str | None = None
    source_audio_path: str | None = None


@dataclass(slots=True)
class _ScopeEntry:
    """One selectable search scope entry in the dialog tree."""

    layer: LayerPresentation
    take: _ScopeTakeRecord
    item: QTreeWidgetItem


class EventShapeGraph(QFrame):
    """Paint one normalized event graph, optionally against a reference graph."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(104)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._samples: tuple[float, ...] | None = None
        self._reference_samples: tuple[float, ...] | None = None
        self._match_state = "idle"

    def sizeHint(self) -> QSize:
        return QSize(240, 104)

    def set_graph(
        self,
        *,
        samples: tuple[float, ...] | None,
        reference_samples: tuple[float, ...] | None = None,
        match_state: str = "idle",
    ) -> None:
        self._samples = samples
        self._reference_samples = reference_samples
        self._match_state = match_state
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(6, 6, -6, -6)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1a1f24"))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        grid_pen = QPen(QColor("#2c343c"), 1.0)
        painter.setPen(grid_pen)
        mid_y = rect.center().y()
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)
        painter.drawLine(rect.left(), rect.top() + (rect.height() // 4), rect.right(), rect.top() + (rect.height() // 4))
        painter.drawLine(rect.left(), rect.bottom() - (rect.height() // 4), rect.right(), rect.bottom() - (rect.height() // 4))
        painter.drawLine(rect.left(), rect.top(), rect.right(), rect.top())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())

        if self._reference_samples:
            reference_pen = QPen(QColor("#6c7a89"), 1.0)
            reference_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(reference_pen)
            self._draw_samples(painter, rect, self._reference_samples)

        if self._samples:
            self._paint_area_fill(painter, rect, self._samples)
            painter.setPen(QPen(self._line_color(), 2.0))
            self._draw_samples(painter, rect, self._samples)
            return

        placeholder_pen = QPen(QColor("#53606c"), 1.0)
        placeholder_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(placeholder_pen)
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)

    def _line_color(self) -> QColor:
        if self._match_state == "match":
            return QColor("#54d08a")
        if self._match_state == "miss":
            return QColor("#f1c75b")
        return QColor("#7fd1ae")

    def _fill_color(self) -> QColor:
        if self._match_state == "match":
            color = QColor("#54d08a")
            color.setAlpha(48)
            return color
        if self._match_state == "miss":
            color = QColor("#f1c75b")
            color.setAlpha(38)
            return color
        color = QColor("#7fd1ae")
        color.setAlpha(44)
        return color

    @staticmethod
    def _draw_samples(
        painter: QPainter,
        rect: QRect,
        samples: tuple[float, ...],
    ) -> None:
        if len(samples) == 1:
            x = float(rect.left())
            y = float(rect.bottom()) - (float(samples[0]) * float(rect.height()))
            painter.drawPoint(QPointF(x, y))
            return
        points: list[QPointF] = []
        width = max(1.0, float(rect.width()))
        height = max(1.0, float(rect.height()))
        last_index = max(1, len(samples) - 1)
        for index, value in enumerate(samples):
            x = float(rect.left()) + (width * (index / last_index))
            y = float(rect.bottom()) - (max(0.0, min(1.0, float(value))) * height)
            points.append(QPointF(x, y))
        for point_a, point_b in zip(points, points[1:]):
            painter.drawLine(point_a, point_b)

    def _paint_area_fill(
        self,
        painter: QPainter,
        rect: QRect,
        samples: tuple[float, ...],
    ) -> None:
        if len(samples) < 2:
            return
        width = max(1.0, float(rect.width()))
        height = max(1.0, float(rect.height()))
        last_index = max(1, len(samples) - 1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._fill_color())
        baseline = rect.bottom()
        for index, value in enumerate(samples):
            x = float(rect.left()) + (width * (index / last_index))
            y = float(rect.bottom()) - (max(0.0, min(1.0, float(value))) * height)
            next_index = min(last_index, index + 1)
            next_x = float(rect.left()) + (width * (next_index / last_index))
            painter.drawRect(
                int(round(x)),
                int(round(y)),
                max(1, int(round(next_x - x))),
                max(1, int(round(baseline - y))),
            )


class EventComparisonDialog(QDialog):
    """Single-panel operator workflow for one event-comparison strategy."""

    def __init__(
        self,
        *,
        presentation: TimelinePresentation,
        layer_id: object,
        take_id: object | None,
        event_id: object,
        default_scope_mode: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presentation = presentation
        self._anchor_match = find_event(
            presentation,
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
        )
        if self._anchor_match is None:
            raise ValueError("EventComparisonDialog requires a resolvable anchor event")
        self._anchor_layer, self._anchor_take_presentation, self._anchor_event = self._anchor_match
        self._anchor_take = self._resolve_anchor_take_record()
        self._anchor_preview = self._build_preview_state(
            self._anchor_layer,
            self._anchor_take_presentation,
            self._anchor_event,
        )
        self._comparison_service = EventComparisonService()
        self._scope_entries: list[_ScopeEntry] = []
        self._syncing_scope_tree = False
        self._latest_results: list[EventComparisonScoredCandidate] = []
        self._selected_result_refs: list[EventRef] = []
        self._selected_layer_ids: list[object] = []
        self._preview_by_event_ref: dict[EventRef, EventPreviewState | None] = {}
        self._row_by_event_ref: dict[EventRef, int] = {}
        self._result_by_event_ref: dict[EventRef, EventComparisonScoredCandidate] = {}
        self._anchor_shape: tuple[float, ...] | None = None
        self._audio_cache: dict[str, tuple[np.ndarray, int]] = {}

        self.setWindowTitle("Compare Events")
        self.resize(1160, 860)
        self._apply_dialog_styles()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        intro = QLabel(
            "Inspect the anchor preview, tune the comparison settings, "
            "then watch EchoZero compare every checked event live.",
            self,
        )
        intro.setObjectName("findSimilarIntro")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        anchor_frame = QFrame(self)
        anchor_frame.setObjectName("findSimilarCard")
        anchor_layout = QVBoxLayout(anchor_frame)
        anchor_layout.setContentsMargins(14, 14, 14, 14)
        anchor_layout.setSpacing(8)
        anchor_title = QLabel("Anchor Preview", anchor_frame)
        anchor_title.setObjectName("findSimilarSectionTitle")
        anchor_layout.addWidget(anchor_title)
        self._anchor_graph = EventShapeGraph(anchor_frame)
        self._anchor_graph.setToolTip(
            "The current preview for the anchor event under the active comparison mode."
        )
        anchor_layout.addWidget(self._anchor_graph)
        self._anchor_meta = QLabel("", anchor_frame)
        self._anchor_meta.setObjectName("findSimilarMeta")
        self._anchor_meta.setWordWrap(True)
        anchor_layout.addWidget(self._anchor_meta)
        top_row.addWidget(anchor_frame, 3)
        self._set_anchor_preview()

        controls_frame = QFrame(self)
        controls_frame.setObjectName("findSimilarCard")
        controls_layout = QGridLayout(controls_frame)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        controls_title = QLabel("Comparison Controls", controls_frame)
        controls_title.setObjectName("findSimilarSectionTitle")
        controls_layout.addWidget(controls_title, 0, 0, 1, 4)

        controls_layout.addWidget(QLabel("Mode", controls_frame), 1, 0)
        self._mode_combo = QComboBox(controls_frame)
        for label, value in _COMPARISON_MODE_OPTIONS:
            self._mode_combo.addItem(label, value)
        controls_layout.addWidget(self._mode_combo, 1, 1, 1, 3)

        controls_layout.addWidget(QLabel("Tolerance %", controls_frame), 2, 0)
        self._tolerance_spin = QDoubleSpinBox(controls_frame)
        self._tolerance_spin.setRange(0.0, 100.0)
        self._tolerance_spin.setDecimals(1)
        self._tolerance_spin.setSingleStep(1.0)
        self._tolerance_spin.setValue(_DEFAULT_TOLERANCE_PERCENTAGE)
        self._tolerance_spin.setToolTip(
            "Minimum similarity percentage required for a candidate to count as a match."
        )
        controls_layout.addWidget(self._tolerance_spin, 2, 1)

        controls_layout.addWidget(QLabel("Points", controls_frame), 2, 2)
        self._sample_count_spin = QSpinBox(controls_frame)
        self._sample_count_spin.setRange(16, 256)
        self._sample_count_spin.setSingleStep(8)
        self._sample_count_spin.setValue(64)
        self._sample_count_spin.setToolTip(
            "How many preview points are used for the active comparison mode."
        )
        controls_layout.addWidget(self._sample_count_spin, 2, 3)

        controls_layout.addWidget(QLabel("Smoothing ms", controls_frame), 3, 0)
        self._smoothing_spin = QDoubleSpinBox(controls_frame)
        self._smoothing_spin.setRange(0.0, 250.0)
        self._smoothing_spin.setDecimals(1)
        self._smoothing_spin.setSingleStep(2.0)
        self._smoothing_spin.setValue(12.0)
        self._smoothing_spin.setToolTip(
            "Shape Envelope mode smooths the event envelope by this amount before normalization."
        )
        controls_layout.addWidget(self._smoothing_spin, 3, 1)

        controls_layout.addWidget(QLabel("Padding ms", controls_frame), 3, 2)
        self._padding_spin = QDoubleSpinBox(controls_frame)
        self._padding_spin.setRange(0.0, 250.0)
        self._padding_spin.setDecimals(1)
        self._padding_spin.setSingleStep(5.0)
        self._padding_spin.setValue(20.0)
        self._padding_spin.setToolTip(
            "Extra clip padding added before feature extraction to preserve onset and decay context."
        )
        controls_layout.addWidget(self._padding_spin, 3, 3)

        self._controls_help = QLabel("", controls_frame)
        self._controls_help.setObjectName("findSimilarHelp")
        self._controls_help.setWordWrap(True)
        self._controls_help.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout.addWidget(self._controls_help, 4, 0, 1, 4)
        top_row.addWidget(controls_frame, 2)
        layout.addLayout(top_row)

        middle_row = QHBoxLayout()
        middle_row.setSpacing(12)

        scope_frame = QFrame(self)
        scope_frame.setObjectName("findSimilarCard")
        scope_layout = QVBoxLayout(scope_frame)
        scope_layout.setContentsMargins(14, 14, 14, 14)
        scope_layout.setSpacing(6)
        scope_title = QLabel("Comparison Scope", scope_frame)
        scope_title.setObjectName("findSimilarSectionTitle")
        scope_layout.addWidget(scope_title)
        scope_hint = QLabel(
            "Choose the lanes EchoZero should scan for this comparison run.",
            scope_frame,
        )
        scope_hint.setObjectName("findSimilarHint")
        scope_hint.setWordWrap(True)
        scope_layout.addWidget(scope_hint)
        self._scope_tree = QTreeWidget(scope_frame)
        self._scope_tree.setHeaderLabels(["Layer / Take", "Events"])
        self._scope_tree.setMinimumHeight(190)
        self._scope_tree.itemChanged.connect(self._handle_scope_item_changed)
        scope_layout.addWidget(self._scope_tree)
        middle_row.addWidget(scope_frame, 2)
        self._populate_scope_tree(default_scope_mode)

        candidate_frame = QFrame(self)
        candidate_frame.setObjectName("findSimilarCard")
        candidate_layout = QVBoxLayout(candidate_frame)
        candidate_layout.setContentsMargins(14, 14, 14, 14)
        candidate_layout.setSpacing(6)
        candidate_header = QHBoxLayout()
        candidate_header.setSpacing(8)
        self._candidate_title = QLabel("Candidate Preview", candidate_frame)
        self._candidate_title.setObjectName("findSimilarSectionTitle")
        candidate_header.addWidget(self._candidate_title, 1)
        self._candidate_badge = QLabel("Idle", candidate_frame)
        self._candidate_badge.setObjectName("findSimilarBadgeIdle")
        candidate_header.addWidget(self._candidate_badge, 0, Qt.AlignmentFlag.AlignRight)
        candidate_layout.addLayout(candidate_header)
        self._candidate_graph = EventShapeGraph(candidate_frame)
        candidate_layout.addWidget(self._candidate_graph)
        self._candidate_meta = QLabel(
            "Run the comparison, then click a result row to inspect it against the anchor.",
            candidate_frame,
        )
        self._candidate_meta.setObjectName("findSimilarMeta")
        self._candidate_meta.setWordWrap(True)
        candidate_layout.addWidget(self._candidate_meta)
        middle_row.addWidget(candidate_frame, 3)
        layout.addLayout(middle_row)

        run_row = QFrame(self)
        run_row.setObjectName("findSimilarStatusCard")
        run_layout = QGridLayout(run_row)
        run_layout.setContentsMargins(14, 14, 14, 14)
        run_layout.setHorizontalSpacing(12)
        run_layout.setVerticalSpacing(8)
        self._summary = QLabel("", run_row)
        self._summary.setObjectName("findSimilarSummary")
        self._summary.setWordWrap(True)
        run_layout.addWidget(self._summary, 0, 0, 1, 2)
        self._progress_label = QLabel("Ready to compare.", run_row)
        self._progress_label.setObjectName("findSimilarHint")
        self._progress_label.setWordWrap(True)
        run_layout.addWidget(self._progress_label, 1, 0)
        self._run_button = QPushButton("Run Comparison", run_row)
        self._run_button.clicked.connect(self._run_search)
        run_layout.addWidget(self._run_button, 1, 1)
        self._progress_bar = QProgressBar(run_row)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        run_layout.addWidget(self._progress_bar, 2, 0, 1, 2)
        layout.addWidget(run_row)

        results_frame = QFrame(self)
        results_frame.setObjectName("findSimilarCard")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(14, 14, 14, 14)
        results_layout.setSpacing(6)
        results_title = QLabel("Comparison Results", results_frame)
        results_title.setObjectName("findSimilarSectionTitle")
        results_layout.addWidget(results_title)
        self._results_table = QTableWidget(results_frame)
        self._results_table.setColumnCount(7)
        self._results_table.setHorizontalHeaderLabels(
            ["Use", "Score", "Preview", "Layer", "Take", "Event", "Start"]
        )
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.verticalHeader().setDefaultSectionSize(46)
        self._results_table.setMinimumHeight(220)
        self._results_table.itemSelectionChanged.connect(self._sync_candidate_preview)
        header = self._results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self._results_table)
        layout.addWidget(results_frame, stretch=1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Apply Selection")
            ok_button.setEnabled(False)
        layout.addWidget(self._buttons)

        self._tolerance_spin.valueChanged.connect(lambda _value: self._refresh_summary())
        self._mode_combo.currentIndexChanged.connect(lambda _value: self._refresh_mode_controls())
        self._mode_combo.currentIndexChanged.connect(lambda _value: self._refresh_controls_help())
        self._mode_combo.currentIndexChanged.connect(lambda _value: self._refresh_summary())
        self._mode_combo.currentIndexChanged.connect(lambda _value: self._refresh_anchor_shape_preview())
        self._sample_count_spin.valueChanged.connect(lambda _value: self._refresh_controls_help())
        self._sample_count_spin.valueChanged.connect(lambda _value: self._refresh_summary())
        self._sample_count_spin.valueChanged.connect(lambda _value: self._refresh_anchor_shape_preview())
        self._smoothing_spin.valueChanged.connect(lambda _value: self._refresh_controls_help())
        self._smoothing_spin.valueChanged.connect(lambda _value: self._refresh_summary())
        self._smoothing_spin.valueChanged.connect(lambda _value: self._refresh_anchor_shape_preview())
        self._padding_spin.valueChanged.connect(lambda _value: self._refresh_controls_help())
        self._padding_spin.valueChanged.connect(lambda _value: self._refresh_summary())
        self._padding_spin.valueChanged.connect(lambda _value: self._refresh_anchor_shape_preview())
        self._refresh_mode_controls()
        self._refresh_controls_help()
        self._refresh_summary()
        self._refresh_anchor_shape_preview()

    def selected_payload(self) -> dict[str, object]:
        """Return the selection payload to dispatch when the dialog is accepted."""

        return {
            "event_refs": list(self._selected_result_refs),
            "event_ids": [event_ref.event_id for event_ref in self._selected_result_refs],
            "anchor_layer_id": self._anchor_layer.layer_id,
            "anchor_take_id": self._anchor_take.take_id,
            "selected_layer_ids": list(self._selected_layer_ids),
        }

    def _set_anchor_preview(self) -> None:
        if self._anchor_preview is None:
            self._anchor_meta.setText("Anchor preview unavailable for this event.")
            self._anchor_graph.set_graph(samples=None)
            return
        self._anchor_meta.setText(event_preview_meta_text(self._anchor_preview))
        self._anchor_graph.set_graph(samples=None, match_state="idle")

    def _refresh_anchor_shape_preview(self) -> None:
        shape = self._load_preview_shape(self._anchor_preview)
        self._anchor_shape = shape
        self._anchor_graph.set_graph(samples=shape, match_state="idle")
        if self._anchor_preview is None:
            return
        if shape is None:
            self._anchor_meta.setText(
                f"{event_preview_meta_text(self._anchor_preview)}\nPreview unavailable for the active comparison mode."
            )
            return
        mode_label = self._current_comparison_mode_label()
        self._anchor_meta.setText(
            f"{event_preview_meta_text(self._anchor_preview)}\n"
            f"{mode_label} preview with {len(shape)} normalized points."
        )

    def _resolve_anchor_take_record(self) -> _ScopeTakeRecord:
        if self._anchor_take_presentation is not None:
            return _ScopeTakeRecord(
                take_id=self._anchor_take_presentation.take_id,
                name=self._anchor_take_presentation.name,
                events=list(self._anchor_take_presentation.events),
                source_content_ref=self._anchor_take_presentation.source_content_ref,
                waveform_key=self._anchor_take_presentation.waveform_key,
                source_audio_path=self._anchor_take_presentation.source_audio_path,
            )
        return self._main_take_record(self._anchor_layer)

    def _populate_scope_tree(self, default_scope_mode: str) -> None:
        self._scope_tree.clear()
        self._scope_entries = []
        self._syncing_scope_tree = True
        try:
            for layer in self._presentation.layers:
                if not _layer_has_searchable_events(layer):
                    continue
                layer_item = QTreeWidgetItem([layer.title, str(_layer_event_count(layer))])
                layer_item.setFlags(
                    layer_item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                layer_item.setCheckState(0, Qt.CheckState.Unchecked)
                self._scope_tree.addTopLevelItem(layer_item)

                main_take = self._main_take_record(layer)
                main_item = QTreeWidgetItem([f"Main: {main_take.name}", str(len(main_take.events))])
                main_item.setFlags(main_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                layer_item.addChild(main_item)
                self._scope_entries.append(_ScopeEntry(layer=layer, take=main_take, item=main_item))

                for take in layer.takes:
                    take_item = QTreeWidgetItem([take.name, str(len(take.events))])
                    take_item.setFlags(take_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    layer_item.addChild(take_item)
                    self._scope_entries.append(
                        _ScopeEntry(
                            layer=layer,
                            take=_ScopeTakeRecord(
                                take_id=take.take_id,
                                name=take.name,
                                events=list(take.events),
                                source_content_ref=take.source_content_ref,
                                waveform_key=take.waveform_key,
                                source_audio_path=take.source_audio_path,
                            ),
                            item=take_item,
                        )
                    )
                layer_item.setExpanded(layer.layer_id == self._anchor_layer.layer_id)
            self._apply_default_scope_checks(default_scope_mode)
        finally:
            self._syncing_scope_tree = False
        self._refresh_summary()

    def _apply_default_scope_checks(self, default_scope_mode: str) -> None:
        for entry in self._scope_entries:
            should_check = False
            if default_scope_mode == "take":
                should_check = (
                    entry.layer.layer_id == self._anchor_layer.layer_id
                    and entry.take.take_id == self._anchor_take.take_id
                )
            elif default_scope_mode == "layer":
                should_check = entry.layer.layer_id == self._anchor_layer.layer_id
            else:
                should_check = entry.layer.layer_id in (
                    self._presentation.selected_layer_ids or [self._anchor_layer.layer_id]
                )
            entry.item.setCheckState(
                0,
                Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked,
            )
        self._refresh_parent_scope_checks()

    def _handle_scope_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._syncing_scope_tree:
            return
        self._syncing_scope_tree = True
        try:
            if item.parent() is None:
                for child_index in range(item.childCount()):
                    item.child(child_index).setCheckState(0, item.checkState(0))
            self._refresh_parent_scope_checks()
        finally:
            self._syncing_scope_tree = False
        self._refresh_summary()

    def _refresh_parent_scope_checks(self) -> None:
        for layer_index in range(self._scope_tree.topLevelItemCount()):
            layer_item = self._scope_tree.topLevelItem(layer_index)
            child_states = {layer_item.child(i).checkState(0) for i in range(layer_item.childCount())}
            if not child_states or child_states == {Qt.CheckState.Unchecked}:
                layer_item.setCheckState(0, Qt.CheckState.Unchecked)
            elif child_states == {Qt.CheckState.Checked}:
                layer_item.setCheckState(0, Qt.CheckState.Checked)
            else:
                layer_item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _main_take_record(self, layer: LayerPresentation) -> _ScopeTakeRecord:
        return _ScopeTakeRecord(
            take_id=layer.main_take_id,
            name="Main Take",
            events=list(layer.events),
            source_content_ref=layer.source_content_ref,
            waveform_key=layer.waveform_key,
            source_audio_path=layer.source_audio_path,
        )

    def _refresh_controls_help(self) -> None:
        settings = self._current_normalization_settings()
        if self._current_comparison_mode() == "timbre_fingerprint":
            self._controls_help.setText(
                f"Timbre Fingerprint mode distills each event into a {settings.sample_count}-band log-mel spectral profile. "
                f"Padding stays active to capture a little more onset and decay context; smoothing is not used in this mode."
            )
            return
        self._controls_help.setText(
            f"Shape Envelope mode reduces each event to a {settings.sample_count}-point preview with "
            f"{settings.smoothing_ms:.1f}ms smoothing and {settings.padding_ms:.1f}ms padding."
        )

    def _refresh_mode_controls(self) -> None:
        is_timbre = self._current_comparison_mode() == "timbre_fingerprint"
        self._smoothing_spin.setEnabled(not is_timbre)

    def _refresh_summary(self) -> None:
        if not hasattr(self, "_summary"):
            return
        checked_entries = self._checked_scope_entries()
        checked_event_count = sum(len(entry.take.events) for entry in checked_entries)
        settings = self._current_normalization_settings()
        mode_label = self._current_comparison_mode_label()
        mode_suffix = (
            f"{settings.sample_count} points, {settings.padding_ms:.1f}ms padding."
            if self._current_comparison_mode() == "timbre_fingerprint"
            else f"{settings.sample_count} points, {settings.smoothing_ms:.1f}ms smoothing, "
            f"{settings.padding_ms:.1f}ms padding."
        )
        self._summary.setText(
            f"{checked_event_count} candidate events across {len(checked_entries)} checked scopes. "
            f"Match at or above {self._tolerance_spin.value():.1f}% similarity. "
            f"Mode: {mode_label}. {mode_suffix}"
        )

    def _run_search(self) -> None:
        checked_entries = self._checked_scope_entries()
        if not checked_entries:
            self._progress_label.setText("Check at least one take or main lane before running the comparison.")
            return

        candidate_records = [
            EventComparisonCandidateRecord(
                layer_id=entry.layer.layer_id,
                take_id=entry.take.take_id,
                event=event,
                layer=entry.layer,
                take=entry.take,
            )
            for entry in checked_entries
            for event in entry.take.events
        ]
        if not candidate_records:
            self._progress_label.setText("The checked scope does not contain any events to compare.")
            return

        self._prepare_results_table(candidate_records)
        request = EventComparisonRequest(
            anchor_event_id=self._anchor_event.event_id,
            comparison_mode=self._current_comparison_mode(),
            similarity_threshold=max(0.0, min(1.0, self._tolerance_spin.value() / 100.0)),
            comparison_settings=self._current_comparison_settings(),
        )
        self._progress_bar.setRange(0, max(1, len(candidate_records)))
        self._progress_bar.setValue(0)
        self._progress_label.setText(
            f"Running {self._current_comparison_mode_label()} comparison..."
        )
        QApplication.processEvents()
        self._latest_results = self._comparison_service.analyze_candidates(
            anchor_layer=self._anchor_layer,
            anchor_take=self._anchor_take,
            candidate_records=candidate_records,
            request=request,
            on_progress=self._handle_progress,
        )
        self._result_by_event_ref = {
            result.event_ref: result for result in self._latest_results
        }
        anchor_result = self._result_by_event_ref.get(
            EventRef(
                layer_id=self._anchor_layer.layer_id,
                take_id=self._anchor_take.take_id,
                event_id=self._anchor_event.event_id,
            )
        )
        self._anchor_shape = None if anchor_result is None else anchor_result.normalized_shape
        self._anchor_graph.set_graph(samples=self._anchor_shape, match_state="idle")
        self._selected_result_refs = [
            result.event_ref for result in self._latest_results if result.is_selected
        ]
        self._selected_layer_ids = list(dict.fromkeys(entry.layer.layer_id for entry in checked_entries))
        self._progress_label.setText(
            f"Comparison complete. {len(self._selected_result_refs)} event(s) meet the tolerance."
        )
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(self._selected_result_refs))
        self._select_first_result_row()
        self._sync_candidate_preview()

    def _handle_progress(
        self,
        current: int,
        total: int,
        result: EventComparisonScoredCandidate,
        record: EventComparisonCandidateRecord,
    ) -> None:
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(current)
        self._update_result_row(result, record)
        self._progress_label.setText(
            f"Comparing {current} of {total}: {record.layer.title} / {record.take.name} / "
            f"{record.event.label or 'Event'} at {record.event.start:.2f}s"
        )
        if (
            self._anchor_shape is None
            and result.event_ref.layer_id == self._anchor_layer.layer_id
            and result.event_ref.take_id == self._anchor_take.take_id
            and result.event_ref.event_id == self._anchor_event.event_id
        ):
            self._anchor_shape = result.normalized_shape
            self._anchor_graph.set_graph(samples=self._anchor_shape, match_state="idle")
        self._show_live_candidate(result, current=current, total=total)
        QApplication.processEvents()

    def _prepare_results_table(
        self,
        candidate_records: list[EventComparisonCandidateRecord],
    ) -> None:
        self._results_table.setRowCount(len(candidate_records))
        self._preview_by_event_ref = {}
        self._row_by_event_ref = {}
        self._result_by_event_ref = {}
        self._anchor_shape = None
        self._refresh_anchor_shape_preview()

        for row_index, record in enumerate(candidate_records):
            event_ref = EventRef(
                layer_id=record.layer_id,
                take_id=record.take_id,
                event_id=record.event.event_id,
            )
            self._row_by_event_ref[event_ref] = row_index
            preview = self._build_preview_state(record.layer, self._resolve_take_presentation(record), record.event)
            self._preview_by_event_ref[event_ref] = preview

            self._results_table.setItem(row_index, 0, QTableWidgetItem("Pending"))
            self._results_table.setItem(row_index, 1, QTableWidgetItem("..."))
            graph = EventShapeGraph(self._results_table)
            graph.set_graph(samples=None, match_state="idle")
            self._results_table.setCellWidget(row_index, 2, graph)
            self._results_table.setItem(row_index, 3, QTableWidgetItem(record.layer.title))
            self._results_table.setItem(row_index, 4, QTableWidgetItem(record.take.name))
            self._results_table.setItem(row_index, 5, QTableWidgetItem(record.event.label or "Event"))
            self._results_table.setItem(row_index, 6, QTableWidgetItem(f"{record.event.start:.2f}s"))
        self._results_table.clearSelection()
        self._candidate_graph.set_graph(samples=None, reference_samples=None, match_state="idle")
        self._candidate_title.setText("Candidate Preview")
        self._set_candidate_badge("idle")
        self._candidate_meta.setText(
            "Run the comparison, then click a result row to inspect its preview against the anchor."
        )

    def _update_result_row(
        self,
        result: EventComparisonScoredCandidate,
        record: EventComparisonCandidateRecord,
    ) -> None:
        row_index = self._row_by_event_ref.get(result.event_ref)
        if row_index is None:
            return
        use_item = self._results_table.item(row_index, 0)
        score_item = self._results_table.item(row_index, 1)
        if use_item is None or score_item is None:
            return
        use_item.setText("Match" if result.is_selected else "Skip")
        score_item.setText(
            f"{result.similarity_percentage:.1f}%"
            if result.similarity_percentage is not None
            else "n/a"
        )
        graph = self._results_table.cellWidget(row_index, 2)
        if isinstance(graph, EventShapeGraph):
            graph.set_graph(
                samples=result.normalized_shape,
                reference_samples=self._anchor_shape,
                match_state="match" if result.is_selected else "miss",
            )
        background = QColor("#103525") if result.is_selected else QColor("#2a2620")
        for column in (0, 1, 3, 4, 5, 6):
            item = self._results_table.item(row_index, column)
            if item is not None:
                item.setBackground(background)
        if (
            self._candidate_meta.text().startswith("Run the comparison")
            and row_index == 0
        ):
            self._results_table.selectRow(0)
            self._sync_candidate_preview()
        if self._results_table.currentRow() == row_index:
            self._sync_candidate_preview()

    def _sync_candidate_preview(self) -> None:
        row_index = self._results_table.currentRow()
        if row_index < 0:
            return
        event_ref = next(
            (candidate_ref for candidate_ref, candidate_row in self._row_by_event_ref.items() if candidate_row == row_index),
            None,
        )
        if event_ref is None:
            return
        preview = self._preview_by_event_ref.get(event_ref)
        result = self._result_by_event_ref.get(event_ref)
        if result is None:
            self._candidate_graph.set_graph(samples=None, reference_samples=self._anchor_shape, match_state="idle")
            self._candidate_title.setText("Candidate Preview")
            self._candidate_badge.setText("Pending")
            self._candidate_badge.setObjectName("findSimilarBadgeIdle")
            self._candidate_badge.style().unpolish(self._candidate_badge)
            self._candidate_badge.style().polish(self._candidate_badge)
            self._candidate_meta.setText("This row has not been scored yet.")
            return
        match_state = "match" if result.is_selected else "miss"
        self._candidate_graph.set_graph(
            samples=result.normalized_shape,
            reference_samples=self._anchor_shape,
            match_state=match_state,
        )
        self._set_candidate_badge(match_state)
        preview_text = (
            event_preview_meta_text(preview)
            if preview is not None
            else "Preview metadata unavailable for this event."
        )
        score_text = (
            f"Similarity: {result.similarity_percentage:.1f}%"
            if result.similarity_percentage is not None
            else "Similarity: unavailable"
        )
        status_text = "MATCH" if result.is_selected else "BELOW TOLERANCE"
        self._candidate_title.setText("Selected Result")
        self._candidate_meta.setText(f"{score_text} · {status_text}\n{preview_text}")

    def _show_live_candidate(
        self,
        result: EventComparisonScoredCandidate,
        *,
        current: int,
        total: int,
    ) -> None:
        row_index = self._row_by_event_ref.get(result.event_ref)
        if row_index is not None:
            self._results_table.selectRow(row_index)
            item = self._results_table.item(row_index, 0)
            if item is not None:
                self._results_table.scrollToItem(item)
        preview = self._preview_by_event_ref.get(result.event_ref)
        match_state = "match" if result.is_selected else "miss"
        self._candidate_graph.set_graph(
            samples=result.normalized_shape,
            reference_samples=self._anchor_shape,
            match_state=match_state,
        )
        self._set_candidate_badge(match_state)
        preview_text = (
            event_preview_meta_text(preview)
            if preview is not None
            else "Preview metadata unavailable for this event."
        )
        score_text = (
            f"{result.similarity_percentage:.1f}%"
            if result.similarity_percentage is not None
            else "unavailable"
        )
        verdict = "PASS" if result.is_selected else "FAIL"
        self._candidate_title.setText(f"Checking Candidate {current} of {total}")
        self._candidate_meta.setText(
            f"Checking {current} of {total} · {verdict} · Similarity {score_text}\n{preview_text}"
        )

    def _set_candidate_badge(self, match_state: str) -> None:
        if match_state == "match":
            text = "PASS"
            object_name = "findSimilarBadgeMatch"
        elif match_state == "miss":
            text = "FAIL"
            object_name = "findSimilarBadgeMiss"
        else:
            text = "Idle"
            object_name = "findSimilarBadgeIdle"
        self._candidate_badge.setText(text)
        self._candidate_badge.setObjectName(object_name)
        self._candidate_badge.style().unpolish(self._candidate_badge)
        self._candidate_badge.style().polish(self._candidate_badge)

    def _select_first_result_row(self) -> None:
        if self._results_table.rowCount() <= 0:
            return
        selected_row = None
        for result in self._latest_results:
            if result.is_selected:
                selected_row = self._row_by_event_ref.get(result.event_ref)
                if selected_row is not None:
                    break
        if selected_row is None:
            selected_row = 0
        self._results_table.selectRow(selected_row)

    def _checked_scope_entries(self) -> list[_ScopeEntry]:
        return [
            entry
            for entry in self._scope_entries
            if entry.item.checkState(0) == Qt.CheckState.Checked
        ]

    def _current_normalization_settings(self) -> ShapeNormalizationSettings:
        return ShapeNormalizationSettings(
            sample_count=int(self._sample_count_spin.value()),
            smoothing_ms=float(self._smoothing_spin.value()),
            padding_ms=float(self._padding_spin.value()),
        )

    def _current_comparison_mode(self) -> str:
        value = self._mode_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        return _DEFAULT_COMPARISON_MODE

    def _current_comparison_mode_label(self) -> str:
        label = self._mode_combo.currentText().strip()
        return label or "Shape Envelope"

    def _current_comparison_settings(self) -> object:
        settings = self._current_normalization_settings()
        if self._current_comparison_mode() == "timbre_fingerprint":
            return TimbreFingerprintSettings(
                sample_count=settings.sample_count,
                padding_ms=settings.padding_ms,
            )
        return settings

    def _resolve_take_presentation(
        self,
        record: EventComparisonCandidateRecord,
    ) -> object | None:
        match = find_event(
            self._presentation,
            layer_id=record.layer_id,
            take_id=record.take_id,
            event_id=record.event.event_id,
        )
        if match is None:
            return None
        _layer, take, _event = match
        return take

    def _build_preview_state(
        self,
        layer: LayerPresentation,
        take: object | None,
        event: EventPresentation,
    ) -> EventPreviewState | None:
        params = event_preview_params(
            self._presentation,
            layer=layer,
            take=take,
            event=event,
        )
        if params is None:
            return None
        return EventPreviewState(
            layer_id=params.get("layer_id"),
            take_id=params.get("take_id"),
            event_id=params.get("event_id"),
            source_ref=str(params.get("source_ref") or ""),
            source_audio_path=(str(params.get("source_audio_path") or "").strip() or None),
            waveform_key=str(params.get("waveform_key") or "").strip() or None,
            start_seconds=float(params.get("start_seconds") or 0.0),
            end_seconds=float(params.get("end_seconds") or 0.0),
            duration_seconds=float(params.get("duration_seconds") or 0.0),
        )

    def _load_preview_shape(
        self,
        preview: EventPreviewState | None,
    ) -> tuple[float, ...] | None:
        if preview is None:
            return None
        source_path = str(preview.source_audio_path or preview.source_ref).strip()
        if not source_path:
            return None
        candidate = Path(source_path).expanduser()
        if not candidate.exists():
            return None
        if self._current_comparison_mode() == "timbre_fingerprint":
            return build_timbre_fingerprint_preview(
                audio_path=str(candidate),
                start_seconds=preview.start_seconds,
                end_seconds=preview.end_seconds,
                settings=TimbreFingerprintSettings(
                    sample_count=int(self._sample_count_spin.value()),
                    padding_ms=float(self._padding_spin.value()),
                ),
                audio_cache=self._audio_cache,
            )
        bundle = load_event_shape_bundle(
            audio_path=str(candidate),
            start_seconds=preview.start_seconds,
            end_seconds=preview.end_seconds,
            settings=self._current_normalization_settings(),
            audio_cache=self._audio_cache,
        )
        if bundle is None:
            return None
        return bundle.normalized_samples

    def _apply_dialog_styles(self) -> None:
        self.setStyleSheet(
            """
            QFrame#findSimilarCard, QFrame#findSimilarStatusCard {
                background-color: #141a22;
                border: 1px solid #2b3340;
                border-radius: 12px;
            }
            QLabel#findSimilarSectionTitle {
                font-size: 15px;
                font-weight: 600;
                color: #f3f6fa;
            }
            QLabel#findSimilarBadgeIdle,
            QLabel#findSimilarBadgeMatch,
            QLabel#findSimilarBadgeMiss {
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.04em;
            }
            QLabel#findSimilarBadgeIdle {
                background-color: #28303a;
                color: #c8d1dc;
                border: 1px solid #3b4654;
            }
            QLabel#findSimilarBadgeMatch {
                background-color: #103525;
                color: #83ebb0;
                border: 1px solid #2d8d57;
            }
            QLabel#findSimilarBadgeMiss {
                background-color: #342b14;
                color: #f3d37a;
                border: 1px solid #8b6c1f;
            }
            QLabel#findSimilarIntro {
                color: #c8d2de;
                font-size: 13px;
            }
            QLabel#findSimilarHint {
                color: #9aabbe;
                font-size: 12px;
            }
            QLabel#findSimilarMeta {
                color: #c7d3df;
                font-size: 12px;
            }
            QLabel#findSimilarHelp {
                color: #d9e3ed;
                background-color: #10151c;
                border: 1px solid #27303b;
                border-radius: 8px;
                padding: 8px 10px;
            }
            QLabel#findSimilarSummary {
                color: #eaf1f7;
                font-size: 13px;
                font-weight: 500;
            }
            QTreeWidget, QTableWidget {
                background-color: #0f141b;
                alternate-background-color: #131a22;
                border: 1px solid #27303b;
                border-radius: 10px;
                gridline-color: #27303b;
            }
            QHeaderView::section {
                background-color: #202838;
                color: #e9eef5;
                border: none;
                padding: 6px 8px;
                font-weight: 600;
            }
            QProgressBar {
                min-height: 10px;
                border: 1px solid #27303b;
                border-radius: 5px;
                background-color: #0f141b;
            }
            QProgressBar::chunk {
                background-color: #54d08a;
                border-radius: 4px;
            }
            """
        )


def _layer_has_searchable_events(layer: LayerPresentation) -> bool:
    return bool(layer.events or any(take.events for take in layer.takes))


def _layer_event_count(layer: LayerPresentation) -> int:
    return len(layer.events) + sum(len(take.events) for take in layer.takes)


FindSimilarSoundsDialog = EventComparisonDialog
