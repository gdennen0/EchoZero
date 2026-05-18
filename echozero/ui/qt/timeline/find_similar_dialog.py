"""
Find Similar review dialog for timeline event matching.
Exists so operators teach the matcher with quick audio review instead of tuning scorer knobs.
Connects the Qt popup to the application Find Similar review service and timeline action payloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QKeyEvent, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.find_similar_review_service import (
    FindSimilarCandidate,
    FindSimilarReviewService,
    FindSimilarReviewSession,
    REVIEW_CANDIDATE_LIMIT,
    ReviewLabel,
    TOP_CANDIDATE_LIMIT,
    save_find_similar_review_model,
)
from echozero.application.timeline.models import EventRef
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.style.qt import ensure_qt_theme_installed

PreviewEventCallback = Callable[[LayerId, TakeId, EventId], None]


@dataclass(frozen=True, slots=True)
class ShapePreviewRow:
    """Compatibility row for older imports that inspect dialog preview rows."""

    event_ref: EventRef
    label: str
    shape: tuple[float, ...]
    score: float | None = None
    is_anchor: bool = False
    is_match: bool = False


class CandidateRailWidget(QWidget):
    """Compact ranked confidence strip for Find Similar candidates."""

    def __init__(
        self,
        session: FindSimilarReviewSession,
        active_ref: EventRef | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._active_ref = active_ref
        self.setMinimumHeight(122)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def rows(self) -> tuple[ShapePreviewRow, ...]:
        return tuple(
            ShapePreviewRow(
                event_ref=candidate.event_ref,
                label=candidate.label,
                shape=candidate.preview_shape,
                score=candidate.score,
                is_anchor=candidate.is_anchor,
                is_match=candidate.review_label == ReviewLabel.POSITIVE,
            )
            for candidate in self._session.top_candidates
        )

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor(TIMELINE_STYLE.canvas.background_hex))
        painter.setPen(QPen(QColor(TIMELINE_STYLE.canvas.row_divider_hex), 1.0))
        painter.drawRoundedRect(rect, 3, 3)
        candidates = self._session.top_candidates
        if not candidates:
            painter.setPen(QColor(TIMELINE_STYLE.object_palette.body_hex))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No reviewable events")
            painter.end()
            return
        rail = rect.adjusted(18, 40, -18, -18)
        gap = 3.0
        bar_width = max(4.0, (rail.width() - gap * (len(candidates) - 1)) / len(candidates))
        painter.setPen(QColor(TIMELINE_STYLE.ruler.label_hex))
        painter.drawText(
            rect.adjusted(10, 4, -10, -4),
            Qt.AlignmentFlag.AlignTop,
            f"TOP {TOP_CANDIDATE_LIMIT} MATCH CANDIDATES",
        )
        review_boundary_x = rail.left() + REVIEW_CANDIDATE_LIMIT * (bar_width + gap) - gap / 2.0
        painter.setPen(QPen(QColor(TIMELINE_STYLE.canvas.row_divider_hex), 1.0))
        painter.drawLine(
            QPointF(review_boundary_x, rail.top()),
            QPointF(review_boundary_x, rail.bottom()),
        )
        for rank, candidate in enumerate(candidates, start=1):
            x = rail.left() + (rank - 1) * (bar_width + gap)
            self._paint_candidate_bar(
                painter,
                QRectF(x, rail.top(), bar_width, rail.height()),
                candidate,
                rank,
            )
        painter.setPen(QColor(TIMELINE_STYLE.ruler.label_hex))
        painter.drawText(
            QRectF(rail.left(), rail.top() - 18, rail.width(), 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"REVIEW FIRST {REVIEW_CANDIDATE_LIMIT}",
        )
        painter.end()

    def _paint_candidate_bar(
        self,
        painter: QPainter,
        rail_rect: QRectF,
        candidate: FindSimilarCandidate,
        rank: int,
    ) -> None:
        color = _candidate_color(candidate)
        active = self._active_ref is not None and _same_ref(candidate.event_ref, self._active_ref)
        height = max(5.0, rail_rect.height() * max(0.0, min(1.0, candidate.score)))
        bar = QRectF(
            rail_rect.left(),
            rail_rect.bottom() - height,
            rail_rect.width(),
            height,
        )
        fill = QColor(color)
        fill.setAlpha(215 if rank <= REVIEW_CANDIDATE_LIMIT else 115)
        painter.fillRect(bar, fill)
        painter.setPen(QPen(QColor(color), 2.0 if active else 0.8))
        painter.drawRect(bar)
        if active:
            painter.setPen(QPen(QColor(TIMELINE_STYLE.object_palette.title_hex), 1.4))
            painter.drawRect(rail_rect.adjusted(-1, -1, 1, 1))
        if rank <= 3:
            painter.setPen(QColor(TIMELINE_STYLE.canvas.background_hex))
            painter.drawText(
                bar.adjusted(0, 0, 0, -1),
                Qt.AlignmentFlag.AlignCenter,
                str(rank),
            )


class FindSimilarSoundsDialog(QDialog):
    """Review-and-train dialog for finding events like a selected timeline event."""

    def __init__(
        self,
        *,
        presentation: TimelinePresentation,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        default_scope_mode: str = "song",
        preview_event_callback: PreviewEventCallback | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        ensure_qt_theme_installed()
        self._presentation = presentation
        self._layer_id = layer_id
        self._take_id = take_id
        self._event_id = event_id
        self._default_scope_mode = default_scope_mode
        self._preview_event_callback = preview_event_callback
        self._saved_model_path = None
        self._service = FindSimilarReviewService()
        self._seed_event_refs = _initial_seed_event_refs(presentation, layer_id, take_id, event_id)
        self._session = self._service.start_session(
            presentation=presentation,
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
            scope_mode=default_scope_mode,
            seed_event_refs=self._seed_event_refs,
        )
        self._active_ref = self._session.next_candidate_ref
        self.setWindowTitle("Find Similar")
        self.resize(1040, 680)
        self.setMinimumSize(900, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        root.addWidget(self._build_header())
        self._rail = CandidateRailWidget(self._session, self._active_ref, self)
        self._preview_widget = self._rail
        root.addWidget(self._rail)

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_candidate_list(), stretch=3)
        body.addWidget(self._build_review_panel(), stretch=2)
        root.addLayout(body, stretch=1)
        root.addWidget(self._build_footer())
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)
        self._apply_styles()
        self._install_shortcuts()
        self._refresh_all()

    def selected_payload(self) -> dict[str, object]:
        selection = self._service.select_similar_events(self._session)
        reviewed_refs = list(self._session.seed_event_refs)
        profile_selected_refs = list(selection.event_refs)
        if self._save_model_checkbox.isChecked() and self._saved_model_path is None:
            self._saved_model_path = save_find_similar_review_model(self._session)
            self._model_status_label.setText(f"Saved profile {self._saved_model_path.name}")
        selected_layer_ids = _selected_layer_ids(
            self._presentation, self._layer_id, self._scope_combo.currentData()
        )
        payload: dict[str, object] = {
            "event_ids": [event_ref.event_id for event_ref in profile_selected_refs],
            "event_refs": profile_selected_refs,
            "matched_event_refs": profile_selected_refs,
            "match_count": len(profile_selected_refs),
            "anchor_layer_id": self._layer_id,
            "anchor_take_id": self._take_id,
            "selected_layer_ids": selected_layer_ids,
            "comparison_mode": "find_similar_review_profile",
            "scope_mode": str(self._scope_combo.currentData()),
            "match_strength": "reviewed",
            "match_threshold": self._session.confidence_threshold,
            "confidence_threshold": self._session.confidence_threshold,
            "shape_smoothing": 0,
            "shape_control_points": 32,
            "shape_fuzziness": 0,
            "outcome_action": str(self._outcome_combo.currentData()),
            "new_layer_title": self._layer_name_edit.text().strip() or "Similar Events",
            "reviewed_event_refs": reviewed_refs,
            "seed_event_refs": list(self._session.seed_event_refs),
            "negative_event_refs": list(self._session.negative_event_refs),
            "model_applied_event_refs": profile_selected_refs,
            "profile_selected_event_refs": profile_selected_refs,
            "profile_ready": self._session.can_select_similar,
            "profile_readiness_reason": self._session.match_profile.readiness_reason,
            "review_model_schema": "echozero.find-similar-review-model.v1",
        }
        if self._saved_model_path is not None:
            payload["review_model_path"] = str(self._saved_model_path)
        return payload

    def set_scan_preview_limit(self, value: int | None) -> None:
        """Compatibility no-op for automation that used the legacy preview cap."""

        del value
        self._refresh_all()

    def _install_shortcuts(self) -> None:
        shortcuts = (
            ("Shift+Space", self._play_active),
            ("Shift+C", lambda: self._mark_active(ReviewLabel.POSITIVE)),
        )
        self._shortcuts = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for widget in (
            self,
            self._candidate_list,
            self._play_button,
            self._match_button,
            self._reject_button,
            self._skip_button,
            self._primary_button,
        ):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.KeyPress and self._handle_shortcut_key(event):
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if self._handle_shortcut_key(event):
            return
        key = event.key()
        if key in {Qt.Key.Key_M, Qt.Key.Key_E}:
            self._mark_active(ReviewLabel.POSITIVE)
            return
        if key == Qt.Key.Key_N:
            self._mark_active(ReviewLabel.NEGATIVE)
            return
        if key == Qt.Key.Key_S:
            self._mark_active(ReviewLabel.SKIPPED)
            return
        if key == Qt.Key.Key_Space:
            self._play_active()
            return
        if key in {Qt.Key.Key_A, Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self._primary_action()
            return
        super().keyPressEvent(event)

    def _handle_shortcut_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        has_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if has_shift and key == Qt.Key.Key_C:
            self._mark_active(ReviewLabel.POSITIVE)
            event.accept()
            return True
        if has_shift and key == Qt.Key.Key_Space:
            self._play_active()
            event.accept()
            return True
        return False

    def _build_header(self) -> QWidget:
        frame = QFrame(self)
        layout = QGridLayout(frame)
        self._title_label = QLabel("Find Similar Sounds", frame)
        self._anchor_label = QLabel("Anchor", frame)
        self._scope_combo = QComboBox(frame)
        self._scope_combo.addItem("Whole Song", "song")
        self._scope_combo.addItem("Current Take", "take")
        self._scope_combo.addItem("Current Layer", "layer")
        self._scope_combo.addItem("Selected Layers · Main Takes", "selected_layers_main")
        index = self._scope_combo.findData(self._default_scope_mode)
        if index >= 0:
            self._scope_combo.setCurrentIndex(index)
        self._scope_combo.currentIndexChanged.connect(self._restart_session_for_scope)
        self._candidate_count_label = QLabel("", frame)
        self._review_count_label = QLabel("", frame)
        layout.addWidget(self._title_label, 0, 0)
        layout.addWidget(self._anchor_label, 1, 0, 1, 2)
        layout.addWidget(QLabel("Search in", frame), 0, 2)
        layout.addWidget(self._scope_combo, 1, 2)
        layout.addWidget(QLabel("Candidates", frame), 0, 3)
        layout.addWidget(self._candidate_count_label, 1, 3)
        layout.addWidget(QLabel("Review lane", frame), 0, 4)
        layout.addWidget(self._review_count_label, 1, 4)
        return frame

    def _build_candidate_list(self) -> QWidget:
        frame = QFrame(self)
        layout = QVBoxLayout(frame)
        header = QHBoxLayout()
        self._candidate_list_label = QLabel("Best Next", frame)
        self._view_combo = QComboBox(frame)
        self._view_combo.addItem("Best next", "best_next")
        self._view_combo.addItem("All unreviewed", "all_unreviewed")
        self._view_combo.addItem("Examples", "examples")
        self._view_combo.addItem("Rejected", "rejected")
        self._view_combo.currentIndexChanged.connect(self._refresh_candidate_list)
        header.addWidget(self._candidate_list_label)
        header.addStretch(1)
        header.addWidget(self._view_combo)
        self._candidate_list = QListWidget(frame)
        self._candidate_list.currentRowChanged.connect(self._on_candidate_row_changed)
        layout.addLayout(header)
        layout.addWidget(self._candidate_list)
        return frame

    def _build_review_panel(self) -> QWidget:
        frame = QFrame(self)
        layout = QVBoxLayout(frame)
        self._active_title_label = QLabel("Candidate", frame)
        self._active_meta_label = QLabel("", frame)
        self._active_score_label = QLabel("", frame)
        self._play_button = QPushButton("Play", frame)
        self._match_button = QPushButton("Sounds Like This", frame)
        self._reject_button = QPushButton("Not This Sound", frame)
        self._skip_button = QPushButton("Unsure", frame)
        self._play_button.clicked.connect(self._play_active)
        self._match_button.clicked.connect(lambda: self._mark_active(ReviewLabel.POSITIVE))
        self._reject_button.clicked.connect(lambda: self._mark_active(ReviewLabel.NEGATIVE))
        self._skip_button.clicked.connect(lambda: self._mark_active(ReviewLabel.SKIPPED))
        layout.addWidget(self._active_title_label)
        layout.addWidget(self._active_meta_label)
        layout.addWidget(self._active_score_label)
        layout.addWidget(self._play_button)
        layout.addWidget(self._match_button)
        layout.addWidget(self._reject_button)
        layout.addWidget(self._skip_button)
        layout.addStretch(1)
        return frame

    def _build_footer(self) -> QWidget:
        frame = QFrame(self)
        layout = QGridLayout(frame)
        self._status_label = QLabel("Ready", frame)
        self._model_status_label = QLabel("Profile updates as you review", frame)
        self._save_model_checkbox = QCheckBox("Remember this match profile", frame)
        self._primary_button = QPushButton("Select Matches", frame)
        self._primary_button.clicked.connect(self._primary_action)
        self._outcome_combo = QComboBox(frame)
        self._outcome_combo.addItem("Select matched events", "select")
        self._outcome_combo.addItem("Promote matched events", "promote")
        self._outcome_combo.addItem("Demote matched events", "demote")
        self._outcome_combo.addItem("Create new layer from matches", "create_layer")
        self._layer_name_edit = QLineEdit("Similar Events", frame)
        layout.addWidget(self._status_label, 0, 0, 1, 2)
        layout.addWidget(self._model_status_label, 1, 0, 1, 2)
        layout.addWidget(self._save_model_checkbox, 0, 2)
        layout.addWidget(self._primary_button, 1, 2)
        layout.addWidget(self._outcome_combo, 0, 3)
        layout.addWidget(self._layer_name_edit, 1, 3)
        return frame

    def _restart_session_for_scope(self) -> None:
        self._session = self._service.start_session(
            presentation=self._presentation,
            layer_id=self._layer_id,
            take_id=self._take_id,
            event_id=self._event_id,
            scope_mode=str(self._scope_combo.currentData()),
            seed_event_refs=self._seed_event_refs,
        )
        self._active_ref = self._session.next_candidate_ref
        self._saved_model_path = None
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._refresh_candidate_list()
        self._refresh_active_panel()
        self._rail._session = self._session
        self._rail._active_ref = self._active_ref
        self._rail.update()
        seed_count = len(self._session.seed_event_refs)
        required = self._session.required_seed_count
        top_count = len(self._session.top_candidates)
        review_count = len(self._session.review_candidates)
        reviewed_in_lane = sum(
            1
            for candidate in self._session.review_candidates
            if candidate.review_label != ReviewLabel.UNKNOWN
        )
        self._candidate_count_label.setText(f"{top_count} shown")
        self._review_count_label.setText(f"{reviewed_in_lane} / {review_count} reviewed")
        hidden_count = max(0, len(self._session.ranked_candidates) - top_count)
        self._status_label.setText(
            f"{seed_count} / {required} matches · "
            f"{self._session.negative_count} not similar · "
            f"{hidden_count} outside top {TOP_CANDIDATE_LIMIT}"
        )
        self._model_status_label.setText(self._session.match_profile.readiness_reason)
        self._primary_button.setText("Select Matches")
        self._primary_button.setEnabled(self._session.can_select_similar)
        self._match_button.setText("Sounds Like This")
        self._reject_button.setText("Not This Sound")
        self._skip_button.setText("Unsure")
        anchor = next(
            (candidate for candidate in self._session.candidates if candidate.is_anchor), None
        )
        self._anchor_label.setText(
            f"Anchor · {anchor.label if anchor is not None else self._event_id}"
        )

    def _refresh_candidate_list(self) -> None:
        self._candidate_list.blockSignals(True)
        self._candidate_list.clear()
        active_row = -1
        candidates = self._ordered_candidates()
        self._candidate_list_label.setText(
            _candidate_view_label(str(self._view_combo.currentData()))
        )
        for index, candidate in enumerate(candidates):
            rank = index + 1
            prefix = _candidate_prefix(candidate, self._session.next_candidate_ref, rank)
            status = _candidate_status(candidate, rank)
            item = QListWidgetItem(
                f"{prefix} {candidate.start_seconds:.2f}s {candidate.label} · "
                f"{candidate.score:.2f} · {status}"
            )
            item.setData(Qt.ItemDataRole.UserRole, candidate.event_ref)
            color = _candidate_color(candidate)
            if not candidate.passes_confidence and not candidate.is_anchor:
                color = TIMELINE_STYLE.ruler.label_hex
            item.setForeground(QColor(color))
            self._candidate_list.addItem(item)
            if self._active_ref is not None and _same_ref(candidate.event_ref, self._active_ref):
                active_row = index
        if active_row >= 0:
            self._candidate_list.setCurrentRow(active_row)
        elif self._candidate_list.count() > 0:
            self._candidate_list.setCurrentRow(0)
            item = self._candidate_list.item(0)
            if item is not None:
                self._active_ref = item.data(Qt.ItemDataRole.UserRole)
        else:
            self._active_ref = None
        self._candidate_list.blockSignals(False)

    def _ordered_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        view_mode = str(self._view_combo.currentData())
        if view_mode == "all_unreviewed":
            return tuple(
                candidate
                for candidate in self._session.ranked_candidates
                if candidate.review_label == ReviewLabel.UNKNOWN
            )
        if view_mode == "examples":
            return tuple(
                candidate
                for candidate in sorted(self._session.candidates, key=lambda row: row.timeline_index)
                if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE
            )
        if view_mode == "rejected":
            return tuple(
                candidate
                for candidate in sorted(self._session.candidates, key=lambda row: row.timeline_index)
                if candidate.review_label == ReviewLabel.NEGATIVE
            )
        return tuple(
            candidate
            for candidate in self._session.top_candidates
            if candidate.review_label == ReviewLabel.UNKNOWN
        )

    def _refresh_active_panel(self) -> None:
        candidate = self._active_candidate()
        if candidate is None:
            self._active_title_label.setText("No candidate")
            self._active_meta_label.setText("")
            self._active_score_label.setText("")
            return
        self._active_title_label.setText(candidate.label)
        self._active_meta_label.setText(
            f"{candidate.start_seconds:.2f}s - {candidate.end_seconds:.2f}s · "
            f"{_label_text(candidate.review_label)}"
        )
        score_label = "Similarity confidence"
        rank = self._candidate_rank(candidate.event_ref)
        rank_label = f"Rank {rank} of {len(self._session.ranked_candidates)}"
        self._active_score_label.setText(f"{rank_label} · {score_label} {candidate.score:.2f}")
        can_review = self._can_review_candidate(candidate)
        self._match_button.setEnabled(can_review)
        self._reject_button.setEnabled(can_review)
        self._skip_button.setEnabled(can_review)

    def _on_candidate_row_changed(self, row: int) -> None:
        item = self._candidate_list.item(row)
        if item is None:
            return
        self._active_ref = item.data(Qt.ItemDataRole.UserRole)
        self._refresh_active_panel()
        self._rail._active_ref = self._active_ref
        self._rail.update()

    def _mark_active(self, label: ReviewLabel) -> None:
        candidate = self._active_candidate()
        if candidate is None or candidate.is_anchor:
            return
        if not self._can_review_candidate(candidate):
            self._model_status_label.setText(
                f"Review is limited to the first {REVIEW_CANDIDATE_LIMIT} candidates"
            )
            return
        self._session = self._service.mark_candidate(self._session, candidate.event_ref, label)
        self._active_ref = self._session.next_candidate_ref or candidate.event_ref
        self._refresh_all()

    def _primary_action(self) -> None:
        if self._session.can_select_similar:
            self.accept()
            return
        self._model_status_label.setText(self._session.match_profile.readiness_reason)

    def _accept_payload(self) -> None:
        self.accept()

    def _play_active(self) -> None:
        candidate = self._active_candidate()
        if candidate is None:
            return
        if self._preview_event_callback is None:
            self._model_status_label.setText("Preview unavailable in this runtime")
            return
        self._preview_event_callback(
            candidate.event_ref.layer_id,
            candidate.event_ref.take_id,
            candidate.event_ref.event_id,
        )

    def _active_candidate(self) -> FindSimilarCandidate | None:
        if self._active_ref is None:
            return None
        return next(
            (
                candidate
                for candidate in self._session.candidates
                if _same_ref(candidate.event_ref, self._active_ref)
            ),
            None,
        )

    def _candidate_rank(self, event_ref: EventRef) -> int:
        for rank, candidate in enumerate(self._session.ranked_candidates, start=1):
            if _same_ref(candidate.event_ref, event_ref):
                return rank
        return 0

    def _can_review_candidate(self, candidate: FindSimilarCandidate) -> bool:
        review_ref_keys = {
            _event_ref_key(review_candidate.event_ref)
            for review_candidate in self._session.review_candidates
        }
        return _event_ref_key(candidate.event_ref) in review_ref_keys

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QDialog {{
                background: {TIMELINE_STYLE.canvas.background_hex};
                color: {TIMELINE_STYLE.object_palette.body_hex};
            }}
            QFrame {{
                background: {TIMELINE_STYLE.canvas.row_fill_hex};
                border: 1px solid {TIMELINE_STYLE.canvas.row_divider_hex};
                border-radius: 3px;
            }}
            QLabel {{
                color: {TIMELINE_STYLE.object_palette.body_hex};
                border: none;
                background: transparent;
            }}
            QPushButton, QComboBox, QLineEdit {{
                background: {TIMELINE_STYLE.object_palette.button_bg_hex};
                color: {TIMELINE_STYLE.object_palette.button_fg_hex};
                border: 1px solid {TIMELINE_STYLE.object_palette.button_border_hex};
                border-radius: 2px;
                padding: 5px 8px;
                min-height: 22px;
            }}
            QListWidget {{
                background: {TIMELINE_STYLE.canvas.background_hex};
                color: {TIMELINE_STYLE.object_palette.body_hex};
                border: 1px solid {TIMELINE_STYLE.canvas.row_divider_hex};
            }}
            QListWidget::item:selected {{
                background: {TIMELINE_STYLE.canvas.selected_row_fill_hex};
                color: {TIMELINE_STYLE.object_palette.title_hex};
            }}
            QCheckBox {{
                color: {TIMELINE_STYLE.object_palette.body_hex};
                border: none;
                background: transparent;
            }}
            """)


class EventComparisonDialog(FindSimilarSoundsDialog):
    """Canonical name for the Find Similar review dialog."""


class FindSimilarShapesDialog(FindSimilarSoundsDialog):
    """Compatibility alias for older tests/imports."""


def _candidate_color(candidate: FindSimilarCandidate) -> str:
    if candidate.is_anchor:
        return TIMELINE_STYLE.playhead.color_hex
    if candidate.review_label == ReviewLabel.POSITIVE:
        return "#7fd1ae"
    if candidate.review_label == ReviewLabel.NEGATIVE:
        return "#a6533e"
    if candidate.review_label == ReviewLabel.SKIPPED:
        return TIMELINE_STYLE.ruler.label_hex
    return TIMELINE_STYLE.object_palette.body_hex


def _candidate_view_label(view_mode: str) -> str:
    return {
        "all_unreviewed": "All Unreviewed",
        "examples": "Examples",
        "rejected": "Rejected",
    }.get(view_mode, "Best Next")


def _candidate_prefix(
    candidate: FindSimilarCandidate,
    next_ref: EventRef | None,
    rank: int,
) -> str:
    if candidate.review_label == ReviewLabel.POSITIVE:
        return f"MATCH {rank:02d}"
    if candidate.review_label == ReviewLabel.NEGATIVE:
        return f"NO {rank:02d}"
    if candidate.review_label == ReviewLabel.SKIPPED:
        return f"SKIP {rank:02d}"
    if next_ref is not None and _same_ref(candidate.event_ref, next_ref):
        return f"NEXT {rank:02d}"
    if rank <= REVIEW_CANDIDATE_LIMIT:
        return f"REVIEW {rank:02d}"
    return f"SCAN {rank:02d}"


def _candidate_status(candidate: FindSimilarCandidate, rank: int) -> str:
    if candidate.review_label == ReviewLabel.POSITIVE:
        return "match"
    if candidate.review_label == ReviewLabel.NEGATIVE:
        return "not similar"
    if candidate.review_label == ReviewLabel.SKIPPED:
        return "unsure"
    if rank <= REVIEW_CANDIDATE_LIMIT:
        return "needs review"
    if candidate.passes_confidence:
        return "queued"
    return "low confidence"


def _label_text(label: ReviewLabel) -> str:
    return {
        ReviewLabel.POSITIVE: "match",
        ReviewLabel.NEGATIVE: "not similar",
        ReviewLabel.SKIPPED: "unsure",
        ReviewLabel.UNKNOWN: "unreviewed",
    }[label]


def _initial_seed_event_refs(
    presentation: TimelinePresentation,
    layer_id: LayerId,
    take_id: TakeId,
    event_id: EventId,
) -> tuple[EventRef, ...]:
    anchor_ref = EventRef(layer_id, take_id, event_id)
    return tuple(
        event_ref
        for event_ref in presentation.resolved_selected_event_refs()
        if not _same_ref(event_ref, anchor_ref)
    )


def _same_ref(left: EventRef, right: EventRef) -> bool:
    return (
        left.layer_id == right.layer_id
        and left.take_id == right.take_id
        and left.event_id == right.event_id
    )


def _event_ref_key(event_ref: EventRef) -> tuple[str, str, str]:
    return (str(event_ref.layer_id), str(event_ref.take_id), str(event_ref.event_id))


def _selected_layer_ids(
    presentation: TimelinePresentation,
    fallback_layer_id: LayerId,
    scope_mode: object,
) -> list[LayerId]:
    if str(scope_mode) == "selected_layers_main":
        return [layer_id for layer_id in presentation.selected_layer_ids if layer_id] or [
            fallback_layer_id
        ]
    return [fallback_layer_id]


__all__ = [
    "CandidateRailWidget",
    "EventComparisonDialog",
    "FindSimilarShapesDialog",
    "FindSimilarSoundsDialog",
    "ShapePreviewRow",
]
