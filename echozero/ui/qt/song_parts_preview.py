"""
Song-parts preview widget: live 2D structure map for the extract-song-parts settings flow.
Exists so operators can inspect merged structure evidence before running a section detector.
Connects object-action session values to an application preview service and a custom painted graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QTimer, Qt, QSize
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from echozero.application.timeline.object_actions import ObjectActionSettingsSession
from echozero.application.timeline.song_parts_preview_service import (
    SongPartsPreviewData,
    SongPartsPreviewPoint,
    SongPartsPreviewSegment,
    build_song_parts_preview,
)


_SECTION_COLORS = (
    QColor("#7fd1ae"),
    QColor("#65b7ff"),
    QColor("#f1c75b"),
    QColor("#ff8b7b"),
    QColor("#b79cff"),
    QColor("#57d0d9"),
)


@dataclass(frozen=True)
class _PreviewRequest:
    source_audio_path: str
    settings: dict[str, object]


class SongPartsStructureGraph(QFrame):
    """Paint a merged song-structure vector space with section overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("songPartsStructureGraph")
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview: SongPartsPreviewData | None = None
        self._message = "Preview unavailable."

    def sizeHint(self) -> QSize:
        return QSize(520, 320)

    def set_preview(self, preview: SongPartsPreviewData | None, *, message: str = "") -> None:
        self._preview = preview
        self._message = message
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 8, -8, -8)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#161c21"))
        painter.drawRoundedRect(rect, 12.0, 12.0)
        self._paint_grid(painter, rect)

        if self._preview is None or not self._preview.points:
            painter.setPen(QColor("#91a0ad"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._message)
            return

        self._paint_path(painter, rect, self._preview.points)
        self._paint_points(painter, rect, self._preview.points)
        self._paint_segment_labels(painter, rect, self._preview.points, self._preview.segments)

    def _paint_grid(self, painter: QPainter, rect) -> None:
        painter.setPen(QPen(QColor("#26313a"), 1.0))
        for fraction in (0.2, 0.4, 0.6, 0.8):
            x = rect.left() + int(rect.width() * fraction)
            y = rect.top() + int(rect.height() * fraction)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            painter.drawLine(rect.left(), y, rect.right(), y)
        painter.setPen(QPen(QColor("#31404b"), 1.0))
        painter.drawLine(rect.center().x(), rect.top(), rect.center().x(), rect.bottom())
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())

    def _paint_path(self, painter: QPainter, rect, points: tuple[SongPartsPreviewPoint, ...]) -> None:
        if len(points) <= 1:
            return
        for point_a, point_b in zip(points, points[1:]):
            pen = QPen(self._section_color(point_a.segment_index), 1.8)
            pen.setCosmetic(True)
            color = pen.color()
            color.setAlphaF(0.42 + (0.30 * point_a.repetition))
            pen.setColor(color)
            painter.setPen(pen)
            painter.drawLine(self._graph_point(rect, point_a), self._graph_point(rect, point_b))

    def _paint_points(self, painter: QPainter, rect, points: tuple[SongPartsPreviewPoint, ...]) -> None:
        for point in points:
            center = self._graph_point(rect, point)
            radius = 2.6 + (3.6 * point.novelty)
            fill = self._section_color(point.segment_index)
            fill.setAlphaF(0.22 + (0.58 * point.repetition))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawEllipse(center, radius, radius)
            if point.is_boundary:
                ring_pen = QPen(self._section_color(point.segment_index).lighter(130), 2.0)
                ring_pen.setCosmetic(True)
                painter.setPen(ring_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, radius + 3.0, radius + 3.0)

    def _paint_segment_labels(
        self,
        painter: QPainter,
        rect,
        points: tuple[SongPartsPreviewPoint, ...],
        segments: tuple[SongPartsPreviewSegment, ...],
    ) -> None:
        font_metrics = QFontMetrics(painter.font())
        for segment_index, segment in enumerate(segments):
            segment_points = [
                point for point in points if point.segment_index == segment_index
            ]
            if not segment_points:
                continue
            centroid_x = sum(point.x for point in segment_points) / len(segment_points)
            centroid_y = sum(point.y for point in segment_points) / len(segment_points)
            label_text = f"{segment.label} {segment.confidence:.2f}"
            text_width = font_metrics.horizontalAdvance(label_text)
            pill_width = text_width + 16
            pill_height = max(20, font_metrics.height() + 8)
            center = self._graph_xy(rect, centroid_x, centroid_y)
            left = max(rect.left() + 6, min(int(center.x() - (pill_width / 2)), rect.right() - pill_width - 6))
            top = max(rect.top() + 6, min(int(center.y() - pill_height - 8), rect.bottom() - pill_height - 6))
            pill_rect = rect.adjusted(0, 0, 0, 0)
            pill_rect.setLeft(left)
            pill_rect.setTop(top)
            pill_rect.setWidth(pill_width)
            pill_rect.setHeight(pill_height)
            fill = self._section_color(segment_index)
            fill.setAlpha(215)
            painter.setPen(QPen(fill.lighter(135), 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(pill_rect, 10.0, 10.0)
            painter.setPen(QColor("#0f1418"))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, label_text)

    @staticmethod
    def _graph_point(rect, point: SongPartsPreviewPoint) -> QPointF:
        return SongPartsStructureGraph._graph_xy(rect, point.x, point.y)

    @staticmethod
    def _graph_xy(rect, x_value: float, y_value: float) -> QPointF:
        x = rect.left() + ((x_value + 1.0) * 0.5 * rect.width())
        y = rect.bottom() - ((y_value + 1.0) * 0.5 * rect.height())
        return QPointF(float(x), float(y))

    @staticmethod
    def _section_color(index: int) -> QColor:
        base = _SECTION_COLORS[index % len(_SECTION_COLORS)]
        return QColor(base)


class SongPartsPreviewPanel(QFrame):
    """Reusable live preview panel for the extract-song-parts dialogs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("songPartsPreviewPanel")
        self.setProperty("section", True)
        self._session: ObjectActionSettingsSession | None = None
        self._cache: dict[tuple[object, ...], SongPartsPreviewData] = {}
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_preview)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._title = QLabel("Song Parts Preview", self)
        self._title.setObjectName("songPartsPreviewTitle")
        layout.addWidget(self._title)

        self._summary = QLabel(
            "Preview unavailable until a source song audio file is resolved.",
            self,
        )
        self._summary.setObjectName("songPartsPreviewSummary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._graph = SongPartsStructureGraph(self)
        layout.addWidget(self._graph, 1)

        self._detail = QLabel("", self)
        self._detail.setObjectName("songPartsPreviewDetail")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        self.hide()

    def set_session(self, session: ObjectActionSettingsSession | None) -> None:
        self._session = session
        is_song_parts = session is not None and session.action_id == "timeline.extract_song_sections"
        self.setVisible(is_song_parts)
        if not is_song_parts:
            self._refresh_timer.stop()
            return
        self._summary.setText("Analyzing structure preview...")
        self._detail.setText("")
        self._graph.set_preview(None, message="Analyzing structure preview...")
        self._refresh_timer.start(180)

    def _refresh_preview(self) -> None:
        request = self._preview_request()
        if request is None:
            self._summary.setText("Preview unavailable until the source song audio layer is resolved.")
            self._detail.setText("")
            self._graph.set_preview(None, message="Source song audio is required for preview.")
            return
        cache_key = _preview_cache_key(request)
        preview = self._cache.get(cache_key)
        if preview is None:
            try:
                preview = build_song_parts_preview(
                    source_audio_path=request.source_audio_path,
                    settings=request.settings,
                )
            except Exception as exc:
                self._summary.setText("Unable to build the song-parts structure preview.")
                self._detail.setText(str(exc))
                self._graph.set_preview(None, message="Preview generation failed.")
                return
            self._cache[cache_key] = preview
        self._summary.setText(preview.summary_text)
        self._detail.setText(_preview_detail_text(preview))
        self._graph.set_preview(preview, message="")

    def _preview_request(self) -> _PreviewRequest | None:
        session = self._session
        if session is None:
            return None
        runtime_bindings = dict(session.plan.runtime_bindings)
        source_audio_path = str(runtime_bindings.get("audio_file", "")).strip()
        if not source_audio_path:
            locked = dict(session.plan.locked_bindings)
            source_audio_path = str(locked.get("audio_file", "")).strip()
        if not source_audio_path:
            return None
        relevant_settings = {
            key: value
            for key, value in session.values.items()
            if key
            in {
                "detect_method",
                "sample_rate",
                "n_mfcc",
                "n_fft",
                "hop_length",
                "history_pool_frames",
                "boundary_sensitivity",
                "min_section_seconds",
                "max_sections",
                "similarity_threshold",
                "intro_tail_seconds",
                "end_tail_seconds",
            }
        }
        return _PreviewRequest(
            source_audio_path=source_audio_path,
            settings=relevant_settings,
        )


def _preview_cache_key(request: _PreviewRequest) -> tuple[object, ...]:
    return (
        request.source_audio_path,
        *sorted((key, _freeze_value(value)) for key, value in request.settings.items()),
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    return value


def _preview_detail_text(preview: SongPartsPreviewData) -> str:
    segment_summary = " · ".join(
        f"{segment.label} ({segment.start_seconds:.1f}s-{segment.end_seconds:.1f}s)"
        for segment in preview.segments[:6]
    )
    if len(preview.segments) > 6:
        segment_summary = f"{segment_summary} · +{len(preview.segments) - 6} more"
    return f"{preview.detail_text}\n{segment_summary}".strip()
