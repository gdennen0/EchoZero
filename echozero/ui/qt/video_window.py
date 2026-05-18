"""
Separate Qt video reference window synced to the song timeline transport.
Exists because video display is a reference surface, not a mixed timeline audio track.
Connects timeline presentation and transport intents to Qt Multimedia playback.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QResizeEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink
from PyQt6.QtWidgets import QMainWindow, QWidget

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.video import (
    VideoClockSync,
    VideoTimelineMapping,
    video_mapping_from_presentation,
)


class _VideoWindow(QMainWindow):
    """Top-level reference-video window with an explicit close callback."""

    def __init__(self, on_closed: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._on_closed = on_closed

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().closeEvent(event)
        if self._on_closed is not None:
            self._on_closed()


class _VideoFrameWidget(QWidget):
    """Paints decoded video frames into a normal QWidget surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._status_text = "Video Reference"
        self._status_is_error = False
        self.setMinimumSize(480, 270)
        self.setAutoFillBackground(False)

    def set_frame(self, frame: QVideoFrame) -> None:
        """Store the latest decoded frame and schedule a repaint."""

        self.set_image(frame.toImage())

    def set_image(self, image: QImage) -> None:
        """Store a rendered video image and schedule a repaint."""

        if image.isNull():
            return
        self._image = image
        self.update()

    def set_status(self, text: str, *, is_error: bool = False) -> None:
        """Show a video status message when no decoded frame is available."""

        self._status_text = str(text or "Video Reference")
        self._status_is_error = bool(is_error)
        if is_error:
            self._image = QImage()
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#050505"))
            if self._image.isNull():
                color = QColor("#d88939") if self._status_is_error else QColor("#7f7f7f")
                painter.setPen(QPen(color, 1))
                painter.drawText(
                    self.rect().adjusted(24, 24, -24, -24),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                    self._status_text,
                )
                return
            target = self._scaled_target_rect(self._image.width(), self._image.height())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(target, self._image)
        finally:
            painter.end()

    def resizeEvent(self, event: QResizeEvent | None) -> None:  # noqa: N802
        del event
        self.update()

    def _scaled_target_rect(self, image_width: int, image_height: int) -> QRectF:
        if image_width <= 0 or image_height <= 0:
            return QRectF(self.rect())
        widget_width = max(1, self.width())
        widget_height = max(1, self.height())
        image_aspect = float(image_width) / float(image_height)
        widget_aspect = float(widget_width) / float(widget_height)
        if widget_aspect > image_aspect:
            height = float(widget_height)
            width = height * image_aspect
            left = (float(widget_width) - width) * 0.5
            top = 0.0
        else:
            width = float(widget_width)
            height = width / image_aspect
            left = 0.0
            top = (float(widget_height) - height) * 0.5
        return QRectF(left, top, width, height)


class VideoPlaybackController:
    """Small QMediaPlayer wrapper that treats the EZ timeline as clock authority."""

    def __init__(self, on_closed: Callable[[], None] | None = None) -> None:
        self._window = _VideoWindow(on_closed)
        self._window.setWindowTitle("EchoZero Video Reference")
        self._video_widget = _VideoFrameWidget(self._window)
        self._window.setCentralWidget(self._video_widget)
        self._player = QMediaPlayer(self._window)
        self._audio_output = QAudioOutput(self._window)
        self._audio_output.setVolume(0.0)
        self._video_sink = QVideoSink(self._window)
        self._video_sink.videoFrameChanged.connect(self._video_widget.set_frame)
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoSink(self._video_sink)
        self._player.errorOccurred.connect(self._on_error)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._mapping: VideoTimelineMapping | None = None
        self._loaded_path: str | None = None
        self._clock_sync = VideoClockSync()
        self._error_text = ""
        self._media_status_text = ""

    @property
    def error_text(self) -> str:
        """Return the latest Qt media error text, if any."""

        return self._error_text

    @property
    def media_status_text(self) -> str:
        """Return the latest Qt media status text."""

        return self._media_status_text

    def show(self) -> None:
        """Show the video reference window."""

        self._window.resize(960, 540)
        self._window.show()
        self._window.raise_()

    def close(self) -> None:
        """Stop playback and close the video reference window."""

        self._player.stop()
        self._window.close()

    def sync_presentation(self, presentation: TimelinePresentation) -> None:
        """Load and seek the video reference represented by a timeline presentation."""

        mapping = video_mapping_from_presentation(presentation)
        previous_mapping = self._mapping
        self._mapping = mapping
        if mapping is None:
            self._player.stop()
            self._loaded_path = None
            self._video_widget.set_status("No video reference")
            return
        mapping_changed = mapping != previous_mapping
        if mapping.video_path != self._loaded_path:
            self._loaded_path = mapping.video_path
            self._error_text = ""
            self._video_widget.set_status("Loading video reference")
            self._player.setSource(QUrl.fromLocalFile(mapping.video_path))
            mapping_changed = True
        if mapping_changed:
            self.seek(float(presentation.playhead))

    def play(self, song_seconds: float) -> None:
        """Start video playback if the song playhead is inside the video range."""

        self.update(song_seconds, True)

    def pause(self, song_seconds: float) -> None:
        """Pause video playback after seeking to the song playhead."""

        self.update(song_seconds, False)

    def stop(self) -> None:
        """Pause video playback at the song timeline zero mapping."""

        self.update(0.0, False)

    def seek(self, song_seconds: float) -> None:
        """Seek video media to the mapped song timeline position."""

        if self._mapping is None:
            return
        media_seconds = self._mapping.media_seconds_for_song_time(song_seconds)
        media_ms = int(round(media_seconds * 1000.0))
        if abs(int(self._player.position()) - media_ms) > 35:
            self._player.setPosition(media_ms)

    def update(self, song_seconds: float, audio_is_playing: bool) -> None:
        """Slave video playback to one sampled song transport clock value."""

        decision = self._clock_sync.decision(
            self._mapping,
            song_seconds=float(song_seconds),
            audio_is_playing=bool(audio_is_playing),
            media_seconds=float(self._player.position()) / 1000.0,
        )
        media_ms = int(round(decision.media_seconds * 1000.0))
        if decision.should_seek:
            self._player.setPosition(media_ms)
        playback_state = self._player.playbackState()
        if decision.should_play:
            if playback_state != QMediaPlayer.PlaybackState.PlayingState:
                self._player.play()
        elif playback_state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()

    def _on_error(self, *args: object) -> None:
        text = ""
        if args:
            text = str(args[-1] or "").strip()
        if not text:
            text = str(self._player.errorString() or "").strip()
        if not text:
            text = "Video could not be decoded by Qt Multimedia."
        self._error_text = text
        self._video_widget.set_status(text, is_error=True)

    def _on_media_status_changed(self, status: object) -> None:
        self._media_status_text = str(status)
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            text = str(self._player.errorString() or "").strip() or "Invalid video media"
            self._error_text = text
            self._video_widget.set_status(text, is_error=True)
        elif status == QMediaPlayer.MediaStatus.NoMedia:
            self._video_widget.set_status("No video reference")
        elif status == QMediaPlayer.MediaStatus.LoadingMedia:
            self._video_widget.set_status("Loading video reference")
