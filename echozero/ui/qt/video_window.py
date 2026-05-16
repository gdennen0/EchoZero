"""
Separate Qt video reference window synced to the song timeline transport.
Exists because video display is a reference surface, not a mixed timeline audio track.
Connects timeline presentation and transport intents to Qt Multimedia playback.
"""

from __future__ import annotations

import subprocess

from PyQt6.QtCore import QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QResizeEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink
from PyQt6.QtWidgets import QMainWindow, QWidget

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.video import VideoTimelineMapping, video_mapping_from_presentation


class _VideoFrameWidget(QWidget):
    """Paints decoded video frames into a normal QWidget surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
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

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#050505"))
            if self._image.isNull():
                painter.setPen(QPen(QColor("#7f7f7f"), 1))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Video Reference")
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

    def __init__(self) -> None:
        self._window = QMainWindow()
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
        self._mapping: VideoTimelineMapping | None = None
        self._loaded_path: str | None = None
        self._seek_frame_cache: dict[int, QImage] = {}

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
        self._mapping = mapping
        if mapping is None:
            self._player.stop()
            self._loaded_path = None
            return
        if mapping.video_path != self._loaded_path:
            self._loaded_path = mapping.video_path
            self._seek_frame_cache.clear()
            self._player.setSource(QUrl.fromLocalFile(mapping.video_path))
        self.seek(float(presentation.playhead))

    def play(self, song_seconds: float) -> None:
        """Start video playback if the song playhead is inside the video range."""

        if self._mapping is None:
            return
        self.seek(song_seconds)
        if self._mapping.contains_song_time(song_seconds):
            self._player.play()
        else:
            self._player.pause()

    def pause(self, song_seconds: float) -> None:
        """Pause video playback after seeking to the song playhead."""

        self.seek(song_seconds)
        self._player.pause()

    def stop(self) -> None:
        """Pause video playback at the song timeline zero mapping."""

        self.seek(0.0)
        self._player.pause()

    def seek(self, song_seconds: float) -> None:
        """Seek video media to the mapped song timeline position."""

        if self._mapping is None:
            return
        media_seconds = self._mapping.media_seconds_for_song_time(song_seconds)
        media_ms = int(round(media_seconds * 1000.0))
        if abs(int(self._player.position()) - media_ms) > 35:
            self._player.setPosition(media_ms)
        self._show_seek_frame(media_seconds)

    def _show_seek_frame(self, media_seconds: float) -> None:
        if not self._loaded_path:
            return
        cache_key = int(round(max(0.0, float(media_seconds)) * 10.0))
        cached = self._seek_frame_cache.get(cache_key)
        if cached is not None:
            self._video_widget.set_image(cached)
            return
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(0.0, float(media_seconds)):.3f}",
                    "-i",
                    self._loaded_path,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=3.0,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return
        image = QImage()
        if not image.loadFromData(completed.stdout, "PNG"):
            return
        if len(self._seek_frame_cache) > 24:
            self._seek_frame_cache.clear()
        self._seek_frame_cache[cache_key] = image
        self._video_widget.set_image(image)
