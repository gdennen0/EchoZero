"""
Waveform Widget

Interactive waveform display widget for audio clip preview.
Supports click-to-seek, playhead visualization, and zoom/pan.
"""

import numpy as np
from typing import Optional, List
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath

try:
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

from ..core.style import TimelineStyle as Colors
from ..logging import TimelineLog as Log


class WaveformWidget(QWidget):
    """
    Interactive waveform display widget.
    
    Features:
    - Displays audio waveform as filled path
    - Click to seek
    - Playhead position indicator
    - Zoom/pan support for long clips
    """
    
    # Signal emitted when user clicks to seek
    seek_requested = pyqtSignal(float)  # Position in seconds
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._waveform_data: Optional[np.ndarray] = None  # Shape: (N,) - amplitude values
        self._duration: float = 0.0
        self._playhead_position: float = 0.0  # Position in seconds
        self._zoom_level: float = 1.0  # 1.0 = fit to width, >1.0 = zoomed in
        self._pan_offset: float = 0.0  # Pan offset in seconds
        
        # Visual settings
        self._waveform_color = Colors.ACCENT_BLUE
        self._playhead_color = Colors.PLAYHEAD_COLOR
        self._background_color = Colors.BG_DARK
        self._grid_color = Colors.BORDER
        
        # Interaction state
        self._is_dragging = False
        self._drag_start_x = 0.0
        
        # Set minimum size -- allow compact mode (48px) for inspector panels
        self.setMinimumHeight(48)
        self.setMinimumWidth(100)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Style
        self.setStyleSheet(f"background-color: {self._background_color.name()};")
    
    def set_waveform_data(self, audio_data: np.ndarray, sample_rate: int, duration: float):
        """
        Set waveform data to display.
        
        Args:
            audio_data: Audio samples (1D array, mono)
            sample_rate: Sample rate in Hz
            duration: Duration in seconds
        """
        if not HAS_AUDIO_LIBS:
            Log.warning("WaveformWidget: Audio libraries not available")
            return
        
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Downsample to reasonable resolution for display
            # Target: ~500 points for good visualization
            target_points = 500
            if len(audio_data) > target_points:
                # Downsample using decimation
                step = len(audio_data) // target_points
                audio_data = audio_data[::step]
            
            # Compute RMS envelope for smoother visualization
            # Use small window for better detail
            window_size = max(1, len(audio_data) // 200)
            if window_size > 1:
                # Simple moving average
                kernel = np.ones(window_size) / window_size
                audio_data = np.convolve(np.abs(audio_data), kernel, mode='same')
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            self._waveform_data = audio_data
            self._duration = duration
            
            # Reset zoom/pan when new data is loaded
            self._zoom_level = 1.0
            self._pan_offset = 0.0
            
            self.update()
            Log.debug(f"WaveformWidget: Set waveform data ({len(audio_data)} points, {duration:.3f}s)")
            
        except Exception as e:
            Log.error(f"WaveformWidget: Failed to process waveform data: {e}")
            self._waveform_data = None
    
    def set_playhead_position(self, seconds: float):
        """
        Set playhead position.
        
        Args:
            seconds: Position in seconds (0 to duration)
        """
        self._playhead_position = max(0.0, min(seconds, self._duration))
        self.update()
    
    def set_zoom(self, level: float):
        """
        Set zoom level.
        
        Args:
            level: Zoom level (1.0 = fit to width, >1.0 = zoomed in)
        """
        self._zoom_level = max(1.0, level)
        self.update()
    
    def set_pan(self, offset_seconds: float):
        """
        Set pan offset.
        
        Args:
            offset_seconds: Pan offset in seconds
        """
        self._pan_offset = max(0.0, offset_seconds)
        self.update()
    
    def clear(self):
        """Clear waveform data."""
        self._waveform_data = None
        self._duration = 0.0
        self._playhead_position = 0.0
        self.update()
    
    def paintEvent(self, event):
        """Paint the waveform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Draw background
        painter.fillRect(rect, QBrush(self._background_color))
        
        if self._waveform_data is None or len(self._waveform_data) == 0 or self._duration <= 0:
            # Draw placeholder text
            painter.setPen(QPen(Colors.TEXT_DISABLED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No waveform data")
            return
        
        # Calculate visible time range based on zoom and pan
        visible_duration = self._duration / self._zoom_level
        start_time = self._pan_offset
        end_time = min(start_time + visible_duration, self._duration)
        
        # Draw grid lines (optional, subtle)
        self._draw_grid(painter, rect, start_time, end_time)
        
        # Draw waveform
        self._draw_waveform(painter, rect, start_time, end_time)
        
        # Draw playhead
        if start_time <= self._playhead_position <= end_time:
            self._draw_playhead(painter, rect, start_time, end_time)
    
    def _draw_grid(self, painter: QPainter, rect: QRectF, start_time: float, end_time: float):
        """Draw subtle grid lines."""
        if end_time - start_time <= 0:
            return
        
        painter.setPen(QPen(self._grid_color, 1))
        
        # Draw center line
        center_y = rect.height() / 2
        painter.drawLine(int(rect.left()), int(center_y), int(rect.right()), int(center_y))
        
        # Draw time markers (every 0.1s if visible)
        if end_time - start_time < 1.0:
            step = 0.1
            time = start_time
            while time <= end_time:
                x = self._time_to_x(rect, time, start_time, end_time)
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
                time += step
    
    def _draw_waveform(self, painter: QPainter, rect: QRectF, start_time: float, end_time: float):
        """Draw the waveform as a filled path."""
        if self._waveform_data is None or len(self._waveform_data) == 0:
            return
        
        duration = end_time - start_time
        if duration <= 0:
            return
        
        # Calculate sample indices for visible range
        total_samples = len(self._waveform_data)
        start_idx = int((start_time / self._duration) * total_samples)
        end_idx = int((end_time / self._duration) * total_samples)
        start_idx = max(0, min(start_idx, total_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_samples))
        
        visible_samples = self._waveform_data[start_idx:end_idx]
        num_samples = len(visible_samples)
        
        if num_samples == 0:
            return
        
        # Create path for waveform
        path = QPainterPath()
        center_y = rect.height() / 2
        max_amplitude = rect.height() / 2 - 2  # Tight padding for compact mode
        
        # Draw top half (positive)
        path.moveTo(rect.left(), center_y)
        for i, amplitude in enumerate(visible_samples):
            x = rect.left() + (i / (num_samples - 1)) * rect.width() if num_samples > 1 else rect.left()
            y = center_y - amplitude * max_amplitude
            path.lineTo(x, y)
        
        # Draw bottom half (mirrored)
        for i in range(num_samples - 1, -1, -1):
            amplitude = abs(visible_samples[i])
            x = rect.left() + (i / (num_samples - 1)) * rect.width() if num_samples > 1 else rect.left()
            y = center_y + amplitude * max_amplitude
            path.lineTo(x, y)
        
        path.closeSubpath()
        
        # Fill waveform
        fill_color = QColor(self._waveform_color)
        fill_color.setAlpha(180)  # Semi-transparent
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(self._waveform_color, 1))
        painter.drawPath(path)
    
    def _draw_playhead(self, painter: QPainter, rect: QRectF, start_time: float, end_time: float):
        """Draw playhead indicator."""
        x = self._time_to_x(rect, self._playhead_position, start_time, end_time)
        
        painter.setPen(QPen(self._playhead_color, 1))
        painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
    
    def _time_to_x(self, rect: QRectF, time: float, start_time: float, end_time: float) -> float:
        """Convert time to X coordinate."""
        if end_time - start_time <= 0:
            return rect.left()
        
        progress = (time - start_time) / (end_time - start_time)
        return rect.left() + progress * rect.width()
    
    def _x_to_time(self, rect: QRectF, x: float, start_time: float, end_time: float) -> float:
        """Convert X coordinate to time."""
        if rect.width() <= 0:
            return start_time
        
        progress = (x - rect.left()) / rect.width()
        return start_time + progress * (end_time - start_time)
    
    def mousePressEvent(self, event):
        """Handle mouse press - start seek drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_start_x = event.position().x()
            self._seek_to_position(event.position().x())
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move - continue seek drag."""
        if self._is_dragging:
            self._seek_to_position(event.position().x())
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - end seek drag."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def _seek_to_position(self, x: float):
        """Seek to position based on X coordinate."""
        if self._duration <= 0:
            return
        
        rect = self.rect()
        visible_duration = self._duration / self._zoom_level
        start_time = self._pan_offset
        end_time = min(start_time + visible_duration, self._duration)
        
        time = self._x_to_time(rect, x, start_time, end_time)
        time = max(0.0, min(time, self._duration))
        
        self._playhead_position = time
        self.seek_requested.emit(time)
        self.update()





