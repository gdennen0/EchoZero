"""
Audio Player Block Item

Custom BlockItem subclass with embedded playback controls
(play/pause, timeline slider, audio source selector) rendered
directly inside the node editor via QGraphicsProxyWidget.
"""
from typing import Optional, List, TYPE_CHECKING
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSlider, QLabel, QListWidget, QListWidgetItem,
    QGraphicsProxyWidget, QSizePolicy, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush

from ui.qt_gui.node_editor.block_item import BlockItem
from ui.qt_gui.design_system import Colors, Sizes, Spacing, Typography
from src.utils.message import Log

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack
    from src.features.blocks.domain import Block
    from src.application.api.application_facade import ApplicationFacade


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


class WaveformWidget(QWidget):
    """
    Waveform visualization widget for audio data.

    Displays a mirrored amplitude envelope computed from the audio file.
    Shows a playhead indicator that tracks the current playback position.
    Uses downsampled peak bins for efficient rendering at any width.
    """

    WAVEFORM_HEIGHT = 80
    NUM_BINS = 512  # Fixed bin count, scaled to widget width during paint
    MAX_ZOOM = 50.0  # Maximum x-axis zoom level

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peaks = []  # List of (min_val, max_val) normalized -1..1
        self._position_ratio = 0.0
        self._file_path = None

        # Zoom / pan state
        self._zoom = 1.0  # 1.0 = full view, higher = zoomed in
        self._view_start = 0.0  # Start of visible window (0.0 - 1.0)
        self._dragging_pan = False
        self._pan_start_x = 0.0
        self._pan_start_offset = 0.0

        self.setFixedHeight(self.WAVEFORM_HEIGHT)
        self.setMinimumWidth(100)

        # Zoom buttons overlaid on top-right corner
        self._zoom_out_btn = QPushButton("\u2212", self)  # minus sign
        self._zoom_out_btn.setFixedSize(20, 16)
        self._zoom_out_btn.clicked.connect(self._zoom_out)

        self._zoom_in_btn = QPushButton("+", self)
        self._zoom_in_btn.setFixedSize(20, 16)
        self._zoom_in_btn.clicked.connect(self._zoom_in)

        self._apply_zoom_btn_style()
        self._position_zoom_buttons()
        self._update_zoom_buttons()

    def load_audio(self, file_path: str):
        """Load audio file and compute waveform peak data."""
        if file_path == self._file_path and self._peaks:
            return  # Already loaded

        self._file_path = file_path
        self._peaks = []
        self._position_ratio = 0.0
        self._zoom = 1.0
        self._view_start = 0.0
        self._update_zoom_buttons()

        try:
            import soundfile as sf
            import numpy as np

            data, _sr = sf.read(file_path, dtype='float32')

            # Convert to mono if multi-channel
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            num_bins = self.NUM_BINS
            if len(data) == 0:
                self.update()
                return

            bin_size = len(data) / num_bins
            peaks = []
            for i in range(num_bins):
                start = int(i * bin_size)
                end = min(int((i + 1) * bin_size), len(data))
                chunk = data[start:end]
                if len(chunk) == 0:
                    peaks.append((0.0, 0.0))
                else:
                    peaks.append((float(chunk.min()), float(chunk.max())))

            self._peaks = peaks
        except ImportError:
            Log.warning("WaveformWidget: soundfile not available, cannot render waveform")
        except Exception as e:
            Log.warning(f"WaveformWidget: Error loading audio: {e}")

        self.update()

    def set_position(self, ratio: float):
        """Update playhead position (0.0 to 1.0)."""
        ratio = max(0.0, min(1.0, ratio))
        if abs(ratio - self._position_ratio) > 0.002:
            self._position_ratio = ratio
            self.update()

    def clear(self):
        """Clear waveform data, reset playhead and zoom."""
        self._peaks = []
        self._position_ratio = 0.0
        self._file_path = None
        self._zoom = 1.0
        self._view_start = 0.0
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_zoom_buttons()
        self.update()

    # ------------------------------------------------------------------
    # Zoom & Pan
    # ------------------------------------------------------------------

    def reset_zoom(self):
        """Reset to full waveform view."""
        self._zoom = 1.0
        self._view_start = 0.0
        self._update_cursor()
        self._update_zoom_buttons()
        self.update()

    def _clamp_view(self):
        """Keep the visible window within valid bounds."""
        view_width = 1.0 / self._zoom
        self._view_start = max(0.0, min(1.0 - view_width, self._view_start))

    def _update_cursor(self):
        """Update mouse cursor based on zoom state."""
        if self._zoom > 1.01:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _zoom_in(self):
        """Zoom in on the center of the current view."""
        if not self._peaks:
            return
        center = self._view_start + 0.5 / self._zoom
        new_zoom = min(self.MAX_ZOOM, self._zoom * 1.5)
        new_view_width = 1.0 / new_zoom
        self._zoom = new_zoom
        self._view_start = center - new_view_width / 2
        self._clamp_view()
        self._update_cursor()
        self._update_zoom_buttons()
        self.update()

    def _zoom_out(self):
        """Zoom out from the center of the current view."""
        if not self._peaks:
            return
        center = self._view_start + 0.5 / self._zoom
        new_zoom = max(1.0, self._zoom / 1.5)
        new_view_width = 1.0 / new_zoom
        self._zoom = new_zoom
        self._view_start = center - new_view_width / 2
        self._clamp_view()
        self._update_cursor()
        self._update_zoom_buttons()
        self.update()

    def _update_zoom_buttons(self):
        """Enable/disable zoom buttons based on current zoom level."""
        self._zoom_out_btn.setEnabled(self._zoom > 1.01)
        self._zoom_in_btn.setEnabled(self._zoom < self.MAX_ZOOM - 0.1)

    def _apply_zoom_btn_style(self):
        """Style the zoom buttons as small overlaid controls."""
        bg = Colors.BG_DARK
        bg_hover = Colors.BG_LIGHT
        text = Colors.TEXT_SECONDARY.name()
        text_hover = Colors.TEXT_PRIMARY.name()
        text_off = Colors.TEXT_DISABLED.name()
        border = Colors.BORDER.name()

        style = f"""
            QPushButton {{
                background: rgba({bg.red()}, {bg.green()}, {bg.blue()}, 200);
                color: {text};
                border: 1px solid {border};
                font-size: 12px;
                font-weight: bold;
                padding: 0px;
                min-width: 0px;
            }}
            QPushButton:hover {{
                background: rgba({bg_hover.red()}, {bg_hover.green()}, {bg_hover.blue()}, 230);
                color: {text_hover};
            }}
            QPushButton:disabled {{
                color: {text_off};
                background: rgba({bg.red()}, {bg.green()}, {bg.blue()}, 120);
            }}
        """
        self._zoom_in_btn.setStyleSheet(style)
        self._zoom_out_btn.setStyleSheet(style)

    def _position_zoom_buttons(self):
        """Position zoom buttons in the top-right corner of the waveform."""
        w = self.width() if self.width() > 0 else 338
        self._zoom_out_btn.move(w - 44, 3)
        self._zoom_in_btn.move(w - 22, 3)

    def resizeEvent(self, event):
        """Reposition zoom buttons when widget is resized."""
        super().resizeEvent(event)
        self._position_zoom_buttons()

    def mousePressEvent(self, event):
        """Start panning when zoomed in."""
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.01:
            self._dragging_pan = True
            self._pan_start_x = event.position().x()
            self._pan_start_offset = self._view_start
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Pan the visible window while dragging."""
        if self._dragging_pan:
            margin_x = 2
            usable_w = self.width() - 2 * margin_x
            if usable_w > 0:
                dx = event.position().x() - self._pan_start_x
                view_width = 1.0 / self._zoom
                delta_ratio = -(dx / usable_w) * view_width
                self._view_start = self._pan_start_offset + delta_ratio
                self._clamp_view()
                self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End panning."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_pan:
            self._dragging_pan = False
            self._update_cursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom to full view."""
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.01:
            self.reset_zoom()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        """Paint the waveform visualization with amplitude-based coloring."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cy = h / 2.0

        # Background with border
        painter.setPen(QPen(Colors.BORDER, 1.0))
        painter.setBrush(QBrush(Colors.BG_DARK))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._peaks:
            # Placeholder text
            painter.setPen(QPen(Colors.TEXT_DISABLED))
            font = QFont()
            font.setPixelSize(10)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "No audio loaded"
            )
            painter.end()
            return

        # Faint center line
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1.0))
        painter.drawLine(QPointF(2, cy), QPointF(w - 2, cy))

        # Draw waveform bars (respecting zoom / pan)
        accent = Colors.BLOCK_PLAYER
        margin_x = 2
        margin_y = 4
        usable_w = w - 2 * margin_x
        half_h = (h - 2 * margin_y) / 2.0

        num_peaks = len(self._peaks)
        view_width = 1.0 / self._zoom
        start_frac = self._view_start
        end_frac = start_frac + view_width

        # Map visible window to peak indices
        start_idx = max(0, int(start_frac * num_peaks))
        end_idx = min(num_peaks, int(end_frac * num_peaks) + 1)
        visible_peaks = self._peaks[start_idx:end_idx]
        num_visible = len(visible_peaks)

        if num_visible > 0:
            x_scale = usable_w / num_visible
            bar_width = max(1.0, x_scale)

            painter.setPen(Qt.PenStyle.NoPen)
            for i, (min_v, max_v) in enumerate(visible_peaks):
                x = margin_x + i * x_scale
                if x > w - margin_x:
                    break

                y_top = cy - max_v * half_h
                y_bot = cy - min_v * half_h

                # Ensure minimum visible bar height
                if y_bot - y_top < 0.5:
                    y_top = cy - 0.25
                    y_bot = cy + 0.25

                # Amplitude-based alpha for visual depth
                amp = max(abs(min_v), abs(max_v))
                alpha = int(80 + 175 * amp)
                color = QColor(accent.red(), accent.green(), accent.blue(), min(255, alpha))

                painter.setBrush(QBrush(color))
                painter.drawRect(QRectF(x, y_top, bar_width, y_bot - y_top))

        # Playhead indicator (only if visible in current view)
        if view_width > 0 and start_frac <= self._position_ratio <= end_frac:
            relative_pos = (self._position_ratio - start_frac) / view_width
            px = margin_x + relative_pos * usable_w
            painter.setPen(QPen(Colors.TEXT_PRIMARY, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(QPointF(px, 2), QPointF(px, h - 2))

        # Zoom overview bar (shows visible range when zoomed in)
        if self._zoom > 1.05:
            bar_h = 2
            bar_y = h - bar_h - 1
            # Full extent (dim)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 20)))
            painter.drawRect(QRectF(margin_x, bar_y, usable_w, bar_h))
            # Visible portion (bright accent)
            vis_x = margin_x + start_frac * usable_w
            vis_w = max(2.0, view_width * usable_w)
            painter.setBrush(QBrush(QColor(
                accent.red(), accent.green(), accent.blue(), 150
            )))
            painter.drawRect(QRectF(vis_x, bar_y, vis_w, bar_h))

        painter.end()


class AudioPlayerWidget(QWidget):
    """
    Compact audio player widget designed to be embedded inside a node.

    Contains:
    - Source list (QListWidget) showing all available audio items
    - Waveform visualization of the selected audio
    - Play/pause button
    - Timeline slider
    - Time label (current / total)
    """

    def __init__(self, block_id: str, facade: "ApplicationFacade", parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade
        self._player = None
        self._is_playing = False
        self._duration = 0.0
        self._current_audio_items: List = []
        self._current_source_index: int = -1

        self.setFixedWidth(Sizes.PLAYER_BLOCK_WIDTH - 12)
        self._build_ui()
        self._init_player()
        self._start_update_timer()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the compact player layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # --- Source list ---
        # QListWidget instead of QComboBox because combo popups get clipped
        # inside QGraphicsProxyWidget.  The list shows all items at once;
        # clicking a row selects it for playback.
        self.source_list = QListWidget()
        self.source_list.setObjectName("sourceList")
        self.source_list.setFixedHeight(60)
        self.source_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.source_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_list.currentRowChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_list)

        # --- Waveform visualization ---
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        # --- Scrubber: full-width slider + time label ---
        scrubber_row = QHBoxLayout()
        scrubber_row.setContentsMargins(0, 0, 0, 0)
        scrubber_row.setSpacing(3)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setFixedHeight(16)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        scrubber_row.addWidget(self.slider, 1)

        self.time_label = QLabel("0:00")
        self.time_label.setFixedWidth(36)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        scrubber_row.addWidget(self.time_label)

        layout.addLayout(scrubber_row)

        # --- Play button below scrubber ---
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedHeight(20)
        self.play_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.play_btn.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_btn)

        self._apply_stylesheet()

    def _apply_stylesheet(self):
        """Apply dark theme styling matching the node editor."""
        bg = Colors.BG_DARK.name()
        bg_light = Colors.BG_LIGHT.name()
        text = Colors.TEXT_PRIMARY.name()
        text_sec = Colors.TEXT_SECONDARY.name()
        accent = Colors.BLOCK_PLAYER.name()
        border = Colors.BORDER.name()

        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                color: {text};
                font-size: 10px;
            }}
            QListWidget#sourceList {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 2px;
                font-size: 9px;
                outline: none;
            }}
            QListWidget#sourceList::item {{
                padding: 2px 4px;
            }}
            QListWidget#sourceList::item:selected {{
                background: {accent};
                color: {text};
            }}
            QListWidget#sourceList::item:hover {{
                background: {bg_light};
            }}
            QPushButton {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 3px;
                font-size: 10px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background: {bg_light};
            }}
            QPushButton:pressed {{
                background: {accent};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {bg};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 10px;
                height: 10px;
                margin: -3px 0;
                background: {accent};
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 2px;
            }}
            QLabel {{
                color: {text_sec};
                font-size: 9px;
                background: transparent;
            }}
        """)

    # ------------------------------------------------------------------
    # Audio Player Backend
    # ------------------------------------------------------------------

    def _init_player(self):
        """Initialize the SimpleAudioPlayer backend."""
        try:
            from ui.qt_gui.widgets.timeline.playback.controller import SimpleAudioPlayer
            self._player = SimpleAudioPlayer()
        except Exception as e:
            Log.warning(f"AudioPlayerWidget: Could not init audio player: {e}")
            self._player = None

    def _start_update_timer(self):
        """Timer to update slider and time label during playback (~30 FPS)."""
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(33)  # ~30 FPS
        self._update_timer.timeout.connect(self._on_update_tick)

    # ------------------------------------------------------------------
    # Source Management
    # ------------------------------------------------------------------

    def refresh_sources(self):
        """
        Query the facade for audio items available on this block's input port.

        Populates the list widget so the user can click to select
        different audio sources (e.g., separated stems).
        """
        old_row = self.source_list.currentRow()
        old_name = ""
        if old_row >= 0 and old_row < len(self._current_audio_items):
            old_name = getattr(self._current_audio_items[old_row], 'name', '')

        self._current_audio_items = []

        self.source_list.blockSignals(True)
        self.source_list.clear()

        try:
            items = self._get_upstream_audio_items()
            self._current_audio_items = items
            Log.debug(f"AudioPlayerWidget: refresh_sources found {len(items)} audio items")

            if not items:
                self.source_list.addItem("(no audio)")
                self.waveform.clear()
                self.source_list.blockSignals(False)
                return

            match_idx = 0
            for i, item in enumerate(items):
                display = getattr(item, 'name', 'audio')
                self.source_list.addItem(display)
                Log.debug(f"AudioPlayerWidget:   - {display} ({item.file_path})")
                if display == old_name:
                    match_idx = i

            # #region agent log
            try:
                import json, time as _trs
                _src_entries = [{"idx": i, "name": getattr(it, 'name', '?'), "file_path": getattr(it, 'file_path', '?')} for i, it in enumerate(items)]
                with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                    _f.write(json.dumps({"hypothesisId": "H-PLAYER-PATH", "location": "audio_player_block_item.py:refresh_sources", "message": "source list entries", "data": {"block_id": self.block_id, "num_items": len(items), "entries": _src_entries, "match_idx": match_idx}, "timestamp": int(_trs.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion

            self.source_list.setCurrentRow(match_idx)
            # Manually sync _current_source_index because signals are blocked
            self._current_source_index = match_idx
            # Also set duration for immediate display and load waveform
            if 0 <= match_idx < len(items):
                self._duration = (items[match_idx].length_ms or 0) / 1000.0
                self.time_label.setText(_format_time(self._duration))
                # Defer waveform loading slightly so UI stays responsive
                QTimer.singleShot(50, self._load_waveform_for_current)
        except Exception as e:
            Log.warning(f"AudioPlayerWidget: Error refreshing sources: {e}")
            self.source_list.addItem("(error)")
            self.waveform.clear()
        finally:
            self.source_list.blockSignals(False)

    def _load_waveform_for_current(self):
        """Load waveform data for the currently selected audio source."""
        idx = self._current_source_index
        if 0 <= idx < len(self._current_audio_items):
            item = self._current_audio_items[idx]
            if item.file_path:
                self.waveform.load_audio(item.file_path)
                return
        self.waveform.clear()

    def _get_upstream_audio_items(self) -> list:
        """
        Get audio data items from the upstream block connected to
        this block's 'audio' input port.
        """
        if not self.facade:
            Log.debug("AudioPlayerWidget: No facade available")
            return []

        try:
            # Get connections for this block
            conn_result = self.facade.list_connections()
            if not conn_result.success or not conn_result.data:
                Log.debug(f"AudioPlayerWidget: No connections found (success={conn_result.success})")
                return []

            # Find incoming connections to our 'audio' port
            incoming = [
                c for c in conn_result.data
                if c.target_block_id == self.block_id
                and c.target_input_name == "audio"
            ]

            Log.debug(
                f"AudioPlayerWidget: {len(conn_result.data)} total connections, "
                f"{len(incoming)} incoming to audio port on block {self.block_id}"
            )

            if not incoming:
                return []

            # Collect audio items from all source blocks
            all_items = []
            for conn in incoming:
                source_items_result = self.facade.get_block_data(conn.source_block_id)
                if source_items_result.success and source_items_result.data:
                    # get_block_data returns {"block_id": ..., "data_items": [...]}
                    data_items = source_items_result.data.get("data_items", [])
                    Log.debug(
                        f"AudioPlayerWidget: Source block {conn.source_block_id} "
                        f"port '{conn.source_output_name}' has {len(data_items)} data items"
                    )
                    for item_dict in data_items:
                        item_type = item_dict.get("type")
                        item_port = item_dict.get("output_port")
                        has_file = bool(item_dict.get("file_path"))
                        Log.debug(
                            f"AudioPlayerWidget:   item type={item_type} port={item_port} "
                            f"has_file={has_file} name={item_dict.get('name')}"
                        )
                        if item_type == "Audio" and has_file:
                            # Check that the file belongs to the correct output port
                            if item_port == conn.source_output_name:
                                all_items.append(_AudioItemRef(
                                    name=item_dict.get("name", "audio"),
                                    file_path=item_dict["file_path"],
                                    length_ms=item_dict.get("length_ms", 0),
                                ))
                else:
                    Log.debug(
                        f"AudioPlayerWidget: get_block_data failed for {conn.source_block_id}: "
                        f"{source_items_result.message if hasattr(source_items_result, 'message') else '?'}"
                    )
            return all_items
        except Exception as e:
            Log.warning(f"AudioPlayerWidget: Error getting upstream items: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Transport Controls
    # ------------------------------------------------------------------

    def _toggle_playback(self):
        """Play or pause the currently selected audio source."""
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Start playback of the selected audio item."""
        if not self._player:
            return

        idx = self._current_source_index
        if idx < 0 or idx >= len(self._current_audio_items):
            return

        item = self._current_audio_items[idx]
        if not item.file_path:
            return

        # Load if needed (check if source changed)
        if not hasattr(self, '_loaded_path') or self._loaded_path != item.file_path:
            if not self._player.load(item.file_path):
                Log.warning(f"AudioPlayerWidget: Failed to load {item.file_path}")
                return
            self._loaded_path = item.file_path
            # Wait briefly for duration to become available
            QTimer.singleShot(100, self._update_duration)

        # #region agent log
        try:
            import json, time as _tpl
            with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H-PLAYER-PATH", "location": "audio_player_block_item.py:_play", "message": "playing file", "data": {"block_id": self.block_id, "idx": idx, "file_path": item.file_path, "item_name": getattr(item, 'name', '?'), "loaded_path": getattr(self, '_loaded_path', None)}, "timestamp": int(_tpl.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        self._player.play()
        self._is_playing = True
        self.play_btn.setText("⏸")
        self._update_timer.start()

    def _pause(self):
        """Pause current playback."""
        if self._player:
            self._player.pause()
        self._is_playing = False
        self.play_btn.setText("▶")
        self._update_timer.stop()

    def _stop(self):
        """Stop playback and reset position."""
        if self._player:
            self._player.stop()
        self._is_playing = False
        self.play_btn.setText("▶")
        self._update_timer.stop()
        self.slider.setValue(0)
        self.time_label.setText("0:00")
        self.waveform.set_position(0.0)

    def _on_source_changed(self, row: int):
        """Handle user clicking an audio source in the list."""
        self._stop()
        self._current_source_index = row
        if 0 <= row < len(self._current_audio_items):
            item = self._current_audio_items[row]
            self._duration = (item.length_ms or 0) / 1000.0
            self.time_label.setText(_format_time(self._duration))
            # #region agent log
            try:
                import json, time as _tsc
                with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                    _f.write(json.dumps({"hypothesisId": "H-PLAYER-PATH", "location": "audio_player_block_item.py:_on_source_changed", "message": "source changed", "data": {"block_id": self.block_id, "row": row, "item_name": getattr(item, 'name', '?'), "file_path": getattr(item, 'file_path', '?')}, "timestamp": int(_tsc.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion
            # Load waveform for newly selected source
            self._load_waveform_for_current()
        else:
            self.waveform.clear()

    # ------------------------------------------------------------------
    # Slider Interaction
    # ------------------------------------------------------------------

    def _on_slider_pressed(self):
        """User grabbed the slider -- pause updates."""
        self._slider_dragging = True

    def _on_slider_released(self):
        """User released the slider -- seek."""
        self._slider_dragging = False
        if self._player and self._duration > 0:
            ratio = self.slider.value() / 1000.0
            self._player.set_position(ratio * self._duration)

    def _on_slider_moved(self, value: int):
        """Update time label and waveform playhead while dragging."""
        if self._duration > 0:
            seconds = (value / 1000.0) * self._duration
            self.time_label.setText(_format_time(seconds))
            self.waveform.set_position(value / 1000.0)

    # ------------------------------------------------------------------
    # Timer Updates
    # ------------------------------------------------------------------

    def _on_update_tick(self):
        """Update slider and time label from player position."""
        if not self._player or not self._is_playing:
            return

        # Check if playback ended
        if not self._player.is_playing():
            self._stop()
            return

        pos = self._player.get_position()
        self._update_duration()

        if self._duration > 0 and not getattr(self, '_slider_dragging', False):
            ratio = pos / self._duration
            self.slider.setValue(int(ratio * 1000))
            self.waveform.set_position(ratio)

        self.time_label.setText(_format_time(pos))

    def _update_duration(self):
        """Refresh duration from the player backend."""
        if self._player:
            d = self._player.get_duration()
            if d > 0:
                self._duration = d

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Stop playback and release resources."""
        self._update_timer.stop()
        if self._player:
            try:
                self._player.cleanup()
            except Exception as e:
                Log.debug(f"AudioPlayerWidget: cleanup error: {e}")
            self._player = None


class _AudioItemRef:
    """Lightweight reference to an upstream audio item for the combo box."""
    __slots__ = ('name', 'file_path', 'length_ms')

    def __init__(self, name: str, file_path: str, length_ms: float):
        self.name = name
        self.file_path = file_path
        self.length_ms = length_ms


# ======================================================================
# AudioPlayerBlockItem -- custom node with embedded player
# ======================================================================

class AudioPlayerBlockItem(BlockItem):
    """
    BlockItem subclass that embeds an AudioPlayerWidget via
    QGraphicsProxyWidget inside the node body.

    The widget sits below the port zone and provides play/pause,
    timeline scrubbing, and source selection directly in the node.
    """

    def __init__(self, block: "Block", facade: "ApplicationFacade",
                 undo_stack: Optional["QUndoStack"] = None):
        # Must call super first -- sets up block, facade, dimensions, flags
        super().__init__(block, facade, undo_stack)

        # Build and embed the player widget
        self._player_widget = AudioPlayerWidget(block.id, facade)
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(self._player_widget)

        # Position the proxy inside the block body (below ports)
        self._position_proxy()

        # Subscribe to data change events to auto-refresh sources
        self._subscribe_player_events()

        # Initial source population (delayed to ensure scene is ready)
        QTimer.singleShot(200, self._player_widget.refresh_sources)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def _calculate_dimensions(self):
        """Extend base dimensions to accommodate the player controls and wider waveform."""
        super()._calculate_dimensions()
        self._width = Sizes.PLAYER_BLOCK_WIDTH
        self._height += Sizes.PLAYER_CONTROL_HEIGHT

    def _position_proxy(self):
        """
        Position the proxy widget inside the block, below the port zone.
        The block is centered at (0,0), so we offset from top-left.
        """
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height
        )
        # Place player at bottom of the block body with small margin
        proxy_x = rect.left() + 6
        proxy_y = rect.bottom() - Sizes.PLAYER_CONTROL_HEIGHT - 4
        self._proxy.setPos(proxy_x, proxy_y)

    # ------------------------------------------------------------------
    # Event Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_player_events(self):
        """Subscribe to events that indicate upstream data changed."""
        if not self.facade or not self.facade.event_bus:
            return
        self.facade.event_bus.subscribe("BlockChanged", self._on_player_data_changed)
        self.facade.event_bus.subscribe("BlockExecuted", self._on_player_data_changed)
        self.facade.event_bus.subscribe("ConnectionCreated", self._on_player_connection_changed)
        self.facade.event_bus.subscribe("ConnectionRemoved", self._on_player_connection_changed)

    def _unsubscribe_player_events(self):
        """Clean up player-specific event subscriptions."""
        if not self.facade or not self.facade.event_bus:
            return
        try:
            self.facade.event_bus.unsubscribe("BlockChanged", self._on_player_data_changed)
            self.facade.event_bus.unsubscribe("BlockExecuted", self._on_player_data_changed)
            self.facade.event_bus.unsubscribe("ConnectionCreated", self._on_player_connection_changed)
            self.facade.event_bus.unsubscribe("ConnectionRemoved", self._on_player_connection_changed)
        except Exception as e:
            Log.debug(f"AudioPlayerBlockItem: Error unsubscribing: {e}")

    def _on_player_data_changed(self, event):
        """Upstream block executed -- refresh available audio sources."""
        if not self._is_valid():
            return
        # Refresh on any block change (upstream might have produced new data)
        QTimer.singleShot(300, self._safe_refresh_sources)

    def _on_player_connection_changed(self, event):
        """Connection topology changed -- refresh sources."""
        if not self._is_valid():
            return
        try:
            data = event.data if hasattr(event, 'data') else {}
            source_id = data.get('source_block_id')
            target_id = data.get('target_block_id')
            if target_id == self.block.id or source_id == self.block.id:
                QTimer.singleShot(200, self._safe_refresh_sources)
        except Exception:
            pass

    def _safe_refresh_sources(self):
        """Safely refresh sources -- handles deleted C++ objects."""
        try:
            if self._is_valid() and self._player_widget:
                self._player_widget.refresh_sources()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        """Override to clean up player when removed from scene."""
        from PyQt6.QtWidgets import QGraphicsItem
        if change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            if value is None:
                # Removed from scene -- clean up
                self._unsubscribe_player_events()
                if self._player_widget:
                    self._player_widget.cleanup()
        return super().itemChange(change, value)
