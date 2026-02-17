"""
Event Inspector Panel

Panel showing properties of selected timeline events.
Designed to be used inside a QDockWidget for flexible positioning.
Handles single selection, multi-selection, and empty selection states.

Layout (single event):
  [Header]           -- "1 event selected" subtle count
  [Hero Section]     -- Time, Duration, Classification, Confidence (primary info)
  [Clip Player]      -- Compact waveform (64px) + play controls
  [Details]          -- ID, display mode, remaining metadata (de-emphasized)
"""

from typing import List, Dict, Any, Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QGridLayout, QSizePolicy,
    QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QColor

try:
    import librosa
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    np = None

# Local imports (makes package standalone)
from ..core.style import TimelineStyle as Colors, TimelineStyle as Spacing, TimelineStyle as Typography
from ..logging import TimelineLog as Log
from .clip_player import ClipAudioPlayer
from .waveform_widget import WaveformWidget
from ui.qt_gui.design_system import border_radius, Colors


# ---------------------------------------------------------------------------
# Typography helpers (consistent across all sections)
# ---------------------------------------------------------------------------

def _mono_font(size: int = 12, bold: bool = False) -> QFont:
    """Monospace font for numeric values (timestamps, IDs)."""
    f = QFont("SF Mono, Consolas, Monaco, monospace")
    f.setPixelSize(size)
    if bold:
        f.setBold(True)
    return f


def _label_font(size: int = 11) -> QFont:
    """Standard label font."""
    f = QFont("SF Pro Text, Segoe UI, -apple-system, system-ui, sans-serif")
    f.setPixelSize(size)
    return f


def _section_header_font() -> QFont:
    """Section header (e.g. 'Clip Player', 'Details')."""
    f = QFont("SF Pro Text, Segoe UI, -apple-system, system-ui, sans-serif")
    f.setPixelSize(10)
    f.setBold(True)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
    return f


class EventInspector(QWidget):
    """
    Panel for displaying selected event properties.

    Designed to be used inside a QDockWidget (TimelineWidget adds this).
    The dock widget provides the title bar and close/float controls.

    Layout hierarchy:
      Outer VBoxLayout
        -> count_label (header)
        -> QScrollArea (single scroll for everything)
             -> _content_widget  (VBoxLayout)
                  -> hero card   (time, duration, classification, confidence)
                  -> player card (waveform + controls)
                  -> details card (ID, display mode, extra metadata)

    Signals:
        selection_changed(int): Number of selected events changed
    """

    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._selected_events: List[Dict[str, Any]] = []
        self._grid_system = None  # Will be set by parent

        # Clip playback
        self._clip_player: Optional[ClipAudioPlayer] = None
        self._play_clip_button: Optional[QPushButton] = None
        self._clip_status_label: Optional[QLabel] = None
        self._waveform_widget: Optional[WaveformWidget] = None
        self._seek_slider: Optional[QSlider] = None
        self._loop_checkbox: Optional[QCheckBox] = None
        self._position_label: Optional[QLabel] = None
        self._current_clip_event: Optional[Dict[str, Any]] = None

        # Event update callback (for updating event properties)
        self._event_update_callback: Optional[Callable] = None

        # Thread management for waveform loading
        self._waveform_loader_thread: Optional[QThread] = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the top-level layout: header + single scroll area."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- Header (selection count, sits outside the scroll) --
        header = QWidget()
        header.setFixedHeight(28)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 4, 10, 4)

        self._count_label = QLabel("No selection")
        self._count_label.setFont(_label_font(11))
        self._count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        header_layout.addWidget(self._count_label)
        header_layout.addStretch()
        root.addWidget(header)

        # -- Thin separator under header --
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
        root.addWidget(sep)

        # -- Single scroll area for ALL content --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                width: 6px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER.name()};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        # Content widget lives inside the scroll
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 6, 8, 8)
        self._content_layout.setSpacing(6)
        self._content_widget.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")

        scroll.setWidget(self._content_widget)
        root.addWidget(scroll, 1)

        # Set minimum size
        self.setMinimumHeight(120)
        self.setMinimumWidth(200)

        # Root background
        self.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")

        # Initialize with empty state
        self._update_display()

    # ------------------------------------------------------------------
    # Card helper -- subtle grouped sections
    # ------------------------------------------------------------------

    def _make_card(self, parent_layout: QVBoxLayout) -> QVBoxLayout:
        """
        Create a subtle card container and return its inner QVBoxLayout.

        Cards have a slightly lighter background and thin border-radius
        to visually group related properties without heavy chrome.
        """
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(6)};
            }}
        """)
        inner = QVBoxLayout(card)
        inner.setContentsMargins(10, 8, 10, 8)
        inner.setSpacing(4)
        parent_layout.addWidget(card)
        return inner

    def _add_section_header(self, layout: QVBoxLayout, text: str):
        """Add an uppercase, small section header label."""
        lbl = QLabel(text.upper())
        lbl.setFont(_section_header_font())
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; border: none; background: transparent;")
        lbl.setContentsMargins(0, 0, 0, 2)
        layout.addWidget(lbl)
    
    def set_grid_system(self, grid_system):
        """Set grid system for time formatting"""
        self._grid_system = grid_system

    def set_data_item_repo(self, repo):
        """
        Set data_item_repo for direct audio item lookup by ID.

        Args:
            repo: DataItemRepository instance for repo.get(audio_id) lookups.
        """
        if repo:
            self._clip_player = ClipAudioPlayer(data_item_repo=repo)
        else:
            if self._clip_player:
                self._clip_player.cleanup()
            self._clip_player = None

    def set_audio_lookup_callback(self, callback: Optional[Callable]):
        """Deprecated: Use set_data_item_repo() instead. Kept for backward compatibility."""
        if callback:
            self._clip_player = ClipAudioPlayer(audio_lookup_callback=callback)
        else:
            if self._clip_player:
                self._clip_player.cleanup()
            self._clip_player = None

    def set_event_update_callback(self, callback: Optional[Callable]):
        """
        Set callback function for updating events.

        Args:
            callback: Function that accepts (event_id: str, metadata: Dict[str, Any])
                     and updates the event. Should return True if successful.
        """
        self._event_update_callback = callback

    def update_selection(self, selected_events: List[Dict[str, Any]]):
        """
        Update the display with selected events.

        Args:
            selected_events: List of event data dicts from TimelineScene
        """
        self._selected_events = selected_events
        self._update_display()
        self.selection_changed.emit(len(selected_events))

    # ------------------------------------------------------------------
    # Display dispatch
    # ------------------------------------------------------------------

    def _update_display(self):
        """Update the display based on current selection."""
        # Clear all children inside the content widget
        layout = self._content_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        count = len(self._selected_events)
        if count == 0:
            self._show_empty_state()
        elif count == 1:
            self._show_single_event(self._selected_events[0])
        else:
            self._show_multi_selection()

    # ------------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------------

    def _show_empty_state(self):
        """Show empty selection state."""
        self._count_label.setText("No selection")

        label = QLabel("Select an event to inspect")
        label.setFont(_label_font(12))
        label.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()};")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setContentsMargins(0, 24, 0, 24)
        self._content_layout.addWidget(label)
        self._content_layout.addStretch()

    # ------------------------------------------------------------------
    # Single-event display
    # ------------------------------------------------------------------

    def _show_single_event(self, event: Dict[str, Any]):
        """
        Show single event details in three visual sections:
          1. Hero card  -- timestamps, classification, confidence
          2. Player card -- compact waveform + play controls
          3. Details card -- ID, display mode, remaining metadata
        """
        self._count_label.setText("1 event selected")
        layout = self._content_layout
        metadata = event.get('metadata', {})

        # ============================================================
        # 1. HERO CARD -- primary info the user scans first
        # ============================================================
        hero = self._make_card(layout)

        # -- Timestamps row --
        self._add_section_header(hero, "Timing")

        time_grid = QGridLayout()
        time_grid.setContentsMargins(0, 0, 0, 0)
        time_grid.setSpacing(2)
        time_grid.setColumnStretch(1, 1)

        trow = 0
        trow = self._add_hero_row(time_grid, trow, "Start", self._format_time(event['start_time']))

        if event['duration'] > 0:
            trow = self._add_hero_row(time_grid, trow, "End", self._format_time(event['end_time']))
            trow = self._add_hero_row(time_grid, trow, "Duration", self._format_duration(event['duration']))

        # Wrap grid in a widget so it can be added to the VBox
        time_w = QWidget()
        time_w.setLayout(time_grid)
        time_w.setStyleSheet("background: transparent; border: none;")
        hero.addWidget(time_w)

        # -- Classification / Layer --
        layer_name = (
            event.get('layer_name')
            or event.get('classification')
            or f"Layer {event.get('layer_index', 0) + 1}"
        )
        self._add_hero_value(hero, "Layer", layer_name, accent=True)

        # -- Confidence (if present in metadata) --
        confidence = metadata.get('confidence')
        if confidence is not None:
            try:
                conf_val = float(confidence)
                conf_text = f"{conf_val:.1%}" if conf_val <= 1.0 else f"{conf_val:.1f}"
            except (ValueError, TypeError):
                conf_text = str(confidence)
            self._add_hero_value(hero, "Confidence", conf_text)

        # ============================================================
        # 2. CLIP PLAYER CARD -- compact waveform + controls
        # ============================================================
        is_clip_event = (
            metadata.get('clip_start_time') is not None
            and metadata.get('clip_end_time') is not None
        )

        if is_clip_event and self._clip_player:
            player_card = self._make_card(layout)
            self._add_section_header(player_card, "Clip Player")

            # Source name (small, secondary)
            audio_name = metadata.get('audio_name', 'Unknown')
            src_lbl = QLabel(audio_name)
            src_lbl.setFont(_label_font(10))
            src_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; border: none; background: transparent;")
            src_lbl.setWordWrap(True)
            player_card.addWidget(src_lbl)

            # Store current clip event for waveform loading
            self._current_clip_event = event

            # Compact waveform (64px)
            if HAS_AUDIO_LIBS:
                self._waveform_widget = WaveformWidget()
                self._waveform_widget.setMinimumHeight(48)
                self._waveform_widget.setMaximumHeight(64)
                self._waveform_widget.setStyleSheet(f"""
                    background-color: {Colors.BG_DARK.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                """)
                self._waveform_widget.seek_requested.connect(self._on_waveform_seek)
                player_card.addWidget(self._waveform_widget)

                # Load waveform data asynchronously
                QTimer.singleShot(0, lambda: self._load_waveform_data(event))

            # Seek slider (thin, under waveform)
            self._seek_slider = QSlider(Qt.Orientation.Horizontal)
            self._seek_slider.setMinimum(0)
            self._seek_slider.setMaximum(1000)
            self._seek_slider.setValue(0)
            self._seek_slider.setEnabled(False)
            self._seek_slider.setFixedHeight(14)
            self._seek_slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    height: 3px;
                    background: {Colors.BG_LIGHT.name()};
                    border-radius: 1px;
                }}
                QSlider::handle:horizontal {{
                    width: 8px;
                    height: 8px;
                    margin: -3px 0;
                    background: {Colors.TEXT_PRIMARY.name()};
                    border-radius: 4px;
                }}
                QSlider::sub-page:horizontal {{
                    background: {Colors.ACCENT_BLUE.name()};
                    border-radius: 1px;
                }}
            """)
            self._seek_slider.valueChanged.connect(self._on_seek_slider_changed)
            self._seek_slider.sliderPressed.connect(lambda: setattr(self, '_seeking', True))
            self._seek_slider.sliderReleased.connect(lambda: setattr(self, '_seeking', False))
            self._seeking = False
            player_card.addWidget(self._seek_slider)

            # Play button + position label row
            ctrl_row = QHBoxLayout()
            ctrl_row.setContentsMargins(0, 2, 0, 0)
            ctrl_row.setSpacing(6)

            self._play_clip_button = QPushButton("Play")
            self._play_clip_button.setFixedHeight(24)
            self._play_clip_button.setFont(_label_font(11))
            self._play_clip_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._play_clip_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    padding: 0 14px;
                    border-radius: {border_radius(4)};
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_BLUE.lighter(115).name()};
                }}
                QPushButton:disabled {{
                    background-color: {Colors.BG_LIGHT.name()};
                    color: {Colors.TEXT_DISABLED.name()};
                }}
            """)
            self._play_clip_button.clicked.connect(lambda: self._on_play_clip_clicked(event))
            ctrl_row.addWidget(self._play_clip_button)

            # Position readout (monospace)
            self._position_label = QLabel("")
            self._position_label.setFont(_mono_font(10))
            self._position_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; border: none; background: transparent;")
            ctrl_row.addWidget(self._position_label)

            # Status label (errors, loading state)
            self._clip_status_label = QLabel("")
            self._clip_status_label.setFont(_label_font(10))
            self._clip_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; border: none; background: transparent;")
            ctrl_row.addWidget(self._clip_status_label)
            ctrl_row.addStretch()

            ctrl_w = QWidget()
            ctrl_w.setLayout(ctrl_row)
            ctrl_w.setStyleSheet("background: transparent; border: none;")
            player_card.addWidget(ctrl_w)

            # Update button state based on current playback
            self._update_clip_button_state()

            # Start timer to monitor playback state and update UI
            if not hasattr(self, '_playback_check_timer'):
                self._playback_check_timer = QTimer(self)
                self._playback_check_timer.timeout.connect(self._update_playback_ui)
                self._playback_check_timer.start(50)

        # ============================================================
        # 3. DETAILS CARD -- debug / secondary info
        # ============================================================
        # Collect detail rows: ID, display mode, and remaining metadata
        detail_rows: List[tuple] = []

        # Event ID
        event_id = event['event_id']
        if len(event_id) > 24:
            event_id = event_id[:21] + "..."
        detail_rows.append(("ID", event_id))

        # Display mode
        render_as_marker = metadata.get('render_as_marker', False)
        detail_rows.append(("Display", "Marker" if render_as_marker else "Clip"))

        # Remaining metadata (exclude keys already surfaced)
        shown_keys = {
            'clip_start_time', 'clip_end_time', 'audio_name', 'audio_id',
            'render_as_marker', 'confidence',
        }
        for key, value in metadata.items():
            if key in shown_keys:
                continue
            value_str = str(value)
            if len(value_str) > 40:
                value_str = value_str[:37] + "..."
            # Humanize key name: snake_case -> Title Case
            display_key = key.replace('_', ' ').title()
            detail_rows.append((display_key, value_str))

        if detail_rows:
            details = self._make_card(layout)
            self._add_section_header(details, "Details")

            detail_grid = QGridLayout()
            detail_grid.setContentsMargins(0, 0, 0, 0)
            detail_grid.setSpacing(2)
            detail_grid.setColumnStretch(1, 1)

            for drow_idx, (lbl, val) in enumerate(detail_rows):
                self._add_detail_row(detail_grid, drow_idx, lbl, val)

            detail_w = QWidget()
            detail_w.setLayout(detail_grid)
            detail_w.setStyleSheet("background: transparent; border: none;")
            details.addWidget(detail_w)

        # Push remaining space to bottom
        layout.addStretch()

    # ------------------------------------------------------------------
    # Multi-selection display
    # ------------------------------------------------------------------

    def _show_multi_selection(self):
        """Show multi-selection summary."""
        count = len(self._selected_events)
        self._count_label.setText(f"{count} events selected")

        layout = self._content_layout
        card = self._make_card(layout)
        self._add_section_header(card, "Selection Summary")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.setColumnStretch(1, 1)

        row = 0
        row = self._add_hero_row(grid, row, "Selected", f"{count} events")

        # Aggregates
        total_duration = sum(e['duration'] for e in self._selected_events)
        min_time = min(e['start_time'] for e in self._selected_events)
        max_time = max(e['end_time'] for e in self._selected_events)

        row = self._add_hero_row(
            grid, row, "Range",
            f"{self._format_time(min_time)} -- {self._format_time(max_time)}"
        )
        row = self._add_hero_row(grid, row, "Total Dur.", self._format_duration(total_duration))

        span = max_time - min_time
        row = self._add_hero_row(grid, row, "Span", self._format_duration(span))

        # Layers involved
        layers = set(e.get('layer_name') or e['classification'] for e in self._selected_events)
        if len(layers) == 1:
            row = self._add_hero_row(grid, row, "Layer", list(layers)[0])
        else:
            row = self._add_hero_row(grid, row, "Layers", f"{len(layers)} layers")

        grid_w = QWidget()
        grid_w.setLayout(grid)
        grid_w.setStyleSheet("background: transparent; border: none;")
        card.addWidget(grid_w)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Row helpers
    # ------------------------------------------------------------------

    def _add_hero_row(self, grid: QGridLayout, row: int, label: str, value: str) -> int:
        """
        Add a primary info row: left-aligned dim label, monospace value.
        Used in the hero card and multi-selection summary.
        """
        lbl = QLabel(label)
        lbl.setFont(_label_font(10))
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; background: transparent; border: none;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lbl.setFixedWidth(64)

        val = QLabel(value)
        val.setFont(_mono_font(12))
        val.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background: transparent; border: none;")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        grid.addWidget(lbl, row, 0)
        grid.addWidget(val, row, 1)
        return row + 1

    def _add_hero_value(self, card_layout: QVBoxLayout, label: str, value: str, accent: bool = False):
        """
        Add a stacked label/value pair inside a card.
        Used for classification and confidence where a single prominent value matters.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFont(_label_font(10))
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; background: transparent; border: none;")
        lbl.setFixedWidth(64)

        val = QLabel(value)
        if accent:
            val.setFont(_mono_font(12, bold=True))
            val.setStyleSheet(f"color: {Colors.ACCENT_BLUE.name()}; background: transparent; border: none;")
        else:
            val.setFont(_mono_font(12))
            val.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background: transparent; border: none;")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        row.addWidget(lbl)
        row.addWidget(val)
        row.addStretch()

        row_w = QWidget()
        row_w.setLayout(row)
        row_w.setStyleSheet("background: transparent; border: none;")
        card_layout.addWidget(row_w)

    def _add_detail_row(self, grid: QGridLayout, row: int, label: str, value: str):
        """
        Add a de-emphasized detail row (smaller, dimmer).
        Used in the details/debug card for metadata.
        """
        lbl = QLabel(label)
        lbl.setFont(_label_font(10))
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; background: transparent; border: none;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        val = QLabel(value)
        val.setFont(_mono_font(10))
        val.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; background: transparent; border: none;")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        val.setWordWrap(True)

        grid.addWidget(lbl, row, 0)
        grid.addWidget(val, row, 1)
    
    def _format_time(self, seconds: float) -> str:
        """Format time value"""
        if self._grid_system:
            return self._grid_system.format_time(seconds)
        return f"{seconds:.3f}s"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration value"""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.3f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    
    def _load_waveform_data(self, event: Dict[str, Any]):
        """
        Load waveform data for the clip event (async to prevent UI freeze).
        
        Uses WaveformService to get cached waveforms or generate on-demand.
        Handles file path resolution for Windows/Mac compatibility.
        """
        if not HAS_AUDIO_LIBS or not self._waveform_widget or not self._clip_player:
            return
        
        metadata = event.get('metadata', {})
        audio_id = metadata.get('audio_id')
        audio_name = metadata.get('audio_name')
        clip_start = metadata.get('clip_start_time')
        clip_end = metadata.get('clip_end_time')
        
        # All events must have valid start_time and end_time - no fallbacks
        # Hard fail if clip times are missing or invalid
        if clip_start is None or clip_end is None:
            error_msg = (
                f"EventInspector: Missing clip timing for event {event.get('event_id', 'unknown')}. "
                f"clip_start_time: {clip_start}, clip_end_time: {clip_end}. "
                f"All events must have valid start and end times."
            )
            Log.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate time range - hard fail if invalid
        if clip_end <= clip_start:
            error_msg = (
                f"EventInspector: Invalid clip time range for event {event.get('event_id', 'unknown')}. "
                f"clip_start_time: {clip_start:.3f}s, clip_end_time: {clip_end:.3f}s. "
                f"End time must be greater than start time."
            )
            Log.error(error_msg)
            raise ValueError(error_msg)
        
        clip_duration = clip_end - clip_start
        
        # Validate minimum duration - hard fail if too small
        from ..types import MIN_EVENT_DURATION
        if clip_duration < MIN_EVENT_DURATION:
            error_msg = (
                f"EventInspector: Clip duration too small for event {event.get('event_id', 'unknown')}. "
                f"Duration: {clip_duration:.6f}s, minimum: {MIN_EVENT_DURATION}s. "
                f"All events must have duration >= {MIN_EVENT_DURATION}s."
            )
            Log.error(error_msg)
            raise ValueError(error_msg)
        
        # Look up audio item -- direct repo lookup (preferred) or legacy callback
        audio_item = None
        if self._clip_player and hasattr(self._clip_player, '_data_item_repo') and self._clip_player._data_item_repo and audio_id:
            audio_item = self._clip_player._data_item_repo.get(audio_id)
        elif self._clip_player and self._clip_player._audio_lookup_callback:
            audio_item = self._clip_player._audio_lookup_callback(audio_id, audio_name)
        
        if not audio_item:
            Log.warning(f"EventInspector: Audio item not found (id: {audio_id}, name: {audio_name})")
            return
        
        # Stop any existing loader thread before starting a new one
        if self._waveform_loader_thread and self._waveform_loader_thread.isRunning():
            self._waveform_loader_thread.requestInterruption()
            # Use shorter timeout to prevent blocking
            if not self._waveform_loader_thread.wait(500):
                Log.warning("EventInspector: Previous waveform loader thread did not stop in time - continuing anyway")
            self._waveform_loader_thread.deleteLater()
            self._waveform_loader_thread = None
        
        # Load waveform in background thread to prevent UI freeze
        class WaveformLoaderThread(QThread):
            finished = pyqtSignal(object, int, float)  # waveform_data, sample_rate, duration
            
            def __init__(self, audio_item, start_time, end_time):
                super().__init__()
                self._audio_item = audio_item
                self._start_time = start_time
                self._end_time = end_time
            
            def run(self):
                try:
                    # Check for interruption
                    if self.isInterruptionRequested():
                        return
                    
                    # Use waveform_simple module which handles caching and WaveformService
                    from .waveform_simple import get_waveform_for_event
                    
                    # Calculate clip duration (this is what we need for the widget)
                    clip_duration = self._end_time - self._start_time
                    
                    waveform_data, returned_duration = get_waveform_for_event(
                        self._audio_item.id if hasattr(self._audio_item, 'id') else None,
                        self._audio_item.name if hasattr(self._audio_item, 'name') else None,
                        self._start_time,
                        self._end_time
                    )
                    
                    # Use clip duration (not returned duration which might be wrong)
                    duration = clip_duration
                    
                    if waveform_data is None or len(waveform_data) == 0:
                        # Try to generate waveform on-demand if not cached
                        if not self.isInterruptionRequested():
                            waveform_data, generated_duration = self._generate_waveform_on_demand()
                            # Use clip duration, not generated duration
                            duration = clip_duration
                    
                    # Validate we have valid waveform data
                    if waveform_data is None or len(waveform_data) == 0 or duration <= 0:
                        Log.warning(f"EventInspector: Invalid waveform data (data: {waveform_data is not None}, len: {len(waveform_data) if waveform_data is not None else 0}, duration: {duration})")
                        if not self.isInterruptionRequested():
                            self.finished.emit(None, 0, 0.0)
                        return
                    
                    # Estimate sample rate from duration and data length
                    # For waveform display, we don't need exact sample rate
                    # Use a reasonable default for visualization
                    sample_rate = 44100  # Default for display purposes
                    if len(waveform_data) > 0 and duration > 0:
                        # Estimate based on data points (rough approximation)
                        # This gives us points per second, which we can use to estimate sample rate
                        points_per_second = len(waveform_data) / duration
                        # For display purposes, use a reasonable sample rate
                        # The actual sample rate doesn't matter for visualization
                        sample_rate = max(8000, min(int(points_per_second * 100), 48000))  # Reasonable range
                    
                    # Check again before emitting (thread might have been interrupted)
                    if not self.isInterruptionRequested():
                        self.finished.emit(waveform_data, sample_rate, duration)
                except Exception as e:
                    Log.error(f"EventInspector: Failed to load waveform data: {e}")
                    import traceback
                    traceback.print_exc()
                    if not self.isInterruptionRequested():
                        self.finished.emit(None, 0, 0.0)
            
            def _generate_waveform_on_demand(self):
                """
                Generate waveform on-demand if not cached.
                Uses WaveformService to compute and cache waveform.
                Handles file path resolution for Windows/Mac compatibility.
                """
                try:
                    from src.shared.application.services.waveform_service import get_waveform_service
                    from .waveform_simple import get_waveform_for_event
                    from pathlib import Path
                    import os
                    
                    waveform_service = get_waveform_service()
                    
                    # Check if audio file exists and is accessible
                    file_path = getattr(self._audio_item, 'file_path', None)
                    if not file_path:
                        # Try metadata fallback
                        file_path = self._audio_item.metadata.get('file_path') if hasattr(self._audio_item, 'metadata') else None
                    
                    if not file_path:
                        Log.warning(f"EventInspector: No file path for audio item {getattr(self._audio_item, 'name', 'unknown')}")
                        return None, 0.0
                    
                    # Validate file exists (handle both absolute and relative paths)
                    # Use pathlib.Path for cross-platform compatibility (Windows/Mac)
                    path_obj = Path(file_path)
                    
                    # Check if file exists
                    if not path_obj.exists():
                        # Try to resolve relative paths
                        # Check if it's a relative path that might need resolution
                        if not os.path.isabs(file_path):
                            # Try resolving relative to current working directory
                            cwd_path = Path.cwd() / file_path
                            if cwd_path.exists():
                                path_obj = cwd_path
                                file_path = str(path_obj)
                                # Update audio item with resolved path
                                if hasattr(self._audio_item, 'file_path'):
                                    self._audio_item.file_path = file_path
                                if hasattr(self._audio_item, 'metadata'):
                                    self._audio_item.metadata['file_path'] = file_path
                            else:
                                # File not found - log warning and return None
                                Log.warning(
                                    f"EventInspector: Audio file not found at {file_path} "
                                    f"(tried absolute and relative to {Path.cwd()})"
                                )
                                return None, 0.0
                        else:
                            # Absolute path doesn't exist
                            Log.warning(f"EventInspector: Audio file not found at absolute path {file_path}")
                            return None, 0.0
                    
                    # Ensure waveform exists (generate if needed)
                    # WaveformService.compute_and_store() will handle file loading internally
                    if not waveform_service.has_waveform(self._audio_item):
                        Log.info(f"EventInspector: Generating waveform on-demand for {getattr(self._audio_item, 'name', 'unknown')}")
                        success = waveform_service.compute_and_store(self._audio_item)
                        if not success:
                            Log.warning(f"EventInspector: Failed to generate waveform for {getattr(self._audio_item, 'name', 'unknown')}")
                            return None, 0.0
                    
                    # Now get the waveform slice
                    waveform_data, duration = get_waveform_for_event(
                        self._audio_item.id if hasattr(self._audio_item, 'id') else None,
                        self._audio_item.name if hasattr(self._audio_item, 'name') else None,
                        self._start_time,
                        self._end_time
                    )
                    
                    return waveform_data, duration
                    
                except Exception as e:
                    Log.error(f"EventInspector: Failed to generate waveform on-demand: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, 0.0
        
        # Create and start loader thread
        loader = WaveformLoaderThread(audio_item, clip_start, clip_end)
        loader.finished.connect(lambda data, sr, dur: self._on_waveform_loaded(data, sr, dur))
        self._waveform_loader_thread = loader
        loader.start()
    
    def _on_waveform_loaded(self, audio_data, sample_rate: int, duration: float):
        """Handle waveform data loaded from background thread."""
        if audio_data is not None and len(audio_data) > 0 and duration > 0 and self._waveform_widget:
            self._waveform_widget.set_waveform_data(audio_data, sample_rate, duration)
        else:
            Log.warning(f"EventInspector: Invalid waveform data received (data: {audio_data is not None}, len: {len(audio_data) if audio_data is not None else 0}, duration: {duration})")
            if self._waveform_widget:
                self._waveform_widget.clear()
    
    def _on_play_clip_clicked(self, event: Dict[str, Any]):
        """Handle play clip button click"""
        if not self._clip_player:
            return
        
        metadata = event.get('metadata', {})
        audio_id = metadata.get('audio_id')
        audio_name = metadata.get('audio_name')
        clip_start = metadata.get('clip_start_time')
        clip_end = metadata.get('clip_end_time')
        
        # All events must have valid start_time and end_time - hard fail if invalid
        from ..types import MIN_EVENT_DURATION
        
        if clip_start is None or clip_end is None:
            error_msg = f"Missing clip timing (start: {clip_start}, end: {clip_end})"
            Log.error(f"EventInspector: {error_msg}")
            self._set_clip_status(f"Error: {error_msg}", is_error=True)
            return
        
        if clip_end <= clip_start:
            error_msg = f"Invalid time range ({clip_start:.3f}s to {clip_end:.3f}s)"
            Log.error(f"EventInspector: {error_msg}")
            self._set_clip_status(f"Error: {error_msg}", is_error=True)
            return
        
        clip_duration = clip_end - clip_start
        if clip_duration < MIN_EVENT_DURATION:
            error_msg = f"Duration too small ({clip_duration:.6f}s, minimum: {MIN_EVENT_DURATION}s)"
            Log.error(f"EventInspector: {error_msg}")
            self._set_clip_status(f"Error: {error_msg}", is_error=True)
            return
        
        # Check if already playing (stop first)
        if self._clip_player.is_playing():
            self._clip_player.stop()
            self._update_clip_button_state()
            if self._seek_slider:
                self._seek_slider.setValue(0)
            if self._waveform_widget:
                self._waveform_widget.set_playhead_position(0.0)
            return
        
        # Update button to loading state
        if self._play_clip_button:
            self._play_clip_button.setText("Loading...")
            self._play_clip_button.setEnabled(False)
        self._set_clip_status("Extracting audio clip...")
        
        # Play clip (on a slight delay to allow UI to update)
        QTimer.singleShot(50, lambda: self._do_play_clip(audio_id, audio_name, clip_start, clip_end))
    
    def _do_play_clip(self, audio_id: Optional[str], audio_name: Optional[str], start_time: float, end_time: float):
        """Actually perform the clip playback"""
        if not self._clip_player:
            return
        
        success, message = self._clip_player.play_clip(audio_id, audio_name, start_time, end_time)
        
        if success:
            self._set_clip_status(message, is_error=False)
            if self._play_clip_button:
                self._play_clip_button.setText("Stop")
                self._play_clip_button.setEnabled(True)
            
            # Enable seek slider
            if self._seek_slider:
                duration = self._clip_player.get_duration()
                self._seek_slider.setMaximum(int(duration * 1000))  # Convert to milliseconds
                self._seek_slider.setEnabled(True)
            
            # Connect position updates
            if self._clip_player:
                self._clip_player.position_changed.connect(self._on_position_changed)
                self._clip_player.playback_finished.connect(self._on_playback_finished)
        else:
            self._set_clip_status(message, is_error=True)
            if self._play_clip_button:
                self._play_clip_button.setText("Play")
                self._play_clip_button.setEnabled(True)
    
    def _on_position_changed(self, position: float):
        """Handle position change from player."""
        try:
            if not self._seeking and self._seek_slider:
                # Check if slider still exists before accessing
                if hasattr(self._seek_slider, 'setValue'):
                    # Update slider (convert seconds to milliseconds)
                    self._seek_slider.setValue(int(position * 1000))
        except RuntimeError:
            # Slider was deleted - disconnect and stop timer
            if self._clip_player:
                try:
                    self._clip_player.position_changed.disconnect()
                except:
                    pass
            if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                self._playback_check_timer.stop()
            return
        
        try:
            if self._waveform_widget and hasattr(self._waveform_widget, 'set_playhead_position'):
                self._waveform_widget.set_playhead_position(position)
        except RuntimeError:
            # Widget was deleted - ignore
            pass
        
        try:
            if self._position_label and hasattr(self._position_label, 'setText'):
                duration = self._clip_player.get_duration() if self._clip_player else 0.0
                self._position_label.setText(f"{position:.2f}s / {duration:.2f}s")
        except RuntimeError:
            # Label was deleted - ignore
            pass
    
    def _on_seek_slider_changed(self, value: int):
        """Handle seek slider change."""
        if not self._clip_player or not self._seeking:
            return
        
        try:
            position = value / 1000.0  # Convert milliseconds to seconds
            self._clip_player.set_position(position)
            
            if self._waveform_widget and hasattr(self._waveform_widget, 'set_playhead_position'):
                self._waveform_widget.set_playhead_position(position)
            
            if self._position_label and hasattr(self._position_label, 'setText'):
                duration = self._clip_player.get_duration()
                self._position_label.setText(f"{position:.2f}s / {duration:.2f}s")
        except RuntimeError:
            # UI elements were deleted - ignore
            pass
    
    def _on_waveform_seek(self, position: float):
        """Handle seek request from waveform widget."""
        try:
            if self._clip_player:
                self._clip_player.set_position(position)
            
            if self._seek_slider and hasattr(self._seek_slider, 'setValue'):
                self._seek_slider.setValue(int(position * 1000))
        except RuntimeError:
            # UI elements were deleted - ignore
            pass
    
    def _on_loop_changed(self, state: int):
        """Handle loop checkbox change."""
        if self._clip_player:
            enabled = state == Qt.CheckState.Checked.value
            self._clip_player.set_loop(enabled)
    
    def _on_playback_finished(self):
        """Handle playback finished signal."""
        try:
            if self._seek_slider and hasattr(self._seek_slider, 'setValue'):
                self._seek_slider.setValue(0)
        except RuntimeError:
            pass
        
        try:
            if self._waveform_widget and hasattr(self._waveform_widget, 'set_playhead_position'):
                self._waveform_widget.set_playhead_position(0.0)
        except RuntimeError:
            pass
        
        try:
            if self._position_label and self._clip_player and hasattr(self._position_label, 'setText'):
                duration = self._clip_player.get_duration()
                self._position_label.setText(f"0.00s / {duration:.2f}s")
        except RuntimeError:
            pass
    
    def closeEvent(self, event):
        """Clean up resources when widget is closed"""
        # Stop any running waveform loader thread
        if self._waveform_loader_thread and self._waveform_loader_thread.isRunning():
            self._waveform_loader_thread.requestInterruption()
            # During close, use very short timeout to prevent hanging
            # Request interruption and let Qt handle cleanup via deleteLater
            if not self._waveform_loader_thread.wait(100):  # Very short wait - just give it a chance to stop
                Log.debug("EventInspector: Waveform loader thread still running during close - will be cleaned up by Qt")
            # Use deleteLater to let Qt clean up the thread asynchronously
            self._waveform_loader_thread.deleteLater()
            self._waveform_loader_thread = None
        
        if self._clip_player:
            self._clip_player.cleanup()
        if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
            self._playback_check_timer.stop()
        super().closeEvent(event)
    
    def _update_clip_button_state(self):
        """Update play button state based on playback status"""
        # Defensive check: ensure button exists and hasn't been deleted
        if not self._play_clip_button or not self._clip_player:
            return
        
        try:
            # Check if the button object is still valid (not deleted)
            if not hasattr(self._play_clip_button, 'text'):
                return
            
            if self._clip_player.is_playing():
                if self._play_clip_button.text() != "Stop":
                    self._play_clip_button.setText("Stop")
                    self._play_clip_button.setEnabled(True)
            else:
                if self._play_clip_button.text() != "Play":
                    self._play_clip_button.setText("Play")
                    self._play_clip_button.setEnabled(True)
        except RuntimeError:
            # Button has been deleted - stop the timer and return
            if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                self._playback_check_timer.stop()
            return
    
    def _update_playback_ui(self):
        """Update playback UI elements (called by timer)."""
        # Defensive check: ensure player and UI elements exist
        if not self._clip_player:
            # Stop timer if player is gone
            if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                self._playback_check_timer.stop()
            return
        
        # Check if UI elements still exist before updating
        if not self._play_clip_button or not self._seek_slider or not self._position_label:
            # UI elements have been cleared - stop the timer
            if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                self._playback_check_timer.stop()
            return
        
        # Update button state (with defensive checks inside)
        try:
            self._update_clip_button_state()
        except RuntimeError:
            # UI element was deleted - stop timer
            if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                self._playback_check_timer.stop()
            return
        
        # Update position if playing
        if self._clip_player.is_playing():
            try:
                position = self._clip_player.get_position()
                self._on_position_changed(position)
            except RuntimeError:
                # UI element was deleted during update - stop timer
                if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
                    self._playback_check_timer.stop()
                return
    
    def _set_clip_status(self, message: str, is_error: bool = False):
        """Set status label text."""
        if self._clip_status_label:
            self._clip_status_label.setText(message)
            color = Colors.DANGER_FG.name() if is_error else Colors.TEXT_SECONDARY.name()
            self._clip_status_label.setStyleSheet(
                f"color: {color}; border: none; background: transparent;"
            )
    
    def clear(self):
        """Clear the selection display"""
        # Stop playback check timer FIRST before clearing UI elements
        # This prevents the timer from trying to access deleted widgets
        if hasattr(self, '_playback_check_timer') and self._playback_check_timer:
            self._playback_check_timer.stop()
            self._playback_check_timer = None
        
        # Stop any playing clip
        if self._clip_player:
            self._clip_player.stop()
            # Disconnect signals
            try:
                self._clip_player.position_changed.disconnect()
                self._clip_player.playback_finished.disconnect()
            except:
                pass
        
        # Stop any running waveform loader thread
        if self._waveform_loader_thread and self._waveform_loader_thread.isRunning():
            self._waveform_loader_thread.requestInterruption()
            # Use shorter timeout to prevent blocking
            if not self._waveform_loader_thread.wait(300):  # Reduced to 300ms for faster cleanup
                Log.debug("EventInspector: Waveform loader thread did not stop in time during clear - continuing anyway")
            self._waveform_loader_thread.deleteLater()
            self._waveform_loader_thread = None
        
        # Clear UI elements (after timer is stopped)
        self._play_clip_button = None
        self._clip_status_label = None
        self._waveform_widget = None
        self._seek_slider = None
        self._loop_checkbox = None
        self._position_label = None
        self._current_clip_event = None
        
        self._selected_events = []
        self._update_display()

