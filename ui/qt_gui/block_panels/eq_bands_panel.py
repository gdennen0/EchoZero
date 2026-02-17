"""
EQ Bands block panel.

Provides UI for configuring multi-band parametric EQ settings:
- Table widget with rows for each frequency band (low Hz, high Hz, gain dB)
- Add / Remove band buttons
- Filter order control

Uses EQBandsSettingsManager for unified settings handling.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QDoubleSpinBox, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Spacing, Colors
from src.utils.message import Log

from src.application.blocks.eq_bands_block import DEFAULT_BANDS
from src.application.settings.eq_bands_settings import EQBandsSettingsManager


@register_block_panel("EQBands")
class EQBandsPanel(BlockPanelBase):
    """Panel for EQBands block configuration with table-based band editing."""

    def __init__(self, block_id: str, facade, parent=None):
        # Track programmatic table updates to avoid feedback loops
        self._updating_table = False

        # Call parent init first (sets up UI structure)
        super().__init__(block_id, facade, parent)

        # Initialize settings manager AFTER parent init
        self._settings_manager = EQBandsSettingsManager(facade, block_id, parent=self)
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create EQBands-specific UI with band table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # -- Description --
        desc_label = QLabel(
            "Configure frequency bands with individual gain.\n"
            "Each band boosts or cuts frequencies in the specified range."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;"
        )
        layout.addWidget(desc_label)

        # -- Bands Table Group --
        bands_group = QGroupBox("Frequency Bands")
        bands_layout = QVBoxLayout(bands_group)
        bands_layout.setSpacing(Spacing.SM)

        # Table widget
        self.bands_table = QTableWidget()
        self.bands_table.setColumnCount(3)
        self.bands_table.setHorizontalHeaderLabels(["Low (Hz)", "High (Hz)", "Gain (dB)"])
        self.bands_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bands_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.bands_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.bands_table.verticalHeader().setVisible(True)
        self.bands_table.setMinimumHeight(150)
        self.bands_table.setAlternatingRowColors(True)

        # Connect cell change signal
        self.bands_table.cellChanged.connect(self._on_cell_changed)

        bands_layout.addWidget(self.bands_table)

        # Add / Remove buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(Spacing.SM)

        self.add_band_btn = QPushButton("+ Add Band")
        self.add_band_btn.setToolTip("Add a new frequency band")
        self.add_band_btn.clicked.connect(self._on_add_band)
        btn_layout.addWidget(self.add_band_btn)

        self.remove_band_btn = QPushButton("- Remove Band")
        self.remove_band_btn.setToolTip("Remove the selected frequency band")
        self.remove_band_btn.clicked.connect(self._on_remove_band)
        btn_layout.addWidget(self.remove_band_btn)

        btn_layout.addStretch()

        self.reset_bands_btn = QPushButton("Reset to Default")
        self.reset_bands_btn.setToolTip("Reset bands to default 3-band configuration")
        self.reset_bands_btn.clicked.connect(self._on_reset_bands)
        btn_layout.addWidget(self.reset_bands_btn)

        bands_layout.addLayout(btn_layout)

        layout.addWidget(bands_group)

        # -- Filter Settings Group --
        filter_group = QGroupBox("Filter Settings")
        filter_layout = QFormLayout(filter_group)
        filter_layout.setSpacing(Spacing.SM)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(4)
        self.order_spin.setSingleStep(1)
        self.order_spin.setToolTip(
            "Butterworth filter order (steepness of band edges):\n"
            "  1 = gentle slope (6 dB/octave)\n"
            "  4 = default (24 dB/octave)\n"
            "  8 = very steep (48 dB/octave)"
        )
        self.order_spin.valueChanged.connect(self._on_order_changed)
        filter_layout.addRow("Filter Order:", self.order_spin)

        layout.addWidget(filter_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return widget

    def refresh(self):
        """Update UI with current block settings."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        if not self.block or not self._settings_manager.is_loaded():
            return

        try:
            bands = self._settings_manager.bands
            order = self._settings_manager.order
        except Exception as e:
            Log.error(f"EQBandsPanel: Failed to load settings: {e}")
            return

        # Block signals during update
        self._updating_table = True
        self.bands_table.blockSignals(True)
        self.order_spin.blockSignals(True)

        # Populate table
        self.bands_table.setRowCount(len(bands))
        for row, band in enumerate(bands):
            freq_low = float(band.get("freq_low", 20.0))
            freq_high = float(band.get("freq_high", 20000.0))
            gain_db = float(band.get("gain_db", 0.0))

            low_item = QTableWidgetItem(f"{freq_low:.1f}")
            low_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bands_table.setItem(row, 0, low_item)

            high_item = QTableWidgetItem(f"{freq_high:.1f}")
            high_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bands_table.setItem(row, 1, high_item)

            gain_item = QTableWidgetItem(f"{gain_db:+.1f}")
            gain_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bands_table.setItem(row, 2, gain_item)

        # Set order
        self.order_spin.setValue(order)

        # Update remove button state
        self.remove_band_btn.setEnabled(len(bands) > 0)

        # Unblock signals
        self.bands_table.blockSignals(False)
        self.order_spin.blockSignals(False)
        self._updating_table = False

        self.set_status_message(f"{len(bands)} band(s) configured")

    # =========================================================================
    # Event Handlers - Table Edits
    # =========================================================================

    def _on_cell_changed(self, row: int, column: int):
        """Handle direct cell editing in the table."""
        if self._updating_table:
            return

        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        item = self.bands_table.item(row, column)
        if item is None:
            return

        try:
            value = float(item.text())
        except ValueError:
            self.set_status_message("Invalid number - please enter a numeric value", error=True)
            self.refresh()
            return

        col_map = {0: "freq_low", 1: "freq_high", 2: "gain_db"}
        field_name = col_map.get(column)
        if not field_name:
            return

        try:
            kwargs = {field_name: value}
            self._settings_manager.update_band(row, **kwargs)
            band = self._settings_manager.bands[row]
            self.set_status_message(
                f"Band {row + 1}: {band['freq_low']:.0f}-{band['freq_high']:.0f} Hz, "
                f"{band['gain_db']:+.1f} dB",
                error=False,
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    # =========================================================================
    # Event Handlers - Buttons
    # =========================================================================

    def _on_add_band(self):
        """Add a new frequency band."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        # Default new band: pick a reasonable range based on existing bands
        bands = self._settings_manager.bands
        if bands:
            last_band = bands[-1]
            # Start new band after the last one
            new_low = float(last_band.get("freq_high", 4000.0))
            new_high = min(new_low * 2.0, 20000.0)
            if new_low >= 19999.0:
                new_low = 1000.0
                new_high = 4000.0
        else:
            new_low = 1000.0
            new_high = 4000.0

        try:
            self._settings_manager.add_band(
                freq_low=new_low,
                freq_high=new_high,
                gain_db=0.0,
            )
            self.refresh()
            self.set_status_message(
                f"Added band: {new_low:.0f}-{new_high:.0f} Hz", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)

    def _on_remove_band(self):
        """Remove the selected band."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        selected = self.bands_table.currentRow()
        if selected < 0:
            self.set_status_message("Select a band to remove", error=True)
            return

        try:
            band = self._settings_manager.bands[selected]
            self._settings_manager.remove_band(selected)
            self.refresh()
            self.set_status_message(
                f"Removed band: {band['freq_low']:.0f}-{band['freq_high']:.0f} Hz",
                error=False,
            )
        except (ValueError, IndexError) as e:
            self.set_status_message(str(e), error=True)

    def _on_reset_bands(self):
        """Reset bands to the default 3-band configuration."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        try:
            self._settings_manager.bands = list(DEFAULT_BANDS)
            self.refresh()
            self.set_status_message("Reset to default 3-band configuration", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)

    def _on_order_changed(self, value: int):
        """Handle filter order change."""
        try:
            self._settings_manager.order = value
            db_per_oct = value * 6
            self.set_status_message(
                f"Filter order set to {value} ({db_per_oct} dB/octave)", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    # =========================================================================
    # External Updates
    # =========================================================================

    def refresh_for_undo(self):
        """Refresh panel after undo/redo operation."""
        if hasattr(self, "_settings_manager") and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()

    def _on_block_updated(self, event):
        """Handle block update event - reload settings and refresh UI."""
        updated_block_id = event.data.get("id")
        if updated_block_id == self.block_id:
            if self._is_saving:
                Log.debug(
                    f"EQBandsPanel: Skipping refresh during save for {self.block_id}"
                )
                return

            Log.debug(
                f"EQBandsPanel: Block {self.block_id} updated externally, refreshing UI"
            )

            # Reload block data from database
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(
                    f"EQBandsPanel: Failed to reload block {self.block_id}"
                )
                return

            # Reload settings from database
            if hasattr(self, "_settings_manager") and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug("EQBandsPanel: Settings manager reloaded from database")
            else:
                Log.warning("EQBandsPanel: Settings manager not available")
                return

            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        relevant_settings = ["bands", "order"]
        if setting_name in relevant_settings:
            self.refresh()
