"""
DetectOnsets block panel.

Provides UI for configuring onset detection settings.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox, QFrame, QStackedWidget
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.application.settings.detect_onsets_settings import DetectOnsetsSettingsManager
from src.utils.message import Log


@register_block_panel("DetectOnsets")
class DetectOnsetsPanel(BlockPanelBase):
    """Panel for DetectOnsets block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = DetectOnsetsSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create DetectOnsets-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Info group
        info_group = QGroupBox("About Onset Detection")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Onset detection identifies the start times of musical events "
            "(notes, drum hits, etc.) in audio.\n\n"
            "The onset detector analyzes the audio signal to find sudden "
            "changes in energy or spectral content.\n\n"
            "Clip end detection always runs to find the end of each event "
            "by analyzing energy decay. All events have duration from onset "
            "to end time. The 'Output Mode' setting controls visual display "
            "(marker vs clip), not the data structure."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11pt;")
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Detection settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setSpacing(Spacing.SM)
        
        # Onset method
        self.method_combo = QComboBox()
        self.method_combo.addItem("Default (Auto)", "default")
        self.method_combo.addItem("Energy-based", "energy")
        self.method_combo.addItem("Spectral Flux", "flux")
        self.method_combo.addItem("High Frequency Content", "hfc")
        self.method_combo.addItem("Complex Domain", "complex")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.method_combo.setToolTip(
            "Detection method for finding onset start times:\n"
            "• Default (Auto): Automatically selects best method for your audio\n"
            "• Energy-based: Good for percussive sounds (drums, hits)\n"
            "• Spectral Flux: Good for pitched instruments (piano, guitar)\n"
            "• High Frequency Content: Good for sharp transients\n"
            "• Complex Domain: Advanced method for complex audio"
        )
        detection_layout.addRow("Method:", self.method_combo)
        
        # Threshold
        threshold_row = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setMinimumWidth(100)
        self.threshold_spin.setMaximumWidth(150)
        # Use editingFinished to only trigger when user finishes typing (Enter or focus lost)
        self.threshold_spin.editingFinished.connect(self._on_threshold_editing_finished)
        self.threshold_spin.setToolTip(
            "Sensitivity threshold for onset detection (0.0-1.0):\n"
            "• Lower values (0.1-0.3): More sensitive, detects more onsets\n"
            "  Use when missing weak onsets or quiet events\n"
            "• Medium values (0.4-0.6): Balanced detection\n"
            "  Good starting point for most audio\n"
            "• Higher values (0.7-0.9): Less sensitive, only strong onsets\n"
            "  Use when detecting false positives or noise\n\n"
            "Tip: Start at 0.5 and adjust based on results"
        )
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        threshold_widget = QWidget()
        threshold_widget.setLayout(threshold_row)
        detection_layout.addRow("Threshold:", threshold_widget)
        
        # Min silence
        silence_row = QHBoxLayout()
        self.silence_spin = QDoubleSpinBox()
        self.silence_spin.setRange(0.0, 1.0)
        self.silence_spin.setSingleStep(0.01)
        self.silence_spin.setValue(0.02)
        self.silence_spin.setDecimals(3)
        self.silence_spin.setSuffix(" sec")
        self.silence_spin.setMinimumWidth(100)
        self.silence_spin.setMaximumWidth(150)
        self.silence_spin.editingFinished.connect(self._on_silence_editing_finished)
        self.silence_spin.setToolTip(
            "Minimum silence duration between onsets (in seconds):\n"
            "• Prevents detecting multiple onsets from the same event\n"
            "• Lower values (0.01-0.05s): Allow rapid events (fast drums, arpeggios)\n"
            "• Higher values (0.1-0.5s): Filter out rapid repeats\n\n"
            "Default: 0.02s (20ms) works well for most audio"
        )
        silence_row.addWidget(self.silence_spin)
        silence_help = QLabel("(Prevents duplicates)")
        silence_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        silence_help.setToolTip("Hover over spinbox for detailed help")
        silence_row.addWidget(silence_help)
        silence_row.addStretch()
        silence_widget = QWidget()
        silence_widget.setLayout(silence_row)
        detection_layout.addRow("Min Silence:", silence_widget)
        
        # Use backtrack checkbox
        backtrack_row = QHBoxLayout()
        self.use_backtrack_checkbox = QCheckBox()
        self.use_backtrack_checkbox.setChecked(True)
        self.use_backtrack_checkbox.stateChanged.connect(self._on_use_backtrack_changed)
        self.use_backtrack_checkbox.setToolTip(
            "Enable librosa backtrack to align onsets to energy minima:\n"
            "• ON (default): More accurate onset times, aligns to actual attack start\n"
            "  Better for: Precise timing, accurate clip boundaries\n"
            "• OFF: Uses raw onset detection without backtrack refinement\n"
            "  May be slightly faster but less accurate\n\n"
            "Backtrack finds the local energy minimum before the detected onset,\n"
            "providing better alignment with the actual start of the audio event."
        )
        backtrack_row.addWidget(self.use_backtrack_checkbox)
        backtrack_help = QLabel("(Better alignment)")
        backtrack_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        backtrack_help.setToolTip("Hover over checkbox for detailed help")
        backtrack_row.addWidget(backtrack_help)
        backtrack_row.addStretch()
        backtrack_widget = QWidget()
        backtrack_widget.setLayout(backtrack_row)
        detection_layout.addRow("Use Backtrack:", backtrack_widget)
        
        layout.addWidget(detection_group)
        
        # Audio preprocessing group
        preprocessing_group = QGroupBox("Audio Preprocessing")
        preprocessing_layout = QFormLayout(preprocessing_group)
        preprocessing_layout.setSpacing(Spacing.SM)
        
        preprocessing_help = QLabel(
            "Signal enhancement techniques to improve onset detection accuracy, "
            "especially for closely-spaced onsets. Preprocessing emphasizes transients "
            "and improves signal quality before onset detection."
        )
        preprocessing_help.setWordWrap(True)
        preprocessing_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        preprocessing_layout.addRow("", preprocessing_help)
        
        # Preprocessing enabled (master switch)
        preprocessing_enabled_row = QHBoxLayout()
        self.preprocessing_enabled_checkbox = QCheckBox()
        self.preprocessing_enabled_checkbox.setChecked(True)
        self.preprocessing_enabled_checkbox.stateChanged.connect(self._on_preprocessing_enabled_changed)
        self.preprocessing_enabled_checkbox.setToolTip(
            "Enable/disable all audio preprocessing:\n"
            "• ON (default): Apply preprocessing to improve onset detection\n"
            "• OFF: Skip preprocessing, use raw audio signal\n\n"
            "When disabled, all preprocessing steps below are ignored."
        )
        preprocessing_enabled_row.addWidget(self.preprocessing_enabled_checkbox)
        preprocessing_enabled_help = QLabel("(Master switch)")
        preprocessing_enabled_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        preprocessing_enabled_help.setToolTip("Hover over checkbox for detailed help")
        preprocessing_enabled_row.addWidget(preprocessing_enabled_help)
        preprocessing_enabled_row.addStretch()
        preprocessing_enabled_widget = QWidget()
        preprocessing_enabled_widget.setLayout(preprocessing_enabled_row)
        preprocessing_layout.addRow("Enable Preprocessing:", preprocessing_enabled_widget)
        
        # Pre-emphasis filter
        preemphasis_row = QHBoxLayout()
        self.preemphasis_checkbox = QCheckBox()
        self.preemphasis_checkbox.setChecked(True)
        self.preemphasis_checkbox.stateChanged.connect(self._on_preemphasis_enabled_changed)
        self.preemphasis_checkbox.setToolTip(
            "Pre-emphasis filter (high-pass) to emphasize transients:\n"
            "• ON (default): Emphasizes high-frequency content (attacks/transients)\n"
            "  Helps separate closely-spaced onsets by making transients more distinct\n"
            "• OFF: Skip pre-emphasis, use original signal\n\n"
            "Most important preprocessing step for closely-spaced onsets.\n"
            "Standard technique in speech/audio processing."
        )
        preemphasis_row.addWidget(self.preemphasis_checkbox)
        preemphasis_help = QLabel("(Emphasizes transients)")
        preemphasis_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        preemphasis_help.setToolTip("Hover over checkbox for detailed help")
        preemphasis_row.addWidget(preemphasis_help)
        preemphasis_row.addStretch()
        preemphasis_widget = QWidget()
        preemphasis_widget.setLayout(preemphasis_row)
        preprocessing_layout.addRow("Pre-emphasis Filter:", preemphasis_widget)
        
        # Pre-emphasis coefficient
        self.preemphasis_coefficient_spin = QDoubleSpinBox()
        self.preemphasis_coefficient_spin.setRange(0.0, 1.0)
        self.preemphasis_coefficient_spin.setSingleStep(0.01)
        self.preemphasis_coefficient_spin.setValue(0.97)
        self.preemphasis_coefficient_spin.setDecimals(2)
        self.preemphasis_coefficient_spin.setMinimumWidth(100)
        self.preemphasis_coefficient_spin.setMaximumWidth(150)
        self.preemphasis_coefficient_spin.editingFinished.connect(self._on_preemphasis_coefficient_editing_finished)
        self.preemphasis_coefficient_spin.setToolTip(
            "Pre-emphasis coefficient (0.0-1.0):\n"
            "Controls how much high-frequency content is emphasized.\n\n"
            "• Lower values (0.90-0.95): Less emphasis, more subtle\n"
            "• Medium values (0.95-0.97): Standard range (default: 0.97)\n"
            "• Higher values (0.97-0.99): Strong emphasis, more aggressive\n\n"
            "Typical values: 0.95-0.97 for most audio.\n"
            "Only used when Pre-emphasis Filter is enabled."
        )
        preprocessing_layout.addRow("Pre-emphasis Coefficient:", self.preemphasis_coefficient_spin)
        
        # DC offset removal
        dc_offset_row = QHBoxLayout()
        self.remove_dc_offset_checkbox = QCheckBox()
        self.remove_dc_offset_checkbox.setChecked(True)
        self.remove_dc_offset_checkbox.stateChanged.connect(self._on_remove_dc_offset_changed)
        self.remove_dc_offset_checkbox.setToolTip(
            "Remove DC offset (bias) from audio signal:\n"
            "• ON (default): Remove DC bias, always safe to do\n"
            "• OFF: Keep DC offset (may affect energy calculations)\n\n"
            "DC offset can affect energy-based detection, so removal is recommended.\n"
            "Minimal overhead, no negative effects."
        )
        dc_offset_row.addWidget(self.remove_dc_offset_checkbox)
        dc_offset_help = QLabel("(Always safe)")
        dc_offset_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        dc_offset_help.setToolTip("Hover over checkbox for detailed help")
        dc_offset_row.addWidget(dc_offset_help)
        dc_offset_row.addStretch()
        dc_offset_widget = QWidget()
        dc_offset_widget.setLayout(dc_offset_row)
        preprocessing_layout.addRow("Remove DC Offset:", dc_offset_widget)
        
        # High-pass filter
        highpass_row = QHBoxLayout()
        self.highpass_checkbox = QCheckBox()
        self.highpass_checkbox.setChecked(False)
        self.highpass_checkbox.stateChanged.connect(self._on_highpass_enabled_changed)
        self.highpass_checkbox.setToolTip(
            "High-pass filter to remove low-frequency content:\n"
            "• ON: Remove frequencies below cutoff (removes low-frequency noise)\n"
            "  Use when low-frequency noise interferes with detection\n"
            "• OFF (default): Keep all frequencies\n\n"
            "Optional: Only enable if low-frequency noise is masking transients.\n"
            "Typical cutoff: 80-200 Hz."
        )
        highpass_row.addWidget(self.highpass_checkbox)
        highpass_help = QLabel("(Optional)")
        highpass_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        highpass_help.setToolTip("Hover over checkbox for detailed help")
        highpass_row.addWidget(highpass_help)
        highpass_row.addStretch()
        highpass_widget = QWidget()
        highpass_widget.setLayout(highpass_row)
        preprocessing_layout.addRow("High-Pass Filter:", highpass_widget)
        
        # High-pass cutoff
        self.highpass_cutoff_spin = QDoubleSpinBox()
        self.highpass_cutoff_spin.setRange(20.0, 500.0)
        self.highpass_cutoff_spin.setSingleStep(10.0)
        self.highpass_cutoff_spin.setValue(80.0)
        self.highpass_cutoff_spin.setDecimals(0)
        self.highpass_cutoff_spin.setSuffix(" Hz")
        self.highpass_cutoff_spin.setMinimumWidth(100)
        self.highpass_cutoff_spin.setMaximumWidth(150)
        self.highpass_cutoff_spin.editingFinished.connect(self._on_highpass_cutoff_editing_finished)
        self.highpass_cutoff_spin.setToolTip(
            "High-pass cutoff frequency (Hz):\n"
            "Frequencies below this value will be filtered out.\n\n"
            "• Lower values (40-80 Hz): Removes only very low frequencies\n"
            "  Good for: Preserving bass content while removing rumble\n"
            "• Medium values (80-200 Hz): Standard range (default: 80 Hz)\n"
            "  Good for: Most audio, removes low-frequency noise\n"
            "• Higher values (200-500 Hz): Aggressive filtering\n"
            "  Good for: Very noisy audio, but may affect sound quality\n\n"
            "Only used when High-Pass Filter is enabled."
        )
        preprocessing_layout.addRow("High-Pass Cutoff:", self.highpass_cutoff_spin)
        
        # Normalization
        normalize_row = QHBoxLayout()
        self.normalize_audio_checkbox = QCheckBox()
        self.normalize_audio_checkbox.setChecked(False)
        self.normalize_audio_checkbox.stateChanged.connect(self._on_normalize_audio_changed)
        self.normalize_audio_checkbox.setToolTip(
            "Normalize audio to ensure consistent signal levels:\n"
            "• ON: Normalize audio before detection\n"
            "  Helps with threshold-based detection when audio levels vary\n"
            "• OFF (default): Use original signal levels\n\n"
            "Optional: Only enable if audio levels are inconsistent.\n"
            "Can help when some parts of audio are much quieter than others."
        )
        normalize_row.addWidget(self.normalize_audio_checkbox)
        normalize_help = QLabel("(Optional)")
        normalize_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        normalize_help.setToolTip("Hover over checkbox for detailed help")
        normalize_row.addWidget(normalize_help)
        normalize_row.addStretch()
        normalize_widget = QWidget()
        normalize_widget.setLayout(normalize_row)
        preprocessing_layout.addRow("Normalize Audio:", normalize_widget)
        
        # Normalization method
        self.normalization_method_combo = QComboBox()
        self.normalization_method_combo.addItem("Peak", "peak")
        self.normalization_method_combo.addItem("RMS", "rms")
        self.normalization_method_combo.currentIndexChanged.connect(self._on_normalization_method_changed)
        self.normalization_method_combo.setToolTip(
            "Normalization method:\n"
            "• Peak: Scale to [-1, 1] range (normalize to maximum amplitude)\n"
            "  Good for: Ensuring consistent peak levels\n"
            "• RMS: Scale to target RMS level (normalize average energy)\n"
            "  Good for: Preserving relative dynamics\n\n"
            "Only used when Normalize Audio is enabled."
        )
        preprocessing_layout.addRow("Normalization Method:", self.normalization_method_combo)
        
        layout.addWidget(preprocessing_group)
        
        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        output_layout.setSpacing(Spacing.SM)
        
        # Output mode
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("Markers (visual)", "markers")
        self.output_mode_combo.addItem("Clips (visual)", "clips")
        self.output_mode_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        self.output_mode_combo.setToolTip(
            "Visual display mode for detected events:\n"
            "• Markers: Events displayed as markers/diamonds\n"
            "  All events still have duration (clip end detection always runs)\n"
            "  Use for: Visual distinction when you want marker-style display\n\n"
            "• Clips: Events displayed as clips/ranges\n"
            "  All events have duration from onset to energy decay\n"
            "  Use for: Visual distinction when you want clip-style display\n\n"
            "Note: This setting applies to all events created by this block.\n"
            "Individual events can have their display mode changed in the Editor."
        )
        output_layout.addRow("Output Mode:", self.output_mode_combo)
        
        # Clip classification name
        clip_name_row = QHBoxLayout()
        self.clip_classification_edit = QLineEdit()
        self.clip_classification_edit.setText("clip")
        self.clip_classification_edit.setMinimumWidth(150)
        self.clip_classification_edit.setMaximumWidth(200)
        self.clip_classification_edit.editingFinished.connect(self._on_clip_classification_changed)
        self.clip_classification_edit.setToolTip(
            "Classification name for clip events:\n"
            "This name will be used as the classification for all detected clip events.\n\n"
            "Examples:\n"
            "• 'clip' - Generic audio clips\n"
            "• 'note' - Musical notes\n"
            "• 'sample' - Audio samples\n"
            "• 'hit' - Percussive hits\n\n"
            "Only used when Output Mode is set to 'Clips'."
        )
        clip_name_row.addWidget(self.clip_classification_edit)
        clip_name_help = QLabel("(Event classification)")
        clip_name_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        clip_name_help.setToolTip("Hover over text field for detailed help")
        clip_name_row.addWidget(clip_name_help)
        clip_name_row.addStretch()
        clip_name_widget = QWidget()
        clip_name_widget.setLayout(clip_name_row)
        output_layout.addRow("Clip Name:", clip_name_widget)
        
        layout.addWidget(output_group)
        
        # Clip detection settings group (only visible in clips mode)
        self.clip_detection_group = QGroupBox("Clip Detection")
        clip_detection_layout = QFormLayout(self.clip_detection_group)
        clip_detection_layout.setSpacing(Spacing.SM)
        
        # Section: Duration Constraints
        duration_label = QLabel("Duration Constraints")
        duration_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; font-size: 11pt;")
        clip_detection_layout.addRow(duration_label)
        
        duration_help = QLabel("Safety limits for clip duration (applied after end detection)")
        duration_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        clip_detection_layout.addRow("", duration_help)
        
        # Max clip duration
        self.max_clip_duration_spin = QDoubleSpinBox()
        self.max_clip_duration_spin.setRange(0.01, 10.0)
        self.max_clip_duration_spin.setSingleStep(0.1)
        self.max_clip_duration_spin.setValue(2.0)
        self.max_clip_duration_spin.setDecimals(2)
        self.max_clip_duration_spin.setSuffix(" sec")
        self.max_clip_duration_spin.setMinimumWidth(100)
        self.max_clip_duration_spin.setMaximumWidth(150)
        self.max_clip_duration_spin.editingFinished.connect(self._on_max_clip_duration_editing_finished)
        self.max_clip_duration_spin.setToolTip(
            "Maximum clip duration (failsafe, in seconds):\n"
            "Prevents clips from extending too long if energy decay detection fails.\n\n"
            "• Percussive hits: Typically 0.1-2 seconds\n"
            "• Sustained notes: Can be longer (2-5 seconds)\n"
            "• Very long sounds: May need 5-10 seconds\n\n"
            "Clips longer than this will be truncated to prevent excessive duration.\n"
            "Default: 2.0 seconds works well for most percussive sounds."
        )
        clip_detection_layout.addRow("Max Duration:", self.max_clip_duration_spin)
        
        # Min clip duration
        self.min_clip_duration_spin = QDoubleSpinBox()
        self.min_clip_duration_spin.setRange(0.001, 1.0)
        self.min_clip_duration_spin.setSingleStep(0.01)
        self.min_clip_duration_spin.setValue(0.01)
        self.min_clip_duration_spin.setDecimals(3)
        self.min_clip_duration_spin.setSuffix(" sec")
        self.min_clip_duration_spin.setMinimumWidth(100)
        self.min_clip_duration_spin.setMaximumWidth(150)
        self.min_clip_duration_spin.editingFinished.connect(self._on_min_clip_duration_editing_finished)
        self.min_clip_duration_spin.setToolTip(
            "Minimum clip duration (in seconds):\n"
            "Very short clips (< 10ms) are often noise or false detections.\n\n"
            "• Clips shorter than this will be extended to the minimum duration\n"
            "• Prevents creating unusably short clips\n"
            "• Default: 0.01s (10ms) - minimum practical duration for audio clips\n\n"
            "Tip: Increase if detecting very short noise bursts"
        )
        clip_detection_layout.addRow("Min Duration:", self.min_clip_duration_spin)
        
        # Separator between duration constraints and clip end detection
        separator_duration = QFrame()
        separator_duration.setFrameShape(QFrame.Shape.HLine)
        separator_duration.setFrameShadow(QFrame.Shadow.Sunken)
        clip_detection_layout.addRow(separator_duration)
        
        # Section: Clip End Detection - Top level selection
        clip_end_label = QLabel("Clip End Detection")
        clip_end_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; font-size: 11pt;")
        clip_detection_layout.addRow(clip_end_label)
        
        clip_end_help = QLabel("How to detect where the sound stops (determines clip end time). Only one method can be active at a time.")
        clip_end_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        clip_detection_layout.addRow("", clip_end_help)
        
        # Top-level detection method selection
        detection_method_row = QHBoxLayout()
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems([
            "Librosa End Detection",
            "Decay Modes"
        ])
        self.detection_mode_combo.setCurrentText("Librosa End Detection")
        self.detection_mode_combo.currentTextChanged.connect(self._on_detection_mode_changed)
        self.detection_mode_combo.setToolTip(
            "Clip End Detection Method:\n"
            "Choose how to detect when a clip ends. Only one method can be active at a time.\n\n"
            "• Librosa End Detection: Uses librosa's energy-based detection\n"
            "  Waits for multiple consecutive frames below threshold\n"
            "  Better for: Capturing tails, reverb, full decay\n\n"
            "• Decay Modes: Cuts immediately when energy drops to % of peak\n"
            "  Better for: Rapid hits, tight cuts, avoiding multiple hits in one clip"
        )
        self.detection_mode_combo.setMinimumWidth(200)
        self.detection_mode_combo.setMaximumWidth(250)
        detection_method_row.addWidget(self.detection_mode_combo)
        detection_method_help = QLabel("(Select detection method)")
        detection_method_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        detection_method_help.setToolTip("Hover over combo box for detailed help")
        detection_method_row.addWidget(detection_method_help)
        detection_method_row.addStretch()
        detection_method_widget = QWidget()
        detection_method_widget.setLayout(detection_method_row)
        clip_detection_layout.addRow("Detection Method:", detection_method_widget)
        
        # Create stacked widget to flip between the two detection modes
        self.detection_mode_stack = QStackedWidget()
        
        # Page 0: Librosa End Detection
        librosa_page = QWidget()
        librosa_page_layout = QVBoxLayout(librosa_page)
        librosa_page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group 1: Librosa End Detection
        self.librosa_group = QGroupBox("Librosa End Detection")
        librosa_group_layout = QFormLayout(self.librosa_group)
        librosa_group_layout.setSpacing(Spacing.SM)
        
        librosa_help = QLabel("Uses librosa's energy-based detection. Waits for multiple consecutive frames below threshold.")
        librosa_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        librosa_group_layout.addRow("", librosa_help)
        
        # Energy decay threshold
        self.energy_decay_threshold_spin = QDoubleSpinBox()
        self.energy_decay_threshold_spin.setRange(0.0, 1.0)
        self.energy_decay_threshold_spin.setSingleStep(0.05)
        self.energy_decay_threshold_spin.setValue(0.1)
        self.energy_decay_threshold_spin.setDecimals(2)
        self.energy_decay_threshold_spin.setSuffix(" (10% = 0.1)")
        self.energy_decay_threshold_spin.setMinimumWidth(100)
        self.energy_decay_threshold_spin.setMaximumWidth(150)
        self.energy_decay_threshold_spin.editingFinished.connect(self._on_energy_decay_threshold_editing_finished)
        self.energy_decay_threshold_spin.setToolTip(
            "Energy decay threshold for detecting clip end times (0.0-1.0):\n"
            "This determines when a percussive event has ended by measuring "
            "when the audio energy decays below a percentage of the peak energy.\n\n"
            "• Lower values (0.05-0.15): Longer clips, includes more tail/reverb\n"
            "  Use for: Sustained sounds, reverb-heavy audio, full decay\n"
            "• Medium values (0.15-0.25): Balanced, cuts most tails\n"
            "  Good starting point for most percussive sounds\n"
            "• Higher values (0.25-0.5): Shorter clips, cuts early\n"
            "  Use for: Sharp transients, tight cuts, minimal tail\n\n"
            "Default: 0.1 (10% of peak) - cuts when energy drops to 10% of maximum"
        )
        librosa_group_layout.addRow("Decay Threshold:", self.energy_decay_threshold_spin)
        
        # Adaptive threshold checkbox
        self.adaptive_threshold_checkbox = QCheckBox()
        self.adaptive_threshold_checkbox.setChecked(True)
        self.adaptive_threshold_checkbox.stateChanged.connect(self._on_adaptive_threshold_changed)
        self.adaptive_threshold_checkbox.setToolTip(
            "Enable adaptive threshold for quiet events (like soft hi-hats):\n"
            "• Uses local energy context, not just peak energy\n"
            "• Better handles events with low absolute energy\n"
            "• Combines peak-relative and baseline-relative thresholds\n\n"
            "Disable to use simple peak-relative threshold only."
        )
        adaptive_row = QHBoxLayout()
        adaptive_row.addWidget(self.adaptive_threshold_checkbox)
        adaptive_help = QLabel("(Better for quiet events)")
        adaptive_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        adaptive_help.setToolTip("Hover over checkbox for detailed help")
        adaptive_row.addWidget(adaptive_help)
        adaptive_row.addStretch()
        adaptive_widget = QWidget()
        adaptive_widget.setLayout(adaptive_row)
        librosa_group_layout.addRow("Adaptive Threshold:", adaptive_widget)
        
        # Adaptive threshold factor
        self.adaptive_threshold_factor_spin = QDoubleSpinBox()
        self.adaptive_threshold_factor_spin.setRange(0.0, 1.0)
        self.adaptive_threshold_factor_spin.setSingleStep(0.1)
        self.adaptive_threshold_factor_spin.setValue(0.5)
        self.adaptive_threshold_factor_spin.setDecimals(2)
        self.adaptive_threshold_factor_spin.setMinimumWidth(100)
        self.adaptive_threshold_factor_spin.setMaximumWidth(150)
        self.adaptive_threshold_factor_spin.editingFinished.connect(self._on_adaptive_threshold_factor_editing_finished)
        self.adaptive_threshold_factor_spin.setToolTip(
            "Adaptive threshold factor (0.0-1.0):\n"
            "Controls how much baseline energy affects the threshold.\n\n"
            "• Lower values (0.2-0.4): More peak-relative, better for loud events\n"
            "• Medium values (0.4-0.6): Balanced (default: 0.5)\n"
            "• Higher values (0.6-0.8): More baseline-relative, better for quiet events\n\n"
            "Only used when Adaptive Threshold is enabled."
        )
        librosa_group_layout.addRow("Adaptive Factor:", self.adaptive_threshold_factor_spin)
        
        # Onset lookahead time
        self.onset_lookahead_spin = QDoubleSpinBox()
        self.onset_lookahead_spin.setRange(0.0, 1.0)
        self.onset_lookahead_spin.setSingleStep(0.01)
        self.onset_lookahead_spin.setValue(0.1)
        self.onset_lookahead_spin.setDecimals(3)
        self.onset_lookahead_spin.setSuffix(" sec")
        self.onset_lookahead_spin.setMinimumWidth(100)
        self.onset_lookahead_spin.setMaximumWidth(150)
        self.onset_lookahead_spin.editingFinished.connect(self._on_onset_lookahead_editing_finished)
        self.onset_lookahead_spin.setToolTip(
            "Onset lookahead time (in seconds):\n"
            "How far ahead to look for the next onset when detecting clip end.\n\n"
            "• Lower values (0.05-0.1s): Faster detection, may miss very quick onsets\n"
            "• Medium values (0.1-0.2s): Balanced (default: 0.1s)\n"
            "• Higher values (0.2-0.5s): Better for detecting rapid successive hits\n\n"
            "Used to detect when a new hit starts and cut the previous clip before it."
        )
        librosa_group_layout.addRow("Onset Lookahead:", self.onset_lookahead_spin)
        
        # Energy rise threshold
        self.energy_rise_threshold_spin = QDoubleSpinBox()
        self.energy_rise_threshold_spin.setRange(1.0, 5.0)
        self.energy_rise_threshold_spin.setSingleStep(0.1)
        self.energy_rise_threshold_spin.setValue(1.5)
        self.energy_rise_threshold_spin.setDecimals(2)
        self.energy_rise_threshold_spin.setSuffix("x")
        self.energy_rise_threshold_spin.setMinimumWidth(100)
        self.energy_rise_threshold_spin.setMaximumWidth(150)
        self.energy_rise_threshold_spin.editingFinished.connect(self._on_energy_rise_threshold_editing_finished)
        self.energy_rise_threshold_spin.setToolTip(
            "Energy rise threshold (1.0+):\n"
            "Factor by which energy must rise to indicate a new onset.\n\n"
            "• Lower values (1.2-1.5): More sensitive, detects smaller energy increases\n"
            "  Use when onsets are close together (default: 1.5)\n"
            "• Higher values (1.5-2.5): Less sensitive, only strong energy rises\n"
            "  Use when there's background noise or reverb\n\n"
            "Energy must rise by this factor from decay level to trigger new onset detection."
        )
        librosa_group_layout.addRow("Energy Rise Threshold:", self.energy_rise_threshold_spin)
        
        # Minimum separation time
        self.min_separation_spin = QDoubleSpinBox()
        self.min_separation_spin.setRange(0.0, 0.5)
        self.min_separation_spin.setSingleStep(0.01)
        self.min_separation_spin.setValue(0.02)
        self.min_separation_spin.setDecimals(3)
        self.min_separation_spin.setSuffix(" sec")
        self.min_separation_spin.setMinimumWidth(100)
        self.min_separation_spin.setMaximumWidth(150)
        self.min_separation_spin.editingFinished.connect(self._on_min_separation_editing_finished)
        self.min_separation_spin.setToolTip(
            "Minimum separation time (in seconds):\n"
            "Minimum time to cut before the next onset to ensure proper separation.\n\n"
            "• Lower values (0.01-0.02s): Tighter cuts, may overlap if onsets are very close\n"
            "• Medium values (0.02-0.05s): Balanced separation (default: 0.02s)\n"
            "• Higher values (0.05-0.1s): More separation, prevents any overlap\n\n"
            "Ensures clips don't extend into the next onset's attack phase."
        )
        librosa_group_layout.addRow("Min Separation:", self.min_separation_spin)
        
        librosa_page_layout.addWidget(self.librosa_group)
        librosa_page_layout.addStretch()
        self.detection_mode_stack.addWidget(librosa_page)
        
        # Page 1: Decay Modes
        decay_modes_page = QWidget()
        decay_modes_page_layout = QVBoxLayout(decay_modes_page)
        decay_modes_page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group 2: Decay Modes
        self.decay_modes_group = QGroupBox("Decay Modes")
        decay_modes_group_layout = QFormLayout(self.decay_modes_group)
        decay_modes_group_layout.setSpacing(Spacing.SM)
        
        decay_modes_help = QLabel("Cuts immediately when energy drops to % of peak. Better for rapid hits and tight cuts.")
        decay_modes_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        decay_modes_group_layout.addRow("", decay_modes_help)
        
        # Peak decay ratio
        self.peak_decay_ratio_spin = QDoubleSpinBox()
        self.peak_decay_ratio_spin.setRange(0.01, 1.0)
        self.peak_decay_ratio_spin.setSingleStep(0.01)
        self.peak_decay_ratio_spin.setValue(0.5)
        self.peak_decay_ratio_spin.setDecimals(3)
        self.peak_decay_ratio_spin.setSuffix(" (50% = 0.5)")
        self.peak_decay_ratio_spin.setMinimumWidth(100)
        self.peak_decay_ratio_spin.setMaximumWidth(150)
        self.peak_decay_ratio_spin.editingFinished.connect(self._on_peak_decay_ratio_editing_finished)
        self.peak_decay_ratio_spin.setToolTip(
            "Peak decay ratio (0.01-1.0):\n"
            "Cut when energy drops to this percentage of peak energy.\n\n"
            "• Very low values (0.01-0.1): Extremely tight cuts, captures just the peak\n"
            "  Use for: Very sharp transients, minimal decay wanted\n"
            "• Lower values (0.2-0.4): Very tight cuts, cuts early in decay\n"
            "  Use for: Rapid hits, sharp transients, minimal tail\n"
            "• Medium values (0.4-0.6): Balanced early cuts (default: 0.5)\n"
            "  Good starting point for most percussive sounds\n"
            "• Higher values (0.6-0.9): Later cuts, includes more decay\n"
            "  Use for: Capturing more of the hit's character\n\n"
            "Example: 0.05 means cut when energy drops to 5% of peak."
        )
        decay_modes_group_layout.addRow("Peak Decay Ratio:", self.peak_decay_ratio_spin)

        # Decay detection method
        self.decay_method_combo = QComboBox()
        self.decay_method_combo.addItems([
            "threshold",
            "rate_of_change",
            "slope",
            "confirmed_threshold",
            "all"
        ])
        self.decay_method_combo.setCurrentText("threshold")
        self.decay_method_combo.currentTextChanged.connect(self._on_decay_method_changed)
        self.decay_method_combo.setToolTip(
            "Decay Detection Method:\n"
            "Method used to detect when energy decays after the peak.\n\n"
            "• threshold: Cut on first frame below peak_decay_ratio (simple, fast)\n"
            "  Default method, works well for most cases\n\n"
            "• rate_of_change: Detect rapid energy decrease (derivative-based)\n"
            "  Best for detecting when decay actually starts\n"
            "  Good for very tight cuts, catches initial falloff\n\n"
            "• slope: Detect consistent negative slope after peak\n"
            "  Uses sliding window to detect decay trend\n"
            "  More stable than single-frame methods\n\n"
            "• confirmed_threshold: Require 2-3 consecutive frames below threshold\n"
            "  More robust than simple threshold, reduces noise sensitivity\n"
            "  Balance between speed and accuracy\n\n"
            "• all: Try ALL methods and use the EARLIEST cut found\n"
            "  Most aggressive - cuts at first sign of decay from any method\n"
            "  Best for very tight cuts, catches decay as early as possible"
        )
        self.decay_method_combo.setMinimumWidth(150)
        self.decay_method_combo.setMaximumWidth(200)
        decay_modes_group_layout.addRow("Decay Detection Method:", self.decay_method_combo)
        
        decay_modes_page_layout.addWidget(self.decay_modes_group)
        decay_modes_page_layout.addStretch()
        self.detection_mode_stack.addWidget(decay_modes_page)
        
        # Add stacked widget to layout
        clip_detection_layout.addRow(self.detection_mode_stack)

        # Separator between clip end detection and advanced settings
        separator_advanced = QFrame()
        separator_advanced.setFrameShape(QFrame.Shape.HLine)
        separator_advanced.setFrameShadow(QFrame.Shadow.Sunken)
        clip_detection_layout.addRow(separator_advanced)
        
        # Section: Advanced Clip End Detection Settings
        advanced_label = QLabel("Advanced Clip End Detection")
        advanced_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; font-size: 11pt;")
        clip_detection_layout.addRow(advanced_label)
        
        advanced_help = QLabel("Fine-tuning for clip end detection (optional)")
        advanced_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic; padding-bottom: 8px;")
        clip_detection_layout.addRow("", advanced_help)
        
        # Use numba acceleration checkbox
        self.use_numba_checkbox = QCheckBox()
        self.use_numba_checkbox.setChecked(True)
        self.use_numba_checkbox.stateChanged.connect(self._on_use_numba_changed)
        self.use_numba_checkbox.setToolTip(
            "Enable numba JIT acceleration for faster clip end detection (3-10x speedup). "
            "Disable to use librosa/numpy only (slower but may be more compatible)."
        )
        clip_detection_layout.addRow("Use Numba Acceleration:", self.use_numba_checkbox)
        
        # Separator for enhanced features
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        clip_detection_layout.addRow(separator3)

        # Post-processing label
        postprocess_label = QLabel("Post-Processing")
        postprocess_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
        clip_detection_layout.addRow(postprocess_label)

        # Split clips with multiple onsets checkbox
        split_clips_row = QHBoxLayout()
        self.split_clips_checkbox = QCheckBox()
        self.split_clips_checkbox.setChecked(False)
        self.split_clips_checkbox.stateChanged.connect(self._on_split_clips_changed)
        self.split_clips_checkbox.setToolTip(
            "Enable post-processing to split clips with multiple onsets:\n"
            "After creating clips, run onset detection on each clip locally.\n"
            "If a clip contains more than 1 onset, split it into multiple clips.\n\n"
            "This is a safety net to catch cases where multiple hits ended up\n"
            "in one clip despite decay detection. Can be slow for many clips.\n\n"
            "• ON: Check each clip and split if multiple onsets found\n"
            "• OFF: Skip post-processing (default, faster)"
        )
        split_clips_row.addWidget(self.split_clips_checkbox)
        split_clips_help = QLabel("(Split clips with multiple hits)")
        split_clips_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; font-style: italic;")
        split_clips_help.setToolTip("Hover over checkbox for detailed help")
        split_clips_row.addWidget(split_clips_help)
        split_clips_row.addStretch()
        split_clips_widget = QWidget()
        split_clips_widget.setLayout(split_clips_row)
        clip_detection_layout.addRow("Split Clips with Multiple Onsets:", split_clips_widget)

        # Initially show Librosa End Detection (page 0)
        # This will be set correctly in refresh() based on settings
        self.detection_mode_stack.setCurrentIndex(0)
        
        # Initially hide (only show in clips mode)
        # Always show clip detection settings (clip end detection always runs)
        layout.addWidget(self.clip_detection_group)
        
        # Tips
        tips_label = QLabel(
            " Tips:\n"
            "• Lower threshold if missing onsets\n"
            "• Higher threshold if detecting false onsets\n"
            "• Increase min silence to avoid detecting rapid repeats\n"
            "• Enable preprocessing (default) to improve closely-spaced onset detection\n"
            "• Pre-emphasis filter (default ON) emphasizes transients for better separation\n"
            "• In Clips mode: Lower decay threshold for longer clips with reverb\n"
            "• In Clips mode: Higher decay threshold for tight cuts with minimal tail\n"
            "• Enable Adaptive Threshold for quiet events (soft hi-hats, quiet hits)\n"
            "• Increase Onset Lookahead if clips are extending into next hits\n"
            "• Lower Energy Rise Threshold if missing quick successive hits\n"
            "• Increase Min Separation if clips are overlapping"
        )
        tips_label.setWordWrap(True)
        tips_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(tips_label)
        
        # Execution Summary section
        summary_group = QGroupBox("Last Execution Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(Spacing.SM)
        
        self.summary_label = QLabel("No execution data available.\nRun the block to see results here.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current settings from settings manager"""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        if not self.block or not self._settings_manager.is_loaded():
            return
        
        # Load settings from settings manager (single source of truth)
        try:
            method = self._settings_manager.onset_method
            threshold = self._settings_manager.onset_threshold
            min_silence = self._settings_manager.min_silence
            output_mode = self._settings_manager.output_mode
            clip_classification = self._settings_manager.clip_classification
            max_clip_duration = self._settings_manager.max_clip_duration
            min_clip_duration = self._settings_manager.min_clip_duration
            energy_decay_threshold = self._settings_manager.energy_decay_threshold
            use_numba_acceleration = self._settings_manager.use_numba_acceleration
            adaptive_threshold_enabled = self._settings_manager.adaptive_threshold_enabled
            adaptive_threshold_factor = self._settings_manager.adaptive_threshold_factor
            onset_lookahead_time = self._settings_manager.onset_lookahead_time
            energy_rise_threshold = self._settings_manager.energy_rise_threshold
            min_separation_time = self._settings_manager.min_separation_time
            early_cut_mode = self._settings_manager.early_cut_mode
            peak_decay_ratio = self._settings_manager.peak_decay_ratio
            preprocessing_enabled = self._settings_manager.preprocessing_enabled
            preemphasis_enabled = self._settings_manager.preemphasis_enabled
            preemphasis_coefficient = self._settings_manager.preemphasis_coefficient
            remove_dc_offset = self._settings_manager.remove_dc_offset
            highpass_enabled = self._settings_manager.highpass_enabled
            highpass_cutoff = self._settings_manager.highpass_cutoff
            normalize_audio = self._settings_manager.normalize_audio
            normalization_method = self._settings_manager.normalization_method
        except Exception as e:
            Log.error(f"DetectOnsetsPanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.method_combo.blockSignals(True)
        self.threshold_spin.blockSignals(True)
        self.silence_spin.blockSignals(True)
        self.output_mode_combo.blockSignals(True)
        self.clip_classification_edit.blockSignals(True)
        self.max_clip_duration_spin.blockSignals(True)
        self.min_clip_duration_spin.blockSignals(True)
        self.energy_decay_threshold_spin.blockSignals(True)
        self.use_numba_checkbox.blockSignals(True)
        self.adaptive_threshold_checkbox.blockSignals(True)
        self.adaptive_threshold_factor_spin.blockSignals(True)
        self.onset_lookahead_spin.blockSignals(True)
        self.energy_rise_threshold_spin.blockSignals(True)
        self.min_separation_spin.blockSignals(True)
        self.detection_mode_combo.blockSignals(True)
        self.peak_decay_ratio_spin.blockSignals(True)
        if hasattr(self, 'decay_method_combo'):
            self.decay_method_combo.blockSignals(True)
        self.preprocessing_enabled_checkbox.blockSignals(True)
        self.preemphasis_checkbox.blockSignals(True)
        self.preemphasis_coefficient_spin.blockSignals(True)
        self.remove_dc_offset_checkbox.blockSignals(True)
        self.highpass_checkbox.blockSignals(True)
        self.highpass_cutoff_spin.blockSignals(True)
        self.normalize_audio_checkbox.blockSignals(True)
        self.normalization_method_combo.blockSignals(True)
        
        # Set method
        method_found = False
        for i in range(self.method_combo.count()):
            if self.method_combo.itemData(i) == method:
                self.method_combo.setCurrentIndex(i)
                method_found = True
                Log.debug(f"DetectOnsetsPanel: Set method combo to index {i} (value: {method})")
                break
        if not method_found:
            Log.warning(f"DetectOnsetsPanel: Method '{method}' not found in combo box")
        
        # Set threshold
        self.threshold_spin.setValue(threshold)
        Log.debug(f"DetectOnsetsPanel: Set threshold to {threshold}")
        
        # Set min silence
        self.silence_spin.setValue(min_silence)
        Log.debug(f"DetectOnsetsPanel: Set min_silence to {min_silence}")
        
        # Set output mode
        output_mode_found = False
        for i in range(self.output_mode_combo.count()):
            if self.output_mode_combo.itemData(i) == output_mode:
                self.output_mode_combo.setCurrentIndex(i)
                output_mode_found = True
                Log.debug(f"DetectOnsetsPanel: Set output_mode combo to index {i} (value: {output_mode})")
                break
        if not output_mode_found:
            Log.warning(f"DetectOnsetsPanel: Output mode '{output_mode}' not found in combo box")
        
        # Set clip classification
        self.clip_classification_edit.setText(clip_classification)
        Log.debug(f"DetectOnsetsPanel: Set clip_classification to {clip_classification}")
        
        # Get backtrack setting
        use_backtrack = self._settings_manager.use_backtrack
        
        # Set onset detection settings
        self.use_backtrack_checkbox.setChecked(use_backtrack)
        
        # Get early cut settings
        early_cut_mode = self._settings_manager.early_cut_mode
        peak_decay_ratio = self._settings_manager.peak_decay_ratio
        
        # Set clip detection settings
        self.max_clip_duration_spin.setValue(max_clip_duration)
        self.min_clip_duration_spin.setValue(min_clip_duration)
        self.energy_decay_threshold_spin.setValue(energy_decay_threshold)
        self.use_numba_checkbox.setChecked(use_numba_acceleration)
        self.adaptive_threshold_checkbox.setChecked(adaptive_threshold_enabled)
        self.adaptive_threshold_factor_spin.setValue(adaptive_threshold_factor)
        self.onset_lookahead_spin.setValue(onset_lookahead_time)
        self.energy_rise_threshold_spin.setValue(energy_rise_threshold)
        self.min_separation_spin.setValue(min_separation_time)
        
        # Set detection mode combo box based on early_cut_mode
        # "Librosa End Detection" -> early_cut_mode = False
        # "Decay Modes" -> early_cut_mode = True
        detection_mode = "Decay Modes" if early_cut_mode else "Librosa End Detection"
        if detection_mode in [self.detection_mode_combo.itemText(i) for i in range(self.detection_mode_combo.count())]:
            self.detection_mode_combo.setCurrentText(detection_mode)
        else:
            self.detection_mode_combo.setCurrentText("Librosa End Detection")  # Fallback to default
        
        # Switch stacked widget page based on mode
        # Page 0: Librosa End Detection
        # Page 1: Decay Modes
        if detection_mode == "Librosa End Detection":
            self.detection_mode_stack.setCurrentIndex(0)
        elif detection_mode == "Decay Modes":
            self.detection_mode_stack.setCurrentIndex(1)
        
        # Set values for both groups
        self.peak_decay_ratio_spin.setValue(peak_decay_ratio)
        
        # Load decay detection method
        decay_method = self._settings_manager.decay_detection_method
        if decay_method in [self.decay_method_combo.itemText(i) for i in range(self.decay_method_combo.count())]:
            self.decay_method_combo.setCurrentText(decay_method)
        else:
            self.decay_method_combo.setCurrentText("threshold")  # Fallback to default
        
        # Set preprocessing settings
        self.preprocessing_enabled_checkbox.setChecked(preprocessing_enabled)
        self.preemphasis_checkbox.setChecked(preemphasis_enabled)
        self.preemphasis_coefficient_spin.setValue(preemphasis_coefficient)
        self.remove_dc_offset_checkbox.setChecked(remove_dc_offset)
        self.highpass_checkbox.setChecked(highpass_enabled)
        self.highpass_cutoff_spin.setValue(highpass_cutoff)
        self.normalize_audio_checkbox.setChecked(normalize_audio)
        normalization_method_found = False
        for i in range(self.normalization_method_combo.count()):
            if self.normalization_method_combo.itemData(i) == normalization_method:
                self.normalization_method_combo.setCurrentIndex(i)
                normalization_method_found = True
                break
        if not normalization_method_found:
            self.normalization_method_combo.setCurrentIndex(0)  # Fallback to "peak"
        
        Log.debug(f"DetectOnsetsPanel: Set settings: backtrack={use_backtrack}, early_cut={early_cut_mode}, peak_decay={peak_decay_ratio}, decay_method={decay_method}, max={max_clip_duration}, min={min_clip_duration}, decay={energy_decay_threshold}, numba={use_numba_acceleration}, adaptive={adaptive_threshold_enabled}, preprocessing={preprocessing_enabled}")
        
        # Clip detection settings are always visible (clip end detection always runs)
        # Clip classification is always enabled
        
        # Unblock signals
        self.method_combo.blockSignals(False)
        self.threshold_spin.blockSignals(False)
        self.silence_spin.blockSignals(False)
        self.use_backtrack_checkbox.blockSignals(False)
        self.output_mode_combo.blockSignals(False)
        self.clip_classification_edit.blockSignals(False)
        self.max_clip_duration_spin.blockSignals(False)
        self.min_clip_duration_spin.blockSignals(False)
        self.energy_decay_threshold_spin.blockSignals(False)
        self.use_numba_checkbox.blockSignals(False)
        self.adaptive_threshold_checkbox.blockSignals(False)
        self.adaptive_threshold_factor_spin.blockSignals(False)
        self.onset_lookahead_spin.blockSignals(False)
        self.energy_rise_threshold_spin.blockSignals(False)
        self.min_separation_spin.blockSignals(False)
        self.detection_mode_combo.blockSignals(False)
        self.peak_decay_ratio_spin.blockSignals(False)
        if hasattr(self, 'decay_method_combo'):
            self.decay_method_combo.blockSignals(False)
        self.preprocessing_enabled_checkbox.blockSignals(False)
        self.preemphasis_checkbox.blockSignals(False)
        self.preemphasis_coefficient_spin.blockSignals(False)
        self.remove_dc_offset_checkbox.blockSignals(False)
        self.highpass_checkbox.blockSignals(False)
        self.highpass_cutoff_spin.blockSignals(False)
        self.normalize_audio_checkbox.blockSignals(False)
        self.normalization_method_combo.blockSignals(False)
        
        # Force Qt to update the visuals
        self.method_combo.update()
        self.threshold_spin.update()
        self.silence_spin.update()
        self.output_mode_combo.update()
        self.clip_classification_edit.update()
        
        # Update status
        self.set_status_message("Settings loaded")
    
    def _on_method_changed(self, index: int):
        """Handle onset method change (undoable via settings manager)"""
        method = self.method_combo.itemData(index)
        if not method:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.onset_method = method
            self.set_status_message(f"Method set to {method}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_threshold_editing_finished(self):
        """Handle threshold change when user finishes editing (undoable via settings manager)"""
        value = self.threshold_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.onset_threshold = value
            self.set_status_message(f"Threshold set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_silence_editing_finished(self):
        """Handle min silence change when user finishes editing (undoable via settings manager)"""
        value = self.silence_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.min_silence = value
            self.set_status_message(f"Min silence set to {value:.3f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_output_mode_changed(self, index: int):
        """Handle output mode change (undoable via settings manager)"""
        mode = self.output_mode_combo.itemData(index)
        if not mode:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        # This sets the default render_as_marker property for events created by this block
        try:
            self._settings_manager.output_mode = mode
            self.set_status_message(
                f"Visual display mode set to {mode}. "
                f"All events will be created with render_as_marker={'True' if mode == 'markers' else 'False'}. "
                f"Individual events can be changed in the Editor.",
                error=False
            )
            # Clip detection settings are always visible (clip end detection always runs)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_clip_classification_changed(self):
        """Handle clip classification name change (undoable via settings manager)"""
        value = self.clip_classification_edit.text().strip()
        
        if not value:
            self.set_status_message("Clip name cannot be empty", error=True)
            self.refresh()
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.clip_classification = value
            self.set_status_message(f"Clip name set to '{value}'", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_max_clip_duration_editing_finished(self):
        """Handle max clip duration change when user finishes editing (undoable via settings manager)"""
        value = self.max_clip_duration_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.max_clip_duration = value
            self.set_status_message(f"Max clip duration set to {value:.2f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_min_clip_duration_editing_finished(self):
        """Handle min clip duration change when user finishes editing (undoable via settings manager)"""
        value = self.min_clip_duration_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.min_clip_duration = value
            self.set_status_message(f"Min clip duration set to {value:.3f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_energy_decay_threshold_editing_finished(self):
        """Handle energy decay threshold change when user finishes editing (undoable via settings manager)"""
        value = self.energy_decay_threshold_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.energy_decay_threshold = value
            self.set_status_message(f"Energy decay threshold set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_use_backtrack_changed(self, state: int):
        """Handle use_backtrack checkbox change (undoable via settings manager)"""
        value = self.use_backtrack_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.use_backtrack = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Backtrack {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_use_numba_changed(self, state: int):
        """Handle use numba acceleration checkbox change (undoable via settings manager)"""
        value = self.use_numba_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.use_numba_acceleration = value
            status_text = "enabled" if value else "disabled (using librosa only)"
            self.set_status_message(f"Numba acceleration {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_adaptive_threshold_changed(self, state: int):
        """Handle adaptive threshold checkbox change (undoable via settings manager)"""
        value = self.adaptive_threshold_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.adaptive_threshold_enabled = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Adaptive threshold {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_adaptive_threshold_factor_editing_finished(self):
        """Handle adaptive threshold factor change when user finishes editing (undoable via settings manager)"""
        value = self.adaptive_threshold_factor_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.adaptive_threshold_factor = value
            self.set_status_message(f"Adaptive threshold factor set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_onset_lookahead_editing_finished(self):
        """Handle onset lookahead time change when user finishes editing (undoable via settings manager)"""
        value = self.onset_lookahead_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.onset_lookahead_time = value
            self.set_status_message(f"Onset lookahead time set to {value:.3f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_energy_rise_threshold_editing_finished(self):
        """Handle energy rise threshold change when user finishes editing (undoable via settings manager)"""
        value = self.energy_rise_threshold_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.energy_rise_threshold = value
            self.set_status_message(f"Energy rise threshold set to {value:.2f}x", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_min_separation_editing_finished(self):
        """Handle minimum separation time change when user finishes editing (undoable via settings manager)"""
        value = self.min_separation_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.min_separation_time = value
            self.set_status_message(f"Minimum separation time set to {value:.3f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_detection_mode_changed(self, mode: str):
        """Handle detection mode combo box change (undoable via settings manager)"""
        # Map UI selection to early_cut_mode setting
        # "Librosa End Detection" -> early_cut_mode = False
        # "Decay Modes" -> early_cut_mode = True
        early_cut_mode = (mode == "Decay Modes")
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.early_cut_mode = early_cut_mode
            
            # Switch stacked widget page based on detection mode
            # Page 0: Librosa End Detection
            # Page 1: Decay Modes
            if mode == "Librosa End Detection":
                self.detection_mode_stack.setCurrentIndex(0)
            elif mode == "Decay Modes":
                self.detection_mode_stack.setCurrentIndex(1)
            
            status_text = f"set to '{mode}'"
            self.set_status_message(f"Detection method {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_peak_decay_ratio_editing_finished(self):
        """Handle peak decay ratio change when user finishes editing (undoable via settings manager)"""
        value = self.peak_decay_ratio_spin.value()
        
        # Clamp value to valid range (in case spinbox allows invalid values)
        value = max(0.01, min(1.0, value))
        self.peak_decay_ratio_spin.setValue(value)  # Ensure spinbox reflects clamped value

        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            Log.debug(f"DetectOnsetsPanel: Setting peak_decay_ratio to {value:.4f}")
            self._settings_manager.peak_decay_ratio = value
            # Verify it was saved correctly
            saved_value = self._settings_manager.peak_decay_ratio
            Log.debug(f"DetectOnsetsPanel: Saved peak_decay_ratio is {saved_value:.4f}")
            self.set_status_message(f"Peak decay ratio set to {value:.3f} ({value*100:.1f}% of peak)", error=False)
        except ValueError as e:
            Log.error(f"DetectOnsetsPanel: Failed to set peak_decay_ratio: {e}")
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_decay_method_changed(self, value: str):
        """Handle decay detection method change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.decay_detection_method = value
            method_descriptions = {
                "threshold": "threshold (first frame below ratio)",
                "rate_of_change": "rate of change (rapid decrease detection)",
                "slope": "slope (consistent negative slope)",
                "confirmed_threshold": "confirmed threshold (2-3 frames below)",
                "all": "all methods (uses earliest cut found)"
            }
            description = method_descriptions.get(value, value)
            self.set_status_message(f"Decay detection method: {description}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def refresh_for_undo(self):
        """
        Refresh panel after undo/redo operation.
        
        Reloads settings from database to ensure UI reflects current state.
        Single source of truth: block.metadata in database.
        """
        # Reload settings manager from database (undo may have changed metadata)
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        
        # Refresh UI with current settings
        self.refresh()
    
    def _on_block_updated(self, event):
        """
        Handle block update event - reload settings and refresh UI.
        
        This ensures panel stays in sync when settings change via quick actions
        or other sources. Single source of truth: block.metadata in database.
        """
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            # Skip if we triggered this update (prevents refresh loop)
            if self._is_saving:
                Log.debug(f"DetectOnsetsPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"DetectOnsetsPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"DetectOnsetsPanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"DetectOnsetsPanel: Settings manager reloaded from database")
            else:
                Log.warning(f"DetectOnsetsPanel: Settings manager not available")
                return
            
            # Refresh UI to reflect changes (now that both block and settings are reloaded)
            # Use QTimer.singleShot to ensure refresh happens after event processing
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)
    
    def closeEvent(self, event):
        """Handle panel close - ensure settings are saved before closing."""
        # Force save any pending settings before closing
        if hasattr(self, '_settings_manager') and self._settings_manager:
            if self._settings_manager.has_pending_save():
                Log.debug("DetectOnsetsPanel: Force saving pending settings before close")
                self._settings_manager.force_save()
        
        # Call parent closeEvent
        super().closeEvent(event)
    
    def _on_split_clips_changed(self, state: int):
        """Handle split clips with multiple onsets checkbox change (undoable via settings manager)"""
        value = self.split_clips_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.split_clips_with_multiple_onsets = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Split clips with multiple onsets {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_preprocessing_enabled_changed(self, state: int):
        """Handle preprocessing enabled checkbox change (undoable via settings manager)"""
        value = self.preprocessing_enabled_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.preprocessing_enabled = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Audio preprocessing {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_preemphasis_enabled_changed(self, state: int):
        """Handle pre-emphasis enabled checkbox change (undoable via settings manager)"""
        value = self.preemphasis_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.preemphasis_enabled = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Pre-emphasis filter {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_preemphasis_coefficient_editing_finished(self):
        """Handle pre-emphasis coefficient change when user finishes editing (undoable via settings manager)"""
        value = self.preemphasis_coefficient_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.preemphasis_coefficient = value
            self.set_status_message(f"Pre-emphasis coefficient set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_remove_dc_offset_changed(self, state: int):
        """Handle remove DC offset checkbox change (undoable via settings manager)"""
        value = self.remove_dc_offset_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.remove_dc_offset = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"DC offset removal {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_highpass_enabled_changed(self, state: int):
        """Handle high-pass filter enabled checkbox change (undoable via settings manager)"""
        value = self.highpass_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.highpass_enabled = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"High-pass filter {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_highpass_cutoff_editing_finished(self):
        """Handle high-pass cutoff change when user finishes editing (undoable via settings manager)"""
        value = self.highpass_cutoff_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.highpass_cutoff = value
            self.set_status_message(f"High-pass cutoff set to {value:.0f} Hz", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_normalize_audio_changed(self, state: int):
        """Handle normalize audio checkbox change (undoable via settings manager)"""
        value = self.normalize_audio_checkbox.isChecked()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.normalize_audio = value
            status_text = "enabled" if value else "disabled"
            self.set_status_message(f"Audio normalization {status_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_normalization_method_changed(self, index: int):
        """Handle normalization method change (undoable via settings manager)"""
        method = self.normalization_method_combo.itemData(index)
        if not method:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.normalization_method = method
            self.set_status_message(f"Normalization method set to {method}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _update_execution_summary(self):
        """Update the execution summary display with last execution results"""
        if not hasattr(self, 'summary_label') or not self.block:
            return
        
        try:
            last_execution = self.block.metadata.get("last_execution")
            if not last_execution:
                self.summary_label.setText("No execution data available.\nRun the block to see results here.")
                self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
                return
            
            # Build summary text
            summary_lines = []
            summary_lines.append(f"<b>Last Execution Summary</b>")
            
            audio_items_processed = last_execution.get("audio_items_processed", 0)
            total_onsets = last_execution.get("total_onsets_detected", 0)
            total_events = last_execution.get("total_events_created", 0)
            
            summary_lines.append(f"")
            summary_lines.append(f"Audio Items Processed: <b>{audio_items_processed}</b>")
            summary_lines.append(f"Total Onsets Detected: <b>{total_onsets}</b>")
            summary_lines.append(f"Total Events Created: <b>{total_events}</b>")
            
            # Show details for each audio item
            details = last_execution.get("details", [])
            if details:
                summary_lines.append(f"")
                summary_lines.append(f"<b>Details:</b>")
                for detail in details:
                    audio_name = detail.get("audio_name", "Unknown")
                    onset_count = detail.get("onset_count", 0)
                    event_count = detail.get("event_count", 0)
                    output_mode = detail.get("output_mode", "markers")
                    
                    summary_lines.append(f"  • {audio_name}:")
                    summary_lines.append(f"    - Onsets: {onset_count}, Events: {event_count} ({output_mode})")
                    
                    if output_mode == "clips":
                        split_enabled = detail.get("split_clips_enabled", False)
                        if split_enabled:
                            split_count = detail.get("split_clips_count", 0)
                            if split_count > 0:
                                summary_lines.append(f"    - Split {split_count} clip(s) with multiple onsets")
            
            summary_text = "\n".join(summary_lines)
            self.summary_label.setText(summary_text)
            self.summary_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 10pt; padding: 8px;")
            
        except Exception as e:
            Log.warning(f"DetectOnsetsPanel: Failed to update execution summary: {e}")
            self.summary_label.setText("Error loading execution summary.")
            self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
    
    def _on_setting_changed(self, setting_name: str):
        """
        React to settings changes from this panel's settings manager.
        
        Note: Changes from other sources (quick actions) are handled via
        _on_block_updated() which reloads from database.
        """
        if setting_name in ['onset_method', 'onset_threshold', 'min_silence', 'output_mode', 'clip_classification',
                           'max_clip_duration', 'min_clip_duration', 'energy_decay_threshold', 'use_numba_acceleration',
                           'adaptive_threshold_enabled', 'adaptive_threshold_factor', 'onset_lookahead_time',
                           'energy_rise_threshold', 'min_separation_time', 'split_clips_with_multiple_onsets',
                           'preprocessing_enabled', 'preemphasis_enabled', 'preemphasis_coefficient',
                           'remove_dc_offset', 'highpass_enabled', 'highpass_cutoff', 'normalize_audio',
                           'normalization_method']:
            # Refresh UI to reflect change
            self.refresh()

