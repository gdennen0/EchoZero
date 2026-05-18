"""Compact modal progress overlays for EchoZero Qt workflows.
Exists to keep short blocking operations visually aligned with the shell.
Connects launcher and timeline wait states to shared app styling tokens.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget

from echozero.ui.style.tokens import SHELL_TOKENS


def begin_operation_progress_overlay(
    parent: QWidget | None,
    *,
    title: str,
    message: str,
) -> QDialog | None:
    """Show a compact indeterminate progress overlay when a Qt app is active."""
    app = QApplication.instance()
    if app is None:
        return None
    dialog_parent = parent if isinstance(parent, QWidget) else None
    dialog = QDialog(dialog_parent)
    dialog.setObjectName("operationProgressOverlay")
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setModal(True)
    dialog.setFixedWidth(320)
    dialog.setStyleSheet(_build_operation_progress_overlay_qss())

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 14, 18, 16)
    layout.setSpacing(6)

    title_label = QLabel(title, dialog)
    title_label.setObjectName("operationProgressOverlayTitle")
    layout.addWidget(title_label)

    message_label = QLabel(message, dialog)
    message_label.setObjectName("operationProgressOverlayMessage")
    layout.addWidget(message_label)

    progress = QProgressBar(dialog)
    progress.setRange(0, 0)
    progress.setTextVisible(False)
    layout.addWidget(progress)

    dialog.show()
    app.processEvents()
    return dialog


def finish_operation_progress_overlay(dialog: QDialog | None) -> None:
    """Close a progress overlay and flush pending Qt events."""
    if dialog is None:
        return
    dialog.close()
    dialog.deleteLater()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def _build_operation_progress_overlay_qss() -> str:
    tokens = SHELL_TOKENS
    scales = tokens.scales
    return f"""
        QDialog#operationProgressOverlay {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-left: 3px solid #CC8844;
            border-radius: {scales.panel_radius}px;
        }}
        QDialog#operationProgressOverlay QLabel#operationProgressOverlayTitle {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 700;
        }}
        QDialog#operationProgressOverlay QLabel#operationProgressOverlayMessage {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        QDialog#operationProgressOverlay QProgressBar {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: 1px;
            min-height: 4px;
            max-height: 4px;
            text-align: center;
        }}
        QDialog#operationProgressOverlay QProgressBar::chunk {{
            background: #CC8844;
            border-radius: 1px;
        }}
    """
