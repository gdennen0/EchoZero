"""Visual Lab preview: standalone Qt runner for cataloged EchoZero UI objects.
Exists to browse individual primitives and compositions outside the canonical app workflow.
Run with python -m dev.visual_lab.preview, optionally adding --capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from dev.visual_lab.capture import capture_widget, default_capture_path
from dev.visual_lab.catalog import CatalogEntry, CatalogRegistry
from dev.visual_lab.catalog_entries import build_visual_lab_catalog
from dev.visual_lab.token_editor import TokenEditorWidget
from dev.visual_lab.tokens import VisualLabTokens, load_tokens

WINDOW_TITLE = "EchoZero Visual Lab"


class VisualLabWindow(QWidget):
    """Standalone Visual Lab catalog preview window."""

    def __init__(
        self,
        tokens: VisualLabTokens,
        registry: CatalogRegistry,
        *,
        token_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.tokens = tokens
        self.registry = registry
        self.selected_entry_id = registry.first().entry_id
        self._preview_widget: QWidget | None = None
        self._preview_layout: QVBoxLayout | None = None
        self._is_selecting = False

        self.setWindowTitle(WINDOW_TITLE)
        self.setObjectName("visual_lab_window")
        self.setAutoFillBackground(True)
        self.setStyleSheet(_build_stylesheet(tokens))

        root = QVBoxLayout(self)
        root.setContentsMargins(
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
        )
        root.setSpacing(tokens.metrics.gap_px)
        root.addWidget(_title_strip())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_preview_pane())
        self.token_editor = TokenEditorWidget(
            tokens,
            token_path=token_path,
            on_tokens_changed=self.apply_tokens,
        )
        splitter.addWidget(self.token_editor)
        splitter.setSizes([260, 820, 340])
        root.addWidget(splitter, stretch=1)

        self.select_entry(self.selected_entry_id)
        self.resize(1420, 760)

    def apply_tokens(self, tokens: VisualLabTokens) -> None:
        """Apply edited lab tokens and refresh the active preview."""
        self.tokens = tokens
        self.token_editor.set_tokens(tokens)
        self.setStyleSheet(_build_stylesheet(tokens))
        self._render_entry(self.registry.get(self.selected_entry_id), refresh_editor=False)

    def select_entry(self, entry_id: str) -> None:
        """Select and render one catalog entry by id."""
        entry = self.registry.get(entry_id)
        if entry.entry_id == self.selected_entry_id and self._preview_widget is not None:
            return
        self.selected_entry_id = entry.entry_id
        self._is_selecting = True
        try:
            for index in range(self.entry_list.count()):
                item = self.entry_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == entry.entry_id:
                    self.entry_list.setCurrentRow(index)
                    break
        finally:
            self._is_selecting = False
        self._render_entry(entry)

    def select_next(self) -> None:
        """Move to the next catalog object."""
        self.select_entry(self.registry.next_id(self.selected_entry_id))

    def select_previous(self) -> None:
        """Move to the previous catalog object."""
        self.select_entry(self.registry.previous_id(self.selected_entry_id))

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("visual_lab_sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.entry_list = QListWidget()
        self.entry_list.setObjectName("visual_lab_catalog_list")
        for category in self.registry.categories():
            heading = QListWidgetItem(category)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            heading.setData(Qt.ItemDataRole.UserRole, "")
            self.entry_list.addItem(heading)
            for entry in self.registry.entries_for_category(category):
                item = QListWidgetItem(f"  {entry.name}")
                item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
                self.entry_list.addItem(item)
        self.entry_list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.entry_list, stretch=1)

        controls = QHBoxLayout()
        previous_button = QPushButton("Previous")
        next_button = QPushButton("Next")
        previous_button.clicked.connect(self.select_previous)
        next_button.clicked.connect(self.select_next)
        controls.addWidget(previous_button)
        controls.addWidget(next_button)
        layout.addLayout(controls)
        return sidebar

    def _build_preview_pane(self) -> QWidget:
        pane = QWidget()
        pane.setObjectName("visual_lab_preview_pane")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.metadata = QLabel()
        self.metadata.setObjectName("visual_lab_metadata")
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)

        preview_frame = QFrame()
        preview_frame.setObjectName("visual_lab_preview_frame")
        self._preview_layout = QVBoxLayout(preview_frame)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_layout.setSpacing(0)
        layout.addWidget(preview_frame, stretch=1)
        return pane

    def _on_current_item_changed(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        del previous
        if self._is_selecting:
            return
        if current is None:
            return
        entry_id = current.data(Qt.ItemDataRole.UserRole)
        if entry_id:
            self.select_entry(str(entry_id))

    def _render_entry(self, entry: CatalogEntry, *, refresh_editor: bool = True) -> None:
        if self._preview_layout is None:
            return
        if self._preview_widget is not None:
            self._preview_layout.removeWidget(self._preview_widget)
            self._preview_widget.deleteLater()
        self._preview_widget = entry.render(self.tokens)
        self._preview_layout.addWidget(self._preview_widget, stretch=1)
        parts = f"\nParts: {', '.join(entry.part_ids)}" if entry.part_ids else ""
        if refresh_editor:
            self.token_editor.set_editable_targets(
                entry.name,
                self.registry.editable_targets_for(entry.entry_id, self.tokens),
            )
        self.metadata.setText(
            f"{entry.category} / {entry.kind} / {entry.source_kind}\n"
            f"{entry.entry_id}\n"
            f"Source: {entry.source_path}\n"
            f"{entry.description}{parts}"
        )


def build_preview(
    tokens: VisualLabTokens | None = None,
    *,
    selected_entry_id: str | None = None,
    token_path: str | Path | None = None,
) -> VisualLabWindow:
    """Build the standalone Visual Lab catalog widget."""
    window = VisualLabWindow(
        tokens or load_tokens(token_path),
        build_visual_lab_catalog(),
        token_path=token_path,
    )
    if selected_entry_id is not None:
        window.select_entry(selected_entry_id)
    return window


def main(argv: list[str] | None = None) -> int:
    """Run the Visual Lab Qt preview."""
    parser = argparse.ArgumentParser(description="Run the EchoZero Visual Lab catalog preview.")
    parser.add_argument(
        "--tokens", type=Path, default=None, help="Path to Visual Lab tokens TOML."
    )
    parser.add_argument("--item", default=None, help="Catalog item id to select at startup.")
    parser.add_argument("--capture", action="store_true", help="Save a screenshot and exit.")
    parser.add_argument(
        "--output", type=Path, default=default_capture_path(), help="Capture path."
    )
    parser.add_argument("--no-peekaboo", action="store_true", help="Force the Qt grab fallback.")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = build_preview(
        load_tokens(args.tokens),
        selected_entry_id=args.item,
        token_path=args.tokens,
    )
    window.show()
    app.processEvents()

    if args.capture:
        result = capture_widget(
            window,
            args.output,
            prefer_peekaboo=not args.no_peekaboo,
            window_title=WINDOW_TITLE,
        )
        print(f"captured {result.path} via {result.backend}")
        window.close()
        app.processEvents()
        return 0

    return app.exec()


def _title_strip() -> QWidget:
    strip = QWidget()
    strip.setObjectName("visual_lab_title_strip")
    layout = QHBoxLayout(strip)
    layout.setContentsMargins(0, 0, 0, 0)
    title = QLabel("EchoZero Visual Lab")
    title.setObjectName("visual_lab_title")
    subtitle = QLabel("Catalog browser / individual objects / support-only previews")
    subtitle.setObjectName("visual_lab_subtitle")
    layout.addWidget(title)
    layout.addWidget(subtitle)
    layout.addStretch(1)
    return strip


def _build_stylesheet(tokens: VisualLabTokens) -> str:
    return f"""
        QWidget#visual_lab_window {{
            background: {tokens.global_colors.app_background};
            color: {tokens.global_colors.text_primary};
            font-family: {tokens.fonts.family};
            font-size: {tokens.fonts.base_px}px;
        }}
        QWidget#visual_lab_sidebar, QWidget#visual_lab_preview_pane {{
            background: {tokens.global_colors.surface};
            border: 1px solid {tokens.palette.border};
            border-radius: {tokens.metrics.corner_radius_px}px;
        }}
        QFrame#visual_lab_preview_frame {{
            background: {tokens.palette.panel};
            border: 1px solid {tokens.palette.border};
            border-radius: {tokens.metrics.corner_radius_px}px;
        }}
        QWidget#visual_lab_catalog_frame {{
            background: {tokens.palette.panel};
        }}
        QListWidget#visual_lab_catalog_list {{
            background: {tokens.palette.panel_raised};
            border: 1px solid {tokens.palette.border};
            color: {tokens.palette.text};
        }}
        QPushButton {{
            background: {tokens.palette.panel_raised};
            border: 1px solid {tokens.palette.border};
            border-radius: {tokens.metrics.control_radius_px}px;
            color: {tokens.palette.text};
            min-height: 28px;
        }}
        QLabel#visual_lab_title {{
            color: {tokens.palette.text};
            font-size: {tokens.fonts.title_px}px;
            font-weight: {tokens.fonts.weight_bold};
        }}
        QLabel#visual_lab_subtitle,
        QLabel#visual_lab_metadata {{
            color: {tokens.palette.text_muted};
            font-size: {tokens.fonts.small_px}px;
        }}
        QLabel#visual_lab_object_title {{
            color: {tokens.palette.text};
            font-size: {tokens.fonts.label_px}px;
            font-weight: {tokens.fonts.weight_bold};
        }}
        QLabel#visual_lab_token_editor_title {{
            color: {tokens.palette.text};
            font-size: {tokens.fonts.label_px}px;
            font-weight: {tokens.fonts.weight_bold};
        }}
        QLabel#visual_lab_token_editor_status {{
            color: {tokens.palette.text_muted};
            font-size: {tokens.fonts.small_px}px;
        }}
        QWidget#visual_lab_token_editor,
        QTabWidget#visual_lab_token_tabs {{
            background: {tokens.palette.panel};
            color: {tokens.palette.text};
        }}
        QLineEdit, QSpinBox {{
            background: {tokens.palette.panel_raised};
            border: 1px solid {tokens.palette.border};
            color: {tokens.palette.text};
            min-height: 24px;
        }}
    """


if __name__ == "__main__":
    raise SystemExit(main())
