"""Visual Lab token editor.
Exists to expose selected-object style knobs inside the preview harness.
The editor is schema-driven from catalog metadata and stays outside production runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dev.visual_lab.catalog import ResolvedStyleTarget
from dev.visual_lab.font_assets import import_lab_fonts
from dev.visual_lab.tokens import (
    DEFAULT_TOKEN_PATH,
    TokenFieldSpec,
    VisualLabTokens,
    save_tokens,
    token_field_specs,
    update_token_values,
)

TokenChangedCallback = Callable[[VisualLabTokens], None]


class ColorSwatchButton(QPushButton):
    """Small color affordance that opens a picker on double-click."""

    double_clicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.double_clicked.emit()
        event.accept()


class ColorLineEdit(QLineEdit):
    """Line edit that exposes a double-click signal for color picking."""

    double_clicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.double_clicked.emit()
        event.accept()


class ColorTokenEditor(QWidget):
    """Manual hex editor with a double-click color picker path."""

    value_changed = pyqtSignal()

    def __init__(self, spec: TokenFieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.swatch = ColorSwatchButton(self)
        self.swatch.setFixedSize(24, 22)
        self.swatch.setToolTip(f"Double-click to pick {spec.path}")
        self.editor = ColorLineEdit(str(spec.value), self)
        self.editor.setMinimumWidth(104)
        self.editor.setToolTip("Double-click to pick a color, or type #RRGGBB.")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.swatch)
        layout.addWidget(self.editor, stretch=1)

        self.swatch.double_clicked.connect(self.open_color_picker)
        self.editor.double_clicked.connect(self.open_color_picker)
        self.editor.textChanged.connect(self._on_text_changed)
        self._refresh_swatch()

    def value(self) -> str:
        """Return the current hex text value."""
        return self.editor.text()

    def set_value(self, value: str) -> None:
        """Set the current hex text value and refresh the swatch."""
        self.editor.setText(value)
        self._refresh_swatch()

    def choose_color(self, initial: QColor) -> QColor:
        """Choose a color; tests can monkeypatch this to avoid a real dialog."""
        return QColorDialog.getColor(initial, self, f"Pick {self.spec.name}")

    def open_color_picker(self) -> None:
        """Open the color picker and write the selected color into the text field."""
        selected = self.choose_color(QColor(self.value()))
        if selected.isValid():
            self.set_value(selected.name(QColor.NameFormat.HexRgb))

    def _on_text_changed(self) -> None:
        self._refresh_swatch()
        self.value_changed.emit()

    def _refresh_swatch(self) -> None:
        color = self.value().strip()
        if QColor(color).isValid():
            self.swatch.setStyleSheet(f"QPushButton {{ background: {color}; }}")
            return
        self.swatch.setStyleSheet("")


class FontFamilyEditor(QWidget):
    """Editable font-family dropdown backed by Qt's available font families."""

    value_changed = pyqtSignal()

    def __init__(
        self,
        spec: TokenFieldSpec,
        *,
        font_families: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setMinimumWidth(150)
        self.combo.setToolTip("Choose an installed Qt font family, or type a family name.")
        for family in _font_family_options(str(spec.value), font_families):
            self.combo.addItem(family)
        self.combo.setCurrentText(str(spec.value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        self.combo.currentTextChanged.connect(self.value_changed.emit)

    def value(self) -> str:
        """Return the selected or manually typed font family."""
        return self.combo.currentText()

    def set_value(self, value: str) -> None:
        """Set the active font family and keep missing values selectable."""
        if self.combo.findText(value) < 0:
            self.combo.insertItem(0, value)
        self.combo.setCurrentText(value)

    def refresh_families(self, font_families: Sequence[str] | None = None) -> None:
        """Refresh dropdown options while preserving the current value."""
        current_value = self.value()
        self.combo.blockSignals(True)
        try:
            self.combo.clear()
            for family in _font_family_options(current_value, font_families):
                self.combo.addItem(family)
            self.combo.setCurrentText(current_value)
        finally:
            self.combo.blockSignals(False)


class TokenEditorWidget(QWidget):
    """Generated editor for the selected Visual Lab catalog object's tokens."""

    def __init__(
        self,
        tokens: VisualLabTokens,
        *,
        token_path: str | Path | None = None,
        on_tokens_changed: TokenChangedCallback | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tokens = tokens
        self.token_path = Path(token_path) if token_path is not None else DEFAULT_TOKEN_PATH
        self._on_tokens_changed = on_tokens_changed
        self._editors: dict[
            str, ColorTokenEditor | FontFamilyEditor | QLineEdit | QSpinBox
        ] = {}
        self._targets: tuple[ResolvedStyleTarget, ...] = ()
        self._target_specs_by_id: dict[str, tuple[TokenFieldSpec, ...]] = {}
        self._editable_specs: tuple[TokenFieldSpec, ...] = token_field_specs(tokens)
        self._is_loading = False
        self.setObjectName("visual_lab_token_editor")
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.header = QLabel("Style Tokens")
        self.header.setObjectName("visual_lab_token_editor_title")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.part_tree = QTreeWidget(self)
        self.part_tree.setObjectName("visual_lab_style_target_tree")
        self.part_tree.setHeaderHidden(True)
        self.part_tree.setMinimumHeight(120)
        self.part_tree.currentItemChanged.connect(self._on_part_tree_selection_changed)
        layout.addWidget(self.part_tree)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("visual_lab_token_tabs")
        layout.addWidget(self.tabs, stretch=1)

        controls = QHBoxLayout()
        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.apply_edits)
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_edits)
        import_font_button = QPushButton("Import Font", self)
        import_font_button.clicked.connect(self.import_fonts)
        controls.addWidget(apply_button)
        controls.addWidget(save_button)
        controls.addWidget(import_font_button)
        layout.addLayout(controls)

        self.status_label = QLabel(f"Token file: {self.token_path}")
        self.status_label.setObjectName("visual_lab_token_editor_status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.set_editable_specs("All tokens", self._editable_specs)

    def apply_edits(self) -> None:
        """Apply edited tokens to the active lab preview."""
        try:
            self.tokens = self._tokens_from_editors()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText("Applied live in session.")
        if self._on_tokens_changed is not None:
            self._on_tokens_changed(self.tokens)

    def save_edits(self) -> None:
        """Apply edited tokens and save them to the active TOML file."""
        try:
            self.tokens = self._tokens_from_editors()
            save_tokens(self.tokens, self.token_path)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(f"Saved {self.token_path}")
        if self._on_tokens_changed is not None:
            self._on_tokens_changed(self.tokens)

    def import_fonts(self) -> None:
        """Import font files into Visual Lab assets and refresh font dropdowns."""
        font_paths = self.choose_font_files()
        if not font_paths:
            return
        try:
            imported = import_lab_fonts([Path(path) for path in font_paths])
        except (OSError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        imported_families = tuple(family for font in imported for family in font.families)
        self.refresh_font_dropdowns(imported_families)
        family_count = sum(len(font.families) for font in imported)
        copied_count = len(imported)
        if family_count:
            self.status_label.setText(
                f"Imported {copied_count} font file(s), registered {family_count} family name(s)."
            )
            return
        self.status_label.setText(f"Imported {copied_count} font file(s); Qt did not expose families.")

    def choose_font_files(self) -> tuple[str, ...]:
        """Choose font files; tests can monkeypatch this to avoid a real dialog."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Fonts",
            "",
            "Fonts (*.ttf *.otf *.ttc)",
        )
        return tuple(paths)

    def refresh_font_dropdowns(self, extra_families: Sequence[str] = ()) -> None:
        """Refresh every visible font-family dropdown from Qt's active font database."""
        font_families = tuple(dict.fromkeys((*QFontDatabase.families(), *extra_families)))
        for editor in self._editors.values():
            if isinstance(editor, FontFamilyEditor):
                editor.refresh_families(font_families)

    def set_editable_specs(self, title: str, specs: Sequence[TokenFieldSpec]) -> None:
        """Rebuild the editor to show only the selected entry's relevant tokens."""
        self._targets = ()
        self._target_specs_by_id = {}
        self._editable_specs = _dedupe_specs(specs)
        self.header.setText(f"Style Tokens\n{title}")
        self._build_part_tree(())
        self._rebuild_tabs()

    def set_editable_targets(
        self, title: str, targets: Sequence[ResolvedStyleTarget]
    ) -> None:
        """Rebuild the editor around nested component part style targets."""
        self._targets = tuple(targets)
        self._target_specs_by_id = {
            target.target_id: _dedupe_specs(target.specs) for target in self._targets
        }
        self._editable_specs = _dedupe_specs(
            spec for target in self._targets for spec in target.specs
        )
        self.header.setText(f"Style Tokens\n{title}")
        self._build_part_tree(self._targets)
        self._rebuild_tabs()

    def set_tokens(self, tokens: VisualLabTokens) -> None:
        """Refresh editor controls from a token object."""
        self.tokens = tokens
        specs_by_path = {spec.path: spec for spec in token_field_specs(tokens)}
        self._editable_specs = tuple(
            specs_by_path[spec.path]
            for spec in self._editable_specs
            if spec.path in specs_by_path
        )
        self._is_loading = True
        try:
            for path, editor in self._editors.items():
                spec = specs_by_path[path]
                if isinstance(editor, QSpinBox):
                    editor.setValue(int(spec.value))
                elif isinstance(editor, ColorTokenEditor):
                    editor.set_value(str(spec.value))
                elif isinstance(editor, FontFamilyEditor):
                    editor.set_value(str(spec.value))
                else:
                    editor.setText(str(spec.value))
        finally:
            self._is_loading = False

    def _build_part_tree(self, targets: Sequence[ResolvedStyleTarget]) -> None:
        self.part_tree.blockSignals(True)
        try:
            self.part_tree.clear()
            all_item = QTreeWidgetItem(["All parts"])
            all_item.setData(0, Qt.ItemDataRole.UserRole, "__all__")
            self.part_tree.addTopLevelItem(all_item)
            nodes: dict[str, QTreeWidgetItem] = {"": all_item}
            for target in targets:
                parent_key = ""
                for segment in (target.component, *target.part_path.split(".")):
                    key = f"{parent_key}.{segment}" if parent_key else segment
                    if key not in nodes:
                        item = QTreeWidgetItem([segment])
                        item.setData(0, Qt.ItemDataRole.UserRole, "")
                        nodes[parent_key].addChild(item)
                        nodes[key] = item
                    parent_key = key
                nodes[parent_key].setText(0, target.label)
                nodes[parent_key].setData(0, Qt.ItemDataRole.UserRole, target.target_id)
            self.part_tree.expandAll()
            self.part_tree.setCurrentItem(all_item)
        finally:
            self.part_tree.blockSignals(False)

    def _on_part_tree_selection_changed(
        self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None
    ) -> None:
        del previous
        if current is None:
            return
        target_id = current.data(0, Qt.ItemDataRole.UserRole)
        if target_id == "__all__":
            if self._targets:
                self._editable_specs = _dedupe_specs(
                    spec for target in self._targets for spec in target.specs
                )
        elif target_id:
            self._editable_specs = self._target_specs_by_id[str(target_id)]
        else:
            return
        self._rebuild_tabs()

    def _rebuild_tabs(self) -> None:
        self._is_loading = True
        try:
            self._editors.clear()
            while self.tabs.count():
                widget = self.tabs.widget(0)
                self.tabs.removeTab(0)
                widget.deleteLater()
            if not self._editable_specs:
                empty = QLabel("No editable tokens declared for this object.")
                empty.setWordWrap(True)
                self.tabs.addTab(empty, "empty")
                return
            for section in ("global_colors", "palette", "fonts", "metrics"):
                specs = tuple(spec for spec in self._editable_specs if spec.section == section)
                if specs:
                    self.tabs.addTab(self._build_section_tab(specs), section)
        finally:
            self._is_loading = False

    def _build_section_tab(self, specs: Sequence[TokenFieldSpec]) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        panel = QWidget(scroll)
        form = QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        for spec in specs:
            editor = self._build_editor(spec)
            self._editors[spec.path] = editor
            form.addRow(_field_label(spec), editor)
        scroll.setWidget(panel)
        return scroll

    def _build_editor(
        self, spec: TokenFieldSpec
    ) -> ColorTokenEditor | FontFamilyEditor | QLineEdit | QSpinBox:
        if spec.value_type is int:
            editor = QSpinBox()
            editor.setRange(0, 10000)
            editor.setValue(int(spec.value))
            editor.valueChanged.connect(self._on_editor_changed)
            return editor
        if spec.section in {"global_colors", "palette"}:
            editor = ColorTokenEditor(spec)
            editor.value_changed.connect(self._on_editor_changed)
            return editor
        if spec.section == "fonts" and spec.value_type is str:
            editor = FontFamilyEditor(spec)
            editor.value_changed.connect(self._on_editor_changed)
            return editor
        editor = QLineEdit(str(spec.value))
        editor.setMinimumWidth(110)
        editor.textChanged.connect(self._on_editor_changed)
        return editor

    def _on_editor_changed(self) -> None:
        if not self._is_loading:
            self.apply_edits()

    def _tokens_from_editors(self) -> VisualLabTokens:
        values: list[tuple[str, str | int]] = []
        for path, editor in self._editors.items():
            if isinstance(editor, QSpinBox):
                values.append((path, editor.value()))
            elif isinstance(editor, ColorTokenEditor):
                values.append((path, editor.value()))
            elif isinstance(editor, FontFamilyEditor):
                values.append((path, editor.value()))
            else:
                values.append((path, editor.text()))
        return update_token_values(self.tokens, values)


def _field_label(spec: TokenFieldSpec) -> QLabel:
    label = QLabel(spec.name)
    label.setToolTip(spec.path)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _font_family_options(
    current_value: str, font_families: Sequence[str] | None = None
) -> tuple[str, ...]:
    families = (
        tuple(font_families)
        if font_families is not None
        else tuple(QFontDatabase.families())
    )
    options = list(dict.fromkeys(family for family in families if family))
    if current_value and current_value not in options:
        options.insert(0, current_value)
    return tuple(options)


def _dedupe_specs(specs: Iterable[TokenFieldSpec]) -> tuple[TokenFieldSpec, ...]:
    return tuple(dict((spec.path, spec) for spec in specs).values())
