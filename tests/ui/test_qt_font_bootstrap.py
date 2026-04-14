from __future__ import annotations

from pathlib import Path

from echozero.ui.qt import font_bootstrap


class _FakeFont:
    def __init__(self, point_size: int = 9):
        self._point_size = point_size

    def pointSize(self) -> int:
        return self._point_size


class _FakeApp:
    def __init__(self, point_size: int = 9):
        self._props: dict[str, object] = {}
        self._font = _FakeFont(point_size)
        self.set_font_calls: list[object] = []

    def property(self, name: str) -> object | None:
        return self._props.get(name)

    def setProperty(self, name: str, value: object) -> None:
        self._props[name] = value

    def font(self) -> _FakeFont:
        return self._font

    def setFont(self, font: object) -> None:
        self.set_font_calls.append(font)


def test_ensure_qt_fonts_available_noops_when_families_already_present(monkeypatch):
    class _ExistingFonts:
        @staticmethod
        def families() -> list[str]:
            return ["Segoe UI", "Arial"]

    app = _FakeApp()
    monkeypatch.setattr(font_bootstrap, "QFontDatabase", _ExistingFonts)

    families = font_bootstrap.ensure_qt_fonts_available(app)

    assert families == ["Segoe UI", "Arial"]
    assert app.set_font_calls == []


def test_ensure_qt_fonts_available_loads_candidate_font_when_none_present(monkeypatch, tmp_path: Path):
    font_file = tmp_path / "segoeui.ttf"
    font_file.write_bytes(b"stub")

    class _HeadlessFonts:
        loaded = False
        added_paths: list[str] = []

        @staticmethod
        def families() -> list[str]:
            return ["Segoe UI"] if _HeadlessFonts.loaded else []

        @staticmethod
        def addApplicationFont(path: str) -> int:
            _HeadlessFonts.added_paths.append(path)
            _HeadlessFonts.loaded = True
            return 1

        @staticmethod
        def applicationFontFamilies(_font_id: int) -> list[str]:
            return ["Segoe UI"]

    app = _FakeApp(point_size=11)
    monkeypatch.setattr(font_bootstrap, "QFontDatabase", _HeadlessFonts)
    monkeypatch.setattr(font_bootstrap, "QFont", lambda family, size: (family, size))

    families = font_bootstrap.ensure_qt_fonts_available(app, candidates=[font_file])

    assert families == ["Segoe UI"]
    assert _HeadlessFonts.added_paths == [str(font_file)]
    assert app.set_font_calls == [("Segoe UI", 11)]
    assert app.property("echozero.qt.font_bootstrap.done") is True
