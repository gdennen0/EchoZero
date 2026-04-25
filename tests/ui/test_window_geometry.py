from __future__ import annotations

import echozero.ui.qt.window_geometry as window_geometry


class _FakeGeometry:
    def __init__(self, width: int, height: int, *, x: int = 0, y: int = 0) -> None:
        self._width = width
        self._height = height
        self._x = x
        self._y = y

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y

    def left(self) -> int:
        return self._x

    def top(self) -> int:
        return self._y

    def right(self) -> int:
        return self._x + self._width - 1

    def bottom(self) -> int:
        return self._y + self._height - 1


class _FakeScreen:
    def __init__(self, width: int, height: int) -> None:
        self._geometry = _FakeGeometry(width, height)

    def availableGeometry(self) -> _FakeGeometry:
        return self._geometry


class _FakeWidget:
    def __init__(
        self,
        screen,
        *,
        width: int = 1000,
        height: int = 700,
        frame_extra_width: int = 16,
        frame_extra_height: int = 39,
        frame_x: int = 0,
        frame_y: int = 0,
    ) -> None:
        self._screen = screen
        self._width = width
        self._height = height
        self._frame_extra_width = frame_extra_width
        self._frame_extra_height = frame_extra_height
        self._frame_x = frame_x
        self._frame_y = frame_y

    def screen(self):
        return self._screen

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def frameGeometry(self) -> _FakeGeometry:
        return _FakeGeometry(
            self._width + self._frame_extra_width,
            self._height + self._frame_extra_height,
            x=self._frame_x,
            y=self._frame_y,
        )

    def move(self, x: int, y: int) -> None:
        self._frame_x = x
        self._frame_y = y


def test_compute_initial_window_size_scales_from_screen_size() -> None:
    width, height = window_geometry.compute_initial_window_size_for_screen(1366, 768)

    assert width == int(1366 * window_geometry.INITIAL_WINDOW_WIDTH_RATIO)
    assert height == int(768 * window_geometry.INITIAL_WINDOW_HEIGHT_RATIO)


def test_compute_initial_window_size_never_exceeds_screen_size() -> None:
    width, height = window_geometry.compute_initial_window_size_for_screen(800, 480)

    assert width == int(800 * window_geometry.INITIAL_WINDOW_WIDTH_RATIO)
    assert height == int(480 * window_geometry.INITIAL_WINDOW_HEIGHT_RATIO)


def test_resolve_initial_window_size_prefers_widget_screen(monkeypatch) -> None:
    widget = _FakeWidget(_FakeScreen(1280, 720))
    monkeypatch.setattr(window_geometry, "_primary_screen", lambda: _FakeScreen(1920, 1080))

    width, height = window_geometry.resolve_initial_window_size(widget)

    assert width == int(1280 * window_geometry.INITIAL_WINDOW_WIDTH_RATIO)
    assert height == int(720 * window_geometry.INITIAL_WINDOW_HEIGHT_RATIO)


def test_resolve_initial_window_size_falls_back_without_screen(monkeypatch) -> None:
    widget = _FakeWidget(None)
    monkeypatch.setattr(window_geometry, "_primary_screen", lambda: None)

    assert window_geometry.resolve_initial_window_size(widget) == (
        window_geometry.DEFAULT_INITIAL_WINDOW_WIDTH,
        window_geometry.DEFAULT_INITIAL_WINDOW_HEIGHT,
    )


def test_fit_window_to_available_screen_clamps_size_and_position() -> None:
    widget = _FakeWidget(
        _FakeScreen(1000, 800),
        width=980,
        height=760,
        frame_extra_width=40,
        frame_extra_height=40,
        frame_x=-10,
        frame_y=-20,
    )

    window_geometry.fit_window_to_available_screen(widget)

    assert widget.width() == 928
    assert widget.height() == 728
    assert widget.frameGeometry().x() >= window_geometry.INITIAL_WINDOW_EDGE_MARGIN_PX
    assert widget.frameGeometry().y() >= window_geometry.INITIAL_WINDOW_EDGE_MARGIN_PX
