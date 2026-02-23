"""
macOS window title bar styling.

Sets the native title bar (traffic lights + "EZ - Project") to match
the theme. Uses PyObjC to access NSWindow; no-op if PyObjC unavailable.
"""
import sys
from ctypes import c_void_p


def _is_dark_theme() -> bool:
    """Return True if the current theme is dark (use dark Aqua appearance)."""
    from ui.qt_gui.design_system import Colors

    r, g, b = Colors.BG_DARK.red(), Colors.BG_DARK.green(), Colors.BG_DARK.blue()
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return luminance < 0.5


def apply_titlebar_theme(window) -> bool:
    """
    Apply theme colors to the macOS native window title bar.

    Uses three approaches:
    1. NSApp.appearance - forces native title bar elements (traffic lights,
       title text) to use dark or light styling based on current theme.
    2. NSWindow.titlebarAppearsTransparent + backgroundColor - sets the
       title bar background to Colors.BG_DARK.

    Args:
        window: QMainWindow instance (or any QWidget with winId).

    Returns:
        True if applied successfully, False otherwise.
    """
    if sys.platform != "darwin":
        return False
    try:
        from ui.qt_gui.design_system import Colors

        import objc
        from AppKit import (
            NSApp,
            NSColor,
            NSAppearance,
            NSAppearanceNameDarkAqua,
            NSAppearanceNameAqua,
        )

        # 1. Set app-wide appearance so traffic lights + title text match theme
        use_dark = _is_dark_theme()
        name = NSAppearanceNameDarkAqua if use_dark else NSAppearanceNameAqua
        appearance = NSAppearance.appearanceNamed_(name)
        if appearance:
            NSApp.setAppearance_(appearance)

        win_id = window.winId()
        ptr = int(win_id) if win_id else 0
        if ptr == 0:
            return True

        # PyQt6 winId() returns sip.voidptr; c_void_p requires an int
        ns_view = objc.objc_object(c_void_p=c_void_p(ptr))
        ns_window = ns_view.window()
        if ns_window is None:
            return True

        # 2. Force this window's appearance
        ns_window.setAppearance_(appearance)

        # 3. Title bar transparent + our theme background color.
        # Note: We avoid NSFullSizeContentViewWindowMask because it makes the
        # title bar non-draggable (content extends under it, consuming drag).
        # Without it, the title bar stays draggable but uses system appearance.
        ns_window.setTitlebarAppearsTransparent_(True)
        r = Colors.BG_DARK.red() / 255.0
        g = Colors.BG_DARK.green() / 255.0
        b = Colors.BG_DARK.blue() / 255.0
        bg = NSColor.colorWithRed_green_blue_alpha_(r, g, b, 1.0)
        ns_window.setBackgroundColor_(bg)

        return True
    except Exception:
        return False
