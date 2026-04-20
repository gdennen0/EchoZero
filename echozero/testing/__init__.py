"""Testing utilities for EchoZero.
Exists to expose testing helpers without forcing every test import through Qt.
Connects package-level imports to the heavier app-flow harness on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from echozero.testing.app_flow import AppFlowHarness

__all__ = ["AppFlowHarness"]


def __getattr__(name: str):
    """Resolve heavy testing helpers lazily so non-Qt tests stay import-safe."""

    if name == "AppFlowHarness":
        from echozero.testing.app_flow import AppFlowHarness

        return AppFlowHarness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
