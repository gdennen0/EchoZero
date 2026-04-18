"""EchoZero-specific adapter surfaces for ui_automation."""

from .live_client import LiveEchoZeroAutomationBackend, LiveEchoZeroAutomationProvider
from .provider import EchoZeroAutomationBackend, HarnessEchoZeroAutomationProvider

EchoZeroAutomationProvider = LiveEchoZeroAutomationProvider

__all__ = [
    "EchoZeroAutomationBackend",
    "EchoZeroAutomationProvider",
    "HarnessEchoZeroAutomationProvider",
    "LiveEchoZeroAutomationBackend",
    "LiveEchoZeroAutomationProvider",
]
