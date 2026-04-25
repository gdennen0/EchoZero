"""
RemoteCapabilities: Safe allowlist for EchoZero remote wrapper actions.
Exists because phone access should expose only a narrow control subset, not the full raw bridge.
Connects short-lived remote sessions to app-declared actions that are safe for v0 remote use.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ALLOWED_ACTIONS: tuple[str, ...] = (
    "transport.play",
    "transport.pause",
    "transport.stop",
)


@dataclass(frozen=True, slots=True)
class RemoteCapabilities:
    """Allowlisted app actions that one remote session may invoke."""

    allowed_actions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS

    def can_invoke(self, action_id: str) -> bool:
        """Return whether one action id is allowed for this remote session."""
        return action_id in self.allowed_actions

    def as_payload(self) -> dict[str, object]:
        """Return a JSON-safe capability payload for API responses."""
        return {"allowed_actions": list(self.allowed_actions)}
