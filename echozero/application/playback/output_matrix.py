"""
Playback output matrix contracts.
Exists so saved route intent is preserved separately from current hardware capacity.
Connects layer/bus routes to concrete hardware-channel assignments and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.output_routing import (
    DEFAULT_STEREO_OUTPUT_BUS,
    MASTER_OUTPUT_BUS_TOKEN,
    NO_OUTPUT_BUS,
    parse_output_bus_spans,
)


@dataclass(slots=True, frozen=True)
class OutputMatrixAssignment:
    """One valid logical route mapped to current hardware channels."""

    owner_id: str
    token: str
    start_channel: int
    end_channel: int

    @property
    def zero_based_span(self) -> tuple[int, int]:
        width = max(0, int(self.end_channel) - int(self.start_channel) + 1)
        return max(0, int(self.start_channel) - 1), width


@dataclass(slots=True, frozen=True)
class OutputMatrixIssue:
    """One route intent that cannot be fully represented on current hardware."""

    owner_id: str
    token: str
    reason: str


@dataclass(slots=True, frozen=True)
class OutputMatrixSnapshot:
    """Resolved output routing state for one hardware generation."""

    hardware_channels: int
    assignments: tuple[OutputMatrixAssignment, ...]
    issues: tuple[OutputMatrixIssue, ...] = ()

    @property
    def healthy(self) -> bool:
        return not self.issues

    @property
    def diagnostics_label(self) -> str:
        if not self.issues:
            return "routes-fit-hardware"
        return "routes-exceed-hardware;routes-degraded:" + ",".join(
            f"{issue.owner_id}:{issue.token}->{issue.reason}" for issue in self.issues
        )


def resolve_output_matrix(
    routes_by_owner: dict[str, object],
    *,
    hardware_channels: int,
    default_route: object = DEFAULT_STEREO_OUTPUT_BUS,
) -> OutputMatrixSnapshot:
    """Resolve logical output routes against current hardware without losing intent."""

    resolved_channels = max(1, int(hardware_channels or 0))
    assignments: list[OutputMatrixAssignment] = []
    issues: list[OutputMatrixIssue] = []
    for owner_id, route_value in routes_by_owner.items():
        tokens = _route_tokens(route_value)
        if any(_is_no_output_token(token) for token in tokens):
            continue
        include_default = not tokens or any(_is_master_token(token) for token in tokens)
        explicit_route_tokens = [
            token
            for token in tokens
            if not _is_master_token(token) and not _is_no_output_token(token)
        ]
        spans = ()
        if include_default:
            spans = (*spans, *parse_output_bus_spans(default_route))
        spans = (*spans, *parse_output_bus_spans(explicit_route_tokens))
        seen_assignments: set[tuple[int, int]] = set()
        for start_channel, end_channel in spans:
            token = f"outputs_{start_channel}_{end_channel}"
            if start_channel > resolved_channels:
                issues.append(
                    OutputMatrixIssue(
                        owner_id=str(owner_id),
                        token=token,
                        reason="outside-hardware",
                    )
                )
                continue
            clipped_end = min(int(end_channel), resolved_channels)
            if clipped_end < int(end_channel):
                issues.append(
                    OutputMatrixIssue(
                        owner_id=str(owner_id),
                        token=token,
                        reason=f"clipped-to-{resolved_channels}",
                    )
                )
            assignment_key = (int(start_channel), int(clipped_end))
            if assignment_key in seen_assignments:
                continue
            seen_assignments.add(assignment_key)
            assignments.append(
                OutputMatrixAssignment(
                    owner_id=str(owner_id),
                    token=token,
                    start_channel=int(start_channel),
                    end_channel=int(clipped_end),
                )
            )
    return OutputMatrixSnapshot(
        hardware_channels=resolved_channels,
        assignments=tuple(assignments),
        issues=tuple(issues),
    )


def _route_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    try:
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    except TypeError:
        text = str(value or "").strip()
        return [text] if text else []


def _is_master_token(value: str) -> bool:
    return value.strip().lower() in {MASTER_OUTPUT_BUS_TOKEN, "default"}


def _is_no_output_token(value: str) -> bool:
    return value.strip().lower() in {NO_OUTPUT_BUS, "no_output", "off"}


__all__ = [
    "OutputMatrixAssignment",
    "OutputMatrixIssue",
    "OutputMatrixSnapshot",
    "resolve_output_matrix",
]
