"""Object info panel contract-text helpers.
Exists to keep contract and settings-plan formatting out of the panel shell.
Connects inspector contracts and object-action plans to operator-readable text.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract import InspectorContract
from echozero.application.timeline.object_actions import ObjectActionSettingsPlan


def contract_kind_label(contract: InspectorContract) -> str:
    if contract.identity is not None:
        return contract.identity.object_type.replace("_", " ").title()
    if contract.sections:
        return "Timeline"
    return "None"


def contract_detail_text(contract: InspectorContract) -> str:
    if not contract.sections:
        return contract.empty_state
    lines: list[str] = []
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)


def plan_detail_text(plan: ObjectActionSettingsPlan) -> str:
    parts: list[str] = []
    if plan.is_running and plan.operation_message:
        if plan.operation_fraction is not None:
            parts.append(
                f"Run: {plan.operation_message} "
                f"({int(round(plan.operation_fraction * 100.0))}%)"
            )
        else:
            parts.append(f"Run: {plan.operation_message}")
    elif plan.operation_error:
        parts.append(f"Run failed: {plan.operation_error}")
    overrides = plan_override_preview(plan)
    if overrides:
        parts.append(f"Saved: {overrides}")
    if plan.locked_bindings:
        locked = ", ".join(f"{key}: {value}" for key, value in plan.locked_bindings)
        parts.append(f"Locked: {locked}")
    elif plan.rerun_hint:
        parts.append(plan.rerun_hint)
    return "\n".join(parts)


def plan_override_preview(plan: ObjectActionSettingsPlan) -> str:
    highlighted: list[str] = []
    for field in (*plan.editable_fields, *plan.advanced_fields):
        if field.value == field.default_value:
            continue
        highlighted.append(f"{field.label} {field.value}")
        if len(highlighted) == 2:
            break
    return ", ".join(highlighted)


def rendered_contract_text(contract: InspectorContract, *, fallback: str) -> str:
    if contract.identity is None and not contract.sections:
        return contract.empty_state or fallback
    lines: list[str] = [contract.title]
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)
