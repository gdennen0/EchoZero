"""Review writeback applies explicit Foundry review decisions to EZ project truth.
Exists because project-backed review should update canonical event data when provenance is sufficient.
Connects durable review signals to project main-take event mutations without touching UI lanes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from echozero.domain.types import Event, EventData, Layer as DomainLayer
from echozero.foundry.domain.review import (
    ReviewDecision,
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewSession,
    ReviewSignal,
)
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take
from .review_runtime_bridge import get_review_runtime_bridge
from .review_event_state import normalize_review_label, updated_review_metadata


class ReviewWritebackService:
    """Mutates project-backed events from explicit review signals when safe."""

    def apply_review_signal(self, session: ReviewSession, signal: ReviewSignal) -> dict[str, object]:
        """Write reviewed truth back into project data when provenance and access are sufficient."""
        if signal.review_outcome == ReviewOutcome.PENDING or signal.review_decision is None:
            return {"status": "skipped", "reason": "pending_review"}
        if session.metadata.get("queue_source_kind") != "ez_project":
            return {"status": "skipped", "reason": "non_project_session"}

        project_dir = self._resolve_project_dir(session.source_ref)
        if project_dir is None:
            return {"status": "skipped", "reason": "unsupported_project_source"}

        runtime_bridge = get_review_runtime_bridge(project_dir)
        if runtime_bridge is not None:
            bridge_result = runtime_bridge.apply_review_signal(session, signal)
            if isinstance(bridge_result, dict):
                return bridge_result

        layer_id = self._ref_id(signal.source_provenance.get("layer_ref"), prefix="layer")
        if layer_id is None:
            return {"status": "skipped", "reason": "missing_layer_provenance"}

        event_id = self._target_event_id(signal)
        if event_id is None:
            return {"status": "skipped", "reason": "missing_event_provenance"}

        try:
            with ProjectStorage.open_db(project_dir) as project:
                layer = project.layers.get(layer_id)
                if layer is None:
                    return {"status": "skipped", "reason": "layer_not_found"}
                take = project.takes.get_main(layer.id)
                if take is None or not isinstance(take.data, EventData):
                    return {"status": "skipped", "reason": "main_take_missing"}
                update_result = self._rewrite_take(
                    take=take,
                    target_event_id=event_id,
                    signal=signal,
                )
                updated_take = update_result.get("take")
                if not isinstance(updated_take, Take):
                    return {
                        "status": "skipped",
                        "reason": str(update_result.get("reason", "event_not_found")),
                    }
                project.takes.update(updated_take)
                project.save()
                return {
                    "status": "applied",
                    "layer_id": layer.id,
                    "event_id": str(update_result["event_id"]),
                    "decision_kind": signal.review_decision.kind.value,
                }
        except RuntimeError:
            return {"status": "skipped", "reason": "project_locked"}

    def _rewrite_take(
        self,
        *,
        take: Take,
        target_event_id: str,
        signal: ReviewSignal,
    ) -> dict[str, object]:
        if not isinstance(take.data, EventData):
            return {"reason": "main_take_missing"}
        decision = signal.review_decision
        assert decision is not None

        updated_layers: list[DomainLayer] = []
        matched_event_id: str | None = None
        changed = False
        for domain_layer in take.data.layers:
            updated_events: list[Event] = []
            for event in domain_layer.events:
                if event.id != target_event_id:
                    updated_events.append(event)
                    continue
                next_event = self._updated_event(event=event, signal=signal, decision=decision)
                updated_events.append(next_event)
                matched_event_id = event.id
                changed = True
            updated_layers.append(replace(domain_layer, events=tuple(updated_events)))
        if not changed or matched_event_id is None:
            return {"reason": "event_not_found"}
        updated_take = replace(take, data=EventData(layers=tuple(updated_layers)))
        return {"take": updated_take, "event_id": matched_event_id}

    def _updated_event(
        self,
        *,
        event: Event,
        signal: ReviewSignal,
        decision: ReviewDecision,
    ) -> Event:
        label = self._resolved_label(signal=signal, decision=decision)
        next_time = float(event.time)
        next_duration = float(event.duration)
        if decision.kind == ReviewDecisionKind.BOUNDARY_CORRECTED:
            if decision.corrected_start_ms is None or decision.corrected_end_ms is None:
                return event
            next_time = float(decision.corrected_start_ms) / 1000.0
            next_duration = max(
                0.0,
                (float(decision.corrected_end_ms) - float(decision.corrected_start_ms)) / 1000.0,
            )
        elif (
            decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED
            and decision.corrected_start_ms is not None
            and decision.corrected_end_ms is not None
        ):
            next_time = float(decision.corrected_start_ms) / 1000.0
            next_duration = max(
                0.0,
                (float(decision.corrected_end_ms) - float(decision.corrected_start_ms)) / 1000.0,
            )

        classifications = dict(event.classifications or {})
        if label is not None:
            normalized_label = normalize_review_label(label)
            classifications["class"] = normalized_label
            classifications["label"] = normalized_label
        metadata = dict(event.metadata or {})
        promotion_state = "promoted"
        review_state = "signed_off" if signal.review_outcome == ReviewOutcome.CORRECT else "corrected"
        corrected_label = None
        next_origin = event.origin
        if decision.kind == ReviewDecisionKind.REJECTED:
            promotion_state = "demoted"
        elif decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED:
            corrected_label = label
            next_origin = "manual_added"
        elif decision.kind in {
            ReviewDecisionKind.RELABELED,
            ReviewDecisionKind.BOUNDARY_CORRECTED,
        }:
            corrected_label = label
        metadata = updated_review_metadata(
            metadata,
            promotion_state=promotion_state,
            review_state=review_state,
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=normalize_review_label(signal.predicted_label),
            corrected_label=corrected_label,
            review_note=decision.review_note or signal.review_note,
            reviewed_at=signal.reviewed_at,
            original_start_ms=decision.original_start_ms or (float(event.time) * 1000.0),
            original_end_ms=decision.original_end_ms or ((float(event.time) + float(event.duration)) * 1000.0),
            corrected_start_ms=decision.corrected_start_ms,
            corrected_end_ms=decision.corrected_end_ms,
            created_event_ref=decision.created_event_ref,
            surface=decision.provenance.surface if decision.provenance is not None else None,
            workflow=decision.provenance.workflow if decision.provenance is not None else None,
            operator_action=decision.provenance.operator_action if decision.provenance is not None else None,
        )
        metadata["foundry_review"] = {
            "review_outcome": signal.review_outcome.value,
            "decision_kind": decision.kind.value,
            "corrected_label": decision.corrected_label,
            "review_note": decision.review_note,
            "queue_session_ref": (
                decision.provenance.queue_session_ref if decision.provenance is not None else None
            ),
            "signal_id": signal.id,
        }
        return replace(
            event,
            time=next_time,
            duration=next_duration,
            classifications=classifications,
            metadata=metadata,
            origin=next_origin,
        )

    @staticmethod
    def _resolved_label(
        *,
        signal: ReviewSignal,
        decision: ReviewDecision,
    ) -> str | None:
        candidate = decision.corrected_label or signal.target_class or signal.predicted_label
        text = str(candidate).strip() if candidate is not None else ""
        return text or None

    @staticmethod
    def _target_event_id(signal: ReviewSignal) -> str | None:
        decision = signal.review_decision
        if decision is None:
            return None
        if decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED and decision.created_event_ref:
            return ReviewWritebackService._ref_id(decision.created_event_ref, prefix="event")
        return ReviewWritebackService._ref_id(signal.source_provenance.get("event_ref"), prefix="event")

    @staticmethod
    def _resolve_project_dir(source_ref: str | None) -> Path | None:
        if source_ref is None:
            return None
        path = Path(source_ref).expanduser().resolve()
        if path.is_dir() and (path / "project.db").exists():
            return path
        if path.is_file() and path.name == "project.db":
            return path.parent
        return None

    @staticmethod
    def _ref_id(value: object, *, prefix: str) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.startswith(f"{prefix}:"):
            return text.split(":", 1)[1].strip() or None
        return text
