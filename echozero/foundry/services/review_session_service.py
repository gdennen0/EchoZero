"""ReviewSessionService imports and updates persisted human-review queues.
Exists because Foundry needs a product-grade review lane beyond training datasets.
Connects JSON/JSONL input files, repositories, and the mobile review server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain.review import (
    ReviewItem,
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSession,
    ReviewSurface,
    build_review_provenance,
    build_review_decision,
)
from echozero.foundry.persistence import ReviewSessionRepository
from echozero.foundry.review_import import import_review_items_from_folder
from echozero.foundry.services.project_review_queue_builder import ProjectReviewQueueBuilder
from echozero.foundry.services.review_signal_codec import (
    coerce_review_decision,
    serialize_review_decision,
)
from echozero.foundry.services.review_signal_service import ReviewSignalService


class ReviewSessionService:
    """Creates, loads, filters, and updates persisted review sessions."""

    def __init__(
        self,
        root: Path,
        repository: ReviewSessionRepository | None = None,
        signal_service: ReviewSignalService | None = None,
        project_queue_builder: ProjectReviewQueueBuilder | None = None,
    ):
        self._root = root
        self._repo = repository or ReviewSessionRepository(root)
        self._signal_service = signal_service or ReviewSignalService(root)
        self._project_queue_builder = project_queue_builder or ProjectReviewQueueBuilder(root)

    def import_session_file(
        self,
        items_path: str | Path,
        *,
        name: str | None = None,
        session_id: str | None = None,
    ) -> ReviewSession:
        """Import review items from JSON or JSONL into a persisted session."""
        path = Path(items_path)
        rows = self._load_rows(path)
        if not rows:
            raise ValueError(f"No review items found in {path}")
        session = ReviewSession(
            id=session_id or f"rev_{uuid4().hex[:12]}",
            name=name or path.stem.replace("_", " ").strip() or "Review Session",
            items=[self._build_item(row) for row in rows],
            source_ref=str(path.resolve()),
            metadata={"import_format": path.suffix.lower().lstrip(".") or "json"},
        )
        return self._repo.save(session)

    def import_session_folder(
        self,
        folder_path: str | Path,
        *,
        name: str | None = None,
        session_id: str | None = None,
        target_class: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
    ) -> ReviewSession:
        """Import one review session directly from a host audio folder."""
        base = Path(folder_path)
        if not base.exists() or not base.is_dir():
            raise ValueError(f"Review folder not found: {base}")

        imported = import_review_items_from_folder(
            base,
            target_class=target_class,
            polarity=polarity,
        )
        if not imported.items:
            if imported.skipped_sources:
                raise ValueError(
                    f"No valid review items found in folder: {base} "
                    f"(skipped {len(imported.skipped_sources)} invalid file(s))"
                )
            raise ValueError(f"No audio review items found in folder: {base}")

        resolved_target_class = (target_class or "").strip() or None
        session = ReviewSession(
            id=session_id or f"rev_{uuid4().hex[:12]}",
            name=name or base.stem.replace("_", " ").strip() or "Review Session",
            items=imported.items,
            source_ref=str(base.resolve()),
            metadata={
                "import_format": "folder",
                "polarity": polarity.value,
                "target_class": resolved_target_class,
                "skipped_invalid_count": len(imported.skipped_sources),
                "skipped_invalid_by_reason": dict(sorted(imported.skipped_reason_counts.items())),
                "skipped_sources": imported.skipped_sources,
            },
        )
        return self._repo.save(session)

    def create_project_session(
        self,
        project_path: str | Path,
        *,
        name: str | None = None,
        session_id: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
        application_session: dict[str, object] | None = None,
    ) -> ReviewSession:
        """Create one persisted review session from canonical EZ project data."""
        queue = self._project_queue_builder.build_queue(
            project_path,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=polarity,
            review_mode=review_mode,
            questionable_score_threshold=questionable_score_threshold,
            item_limit=item_limit,
        )
        metadata = dict(queue.metadata)
        if application_session:
            metadata["application_session"] = dict(application_session)
        session = ReviewSession(
            id=session_id or f"rev_{uuid4().hex[:12]}",
            name=name or f"{queue.project_name} Review",
            items=queue.items,
            source_ref=queue.source_ref,
            metadata=metadata,
        )
        return self._repo.save(session)

    def get_session(self, session_id: str) -> ReviewSession | None:
        """Load a persisted review session."""
        return self._repo.get(session_id)

    def list_sessions(self) -> list[ReviewSession]:
        """Return all persisted review sessions."""
        return self._repo.list()

    def set_item_outcome(self, session_id: str, item_id: str, outcome: ReviewOutcome) -> ReviewSession:
        """Persist a human review decision for one item in a session."""
        return self.set_item_review(
            session_id,
            item_id,
            outcome=outcome,
            corrected_label=None,
            review_note=None,
        )

    def set_item_review(
        self,
        session_id: str,
        item_id: str,
        *,
        outcome: ReviewOutcome,
        corrected_label: str | None,
        review_note: str | None,
        decision_kind: ReviewDecisionKind | str | None = None,
        original_start_ms: float | None = None,
        original_end_ms: float | None = None,
        corrected_start_ms: float | None = None,
        corrected_end_ms: float | None = None,
        created_event_ref: str | None = None,
        surface: ReviewSurface | str = ReviewSurface.PHONE_REVIEW,
        workflow: str | None = "manual_review",
        operator_action: str | None = None,
    ) -> ReviewSession:
        """Persist a human review decision plus optional relabeling context."""
        session = self._repo.get(session_id)
        if session is None:
            raise ValueError(f"ReviewSession not found: {session_id}")
        timestamp = datetime.now(UTC)
        normalized_label = _normalize_optional_text(corrected_label)
        normalized_note = _normalize_optional_text(review_note)
        updated_items: list[ReviewItem] = []
        found = False
        for item in session.items:
            if item.item_id != item_id:
                updated_items.append(item)
                continue
            updated_items.append(
                ReviewItem(
                    item_id=item.item_id,
                    audio_path=item.audio_path,
                    predicted_label=item.predicted_label,
                    target_class=item.target_class,
                    polarity=item.polarity,
                    score=item.score,
                    source_provenance=item.source_provenance,
                    review_outcome=outcome,
                    review_decision=build_review_decision(
                        outcome,
                        corrected_label=normalized_label if outcome == ReviewOutcome.INCORRECT else None,
                        review_note=normalized_note if outcome == ReviewOutcome.INCORRECT else None,
                        decision_kind=decision_kind,
                        original_start_ms=original_start_ms if outcome == ReviewOutcome.INCORRECT else None,
                        original_end_ms=original_end_ms if outcome == ReviewOutcome.INCORRECT else None,
                        corrected_start_ms=corrected_start_ms if outcome == ReviewOutcome.INCORRECT else None,
                        corrected_end_ms=corrected_end_ms if outcome == ReviewOutcome.INCORRECT else None,
                        created_event_ref=created_event_ref if outcome == ReviewOutcome.INCORRECT else None,
                        provenance=build_review_provenance(
                            item.source_provenance,
                            surface=surface,
                            workflow=workflow,
                            operator_action=operator_action,
                            queue_session_ref=session.id,
                        ),
                    ),
                    corrected_label=normalized_label if outcome == ReviewOutcome.INCORRECT else None,
                    review_note=normalized_note if outcome == ReviewOutcome.INCORRECT else None,
                    reviewed_at=timestamp,
                )
            )
            found = True
        if not found:
            raise ValueError(f"ReviewItem not found: {item_id}")
        updated_session = ReviewSession(
            id=session.id,
            name=session.name,
            items=updated_items,
            source_ref=session.source_ref,
            metadata=session.metadata,
            created_at=session.created_at,
            updated_at=timestamp,
        )
        saved_session = self._repo.save(updated_session)
        reviewed_item = next(item for item in saved_session.items if item.item_id == item_id)
        if reviewed_item.review_outcome != ReviewOutcome.PENDING and reviewed_item.review_decision is not None:
            self._signal_service.record_session_item_review(saved_session, reviewed_item)
        return saved_session

    def filter_items(
        self,
        session: ReviewSession,
        *,
        outcome: str = "pending",
        polarity: str = "all",
        target_class: str = "all",
        song_ref: str = "all",
        layer_ref: str = "all",
    ) -> list[ReviewItem]:
        """Return items matching the requested UI filters."""
        filtered = list(session.items)
        if outcome != "all":
            filtered = [item for item in filtered if item.review_outcome.value == outcome]
        if polarity != "all":
            filtered = [item for item in filtered if item.polarity.value == polarity]
        if target_class != "all":
            filtered = [item for item in filtered if item.target_class == target_class]
        if song_ref != "all":
            filtered = [
                item
                for item in filtered
                if _source_text(item.source_provenance, "song_ref", "songRef") == song_ref
            ]
        if layer_ref != "all":
            filtered = [
                item
                for item in filtered
                if _source_text(item.source_provenance, "layer_ref", "layerRef") == layer_ref
            ]
        return filtered

    def build_snapshot(
        self,
        session_id: str,
        *,
        outcome: str = "pending",
        polarity: str = "all",
        target_class: str = "all",
        song_ref: str = "all",
        layer_ref: str = "all",
        cursor: int = 0,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Build a UI/API payload for the requested review session and filters."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"ReviewSession not found: {session_id}")
        filtered_items = self.filter_items(
            session,
            outcome=outcome,
            polarity=polarity,
            target_class=target_class,
            song_ref=song_ref,
            layer_ref=layer_ref,
        )
        normalized_cursor = _normalize_cursor(cursor, item_count=len(filtered_items))
        focused_item, focused_item_visible = _resolve_focused_item(
            session,
            filtered_items=filtered_items,
            normalized_cursor=normalized_cursor,
            item_id=item_id,
        )
        if focused_item_visible:
            normalized_cursor = next(
                index
                for index, filtered_item in enumerate(filtered_items)
                if filtered_item.item_id == focused_item.item_id
            )
        counts_by_outcome = {
            review_outcome.value: sum(1 for item in session.items if item.review_outcome == review_outcome)
            for review_outcome in ReviewOutcome
        }
        counts_by_polarity = {
            review_polarity.value: sum(1 for item in session.items if item.polarity == review_polarity)
            for review_polarity in ReviewPolarity
        }
        reviewed_count = counts_by_outcome[ReviewOutcome.CORRECT.value] + counts_by_outcome[ReviewOutcome.INCORRECT.value]
        pending_count = counts_by_outcome[ReviewOutcome.PENDING.value]
        return {
            "session": {
                "id": session.id,
                "name": session.name,
                "sourceRef": session.source_ref,
                "createdAt": session.created_at.isoformat(),
                "updatedAt": session.updated_at.isoformat(),
                "totalItems": len(session.items),
                "classMap": session.class_map,
                "reviewMode": str(session.metadata.get("review_mode", "all_events")),
                "applicationSession": _application_session_payload(session.metadata),
                "scopeOptions": _build_scope_options(session, song_ref=song_ref),
                "countsByOutcome": counts_by_outcome,
                "countsByPolarity": counts_by_polarity,
                "reviewedCount": reviewed_count,
                "pendingCount": pending_count,
                "completionRatio": (float(reviewed_count) / float(len(session.items))) if session.items else 0.0,
            },
            "sessions": self.build_session_index(default_session_id=session.id),
            "filters": {
                "outcome": outcome,
                "polarity": polarity,
                "targetClass": target_class,
                "songRef": song_ref,
                "layerRef": layer_ref,
            },
            "filteredCount": len(filtered_items),
            "navigation": {
                "cursor": normalized_cursor if filtered_items else None,
                "currentItemNumber": (normalized_cursor + 1) if filtered_items else 0,
                "filteredCount": len(filtered_items),
                "hasPrevious": bool(filtered_items) and normalized_cursor > 0,
                "hasNext": bool(filtered_items) and normalized_cursor < (len(filtered_items) - 1),
                "previousCursor": (normalized_cursor - 1) if filtered_items and normalized_cursor > 0 else None,
                "nextCursor": (
                    normalized_cursor + 1
                ) if filtered_items and normalized_cursor < (len(filtered_items) - 1) else None,
                "viewMode": "queue" if focused_item is None or focused_item_visible else "history",
                "focusedItemId": focused_item.item_id if focused_item is not None else None,
                "focusedItemVisible": focused_item_visible,
            },
            "currentItem": (
                self._serialize_item(focused_item, session_id=session.id)
                if focused_item is not None
                else None
            ),
        }

    def build_session_index(self, *, default_session_id: str | None = None) -> dict[str, object]:
        """Return a compact list of available review sessions for phone navigation."""
        sessions = sorted(self.list_sessions(), key=lambda session: session.updated_at, reverse=True)
        return {
            "defaultSessionId": default_session_id,
            "items": [self._serialize_session_summary(session) for session in sessions],
        }

    def _load_rows(self, path: Path) -> list[dict]:
        if not path.exists():
            raise ValueError(f"Review items file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                rows = payload["items"]
            else:
                raise ValueError(f"Unsupported review JSON payload in {path}")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Review items in {path} must be JSON objects")
        return rows

    def _build_item(self, row: dict) -> ReviewItem:
        audio_path = self._coerce_required_str(row, "audio_path", "audioPath", "audio", "path")
        target_class = self._coerce_required_str(row, "target_class", "targetClass", "class", "label")
        predicted_label = self._coerce_optional_str(row, "predicted_label", "predictedLabel", "prediction", "label")
        polarity_value = self._coerce_required_str(row, "polarity")
        outcome_value = self._coerce_optional_str(row, "review_outcome", "reviewOutcome", "outcome")
        item_id = self._coerce_optional_str(row, "item_id", "itemId", "id") or f"ri_{uuid4().hex[:12]}"
        score_value = row.get("score", row.get("confidence"))
        source_provenance = row.get("source_provenance", row.get("sourceProvenance", row.get("provenance", {})))
        normalized_outcome = ReviewOutcome(outcome_value or ReviewOutcome.PENDING.value)
        normalized_source_provenance = (
            source_provenance if isinstance(source_provenance, dict) else {"value": source_provenance}
        )
        corrected_label = self._coerce_optional_str(row, "corrected_label", "correctedLabel", "resolvedLabel")
        review_note = self._coerce_optional_str(row, "review_note", "reviewNote", "note", "description")
        review_decision = coerce_review_decision(
            row,
            outcome=normalized_outcome,
            corrected_label=corrected_label,
            review_note=review_note,
            source_provenance=normalized_source_provenance,
        )
        return ReviewItem(
            item_id=item_id,
            audio_path=str(Path(audio_path).expanduser().resolve()),
            predicted_label=predicted_label or target_class,
            target_class=target_class,
            polarity=ReviewPolarity(polarity_value),
            score=float(score_value) if score_value is not None else None,
            source_provenance=normalized_source_provenance,
            review_outcome=normalized_outcome,
            review_decision=review_decision,
            corrected_label=corrected_label,
            review_note=review_note,
        )

    @staticmethod
    def _coerce_required_str(row: dict, *keys: str) -> str:
        value = ReviewSessionService._coerce_optional_str(row, *keys)
        if not value:
            joined_keys = ", ".join(keys)
            raise ValueError(f"Review item is missing one of: {joined_keys}")
        return value

    @staticmethod
    def _coerce_optional_str(row: dict, *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _serialize_item(item: ReviewItem, *, session_id: str) -> dict[str, object]:
        song_ref = _source_text(item.source_provenance, "song_ref", "songRef")
        song_title = _source_text(item.source_provenance, "song_title", "songTitle")
        version_ref = _source_text(item.source_provenance, "version_ref", "versionRef")
        version_label = _source_text(item.source_provenance, "version_label", "versionLabel")
        layer_ref = _source_text(item.source_provenance, "layer_ref", "layerRef")
        layer_name = _source_text(item.source_provenance, "layer_name", "layerName")
        return {
            "itemId": item.item_id,
            "audioPath": item.audio_path,
            "audioUrl": f"/audio/{item.item_id}?sessionId={session_id}",
            "predictedLabel": item.predicted_label,
            "targetClass": item.target_class,
            "polarity": item.polarity.value,
            "score": item.score,
            "songRef": song_ref,
            "songTitle": song_title,
            "versionRef": version_ref,
            "versionLabel": version_label,
            "layerRef": layer_ref,
            "layerName": layer_name,
            "sourceProvenance": item.source_provenance,
            "reviewOutcome": item.review_outcome.value,
            "reviewDecision": serialize_review_decision(item.review_decision),
            "correctedLabel": item.corrected_label,
            "reviewNote": item.review_note,
            "reviewedAt": item.reviewed_at.isoformat() if item.reviewed_at else None,
            "fileName": Path(item.audio_path).name,
            "promptText": f"Does this sound like {item.predicted_label.upper()}?",
            "laneText": f"{item.target_class} lane · {item.polarity.value}",
        }

    @staticmethod
    def _serialize_session_summary(session: ReviewSession) -> dict[str, object]:
        pending_count = sum(
            1 for item in session.items if item.review_outcome == ReviewOutcome.PENDING
        )
        return {
            "id": session.id,
            "name": session.name,
            "updatedAt": session.updated_at.isoformat(),
            "totalItems": len(session.items),
            "pendingCount": pending_count,
            "classMap": session.class_map,
            "reviewMode": str(session.metadata.get("review_mode", "all_events")),
        }



def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_cursor(value: int, *, item_count: int) -> int:
    if item_count <= 0:
        return 0
    return max(0, min(int(value), item_count - 1))


def _resolve_focused_item(
    session: ReviewSession,
    *,
    filtered_items: list[ReviewItem],
    normalized_cursor: int,
    item_id: str | None,
) -> tuple[ReviewItem | None, bool]:
    if item_id:
        focused_item = next((item for item in session.items if item.item_id == item_id), None)
        if focused_item is None:
            raise ValueError(f"ReviewItem not found: {item_id}")
        return focused_item, any(item.item_id == item_id for item in filtered_items)
    if not filtered_items:
        return None, False
    return filtered_items[normalized_cursor], True


def _build_scope_options(session: ReviewSession, *, song_ref: str) -> dict[str, list[dict[str, object]]]:
    songs: dict[str, dict[str, object]] = {}
    for item in session.items:
        item_song_ref = _source_text(item.source_provenance, "song_ref", "songRef")
        if not item_song_ref:
            continue
        entry = songs.setdefault(
            item_song_ref,
            {
                "value": item_song_ref,
                "label": _source_text(item.source_provenance, "song_title", "songTitle")
                or _readable_ref_label(item_song_ref),
                "itemCount": 0,
            },
        )
        entry["itemCount"] = int(entry["itemCount"]) + 1

    layer_source = session.items
    if song_ref != "all":
        layer_source = [
            item
            for item in session.items
            if _source_text(item.source_provenance, "song_ref", "songRef") == song_ref
        ]
    has_multiple_songs = len(songs) > 1
    layers: dict[str, dict[str, object]] = {}
    for item in layer_source:
        item_layer_ref = _source_text(item.source_provenance, "layer_ref", "layerRef")
        if not item_layer_ref:
            continue
        song_label = _source_text(item.source_provenance, "song_title", "songTitle") or _readable_ref_label(
            _source_text(item.source_provenance, "song_ref", "songRef")
        )
        base_label = (
            _source_text(item.source_provenance, "layer_name", "layerName")
            or item.target_class
            or _readable_ref_label(item_layer_ref)
        )
        label = f"{base_label} · {song_label}" if song_ref == "all" and has_multiple_songs else base_label
        entry = layers.setdefault(
            item_layer_ref,
            {
                "value": item_layer_ref,
                "label": label,
                "songRef": _source_text(item.source_provenance, "song_ref", "songRef"),
                "itemCount": 0,
            },
        )
        entry["itemCount"] = int(entry["itemCount"]) + 1

    return {
        "songs": sorted(songs.values(), key=lambda entry: str(entry["label"]).lower()),
        "layers": sorted(layers.values(), key=lambda entry: str(entry["label"]).lower()),
    }


def _application_session_payload(metadata: dict[str, object]) -> dict[str, object] | None:
    payload = metadata.get("application_session")
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _source_text(payload: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _readable_ref_label(value: str | None) -> str:
    if not value:
        return "Unknown"
    _, _, suffix = str(value).partition(":")
    text = suffix or str(value)
    return text.replace("_", " ").strip() or str(value)
