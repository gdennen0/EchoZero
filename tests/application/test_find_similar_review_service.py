"""Find Similar review-service behavior tests.
Exists to prove interactive review labels drive local ranking without Qt widget state.
Connects application-level candidate sessions to saved review model artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import wave
from pathlib import Path

import numpy as np

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.timeline.find_similar_review_service import (
    REVIEW_MODEL_SCHEMA,
    FindSimilarReviewPhase,
    FindSimilarReviewService,
    ReviewLabel,
    save_find_similar_review_model,
    score_saved_review_model,
)
from echozero.application.timeline.models import EventRef


def test_initial_scan_ranks_similar_events_above_different_events(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()

    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    score_by_id = {
        candidate.event_ref.event_id: candidate.score for candidate in session.candidates
    }
    assert score_by_id[EventId("kick_b")] > score_by_id[EventId("hat")]
    assert session.next_candidate_ref == EventRef(
        LayerId("layer"), TakeId("take"), EventId("kick_b")
    )
    assert session.top_candidates[0].event_ref.event_id == EventId("kick_b")
    assert len(session.review_candidates) == min(15, len(session.top_candidates))
    assert session.can_select_similar is False
    assert "Need 4 more matches" in session.match_profile.readiness_reason


def test_positive_and_negative_labels_rerank_remaining_candidates(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    session = service.mark_candidate(
        session,
        EventRef(LayerId("layer"), TakeId("take"), EventId("hat")),
        ReviewLabel.NEGATIVE,
    )

    score_by_id = {
        candidate.event_ref.event_id: candidate.score for candidate in session.candidates
    }
    assert score_by_id[EventId("hat")] == 0.0
    assert score_by_id[EventId("kick_b")] > score_by_id[EventId("hat_b")]
    assert session.negative_count == 1
    assert EventId("hat") not in [ref.event_id for ref in session.model_result_event_refs]


def test_confidence_threshold_filters_without_reordering(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    filtered = service.set_confidence_threshold(session, 0.85)

    assert [candidate.event_ref.event_id for candidate in filtered.candidates] == [
        candidate.event_ref.event_id for candidate in session.candidates
    ]
    assert all(
        candidate.passes_confidence or candidate.score < 0.85
        for candidate in filtered.candidates
        if not candidate.is_anchor
    )


def test_top_candidates_are_ranked_by_confidence_not_timeline_order(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    ranked_ids = [candidate.event_ref.event_id for candidate in session.top_candidates]

    assert ranked_ids.index(EventId("kick_b")) < ranked_ids.index(EventId("hat"))
    assert EventId("kick_a") not in ranked_ids


def test_anchor_plus_four_positive_examples_enables_stable_selection(
    tmp_path: Path,
) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    for event_id in ("kick_b", "kick_c", "kick_d", "kick_e"):
        session = service.mark_candidate(
            session,
            EventRef(LayerId("layer"), TakeId("take"), EventId(event_id)),
            ReviewLabel.POSITIVE,
        )

    assert session.can_select_similar is True
    assert session.can_train is True
    assert session.match_profile.selection_delta_count <= 1
    assert session.phase == FindSimilarReviewPhase.CHOOSE_EXAMPLES
    strict_result_ids = [ref.event_id for ref in session.model_result_event_refs]
    assert strict_result_ids == [
        EventId("kick_a"),
        EventId("kick_b"),
        EventId("kick_c"),
        EventId("kick_d"),
        EventId("kick_e"),
    ]


def test_selection_stays_gated_until_top_set_is_stable(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    seed_refs = [
        EventRef(LayerId("layer"), TakeId("take"), EventId(event_id))
        for event_id in ("kick_b", "kick_c", "kick_d", "kick_e")
    ]

    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
        seed_event_refs=seed_refs,
    )

    assert session.has_enough_profile_evidence is True
    assert session.match_profile.selection_delta_count > 1
    assert session.can_select_similar is False
    assert session.match_profile.readiness_reason == (
        "Selection still changing; review the next candidate."
    )


def test_negative_and_unsure_labels_never_appear_in_final_selection(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    for event_id in ("kick_b", "kick_c", "kick_d", "kick_e"):
        session = service.mark_candidate(
            session,
            EventRef(LayerId("layer"), TakeId("take"), EventId(event_id)),
            ReviewLabel.POSITIVE,
        )
    session = service.mark_candidate(
        session,
        EventRef(LayerId("layer"), TakeId("take"), EventId("hat")),
        ReviewLabel.NEGATIVE,
    )
    session = service.mark_candidate(
        session,
        EventRef(LayerId("layer"), TakeId("take"), EventId("hat_b")),
        ReviewLabel.SKIPPED,
    )
    selection = service.select_similar_events(session)
    selected_ids = [ref.event_id for ref in selection.event_refs]

    assert EventId("hat") not in selected_ids
    assert EventId("hat_b") not in selected_ids


def test_train_review_model_is_profile_compatibility_shim(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    refreshed = service.train_review_model(session)

    assert refreshed.phase == FindSimilarReviewPhase.CHOOSE_EXAMPLES
    assert refreshed.model_result_event_refs == refreshed.match_profile.selected_event_refs


def test_no_audio_candidates_fail_gracefully() -> None:
    service = FindSimilarReviewService()

    session = service.start_session(
        presentation=_presentation(None),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )

    assert all(candidate.score == 0.0 or candidate.is_anchor for candidate in session.candidates)
    assert session.matched_candidates[0].event_ref.event_id == EventId("kick_a")


def test_saved_review_model_scores_and_preserves_review_refs(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    service = FindSimilarReviewService()
    session = service.start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )
    session = service.mark_candidate(
        session,
        EventRef(LayerId("layer"), TakeId("take"), EventId("kick_b")),
        ReviewLabel.POSITIVE,
    )
    session = service.mark_candidate(
        session,
        EventRef(LayerId("layer"), TakeId("take"), EventId("hat")),
        ReviewLabel.NEGATIVE,
    )

    model_path = save_find_similar_review_model(
        session,
        output_dir=tmp_path,
        created_at=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
    )

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["schema"] == REVIEW_MODEL_SCHEMA
    assert payload["confidence_threshold"] == session.confidence_threshold
    assert len(payload["positive_event_refs"]) == 2
    assert len(payload["negative_event_refs"]) == 1
    kick_b = next(
        candidate
        for candidate in session.candidates
        if candidate.event_ref.event_id == EventId("kick_b")
    )
    hat = next(
        candidate
        for candidate in session.candidates
        if candidate.event_ref.event_id == EventId("hat")
    )
    assert score_saved_review_model(model_path, kick_b.embedding) > score_saved_review_model(
        model_path, hat.embedding
    )


def test_legacy_mini_model_artifact_scores_as_fallback(tmp_path: Path) -> None:
    audio_path = _review_audio_path(tmp_path)
    session = FindSimilarReviewService().start_session(
        presentation=_presentation(audio_path),
        layer_id=LayerId("layer"),
        take_id=TakeId("take"),
        event_id=EventId("kick_a"),
    )
    anchor = next(
        candidate
        for candidate in session.candidates
        if candidate.event_ref.event_id == EventId("kick_a")
    )
    kick_b = next(
        candidate
        for candidate in session.candidates
        if candidate.event_ref.event_id == EventId("kick_b")
    )
    legacy_path = tmp_path / "legacy-mini-model.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema": "echozero.find-similar-mini-model.v1",
                "model_id": "legacy",
                "model_kind": "timbre_prototype",
                "created_at": "2026-05-18T12:00:00+00:00",
                "anchor_label": "Kick",
                "anchor_event_ref": {"layer_id": "layer", "take_id": "take", "event_id": "kick_a"},
                "settings": {"sample_count": 64, "padding_ms": 20.0},
                "positive_sample_count": 1,
                "centroid": list(anchor.embedding),
            }
        ),
        encoding="utf-8",
    )

    assert score_saved_review_model(legacy_path, kick_b.embedding) > 0.7


def _presentation(audio_path: Path | None) -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline"),
        title="Find Similar",
        selected_layer_ids=[LayerId("layer")],
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer"),
                title="Layer",
                main_take_id=TakeId("take"),
                kind=LayerKind.EVENT,
                source_audio_path=str(audio_path) if audio_path is not None else None,
                events=[
                    EventPresentation(EventId("kick_a"), 0.0, 0.18, "Kick A"),
                    EventPresentation(EventId("kick_b"), 0.5, 0.68, "Kick B"),
                    EventPresentation(EventId("hat"), 1.0, 1.12, "Hat"),
                    EventPresentation(EventId("hat_b"), 1.44, 1.56, "Hat B"),
                    EventPresentation(EventId("kick_c"), 1.88, 2.06, "Kick C"),
                    EventPresentation(EventId("kick_d"), 2.38, 2.56, "Kick D"),
                    EventPresentation(EventId("kick_e"), 2.88, 3.06, "Kick E"),
                    EventPresentation(EventId("kick_f"), 3.38, 3.56, "Kick F"),
                ],
            )
        ],
    )


def _review_audio_path(tmp_path: Path) -> Path:
    kick_a = _burst(120.0, duration_seconds=0.18)
    kick_b = _burst(120.0, duration_seconds=0.18)
    hat = _burst(2200.0, duration_seconds=0.12)
    hat_b = _burst(2300.0, duration_seconds=0.12)
    kick_c = _burst(120.0, duration_seconds=0.18)
    kick_d = _burst(120.0, duration_seconds=0.18)
    kick_e = _burst(120.0, duration_seconds=0.18)
    kick_f = _burst(120.0, duration_seconds=0.18)
    silence = np.zeros(int(22050 * 0.32), dtype=np.float32)
    audio = np.concatenate(
        (
            kick_a,
            silence,
            kick_b,
            silence,
            hat,
            silence,
            hat_b,
            silence,
            kick_c,
            silence,
            kick_d,
            silence,
            kick_e,
            silence,
            kick_f,
        )
    )
    path = tmp_path / "review.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(np.clip(audio * 32767.0, -32768, 32767).astype("<i2").tobytes())
    return path


def _burst(frequency_hz: float, *, duration_seconds: float) -> np.ndarray:
    sample_rate = 22050
    times = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    envelope = np.exp(-times * 18.0)
    return (np.sin(2.0 * np.pi * frequency_hz * times) * envelope).astype(np.float32)
