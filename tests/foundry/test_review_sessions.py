from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from echozero.domain.types import EventData, Layer as DomainLayer
from echozero.foundry.domain import ReviewDecisionKind, ReviewItem, ReviewOutcome, ReviewPolarity
from echozero.foundry.persistence import ReviewSessionRepository
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.foundry.review_server import create_review_http_server
from echozero.foundry.services.project_review_queue_builder import ProjectReviewQueue
from echozero.foundry.services.review_event_state import updated_review_metadata
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.persistence.session import ProjectStorage
from tests.foundry.audio_fixtures import write_percussion_dataset
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture


def test_review_session_import_round_trip_and_update(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path, name="Mobile Review")
    updated = service.set_item_outcome(session.id, session.items[0].item_id, ReviewOutcome.CORRECT)
    persisted = ReviewSessionRepository(tmp_path).get(session.id)

    assert updated.name == "Mobile Review"
    assert len(updated.items) == 2
    assert persisted is not None
    assert persisted.items[0].review_outcome == ReviewOutcome.CORRECT
    assert persisted.items[0].review_decision is not None
    assert persisted.items[0].review_decision.kind == ReviewDecisionKind.VERIFIED
    assert persisted.items[0].review_decision.provenance is not None
    assert persisted.items[0].review_decision.provenance.surface.value == "phone_review"
    assert persisted.items[0].review_decision.provenance.project_ref == "project:fixture"
    assert persisted.items[0].review_decision.provenance.song_ref == "song:arcade"
    assert persisted.items[0].review_decision.provenance.layer_ref == "layer:kick"
    assert persisted.items[0].review_decision.training_eligibility.allows_positive_signal is True
    assert persisted.items[0].review_decision.training_eligibility.allows_negative_signal is False
    assert persisted.items[0].reviewed_at is not None
    assert persisted.class_map == ["kick", "snare"]


def test_review_session_reclassify_persists_label_and_note(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path)
    updated = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="short mid drum with more body than kick",
    )
    persisted = ReviewSessionRepository(tmp_path).get(session.id)

    assert updated.items[0].corrected_label == "tom"
    assert updated.items[0].review_note == "short mid drum with more body than kick"
    assert updated.items[0].review_decision is not None
    assert updated.items[0].review_decision.kind == ReviewDecisionKind.RELABELED
    assert persisted is not None
    assert persisted.items[0].corrected_label == "tom"
    assert persisted.items[0].review_note == "short mid drum with more body than kick"
    assert persisted.items[0].review_decision is not None
    assert persisted.items[0].review_decision.kind == ReviewDecisionKind.RELABELED
    assert persisted.items[0].review_decision.training_eligibility.allows_positive_signal is True
    assert persisted.items[0].review_decision.training_eligibility.allows_negative_signal is True


def test_review_session_snapshot_supports_cursor_navigation_and_reject_decision(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path)
    second_snapshot = service.build_snapshot(session.id, outcome="pending", cursor=1)
    updated = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label=None,
        review_note=None,
    )
    refreshed = service.build_snapshot(session.id, outcome="all", cursor=0)

    assert second_snapshot["currentItem"]["itemId"] == session.items[1].item_id
    assert second_snapshot["navigation"]["hasPrevious"] is True
    assert second_snapshot["navigation"]["hasNext"] is False
    assert len(second_snapshot["progress"]["items"]) == 2
    assert second_snapshot["progress"]["items"][1]["isCurrent"] is True
    assert updated.items[0].review_decision is not None
    assert updated.items[0].review_decision.kind == ReviewDecisionKind.REJECTED
    assert refreshed["currentItem"]["reviewDecision"]["kind"] == "rejected"
    assert refreshed["currentItem"]["reviewDecision"]["trainingEligibility"]["allowsNegativeSignal"] is True
    assert refreshed["progress"]["items"][0]["reviewOutcome"] == "incorrect"


def test_review_session_pending_snapshot_progress_keeps_reviewed_scope_status(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path)
    service.set_item_outcome(session.id, session.items[0].item_id, ReviewOutcome.CORRECT)
    snapshot = service.build_snapshot(session.id, outcome="pending")

    assert snapshot["filteredCount"] == 1
    assert snapshot["currentItem"]["itemId"] == session.items[1].item_id
    assert len(snapshot["progress"]["items"]) == 2
    assert snapshot["progress"]["items"][0]["reviewOutcome"] == "correct"
    assert snapshot["progress"]["items"][1]["isCurrent"] is True
    assert snapshot["navigation"]["currentScopeItemNumber"] == 2
    assert snapshot["navigation"]["scopeCount"] == 2
    assert snapshot["navigation"]["scopePendingCount"] == 1
    assert snapshot["navigation"]["scopeReviewedCount"] == 1


def test_review_session_supports_boundary_and_missed_event_decisions(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path)
    boundary_updated = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="kick",
        review_note="attack was right but timing drifted late",
        decision_kind=ReviewDecisionKind.BOUNDARY_CORRECTED,
        original_start_ms=112.5,
        original_end_ms=161.0,
        corrected_start_ms=104.0,
        corrected_end_ms=156.5,
        surface="timeline_fix_mode",
        workflow="timeline_event_edit",
        operator_action="retime_event",
    )
    missed_updated = service.set_item_review(
        session.id,
        session.items[1].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="snare",
        review_note="operator added the missed hit in the snare lane",
        decision_kind=ReviewDecisionKind.MISSED_EVENT_ADDED,
        created_event_ref="event:manual-snare-02",
        surface="timeline_fix_mode",
        workflow="timeline_event_edit",
        operator_action="add_missing_event",
    )
    persisted = ReviewSessionRepository(tmp_path).get(session.id)

    assert boundary_updated.items[0].review_decision is not None
    assert boundary_updated.items[0].review_decision.kind == ReviewDecisionKind.BOUNDARY_CORRECTED
    assert boundary_updated.items[0].review_decision.corrected_start_ms == 104.0
    assert boundary_updated.items[0].review_decision.training_eligibility.requires_materialized_correction is True
    assert missed_updated.items[1].review_decision is not None
    assert missed_updated.items[1].review_decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED
    assert missed_updated.items[1].review_decision.created_event_ref == "event:manual-snare-02"
    assert missed_updated.items[1].review_decision.provenance is not None
    assert missed_updated.items[1].review_decision.provenance.surface.value == "timeline_fix_mode"
    assert persisted is not None
    assert persisted.items[0].review_decision is not None
    assert persisted.items[0].review_decision.original_start_ms == 112.5
    assert persisted.items[1].review_decision is not None
    assert persisted.items[1].review_decision.created_event_ref == "event:manual-snare-02"


def test_review_repository_loads_legacy_review_decisions_with_default_semantics(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=1)
    state_dir = tmp_path / "foundry" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "foundry.state.review_sessions.v1",
        "version": 1,
        "items": {
            "rev_legacy": {
                "id": "rev_legacy",
                "name": "Legacy Queue",
                "source_ref": str((tmp_path / "legacy.json").resolve()),
                "metadata": {},
                "created_at": "2026-04-25T00:00:00+00:00",
                "updated_at": "2026-04-25T00:00:00+00:00",
                "items": [
                    {
                        "item_id": "ri_legacy",
                        "audio_path": str(samples / "kick" / "k1.wav"),
                        "predicted_label": "kick",
                        "target_class": "kick",
                        "polarity": "positive",
                        "score": 0.88,
                        "source_provenance": {
                            "project_ref": "project:fixture",
                            "song_ref": "song:arcade",
                            "layer_ref": "layer:kick",
                            "event_ref": "event:legacy-k1",
                            "model_ref": "bundle:fixture-v1",
                        },
                        "review_outcome": "incorrect",
                        "review_decision": {
                            "kind": "relabeled",
                            "corrected_label": "tom",
                            "review_note": "legacy payload without explicit provenance",
                        },
                        "corrected_label": "tom",
                        "review_note": "legacy payload without explicit provenance",
                        "reviewed_at": "2026-04-25T00:00:00+00:00",
                    }
                ],
            }
        },
    }
    (state_dir / "review_sessions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    session = ReviewSessionRepository(tmp_path).get("rev_legacy")

    assert session is not None
    assert session.items[0].review_decision is not None
    assert session.items[0].review_decision.provenance is not None
    assert session.items[0].review_decision.provenance.surface.value == "phone_review"
    assert session.items[0].review_decision.provenance.event_ref == "event:legacy-k1"
    assert session.items[0].review_decision.training_eligibility.allows_positive_signal is True
    assert session.items[0].review_decision.training_eligibility.allows_negative_signal is True


def test_review_session_import_accepts_jsonl(tmp_path: Path):
    review_items_path = _write_review_items_jsonl(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(review_items_path)
    snapshot = service.build_snapshot(session.id, outcome="all", polarity="negative", target_class="snare")

    assert len(session.items) == 2
    assert snapshot["filteredCount"] == 1
    assert snapshot["currentItem"]["targetClass"] == "snare"
    assert snapshot["currentItem"]["polarity"] == "negative"


def test_review_session_import_accepts_class_named_folder(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    service = ReviewSessionService(tmp_path)

    session = service.import_session_folder(samples, name="Arcade Queue")
    snapshot = service.build_snapshot(session.id, outcome="pending")

    assert session.name == "Arcade Queue"
    assert len(session.items) == 8
    assert session.class_map == ["kick", "snare"]
    assert snapshot["session"]["pendingCount"] == 8
    assert snapshot["currentItem"]["promptText"].startswith("Does this sound like")


def test_review_session_import_accepts_flat_folder_with_target_class(tmp_path: Path):
    flat = tmp_path / "flat"
    flat.mkdir(parents=True, exist_ok=True)
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=1)
    source = samples / "kick" / "k1.wav"
    (flat / "clip_a.wav").write_bytes(source.read_bytes())
    (flat / "clip_b.wav").write_bytes(source.read_bytes())
    service = ReviewSessionService(tmp_path)

    session = service.import_session_folder(flat, target_class="kick")

    assert len(session.items) == 2
    assert session.class_map == ["kick"]
    assert all(item.predicted_label == "kick" for item in session.items)


def test_review_session_project_snapshot_includes_application_session_metadata(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        song_id=refs["alpha_song_id"],
        review_mode="all_events",
        application_session={
            "sessionId": "sess_123",
            "projectId": "proj_456",
            "projectName": "Arcade Project",
            "projectRef": refs["project_ref"],
            "activeSongId": refs["alpha_song_id"],
            "activeSongRef": refs["alpha_song_ref"],
            "activeSongTitle": "Alpha Song",
            "activeSongVersionId": "ver_alpha",
            "activeSongVersionRef": refs["alpha_version_ref"],
            "activeSongVersionLabel": "Alpha Song",
        },
    )
    snapshot = service.build_snapshot(session.id, outcome="all")

    assert session.metadata["review_mode"] == "all_events"
    assert session.metadata["application_session"]["sessionId"] == "sess_123"
    assert snapshot["session"]["reviewMode"] == "all_events"
    assert snapshot["session"]["context"] == {
        "projectName": "Arcade Project",
        "projectRef": refs["project_ref"],
        "sourceKind": "ez_project",
        "sourceLabel": "Arcade Project",
        "sourceRef": str(working_dir.resolve()),
    }
    assert snapshot["session"]["applicationSession"] == {
        "sessionId": "sess_123",
        "projectId": "proj_456",
        "projectName": "Arcade Project",
        "projectRef": refs["project_ref"],
        "activeSongId": refs["alpha_song_id"],
        "activeSongRef": refs["alpha_song_ref"],
        "activeSongTitle": "Alpha Song",
        "activeSongVersionId": "ver_alpha",
        "activeSongVersionRef": refs["alpha_version_ref"],
        "activeSongVersionLabel": "Alpha Song",
    }


def test_project_review_snapshot_refreshes_from_canonical_event_state(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Project Queue",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    _mark_project_event_review_state(
        working_dir,
        layer_id="layer_alpha_kick",
        event_id="evt_alpha_kick_01",
        promotion_state="demoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.REJECTED,
        original_label="kick",
        corrected_label=None,
        review_note="timeline rejected this false positive",
    )

    snapshot = service.build_snapshot(session.id, outcome="all", item_id=session.items[0].item_id)

    assert snapshot["currentItem"]["itemId"] == session.items[0].item_id
    assert snapshot["currentItem"]["reviewOutcome"] == "incorrect"
    assert snapshot["currentItem"]["reviewDecision"]["kind"] == "rejected"
    assert snapshot["currentItem"]["sourceProvenance"]["promotion_state"] == "demoted"
    assert snapshot["currentItem"]["sourceProvenance"]["review_state"] == "corrected"


def test_project_review_session_index_refreshes_counts_from_canonical_event_state(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Project Queue",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    _mark_project_event_review_state(
        working_dir,
        layer_id="layer_alpha_kick",
        event_id="evt_alpha_kick_01",
        promotion_state="demoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.REJECTED,
        original_label="kick",
        corrected_label=None,
        review_note="timeline rejected this false positive",
    )

    sessions_index = service.build_session_index(default_session_id=session.id)

    assert sessions_index["defaultSessionId"] == session.id
    assert sessions_index["items"][0]["pendingCount"] == 1
    assert sessions_index["items"][0]["reviewedCount"] == 1


def test_review_session_snapshot_supports_song_layer_filters_and_history_focus(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(working_dir, name="Project Queue")
    alpha_snapshot = service.build_snapshot(
        session.id,
        outcome="pending",
        song_ref=refs["alpha_song_ref"],
    )
    layer_snapshot = service.build_snapshot(
        session.id,
        outcome="pending",
        song_ref=refs["alpha_song_ref"],
        layer_ref="layer:layer_alpha_kick",
    )
    service.set_item_outcome(session.id, session.items[0].item_id, ReviewOutcome.CORRECT)
    history_snapshot = service.build_snapshot(
        session.id,
        outcome="pending",
        song_ref=refs["alpha_song_ref"],
        layer_ref="layer:layer_alpha_kick",
        item_id=session.items[0].item_id,
    )

    assert {item["value"] for item in alpha_snapshot["session"]["scopeOptions"]["songs"]} == {
        refs["alpha_song_ref"],
        refs["bravo_song_ref"],
    }
    assert {item["value"] for item in alpha_snapshot["session"]["scopeOptions"]["layers"]} == {
        "layer:layer_alpha_kick",
        "layer:layer_alpha_snare",
    }
    assert alpha_snapshot["filteredCount"] == 3
    assert layer_snapshot["filteredCount"] == 2
    assert layer_snapshot["currentItem"]["songRef"] == refs["alpha_song_ref"]
    assert layer_snapshot["currentItem"]["songTitle"] == "Alpha Song"
    assert layer_snapshot["currentItem"]["layerRef"] == "layer:layer_alpha_kick"
    assert layer_snapshot["currentItem"]["layerName"] == "kick"
    assert len(layer_snapshot["progress"]["items"]) == 2
    assert history_snapshot["currentItem"]["itemId"] == session.items[0].item_id
    assert history_snapshot["currentItem"]["reviewOutcome"] == "correct"
    assert history_snapshot["progress"]["items"][0]["isCurrent"] is True
    assert history_snapshot["navigation"]["viewMode"] == "history"
    assert history_snapshot["navigation"]["focusedItemVisible"] is False
    assert history_snapshot["navigation"]["currentScopeItemNumber"] == 1
    assert history_snapshot["navigation"]["scopeCount"] == 2
    assert history_snapshot["sessions"]["items"][0]["context"]["projectName"] == "Arcade Project"


def test_project_review_snapshot_reuses_cached_materialization_between_navigation_requests(
    tmp_path: Path,
):
    project_root = _build_fake_project_root(tmp_path)
    builder = _FakeProjectReviewQueueBuilder(project_root)
    service = ReviewSessionService(
        tmp_path,
        signal_service=_FakeReviewSignalService(),
        project_queue_builder=builder,
    )

    session = service.create_project_session(project_root, name="Project Queue")
    builder.build_calls = 0

    first_snapshot = service.build_snapshot(session.id, outcome="pending")
    second_snapshot = service.build_snapshot(session.id, outcome="pending", cursor=0)

    assert first_snapshot["currentItem"]["itemId"] == "ri_fixture_kick"
    assert second_snapshot["currentItem"]["itemId"] == "ri_fixture_kick"
    assert builder.build_calls == 0


def test_project_review_snapshot_uses_cached_session_after_phone_review_save(
    tmp_path: Path,
):
    project_root = _build_fake_project_root(tmp_path)
    builder = _FakeProjectReviewQueueBuilder(project_root)
    signal_service = _FakeReviewSignalService(touch_path=project_root / "project.db")
    service = ReviewSessionService(
        tmp_path,
        signal_service=signal_service,
        project_queue_builder=builder,
    )

    session = service.create_project_session(project_root, name="Project Queue")
    builder.build_calls = 0

    updated = service.set_item_review(
        session.id,
        "ri_fixture_kick",
        outcome=ReviewOutcome.CORRECT,
        corrected_label=None,
        review_note=None,
    )
    snapshot = service.build_snapshot(session.id, outcome="all")

    assert updated.items[0].review_outcome == ReviewOutcome.CORRECT
    assert snapshot["currentItem"]["reviewOutcome"] == "correct"
    assert signal_service.calls == 1
    assert builder.build_calls == 0


def test_project_review_snapshot_rebuilds_after_structural_review_save(tmp_path: Path):
    project_root = _build_fake_project_root(tmp_path)
    builder = _FakeProjectReviewQueueBuilder(project_root)
    service = ReviewSessionService(
        tmp_path,
        signal_service=_FakeReviewSignalService(),
        project_queue_builder=builder,
    )

    session = service.create_project_session(project_root, name="Project Queue")
    builder.build_calls = 0

    service.set_item_review(
        session.id,
        "ri_fixture_kick",
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="kick",
        review_note="retimed in timeline review",
        decision_kind=ReviewDecisionKind.BOUNDARY_CORRECTED,
        original_start_ms=120.0,
        original_end_ms=170.0,
        corrected_start_ms=118.0,
        corrected_end_ms=166.0,
        surface="timeline_fix_mode",
        workflow="timeline_event_edit",
        operator_action="retime_event",
    )
    snapshot = service.build_snapshot(session.id, outcome="all")

    assert snapshot["currentItem"]["reviewDecision"]["kind"] == "boundary_corrected"
    assert builder.build_calls == 1


def test_review_server_controller_requires_explicit_enable(tmp_path: Path):
    controller = ReviewServerController(port=0)

    with pytest.raises(ValueError, match="Phone review service is disabled"):
        controller.build_session_url(tmp_path, "rev_demo")

    controller.enable()
    url = controller.build_session_url(tmp_path, "rev_demo")

    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/")
    assert controller.is_enabled is True

    controller.disable()
    assert controller.is_enabled is False


def test_review_session_index_includes_current_project_context(tmp_path: Path):
    storage = ProjectStorage.create_new(
        name="Olivia Scratch 3",
        working_dir_root=tmp_path / "working",
    )
    service = ReviewSessionService(storage.working_dir)

    try:
        payload = service.build_session_index()

        assert payload["defaultSessionId"] is None
        assert payload["items"] == []
        assert payload["project"] == {
            "projectRef": f"project:{storage.project.id}",
            "projectName": "Olivia Scratch 3",
            "projectRoot": str(storage.working_dir.resolve()),
        }
    finally:
        storage.close()


def test_review_session_index_prefers_live_application_session_context(tmp_path: Path):
    storage = ProjectStorage.create_new(
        name="Olivia Scratch 3",
        working_dir_root=tmp_path / "working",
    )
    service = ReviewSessionService(storage.working_dir)

    try:
        payload = service.build_session_index(
            application_session={
                "sessionId": "sess_live",
                "projectName": "Runtime Project",
                "projectRef": "project:runtime_live",
                "activeSongRef": "song:alpha",
            }
        )

        assert payload["defaultSessionId"] is None
        assert payload["project"] == {
            "projectRef": "project:runtime_live",
            "projectName": "Runtime Project",
            "projectRoot": str(storage.working_dir.resolve()),
            "applicationSession": {
                "sessionId": "sess_live",
                "projectName": "Runtime Project",
                "projectRef": "project:runtime_live",
                "activeSongRef": "song:alpha",
            },
        }
    finally:
        storage.close()


def test_review_server_serves_html_api_audio_and_review_updates(tmp_path: Path):
    review_items_path = _write_review_items_json(tmp_path)
    service = ReviewSessionService(tmp_path)
    session = service.import_session_file(review_items_path, name="Phone Queue")
    other = service.import_session_file(review_items_path, name="Batch B")
    server = create_review_http_server(tmp_path, session.id, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        html = _read_text(f"{base_url}/")
        sessions_payload = _read_json(f"{base_url}/api/sessions")
        snapshot = _read_json(f"{base_url}/api/session?sessionId={session.id}&outcome=pending")
        shifted = _read_json(f"{base_url}/api/session?sessionId={session.id}&outcome=pending&cursor=1")
        scoped = _read_json(
            f"{base_url}/api/session?sessionId={session.id}&outcome=pending&songRef=song:arcade&layerRef=layer:snare"
        )
        audio_bytes = _read_bytes(f"{base_url}{snapshot['currentItem']['audioUrl']}")
        audio_range = _read_bytes(
            f"{base_url}{snapshot['currentItem']['audioUrl']}",
            headers={"Range": "bytes=0-7"},
        )
        updated = _post_json(
            f"{base_url}/api/review?sessionId={session.id}&outcome=pending&targetClass=all",
            {"itemId": snapshot["currentItem"]["itemId"], "outcome": "correct"},
        )
        switched = _read_json(f"{base_url}/api/session?sessionId={other.id}&outcome=pending")
        relabeled = _post_json(
            f"{base_url}/api/review?sessionId={other.id}&outcome=pending&targetClass=all",
            {
                "itemId": switched["currentItem"]["itemId"],
                "outcome": "incorrect",
                "correctedLabel": "tom",
                "reviewNote": "felt more like a tom hit",
            },
        )
        focused = _read_json(
            f"{base_url}/api/session?sessionId={session.id}&outcome=pending&itemId={session.items[0].item_id}"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    persisted = ReviewSessionRepository(tmp_path).get(session.id)
    other_persisted = ReviewSessionRepository(tmp_path).get(other.id)
    assert "EZ Review" in html
    assert "Reclassify" in html
    assert "Back" in html
    assert "Forward" in html
    assert "progress-boxes" in html
    assert 'id="reload-status-btn"' in html
    assert "overflow-x: auto;" in html
    assert "scrollIntoView({ block: 'nearest', inline: 'center' })" in html
    assert "waveform" in html
    assert "audioClipCache: new Map()" in html
    assert "prefetchNeighborMedia(state.snapshot?.navigation)" in html
    assert "fetch(item.audioUrl, { cache: 'no-store' })" not in html
    assert 'class="title-meta"' in html
    assert 'class="replay-button" id="play-button"' in html
    assert 'class="nav-row"' in html
    assert "flex-direction: column;" in html
    assert "overflow-y: auto;" in html
    assert '<h2 class="prediction">' not in html
    assert 'class="hero-label"' in html
    assert 'class="hero panel-inset"' not in html
    assert "grid-template-rows: auto auto auto auto;" in html
    assert "grid-template-columns: repeat(5, minmax(0, 1fr));" not in html
    assert ".bottom-bar .action.nav" not in html
    assert "const initialQuery = new URLSearchParams(window.location.search);" in html
    assert "explicitSessionQuery: initialQuery.has('sessionId')" in html
    assert "function currentSearchParams()" in html
    assert "nextUrl.searchParams.delete('sessionId')" in html
    assert "const REFRESH_INTERVAL_MS = 4000;" in html
    assert "scheduleAutoRefresh();" in html
    assert "manualReloadStatus" in html
    assert "stateRevision" in html
    assert "refreshSessionState().catch(() => undefined);" in html
    assert "function renderIdleProjectState()" in html
    assert "if (!selectedSessionId()) {" in html
    assert "renderIdleProjectState();" in html
    assert "currentScopeItemNumber" in html
    assert "document.addEventListener('visibilitychange'" in html
    assert "Song" in html
    assert "Layer" in html
    assert "focusItemFromCurrentView(button.dataset.progressItem)" in html
    load_sessions_block = html.split("async function loadSessions() {", 1)[1].split(
        "async function requestSnapshot", 1
    )[0]
    load_live_cursor_block = html.split("async function loadLiveCursor(cursor = 0) {", 1)[1].split(
        "async function loadFocusedItem(itemId) {", 1
    )[0]
    post_review_block = html.split("async function postReview(outcome, extra = {}) {", 1)[1].split(
        "async function focusItemFromCurrentView(itemId) {", 1
    )[0]
    apply_navigation_block = html.split("async function applyNavigationTransition(run, { historyStack = null, clearForward = false } = {}) {", 1)[1].split(
        "function isMissingHistoryItemError(error) {", 1
    )[0]
    focus_item_block = html.split("async function focusItemFromCurrentView(itemId) {", 1)[1].split(
        "async function navigateBack() {", 1
    )[0]
    navigate_back_block = html.split("async function navigateBack() {", 1)[1].split(
        "async function navigateForward() {", 1
    )[0]
    navigate_forward_block = html.split("async function navigateForward() {", 1)[1].split(
        "function renderSessionSelect()", 1
    )[0]
    refresh_state_block = html.split(
        "async function refreshSessionState({ forceRender = false, resetCursor = false } = {}) {",
        1,
    )[1].split("async function manualReloadStatus() {", 1)[0]
    manual_reload_block = html.split("async function manualReloadStatus() {", 1)[1].split(
        "function scheduleAutoRefresh()", 1
    )[0]
    assert "resetReviewState();" in load_sessions_block
    assert "state.outcome = 'pending';" in load_sessions_block
    assert "state.songRef = 'all';" in load_sessions_block
    assert "state.layerRef = 'all';" in load_sessions_block
    assert "state.targetClass = 'all';" in load_sessions_block
    assert "const nextCursor = Math.max(0, Number(cursor) || 0);" in load_live_cursor_block
    assert "requestSnapshot({ cursor: nextCursor })" in load_live_cursor_block
    assert "syncNavigationButtons(state.snapshot?.navigation);" in apply_navigation_block
    assert "await applyNavigationTransition(async () => {" in post_review_block
    assert "{ historyStack: state.historyBack, clearForward: true }" in post_review_block
    assert "await applyNavigationTransition(" in focus_item_block
    assert "{ historyStack: state.historyBack, clearForward: true }" in focus_item_block
    assert "while (state.historyBack.length > 0)" in navigate_back_block
    assert "isMissingHistoryItemError(error)" in navigate_back_block
    assert "{ historyStack: state.historyForward }" in navigate_back_block
    assert "() => loadLiveCursor(previousCursor)" in navigate_back_block
    assert "while (state.historyForward.length > 0)" in navigate_forward_block
    assert "isMissingHistoryItemError(error)" in navigate_forward_block
    assert "{ historyStack: state.historyBack }" in navigate_forward_block
    assert "{ historyStack: state.historyBack, clearForward: true }" in navigate_forward_block
    assert "() => loadLiveCursor(nextCursor)" in navigate_forward_block
    assert "const focusedHistoryItemId = !shouldResetCursor && navigation.viewMode === 'history'" in refresh_state_block
    assert "? { itemId: focusedHistoryItemId }" in refresh_state_block
    assert ": { cursor: shouldResetCursor ? 0 : state.cursor }" in refresh_state_block
    assert "clearMediaCaches();" in manual_reload_block
    assert "state.currentAudioUrl = '';" in manual_reload_block
    assert "await refreshSessionState({ forceRender: true });" in manual_reload_block
    assert sessions_payload["defaultSessionId"] == session.id
    assert sessions_payload["stateRevision"] == 0
    assert len(sessions_payload["items"]) == 2
    assert snapshot["currentItem"]["targetClass"] == "kick"
    assert snapshot["stateRevision"] == 0
    assert shifted["currentItem"]["itemId"] == session.items[1].item_id
    assert shifted["navigation"]["hasPrevious"] is True
    assert scoped["filteredCount"] == 1
    assert scoped["currentItem"]["layerRef"] == "layer:snare"
    assert len(audio_bytes["body"]) > 32
    assert audio_range["status"] == 206
    assert audio_range["headers"]["Content-Range"].startswith("bytes 0-7/")
    assert len(audio_range["body"]) == 8
    assert updated["session"]["countsByOutcome"]["correct"] == 1
    assert updated["navigation"]["cursor"] == 0
    assert updated["navigation"]["currentScopeItemNumber"] == 2
    assert updated["navigation"]["scopeCount"] == 2
    assert updated["progress"]["items"][0]["reviewOutcome"] == "correct"
    assert updated["progress"]["items"][1]["isCurrent"] is True
    assert focused["currentItem"]["itemId"] == session.items[0].item_id
    assert focused["currentItem"]["reviewOutcome"] == "correct"
    assert focused["navigation"]["viewMode"] == "history"
    assert persisted is not None
    assert persisted.items[0].review_outcome == ReviewOutcome.CORRECT
    assert relabeled["session"]["countsByOutcome"]["incorrect"] == 1
    assert other_persisted is not None
    assert other_persisted.items[0].corrected_label == "tom"
    assert other_persisted.items[0].review_note == "felt more like a tom hit"
    assert other_persisted.items[0].review_decision is not None
    assert other_persisted.items[0].review_decision.kind == ReviewDecisionKind.RELABELED


def test_review_server_sessions_api_exposes_live_project_context_without_active_session(tmp_path: Path):
    storage = ProjectStorage.create_new(
        name="Idle Runtime Project",
        working_dir_root=tmp_path / "working",
    )
    server = create_review_http_server(
        storage.working_dir,
        None,
        host="127.0.0.1",
        port=0,
        application_session={
            "sessionId": "sess_idle",
            "projectName": "Idle Runtime Project",
            "projectRef": f"project:{storage.project.id}",
            "activeSongRef": "song:alpha",
        },
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        payload = _read_json(f"{base_url}/api/sessions")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        storage.close()

    assert payload["defaultSessionId"] is None
    assert payload["items"] == []
    assert payload["project"] == {
        "projectRef": f"project:{storage.project.id}",
        "projectName": "Idle Runtime Project",
        "projectRoot": str(storage.working_dir.resolve()),
        "applicationSession": {
            "sessionId": "sess_idle",
            "projectName": "Idle Runtime Project",
            "projectRef": f"project:{storage.project.id}",
            "activeSongRef": "song:alpha",
        },
    }


def test_review_server_returns_404_for_missing_audio(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    missing_audio = samples / "kick" / "missing.wav"
    payload = [
        {
            "item_id": "ri_missing",
            "audio_path": str(missing_audio),
            "predicted_label": "kick",
            "target_class": "kick",
            "polarity": "positive",
            "score": 0.9,
        }
    ]
    items_path = tmp_path / "review_items.json"
    items_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session = ReviewSessionService(tmp_path).import_session_file(items_path)
    server = create_review_http_server(tmp_path, session.id, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with pytest.raises(HTTPError) as exc_info:
            _read_bytes(f"{base_url}/audio/ri_missing")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert exc_info.value.code == 404


def _mark_project_event_review_state(
    working_dir: Path,
    *,
    layer_id: str,
    event_id: str,
    promotion_state: str,
    review_state: str,
    review_outcome: ReviewOutcome,
    decision_kind: ReviewDecisionKind,
    original_label: str | None,
    corrected_label: str | None,
    review_note: str | None,
) -> None:
    with ProjectStorage.open_db(working_dir) as project:
        take = project.takes.get_main(layer_id)
        assert take is not None
        assert isinstance(take.data, EventData)
        updated_layers: list[DomainLayer] = []
        for layer in take.data.layers:
            updated_events = []
            for event in layer.events:
                if event.id != event_id:
                    updated_events.append(event)
                    continue
                updated_events.append(
                    replace(
                        event,
                        metadata=updated_review_metadata(
                            event.metadata,
                            promotion_state=promotion_state,
                            review_state=review_state,
                            review_outcome=review_outcome,
                            decision_kind=decision_kind,
                            original_label=original_label,
                            corrected_label=corrected_label,
                            review_note=review_note,
                            reviewed_at="2026-04-26T18:00:00+00:00",
                            original_start_ms=float(event.time) * 1000.0,
                            original_end_ms=float(event.time + event.duration) * 1000.0,
                            corrected_start_ms=None,
                            corrected_end_ms=None,
                            created_event_ref=None,
                            surface="timeline_fix_mode",
                            workflow="timeline_event_review",
                            operator_action="reject_event",
                        ),
                    )
                )
            updated_layers.append(replace(layer, events=tuple(updated_events)))
        project.takes.update(replace(take, data=EventData(layers=tuple(updated_layers))))
        project.save()


def _build_fake_project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "fake_project"
    project_root.mkdir()
    (project_root / "project.db").write_text("project-db", encoding="utf-8")
    return project_root


class _FakeProjectReviewQueueBuilder:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self.build_calls = 0
        self._audio_path = project_root / "kick.wav"
        self._audio_path.write_bytes(b"RIFFfake")

    def build_queue(
        self,
        project_path: str | Path,
        *,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ) -> ProjectReviewQueue:
        del project_path, song_id, song_version_id, layer_id, questionable_score_threshold, item_limit
        self.build_calls += 1
        item = ReviewItem(
            item_id="ri_fixture_kick",
            audio_path=str(self._audio_path.resolve()),
            predicted_label="kick",
            target_class="kick",
            polarity=polarity,
            score=0.92,
            source_provenance={
                "project_ref": "project:fixture",
                "project_name": "Arcade Project",
                "song_ref": "song:arcade",
                "song_title": "Arcade",
                "version_ref": "version:main",
                "version_label": "Main",
                "layer_ref": "layer:kick",
                "layer_name": "kick",
                "event_ref": "event:k1",
                "audio_ref": str(self._audio_path.resolve()),
                "source_audio_ref": str(self._audio_path.resolve()),
                "current_start_ms": 120.0,
                "current_end_ms": 170.0,
            },
        )
        return ProjectReviewQueue(
            project_name="Arcade Project",
            project_ref="project:fixture",
            source_ref=str(self._project_root.resolve()),
            items=[item],
            metadata={
                "import_format": "project",
                "queue_source_kind": "ez_project",
                "project_ref": "project:fixture",
                "project_name": "Arcade Project",
                "review_mode": review_mode or "all_events",
                "selected_item_count": 1,
                "total_item_count": 1,
            },
        )


class _FakeReviewSignalService:
    def __init__(self, *, touch_path: Path | None = None) -> None:
        self._touch_path = touch_path
        self.calls = 0

    def record_session_item_review(self, session, item) -> None:
        del session, item
        self.calls += 1
        if self._touch_path is not None:
            self._touch_path.write_text(f"project-db-{self.calls}", encoding="utf-8")


def _write_review_items_json(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    payload = [
        {
            "item_id": "ri_kick",
            "audio_path": str(samples / "kick" / "k1.wav"),
            "predicted_label": "kick",
            "target_class": "kick",
            "polarity": "positive",
            "score": 0.97,
            "source_provenance": {
                "kind": "fixture",
                "source": "kick_01",
                "project_ref": "project:fixture",
                "song_ref": "song:arcade",
                "version_ref": "version:main",
                "layer_ref": "layer:kick",
                "event_ref": "event:kick-01",
                "source_event_ref": "event:kick-source-01",
                "model_ref": "bundle:fixture-v1",
            },
        },
        {
            "item_id": "ri_snare",
            "audio_path": str(samples / "snare" / "s1.wav"),
            "predicted_label": "kick",
            "target_class": "snare",
            "polarity": "negative",
            "confidence": 0.12,
            "source_provenance": {
                "kind": "fixture",
                "source": "snare_01",
                "project_ref": "project:fixture",
                "song_ref": "song:arcade",
                "version_ref": "version:main",
                "layer_ref": "layer:snare",
                "event_ref": "event:snare-01",
                "source_event_ref": "event:snare-source-01",
                "model_ref": "bundle:fixture-v1",
            },
        },
    ]
    path = tmp_path / "review_items.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_review_items_jsonl(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    rows = [
        {
            "audioPath": str(samples / "kick" / "k1.wav"),
            "predictedLabel": "kick",
            "targetClass": "kick",
            "polarity": "positive",
            "score": 0.93,
        },
        {
            "audioPath": str(samples / "snare" / "s1.wav"),
            "predictedLabel": "kick",
            "targetClass": "snare",
            "polarity": "negative",
            "score": 0.08,
        },
    ]
    path = tmp_path / "review_items.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _read_text(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _read_bytes(url: str, *, headers: dict[str, str] | None = None) -> dict[str, object]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=5) as response:
        return {
            "status": response.status,
            "headers": dict(response.headers.items()),
            "body": response.read(),
        }


def _read_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, object]) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
