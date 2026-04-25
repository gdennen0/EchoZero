from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from echozero.foundry.domain import ReviewDecisionKind, ReviewOutcome
from echozero.foundry.persistence import ReviewSessionRepository
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.foundry.review_server import create_review_http_server
from echozero.foundry.services.review_session_service import ReviewSessionService
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
    assert updated.items[0].review_decision is not None
    assert updated.items[0].review_decision.kind == ReviewDecisionKind.REJECTED
    assert refreshed["currentItem"]["reviewDecision"]["kind"] == "rejected"
    assert refreshed["currentItem"]["reviewDecision"]["trainingEligibility"]["allowsNegativeSignal"] is True


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
            "activeSongId": refs["alpha_song_id"],
            "activeSongVersionId": "ver_alpha",
        },
    )
    snapshot = service.build_snapshot(session.id, outcome="all")

    assert session.metadata["review_mode"] == "all_events"
    assert session.metadata["application_session"]["sessionId"] == "sess_123"
    assert snapshot["session"]["reviewMode"] == "all_events"
    assert snapshot["session"]["applicationSession"] == {
        "sessionId": "sess_123",
        "projectId": "proj_456",
        "activeSongId": refs["alpha_song_id"],
        "activeSongVersionId": "ver_alpha",
    }


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
    assert history_snapshot["currentItem"]["itemId"] == session.items[0].item_id
    assert history_snapshot["currentItem"]["reviewOutcome"] == "correct"
    assert history_snapshot["navigation"]["viewMode"] == "history"
    assert history_snapshot["navigation"]["focusedItemVisible"] is False


def test_review_server_controller_requires_explicit_enable(tmp_path: Path):
    controller = ReviewServerController()

    with pytest.raises(ValueError, match="Phone review service is disabled"):
        controller.build_session_url(tmp_path, "rev_demo")

    controller.enable()
    url = controller.build_session_url(tmp_path, "rev_demo")

    assert "sessionId=rev_demo" in url
    assert controller.is_enabled is True

    controller.disable()
    assert controller.is_enabled is False


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
    assert "Prev" in html
    assert "Next" in html
    assert "Song" in html
    assert "Layer" in html
    assert sessions_payload["defaultSessionId"] == session.id
    assert len(sessions_payload["items"]) == 2
    assert snapshot["currentItem"]["targetClass"] == "kick"
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
