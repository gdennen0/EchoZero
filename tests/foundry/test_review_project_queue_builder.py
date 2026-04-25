"""Focused proof for project-backed Foundry review queue generation.
Exists because review sessions must build from canonical EZ project data without import-side truth drift.
Connects persisted project archives or working dirs to deterministic Foundry review sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from echozero.domain.types import Event as DomainEvent
from echozero.domain.types import EventData, Layer as DomainLayer
from echozero.foundry.services.project_review_queue_builder import ProjectReviewQueueBuilder
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.persistence.entities import LayerRecord
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take, TakeSource
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_project_review_queue_builder_reads_archive_with_deterministic_order(tmp_path: Path):
    ez_path, _working_dir, refs = _build_project_review_fixture(tmp_path)
    builder = ProjectReviewQueueBuilder(tmp_path)

    first_queue = builder.build_queue(ez_path)
    second_queue = builder.build_queue(ez_path)

    assert [item.item_id for item in first_queue.items] == [
        item.item_id for item in second_queue.items
    ]
    assert [item.audio_path for item in first_queue.items] == [
        item.audio_path for item in second_queue.items
    ]
    assert [item.target_class for item in first_queue.items] == ["kick", "kick", "snare", "kick"]
    assert [item.predicted_label for item in first_queue.items] == ["kick", "kick", "snare", "kick"]
    assert [item.source_provenance["event_ref"] for item in first_queue.items] == [
        "event:evt_alpha_kick_01",
        "event:evt_alpha_kick_02",
        "event:evt_alpha_snare_01",
        "event:evt_bravo_kick_01",
    ]
    assert first_queue.items[0].source_provenance["project_ref"] == refs["project_ref"]
    assert first_queue.items[0].source_provenance["song_ref"] == refs["alpha_song_ref"]
    assert first_queue.items[0].source_provenance["version_ref"] == refs["alpha_version_ref"]
    assert first_queue.items[0].source_provenance["layer_ref"] == "layer:layer_alpha_kick"
    assert first_queue.items[0].source_provenance["source_event_ref"] == "event:src_alpha_kick_01"
    assert first_queue.items[0].source_provenance["model_ref"] == "bundle-kick-v1"
    assert first_queue.items[0].source_provenance["audio_ref"] == first_queue.items[0].audio_path
    assert Path(first_queue.items[0].source_provenance["source_audio_ref"]).exists()
    assert (
        Path(first_queue.items[0].source_provenance["source_audio_ref"]).name
        == Path(refs["alpha_source_audio_ref"]).name
    )
    assert first_queue.items[0].score == 0.97
    assert first_queue.items[0].audio_path.startswith(
        str(tmp_path / "foundry" / "cache" / "review_projects")
    )
    assert Path(first_queue.items[0].audio_path).parent.name == "clips"
    assert Path(first_queue.items[0].audio_path).exists()
    assert 0.045 <= sf.info(first_queue.items[0].audio_path).duration <= 0.055
    assert (
        sf.info(first_queue.items[0].source_provenance["source_audio_ref"]).frames
        > sf.info(first_queue.items[0].audio_path).frames
    )
    assert first_queue.metadata["import_format"] == "project"
    assert first_queue.metadata["review_mode"] == "all_events"
    assert first_queue.metadata["song_ids"] == [refs["alpha_song_id"], refs["bravo_song_id"]]
    assert first_queue.metadata["skipped_missing_audio_count"] == 0
    assert first_queue.metadata["skipped_unmaterialized_clip_count"] == 0


def test_review_session_service_creates_project_backed_session_from_working_dir(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Alpha Kick Review",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    snapshot = service.build_snapshot(session.id, outcome="all")

    assert session.name == "Alpha Kick Review"
    assert session.source_ref == str(working_dir.resolve())
    assert len(session.items) == 2
    assert session.class_map == ["kick"]
    assert session.metadata["import_format"] == "project"
    assert session.metadata["song_ids"] == [refs["alpha_song_id"]]
    assert session.metadata["layer_ids"] == ["layer_alpha_kick"]
    assert snapshot["session"]["totalItems"] == 2
    assert snapshot["currentItem"]["targetClass"] == "kick"
    assert snapshot["currentItem"]["sourceProvenance"]["song_ref"] == refs["alpha_song_ref"]
    assert snapshot["currentItem"]["sourceProvenance"]["version_ref"] == refs["alpha_version_ref"]
    assert snapshot["currentItem"]["sourceProvenance"]["layer_ref"] == "layer:layer_alpha_kick"
    assert snapshot["currentItem"]["sourceProvenance"]["audio_ref"] == snapshot["currentItem"]["audioPath"]
    assert snapshot["currentItem"]["sourceProvenance"]["source_audio_ref"] == refs["alpha_source_audio_ref"]
    assert snapshot["currentItem"]["sourceProvenance"]["audio_ref"].endswith(".wav")
    assert snapshot["currentItem"]["sourceProvenance"]["audio_ref"] != refs["alpha_source_audio_ref"]
    assert 0.045 <= sf.info(snapshot["currentItem"]["audioPath"]).duration <= 0.055


def test_project_review_queue_builder_invalidates_clip_cache_when_source_audio_changes(
    tmp_path: Path,
):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    builder = ProjectReviewQueueBuilder(tmp_path)

    first_queue = builder.build_queue(
        working_dir,
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    _rewrite_review_source_audio(Path(refs["alpha_source_audio_ref"]), duration_seconds=4.0)
    second_queue = builder.build_queue(
        working_dir,
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )

    assert first_queue.items[0].audio_path != second_queue.items[0].audio_path
    assert Path(second_queue.items[0].audio_path).exists()


def test_project_review_queue_builder_filters_questionable_items_by_score_and_limit(
    tmp_path: Path,
):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    builder = ProjectReviewQueueBuilder(tmp_path)

    queue = builder.build_queue(
        working_dir,
        song_id=refs["alpha_song_id"],
        review_mode="questionables",
        questionable_score_threshold=0.9,
        item_limit=2,
    )

    assert [item.score for item in queue.items] == [0.72, 0.88]
    assert [item.target_class for item in queue.items] == ["snare", "kick"]
    assert queue.metadata["total_item_count"] == 3
    assert queue.metadata["selected_item_count"] == 2
    assert queue.metadata["review_mode"] == "questionables"
    assert queue.metadata["questionable_score_threshold"] == 0.9
    assert queue.metadata["item_limit"] == 2


def _build_project_review_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=2)
    source_audio_root = tmp_path / "source_audio"
    alpha_source = _write_review_source_audio(
        samples / "kick" / "k1.wav",
        source_audio_root / "alpha_song.wav",
        duration_seconds=3.0,
    )
    bravo_source = _write_review_source_audio(
        samples / "snare" / "s1.wav",
        source_audio_root / "bravo_song.wav",
        duration_seconds=3.0,
    )
    working_root = tmp_path / "project_work"
    session = ProjectStorage.create_new("Arcade Project", working_dir_root=working_root)
    ez_path = tmp_path / "arcade_review.ez"
    try:
        alpha_song, alpha_version = session.import_song(
            "Alpha Song",
            alpha_source,
            default_templates=[],
        )
        bravo_song, bravo_version = session.import_song(
            "Bravo Song",
            bravo_source,
            default_templates=[],
        )
        _create_analysis_layer(
            session,
            version_id=alpha_version.id,
            layer_id="layer_alpha_kick",
            name="kick",
            order=0,
            source_audio_path=alpha_version.audio_file,
            run_id="run-alpha-kick",
            events=(
                _event(
                    "evt_alpha_kick_02",
                    time=2.0,
                    duration=0.05,
                    predicted_label="kick",
                    score=0.88,
                    source_event_id="src_alpha_kick_02",
                    bundle_id="bundle-kick-v1",
                ),
                _event(
                    "evt_alpha_kick_01",
                    time=1.0,
                    duration=0.05,
                    predicted_label="kick",
                    score=0.97,
                    source_event_id="src_alpha_kick_01",
                    bundle_id="bundle-kick-v1",
                ),
            ),
        )
        _create_analysis_layer(
            session,
            version_id=alpha_version.id,
            layer_id="layer_alpha_snare",
            name="snare",
            order=1,
            source_audio_path=alpha_version.audio_file,
            run_id="run-alpha-snare",
            events=(
                _event(
                    "evt_alpha_snare_01",
                    time=1.5,
                    duration=0.06,
                    predicted_label="snare",
                    score=0.72,
                    source_event_id="src_alpha_snare_01",
                    bundle_id="bundle-snare-v1",
                ),
            ),
        )
        _create_manual_layer(
            session,
            version_id=alpha_version.id,
            layer_id="layer_alpha_manual",
            name="manual fixes",
            order=2,
            source_audio_path=alpha_version.audio_file,
            event_id="evt_alpha_manual_01",
        )
        _create_analysis_layer(
            session,
            version_id=bravo_version.id,
            layer_id="layer_bravo_kick",
            name="kick",
            order=0,
            source_audio_path=bravo_version.audio_file,
            run_id="run-bravo-kick",
            events=(
                _event(
                    "evt_bravo_kick_01",
                    time=0.5,
                    duration=0.05,
                    predicted_label="kick",
                    score=0.67,
                    source_event_id="src_bravo_kick_01",
                    bundle_id="bundle-kick-v2",
                ),
            ),
        )
        session.commit()
        session.save_as(ez_path)
        refs = {
            "project_ref": f"project:{session.project.id}",
            "alpha_song_id": alpha_song.id,
            "alpha_song_ref": f"song:{alpha_song.id}",
            "alpha_version_ref": f"version:{alpha_version.id}",
            "alpha_source_audio_ref": str((session.working_dir / alpha_version.audio_file).resolve()),
            "bravo_song_id": bravo_song.id,
            "bravo_song_ref": f"song:{bravo_song.id}",
        }
        return ez_path, session.working_dir, refs
    finally:
        session.close()


def _write_review_source_audio(
    source_path: Path,
    destination_path: Path,
    *,
    duration_seconds: float,
) -> Path:
    audio, sample_rate = sf.read(str(source_path), dtype="float32", always_2d=False)
    target_frames = max(1, int(round(duration_seconds * sample_rate)))
    if getattr(audio, "ndim", 0) == 1:
        repeated = np.tile(audio, max(1, int(np.ceil(target_frames / audio.shape[0]))))[:target_frames]
    else:
        repeated = np.tile(
            audio,
            (max(1, int(np.ceil(target_frames / audio.shape[0]))), 1),
        )[:target_frames, :]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination_path, repeated, sample_rate)
    return destination_path


def _rewrite_review_source_audio(path: Path, *, duration_seconds: float) -> None:
    info = sf.info(str(path))
    target_frames = max(1, int(round(duration_seconds * info.samplerate)))
    rewritten = np.linspace(-0.35, 0.35, target_frames, dtype=np.float32)
    sf.write(path, rewritten, info.samplerate)


def _create_analysis_layer(
    session: ProjectStorage,
    *,
    version_id: str,
    layer_id: str,
    name: str,
    order: int,
    source_audio_path: str,
    run_id: str,
    events: tuple[DomainEvent, ...],
) -> None:
    now = datetime.now(timezone.utc)
    layer = LayerRecord(
        id=layer_id,
        song_version_id=version_id,
        name=name,
        layer_type="analysis",
        color=None,
        order=order,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={"pipeline_id": "extract_song_drum_events"},
        created_at=now,
        provenance={
            "analysis_build": {
                "build_id": f"build-{layer_id}",
                "execution_id": run_id,
                "pipeline_id": "extract_song_drum_events",
            }
        },
    )
    take = Take.create(
        data=EventData(
            layers=(DomainLayer(id=f"domain_{layer_id}", name=name, events=events),)
        ),
        label="Main",
        source=TakeSource(
            block_id="binary_drum_classify",
            block_type="BinaryDrumClassify",
            settings_snapshot={"source_audio_path": source_audio_path},
            run_id=run_id,
        ),
        is_main=True,
    )
    session.layers.create(layer)
    session.takes.create(layer_id, take)


def _create_manual_layer(
    session: ProjectStorage,
    *,
    version_id: str,
    layer_id: str,
    name: str,
    order: int,
    source_audio_path: str,
    event_id: str,
) -> None:
    layer = LayerRecord(
        id=layer_id,
        song_version_id=version_id,
        name=name,
        layer_type="manual",
        color=None,
        order=order,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
    )
    event = _event(
        event_id,
        time=3.0,
        duration=0.04,
        predicted_label="manual",
        score=1.0,
        source_event_id=event_id,
        bundle_id="bundle-manual",
    )
    take = Take.create(
        data=EventData(layers=(DomainLayer(id=f"domain_{layer_id}", name=name, events=(event,)),)),
        label="Main",
        source=TakeSource(
            block_id="manual",
            block_type="ManualEdit",
            settings_snapshot={"source_audio_path": source_audio_path},
            run_id="manual-edit",
        ),
        is_main=True,
    )
    session.layers.create(layer)
    session.takes.create(layer_id, take)


def _event(
    event_id: str,
    *,
    time: float,
    duration: float,
    predicted_label: str,
    score: float,
    source_event_id: str,
    bundle_id: str,
) -> DomainEvent:
    model_artifact = {"artifactIdentity": {"artifactId": bundle_id}}
    return DomainEvent(
        id=event_id,
        time=time,
        duration=duration,
        classifications={
            "class": predicted_label,
            "confidence": score,
            "model_artifact": model_artifact,
        },
        metadata={"model_artifact": model_artifact},
        origin="analysis",
        source_event_id=source_event_id,
    )
