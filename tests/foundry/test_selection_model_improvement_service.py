"""Focused proof for selection-driven candidate model training orchestration.
Exists because EZ's improve-model flow must stay bounded, deterministic, and review-signal correct.
Connects selected review signals to binary dataset creation, base-model comparison, and candidate runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from echozero.foundry.domain import (
    CompatibilityReport,
    Dataset,
    DatasetSample,
    DatasetVersion,
    ModelArtifact,
    TrainRun,
    TrainRunStatus,
)
from echozero.foundry.domain.review import (
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSignal,
    build_review_decision,
)
from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
    SelectionModelImprovementService,
)
from echozero.models.runtime_bundle_selection import InstalledRuntimeBundle


def test_selection_model_improvement_service_trains_candidate_from_selected_signals(
    monkeypatch,
    tmp_path: Path,
) -> None:
    current_artifact = ModelArtifact(
        id="art_current",
        run_id="run_current",
        artifact_version="v1",
        path=(tmp_path / "current.manifest.json"),
        sha256="sha-current",
        manifest={"classes": ["kick", "other"], "sourceManifestPath": str(tmp_path / "installed.manifest.json")},
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    local_artifact = ModelArtifact(
        id="art_local",
        run_id="run_local",
        artifact_version="v1",
        path=(tmp_path / "local.manifest.json"),
        sha256="sha-local",
        manifest={"classes": ["kick", "other"]},
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "echozero.foundry.services.selection_model_improvement_service.resolve_installed_binary_drum_bundles",
        lambda labels: {
            "kick": InstalledRuntimeBundle(
                label="kick",
                manifest_path=(tmp_path / "installed.manifest.json"),
                weights_path=(tmp_path / "installed.pth"),
                bundle_dir=tmp_path,
            )
        },
    )

    positive_signal = _review_signal(
        signal_id="rsig_pos",
        item_id="timeline_review:version_a:layer_kick:event_good",
        polarity=ReviewPolarity.POSITIVE,
        outcome=ReviewOutcome.CORRECT,
        decision_kind=ReviewDecisionKind.VERIFIED,
    )
    negative_signal = _review_signal(
        signal_id="rsig_neg",
        item_id="timeline_review:version_a:layer_kick:event_bad",
        polarity=ReviewPolarity.NEGATIVE,
        outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.REJECTED,
    )

    class _FakeSignalRepository:
        def get(self, signal_id: str):
            return {"rsig_pos": positive_signal, "rsig_neg": negative_signal}.get(signal_id)

    @dataclass
    class _FakeDatasets:
        created_version: DatasetVersion | None = None
        created_dataset: Dataset | None = None
        run_spec: dict[str, object] | None = None

        def __post_init__(self) -> None:
            self._versions = {
                "dsv_pos": DatasetVersion(
                    id="dsv_pos",
                    dataset_id="ds_pos",
                    version=1,
                    manifest_hash="hash-pos",
                    sample_rate=22050,
                    audio_standard="mono_wav_pcm16",
                    class_map=["kick"],
                    samples=[
                        _dataset_sample("sm_pos", "kick", review_polarity="positive"),
                    ],
                ),
                "dsv_neg": DatasetVersion(
                    id="dsv_neg",
                    dataset_id="ds_neg",
                    version=1,
                    manifest_hash="hash-neg",
                    sample_rate=22050,
                    audio_standard="mono_wav_pcm16",
                    class_map=["kick"],
                    samples=[
                        _dataset_sample("sm_neg", "kick", review_polarity="negative"),
                    ],
                ),
                "dsv_review_binary": DatasetVersion(
                    id="dsv_review_binary",
                    dataset_id="ds_review_binary",
                    version=1,
                    manifest_hash="hash-review-binary",
                    sample_rate=22050,
                    audio_standard="mono_wav_pcm16",
                    class_map=["kick", "other"],
                    samples=[
                        _dataset_sample("sm_rel_pos", "kick", review_polarity="positive"),
                        _dataset_sample("sm_rel_neg", "other", review_polarity="negative"),
                    ],
                ),
            }

        def materialize_review_signal(self, session, signal):
            assert session.source_ref == str(tmp_path)
            if signal.id == "rsig_pos":
                return {
                    "status": "materialized",
                    "version_id": "dsv_pos",
                    "materialized_signal_samples": ["sm_pos"],
                }
            return {
                "status": "materialized",
                "version_id": "dsv_neg",
                "materialized_signal_samples": ["sm_neg"],
            }

        def get_version(self, version_id: str) -> DatasetVersion | None:
            return self._versions.get(version_id) if self.created_version is None else (
                self.created_version if self.created_version.id == version_id else self._versions.get(version_id)
            )

        def create_dataset(self, name: str, *, source_kind: str, source_ref: str, metadata: dict[str, object]) -> Dataset:
            self.created_dataset = Dataset(
                id="ds_selection",
                name=name,
                source_kind=source_kind,
                source_ref=source_ref,
                metadata=metadata,
            )
            return self.created_dataset

        def create_version_from_samples(self, dataset_id: str, *, samples: list[DatasetSample], **kwargs) -> DatasetVersion:
            self.created_version = DatasetVersion(
                id="dsv_selection",
                dataset_id=dataset_id,
                version=1,
                manifest_hash="hash-selection",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick", "other"],
                samples=samples,
                taxonomy=kwargs["taxonomy"],
                label_policy=kwargs["label_policy"],
                manifest=kwargs["manifest"],
                stats=kwargs["stats"],
                lineage=kwargs["lineage"],
            )
            return self.created_version

        def derive_binary_dataset_version(self, source_version_id: str, *, positive_label: str, negative_label: str):
            assert source_version_id == "dsv_review_source"
            assert positive_label == "kick"
            assert negative_label == "other"
            return self._versions["dsv_review_binary"]

    fake_datasets = _FakeDatasets()

    class _FakeApp:
        datasets = fake_datasets

        def list_artifacts(self) -> list[ModelArtifact]:
            return [local_artifact, current_artifact]

        def extract_project_review_dataset(self, *args, **kwargs) -> DatasetVersion:
            return DatasetVersion(
                id="dsv_review_source",
                dataset_id="ds_review_source",
                version=1,
                manifest_hash="hash-review-source",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick", "snare"],
            )

        def plan_version(self, version_id: str, **kwargs) -> dict[str, object]:
            assert version_id == "dsv_selection"
            return {"version_id": version_id}

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            fake_datasets.run_spec = run_spec
            return TrainRun(
                id="run_candidate",
                dataset_version_id=dataset_version_id,
                status=TrainRunStatus.QUEUED,
                spec=run_spec,
                spec_hash="hash-run",
            )

        def start_run(self, run_id: str) -> TrainRun:
            return TrainRun(
                id=run_id,
                dataset_version_id="dsv_selection",
                status=TrainRunStatus.COMPLETED,
                spec=fake_datasets.run_spec or {},
                spec_hash="hash-run",
            )

        def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
            return [
                ModelArtifact(
                    id="art_candidate",
                    run_id=run_id,
                    artifact_version="v1",
                    path=(tmp_path / "candidate.manifest.json"),
                    sha256="sha-candidate",
                    manifest={"classes": ["kick", "other"]},
                )
            ]

        def validate_artifact(self, artifact_id: str) -> CompatibilityReport:
            return CompatibilityReport(
                artifact_id=artifact_id,
                consumer="PyTorchAudioClassify",
                ok=True,
            )

    service = SelectionModelImprovementService(
        tmp_path,
        foundry_app_factory=lambda _root: _FakeApp(),
        review_signal_repository=_FakeSignalRepository(),
    )

    options = service.list_base_model_options(target_label="kick")
    result = service.train_candidate_model(
        ImproveModelTrainingRequest(
            target_label="kick",
            selected_signal_ids=("rsig_pos", "rsig_neg"),
            candidate_name="Kick Improve",
            scope_mode="project",
            strength="strong",
            include_related_examples=True,
            base_model_option_id=options[0].option_id,
        )
    )

    assert [option.label for option in options] == [
        "Current installed kick model",
        "Local candidate art_local (2026-05-02)",
    ]
    assert result.selected_signal_count == 2
    assert result.anchor_sample_count == 12
    assert result.related_sample_count == 2
    assert result.base_artifact_id == "art_current"
    assert result.compared_to_base_model is True
    assert fake_datasets.created_version is not None
    assert fake_datasets.created_version.stats["anchor_sample_count"] == 12
    assert fake_datasets.created_version.stats["related_sample_count"] == 2
    assert fake_datasets.run_spec is not None
    assert fake_datasets.run_spec["promotion"] == {"reference_artifact_id": "art_current"}
    assert fake_datasets.run_spec["training"]["epochs"] == 8


def _review_signal(
    *,
    signal_id: str,
    item_id: str,
    polarity: ReviewPolarity,
    outcome: ReviewOutcome,
    decision_kind: ReviewDecisionKind,
) -> ReviewSignal:
    decision = build_review_decision(
        outcome,
        corrected_label="kick" if decision_kind is not ReviewDecisionKind.REJECTED else None,
        review_note="fixture",
        decision_kind=decision_kind,
    )
    assert decision is not None
    return ReviewSignal(
        id=signal_id,
        session_id="timeline_review_project_alpha_version_a",
        item_id=item_id,
        audio_path="/tmp/audio.wav",
        predicted_label="kick",
        target_class="kick",
        polarity=polarity,
        source_provenance={
            "kind": "ez_timeline_review",
            "project_ref": "project:alpha",
            "song_ref": "song:song_a",
            "version_ref": "version:version_a",
            "layer_ref": "layer:layer_kick",
        },
        review_outcome=outcome,
        review_decision=decision,
        corrected_label="kick" if decision.training_eligibility.allows_positive_signal else None,
    )


def _dataset_sample(sample_id: str, label: str, *, review_polarity: str) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        audio_ref=f"/tmp/{sample_id}.wav",
        label=label,
        duration_ms=85.0,
        content_hash=f"hash-{sample_id}",
        source_provenance={"review_polarity": review_polarity},
        quality_flags=["reviewed"],
    )
