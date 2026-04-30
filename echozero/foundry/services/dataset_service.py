from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import CurationState, Dataset, DatasetSample, DatasetVersion
from echozero.foundry.domain.review import (
    ReviewDecision,
    ReviewDecisionKind,
    ReviewItem,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSession,
    ReviewSignal,
)
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository
from echozero.foundry.services.project_review_queue_builder import ProjectReviewQueueBuilder
from echozero.foundry.services.audio_source_validation import (
    AudioSourceMetadata,
    InvalidAudioSourceError,
    inspect_audio_source,
)
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService
from echozero.foundry.services.review_event_state import normalize_review_label
from echozero.foundry.services.split_balance_service import SplitBalanceService


class DatasetService:
    def __init__(
        self,
        root: Path,
        dataset_repo: DatasetRepository | None = None,
        version_repo: DatasetVersionRepository | None = None,
        clip_service: ReviewAudioClipService | None = None,
    ):
        self._root = root
        self._datasets = dataset_repo or DatasetRepository(root)
        self._versions = version_repo or DatasetVersionRepository(root)
        self._clip_service = clip_service or ReviewAudioClipService()
        self._audio_metadata_cache: dict[tuple[str, int, int], AudioSourceMetadata] = {}
        self._audio_hash_cache: dict[tuple[str, int, int], str] = {}

    def create_dataset(
        self,
        name: str,
        source_kind: str = "folder_import",
        *,
        source_ref: str | None = None,
        metadata: dict | None = None,
    ) -> Dataset:
        dataset = Dataset(
            id=f"ds_{uuid4().hex[:12]}",
            name=name,
            source_kind=source_kind,
            source_ref=source_ref,
            metadata=metadata or {},
        )
        return self._datasets.save(dataset)

    def ingest_from_folder(
        self,
        dataset_id: str,
        folder_path: str | Path,
        *,
        sample_rate: int = 22050,
        audio_standard: str = "mono_wav_pcm16",
        taxonomy: dict | None = None,
        label_policy: dict | None = None,
    ) -> DatasetVersion:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        base = Path(folder_path)
        if not base.exists() or not base.is_dir():
            raise ValueError(f"Dataset folder not found: {base}")

        samples: list[DatasetSample] = []
        content_groups: dict[str, list[str]] = {}
        skipped_sources: list[dict[str, str]] = []
        skipped_reason_counts: Counter[str] = Counter()
        for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            label = class_dir.name
            for file in sorted(class_dir.rglob("*")):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}:
                    continue
                rel_path = file.relative_to(base).as_posix()
                rel = file.resolve().as_posix()
                try:
                    audio_metadata = self._inspect_audio_source_cached(file)
                    content_hash = self._content_hash_for_audio(file)
                except InvalidAudioSourceError as exc:
                    skipped_reason_counts[exc.code] += 1
                    skipped_sources.append(
                        {
                            "path": rel,
                            "relative_path": rel_path,
                            "filename": file.name,
                            "label_from_path": label,
                            "reason_code": exc.code,
                            "reason": str(exc),
                        }
                    )
                    continue
                except OSError as exc:
                    skipped_reason_counts["unreadable"] += 1
                    skipped_sources.append(
                        {
                            "path": rel,
                            "relative_path": rel_path,
                            "filename": file.name,
                            "label_from_path": label,
                            "reason_code": "unreadable",
                            "reason": f"audio source bytes could not be read: {exc}",
                        }
                    )
                    continue
                sample_id = f"sm_{uuid4().hex[:12]}"
                content_groups.setdefault(content_hash, []).append(sample_id)
                samples.append(
                    DatasetSample(
                        sample_id=sample_id,
                        audio_ref=rel,
                        label=label,
                        duration_ms=audio_metadata.duration_ms,
                        content_hash=content_hash,
                        source_provenance={
                            "kind": "folder_import",
                            "path": rel,
                            "source_root": str(base.resolve().as_posix()),
                            "relative_path": rel_path,
                            "filename": file.name,
                            "label_from_path": label,
                        },
                        group_id=f"content:{content_hash}",
                        is_synthetic=False,
                        synthetic_provenance={},
                        curation_state=CurationState.UNKNOWN,
                    )
                )

        if not samples:
            if skipped_sources:
                raise ValueError(
                    f"No valid audio samples found in dataset folder: {base} "
                    f"(skipped {len(skipped_sources)} invalid file(s))"
                )
            raise ValueError(f"No audio samples found in dataset folder: {base}")

        existing = self._versions.list_for_dataset(dataset_id)
        next_version_num = (existing[-1].version + 1) if existing else 1

        manifest_hash = self.compute_manifest_hash(samples)
        class_map = sorted({s.label for s in samples})
        resolved_taxonomy = taxonomy or {
            "schema": "foundry.taxonomy.v1",
            "namespace": "percussion.one_shot",
            "version": 1,
            "labels": [
                {"id": label, "display_name": label.replace("_", " "), "aliases": []}
                for label in class_map
            ],
        }
        resolved_label_policy = label_policy or {
            "schema": "foundry.label_policy.v1",
            "classification_mode": "multiclass",
            "unit": "one_shot",
            "allowed_labels": class_map,
            "unknown_label": None,
        }

        stats = {
            "sample_count": len(samples),
            "real_sample_count": sum(1 for s in samples if not s.is_synthetic),
            "synthetic_sample_count": sum(1 for s in samples if s.is_synthetic),
            "class_counts": {
                label: sum(1 for s in samples if s.label == label)
                for label in class_map
            },
            "duplicate_content_hashes": sum(1 for ids in content_groups.values() if len(ids) > 1),
            "skipped_invalid_count": len(skipped_sources),
            "skipped_invalid_by_reason": dict(sorted(skipped_reason_counts.items())),
        }
        dataset_manifest = {
            "schema": "foundry.dataset_manifest.v1",
            "source_kind": dataset.source_kind,
            "source_ref": dataset.source_ref,
            "ingest_root": str(base.resolve().as_posix()),
            "deterministic_order": [s.sample_id for s in samples],
            "content_hash_algorithm": "sha256",
            "content_groups": {key: sorted(ids) for key, ids in sorted(content_groups.items())},
            "synthetic_sample_ids": [s.sample_id for s in samples if s.is_synthetic],
            "real_sample_ids": [s.sample_id for s in samples if not s.is_synthetic],
            "skipped_sources": skipped_sources,
        }

        version = DatasetVersion(
            id=f"dsv_{uuid4().hex[:12]}",
            dataset_id=dataset_id,
            version=next_version_num,
            manifest_hash=manifest_hash,
            sample_rate=sample_rate,
            audio_standard=audio_standard,
            class_map=class_map,
            samples=samples,
            taxonomy=resolved_taxonomy,
            label_policy=resolved_label_policy,
            manifest=dataset_manifest,
            stats=stats,
        )
        return self._versions.save(version)

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        return self._datasets.get(dataset_id)

    def get_version(self, version_id: str) -> DatasetVersion | None:
        return self._versions.get(version_id)

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        return self._versions.list_for_dataset(dataset_id)

    def export_project_review_dataset(
        self,
        project_path: str | Path,
        *,
        project_ref: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        queue_source_kind: str = "ez_project",
    ) -> DatasetVersion:
        """Build and persist one explicit review dataset version from canonical project truth."""

        builder = ProjectReviewQueueBuilder(self._root, clip_service=self._clip_service)
        queue = builder.build_queue(
            project_path,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=ReviewPolarity.POSITIVE,
            review_mode="all_events",
            materialize_pending_clips=False,
        )
        sample_result = self._build_project_review_export_samples(queue.items)
        samples = sample_result["samples"]
        if not samples:
            raise ValueError("No reviewed project events were eligible for dataset export.")

        normalized_project_ref = str(
            project_ref or queue.metadata.get("project_ref") or ""
        ).strip()
        dataset = self._get_or_create_project_review_export_dataset(
            queue_source_kind=queue_source_kind,
            project_ref=normalized_project_ref,
            project_name=str(queue.metadata.get("project_name", queue.project_name)).strip() or queue.project_name,
            source_ref=str(Path(project_path).expanduser().resolve()),
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
        )
        existing_versions = self._versions.list_for_dataset(dataset.id)
        manifest_hash = self.compute_manifest_hash(samples)
        if existing_versions and existing_versions[-1].manifest_hash == manifest_hash:
            return existing_versions[-1]

        class_map = sorted({sample.label for sample in samples})
        manifest = {
            "schema": "foundry.project_review_dataset_manifest.v1",
            "review_dataset_key": dataset.metadata.get("review_dataset_key"),
            "project_ref": normalized_project_ref,
            "queue_source_kind": queue_source_kind,
            "deterministic_order": [sample.sample_id for sample in samples],
            "content_hash_algorithm": "sha256",
            "content_groups": self._content_groups(samples),
            "review_item_ids": sorted(
                {
                    str(sample.source_provenance.get("review_item_id"))
                    for sample in samples
                    if sample.source_provenance.get("review_item_id")
                }
            ),
            "scope": {
                "song_id": song_id,
                "song_version_id": song_version_id,
                "layer_id": layer_id,
            },
        }
        version = DatasetVersion(
            id=f"dsv_{uuid4().hex[:12]}",
            dataset_id=dataset.id,
            version=(existing_versions[-1].version + 1) if existing_versions else 1,
            manifest_hash=manifest_hash,
            sample_rate=22050,
            audio_standard="mono_wav_pcm16",
            class_map=class_map,
            samples=samples,
            taxonomy=self._build_review_taxonomy(class_map),
            label_policy=self._build_review_label_policy(class_map),
            manifest=manifest,
            stats={
                "sample_count": len(samples),
                "real_sample_count": sum(1 for sample in samples if not sample.is_synthetic),
                "synthetic_sample_count": sum(1 for sample in samples if sample.is_synthetic),
                "class_counts": {
                    label: sum(1 for sample in samples if sample.label == label)
                    for label in class_map
                },
                "review_positive_count": sum(
                    1
                    for sample in samples
                    if sample.source_provenance.get("review_polarity") == "positive"
                ),
                "review_negative_count": sum(
                    1
                    for sample in samples
                    if sample.source_provenance.get("review_polarity") == "negative"
                ),
                "reviewed_item_count": int(sample_result["reviewed_item_count"]),
                "decision_counts": dict(sample_result["decision_counts"]),
            },
            lineage={
                "kind": "project_review_export",
                "project_ref": normalized_project_ref,
                "source_ref": str(Path(project_path).expanduser().resolve()),
                "song_id": song_id,
                "song_version_id": song_version_id,
                "layer_id": layer_id,
            },
        )
        return self._versions.save(version)

    def materialize_review_signal(
        self,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> dict[str, object]:
        source_queue_kind = (
            str(session.metadata.get("queue_source_kind", "manual_review")).strip()
            or "manual_review"
        )
        project_dir = self._resolve_project_dir(session.source_ref)
        if project_dir is None:
            return {
                "status": "deferred",
                "reason": "non_project_session",
                "queue_source_kind": source_queue_kind,
                "signal_id": signal.id,
            }
        canonical_project_export = self._refresh_project_review_export(project_dir)

        sample_result = self._build_review_materialization_samples(
            session=session,
            signal=signal,
        )
        samples = list(sample_result["samples"])
        if not samples:
            return {
                "status": "deferred",
                "reason": str(sample_result.get("reason", "no_eligible_training_samples")),
                "details": self._serialize_materialization_details(
                    list(sample_result.get("details", []))
                ),
                "canonical_project_export": canonical_project_export,
                "queue_source_kind": source_queue_kind,
                "signal_id": signal.id,
            }

        dataset = self._get_or_create_review_dataset(session=session, signal=signal)
        existing_versions = self._versions.list_for_dataset(dataset.id)
        previous_samples = list(existing_versions[-1].samples) if existing_versions else []
        merged_samples = [
            sample
            for sample in previous_samples
            if str(sample.source_provenance.get("review_signal_id", "")).strip() != signal.id
        ]
        merged_samples.extend(samples)
        if not merged_samples:
            return {
                "status": "deferred",
                "reason": "no_eligible_training_samples",
                "details": self._serialize_materialization_details(
                    list(sample_result.get("details", []))
                ),
                "canonical_project_export": canonical_project_export,
                "queue_source_kind": source_queue_kind,
                "signal_id": signal.id,
            }

        manifest_hash = self.compute_manifest_hash(merged_samples)
        if existing_versions and existing_versions[-1].manifest_hash == manifest_hash:
            version = existing_versions[-1]
        else:
            class_map = sorted({sample.label for sample in merged_samples})
            version = DatasetVersion(
                id=f"dsv_{uuid4().hex[:12]}",
                dataset_id=dataset.id,
                version=(existing_versions[-1].version + 1) if existing_versions else 1,
                manifest_hash=manifest_hash,
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=class_map,
                samples=merged_samples,
                taxonomy=self._build_review_taxonomy(class_map),
                label_policy=self._build_review_label_policy(class_map),
                manifest={
                    "schema": "foundry.review_dataset_manifest.v1",
                    "review_dataset_key": dataset.metadata.get("review_dataset_key"),
                    "queue_source_kind": source_queue_kind,
                    "project_ref": dataset.metadata.get("project_ref"),
                    "deterministic_order": [sample.sample_id for sample in merged_samples],
                    "content_hash_algorithm": "sha256",
                    "content_groups": self._content_groups(merged_samples),
                    "review_signal_ids": sorted(
                        {
                            str(sample.source_provenance.get("review_signal_id"))
                            for sample in merged_samples
                            if sample.source_provenance.get("review_signal_id")
                        }
                    ),
                },
                stats={
                    "sample_count": len(merged_samples),
                    "real_sample_count": sum(
                        1 for sample in merged_samples if not sample.is_synthetic
                    ),
                    "synthetic_sample_count": sum(
                        1 for sample in merged_samples if sample.is_synthetic
                    ),
                    "class_counts": {
                        label: sum(1 for sample in merged_samples if sample.label == label)
                        for label in class_map
                    },
                    "review_positive_count": sum(
                        1
                        for sample in merged_samples
                        if sample.source_provenance.get("review_polarity") == "positive"
                    ),
                    "review_negative_count": sum(
                        1
                        for sample in merged_samples
                        if sample.source_provenance.get("review_polarity") == "negative"
                    ),
                },
                lineage={
                    "kind": "review_signal_materialization",
                    "project_ref": dataset.metadata.get("project_ref"),
                    "source_ref": session.source_ref,
                },
            )
            version = self._versions.save(version)

        return {
            "status": "materialized",
            "queue_source_kind": source_queue_kind,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "version_id": version.id,
            "version": version.version,
            "sample_count": len(version.samples),
            "signal_id": signal.id,
            "materialized_signal_samples": [sample.sample_id for sample in samples],
            "canonical_project_export": canonical_project_export,
            "details": self._serialize_materialization_details(
                list(sample_result.get("details", []))
            ),
        }

    def update_version_plans(self, version_id: str, *, split_plan: dict, balance_plan: dict) -> DatasetVersion:
        version = self._versions.get(version_id)
        if version is None:
            raise ValueError(f"DatasetVersion not found: {version_id}")
        assignments = split_plan.get("assignments", {})
        version.samples = [
            DatasetSample(
                sample_id=sample.sample_id,
                audio_ref=sample.audio_ref,
                label=sample.label,
                duration_ms=sample.duration_ms,
                content_hash=sample.content_hash,
                source_provenance=sample.source_provenance,
                group_id=sample.group_id,
                is_synthetic=sample.is_synthetic,
                synthetic_provenance=sample.synthetic_provenance,
                quality_flags=sample.quality_flags,
                split_assignment=assignments.get(sample.sample_id),
                curation_state=sample.curation_state,
            )
            for sample in version.samples
        ]
        version.split_plan = split_plan
        version.balance_plan = balance_plan
        return self._versions.save(version)

    def apply_curation(self, version_id: str, decisions: dict[str, CurationState]) -> DatasetVersion:
        version = self._versions.get(version_id)
        if version is None:
            raise ValueError(f"DatasetVersion not found: {version_id}")

        updated_samples: list[DatasetSample] = []
        for sample in version.samples:
            state = decisions.get(sample.sample_id, sample.curation_state)
            updated_samples.append(
                DatasetSample(
                    sample_id=sample.sample_id,
                    audio_ref=sample.audio_ref,
                    label=sample.label,
                    duration_ms=sample.duration_ms,
                    content_hash=sample.content_hash,
                    source_provenance=sample.source_provenance,
                    group_id=sample.group_id,
                    is_synthetic=sample.is_synthetic,
                    synthetic_provenance=sample.synthetic_provenance,
                    quality_flags=sample.quality_flags,
                    split_assignment=sample.split_assignment,
                    curation_state=state,
                )
            )

        accepted = [s for s in updated_samples if s.curation_state != CurationState.REJECTED]
        existing = self._versions.list_for_dataset(version.dataset_id)
        next_version_num = (existing[-1].version + 1) if existing else (version.version + 1)

        manifest_hash = self.compute_manifest_hash(accepted)
        accepted_ids = {sample.sample_id for sample in accepted}
        split_plan = self._filter_split_plan(version.split_plan, accepted_ids)

        next_version = DatasetVersion(
            id=f"dsv_{uuid4().hex[:12]}",
            dataset_id=version.dataset_id,
            version=next_version_num,
            manifest_hash=manifest_hash,
            sample_rate=version.sample_rate,
            audio_standard=version.audio_standard,
            class_map=sorted({s.label for s in accepted}),
            samples=accepted,
            taxonomy=version.taxonomy,
            label_policy=version.label_policy,
            manifest=self._build_curated_manifest(version.manifest, accepted_ids),
            split_plan=split_plan,
            balance_plan=version.balance_plan,
            stats={
                "sample_count": len(accepted),
                "real_sample_count": sum(1 for s in accepted if not s.is_synthetic),
                "synthetic_sample_count": sum(1 for s in accepted if s.is_synthetic),
                "class_counts": {
                    label: sum(1 for s in accepted if s.label == label)
                    for label in sorted({s.label for s in accepted})
                },
                "curated_from": version.id,
            },
            lineage={"source_version_id": version.id, "kind": "curation"},
        )
        return self._versions.save(next_version)

    def derive_binary_dataset_version(
        self,
        source_version_id: str,
        *,
        positive_label: str,
        negative_label: str = "other",
    ) -> DatasetVersion:
        """Create a binary one-vs-rest dataset version from an existing labeled review dataset."""
        source_version = self.get_version(source_version_id)
        if source_version is None:
            raise ValueError(f"DatasetVersion not found: {source_version_id}")
        source_dataset = self.get_dataset(source_version.dataset_id)
        if source_dataset is None:
            raise ValueError(f"Dataset not found: {source_version.dataset_id}")

        normalized_positive = positive_label.strip().lower()
        normalized_negative = negative_label.strip().lower()
        if not normalized_positive:
            raise ValueError("positive_label must be non-empty")
        if normalized_positive == normalized_negative:
            raise ValueError("positive_label and negative_label must be different")

        source_labels = {sample.label.strip().lower() for sample in source_version.samples}
        if normalized_positive not in source_labels:
            raise ValueError(
                f"Source dataset version '{source_version_id}' does not contain label '{normalized_positive}'."
            )
        if source_labels == {normalized_positive}:
            raise ValueError("Binary dataset derivation requires at least one non-positive sample.")

        target_dataset = self._get_or_create_binary_dataset(
            source_dataset=source_dataset,
            positive_label=normalized_positive,
        )
        existing_versions = self._versions.list_for_dataset(target_dataset.id)
        derived_samples = [
            replace(
                sample,
                label=self._binary_label_for_sample(
                    sample,
                    positive_label=normalized_positive,
                    negative_label=normalized_negative,
                ),
            )
            for sample in source_version.samples
        ]
        derived_labels = {sample.label for sample in derived_samples}
        if normalized_positive not in derived_labels:
            raise ValueError(
                f"Source dataset version '{source_version_id}' does not contain any positive training samples for '{normalized_positive}'."
            )
        if derived_labels == {normalized_positive}:
            raise ValueError("Binary dataset derivation requires at least one non-positive sample.")
        manifest_hash = self.compute_manifest_hash(derived_samples)
        next_version = DatasetVersion(
            id=f"dsv_{uuid4().hex[:12]}",
            dataset_id=target_dataset.id,
            version=(existing_versions[-1].version + 1) if existing_versions else 1,
            manifest_hash=manifest_hash,
            sample_rate=source_version.sample_rate,
            audio_standard=source_version.audio_standard,
            class_map=[normalized_positive, normalized_negative],
            samples=derived_samples,
            taxonomy=self._build_binary_taxonomy(
                positive_label=normalized_positive,
                negative_label=normalized_negative,
            ),
            label_policy=self._build_binary_label_policy(
                positive_label=normalized_positive,
                negative_label=normalized_negative,
            ),
            manifest={
                **dict(source_version.manifest),
                "schema": "foundry.derived_binary_dataset_manifest.v1",
                "source_dataset_id": source_dataset.id,
                "source_dataset_version_id": source_version.id,
                "derivation": {
                    "kind": "positive_vs_other",
                    "positive_label": normalized_positive,
                    "negative_label": normalized_negative,
                },
            },
            split_plan=self._copy_source_split_plan(source_version),
            balance_plan=dict(source_version.balance_plan),
            stats=self._binary_dataset_stats(
                derived_samples,
                source_version_id=source_version.id,
                positive_label=normalized_positive,
                negative_label=normalized_negative,
            ),
            lineage={
                "kind": "positive_vs_other",
                "source_dataset_id": source_dataset.id,
                "source_version_id": source_version.id,
                "positive_label": normalized_positive,
                "negative_label": normalized_negative,
            },
        )
        return self._versions.save(next_version)

    @staticmethod
    def _filter_split_plan(split_plan: dict, accepted_ids: set[str]) -> dict:
        if not split_plan:
            return {}
        assignments = {
            sample_id: split_name
            for sample_id, split_name in split_plan.get("assignments", {}).items()
            if sample_id in accepted_ids
        }
        filtered = dict(split_plan)
        filtered["assignments"] = assignments
        filtered["train_ids"] = [sample_id for sample_id in split_plan.get("train_ids", []) if sample_id in accepted_ids]
        filtered["val_ids"] = [sample_id for sample_id in split_plan.get("val_ids", []) if sample_id in accepted_ids]
        filtered["test_ids"] = [sample_id for sample_id in split_plan.get("test_ids", []) if sample_id in accepted_ids]
        return filtered

    @staticmethod
    def _build_curated_manifest(manifest: dict, accepted_ids: set[str]) -> dict:
        if not manifest:
            return {}
        curated_manifest = dict(manifest)
        deterministic_order = [sample_id for sample_id in manifest.get("deterministic_order", []) if sample_id in accepted_ids]
        content_groups = {
            key: [sample_id for sample_id in sample_ids if sample_id in accepted_ids]
            for key, sample_ids in manifest.get("content_groups", {}).items()
        }
        curated_manifest["deterministic_order"] = deterministic_order
        curated_manifest["content_groups"] = {key: ids for key, ids in content_groups.items() if ids}
        curated_manifest["synthetic_sample_ids"] = [
            sample_id for sample_id in manifest.get("synthetic_sample_ids", []) if sample_id in accepted_ids
        ]
        curated_manifest["real_sample_ids"] = [
            sample_id for sample_id in manifest.get("real_sample_ids", []) if sample_id in accepted_ids
        ]
        return curated_manifest

    @classmethod
    def compute_manifest_hash(cls, samples: list[DatasetSample]) -> str:
        manifest = cls._canonical_manifest_rows(samples)
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()

    def _get_or_create_project_review_export_dataset(
        self,
        *,
        queue_source_kind: str,
        project_ref: str,
        project_name: str,
        source_ref: str,
        song_id: str | None,
        song_version_id: str | None,
        layer_id: str | None,
    ) -> Dataset:
        dataset_key = self._project_review_dataset_key(
            queue_source_kind=queue_source_kind,
            project_ref=project_ref,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
        )
        for dataset in self._datasets.list():
            metadata = dataset.metadata or {}
            if dataset.source_kind != "project_review_export":
                continue
            if str(metadata.get("review_dataset_key", "")).strip() == dataset_key:
                return dataset
        return self.create_dataset(
            f"Review Samples - {project_name}",
            source_kind="project_review_export",
            source_ref=source_ref,
            metadata={
                "schema": "foundry.project_review_dataset.v1",
                "review_dataset_key": dataset_key,
                "queue_source_kind": queue_source_kind,
                "project_ref": project_ref,
                "project_name": project_name,
                "song_id": song_id,
                "song_version_id": song_version_id,
                "layer_id": layer_id,
            },
        )

    def _build_project_review_export_samples(
        self,
        items: list[ReviewItem],
    ) -> dict[str, object]:
        samples: list[DatasetSample] = []
        decision_counts: Counter[str] = Counter()
        reviewed_item_count = 0
        for item in items:
            if item.review_outcome == ReviewOutcome.PENDING or item.review_decision is None:
                continue
            reviewed_item_count += 1
            decision_counts[item.review_decision.kind.value] += 1
            lane = self._project_review_export_lane(item)
            if lane is None:
                continue
            sample = self._build_project_review_export_sample(
                item=item,
                decision=item.review_decision,
                lane=lane,
            )
            if sample is not None:
                samples.append(sample)
        return {
            "samples": samples,
            "reviewed_item_count": reviewed_item_count,
            "decision_counts": decision_counts,
        }

    def _build_project_review_export_sample(
        self,
        *,
        item: ReviewItem,
        decision: ReviewDecision,
        lane: str,
    ) -> DatasetSample | None:
        audio_path = self._resolve_project_review_export_audio(
            item=item,
            decision=decision,
            lane=lane,
        )
        if audio_path is None:
            return None
        label = self._project_review_export_label(item=item, decision=decision, lane=lane)
        if label is None:
            return None
        try:
            metadata = self._inspect_audio_source_cached(audio_path)
            content_hash = self._content_hash_for_audio(audio_path)
        except (InvalidAudioSourceError, OSError):
            return None
        return DatasetSample(
            sample_id=self._project_review_sample_id(item_id=item.item_id, lane=lane),
            audio_ref=str(audio_path.resolve()),
            label=label,
            duration_ms=metadata.duration_ms,
            content_hash=content_hash,
            source_provenance={
                "kind": "project_review_export",
                "review_item_id": item.item_id,
                "review_polarity": lane,
                "review_outcome": item.review_outcome.value,
                "review_decision_kind": decision.kind.value,
                **dict(item.source_provenance),
            },
            group_id=f"content:{content_hash}",
            is_synthetic=False,
            synthetic_provenance={},
            quality_flags=["reviewed", f"review_{lane}", f"decision_{decision.kind.value}"],
            curation_state=CurationState.ACCEPTED,
        )

    def _resolve_project_review_export_audio(
        self,
        *,
        item: ReviewItem,
        decision: ReviewDecision,
        lane: str,
    ) -> Path | None:
        del decision, lane
        candidate = Path(item.audio_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        return None

    @staticmethod
    def _project_review_export_label(
        *,
        item: ReviewItem,
        decision: ReviewDecision,
        lane: str,
    ) -> str | None:
        if lane == "negative":
            candidate = (
                item.source_provenance.get("original_label")
                or item.predicted_label
                or item.target_class
            )
        else:
            candidate = (
                item.source_provenance.get("current_target_class")
                or item.target_class
                or decision.corrected_label
                or item.predicted_label
            )
        if candidate is None:
            return None
        return normalize_review_label(candidate)

    @staticmethod
    def _project_review_export_lane(item: ReviewItem) -> str | None:
        decision = item.review_decision
        if decision is None or item.review_outcome == ReviewOutcome.PENDING:
            return None
        promotion_state = str(item.source_provenance.get("promotion_state", "")).strip().lower()
        if promotion_state == "demoted" or decision.kind == ReviewDecisionKind.REJECTED:
            return "negative"
        if promotion_state == "promoted":
            return "positive"
        if item.review_outcome == ReviewOutcome.CORRECT:
            return "positive"
        return "negative"

    @staticmethod
    def _project_review_sample_id(*, item_id: str, lane: str) -> str:
        digest = hashlib.sha1(f"{item_id}|{lane}".encode("utf-8")).hexdigest()[:16]
        return f"prsm_{digest}"

    @staticmethod
    def _project_review_dataset_key(
        *,
        queue_source_kind: str,
        project_ref: str,
        song_id: str | None,
        song_version_id: str | None,
        layer_id: str | None,
    ) -> str:
        parts = [queue_source_kind.strip() or "ez_project", project_ref.strip()]
        if song_id:
            parts.append(f"song:{song_id}")
        if song_version_id:
            parts.append(f"version:{song_version_id}")
        if layer_id:
            parts.append(f"layer:{layer_id}")
        return ":".join(part for part in parts if part)

    def _get_or_create_review_dataset(
        self,
        *,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> Dataset:
        dataset_key = self._review_dataset_key(session=session, signal=signal)
        for dataset in self._datasets.list():
            if dataset.source_kind != "review_signal":
                continue
            if dataset.metadata.get("review_dataset_key") == dataset_key:
                return dataset
        dataset_name = self._review_dataset_name(session=session, signal=signal)
        return self.create_dataset(
            dataset_name,
            source_kind="review_signal",
            source_ref=session.source_ref,
            metadata={
                "schema": "foundry.review_dataset.v1",
                "review_dataset_key": dataset_key,
                "queue_source_kind": session.metadata.get("queue_source_kind", "manual_review"),
                "project_ref": signal.source_provenance.get("project_ref"),
                "source_session_id": session.id,
            },
        )

    def _build_review_materialization_samples(
        self,
        *,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> dict[str, object]:
        decision = signal.review_decision
        if decision is None:
            return {"samples": [], "reason": "missing_review_decision", "details": []}
        details: list[dict[str, object]] = []
        samples: list[DatasetSample] = []
        for lane in self._training_lanes(signal, decision):
            sample_result = self._build_review_sample(
                session=session,
                signal=signal,
                decision=decision,
                lane=lane,
            )
            details.append(sample_result)
            sample = sample_result.get("sample")
            if isinstance(sample, DatasetSample):
                samples.append(sample)
        if samples:
            return {"samples": samples, "reason": None, "details": details}
        reason = next(
            (
                str(detail.get("reason"))
                for detail in details
                if isinstance(detail, dict) and detail.get("reason")
            ),
            "no_eligible_training_samples",
        )
        return {"samples": [], "reason": reason, "details": details}

    def _build_review_sample(
        self,
        *,
        session: ReviewSession,
        signal: ReviewSignal,
        decision: ReviewDecision,
        lane: str,
    ) -> dict[str, object]:
        audio_result = self._resolve_review_sample_audio(
            session=session,
            signal=signal,
            decision=decision,
            lane=lane,
        )
        audio_path = audio_result.get("audio_path")
        if not isinstance(audio_path, Path):
            return {
                "lane": lane,
                "status": "skipped",
                "reason": audio_result.get("reason", "missing_audio"),
            }
        label = self._review_sample_label(signal=signal, decision=decision, lane=lane)
        if label is None:
            return {
                "lane": lane,
                "status": "skipped",
                "reason": "missing_label",
            }
        try:
            metadata = self._inspect_audio_source_cached(audio_path)
            content_hash = self._content_hash_for_audio(audio_path)
        except InvalidAudioSourceError as exc:
            return {
                "lane": lane,
                "status": "skipped",
                "reason": exc.code,
            }
        except OSError:
            return {
                "lane": lane,
                "status": "skipped",
                "reason": "unreadable",
            }
        sample = DatasetSample(
            sample_id=self._review_sample_id(signal_id=signal.id, lane=lane),
            audio_ref=str(audio_path.resolve()),
            label=label,
            duration_ms=metadata.duration_ms,
            content_hash=content_hash,
            source_provenance={
                "kind": "review_signal_materialization",
                "review_signal_id": signal.id,
                "review_session_id": session.id,
                "review_polarity": lane,
                "review_outcome": signal.review_outcome.value,
                "review_decision_kind": decision.kind.value,
                "project_writeback": signal.source_provenance.get("project_writeback"),
                **dict(signal.source_provenance),
            },
            group_id=f"content:{content_hash}",
            is_synthetic=False,
            synthetic_provenance={},
            quality_flags=[
                "reviewed",
                f"review_{lane}",
                f"decision_{decision.kind.value}",
            ],
            curation_state=CurationState.ACCEPTED,
        )
        return {
            "lane": lane,
            "status": "materialized",
            "sample": sample,
            "audio_ref": sample.audio_ref,
        }

    def _resolve_review_sample_audio(
        self,
        *,
        session: ReviewSession,
        signal: ReviewSignal,
        decision: ReviewDecision,
        lane: str,
    ) -> dict[str, object]:
        if lane == "negative":
            original_path = Path(signal.audio_path).expanduser()
            if original_path.exists():
                return {"audio_path": original_path}
            return {"reason": "missing_original_audio"}
        if decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED:
            return self._resolve_materialized_review_clip(
                signal=signal,
                clip_stem=f"{signal.id}_{lane}",
                start_ms=decision.corrected_start_ms,
                end_ms=decision.corrected_end_ms,
                reason="missing_created_event_materialization",
            )
        if not decision.training_eligibility.requires_materialized_correction:
            original_path = Path(signal.audio_path).expanduser()
            if original_path.exists():
                return {"audio_path": original_path}
            return {"reason": "missing_original_audio"}
        return self._resolve_materialized_review_clip(
            signal=signal,
            clip_stem=f"{signal.id}_{lane}",
            start_ms=decision.corrected_start_ms,
            end_ms=decision.corrected_end_ms,
            reason="missing_materialized_correction",
        )

    def _resolve_materialized_review_clip(
        self,
        *,
        signal: ReviewSignal,
        clip_stem: str,
        start_ms: float | None,
        end_ms: float | None,
        reason: str,
    ) -> dict[str, object]:
        source_audio_ref = signal.source_provenance.get("source_audio_ref")
        if not isinstance(source_audio_ref, str) or not source_audio_ref.strip():
            return {"reason": reason}
        if start_ms is None or end_ms is None:
            return {"reason": reason}
        clip_path = self._clip_service.materialize_event_clip(
            source_audio_path=Path(source_audio_ref),
            clip_cache_dir=self._root / "foundry" / "cache" / "review_training_samples",
            clip_stem=clip_stem,
            start_seconds=float(start_ms) / 1000.0,
            end_seconds=float(end_ms) / 1000.0,
        )
        if clip_path is None:
            return {"reason": reason}
        return {"audio_path": clip_path}

    @staticmethod
    def _review_sample_label(
        *,
        signal: ReviewSignal,
        decision: ReviewDecision,
        lane: str,
    ) -> str | None:
        if lane == "negative":
            label = signal.target_class or signal.predicted_label
        else:
            label = decision.corrected_label or signal.target_class or signal.predicted_label
        text = str(label).strip() if label is not None else ""
        return text or None

    @staticmethod
    def _training_lanes(signal: ReviewSignal, decision: ReviewDecision) -> list[str]:
        lanes: list[str] = []
        if decision.training_eligibility.allows_negative_signal:
            lanes.append("negative")
        if decision.training_eligibility.allows_positive_signal:
            lanes.append("positive")
        if not lanes and signal.review_outcome == ReviewOutcome.CORRECT:
            lanes.append("positive")
        return lanes

    @staticmethod
    def _review_sample_id(*, signal_id: str, lane: str) -> str:
        digest = hashlib.sha1(f"{signal_id}|{lane}".encode("utf-8")).hexdigest()[:16]
        return f"rsm_{digest}"

    @staticmethod
    def _review_dataset_key(
        *,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> str:
        queue_kind = str(session.metadata.get("queue_source_kind", "manual_review")).strip() or "manual_review"
        project_ref = str(signal.source_provenance.get("project_ref", "")).strip()
        if project_ref:
            return f"{queue_kind}:{project_ref}"
        return f"{queue_kind}:{session.id}"

    @staticmethod
    def _review_dataset_name(
        *,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> str:
        project_name = str(signal.source_provenance.get("project_name", "")).strip()
        if project_name:
            return f"Review Samples - {project_name}"
        return f"Review Samples - {session.name}"

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

    def _inspect_audio_source_cached(self, audio_path: Path) -> AudioSourceMetadata:
        key = self._audio_cache_key(audio_path)
        if key is not None:
            cached = self._audio_metadata_cache.get(key)
            if cached is not None:
                return cached
        metadata = inspect_audio_source(audio_path)
        if key is not None:
            self._audio_metadata_cache[key] = metadata
        return metadata

    def _content_hash_for_audio(self, audio_path: Path) -> str:
        key = self._audio_cache_key(audio_path)
        if key is not None:
            cached = self._audio_hash_cache.get(key)
            if cached is not None:
                return cached

        digest = hashlib.sha256()
        with audio_path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        content_hash = digest.hexdigest()
        if key is not None:
            self._audio_hash_cache[key] = content_hash
        return content_hash

    @staticmethod
    def _audio_cache_key(audio_path: Path) -> tuple[str, int, int] | None:
        try:
            resolved = audio_path.expanduser().resolve()
            stat = resolved.stat()
        except OSError:
            return None
        return (resolved.as_posix(), int(stat.st_size), int(stat.st_mtime_ns))

    def _refresh_project_review_export(self, project_dir: Path) -> dict[str, object]:
        try:
            version = self.export_project_review_dataset(project_dir, queue_source_kind="ez_project")
        except ValueError:
            return {
                "status": "deferred",
                "reason": "no_reviewed_project_events",
                "source_ref": str(project_dir.resolve()),
            }
        dataset = self.get_dataset(version.dataset_id)
        return {
            "status": "materialized",
            "dataset_id": version.dataset_id,
            "dataset_name": dataset.name if dataset is not None else None,
            "version_id": version.id,
            "version": version.version,
            "sample_count": int(version.stats.get("sample_count", len(version.samples))),
            "review_positive_count": int(version.stats.get("review_positive_count", 0)),
            "review_negative_count": int(version.stats.get("review_negative_count", 0)),
            "reviewed_item_count": int(version.stats.get("reviewed_item_count", len(version.samples))),
            "source_ref": str(project_dir.resolve()),
        }

    @staticmethod
    def _content_groups(samples: list[DatasetSample]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for sample in samples:
            key = sample.group_id or f"content:{sample.content_hash}"
            groups.setdefault(key, []).append(sample.sample_id)
        return {key: sorted(sample_ids) for key, sample_ids in sorted(groups.items())}

    @staticmethod
    def _build_review_taxonomy(class_map: list[str]) -> dict[str, object]:
        return {
            "schema": "foundry.taxonomy.v1",
            "namespace": "percussion.one_shot",
            "version": 1,
            "labels": [
                {"id": label, "display_name": label.replace("_", " "), "aliases": []}
                for label in class_map
            ],
        }

    @staticmethod
    def _build_review_label_policy(class_map: list[str]) -> dict[str, object]:
        return {
            "schema": "foundry.label_policy.v1",
            "classification_mode": "multiclass",
            "unit": "one_shot",
            "allowed_labels": class_map,
            "unknown_label": None,
        }

    @staticmethod
    def _serialize_materialization_details(details: list[dict[str, object]]) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for detail in details:
            row = dict(detail)
            sample = row.pop("sample", None)
            if isinstance(sample, DatasetSample):
                row["sample_id"] = sample.sample_id
                row["label"] = sample.label
                row["audio_ref"] = sample.audio_ref
            serialized.append(row)
        return serialized

    def _get_or_create_binary_dataset(
        self,
        *,
        source_dataset: Dataset,
        positive_label: str,
    ) -> Dataset:
        dataset_key = f"{source_dataset.id}:{positive_label}"
        for dataset in self._datasets.list():
            if dataset.source_kind != "derived_binary_review":
                continue
            if dataset.metadata.get("binary_dataset_key") == dataset_key:
                return dataset
        return self.create_dataset(
            f"{source_dataset.name} - {positive_label} vs other",
            source_kind="derived_binary_review",
            source_ref=source_dataset.source_ref,
            metadata={
                "schema": "foundry.derived_binary_dataset.v1",
                "binary_dataset_key": dataset_key,
                "source_dataset_id": source_dataset.id,
                "positive_label": positive_label,
            },
        )

    @staticmethod
    def _build_binary_taxonomy(*, positive_label: str, negative_label: str) -> dict[str, object]:
        return {
            "schema": "foundry.taxonomy.v1",
            "namespace": "percussion.one_shot",
            "version": 1,
            "labels": [
                {"id": positive_label, "display_name": positive_label.replace("_", " "), "aliases": []},
                {"id": negative_label, "display_name": negative_label.replace("_", " "), "aliases": []},
            ],
        }

    @staticmethod
    def _build_binary_label_policy(*, positive_label: str, negative_label: str) -> dict[str, object]:
        return {
            "schema": "foundry.label_policy.v1",
            "classification_mode": "binary",
            "unit": "one_shot",
            "allowed_labels": [positive_label, negative_label],
            "unknown_label": None,
        }

    @staticmethod
    def _binary_label_for_sample(
        sample: DatasetSample,
        *,
        positive_label: str,
        negative_label: str,
    ) -> str:
        review_polarity = str(sample.source_provenance.get("review_polarity", "")).strip().lower()
        sample_label = sample.label.strip().lower()
        if review_polarity == "negative":
            return negative_label
        if sample_label == positive_label:
            return positive_label
        return negative_label

    @staticmethod
    def _binary_dataset_stats(
        samples: list[DatasetSample],
        *,
        source_version_id: str,
        positive_label: str,
        negative_label: str,
    ) -> dict[str, object]:
        return {
            "sample_count": len(samples),
            "real_sample_count": sum(1 for sample in samples if not sample.is_synthetic),
            "synthetic_sample_count": sum(1 for sample in samples if sample.is_synthetic),
            "class_counts": {
                positive_label: sum(1 for sample in samples if sample.label == positive_label),
                negative_label: sum(1 for sample in samples if sample.label == negative_label),
            },
            "source_version_id": source_version_id,
        }

    @staticmethod
    def _copy_source_split_plan(source_version: DatasetVersion) -> dict[str, object]:
        if not source_version.split_plan:
            return {}
        return json.loads(json.dumps(source_version.split_plan))

    @classmethod
    def validate_version_integrity(cls, version: DatasetVersion) -> dict:
        errors: list[str] = []
        warnings: list[str] = []
        sample_ids = [sample.sample_id for sample in version.samples]
        sample_id_set = set(sample_ids)
        class_map = sorted({sample.label for sample in version.samples})
        class_counts = {
            label: sum(1 for sample in version.samples if sample.label == label)
            for label in class_map
        }
        expected_manifest_hash = cls.compute_manifest_hash(version.samples)

        if expected_manifest_hash != version.manifest_hash:
            errors.append("dataset version manifest_hash does not match samples")
        if sorted(version.class_map) != class_map:
            errors.append("dataset version class_map does not match samples")
        if version.stats.get("sample_count") not in {None, len(version.samples)}:
            errors.append("dataset version stats.sample_count does not match samples")
        if version.stats.get("class_counts") not in ({}, None, class_counts):
            errors.append("dataset version stats.class_counts does not match samples")

        manifest = version.manifest or {}
        if manifest:
            if manifest.get("deterministic_order") not in (None, sample_ids):
                errors.append("dataset manifest deterministic_order does not match sample ordering")
            declared_content_groups = {
                key: sorted(ids) for key, ids in manifest.get("content_groups", {}).items()
            }
            actual_content_groups: dict[str, list[str]] = {}
            for sample in version.samples:
                if not sample.content_hash:
                    continue
                actual_content_groups.setdefault(sample.content_hash, []).append(sample.sample_id)
            actual_content_groups = {key: sorted(ids) for key, ids in sorted(actual_content_groups.items())}
            if declared_content_groups and declared_content_groups != actual_content_groups:
                errors.append("dataset manifest content_groups do not match samples")
            for field in ("synthetic_sample_ids", "real_sample_ids"):
                declared_ids = set(manifest.get(field, []))
                if not declared_ids.issubset(sample_id_set):
                    errors.append(f"dataset manifest {field} references unknown sample ids")

        split_plan = version.split_plan or {}
        split_validation = SplitBalanceService.validate_split_plan(version, split_plan) if split_plan else {"ok": True, "errors": [], "warnings": []}
        if not split_validation["ok"]:
            errors.extend(split_validation["errors"])
        warnings.extend(split_validation.get("warnings", []))

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "expected_manifest_hash": expected_manifest_hash,
        }

    @staticmethod
    def _canonical_manifest_rows(samples: list[DatasetSample]) -> list[dict]:
        return [
            {
                "sample_id": sample.sample_id,
                "audio_ref": sample.audio_ref,
                "label": sample.label,
                "content_hash": sample.content_hash,
                "source_provenance": sample.source_provenance,
                "group_id": sample.group_id,
                "is_synthetic": sample.is_synthetic,
                "synthetic_provenance": sample.synthetic_provenance,
                "quality_flags": sample.quality_flags,
                "curation_state": sample.curation_state.value,
            }
            for sample in samples
        ]
