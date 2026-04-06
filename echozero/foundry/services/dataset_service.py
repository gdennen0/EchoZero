from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import CurationState, Dataset, DatasetSample, DatasetVersion
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository


class DatasetService:
    def __init__(
        self,
        root: Path,
        dataset_repo: DatasetRepository | None = None,
        version_repo: DatasetVersionRepository | None = None,
    ):
        self._root = root
        self._datasets = dataset_repo or DatasetRepository(root)
        self._versions = version_repo or DatasetVersionRepository(root)

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
        for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            label = class_dir.name
            for file in sorted(class_dir.rglob("*")):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}:
                    continue
                rel_path = file.relative_to(base).as_posix()
                rel = file.resolve().as_posix()
                content_hash = hashlib.sha256(file.read_bytes()).hexdigest()
                sample_id = f"sm_{uuid4().hex[:12]}"
                content_groups.setdefault(content_hash, []).append(sample_id)
                samples.append(
                    DatasetSample(
                        sample_id=sample_id,
                        audio_ref=rel,
                        label=label,
                        content_hash=content_hash,
                        source_provenance={
                            "kind": "folder_import",
                            "path": rel,
                            "source_root": str(base.resolve().as_posix()),
                            "relative_path": rel_path,
                            "filename": file.name,
                            "label_from_path": label,
                        },
                        curation_state=CurationState.UNKNOWN,
                    )
                )

        if not samples:
            raise ValueError(f"No audio samples found in dataset folder: {base}")

        existing = self._versions.list_for_dataset(dataset_id)
        next_version_num = (existing[-1].version + 1) if existing else 1

        manifest = [
            {
                "sample_id": s.sample_id,
                "audio_ref": s.audio_ref,
                "label": s.label,
                "content_hash": s.content_hash,
                "source_provenance": s.source_provenance,
                "quality_flags": s.quality_flags,
                "split_assignment": s.split_assignment,
                "curation_state": s.curation_state.value,
            }
            for s in samples
        ]
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
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
            "class_counts": {
                label: sum(1 for s in samples if s.label == label)
                for label in class_map
            },
            "duplicate_content_hashes": sum(1 for ids in content_groups.values() if len(ids) > 1),
        }
        dataset_manifest = {
            "schema": "foundry.dataset_manifest.v1",
            "source_kind": dataset.source_kind,
            "source_ref": dataset.source_ref,
            "ingest_root": str(base.resolve().as_posix()),
            "deterministic_order": [s.sample_id for s in samples],
            "content_hash_algorithm": "sha256",
            "content_groups": {key: sorted(ids) for key, ids in sorted(content_groups.items())},
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
                    quality_flags=sample.quality_flags,
                    split_assignment=sample.split_assignment,
                    curation_state=state,
                )
            )

        accepted = [s for s in updated_samples if s.curation_state != CurationState.REJECTED]
        existing = self._versions.list_for_dataset(version.dataset_id)
        next_version_num = (existing[-1].version + 1) if existing else (version.version + 1)

        manifest = [
            {
                "sample_id": s.sample_id,
                "label": s.label,
                "content_hash": s.content_hash,
                "split_assignment": s.split_assignment,
                "curation_state": s.curation_state.value,
            }
            for s in accepted
        ]
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
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
                "class_counts": {
                    label: sum(1 for s in accepted if s.label == label)
                    for label in sorted({s.label for s in accepted})
                },
                "curated_from": version.id,
            },
            lineage={"source_version_id": version.id, "kind": "curation"},
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
        return curated_manifest
