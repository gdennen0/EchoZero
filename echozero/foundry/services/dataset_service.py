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

    def create_dataset(self, name: str, source_kind: str = "folder_import") -> Dataset:
        dataset = Dataset(
            id=f"ds_{uuid4().hex[:12]}",
            name=name,
            source_kind=source_kind,
        )
        return self._datasets.save(dataset)

    def ingest_from_folder(
        self,
        dataset_id: str,
        folder_path: str | Path,
        *,
        sample_rate: int = 22050,
        audio_standard: str = "mono_wav_pcm16",
    ) -> DatasetVersion:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")

        base = Path(folder_path)
        if not base.exists() or not base.is_dir():
            raise ValueError(f"Dataset folder not found: {base}")

        samples: list[DatasetSample] = []
        for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            label = class_dir.name
            for file in sorted(class_dir.rglob("*")):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}:
                    continue
                rel = file.resolve().as_posix()
                sample_id = f"sm_{uuid4().hex[:12]}"
                samples.append(
                    DatasetSample(
                        sample_id=sample_id,
                        audio_ref=rel,
                        label=label,
                        source_provenance={"kind": "folder_import", "path": rel},
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
                "source_provenance": s.source_provenance,
                "quality_flags": s.quality_flags,
                "curation_state": s.curation_state.value,
            }
            for s in samples
        ]
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        class_map = sorted({s.label for s in samples})

        stats = {
            "sample_count": len(samples),
            "class_counts": {
                label: sum(1 for s in samples if s.label == label)
                for label in class_map
            },
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
                    source_provenance=sample.source_provenance,
                    quality_flags=sample.quality_flags,
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
                "curation_state": s.curation_state.value,
            }
            for s in accepted
        ]
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()

        next_version = DatasetVersion(
            id=f"dsv_{uuid4().hex[:12]}",
            dataset_id=version.dataset_id,
            version=next_version_num,
            manifest_hash=manifest_hash,
            sample_rate=version.sample_rate,
            audio_standard=version.audio_standard,
            class_map=sorted({s.label for s in accepted}),
            samples=accepted,
            split_plan=version.split_plan,
            balance_plan=version.balance_plan,
            stats={
                "sample_count": len(accepted),
                "class_counts": {
                    label: sum(1 for s in accepted if s.label == label)
                    for label in sorted({s.label for s in accepted})
                },
                "curated_from": version.id,
            },
        )
        return self._versions.save(next_version)
