"""Event-span sample materialization for model evolution.
Exists because reviewed timeline Event bounds are the canonical training clip bounds.
Connects fixed Events to deterministic Foundry DatasetSample records.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from echozero.foundry.domain import CurationState, DatasetSample, DatasetVersion
from echozero.foundry.model_evolution.truth import FixedEventTruth
from echozero.foundry.services.audio_source_validation import inspect_audio_source
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService


@dataclass(frozen=True, slots=True)
class RuntimeWindowPolicy:
    """Compatibility policy that preserves fixed Event start/end bounds."""

    name: str = "event_span_v1"
    window_seconds: float | None = None
    sample_rate: int = 22050
    audio_standard: str = "mono_wav_pcm16"
    anchor_position: str = "event_bounds"

    def window_for(self, truth: FixedEventTruth) -> tuple[float, float]:
        """Return the reviewed Event start/end seconds for one truth record."""
        start = max(0.0, float(truth.event_start_seconds))
        end = max(start, float(truth.event_end_seconds))
        return start, end

    def to_payload(self) -> dict[str, object]:
        """Serialize the window policy into dataset metadata."""
        return {
            "name": self.name,
            "window_seconds": self.window_seconds,
            "sample_rate": self.sample_rate,
            "audio_standard": self.audio_standard,
            "anchor_position": self.anchor_position,
            "kind": "event_span",
        }


@dataclass(frozen=True, slots=True)
class MaterializedRuntimeSample:
    """One event-span sample and its originating fixed Event truth."""

    truth: FixedEventTruth
    sample: DatasetSample
    window_start_seconds: float
    window_end_seconds: float
    clip_path: Path


class RuntimeWindowMaterializer:
    """Materializes fixed Event truth into a Foundry dataset version."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_service: DatasetService,
        clip_service: ReviewAudioClipService | None = None,
    ) -> None:
        self._root = Path(root)
        self._datasets = dataset_service
        self._clip_service = clip_service or ReviewAudioClipService()

    def materialize_dataset(
        self,
        truths: list[FixedEventTruth],
        *,
        dataset_name: str,
        policy: RuntimeWindowPolicy | None = None,
        source_scope: str = "fixed_events",
    ) -> DatasetVersion:
        """Create one dataset version from event-span fixed Event samples."""
        resolved_policy = policy or RuntimeWindowPolicy()
        normalized_truths = self._normalize_truths(truths)
        if not normalized_truths:
            raise ValueError("At least one fixed Event truth is required.")

        materialized = [
            self._materialize_truth(truth, policy=resolved_policy)
            for truth in normalized_truths
        ]
        samples = [entry.sample for entry in materialized]
        dataset = self._datasets.create_dataset(
            dataset_name,
            source_kind="model_evolution_runtime_samples",
            metadata={
                "schema": "foundry.model_evolution_dataset.v1",
                "source_scope": source_scope,
                "window_policy": resolved_policy.to_payload(),
            },
        )
        class_counts = Counter(sample.label for sample in samples)
        manifest = {
            "schema": "foundry.model_evolution_runtime_samples_manifest.v1",
            "source_kind": "model_evolution_runtime_samples",
            "source_scope": source_scope,
            "window_policy": resolved_policy.to_payload(),
            "deterministic_order": [sample.sample_id for sample in samples],
            "content_hash_algorithm": "sha256",
            "truth_ids": [entry.truth.truth_id for entry in materialized],
                "sample_windows": [
                {
                    "sample_id": entry.sample.sample_id,
                    "truth_id": entry.truth.truth_id,
                    "event_start_seconds": entry.truth.event_start_seconds,
                    "event_end_seconds": entry.truth.event_end_seconds,
                    "event_duration_ms": entry.truth.event_duration_ms,
                    "window_start_seconds": entry.window_start_seconds,
                    "window_end_seconds": entry.window_end_seconds,
                    "clip_path": str(entry.clip_path),
                }
                for entry in materialized
            ],
        }
        return self._datasets.create_version_from_samples(
            dataset.id,
            samples=samples,
            sample_rate=resolved_policy.sample_rate,
            audio_standard=resolved_policy.audio_standard,
            manifest=manifest,
            stats={
                "sample_count": len(samples),
                "real_sample_count": len(samples),
                "synthetic_sample_count": 0,
                "class_counts": dict(sorted(class_counts.items())),
                "source_scope": source_scope,
                "window_policy": resolved_policy.to_payload(),
            },
            lineage={
                "kind": "model_evolution_runtime_samples",
                "source_scope": source_scope,
                "truth_ids": [truth.truth_id for truth in normalized_truths],
                "window_policy": resolved_policy.to_payload(),
            },
        )

    def _materialize_truth(
        self,
        truth: FixedEventTruth,
        *,
        policy: RuntimeWindowPolicy,
    ) -> MaterializedRuntimeSample:
        source_path = truth.source_audio_path.expanduser().resolve()
        window_start, window_end = policy.window_for(truth)
        clip_path = self._clip_service.materialize_event_clip(
            source_audio_path=source_path,
            clip_cache_dir=self._root / "foundry" / "cache" / "model_evolution_runtime_samples",
            clip_stem=self._clip_stem(truth),
            start_seconds=window_start,
            end_seconds=window_end,
        )
        if clip_path is None:
            raise ValueError(
                "Could not materialize runtime sample for fixed Event truth "
                f"'{truth.truth_id}' from {source_path}"
            )
        metadata = inspect_audio_source(clip_path)
        content_hash = self._content_hash(clip_path)
        sample_id = self._sample_id(
            truth=truth,
            policy=policy,
            window_start=window_start,
            window_end=window_end,
            content_hash=content_hash,
        )
        provenance = truth.provenance()
        provenance.update(
            {
                "source_kind": "fixed_event_truth",
                "runtime_window_start_seconds": window_start,
                "runtime_window_end_seconds": window_end,
                "runtime_window_duration_ms": metadata.duration_ms,
                "sample_window_kind": "event_span",
                "window_policy": policy.to_payload(),
                "materialized_clip_path": str(clip_path.resolve()),
            }
        )
        sample = DatasetSample(
            sample_id=sample_id,
            audio_ref=str(clip_path.resolve()),
            label=truth.normalized_label,
            duration_ms=metadata.duration_ms,
            content_hash=content_hash,
            source_provenance=provenance,
            group_id=f"truth:{truth.truth_id}",
            is_synthetic=False,
            synthetic_provenance={},
            quality_flags=["reviewed", "fixed_event_truth", "event_span"],
            curation_state=CurationState.ACCEPTED,
        )
        return MaterializedRuntimeSample(
            truth=truth,
            sample=sample,
            window_start_seconds=window_start,
            window_end_seconds=window_end,
            clip_path=clip_path.resolve(),
        )

    @staticmethod
    def _normalize_truths(truths: list[FixedEventTruth]) -> list[FixedEventTruth]:
        return sorted(
            (truth for truth in truths if truth.normalized_label),
            key=lambda truth: (
                truth.normalized_label,
                str(truth.source_audio_path.expanduser()),
                float(truth.event_start_seconds),
                float(truth.event_end_seconds),
                truth.truth_id,
            ),
        )

    @staticmethod
    def _clip_stem(truth: FixedEventTruth) -> str:
        label = truth.normalized_label or "event"
        safe_truth_id = "".join(
            character if character.isalnum() else "_" for character in truth.truth_id
        )
        return f"{label}_{safe_truth_id}"

    @staticmethod
    def _sample_id(
        *,
        truth: FixedEventTruth,
        policy: RuntimeWindowPolicy,
        window_start: float,
        window_end: float,
        content_hash: str,
    ) -> str:
        digest = hashlib.sha256(
            (
                f"{truth.truth_id}|{truth.normalized_label}|{truth.source_audio_path.expanduser()}|"
                f"{truth.event_start_seconds:.9f}|{truth.event_end_seconds:.9f}|"
                f"{window_start:.9f}|{window_end:.9f}|{policy.name}|{content_hash}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"mesm_{digest}"

    @staticmethod
    def _content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
