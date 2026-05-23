"""Review sample doctor for shared Foundry training pools.
Exists because review exports can mix positives, negatives, and duplicate audio in class folders.
Connects shared review samples to clean training-ready pools plus explicit quarantine reports.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from echozero.foundry.domain.review import ReviewDecisionKind
from echozero.foundry.review_samples import (
    ReviewSampleTrainingRole,
    review_sample_label_dir,
    review_sample_target_label,
)
from echozero.foundry.services.review_event_state import normalize_review_label
from echozero.foundry.services.review_sample_doctor_models import (
    DoctorAction,
    DoctorSample,
    ReviewSampleDoctorResult,
)

_AUDIO_SAMPLE_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
_RUNTIME_WINDOW_SECONDS = 0.08


class ReviewSampleDoctorService:
    """Audits shared review samples and writes a clean pool plus quarantine."""

    def audit_and_repair(
        self,
        source_root: Path,
        *,
        output_root: Path | None = None,
        labels: tuple[str, ...] | None = None,
        conflict_policy: str = "quarantine",
    ) -> ReviewSampleDoctorResult:
        """Create a cleaned review-sample pool without mutating the raw export."""
        resolved_source = source_root.expanduser().resolve()
        if not resolved_source.is_dir():
            raise ValueError(f"Review sample root not found: {resolved_source}")
        resolved_conflict_policy = self._resolve_conflict_policy(conflict_policy)
        resolved_output = self._resolve_output_root(resolved_source, output_root)
        clean_root = resolved_output / "clean"
        quarantine_root = resolved_output / "quarantine"
        self._prepare_output_root(resolved_source, resolved_output)
        clean_root.mkdir(parents=True, exist_ok=True)
        quarantine_root.mkdir(parents=True, exist_ok=True)
        manifest_lookup, manifest_stats = self._load_manifest(resolved_source / "manifest.jsonl")
        selected_labels = self._select_labels(resolved_source, labels)
        samples, unreadable_actions = self._discover_samples(
            resolved_source,
            selected_labels,
            manifest_lookup,
            quarantine_root,
        )
        actions = self._plan_actions(
            samples,
            clean_root,
            quarantine_root,
            conflict_policy=resolved_conflict_policy,
        )
        actions.extend(unreadable_actions)
        self._write_actions(actions)
        clean_manifest_path = clean_root / "manifest.jsonl"
        quarantine_manifest_path = quarantine_root / "manifest.jsonl"
        self._write_manifest(clean_manifest_path, actions, action_filter={"clean"})
        self._write_manifest(
            quarantine_manifest_path,
            actions,
            action_filter={"quarantine", "dedupe"},
        )
        report = self._build_report(
            resolved_source,
            resolved_output,
            clean_root,
            quarantine_root,
            selected_labels,
            samples,
            actions,
            manifest_stats,
            resolved_conflict_policy,
        )
        report_path = resolved_output / "doctor_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return ReviewSampleDoctorResult(
            source_root=resolved_source,
            output_root=resolved_output,
            clean_root=clean_root,
            quarantine_root=quarantine_root,
            report_path=report_path,
            report=report,
        )
    def _discover_samples(
        self,
        source_root: Path,
        labels: tuple[str, ...],
        manifest_lookup: dict[str, dict[str, Any]],
        quarantine_root: Path,
    ) -> tuple[list[DoctorSample], list[DoctorAction]]:
        samples: list[DoctorSample] = []
        unreadable_actions: list[DoctorAction] = []
        for label in labels:
            class_dir = source_root / label
            if not class_dir.is_dir():
                continue
            for audio_path in sorted(class_dir.rglob("*")):
                if not audio_path.is_file() or audio_path.suffix.lower() not in _AUDIO_SAMPLE_SUFFIXES:
                    continue
                relative_path = audio_path.relative_to(source_root).as_posix()
                manifest_row = manifest_lookup.get(relative_path)
                try:
                    duration_seconds, frames, sample_rate = self._inspect_audio(audio_path)
                    content_hash = self._compute_hash(audio_path)
                except RuntimeError as exc:
                    unreadable_actions.append(
                        DoctorAction(
                            sample=self._unreadable_sample(
                                audio_path,
                                relative_path,
                                label,
                                manifest_row,
                                str(exc),
                            ),
                            action="quarantine",
                            reason="unreadable_audio",
                            output_path=self._quarantine_path(
                                quarantine_root,
                                "unreadable_audio",
                                relative_path,
                            ),
                        )
                    )
                    continue
                decision_kind = self._decision_kind(manifest_row)
                review_polarity = self._review_polarity(decision_kind)
                training_role = self._training_role(review_polarity)
                target_label = review_sample_target_label(
                    class_label=label,
                    training_role=training_role,
                )
                samples.append(
                    DoctorSample(
                        source_path=audio_path,
                        source_relative_path=relative_path,
                        folder_label=normalize_review_label(label),
                        target_label=target_label,
                        content_hash=content_hash,
                        duration_seconds=duration_seconds,
                        frames=frames,
                        sample_rate=sample_rate,
                        manifest_row=manifest_row,
                        decision_kind=decision_kind,
                        review_polarity=review_polarity,
                        quality_flags=self._quality_flags(duration_seconds, manifest_row, review_polarity),
                    )
                )
        return samples, unreadable_actions

    def _plan_actions(
        self,
        samples: list[DoctorSample],
        clean_root: Path,
        quarantine_root: Path,
        *,
        conflict_policy: str,
    ) -> list[DoctorAction]:
        groups: dict[str, list[DoctorSample]] = defaultdict(list)
        for sample in samples:
            groups[sample.content_hash].append(sample)
        actions: list[DoctorAction] = []
        used_clean_paths: set[Path] = set()
        for content_hash in sorted(groups):
            group = sorted(groups[content_hash], key=lambda sample: sample.source_relative_path)
            target_labels = {sample.target_label for sample in group}
            if len(target_labels) > 1:
                if conflict_policy == "latest_review_wins":
                    canonical = self._latest_reviewed_sample(group)
                    canonical_path = self._clean_path(clean_root, canonical, used_clean_paths)
                    used_clean_paths.add(canonical_path)
                    actions.append(
                        DoctorAction(
                            sample=canonical,
                            action="clean",
                            reason="latest_review_wins",
                            output_path=canonical_path,
                        )
                    )
                    for sample in group:
                        if sample == canonical:
                            continue
                        actions.append(
                            DoctorAction(
                                sample=sample,
                                action="quarantine",
                                reason="conflict_superseded_by_latest_review",
                                output_path=self._quarantine_path(
                                    quarantine_root,
                                    "conflict_superseded_by_latest_review",
                                    sample.source_relative_path,
                                ),
                            )
                        )
                    continue
                for sample in group:
                    actions.append(
                        DoctorAction(
                            sample=sample,
                            action="quarantine",
                            reason="conflicting_content",
                            output_path=self._quarantine_path(
                                quarantine_root,
                                "conflicting_content",
                                sample.source_relative_path,
                            ),
                        )
                    )
                continue
            canonical = self._canonical_sample(group)
            canonical_path = self._clean_path(clean_root, canonical, used_clean_paths)
            used_clean_paths.add(canonical_path)
            actions.append(
                DoctorAction(
                    sample=canonical,
                    action="clean",
                    reason="canonical",
                    output_path=canonical_path,
                )
            )
            for duplicate in group:
                if duplicate == canonical:
                    continue
                actions.append(
                    DoctorAction(
                        sample=duplicate,
                        action="dedupe",
                        reason="duplicate_content",
                        output_path=self._quarantine_path(
                            quarantine_root,
                            "duplicate_content",
                            duplicate.source_relative_path,
                        ),
                    )
                )
        return actions

    def _write_actions(self, actions: list[DoctorAction]) -> None:
        for action in actions:
            if action.output_path is None:
                continue
            action.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(action.sample.source_path, action.output_path)

    def _write_manifest(
        self,
        manifest_path: Path,
        actions: list[DoctorAction],
        *,
        action_filter: set[str],
    ) -> None:
        rows = [
            self._manifest_row(action)
            for action in sorted(actions, key=lambda item: item.sample.source_relative_path)
            if action.action in action_filter
        ]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _build_report(
        self,
        source_root: Path,
        output_root: Path,
        clean_root: Path,
        quarantine_root: Path,
        labels: tuple[str, ...],
        samples: list[DoctorSample],
        actions: list[DoctorAction],
        manifest_stats: dict[str, Any],
        conflict_policy: str,
    ) -> dict[str, Any]:
        content_groups: dict[str, list[DoctorSample]] = defaultdict(list)
        for sample in samples:
            content_groups[sample.content_hash].append(sample)
        conflicting_groups = [
            {
                "content_hash": content_hash,
                "target_labels": sorted({sample.target_label for sample in group}),
                "source_relative_paths": [sample.source_relative_path for sample in group],
            }
            for content_hash, group in sorted(content_groups.items())
            if len({sample.target_label for sample in group}) > 1
        ]
        return {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "clean_root": str(clean_root),
            "quarantine_root": str(quarantine_root),
            "labels": list(labels),
            "source_sample_count": len(samples),
            "counts_by_folder_label": dict(sorted(Counter(sample.folder_label for sample in samples).items())),
            "counts_by_target_label": dict(sorted(Counter(sample.target_label for sample in samples).items())),
            "counts_by_training_role": dict(
                sorted(
                    Counter(self._training_role(sample.review_polarity).value for sample in samples).items()
                )
            ),
            "counts_by_decision_kind": dict(
                sorted(Counter(sample.decision_kind or "missing" for sample in samples).items())
            ),
            "counts_by_review_polarity": dict(
                sorted(Counter(sample.review_polarity or "missing" for sample in samples).items())
            ),
            "duration_buckets": self._duration_buckets(samples),
            "manifest": manifest_stats,
            "conflict_policy": conflict_policy,
            "duplicate_content_group_count": sum(1 for group in content_groups.values() if len(group) > 1),
            "conflicting_content_group_count": len(conflicting_groups),
            "conflicting_content_groups": conflicting_groups[:100],
            "action_counts": dict(sorted(Counter(action.action for action in actions).items())),
            "quarantine_reason_counts": dict(
                sorted(
                    Counter(
                        action.reason for action in actions if action.action in {"quarantine", "dedupe"}
                    ).items()
                )
            ),
            "clean_sample_count": sum(1 for action in actions if action.action == "clean"),
            "quarantined_sample_count": sum(
                1 for action in actions if action.action in {"quarantine", "dedupe"}
            ),
        }

    def _manifest_row(self, action: DoctorAction) -> dict[str, Any]:
        sample = action.sample
        return {
            "action": action.action,
            "reason": action.reason,
            "clip_path": (
                None
                if action.output_path is None
                else action.output_path.relative_to(action.output_path.parents[2]).as_posix()
            ),
            "source_clip_path": sample.source_relative_path,
            "folder_label": sample.folder_label,
            "training_role": self._training_role(sample.review_polarity).value,
            "target_label": sample.target_label,
            "export_contract": {
                "schema": "echozero.review_sample_export.v2",
                "layout": "<root>/<training_role>/<class_label>/<clip>.wav",
                "training_role_values": ["positive", "negative"],
                "negative_target_label": "other",
            },
            "content_hash": sample.content_hash,
            "duration_seconds": sample.duration_seconds,
            "frames": sample.frames,
            "sample_rate": sample.sample_rate,
            "decision_kind": sample.decision_kind,
            "review_polarity": sample.review_polarity,
            "quality_flags": list(sample.quality_flags),
            "resolved_by": action.reason if action.reason == "latest_review_wins" else None,
            "source_manifest_row": sample.manifest_row,
        }

    def _resolve_conflict_policy(self, conflict_policy: str) -> str:
        value = str(conflict_policy or "quarantine").strip().lower().replace("-", "_")
        if value not in {"quarantine", "latest_review_wins"}:
            raise ValueError(
                "Review sample doctor conflict policy must be one of: "
                "quarantine, latest_review_wins"
            )
        return value

    def _load_manifest(self, manifest_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if not manifest_path.exists():
            return {}, {"exists": False, "row_count": 0, "duplicate_clip_path_count": 0}

        lookup: dict[str, dict[str, Any]] = {}
        duplicate_clip_paths: Counter[str] = Counter()
        malformed_rows = 0
        row_count = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue
                clip_path = str(row.get("clip_path") or "").strip()
                if not clip_path:
                    malformed_rows += 1
                    continue
                if clip_path in lookup:
                    duplicate_clip_paths[clip_path] += 1
                lookup[clip_path] = row
        return lookup, {
            "exists": True,
            "row_count": row_count,
            "matched_policy": "clip_path_last_row_wins",
            "usable_clip_path_count": len(lookup),
            "duplicate_clip_path_count": len(duplicate_clip_paths),
            "malformed_row_count": malformed_rows,
        }

    def _select_labels(self, source_root: Path, labels: tuple[str, ...] | None) -> tuple[str, ...]:
        if labels:
            return tuple(normalize_review_label(label) for label in labels)
        return tuple(
            sorted(
                path.name
                for path in source_root.iterdir()
                if path.is_dir() and not path.name.startswith("_")
            )
        )

    def _resolve_output_root(self, source_root: Path, output_root: Path | None) -> Path:
        if output_root is None:
            return (source_root.parent / f"{source_root.name}_doctored").resolve()
        return output_root.expanduser().resolve()

    def _prepare_output_root(self, source_root: Path, output_root: Path) -> None:
        if output_root == source_root or source_root in output_root.parents:
            raise ValueError("Doctor output root must be outside the raw review-sample root")
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    def _inspect_audio(self, audio_path: Path) -> tuple[float, int, int]:
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
        except Exception as exc:  # pragma: no cover - exact soundfile exception varies.
            raise RuntimeError(f"Could not inspect audio: {audio_path}") from exc
        duration_seconds = float(info.frames) / float(info.samplerate)
        return duration_seconds, int(info.frames), int(info.samplerate)

    def _compute_hash(self, audio_path: Path) -> str:
        digest = hashlib.sha256()
        with audio_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _unreadable_sample(
        self,
        audio_path: Path,
        relative_path: str,
        label: str,
        manifest_row: dict[str, Any] | None,
        error_message: str,
    ) -> DoctorSample:
        decision_kind = self._decision_kind(manifest_row)
        review_polarity = self._review_polarity(decision_kind)
        return DoctorSample(
            source_path=audio_path,
            source_relative_path=relative_path,
            folder_label=normalize_review_label(label),
            target_label=review_sample_target_label(
                class_label=label,
                training_role=self._training_role(review_polarity),
            ),
            content_hash=f"unreadable:{hashlib.sha256(relative_path.encode()).hexdigest()}",
            duration_seconds=None,
            frames=None,
            sample_rate=None,
            manifest_row=manifest_row,
            decision_kind=decision_kind,
            review_polarity=review_polarity,
            quality_flags=("unreadable_audio", error_message),
        )

    def _decision_kind(self, manifest_row: dict[str, Any] | None) -> str | None:
        if manifest_row is None:
            return None
        decision_kind = str(manifest_row.get("decision_kind") or "").strip().lower()
        return decision_kind or None

    def _review_polarity(self, decision_kind: str | None) -> str | None:
        if decision_kind is None:
            return None
        if decision_kind == ReviewDecisionKind.REJECTED.value:
            return "negative"
        return "positive"

    def _training_role(self, review_polarity: str | None) -> ReviewSampleTrainingRole:
        if review_polarity == "negative":
            return ReviewSampleTrainingRole.NEGATIVE
        return ReviewSampleTrainingRole.POSITIVE

    def _quality_flags(
        self,
        duration_seconds: float | None,
        manifest_row: dict[str, Any] | None,
        review_polarity: str | None,
    ) -> tuple[str, ...]:
        flags: list[str] = []
        if manifest_row is None:
            flags.append("missing_manifest")
        if review_polarity:
            flags.append(f"review_{review_polarity}")
        if duration_seconds is not None and abs(duration_seconds - _RUNTIME_WINDOW_SECONDS) <= 0.002:
            flags.append("runtime_window_80ms")
        if duration_seconds is not None and duration_seconds < 0.12:
            flags.append("very_short_audio")
        return tuple(flags)

    def _canonical_sample(self, samples: list[DoctorSample]) -> DoctorSample:
        return sorted(
            samples,
            key=lambda sample: (
                sample.review_polarity != "positive",
                sample.manifest_row is None,
                sample.source_relative_path,
            ),
        )[0]

    def _latest_reviewed_sample(self, samples: list[DoctorSample]) -> DoctorSample:
        return sorted(
            samples,
            key=lambda sample: (
                self._reviewed_at_sort_key(sample),
                sample.manifest_row is not None,
                sample.source_relative_path,
            ),
            reverse=True,
        )[0]

    def _reviewed_at_sort_key(self, sample: DoctorSample) -> str:
        if not sample.manifest_row:
            return ""
        reviewed_at = str(sample.manifest_row.get("reviewed_at") or "").strip()
        if reviewed_at:
            return reviewed_at
        source_row = sample.manifest_row.get("source_manifest_row")
        if isinstance(source_row, dict):
            return str(source_row.get("reviewed_at") or "").strip()
        return ""

    def _clean_path(
        self,
        clean_root: Path,
        sample: DoctorSample,
        used_paths: set[Path],
    ) -> Path:
        safe_stem = self._safe_stem(sample.source_path.stem)
        label_dir = review_sample_label_dir(
            clean_root,
            class_label=sample.folder_label,
            training_role=self._training_role(sample.review_polarity),
        )
        candidate = (
            label_dir
            / f"{sample.content_hash[:16]}_{safe_stem}{sample.source_path.suffix.lower()}"
        )
        if candidate not in used_paths:
            return candidate
        return (
            label_dir
            / f"{sample.content_hash[:16]}_{hashlib.sha1(sample.source_relative_path.encode()).hexdigest()[:8]}_{safe_stem}{sample.source_path.suffix.lower()}"
        )

    def _quarantine_path(self, quarantine_root: Path, reason: str, source_relative_path: str) -> Path:
        safe_relative = Path(*[self._safe_stem(part) for part in Path(source_relative_path).parts])
        return quarantine_root / reason / safe_relative

    def _safe_stem(self, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
        return safe or "sample"

    def _duration_buckets(self, samples: list[DoctorSample]) -> dict[str, int]:
        buckets: Counter[str] = Counter()
        for sample in samples:
            duration = sample.duration_seconds
            if duration is None:
                buckets["unreadable"] += 1
            elif abs(duration - _RUNTIME_WINDOW_SECONDS) <= 0.002:
                buckets["runtime_window_80ms"] += 1
            elif duration < 0.12:
                buckets["under_120ms"] += 1
            elif duration < 0.5:
                buckets["120ms_to_500ms"] += 1
            elif duration < 1.25:
                buckets["500ms_to_1250ms"] += 1
            else:
                buckets["over_1250ms"] += 1
        return dict(sorted(buckets.items()))
