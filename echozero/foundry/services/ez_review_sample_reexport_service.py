"""EZ review-sample re-export from project bundles.
Exists because legacy shared sample folders can drift from reviewed project truth.
Connects archived .ez timeline reviews to the canonical shared review-sample layout.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echozero.foundry.domain.review import ReviewDecisionKind
from echozero.foundry.review_samples import (
    review_sample_label_dir,
    review_sample_target_label,
    review_sample_training_role,
)
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService
from echozero.foundry.services.review_event_state import normalize_review_label
from echozero.processors.drum_event_span import estimate_drum_event_span

_MARKER_DURATION_SECONDS = 0.08


@dataclass(frozen=True, slots=True)
class EzReviewSampleReexportResult:
    """Summarizes a project-bundle review sample re-export."""

    output_root: Path
    manifest_path: Path
    report_path: Path
    report: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe result payload for CLI callers."""
        return {
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "report_path": str(self.report_path),
            "report": self.report,
        }


@dataclass(frozen=True, slots=True)
class ReviewedEventClip:
    """One reviewed event resolved to bundled audio and canonical training metadata."""

    project_path: Path
    source_audio_member: str
    extracted_source_path: Path
    event_id: str
    class_label: str
    decision_kind: str
    review_outcome: str | None
    reviewed_at: str | None
    start_seconds: float
    end_seconds: float
    source_take_id: str
    bounds_policy: str
    source_kind: str
    span_estimate: dict[str, Any] | None


class EzReviewSampleReexportService:
    """Rebuilds canonical shared review samples from reviewed events in .ez bundles."""

    def __init__(self, *, clip_service: ReviewAudioClipService | None = None) -> None:
        self._clip_service = clip_service or ReviewAudioClipService()
        self._audio_cache: dict[tuple[Path, int, int], tuple[np.ndarray, int]] = {}

    def reexport(
        self,
        project_paths: list[Path],
        *,
        output_root: Path,
        labels: tuple[str, ...] | None = None,
        include_promoted_events: bool = False,
        overwrite: bool = False,
    ) -> EzReviewSampleReexportResult:
        """Write one canonical sample pool from reviewed events in .ez project files."""
        resolved_projects = [path.expanduser().resolve() for path in project_paths]
        if not resolved_projects:
            raise ValueError("At least one .ez project path is required.")
        normalized_labels = (
            tuple(normalize_review_label(label) for label in labels) if labels else None
        )
        resolved_output = output_root.expanduser().resolve()
        if resolved_output.exists():
            if not overwrite:
                raise ValueError(f"Output root already exists: {resolved_output}")
            shutil.rmtree(resolved_output)
        resolved_output.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        for project_path in resolved_projects:
            if not project_path.is_file():
                skipped["missing_project"] += 1
                continue
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                rows.extend(
                    self._reexport_project(
                        project_path,
                        output_root=resolved_output,
                        temp_root=temp_root,
                        labels=normalized_labels,
                        include_promoted_events=include_promoted_events,
                        skipped=skipped,
                    )
                )

        raw_row_count = len(rows)
        rows, dedupe_stats = self._dedupe_rows(rows, output_root=resolved_output)
        rows = sorted(
            rows,
            key=lambda row: (
                str(row["class_label"]),
                str(row["training_role"]),
                str(row["source_project_path"]),
                float(row["event_start_seconds"]),
                str(row["event_id"]),
            ),
        )
        manifest_path = resolved_output / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = self._build_report(
            output_root=resolved_output,
            project_paths=resolved_projects,
            rows=rows,
            skipped=skipped,
            raw_row_count=raw_row_count,
            dedupe_stats=dedupe_stats,
            include_promoted_events=include_promoted_events,
        )
        report_path = resolved_output / "reexport_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return EzReviewSampleReexportResult(
            output_root=resolved_output,
            manifest_path=manifest_path,
            report_path=report_path,
            report=report,
        )

    def _reexport_project(
        self,
        project_path: Path,
        *,
        output_root: Path,
        temp_root: Path,
        labels: tuple[str, ...] | None,
        include_promoted_events: bool,
        skipped: Counter[str],
    ) -> list[dict[str, Any]]:
        with zipfile.ZipFile(project_path) as archive:
            db_path = temp_root / "project.db"
            try:
                db_path.write_bytes(archive.read("project.db"))
            except KeyError:
                skipped["missing_project_db"] += 1
                return []
            source_members = self._source_audio_members(db_path)
            rows: list[dict[str, Any]] = []
            for clip in self._reviewed_clips(
                db_path,
                project_path=project_path,
                archive=archive,
                source_members=source_members,
                temp_root=temp_root,
                labels=labels,
                include_promoted_events=include_promoted_events,
                skipped=skipped,
            ):
                row = self._materialize_clip(clip, output_root=output_root)
                if row is None:
                    skipped["clip_materialization_failed"] += 1
                    continue
                rows.append(row)
            return rows

    def _source_audio_members(self, db_path: Path) -> dict[str, str]:
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                """
                select event_payload.payload_json, source_payload.payload_json
                from object_contents event_payload
                left join object_contents source_payload
                  on source_payload.id = json_extract(event_payload.source_ref_json, '$.content_id')
                where event_payload.content_kind = 'event_set'
                """
            ).fetchall()
        finally:
            connection.close()
        source_members: dict[str, str] = {}
        for event_payload_text, source_payload_text in rows:
            event_payload = self._json_dict(event_payload_text)
            source_payload = self._json_dict(source_payload_text)
            take_id = str(event_payload.get("take_id") or "").strip()
            audio_file = str(source_payload.get("audio_file") or "").strip()
            if take_id and audio_file:
                source_members[take_id] = audio_file
        return source_members

    def _reviewed_clips(
        self,
        db_path: Path,
        *,
        project_path: Path,
        archive: zipfile.ZipFile,
        source_members: dict[str, str],
        temp_root: Path,
        labels: tuple[str, ...] | None,
        include_promoted_events: bool,
        skipped: Counter[str],
    ) -> list[ReviewedEventClip]:
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute("select id, data_json from takes").fetchall()
        finally:
            connection.close()
        clips: list[ReviewedEventClip] = []
        extracted_sources: dict[str, Path] = {}
        for take_id, data_text in rows:
            source_member = source_members.get(str(take_id))
            if not source_member:
                continue
            source_path = self._extract_source_audio(
                archive,
                member=source_member,
                temp_root=temp_root,
                extracted_sources=extracted_sources,
                skipped=skipped,
            )
            if source_path is None:
                continue
            for event in self._events_from_take(data_text):
                review = self._review_payload(event)
                source_kind = "reviewed_event"
                if review is None:
                    if not include_promoted_events or not self._is_promoted_event(event):
                        continue
                    review = {
                        "decision_kind": ReviewDecisionKind.VERIFIED.value,
                        "review_outcome": "correct",
                    }
                    source_kind = "promoted_event"
                else:
                    review = dict(review)
                    review.setdefault("review_outcome", "correct")
                    review.setdefault("decision_kind", ReviewDecisionKind.VERIFIED.value)
                if not review:
                    continue
                decision_kind = str(review.get("decision_kind") or "").strip().lower()
                if decision_kind not in {
                    ReviewDecisionKind.VERIFIED.value,
                    ReviewDecisionKind.REJECTED.value,
                    ReviewDecisionKind.MISSED_EVENT_ADDED.value,
                    ReviewDecisionKind.RELABELED.value,
                    ReviewDecisionKind.BOUNDARY_CORRECTED.value,
                }:
                    skipped["unsupported_decision_kind"] += 1
                    continue
                class_label = self._class_label(event, review)
                if labels is not None and class_label not in labels:
                    continue
                start_seconds, end_seconds, bounds_policy, span_estimate = self._event_bounds(
                    event,
                    review,
                    source_path=source_path,
                )
                if end_seconds <= start_seconds:
                    skipped["empty_event_window"] += 1
                    continue
                clips.append(
                    ReviewedEventClip(
                        project_path=project_path,
                        source_audio_member=source_member,
                        extracted_source_path=source_path,
                        event_id=str(event.get("id") or ""),
                        class_label=class_label,
                        decision_kind=decision_kind,
                        review_outcome=self._optional_text(review.get("review_outcome")),
                        reviewed_at=self._optional_text(review.get("reviewed_at")),
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        source_take_id=str(take_id),
                        bounds_policy=bounds_policy,
                        source_kind=source_kind,
                        span_estimate=span_estimate,
                    )
                )
        return clips

    def _materialize_clip(
        self,
        clip: ReviewedEventClip,
        *,
        output_root: Path,
    ) -> dict[str, Any] | None:
        decision_kind = ReviewDecisionKind(clip.decision_kind)
        training_role = review_sample_training_role(decision_kind)
        class_dir = review_sample_label_dir(
            output_root,
            class_label=clip.class_label,
            training_role=training_role,
        )
        clip_path = self._clip_service.materialize_event_clip(
            source_audio_path=clip.extracted_source_path,
            clip_cache_dir=class_dir,
            clip_stem=self._clip_stem(clip),
            start_seconds=clip.start_seconds,
            end_seconds=clip.end_seconds,
        )
        if clip_path is None:
            return None
        return {
            "schema": "echozero.review_sample_reexport.v1",
            "source_project_path": str(clip.project_path),
            "source_audio_member": clip.source_audio_member,
            "source_take_id": clip.source_take_id,
            "event_id": clip.event_id,
            "class_label": clip.class_label,
            "training_role": training_role.value,
            "target_label": review_sample_target_label(
                class_label=clip.class_label,
                training_role=training_role,
            ),
            "decision_kind": clip.decision_kind,
            "review_outcome": clip.review_outcome,
            "reviewed_at": clip.reviewed_at,
            "source_kind": clip.source_kind,
            "event_start_seconds": clip.start_seconds,
            "event_end_seconds": clip.end_seconds,
            "event_duration_seconds": clip.end_seconds - clip.start_seconds,
            "sample_window_policy": {
                "schema": "echozero.review_sample_reexport_window.v1",
                "kind": clip.bounds_policy,
                "anchor": "event_start",
                "span_estimate": clip.span_estimate,
            },
            "clip_path": clip_path.relative_to(output_root).as_posix(),
            "content_hash": self._content_hash(clip_path),
            "export_contract": {
                "schema": "echozero.review_sample_export.v2",
                "layout": "<root>/<training_role>/<class_label>/<clip>.wav",
                "training_role_values": ["positive", "negative"],
                "negative_target_label": "other",
            },
        }

    def _extract_source_audio(
        self,
        archive: zipfile.ZipFile,
        *,
        member: str,
        temp_root: Path,
        extracted_sources: dict[str, Path],
        skipped: Counter[str],
    ) -> Path | None:
        cached = extracted_sources.get(member)
        if cached is not None:
            return cached
        try:
            payload = archive.read(member)
        except KeyError:
            skipped["missing_source_audio_member"] += 1
            return None
        path = temp_root / "sources" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        extracted_sources[member] = path
        return path

    def _events_from_take(self, data_text: str | None) -> list[dict[str, Any]]:
        data = self._json_dict(data_text)
        events: list[dict[str, Any]] = []
        for layer in data.get("layers", []):
            if not isinstance(layer, dict):
                continue
            for event in layer.get("events", []):
                if isinstance(event, dict):
                    events.append(event)
        return events

    def _review_payload(self, event: dict[str, Any]) -> dict[str, Any] | None:
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            return None
        review = metadata.get("review")
        return review if isinstance(review, dict) else None

    def _is_promoted_event(self, event: dict[str, Any]) -> bool:
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            return True
        review = metadata.get("review")
        if isinstance(review, dict):
            review_state = str(review.get("promotion_state") or "").strip().lower()
            if review_state in {"promoted", "demoted"}:
                return review_state == "promoted"
        detection = metadata.get("detection")
        if isinstance(detection, dict):
            detection_state = str(detection.get("promotion_state") or "").strip().lower()
            if detection_state in {"promoted", "demoted"}:
                return detection_state == "promoted"
            threshold_passed = detection.get("threshold_passed")
            if isinstance(threshold_passed, bool):
                return threshold_passed
        legacy_state = str(metadata.get("promotion_state") or "").strip().lower()
        if legacy_state in {"promoted", "demoted"}:
            return legacy_state == "promoted"
        return True

    def _class_label(self, event: dict[str, Any], review: dict[str, Any]) -> str:
        label = (
            review.get("corrected_label")
            or review.get("original_label")
            or self._classifications(event).get("class")
            or self._classifications(event).get("label")
            or "event"
        )
        return normalize_review_label(label)

    def _event_bounds(
        self,
        event: dict[str, Any],
        review: dict[str, Any],
        *,
        source_path: Path,
    ) -> tuple[float, float, str, dict[str, Any] | None]:
        start = review.get("corrected_start_ms")
        end = review.get("corrected_end_ms")
        if start is not None and end is not None:
            return (
                max(0.0, float(start) / 1000.0),
                max(0.0, float(end) / 1000.0),
                "corrected_event_bounds",
                None,
            )
        event_start = float(event.get("time") or event.get("start") or 0.0)
        duration = float(event.get("duration") or 0.0)
        if duration <= _MARKER_DURATION_SECONDS:
            span_estimate = self._estimate_event_tail_duration(
                source_path=source_path,
                event_start=event_start,
            )
            if (
                span_estimate["consensus_method"] == "agreement"
                and span_estimate["duration_seconds"] > duration
            ):
                duration = float(span_estimate["duration_seconds"])
                return (
                    max(0.0, event_start),
                    max(0.0, event_start + duration),
                    "estimated_audio_tail",
                    span_estimate,
                )
        return max(0.0, event_start), max(0.0, event_start + duration), "event_bounds", None

    def _estimate_event_tail_duration(
        self,
        *,
        source_path: Path,
        event_start: float,
    ) -> dict[str, Any]:
        samples, sample_rate = self._read_audio(source_path)
        estimate = estimate_drum_event_span(
            audio=np.asarray(samples),
            onset_seconds=event_start,
            sample_rate=sample_rate,
        )
        return {
            "schema": "echozero.drum_event_span_estimate.v1",
            "duration_seconds": round(estimate.duration_seconds, 6),
            "consensus_method": estimate.consensus_method,
            "agreement_seconds": estimate.agreement_seconds,
            "method_durations": {
                method: round(duration, 6)
                for method, duration in estimate.method_durations.items()
            },
        }

    def _read_audio(self, path: Path) -> tuple[np.ndarray, int]:
        source = path.expanduser().resolve()
        stat = source.stat()
        cache_key = (source, int(stat.st_mtime_ns), int(stat.st_size))
        cached = self._audio_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "EZ review sample re-export requires soundfile. Install with: pip install soundfile"
            ) from exc
        samples, sample_rate = sf.read(str(source), always_2d=False, dtype="float32")
        payload = (np.asarray(samples, dtype=np.float32), int(sample_rate))
        self._audio_cache[cache_key] = payload
        return payload

    def _build_report(
        self,
        *,
        output_root: Path,
        project_paths: list[Path],
        rows: list[dict[str, Any]],
        skipped: Counter[str],
        raw_row_count: int,
        dedupe_stats: dict[str, int],
        include_promoted_events: bool,
    ) -> dict[str, Any]:
        return {
            "schema": "echozero.review_sample_reexport_report.v1",
            "output_root": str(output_root),
            "project_paths": [str(path) for path in project_paths],
            "raw_exported_sample_count": raw_row_count,
            "exported_sample_count": len(rows),
            "include_promoted_events": include_promoted_events,
            "content_resolution_policy": "latest_reviewed_row_per_content_hash",
            "content_resolution_counts": dict(sorted(dedupe_stats.items())),
            "counts_by_class_label": dict(
                sorted(Counter(str(row["class_label"]) for row in rows).items())
            ),
            "counts_by_training_role": dict(
                sorted(Counter(str(row["training_role"]) for row in rows).items())
            ),
            "counts_by_target_label": dict(
                sorted(Counter(str(row["target_label"]) for row in rows).items())
            ),
            "counts_by_decision_kind": dict(
                sorted(Counter(str(row["decision_kind"]) for row in rows).items())
            ),
            "skipped_counts": dict(sorted(skipped.items())),
        }

    def _dedupe_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        output_root: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["content_hash"]), []).append(row)
        kept: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()
        for group in grouped.values():
            if len(group) == 1:
                kept.append(group[0])
                stats["unique_content"] += 1
                continue
            target_labels = {str(row["target_label"]) for row in group}
            winner = self._latest_reviewed_row(group)
            kept.append(winner)
            if len(target_labels) > 1:
                stats["conflicting_content_resolved"] += len(group) - 1
            else:
                stats["duplicate_content_dropped"] += len(group) - 1
            for row in group:
                if row is winner:
                    continue
                self._remove_clip(output_root / str(row["clip_path"]))
        return kept, dict(stats)

    def _latest_reviewed_row(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return sorted(
            rows,
            key=lambda row: (
                str(row.get("reviewed_at") or ""),
                str(row.get("source_project_path") or ""),
                str(row.get("event_id") or ""),
                str(row.get("clip_path") or ""),
            ),
            reverse=True,
        )[0]

    def _remove_clip(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _classifications(event: dict[str, Any]) -> dict[str, Any]:
        classifications = event.get("classifications")
        return classifications if isinstance(classifications, dict) else {}

    @staticmethod
    def _json_dict(text: str | None) -> dict[str, Any]:
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _optional_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _clip_stem(clip: ReviewedEventClip) -> str:
        payload = (
            f"{clip.project_path}|{clip.source_take_id}|{clip.event_id}|"
            f"{clip.decision_kind}|{clip.class_label}|{clip.start_seconds:.9f}|"
            f"{clip.end_seconds:.9f}"
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"ez_reexport_{clip.class_label}_{digest}"

    @staticmethod
    def _content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
