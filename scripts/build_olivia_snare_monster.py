"""
Olivia snare monster builder: create and promote a specialized snare runtime bundle.
Exists to make the Olivia-weighted Foundry build repeatable instead of a one-off shell session.
Connects EZ project event truth, organized samples, Foundry training, and runtime bundle promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf

from echozero.foundry import FoundryApp
from echozero.foundry.domain import CurationState, Dataset, DatasetSample, DatasetVersion
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository, migrate_foundry_state
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.split_balance_service import SplitBalanceService
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_index import load_binary_drum_bundle_index
from echozero.runtime_models.loader import (
    build_feature_tensor,
    load_runtime_model,
    predict_probabilities,
)


PROJECT_PATH = Path("/Users/march/Desktop/Olivia/Olivia Scratch 3.ez")
SAMPLES_ROOT = Path("/Users/march/Desktop/OrginizedSamples")
WORKSPACE_ROOT = Path("/Users/march/.echozero/foundry_workspaces/olivia_snare_monster_20260501")
DATASET_AUDIO_ROOT = Path("/Users/march/.echozero/foundry_datasets/olivia_snare_monster_20260501")
REPORT_PATH = WORKSPACE_ROOT / "reports" / "olivia_snare_monster_report.json"
BUNDLE_NAME = "binary-drum-snare-olivia-monster-20260501"
POSITIVE_LABEL = "snare"
NEGATIVE_LABEL = "other"
EVENT_CLIP_SECONDS = 1.0
OLIVIA_TRAIN_DUPLICATES = 3
TRAINING_EPOCHS = 8
NEGATIVE_LIBRARY_LIMIT = 7000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}


@dataclass(frozen=True, slots=True)
class EventClip:
    """One project event clip ready to become a Foundry sample."""

    label: str
    source_audio_ref: str
    event_id: str
    song_title: str
    time_seconds: float
    audio_path: Path
    content_hash: str
    group_id: str


@dataclass(frozen=True, slots=True)
class RuntimeScoreSet:
    """Score summary for one runtime model over Olivia event truth."""

    name: str
    model_path: Path
    positive_scores: list[float]
    negative_scores: list[float]

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "modelPath": str(self.model_path),
            "positive": summarize_scores(self.positive_scores),
            "negative": summarize_scores(self.negative_scores),
        }


def parse_args() -> argparse.Namespace:
    """Parse the command-line options for the build."""
    parser = argparse.ArgumentParser(description="Build the Olivia-weighted monster snare binary model.")
    parser.add_argument("--project", type=Path, default=PROJECT_PATH)
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--dataset-audio-root", type=Path, default=DATASET_AUDIO_ROOT)
    parser.add_argument("--bundle-name", default=BUNDLE_NAME)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--negative-library-limit", type=int, default=NEGATIVE_LIBRARY_LIMIT)
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    return parser.parse_args()


def main() -> int:
    """Build, evaluate, and conditionally promote the Olivia snare bundle."""
    args = parse_args()
    started_at = datetime.now(UTC).isoformat()
    workspace_root = args.workspace_root.expanduser().resolve()
    dataset_audio_root = args.dataset_audio_root.expanduser().resolve()
    report_path = workspace_root / "reports" / "olivia_snare_monster_report.json"
    reset_directory(dataset_audio_root)
    (workspace_root / "reports").mkdir(parents=True, exist_ok=True)
    migrate_foundry_state(workspace_root)

    project_cache = workspace_root / "project_cache"
    reset_directory(project_cache)
    project_db = extract_project_db(args.project, project_cache)
    source_audio_refs = collect_project_source_audio_refs(project_db)
    extract_project_audio(args.project, project_cache, source_audio_refs)

    event_clips = build_project_event_clips(project_db, project_cache, dataset_audio_root / "olivia_events")
    base_samples = build_base_samples(
        event_clips=event_clips,
        samples_root=args.samples_root.expanduser().resolve(),
        negative_library_limit=args.negative_library_limit,
    )
    dataset, version = persist_weighted_dataset(
        workspace_root=workspace_root,
        project_path=args.project.expanduser().resolve(),
        samples_root=args.samples_root.expanduser().resolve(),
        base_samples=base_samples,
    )
    integrity = DatasetService.validate_version_integrity(version)
    if not integrity["ok"]:
        raise RuntimeError(f"Dataset integrity failed: {integrity['errors']}")

    app = FoundryApp(workspace_root)
    run = app.create_run(version.id, build_run_spec(version, epochs=args.epochs))
    completed_run = app.start_run(run.id)
    artifacts = app.list_artifacts_for_run(completed_run.id)
    if completed_run.status.value != "completed" or not artifacts:
        raise RuntimeError(f"Training failed or produced no artifacts: {completed_run.status.value}")
    artifact = sorted(artifacts, key=lambda item: item.created_at)[-1]
    compatibility = app.validate_artifact(artifact.id)
    if not compatibility.ok:
        raise RuntimeError(f"Artifact validation failed: {compatibility.errors}")

    metrics_path = completed_run.exports_dir(workspace_root) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    promotion = evaluate_promotion(metrics)
    candidate_model_path = completed_run.exports_dir(workspace_root) / "model.pth"
    score_sets = score_runtime_models(
        project_db=project_db,
        project_cache=project_cache,
        candidate_model_path=candidate_model_path,
    )
    promotion["olivia_score_gate"] = evaluate_olivia_score_gate(score_sets)
    promotion["passed"] = bool(promotion["passed"] and promotion["olivia_score_gate"]["passed"])

    installed_payload: dict[str, object] | None = None
    if promotion["passed"] and not args.no_promote:
        installed = app.runtime_bundles.install_binary_drum_artifact(
            artifact.id,
            bundle_name=args.bundle_name,
            models_dir=ensure_installed_models_dir(),
        )
        installed_payload = {
            "label": installed.label,
            "bundleName": installed.bundle_name,
            "bundleDir": str(installed.bundle_dir),
            "manifestPath": str(installed.manifest_path),
            "weightsPath": str(installed.weights_path),
            "artifactId": installed.artifact_id,
            "runId": installed.run_id,
        }

    report = {
        "schema": "echozero.olivia_snare_monster_report.v1",
        "startedAt": started_at,
        "finishedAt": datetime.now(UTC).isoformat(),
        "projectPath": str(args.project.expanduser().resolve()),
        "samplesRoot": str(args.samples_root.expanduser().resolve()),
        "workspaceRoot": str(workspace_root),
        "datasetAudioRoot": str(dataset_audio_root),
        "datasetId": dataset.id,
        "datasetVersionId": version.id,
        "datasetStats": version.stats,
        "splitDistribution": version.split_plan.get("label_distribution", {}),
        "runId": completed_run.id,
        "artifactId": artifact.id,
        "exportsDir": str(completed_run.exports_dir(workspace_root)),
        "compatibility": compatibility.to_contract_payload(),
        "promotion": promotion,
        "installed": installed_payload,
        "scoreSets": [score_set.to_payload() for score_set in score_sets],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if promotion["passed"] else 1


def reset_directory(path: Path) -> None:
    """Replace a generated directory with an empty one."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_project_db(project_path: Path, cache_dir: Path) -> Path:
    """Extract the project database from an EZ bundle."""
    db_path = cache_dir / "project.db"
    with zipfile.ZipFile(project_path) as archive:
        db_path.write_bytes(archive.read("project.db"))
    return db_path


def collect_project_source_audio_refs(project_db: Path) -> set[str]:
    """Collect audio refs needed to materialize event and scoring clips."""
    refs: set[str] = set()
    with sqlite3.connect(project_db) as connection:
        for (audio_file,) in connection.execute("select audio_file from song_versions"):
            refs.add(str(audio_file))
        for (data_json,) in connection.execute("select data_json from takes where is_main = 1"):
            payload = parse_json_object(data_json)
            if payload.get("type") == "AudioData" and payload.get("file_path"):
                refs.add(str(payload["file_path"]))
    return refs


def extract_project_audio(project_path: Path, cache_dir: Path, audio_refs: set[str]) -> None:
    """Extract only the project audio files needed for clips and scoring."""
    with zipfile.ZipFile(project_path) as archive:
        names = set(archive.namelist())
        for audio_ref in sorted(audio_refs):
            if audio_ref not in names:
                continue
            target = cache_dir / audio_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(audio_ref))


def build_project_event_clips(project_db: Path, project_cache: Path, output_dir: Path) -> list[EventClip]:
    """Materialize Olivia event clips from main Kick and Snare event layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[EventClip] = []
    with sqlite3.connect(project_db) as connection:
        connection.row_factory = sqlite3.Row
        layer_rows = {
            row["id"]: row
            for row in connection.execute("select id, song_version_id, name, provenance_json from layers")
        }
        song_rows = {
            row["version_id"]: row
            for row in connection.execute(
                """
                select song_versions.id as version_id, songs.title, song_versions.audio_file
                from song_versions
                join songs on songs.id = song_versions.song_id
                """
            )
        }
        main_audio_by_layer = collect_main_audio_by_layer(connection)
        query = """
            select takes.id as take_id, takes.layer_id, takes.data_json
            from takes
            join layers on layers.id = takes.layer_id
            where takes.is_main = 1
              and lower(layers.name) in ('kick', 'snare')
        """
        for row in connection.execute(query):
            layer = layer_rows[row["layer_id"]]
            label = str(layer["name"]).strip().lower()
            source_audio_ref = resolve_source_audio_ref(layer, song_rows, main_audio_by_layer)
            source_audio_path = project_cache / source_audio_ref
            if not source_audio_path.exists():
                continue
            events = extract_events(row["data_json"])
            song_title = str(song_rows[layer["song_version_id"]]["title"])
            for event in events:
                event_id = str(event.get("id") or uuid4().hex)
                time_seconds = float(event.get("time", 0.0))
                clip_path = output_dir / label / safe_filename(f"{song_title}_{event_id}.wav")
                write_event_clip(source_audio_path, clip_path, time_seconds)
                content_hash = compute_file_hash(clip_path)
                clips.append(
                    EventClip(
                        label=label,
                        source_audio_ref=source_audio_ref,
                        event_id=event_id,
                        song_title=song_title,
                        time_seconds=time_seconds,
                        audio_path=clip_path,
                        content_hash=content_hash,
                        group_id=f"olivia:{song_title}:{label}:{event_id}",
                    )
                )
    return sorted(clips, key=lambda clip: (clip.label, clip.song_title, clip.time_seconds, clip.event_id))


def collect_main_audio_by_layer(connection: sqlite3.Connection) -> dict[str, str]:
    """Map layer ids to their main AudioData refs."""
    refs: dict[str, str] = {}
    for layer_id, data_json in connection.execute("select layer_id, data_json from takes where is_main = 1"):
        payload = parse_json_object(data_json)
        if payload.get("type") == "AudioData" and payload.get("file_path"):
            refs[str(layer_id)] = str(payload["file_path"])
    return refs


def resolve_source_audio_ref(
    layer: sqlite3.Row,
    song_rows: dict[str, sqlite3.Row],
    main_audio_by_layer: dict[str, str],
) -> str:
    """Resolve the audio ref an event layer should be clipped against."""
    provenance = parse_json_object(layer["provenance_json"])
    source_layer_id = str(provenance.get("source_layer_id") or "").strip()
    if source_layer_id in main_audio_by_layer:
        return main_audio_by_layer[source_layer_id]
    song = song_rows[str(layer["song_version_id"])]
    return str(song["audio_file"])


def extract_events(data_json: str) -> list[dict[str, object]]:
    """Extract EventData events from a serialized take payload."""
    payload = parse_json_object(data_json)
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return []
    events: list[dict[str, object]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        raw_events = layer.get("events")
        if isinstance(raw_events, list):
            events.extend(event for event in raw_events if isinstance(event, dict))
    return events


def write_event_clip(source_audio_path: Path, clip_path: Path, time_seconds: float) -> None:
    """Write a one-second mono event clip at the runtime classifier window."""
    audio, sample_rate = librosa.load(
        source_audio_path,
        sr=22050,
        mono=True,
        offset=max(0.0, time_seconds),
        duration=EVENT_CLIP_SECONDS,
    )
    target_length = int(22050 * EVENT_CLIP_SECONDS)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(clip_path, audio.astype(np.float32), 22050)


def build_base_samples(
    *,
    event_clips: list[EventClip],
    samples_root: Path,
    negative_library_limit: int,
) -> list[DatasetSample]:
    """Create base samples before train-only Olivia weighting duplicates."""
    samples: list[DatasetSample] = []
    for clip in event_clips:
        samples.append(sample_from_event_clip(clip))
    for audio_path in sorted((samples_root / "Snare").rglob("*")):
        if is_audio_file(audio_path):
            samples.append(sample_from_library_file(audio_path, label=POSITIVE_LABEL, source_kind="organized_snare"))
    negative_paths = collect_negative_library_paths(samples_root, limit=negative_library_limit)
    for audio_path in negative_paths:
        samples.append(sample_from_library_file(audio_path, label=NEGATIVE_LABEL, source_kind="organized_negative"))
    return samples


def sample_from_event_clip(clip: EventClip) -> DatasetSample:
    """Create a DatasetSample from an Olivia event clip."""
    label = POSITIVE_LABEL if clip.label == POSITIVE_LABEL else NEGATIVE_LABEL
    return DatasetSample(
        sample_id=f"olivia_{clip.label}_{hashlib.sha1(str(clip.audio_path).encode()).hexdigest()[:16]}",
        audio_ref=str(clip.audio_path),
        label=label,
        duration_ms=EVENT_CLIP_SECONDS * 1000.0,
        content_hash=clip.content_hash,
        source_provenance={
            "source_kind": "olivia_project_event",
            "project_path": str(PROJECT_PATH),
            "song_title": clip.song_title,
            "event_id": clip.event_id,
            "event_time_seconds": clip.time_seconds,
            "source_audio_ref": clip.source_audio_ref,
            "original_label": clip.label,
            "group_id": clip.group_id,
        },
        group_id=clip.group_id,
        curation_state=CurationState.ACCEPTED,
    )


def sample_from_library_file(audio_path: Path, *, label: str, source_kind: str) -> DatasetSample:
    """Create a DatasetSample from an organized sample-library file."""
    content_hash = compute_file_hash(audio_path)
    digest = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()[:16]
    return DatasetSample(
        sample_id=f"{source_kind}_{digest}",
        audio_ref=str(audio_path),
        label=label,
        content_hash=content_hash,
        source_provenance={
            "source_kind": source_kind,
            "source_path": str(audio_path),
            "group_id": f"content:{content_hash}",
        },
        group_id=f"content:{content_hash}",
        curation_state=CurationState.ACCEPTED,
    )


def collect_negative_library_paths(samples_root: Path, *, limit: int) -> list[Path]:
    """Collect deterministic non-snare negatives from the organized sample tree."""
    paths: list[Path] = []
    for child in sorted(samples_root.iterdir()):
        if not child.is_dir() or child.name.lower() == "snare":
            continue
        paths.extend(path for path in sorted(child.rglob("*")) if is_audio_file(path))
    paths = sorted(paths, key=lambda path: hashlib.sha1(str(path).encode("utf-8")).hexdigest())
    return paths[:limit] if limit > 0 else paths


def persist_weighted_dataset(
    *,
    workspace_root: Path,
    project_path: Path,
    samples_root: Path,
    base_samples: list[DatasetSample],
) -> tuple[Dataset, DatasetVersion]:
    """Persist the final train-weighted binary dataset into Foundry state."""
    dataset = Dataset(
        id=f"ds_{uuid4().hex[:12]}",
        name="Olivia Snare Monster 20260501",
        source_kind="olivia_weighted_binary_snare",
        source_ref=str(project_path),
        metadata={
            "schema": "echozero.olivia_snare_monster_dataset.v1",
            "project_path": str(project_path),
            "samples_root": str(samples_root),
            "olivia_train_duplicates": OLIVIA_TRAIN_DUPLICATES,
        },
    )
    base_version = DatasetVersion(
        id=f"dsv_{uuid4().hex[:12]}",
        dataset_id=dataset.id,
        version=1,
        manifest_hash=DatasetService.compute_manifest_hash(base_samples),
        sample_rate=22050,
        audio_standard="mono_wav_pcm16_or_source_audio",
        class_map=[POSITIVE_LABEL, NEGATIVE_LABEL],
        samples=base_samples,
        taxonomy=build_taxonomy(),
        label_policy=build_label_policy(),
        stats=build_stats(base_samples),
    )
    split_service = SplitBalanceService()
    base_split = split_service.plan_splits(base_version, validation_split=0.15, test_split=0.10, seed=42)
    final_samples = apply_train_only_olivia_weighting(base_samples, base_split)
    final_version = DatasetVersion(
        id=base_version.id,
        dataset_id=dataset.id,
        version=1,
        manifest_hash=DatasetService.compute_manifest_hash(final_samples),
        sample_rate=base_version.sample_rate,
        audio_standard=base_version.audio_standard,
        class_map=[POSITIVE_LABEL, NEGATIVE_LABEL],
        samples=final_samples,
        taxonomy=base_version.taxonomy,
        label_policy=base_version.label_policy,
        manifest=build_manifest(final_samples, project_path=project_path, samples_root=samples_root),
        split_plan=build_final_split_plan(final_samples, base_split),
        balance_plan=split_service.plan_balance(
            DatasetVersion(
                id=base_version.id,
                dataset_id=dataset.id,
                version=1,
                manifest_hash=DatasetService.compute_manifest_hash(final_samples),
                sample_rate=base_version.sample_rate,
                audio_standard=base_version.audio_standard,
                class_map=[POSITIVE_LABEL, NEGATIVE_LABEL],
                samples=final_samples,
            ),
            strategy="none",
        ),
        stats=build_stats(final_samples),
        lineage={
            "kind": "olivia_weighted_binary_snare",
            "base_sample_count": len(base_samples),
            "weighted_sample_count": len(final_samples),
        },
    )
    DatasetRepository(workspace_root).save(dataset)
    DatasetVersionRepository(workspace_root).save(final_version)
    return dataset, final_version


def apply_train_only_olivia_weighting(
    samples: list[DatasetSample],
    split_plan: dict[str, object],
) -> list[DatasetSample]:
    """Duplicate only train-split Olivia samples so eval remains honest."""
    assignments = split_plan.get("assignments", {})
    final_samples = list(samples)
    for sample in samples:
        if assignments.get(sample.sample_id) != "train":
            continue
        if sample.source_provenance.get("source_kind") != "olivia_project_event":
            continue
        for copy_index in range(OLIVIA_TRAIN_DUPLICATES):
            final_samples.append(
                DatasetSample(
                    sample_id=f"{sample.sample_id}_w{copy_index + 1}",
                    audio_ref=sample.audio_ref,
                    label=sample.label,
                    duration_ms=sample.duration_ms,
                    content_hash=sample.content_hash,
                    source_provenance={
                        **sample.source_provenance,
                        "weighting_copy": copy_index + 1,
                    },
                    group_id=sample.group_id,
                    curation_state=CurationState.ACCEPTED,
                )
            )
    return final_samples


def build_final_split_plan(samples: list[DatasetSample], base_split: dict[str, object]) -> dict[str, object]:
    """Extend the base split with train-only weighting copies."""
    assignments = dict(base_split["assignments"])
    train_ids = list(base_split["train_ids"])
    for sample in samples:
        if sample.sample_id in assignments:
            continue
        assignments[sample.sample_id] = "train"
        train_ids.append(sample.sample_id)
    split_plan = {
        **base_split,
        "dataset_manifest_hash": DatasetService.compute_manifest_hash(samples),
        "dataset_sample_count": len(samples),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(base_split["val_ids"]),
        "test_ids": sorted(base_split["test_ids"]),
        "assignments": assignments,
    }
    temp_version = DatasetVersion(
        id=str(base_split["dataset_version_id"]),
        dataset_id="",
        version=1,
        manifest_hash=split_plan["dataset_manifest_hash"],
        sample_rate=22050,
        audio_standard="mono_wav_pcm16_or_source_audio",
        class_map=[POSITIVE_LABEL, NEGATIVE_LABEL],
        samples=samples,
    )
    split_plan["label_distribution"] = build_label_distribution(samples, assignments)
    split_plan["group_distribution"] = build_group_distribution(samples, assignments)
    split_plan["content_hash_groups"] = build_content_groups(samples)
    split_plan["leakage"] = SplitBalanceService._build_leakage_report(temp_version, assignments)
    split_plan["reproducibility"] = SplitBalanceService._build_reproducibility_report(temp_version, assignments, seed=42)
    return split_plan


def build_run_spec(version: DatasetVersion, *, epochs: int) -> dict[str, object]:
    """Build the Foundry CRNN run spec."""
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "binary",
        "model": {"type": "crnn"},
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {
            "epochs": epochs,
            "batchSize": 64,
            "learningRate": 0.001,
            "seed": 42,
            "trainerProfile": "stronger_v1",
            "optimizer": "adamw",
            "classWeighting": "balanced",
            "rebalanceStrategy": "oversample",
            "augmentTrain": True,
            "augmentNoiseStd": 0.03,
            "augmentGainJitter": 0.15,
            "augmentCopies": 2,
            "earlyStoppingPatience": 3,
            "minEpochs": 4,
            "regularizationAlpha": 0.0001,
            "weightDecay": 0.0001,
            "gradientClipNorm": 1.0,
        },
        "promotion": {
            "gate_policy": {
                "macro_f1_floor": 0.93,
                "max_regression_vs_reference": 0.01,
                "per_class_recall_floors": {"snare": 0.9006289308176101},
            }
        },
    }


def evaluate_promotion(metrics: dict[str, object]) -> dict[str, object]:
    """Evaluate Foundry metric gates against the canonical CRNN snare model."""
    final_eval = metrics["finalEval"]
    metric_values = final_eval["metrics"]
    per_class = final_eval["per_class_metrics"]
    macro_f1 = float(metric_values["macro_f1"])
    snare_recall = float(per_class[POSITIVE_LABEL]["recall"])
    canonical = load_canonical_metrics()
    canonical_macro_f1 = float(canonical["macro_f1"])
    canonical_snare_recall = float(canonical["snare_recall"])
    reasons: list[str] = []
    if macro_f1 < 0.93:
        reasons.append(f"macro_f1 {macro_f1:.6f} below 0.93")
    if snare_recall < canonical_snare_recall:
        reasons.append(f"snare recall {snare_recall:.6f} below canonical {canonical_snare_recall:.6f}")
    if macro_f1 < canonical_macro_f1 - 0.01:
        reasons.append(f"macro_f1 regression exceeds 0.01 vs canonical {canonical_macro_f1:.6f}")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidate": {"macroF1": macro_f1, "snareRecall": snare_recall},
        "canonical": {"macroF1": canonical_macro_f1, "snareRecall": canonical_snare_recall},
    }


def load_canonical_metrics() -> dict[str, float]:
    """Load canonical installed CRNN snare reference metrics."""
    metrics_path = Path("/Users/march/.echozero/models/binary-drum-snare/metrics.json")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_eval = payload["finalEval"]
    return {
        "macro_f1": float(final_eval["metrics"]["macro_f1"]),
        "snare_recall": float(final_eval["per_class_metrics"][POSITIVE_LABEL]["recall"]),
    }


def score_runtime_models(
    *,
    project_db: Path,
    project_cache: Path,
    candidate_model_path: Path,
) -> list[RuntimeScoreSet]:
    """Score candidate and reference models on Olivia project event truth."""
    positive_events, negative_events = collect_scoring_events(project_db, project_cache)
    model_paths = resolve_reference_model_paths(candidate_model_path)
    score_sets: list[RuntimeScoreSet] = []
    for name, model_path in model_paths:
        runtime_model = load_runtime_model(model_path, device="cpu")
        positive_scores = score_events(runtime_model, positive_events)
        negative_scores = score_events(runtime_model, negative_events)
        score_sets.append(
            RuntimeScoreSet(
                name=name,
                model_path=model_path,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
            )
        )
    return score_sets


def collect_scoring_events(project_db: Path, project_cache: Path) -> tuple[list[tuple[Path, float]], list[tuple[Path, float]]]:
    """Collect positive snare and negative kick event windows for real Olivia scoring."""
    positives: list[tuple[Path, float]] = []
    negatives: list[tuple[Path, float]] = []
    with sqlite3.connect(project_db) as connection:
        connection.row_factory = sqlite3.Row
        layer_rows = {
            row["id"]: row
            for row in connection.execute("select id, song_version_id, name, provenance_json from layers")
        }
        song_rows = {
            row["version_id"]: row
            for row in connection.execute(
                """
                select song_versions.id as version_id, songs.title, song_versions.audio_file
                from song_versions
                join songs on songs.id = song_versions.song_id
                """
            )
        }
        main_audio_by_layer = collect_main_audio_by_layer(connection)
        query = """
            select takes.layer_id, takes.data_json
            from takes
            join layers on layers.id = takes.layer_id
            where takes.is_main = 1
              and lower(layers.name) in ('kick', 'snare')
        """
        for row in connection.execute(query):
            layer = layer_rows[row["layer_id"]]
            label = str(layer["name"]).strip().lower()
            audio_ref = resolve_source_audio_ref(layer, song_rows, main_audio_by_layer)
            audio_path = project_cache / audio_ref
            if not audio_path.exists():
                continue
            target = positives if label == POSITIVE_LABEL else negatives
            for event in extract_events(row["data_json"]):
                target.append((audio_path, float(event.get("time", 0.0))))
    return positives, negatives


def resolve_reference_model_paths(candidate_model_path: Path) -> list[tuple[str, Path]]:
    """Resolve candidate, canonical, and current active snare models."""
    models_dir = ensure_installed_models_dir()
    records = load_binary_drum_bundle_index(models_dir)
    model_paths = [
        ("candidate", candidate_model_path),
        ("canonical_crnn", Path("/Users/march/.echozero/models/binary-drum-snare/model.pth")),
    ]
    active = records.get(POSITIVE_LABEL)
    if active is not None:
        model_paths.append(("active_indexed", models_dir / active.bundle_dir / active.weights_file))
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for name, path in model_paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        unique.append((name, resolved))
        seen.add(resolved)
    return unique


def score_events(runtime_model, events: list[tuple[Path, float]]) -> list[float]:
    """Score snare probability for event windows."""
    try:
        positive_index = runtime_model.classes.index(POSITIVE_LABEL)
    except ValueError as exc:
        raise RuntimeError(f"Runtime model lacks snare class: {runtime_model.source_path}") from exc
    scores: list[float] = []
    audio_cache: dict[Path, np.ndarray] = {}
    for audio_path, time_seconds in events:
        audio = audio_cache.get(audio_path)
        if audio is None:
            audio, _ = librosa.load(audio_path, sr=runtime_model.sample_rate, mono=True)
            audio_cache[audio_path] = audio
        feature = build_feature_tensor(
            audio=audio,
            event_time=time_seconds,
            sample_rate=runtime_model.sample_rate,
            max_length=runtime_model.max_length,
            n_fft=runtime_model.n_fft,
            hop_length=runtime_model.hop_length,
            n_mels=runtime_model.n_mels,
            fmax=runtime_model.fmax,
            feature_mode=runtime_model.feature_mode,
        )
        scores.append(float(predict_probabilities(runtime_model, feature)[positive_index]))
    return scores


def evaluate_olivia_score_gate(score_sets: list[RuntimeScoreSet]) -> dict[str, object]:
    """Require candidate Olivia separation to beat or match active/canonical references."""
    candidate = next(score_set for score_set in score_sets if score_set.name == "candidate")
    references = [score_set for score_set in score_sets if score_set.name != "candidate"]
    candidate_margin = score_margin(candidate)
    best_reference_margin = max((score_margin(score_set) for score_set in references), default=0.0)
    reasons: list[str] = []
    if candidate_margin < best_reference_margin:
        reasons.append(
            f"candidate Olivia score margin {candidate_margin:.6f} below best reference {best_reference_margin:.6f}"
        )
    if summarize_scores(candidate.positive_scores)["mean"] < 0.5:
        reasons.append("candidate mean positive snare score is below 0.5")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidateMargin": candidate_margin,
        "bestReferenceMargin": best_reference_margin,
    }


def score_margin(score_set: RuntimeScoreSet) -> float:
    """Compute positive-vs-negative mean separation."""
    return float(mean(score_set.positive_scores or [0.0]) - mean(score_set.negative_scores or [0.0]))


def summarize_scores(scores: list[float]) -> dict[str, float | int]:
    """Summarize classifier scores compactly for reports."""
    if not scores:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "lowConfidenceCount": 0}
    values = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(len(scores)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "lowConfidenceCount": int(np.sum(values < 0.5)),
    }


def build_taxonomy() -> dict[str, object]:
    """Build the binary percussion taxonomy."""
    return {
        "schema": "foundry.taxonomy.v1",
        "namespace": "percussion.one_shot",
        "version": 1,
        "labels": [
            {"id": POSITIVE_LABEL, "display_name": "snare", "aliases": []},
            {"id": NEGATIVE_LABEL, "display_name": "other", "aliases": []},
        ],
    }


def build_label_policy() -> dict[str, object]:
    """Build the binary label policy."""
    return {
        "schema": "foundry.label_policy.v1",
        "classification_mode": "binary",
        "unit": "one_shot",
        "allowed_labels": [POSITIVE_LABEL, NEGATIVE_LABEL],
        "unknown_label": None,
    }


def build_manifest(samples: list[DatasetSample], *, project_path: Path, samples_root: Path) -> dict[str, object]:
    """Build the dataset manifest."""
    return {
        "schema": "echozero.olivia_snare_monster_dataset_manifest.v1",
        "project_path": str(project_path),
        "samples_root": str(samples_root),
        "deterministic_order": [sample.sample_id for sample in samples],
        "content_hash_algorithm": "sha256",
        "content_groups": build_content_groups(samples),
        "real_sample_ids": [sample.sample_id for sample in samples if not sample.is_synthetic],
        "synthetic_sample_ids": [sample.sample_id for sample in samples if sample.is_synthetic],
    }


def build_stats(samples: list[DatasetSample]) -> dict[str, object]:
    """Build class and source stats."""
    class_counts = {
        label: sum(1 for sample in samples if sample.label == label)
        for label in (POSITIVE_LABEL, NEGATIVE_LABEL)
    }
    source_counts: dict[str, int] = {}
    for sample in samples:
        source_kind = str(sample.source_provenance.get("source_kind", "unknown"))
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
    return {
        "sample_count": len(samples),
        "real_sample_count": len(samples),
        "synthetic_sample_count": 0,
        "class_counts": class_counts,
        "source_counts": source_counts,
    }


def build_content_groups(samples: list[DatasetSample]) -> dict[str, list[str]]:
    """Group sample ids by content hash."""
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(sample.content_hash, []).append(sample.sample_id)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def build_label_distribution(samples: list[DatasetSample], assignments: dict[str, str]) -> dict[str, dict[str, int]]:
    """Build per-split label counts."""
    distribution: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for sample in samples:
        split_name = assignments.get(sample.sample_id, "unassigned")
        distribution.setdefault(split_name, {})
        distribution[split_name][sample.label] = distribution[split_name].get(sample.label, 0) + 1
    return distribution


def build_group_distribution(samples: list[DatasetSample], assignments: dict[str, str]) -> dict[str, int]:
    """Build per-split group counts."""
    groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for sample in samples:
        split_name = assignments.get(sample.sample_id)
        if split_name in groups:
            groups[split_name].add(SplitBalanceService.resolve_group_id(sample))
    return {key: len(value) for key, value in groups.items()}


def compute_file_hash(path: Path) -> str:
    """Compute a SHA-256 content hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_audio_file(path: Path) -> bool:
    """Return true when a path points to a supported audio file."""
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def parse_json_object(raw_value: object) -> dict[str, object]:
    """Parse a JSON object value safely."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        return {}
    payload = json.loads(raw_value)
    return payload if isinstance(payload, dict) else {}


def safe_filename(value: str) -> str:
    """Create a conservative filename for generated clips."""
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)[:180]


if __name__ == "__main__":
    raise SystemExit(main())
