"""Focused proof for Foundry model evolution service boundaries.
Exists because fixed timeline Events must become event-span training samples.
Connects Event truth, all-negative planning, lineage, and candidate run creation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from echozero.foundry import FoundryApp
from echozero.foundry.model_evolution import (
    FixedEventTruth,
    ModelEvolutionRunRequest,
    ModelEvolutionService,
    RuntimeWindowPolicy,
)
from echozero.models.runtime_bundle_index import (
    IndexedBinaryDrumBundle,
    save_binary_drum_bundle_index,
)


def test_model_evolution_materializes_event_spans_and_creates_lineage_runs(
    tmp_path: Path,
) -> None:
    source_audio = tmp_path / "drums.wav"
    _write_impulse_train(source_audio)
    models_dir = tmp_path / "models"
    kick_manifest = _write_installed_bundle(models_dir, "foundry-kick-ovr-crnn-v2", "kick")
    snare_manifest = _write_installed_bundle(
        models_dir,
        "binary-drum-snare-olivia-monster",
        "snare",
    )
    save_binary_drum_bundle_index(
        models_dir,
        {
            "kick": IndexedBinaryDrumBundle(
                label="kick",
                bundle_dir="foundry-kick-ovr-crnn-v2",
                manifest_file=kick_manifest.name,
                weights_file="model.pth",
                artifact_id="art_kick_base",
                run_id="run_kick_base",
            ),
            "snare": IndexedBinaryDrumBundle(
                label="snare",
                bundle_dir="binary-drum-snare-olivia-monster",
                manifest_file=snare_manifest.name,
                weights_file="model.pth",
                artifact_id="art_snare_base",
                run_id="run_snare_base",
            ),
        },
    )
    truths = tuple(
        FixedEventTruth(
            truth_id=f"truth_{index}_{label}",
            label=label,
            source_audio_path=source_audio,
            event_start_seconds=offset,
            event_end_seconds=offset + 0.08,
            event_id=f"evt_{index}",
            decision_kind="verified",
        )
        for index, (label, offset) in enumerate(
            (
                ("kick", 0.1),
                ("kick", 1.3),
                ("snare", 2.5),
                ("snare", 3.7),
                ("clap", 4.9),
                ("clap", 6.1),
                ("cymbal", 7.3),
                ("cymbal", 8.5),
            )
        )
    )

    result = ModelEvolutionService(
        tmp_path,
        models_dir_factory=lambda: models_dir,
    ).create_candidate_runs(
        ModelEvolutionRunRequest(
            identity="Noah Kahan",
            truths=truths,
            labels=("kick", "snare"),
            profile_name="beefy",
            window_policy=RuntimeWindowPolicy(),
        )
    )

    assert result.source_dataset_version.stats["class_counts"] == {
        "clap": 2,
        "cymbal": 2,
        "kick": 2,
        "snare": 2,
    }
    assert {
        round(float(sample.source_provenance["event_duration_ms"]))
        for sample in result.source_dataset_version.samples
    } == {80}
    assert {
        round(float(sample.source_provenance["runtime_window_duration_ms"]))
        for sample in result.source_dataset_version.samples
    } == {80}
    assert {
        sample.source_provenance["sample_window_kind"]
        for sample in result.source_dataset_version.samples
    } == {"event_span"}
    assert len(result.candidate_plans) == 2
    assert len(result.runs) == 2

    plans_by_label = {plan.label: plan for plan in result.candidate_plans}
    kick_plan = plans_by_label["kick"]
    snare_plan = plans_by_label["snare"]
    assert kick_plan.positive_count == 2
    assert kick_plan.negative_count == 6
    assert kick_plan.negative_source_counts == {"clap": 2, "cymbal": 2, "snare": 2}
    assert kick_plan.lineage.initial_model_path == kick_manifest.resolve()
    assert kick_plan.run_spec["model"]["initialWeightsPath"] == str(kick_manifest.resolve())
    assert kick_plan.run_spec["evolution"]["targetIdentity"] == "Noah Kahan"
    assert kick_plan.run_spec["evolution"]["trainingProfile"] == "beefy"
    assert kick_plan.run_spec["evolution"]["negativeSourceCounts"] == {
        "clap": 2,
        "cymbal": 2,
        "snare": 2,
    }
    assert snare_plan.negative_source_counts == {"clap": 2, "cymbal": 2, "kick": 2}
    assert snare_plan.lineage.initial_model_path == snare_manifest.resolve()

    app = FoundryApp(tmp_path)
    persisted_kick_dataset = app.datasets.get_version(kick_plan.dataset_version_id)
    assert persisted_kick_dataset is not None
    assert persisted_kick_dataset.stats["negative_source_counts"] == {
        "clap": 2,
        "cymbal": 2,
        "snare": 2,
    }
    for run in result.runs:
        persisted = app.runs.get_run(run.id)
        assert persisted is not None
        assert persisted.spec["evolution"]["schema"] == "foundry.model_evolution_run.v1"


def _write_installed_bundle(models_dir: Path, bundle_name: str, label: str) -> Path:
    bundle_dir = models_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / f"{label}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": [label, "other"],
                "classificationMode": "binary",
                "weightsPath": "model.pth",
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "model.pth").write_bytes(b"fixture")
    return manifest_path


def _write_impulse_train(path: Path, sample_rate: int = 22050) -> None:
    audio = np.zeros(sample_rate * 12, dtype=np.float32)
    for index, offset_seconds in enumerate((0.1, 1.3, 2.5, 3.7, 4.9, 6.1, 7.3, 8.5)):
        start = int(offset_seconds * sample_rate)
        width = 128 + (index * 8)
        audio[start : start + width] = np.linspace(0.2, 1.0 - (index * 0.05), width)
    sf.write(path, audio, sample_rate)
