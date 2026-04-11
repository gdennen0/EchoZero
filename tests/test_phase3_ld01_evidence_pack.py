from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "phase3_ld01_evidence_pack.py"
SPEC = importlib.util.spec_from_file_location("phase3_ld01_evidence_pack", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
phase3_ld01_evidence_pack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase3_ld01_evidence_pack
SPEC.loader.exec_module(phase3_ld01_evidence_pack)

ArtifactState = phase3_ld01_evidence_pack.ArtifactState
build_run_notes = phase3_ld01_evidence_pack.build_run_notes
evaluate_run_status = phase3_ld01_evidence_pack.evaluate_run_status


def _artifacts(*, external_bootstrap: bool = False, missing: bool = False) -> dict[str, ArtifactState]:
    status = "INCOMPLETE" if missing else "PASS"
    return {
        "01-initial-state.png": ArtifactState(
            status=status,
            path=Path("01.png"),
            external_bootstrap=external_bootstrap and not missing,
        ),
        "02-post-extract-all.png": ArtifactState(status=status, path=Path("02.png")),
        "03-divergence-visible.png": ArtifactState(status=status, path=Path("03.png")),
        "04-post-resolution-or-sync.png": ArtifactState(status=status, path=Path("04.png")),
        "phase3-ld-01-walkthrough.mp4": ArtifactState(status=status, path=Path("walkthrough.mp4")),
    }


def test_evaluate_run_status_returns_incomplete_when_required_artifacts_missing() -> None:
    run_status, signoff_ready, run_outcome_note = evaluate_run_status(
        capture_exit_code=0,
        artifacts=_artifacts(missing=True),
    )

    assert run_status == "INCOMPLETE"
    assert signoff_ready is False
    assert "not signoff-ready" in run_outcome_note


def test_evaluate_run_status_returns_bootstrap_pass_when_external_bootstrap_artifact_required() -> None:
    run_status, signoff_ready, run_outcome_note = evaluate_run_status(
        capture_exit_code=0,
        artifacts=_artifacts(external_bootstrap=True),
    )

    assert run_status == "BOOTSTRAP_PASS"
    assert signoff_ready is False
    assert "bootstrap-only" in run_outcome_note


def test_build_run_notes_marks_signoff_ready_run() -> None:
    artifacts = _artifacts()
    run_status, signoff_ready, run_outcome_note = evaluate_run_status(
        capture_exit_code=0,
        artifacts=artifacts,
    )

    notes = build_run_notes(
        timestamp="2026-04-10T00:00:00+00:00",
        operator="codex",
        commit_hash="abc123",
        run_status=run_status,
        signoff_ready=signoff_ready,
        run_outcome_note=run_outcome_note,
        audio_path=Path("audio.wav"),
        ma3_replay_fixture=Path("fixture.json"),
        evidence_dir=Path("proof"),
        raw_capture_dir=Path("proof/raw"),
        work_root=Path("proof/work"),
        walkthrough_source_dir=Path("walkthrough"),
        capture_result=subprocess.CompletedProcess(
            args=["cmd"],
            returncode=0,
            stdout="ok",
            stderr="",
        ),
        artifacts=artifacts,
    )

    assert run_status == "PASS"
    assert signoff_ready is True
    assert "- signoff_ready: yes" in notes
    assert (
        "- run_classification: signoff-ready: all required artifacts were produced by this run without explicit external "
        "bootstrap dependencies."
    ) in notes
    assert "## Signoff" in notes
    assert "signoff-ready: all required artifacts were produced by this run without explicit external bootstrap dependencies." in notes
