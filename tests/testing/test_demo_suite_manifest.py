from __future__ import annotations

from pathlib import Path

from echozero.testing.demo_suite import (
    build_manifest,
    collect_reference_artifacts,
    find_latest_smoke_report,
    ScenarioResult,
)


def test_manifest_schema_keys_exist(tmp_path: Path):
    run_folder = tmp_path / "demo-suite" / "20260413-010203"
    run_folder.mkdir(parents=True)
    manifest = build_manifest(
        run_id="20260413-010203",
        run_folder=run_folder,
        scenario_results=[
            ScenarioResult(
                group="canonical_app_lifecycle",
                name="baseline",
                status="passed",
                artifacts={"screenshot": "canonical/baseline.png"},
            ),
            ScenarioResult(
                group="real_data_scenario",
                name="real_audio_flow",
                status="skipped",
                notes=["audio not provided"],
            ),
        ],
    )

    assert set(manifest) >= {"run_timestamp", "run_id", "run_folder", "scenario_statuses", "counts", "proof_contract"}
    assert manifest["counts"]["passed"] == 1
    assert manifest["counts"]["skipped"] == 1
    assert manifest["counts"]["artifacts"] == 1
    assert manifest["scenario_statuses"][0]["group"] == "canonical_app_lifecycle"
    assert manifest["proof_contract"]["simulated_artifacts_must_be_labeled"] is True


def test_missing_optional_artifacts_are_marked_without_crashing(tmp_path: Path):
    run_folder = tmp_path / "run"
    run_folder.mkdir()

    results = collect_reference_artifacts(
        run_folder=run_folder,
        reference_root=tmp_path / "missing-timeline-demo",
    )

    assert results
    assert all(result.status == "missing" for result in results)
    assert all(result.notes for result in results)


def test_find_latest_smoke_report_prefers_latest_named_release(tmp_path: Path):
    releases_root = tmp_path / "artifacts" / "releases" / "test"
    older = releases_root / "20260412-101500"
    newer = releases_root / "20260413-090000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "smoke-report.json").write_text("{}", encoding="utf-8")
    (newer / "smoke-report.json").write_text("{\"status\":\"passed\"}", encoding="utf-8")

    latest = find_latest_smoke_report(releases_root)

    assert latest == newer / "smoke-report.json"
