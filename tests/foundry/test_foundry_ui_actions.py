from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from echozero.foundry.app import FoundryApp
from echozero.foundry.ui.main_window import FoundryWindow
from tests.foundry.audio_fixtures import write_percussion_dataset


def _qt_app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _wait_until(app: QApplication, predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _prepare_dataset(app: FoundryApp, samples: Path, *, name: str = "Desktop Drums") -> tuple[object, object]:
    dataset = app.datasets.create_dataset(name, source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=11, balance_strategy="none")
    return dataset, version


def _prepare_completed_run(root: Path) -> tuple[FoundryWindow, QApplication]:
    samples = root / "samples"
    write_percussion_dataset(samples)
    foundry = FoundryApp(root)
    _, version = _prepare_dataset(foundry, samples)
    run = foundry.create_run(
        version.id,
        {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": version.id,
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 7},
        },
    )
    foundry.start_run(run.id)
    qt_app = _qt_app()
    window = FoundryWindow(root)
    return window, qt_app


def test_foundry_window_create_and_start_runs_in_background_with_live_updates(tmp_path: Path, monkeypatch):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)
    window.dataset_name.setText("Async Drums")
    window.dataset_folder.setText(str(samples))
    window._create_and_ingest_dataset()
    window._plan_version()

    original_train = window._app.runs._trainer.train

    def delayed_train(run, dataset_version):
        time.sleep(0.35)
        return original_train(run, dataset_version)

    monkeypatch.setattr(window._app.runs._trainer, "train", delayed_train)

    window._create_and_start_run()

    assert window._run_thread is not None
    assert window.create_run_btn.isEnabled() is False
    assert "background" in window.status_line.text().lower()
    assert _wait_until(qt_app, lambda: "RUN_PREPARING" in window.activity.toPlainText(), timeout=5.0)
    assert _wait_until(qt_app, lambda: window._run_thread is None, timeout=30.0)
    assert window.create_run_btn.isEnabled() is True
    assert "Status: completed" in window.run_summary.toPlainText()
    assert "RUN_COMPLETED" in window.activity.toPlainText()

    window.close()
    qt_app.processEvents()


def test_foundry_window_background_start_handles_failed_run_gracefully(tmp_path: Path, monkeypatch):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)
    window.dataset_name.setText("Failure Drums")
    window.dataset_folder.setText(str(samples))
    window._create_and_ingest_dataset()
    window._plan_version()
    window._create_run()

    def failing_train(run, dataset_version):
        time.sleep(0.2)
        raise RuntimeError("fixture training failure")

    monkeypatch.setattr(window._app.runs._trainer, "train", failing_train)

    window._start_run()

    assert _wait_until(qt_app, lambda: window._run_thread is None, timeout=15.0)
    assert window.start_run_btn.isEnabled() is True
    assert "Status: failed" in window.run_summary.toPlainText()
    assert "RUN_FAILED" in window.activity.toPlainText()

    window.close()
    qt_app.processEvents()


def test_foundry_window_opens_exports_and_artifact_outputs_and_reports_missing_files(tmp_path: Path, monkeypatch):
    window, qt_app = _prepare_completed_run(tmp_path)
    opened_paths: list[str] = []

    def fake_open(url):
        opened_paths.append(url.toLocalFile())
        return True

    monkeypatch.setattr("echozero.foundry.ui.main_window.QDesktopServices.openUrl", staticmethod(fake_open))

    window._open_exports_dir()
    window._open_metrics_json()
    window._open_run_summary_json()
    window._open_artifact_manifest()

    assert len(opened_paths) == 4
    assert opened_paths[0].endswith("exports")
    assert opened_paths[1].endswith("metrics.json")
    assert opened_paths[2].endswith("run_summary.json")
    assert opened_paths[3].endswith(".manifest.json")

    Path(opened_paths[2]).unlink()
    window._open_run_summary_json()
    assert "Error: run_summary.json not found" in window.status_line.text()

    window.close()
    qt_app.processEvents()


def test_foundry_window_opens_latest_artifact_package_from_persisted_state(tmp_path: Path, monkeypatch):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    foundry = FoundryApp(tmp_path)
    _, version = _prepare_dataset(foundry, samples, name="Queued Drums")
    first_run = foundry.create_run(version.id, {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 3},
    })
    foundry.start_run(first_run.id)
    second_run = foundry.create_run(version.id, {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 5},
    })
    foundry.start_run(second_run.id)

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)
    opened_paths: list[str] = []

    def fake_open(url):
        opened_paths.append(url.toLocalFile())
        return True

    monkeypatch.setattr("echozero.foundry.ui.main_window.QDesktopServices.openUrl", staticmethod(fake_open))

    latest_artifact = sorted(foundry.artifacts._artifact_repo.list(), key=lambda item: (item.created_at, item.id))[-1]
    window._open_latest_artifact_package()

    assert [Path(path) for path in opened_paths] == [latest_artifact.path.parent]

    window.close()
    qt_app.processEvents()


def test_foundry_window_latest_artifact_package_reports_missing_path(tmp_path: Path):
    window, qt_app = _prepare_completed_run(tmp_path)

    latest_package = window._resolve_latest_artifact_package_path()
    assert latest_package is not None
    for child in latest_package.iterdir():
        child.unlink()
    latest_package.rmdir()

    window._open_latest_artifact_package()

    assert "Latest artifact package is missing on disk" in window.status_line.text()

    window.close()
    qt_app.processEvents()


def test_foundry_window_dataset_and_version_selectors_support_multi_dataset_workspaces(tmp_path: Path):
    samples_a = tmp_path / "samples_a"
    samples_b = tmp_path / "samples_b"
    write_percussion_dataset(samples_a)
    write_percussion_dataset(samples_b)

    app = FoundryApp(tmp_path)
    dataset_a, version_a = _prepare_dataset(app, samples_a, name="Alpha Drums")
    dataset_b, version_b1 = _prepare_dataset(app, samples_b, name="Beta Drums")
    version_b2 = app.datasets.ingest_from_folder(dataset_b.id, samples_b)
    app.plan_version(version_b2.id, validation_split=0.2, test_split=0.2, seed=17, balance_strategy="none")

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)

    assert window.dataset_selector.count() == 2
    dataset_b_index = window.dataset_selector.findData(dataset_b.id)
    window.dataset_selector.setCurrentIndex(dataset_b_index)
    assert _wait_until(qt_app, lambda: window.dataset_selector.currentData() == dataset_b.id)
    assert window.version_selector.count() == 2

    dataset_a_index = window.dataset_selector.findData(dataset_a.id)
    window.dataset_selector.setCurrentIndex(dataset_a_index)
    assert _wait_until(qt_app, lambda: window.dataset_selector.currentData() == dataset_a.id)
    assert window.version_selector.count() == 1
    assert dataset_a.name in window.dataset_summary.toPlainText()
    assert version_a.id in window.dataset_summary.toPlainText()

    window.dataset_selector.setCurrentIndex(dataset_b_index)
    assert _wait_until(qt_app, lambda: window.dataset_selector.currentData() == dataset_b.id)
    assert window.version_selector.count() == 2

    version_b1_index = window.version_selector.findData(version_b1.id)
    window.version_selector.setCurrentIndex(version_b1_index)
    assert _wait_until(qt_app, lambda: window.version_selector.currentData() == version_b1.id)
    assert version_b1.id in window.dataset_summary.toPlainText()
    assert window.class_names.text() == ",".join(version_b1.class_map)

    window.close()
    qt_app.processEvents()


def test_foundry_window_queue_panel_populates_with_queued_and_recent_runs(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    foundry = FoundryApp(tmp_path)
    _, version = _prepare_dataset(foundry, samples, name="Queue Drums")
    completed_run = foundry.create_run(version.id, {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 13},
    })
    foundry.start_run(completed_run.id)
    queued_run = foundry.create_run(version.id, {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 17},
    })

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)

    queue_entries = [window.queue_list.item(index).text() for index in range(window.queue_list.count())]

    assert any(queued_run.id in entry and "[queued]" in entry and "ACTIVE " in entry for entry in queue_entries)
    assert any(completed_run.id in entry and "[completed]" in entry for entry in queue_entries)

    window.close()
    qt_app.processEvents()


def test_foundry_window_queue_panel_updates_active_run_status_during_background_run(tmp_path: Path, monkeypatch):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    qt_app = _qt_app()
    window = FoundryWindow(tmp_path)
    window.dataset_name.setText("Queue Activity")
    window.dataset_folder.setText(str(samples))
    window._create_and_ingest_dataset()
    window._plan_version()

    original_train = window._app.runs._trainer.train

    def delayed_train(run, dataset_version):
        time.sleep(0.35)
        return original_train(run, dataset_version)

    monkeypatch.setattr(window._app.runs._trainer, "train", delayed_train)

    window._create_and_start_run()

    assert _wait_until(
        qt_app,
        lambda: any(
            "ACTIVE " in window.queue_list.item(index).text() and "[running]" in window.queue_list.item(index).text()
            for index in range(window.queue_list.count())
        ),
        timeout=5.0,
    )
    assert _wait_until(qt_app, lambda: window._run_thread is None, timeout=30.0)
    assert any(
        window._run_id in window.queue_list.item(index).text() and "[completed]" in window.queue_list.item(index).text()
        for index in range(window.queue_list.count())
    )

    window.close()
    qt_app.processEvents()
