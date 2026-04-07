from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.foundry.audio_fixtures import write_percussion_dataset
from echozero.ui.style.qt.qss import build_foundry_shell_qss


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed in this environment",
)


def _run_ui_script(script: str, workspace: Path) -> dict:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(workspace)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_foundry_window_smoke(tmp_path: Path):
    payload = _run_ui_script(
        textwrap.dedent(
            """
            import json
            import sys
            import time
            from pathlib import Path

            from PyQt6.QtWidgets import QApplication
            from echozero.foundry.ui import FoundryWindow

            root = Path(sys.argv[1])
            app = QApplication.instance() or QApplication([])
            window = FoundryWindow(root)
            print(json.dumps({
                "title": window.windowTitle(),
                "status": window.status_line.text(),
                "workspace": window.workspace_summary.toPlainText(),
            }))
            window.close()
            app.quit()
            """
        ),
        tmp_path,
    )

    assert payload["title"].startswith("EchoZero Foundry")
    assert "Workspace ready" in payload["status"]
    assert "Root:" in payload["workspace"]


def test_foundry_window_applies_shared_shell_stylesheet(tmp_path: Path):
    payload = _run_ui_script(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            from PyQt6.QtWidgets import QApplication
            from echozero.foundry.ui import FoundryWindow

            root = Path(sys.argv[1])
            app = QApplication.instance() or QApplication([])
            window = FoundryWindow(root)
            print(json.dumps({
                "style": window.styleSheet(),
                "root_name": window.centralWidget().objectName(),
                "status_name": window.status_line.objectName(),
            }))
            window.close()
            app.quit()
            """
        ),
        tmp_path,
    )

    assert payload["style"] == build_foundry_shell_qss()
    assert payload["root_name"] == "foundryRoot"
    assert payload["status_name"] == "foundryStatusLine"


def test_foundry_window_desktop_workflow_exposes_run_artifact_and_eval(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    payload = _run_ui_script(
        textwrap.dedent(
            """
            import json
            import sys
            import time
            from pathlib import Path

            from PyQt6.QtWidgets import QApplication
            from echozero.foundry.ui import FoundryWindow

            root = Path(sys.argv[1])
            samples = root / "samples"
            app = QApplication.instance() or QApplication([])
            window = FoundryWindow(root)
            window.dataset_name.setText("Desktop Drums")
            window.dataset_folder.setText(str(samples))
            window._create_and_ingest_dataset()
            window._plan_version()
            window._create_and_start_run()
            deadline = time.time() + 30
            while window._run_thread is not None and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            print(json.dumps({
                "dataset_id": window._dataset_id,
                "version_id": window._version_id,
                "run_id": window._run_id,
                "artifact_id": window._artifact_id,
                "dataset_summary": window.dataset_summary.toPlainText(),
                "run_summary": window.run_summary.toPlainText(),
                "artifact_summary": window.artifact_summary.toPlainText(),
                "run_count": window.run_list.count(),
                "artifact_count": window.artifact_list.count(),
                "eval_count": window.eval_list.count(),
            }))
            window.close()
            app.quit()
            """
        ),
        tmp_path,
    )

    assert payload["dataset_id"] is not None
    assert payload["version_id"] is not None
    assert payload["run_id"] is not None
    assert payload["artifact_id"] is not None
    assert "Dataset: Desktop Drums" in payload["dataset_summary"]
    assert "Status: completed" in payload["run_summary"]
    assert "Validation: ok=True" in payload["artifact_summary"]
    assert payload["run_count"] == 1
    assert payload["artifact_count"] == 1
    assert payload["eval_count"] == 1


def test_foundry_window_loads_existing_workspace_state(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    payload = _run_ui_script(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            from PyQt6.QtWidgets import QApplication
            from echozero.foundry.app import FoundryApp
            from echozero.foundry.ui import FoundryWindow

            root = Path(sys.argv[1])
            samples = root / "samples"

            foundry = FoundryApp(root)
            dataset = foundry.datasets.create_dataset("Existing Drums", source_ref=str(samples))
            version = foundry.datasets.ingest_from_folder(dataset.id, samples)
            foundry.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=11, balance_strategy="none")
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

            app = QApplication.instance() or QApplication([])
            window = FoundryWindow(root)
            print(json.dumps({
                "run_count": window.run_list.count(),
                "artifact_count": window.artifact_list.count(),
                "eval_count": window.eval_list.count(),
                "dataset_summary": window.dataset_summary.toPlainText(),
                "run_summary": window.run_summary.toPlainText(),
            }))
            window.close()
            app.quit()
            """
        ),
        tmp_path,
    )

    assert payload["run_count"] == 1
    assert payload["artifact_count"] == 1
    assert payload["eval_count"] == 1
    assert "Existing Drums" in payload["dataset_summary"]
    assert "Status: completed" in payload["run_summary"]
