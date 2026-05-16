"""
Opt-in Audio Engine v2 real-project smoke coverage.
Exists so private .ez files can exercise the app presentation path outside CI.
Connects pytest to the reusable fake-output smoke command.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from echozero.application.playback.engine_selection import ENGINE_BACKEND_ENV
from scripts.audio_engine_v2_real_project_smoke import run_real_project_smoke


def test_audio_engine_v2_real_project_smoke_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise a private local .ez through app load, v2 fake output, and callbacks."""

    if not os.environ.get("ECHOZERO_REAL_PROJECT_SMOKE"):
        pytest.skip("set ECHOZERO_REAL_PROJECT_SMOKE=/path/to/project.ez")
    monkeypatch.setenv(ENGINE_BACKEND_ENV, "v2")
    project_path = Path(os.environ["ECHOZERO_REAL_PROJECT_SMOKE"]).expanduser().resolve()

    result = run_real_project_smoke(project_path)

    assert result.project_path == project_path
    assert result.playable_layer_count > 0
    assert result.track_count > 0
    assert result.graph_hash
    assert result.paused_peak == pytest.approx(0.0)
