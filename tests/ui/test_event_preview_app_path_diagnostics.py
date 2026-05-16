"""
Event preview app-path diagnostics tests.
Exists so private .ez projects can exercise preview playback without speaker hardware.
Connects pytest env gating to the reusable fake-output diagnostic command.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.event_preview_app_path_diagnostics import (
    resolve_project_paths,
    run_event_preview_diagnostics,
)


def test_event_preview_diagnostics_env_absent_is_ci_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no project paths when private real-project smoke env is absent."""

    monkeypatch.delenv("ECHOZERO_REAL_PROJECT_SMOKE", raising=False)

    assert resolve_project_paths([]) == []


def test_event_preview_diagnostics_real_project_from_env(tmp_path: Path) -> None:
    """Exercise private .ez fixtures through real app preview path with fake output."""

    raw_value = os.environ.get("ECHOZERO_REAL_PROJECT_SMOKE", "").strip()
    if not raw_value:
        pytest.skip("set ECHOZERO_REAL_PROJECT_SMOKE=/path/to/project.ez")
    project_paths = resolve_project_paths([])
    if not project_paths:
        pytest.skip("ECHOZERO_REAL_PROJECT_SMOKE did not contain project paths")

    result = run_event_preview_diagnostics(
        project_paths[0],
        sample_rate=48000,
        channels=2,
        blocks=6,
        output_dir=tmp_path,
    )

    assert result.project_path == str(project_paths[0])
    assert result.fake_output is True
    assert result.speaker_audio is False
    assert result.source_ref
    assert result.source_sample_rate > 0
    assert result.output_sample_rate == 48000
    assert result.output_channels == 2
    assert result.stream_blocksize > 0
    assert result.clip_frame_count > 0
    assert result.rendered_blocks
    assert max(float(block["peak_abs"]) for block in result.rendered_blocks) >= 0.0
    assert {event.get("kind") for event in result.runtime_events} >= {
        "preview-start",
        "overlay-start",
    }
    assert result.glitch_count == 0
