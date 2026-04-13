from __future__ import annotations

import json

import pytest

from echozero.testing.gui_dsl import load_scenario


def test_load_scenario_parses_valid_json(tmp_path):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Valid Scenario",
                "seed": 11,
                "steps": [
                    {"action": "select_first_event", "params": {"layer_id": "layer_kick"}},
                    {"action": "enable_sync"},
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "Valid Scenario"
    assert scenario.seed == 11
    assert [step.action for step in scenario.steps] == ["select_first_event", "enable_sync"]


def test_load_scenario_parses_real_input_pipeline_actions(tmp_path):
    scenario_path = tmp_path / "pipeline_actions.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Pipeline Actions",
                "steps": [
                    {
                        "action": "add_song_from_path",
                        "params": {
                            "title": "Test Song",
                            "audio_path": "C:/audio/test.wav",
                        },
                    },
                    {
                        "action": "extract_stems",
                        "params": {"layer_id": "layer_song"},
                    },
                    {
                        "action": "extract_drum_events",
                        "params": {"layer_id": "layer_drums"},
                    },
                    {
                        "action": "classify_drum_events",
                        "params": {"layer_id": "layer_drums"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert [step.action for step in scenario.steps] == [
        "add_song_from_path",
        "extract_stems",
        "extract_drum_events",
        "classify_drum_events",
    ]


def test_load_scenario_rejects_unsupported_action(tmp_path):
    scenario_path = tmp_path / "bad_action.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Bad Action",
                "steps": [{"action": "teleport"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        load_scenario(scenario_path)


def test_load_scenario_rejects_missing_required_fields(tmp_path):
    scenario_path = tmp_path / "missing_fields.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Missing Fields",
                "steps": [{"action": "trigger_action", "params": {}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_id"):
        load_scenario(scenario_path)


def test_load_scenario_rejects_missing_pipeline_action_fields(tmp_path):
    scenario_path = tmp_path / "missing_pipeline_fields.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Missing Pipeline Fields",
                "steps": [{"action": "extract_stems", "params": {}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="layer_id"):
        load_scenario(scenario_path)
