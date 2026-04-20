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
                    {"action": "selection.first_event", "params": {"layer_id": "layer_kick"}},
                    {"action": "sync.enable"},
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "Valid Scenario"
    assert scenario.seed == 11
    assert [step.action for step in scenario.steps] == ["selection.first_event", "sync.enable"]


def test_load_scenario_parses_real_input_pipeline_actions(tmp_path):
    scenario_path = tmp_path / "pipeline_actions.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Pipeline Actions",
                "steps": [
                    {
                        "action": "song.add",
                        "params": {
                            "title": "Test Song",
                            "audio_path": "C:/audio/test.wav",
                        },
                    },
                    {
                        "action": "timeline.extract_stems",
                        "params": {"layer_id": "layer_song"},
                    },
                    {
                        "action": "timeline.extract_drum_events",
                        "params": {"layer_title": "Drums"},
                    },
                    {
                        "action": "timeline.classify_drum_events",
                        "params": {"layer_title": "Drums", "model_path": "C:/models/drums.pth"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert [step.action for step in scenario.steps] == [
        "song.add",
        "timeline.extract_stems",
        "timeline.extract_drum_events",
        "timeline.classify_drum_events",
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
                "steps": [{"action": "song.add", "params": {}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="title"):
        load_scenario(scenario_path)


def test_load_scenario_rejects_missing_pipeline_action_fields(tmp_path):
    scenario_path = tmp_path / "missing_pipeline_fields.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Missing Pipeline Fields",
                "steps": [{"action": "timeline.extract_stems", "params": {}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="layer_id"):
        load_scenario(scenario_path)


def test_load_scenario_accepts_layer_title_targets_and_action_aliases(tmp_path):
    scenario_path = tmp_path / "layer_title.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Layer Titles",
                "steps": [
                    {"action": "selection.first_event", "params": {"layer_title": "Kick"}},
                    {
                        "action": "transfer.workspace_open",
                        "params": {"layer_title": "Kick", "direction": "push"},
                    },
                    {
                        "action": "timeline.nudge_selection",
                        "params": {"direction": "right", "steps": 1},
                    },
                    {"action": "timeline.duplicate_selection", "params": {"steps": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert [step.action for step in scenario.steps] == [
        "selection.first_event",
        "transfer.workspace_open",
        "timeline.nudge_selection",
        "timeline.duplicate_selection",
    ]


def test_load_scenario_rejects_classify_without_model_path(tmp_path):
    scenario_path = tmp_path / "classify_missing_model.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "Missing Model",
                "steps": [
                    {
                        "action": "timeline.classify_drum_events",
                        "params": {"layer_title": "Drums"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_path"):
        load_scenario(scenario_path)
