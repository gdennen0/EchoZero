from echozero.testing.e2e.scenario import (
    AssertStep,
    load_scenario,
    scenario_from_mapping,
)


def test_load_json_scenario(tmp_path):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        '{"name":"sample","steps":[{"kind":"assert","name":"a","query":"foo","expected":1}]}',
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "sample"
    assert scenario.source_path == scenario_path
    assert isinstance(scenario.steps[0], AssertStep)


def test_load_yaml_scenario_without_pyyaml_dependency(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: sample_yaml",
                "steps:",
                "  - kind: assert",
                "    name: ready",
                "    query: status",
                "    expected: true",
            ]
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "sample_yaml"
    assert scenario.steps[0].name == "ready"


def test_scenario_from_mapping_rejects_missing_steps():
    try:
        scenario_from_mapping({"name": "broken", "steps": []})
    except ValueError as exc:
        assert "non-empty steps list" in str(exc)
    else:
        raise AssertionError("Expected validation failure")
