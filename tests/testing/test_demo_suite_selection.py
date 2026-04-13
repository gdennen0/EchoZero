from __future__ import annotations

from pathlib import Path

import pytest

from echozero.testing import demo_suite
from echozero.testing.demo_suite_scenarios import ScenarioResult


def test_list_scenarios_prints_declared_order(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(demo_suite, "SCENARIO_ORDER", ["alpha", "beta", "gamma"])

    result = demo_suite.main(["--list-scenarios"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.strip().splitlines() == ["alpha", "beta", "gamma"]


def test_run_demo_suite_only_runs_requested_scenarios_in_declared_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[str] = []

    def fake_runner(name: str):
        def _runner(**_: object) -> list[ScenarioResult]:
            calls.append(name)
            return [ScenarioResult(group=name, name="result", status="passed")]

        return _runner

    monkeypatch.setattr(demo_suite, "SCENARIO_ORDER", ["alpha", "beta", "gamma"])
    monkeypatch.setattr(
        demo_suite,
        "SCENARIO_RUNNERS",
        {
            "alpha": fake_runner("alpha"),
            "beta": fake_runner("beta"),
            "gamma": fake_runner("gamma"),
        },
    )
    monkeypatch.setattr(demo_suite, "timestamp_run_id", lambda now=None: "20260413-010203")

    _, manifest_path, manifest = demo_suite.run_demo_suite(
        output_root=tmp_path / "demo-suite",
        scenarios=["gamma", "alpha"],
    )

    assert calls == ["alpha", "gamma"]
    assert [item["group"] for item in manifest["scenario_statuses"]] == ["alpha", "gamma"]
    assert manifest_path.name == "manifest.json"


def test_run_demo_suite_defaults_to_all_scenarios_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[str] = []

    def fake_runner(name: str):
        def _runner(**_: object) -> list[ScenarioResult]:
            calls.append(name)
            return [ScenarioResult(group=name, name="result", status="passed")]

        return _runner

    monkeypatch.setattr(demo_suite, "SCENARIO_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        demo_suite,
        "SCENARIO_RUNNERS",
        {
            "one": fake_runner("one"),
            "two": fake_runner("two"),
            "three": fake_runner("three"),
        },
    )
    monkeypatch.setattr(demo_suite, "timestamp_run_id", lambda now=None: "20260413-010204")

    _, _, manifest = demo_suite.run_demo_suite(output_root=tmp_path / "demo-suite")

    assert calls == ["one", "two", "three"]
    assert [item["group"] for item in manifest["scenario_statuses"]] == ["one", "two", "three"]
