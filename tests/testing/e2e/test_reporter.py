import json

from echozero.testing.e2e.reporter import write_run_report
from echozero.testing.e2e.runner import RunResult, StepResult


def test_write_run_report_emits_json_and_markdown(tmp_path):
    result = RunResult(
        scenario_name="demo",
        driver_name="FakeDriver",
        artifacts_dir=str(tmp_path),
        passed=True,
        duration_ms=12,
        started_at=1.0,
        finished_at=2.0,
        steps=[
            StepResult(
                index=0,
                name="step_one",
                kind="assert",
                passed=True,
                duration_ms=4,
                output={"actual": "ok"},
            )
        ],
    )

    json_path, markdown_path = write_run_report(result, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["scenario_name"] == "demo"
    assert "Status: PASS" in markdown
    assert "step_one" in markdown
