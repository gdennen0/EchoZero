from pathlib import Path

from echozero.testing.e2e.runner import Runner
from echozero.testing.e2e.scenario import AssertStep, CaptureStep, Scenario, WaitStep


class FakeDriver:
    def __init__(self) -> None:
        self.queries = {"status": "ready"}
        self.wait_calls: list[int] = []

    def click(self, target: str, *, args=None):
        return None

    def type_text(self, target: str, text: str, *, args=None):
        return None

    def press_key(self, key: str, *, args=None):
        return None

    def drag(self, target: str, destination, *, args=None):
        return None

    def dispatch_intent(self, intent_name: str, payload=None):
        return None

    def query_state(self, query: str):
        return self.queries[query]

    def capture_screenshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text("fake-image", encoding="utf-8")
        return output

    def wait(self, duration_ms: int) -> None:
        self.wait_calls.append(duration_ms)


def test_runner_executes_assert_wait_and_capture_steps(tmp_path):
    scenario = Scenario(
        name="runner_sample",
        steps=[
            AssertStep(name="status_ready", query="status", expected="ready"),
            WaitStep(name="brief_wait", duration_ms=5),
            CaptureStep(name="final_frame", artifact="screenshot"),
        ],
    )

    result = Runner(FakeDriver()).run(scenario, tmp_path)

    assert result.passed is True
    assert len(result.steps) == 3
    assert result.steps[2].output["path"].endswith("final_frame.png")
    assert (tmp_path / "final_frame.png").exists()


def test_runner_stops_on_assertion_failure(tmp_path):
    scenario = Scenario(
        name="runner_failure",
        steps=[
            AssertStep(name="bad_assert", query="status", expected="not-ready"),
            CaptureStep(name="should_not_run", artifact="screenshot"),
        ],
    )

    result = Runner(FakeDriver()).run(scenario, tmp_path)

    assert result.passed is False
    assert len(result.steps) == 1
    assert "Assertion failed" in (result.steps[0].error or "")
