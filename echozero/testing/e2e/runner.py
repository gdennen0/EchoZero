"""Scenario runner and structured run results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import VideoCaptureHandle, capture_screenshot, start_video_capture
from .driver import E2EDriver
from .scenario import ActStep, AssertStep, CaptureStep, Scenario, ScenarioStep, WaitStep


@dataclass(slots=True)
class StepResult:
    index: int
    name: str
    kind: str
    passed: bool
    duration_ms: int
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    scenario_name: str
    driver_name: str
    artifacts_dir: str
    passed: bool
    duration_ms: int
    started_at: float
    finished_at: float
    steps: list[StepResult]


class Runner:
    def __init__(self, driver: E2EDriver, *, stop_on_failure: bool = True):
        self._driver = driver
        self._stop_on_failure = stop_on_failure
        self._video_handles: dict[str, VideoCaptureHandle] = {}

    def run(self, scenario: Scenario, artifacts_dir: str | Path) -> RunResult:
        root = Path(artifacts_dir)
        root.mkdir(parents=True, exist_ok=True)
        started = time.time()
        started_perf = time.perf_counter()
        step_results: list[StepResult] = []
        passed = True
        for index, step in enumerate(scenario.steps):
            step_result = self._run_step(index, step, root)
            step_results.append(step_result)
            if not step_result.passed:
                passed = False
                if self._stop_on_failure:
                    break
        for handle in self._video_handles.values():
            handle.stop()
        self._video_handles.clear()
        finished = time.time()
        duration_ms = int((time.perf_counter() - started_perf) * 1000)
        return RunResult(
            scenario_name=scenario.name,
            driver_name=type(self._driver).__name__,
            artifacts_dir=str(root),
            passed=passed,
            duration_ms=duration_ms,
            started_at=started,
            finished_at=finished,
            steps=step_results,
        )

    def _run_step(self, index: int, step: ScenarioStep, artifacts_dir: Path) -> StepResult:
        started = time.perf_counter()
        try:
            output = self._execute_step(step, artifacts_dir)
            return StepResult(
                index=index,
                name=step.name,
                kind=step.kind,
                passed=True,
                duration_ms=int((time.perf_counter() - started) * 1000),
                output=output,
            )
        except Exception as exc:
            return StepResult(
                index=index,
                name=step.name,
                kind=step.kind,
                passed=False,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    def _execute_step(self, step: ScenarioStep, artifacts_dir: Path) -> dict[str, Any]:
        if isinstance(step, ActStep):
            return self._execute_act(step)
        if isinstance(step, AssertStep):
            return self._execute_assert(step)
        if isinstance(step, CaptureStep):
            return self._execute_capture(step, artifacts_dir)
        if isinstance(step, WaitStep):
            return self._execute_wait(step)
        raise TypeError(f"Unsupported step type: {type(step)!r}")

    def _execute_act(self, step: ActStep) -> dict[str, Any]:
        if step.action == "click":
            self._driver.click(step.target or "", args=step.args)
        elif step.action == "type":
            self._driver.type_text(step.target or "", str(step.value or ""), args=step.args)
        elif step.action == "key":
            self._driver.press_key(str(step.value or step.target or ""), args=step.args)
        elif step.action == "drag":
            self._driver.drag(step.target or "", step.value, args=step.args)
        elif step.action == "intent":
            intent_name = str(step.value or step.target or "")
            self._driver.dispatch_intent(intent_name, step.args)
        else:
            raise ValueError(f"Unsupported action: {step.action}")
        return {}

    def _execute_assert(self, step: AssertStep) -> dict[str, Any]:
        actual = self._driver.query_state(step.query)
        if not _compare(actual, step.expected, step.comparator):
            raise AssertionError(
                f"Assertion failed for {step.query}: actual={actual!r}, "
                f"expected={step.expected!r}, comparator={step.comparator}"
            )
        return {"actual": actual}

    def _execute_capture(self, step: CaptureStep, artifacts_dir: Path) -> dict[str, Any]:
        output_path = artifacts_dir / (step.path or f"{step.name}.{_capture_extension(step.artifact)}")
        if step.artifact == "screenshot":
            captured = capture_screenshot(self._driver, output_path)
            return {"path": str(captured)}
        if step.artifact == "video_start":
            handle = start_video_capture(
                output_path,
                command_template=step.options.get("command_template"),
                context=step.options.get("context"),
            )
            self._video_handles[step.name] = handle
            return {"path": str(handle.output_path), "command": list(handle.command or ())}
        if step.artifact == "video_stop":
            handle_name = str(step.options.get("handle", step.name))
            handle = self._video_handles.pop(handle_name, None)
            if handle is None:
                raise ValueError(f"Unknown video capture handle: {handle_name}")
            return {"path": str(handle.stop())}
        raise ValueError(f"Unsupported capture artifact: {step.artifact}")

    def _execute_wait(self, step: WaitStep) -> dict[str, Any]:
        if step.duration_ms > 0:
            self._driver.wait(step.duration_ms)
        if step.until_query is None:
            return {}

        timeout_at = time.perf_counter() + (step.timeout_ms / 1000.0)
        while True:
            actual = self._driver.query_state(step.until_query)
            if actual == step.expected:
                return {"actual": actual}
            if time.perf_counter() >= timeout_at:
                raise TimeoutError(
                    f"Timed out waiting for {step.until_query} to equal {step.expected!r}; "
                    f"last value was {actual!r}"
                )
            self._driver.wait(step.poll_interval_ms)


def _compare(actual: Any, expected: Any, comparator: str) -> bool:
    if comparator == "equals":
        return actual == expected
    if comparator == "contains":
        if isinstance(actual, str):
            return str(expected) in actual
        if isinstance(actual, list):
            return expected in actual
        if isinstance(actual, dict):
            return expected in actual.values() or expected in actual.keys()
    if comparator == "truthy":
        return bool(actual)
    raise ValueError(f"Unsupported comparator: {comparator}")


def _capture_extension(artifact: str) -> str:
    if artifact == "screenshot":
        return "png"
    if artifact.startswith("video"):
        return "mp4"
    return "bin"
