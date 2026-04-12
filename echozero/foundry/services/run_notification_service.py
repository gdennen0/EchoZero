from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from echozero.foundry.domain import TrainRun, TrainRunStatus


_TERMINAL_STATUSES = {TrainRunStatus.COMPLETED, TrainRunStatus.FAILED, TrainRunStatus.CANCELED}
_ACTIVE_STATUSES = {
    TrainRunStatus.QUEUED,
    TrainRunStatus.PREPARING,
    TrainRunStatus.RUNNING,
    TrainRunStatus.EVALUATING,
    TrainRunStatus.EXPORTING,
}


class RunNotificationService:
    def __init__(self, root: Path):
        self._root = root

    def emit_notification_cadence(
        self,
        run: TrainRun,
        *,
        event_type: str,
        list_runs: Callable[[], list[TrainRun]],
        notify: Callable[[str], None],
    ) -> None:
        state = self.read_notification_state()

        if event_type == "RUN_STARTED":
            self.notify_openclaw_deduped(
                f"start:{run.id}",
                f"Foundry run started: {run.id} ({run.spec.get('model', {}).get('type', 'baseline_sgd')})",
                cooldown_seconds=45,
                state=state,
                notify=notify,
            )

        if run.status in _TERMINAL_STATUSES:
            runs = list_runs()
            terminal_runs = [item for item in runs if item.status in _TERMINAL_STATUSES]
            active_runs = [item for item in runs if item.status in _ACTIVE_STATUSES]

            if run.status == TrainRunStatus.FAILED and not state.get("first_failure_sent", False):
                self.notify_openclaw_deduped(
                    "first_failure",
                    f"Foundry first failure detected: {run.id}. Check dashboard for details.",
                    cooldown_seconds=60,
                    state=state,
                    notify=notify,
                )
                state["first_failure_sent"] = True

            terminal_count = len(terminal_runs)
            last_milestone = int(state.get("last_milestone_notified", 0))
            if terminal_count >= 3 and terminal_count % 3 == 0 and terminal_count != last_milestone:
                self.notify_openclaw_deduped(
                    f"milestone:{terminal_count}",
                    f"Foundry milestone: {terminal_count} runs reached terminal state.",
                    cooldown_seconds=120,
                    state=state,
                    notify=notify,
                )
                state["last_milestone_notified"] = terminal_count

            if not active_runs:
                completed = sum(1 for item in terminal_runs if item.status == TrainRunStatus.COMPLETED)
                failed = sum(1 for item in terminal_runs if item.status == TrainRunStatus.FAILED)
                canceled = sum(1 for item in terminal_runs if item.status == TrainRunStatus.CANCELED)
                digest_signature = f"{terminal_count}:{completed}:{failed}:{canceled}"
                if digest_signature != state.get("last_digest_signature"):
                    self.notify_openclaw_deduped(
                        f"digest:{digest_signature}",
                        "Foundry final digest: "
                        f"terminal={terminal_count}, completed={completed}, failed={failed}, canceled={canceled}.",
                        cooldown_seconds=180,
                        state=state,
                        notify=notify,
                    )
                    state["last_digest_signature"] = digest_signature

        self.write_notification_state(state)

    def notification_state_path(self) -> Path:
        path = self._root / "foundry" / "tracking" / "notification_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_notification_state(self) -> dict[str, Any]:
        path = self.notification_state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_notification_state(self, state: dict[str, Any]) -> None:
        path = self.notification_state_path()
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def notify_openclaw_deduped(
        self,
        key: str,
        text: str,
        *,
        cooldown_seconds: int,
        state: dict[str, Any],
        notify: Callable[[str], None],
    ) -> None:
        now = datetime.now(UTC).timestamp()
        sent_at = float((state.get("sent", {}) or {}).get(key, 0.0))
        if now - sent_at < max(0, cooldown_seconds):
            return
        notify(text)
        sent = dict(state.get("sent", {}) or {})
        sent[key] = now
        state["sent"] = sent

    def notify_openclaw(self, text: str) -> None:
        try:
            subprocess.run(
                ["openclaw", "system", "event", "--text", text, "--mode", "now"],
                cwd=str(self._root),
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            pass
