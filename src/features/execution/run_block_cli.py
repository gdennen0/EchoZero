"""
Run Block CLI – dumb subprocess runner for one block.

Runs in a separate process so the main (Qt) process is fully decoupled:
no GIL sharing, no blocking. Single responsibility: run one block, stream
progress as JSON lines to stdout, then one result line and exit.

Usage (from repo root):
    python -m src.features.execution.run_block_cli --db <path> --project <id> --block <id>

Protocol (one JSON object per line, utf-8):
    {"type": "progress", "message": "...", "percentage": 0, "current": 0, "total": 100}
    {"type": "result", "success": true, "message": "...", "errors": []}
    {"type": "error", "message": "...", "traceback": "..."}

Main app integration (optional):
    Set env ECHOZERO_USE_SUBPROCESS_RUNNER=1 to run blocks via this CLI
    instead of RunBlockThread. MainWindow will spawn:
      python -m src.features.execution.run_block_cli --db <path> --project <id> --block <id>
    and read stdout for progress (type=progress) and final result (type=result/error).
    Fully decoupled: no GIL sharing, no blocking of the UI process.
"""
import os

# The parent Qt process sets *_NUM_THREADS=1 to prevent NumPy/Qt threading
# crashes. This subprocess has no Qt, so restore full CPU parallelism BEFORE
# importing NumPy/PyTorch (they read these at import time).
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.pop(_var, None)

import argparse
import json
import sys
import warnings
from datetime import datetime
from typing import Optional, Any

# Suppress known third-party warnings that clutter the execution log (stderr).
# Applied before bootstrap so they take effect when deps (google.api_core, librosa) load.
def _suppress_noisy_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="google.api_core._python_version_support",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )


def _emit(line: dict) -> None:
    """Print one JSON line to stdout and flush (for parent process)."""
    if "timestamp" not in line:
        line = {**line, "timestamp": datetime.now().strftime("%H:%M:%S")}
    print(json.dumps(line), flush=True)


class StdoutProgressTracker:
    """
    Minimal progress tracker that writes JSON lines to stdout.
    Same interface as ProgressTracker (start, update, complete) so block
    processors work unchanged; used as progress_tracker_override in subprocess.
    """

    def __init__(self) -> None:
        self._total: Optional[int] = None
        self._current: int = 0
        self._started: bool = False

    def start(
        self,
        message: str = "",
        total: Optional[int] = None,
        current: int = 0,
    ) -> None:
        self._total = total
        self._current = current
        self._started = True
        pct = int((current / total) * 100) if total and total > 0 else 0
        _emit({
            "type": "progress",
            "message": message or "Starting...",
            "percentage": max(0, min(100, pct)),
            "current": current,
            "total": total,
        })

    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        increment: int = 0,
    ) -> None:
        if not self._started:
            self.start(message or "", total=total, current=current or 0)
            return
        if current is not None:
            self._current = current
        elif increment != 0:
            self._current += increment
        if total is not None:
            self._total = total
        pct = int((self._current / self._total) * 100) if self._total and self._total > 0 else 0
        _emit({
            "type": "progress",
            "message": message or "",
            "percentage": max(0, min(100, pct)),
            "current": self._current,
            "total": self._total,
        })

    def complete(self, message: Optional[str] = None) -> None:
        _emit({
            "type": "progress",
            "message": message or "Complete",
            "percentage": 100,
            "current": self._total or self._current,
            "total": self._total,
        })


def _redirect_logging_to_stderr() -> None:
    """Send all Log/logging output to stderr so it doesn't corrupt the stdout JSON protocol.

    The parent process captures stderr and shows it in the execution panel.
    """
    import logging

    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.stream = sys.stderr
    for logger_name in list(logging.Logger.manager.loggerDict):
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.stream = sys.stderr


def main() -> int:
    _suppress_noisy_warnings()

    parser = argparse.ArgumentParser(
        description="Run one EchoZero block in a subprocess. Progress and result on stdout as JSON lines."
    )
    parser.add_argument("--db", required=True, help="Path to EchoZero database (e.g. .ez file)")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--block", required=True, help="Block ID or name")
    args = parser.parse_args()

    try:
        from src.application.bootstrap import initialize_services
    except Exception as e:
        _emit({"type": "error", "message": f"Import failed: {e}", "traceback": ""})
        return 1

    container = None
    try:
        container = initialize_services(
            db_path=args.db,
            progress_tracker=None,
            clear_runtime_tables=False,
        )
        _redirect_logging_to_stderr()
        import time as _time
        _t_bootstrap = _time.perf_counter()

        facade = container.facade
        facade.current_project_id = args.project
        progress = StdoutProgressTracker()

        _t_exec_start = _time.perf_counter()
        print(
            f"[run_block_cli] bootstrap={_t_exec_start - _t_bootstrap:.1f}s, "
            f"device env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}",
            file=sys.stderr, flush=True,
        )

        result = facade.execute_block(args.block, progress_tracker_override=progress)

        _t_done = _time.perf_counter()
        print(
            f"[run_block_cli] execution={_t_done - _t_exec_start:.1f}s, "
            f"total={_t_done - _t_bootstrap:.1f}s",
            file=sys.stderr, flush=True,
        )

        _emit({
            "type": "result",
            "success": result.success,
            "message": result.message or "",
            "errors": list(result.errors) if result.errors else [],
        })
        return 0 if result.success else 1
    except Exception as e:
        import traceback
        _emit({
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
        return 1
    finally:
        if container is not None:
            try:
                container.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
