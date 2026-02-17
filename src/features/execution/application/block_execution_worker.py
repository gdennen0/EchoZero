"""
Block execution worker for subprocess execution.

Reserved for future process-isolated execution (e.g. sandboxing). Not used for
normal single-block execution: the UI uses RunBlockThread (in-process QThread)
so progress and errors stay in-process without subprocess bootstrap cost.

Supports two modes if used in the future:
- One-off: run_block_in_process() spawns a process that bootstraps once, runs
  one block, then exits (high per-run cost).
- Persistent: run_persistent_worker() runs in a long-lived process that
  bootstraps once and reuses the same container for each request.
"""
import traceback
from typing import Any, Optional

from src.utils.message import Log


def run_block_in_process(
    project_id: str,
    block_id: str,
    result_queue: Any,
    db_path: Optional[str] = None,
) -> None:
    """
    Run a single block execution in this process. Called via multiprocessing.Process.

    Puts a result tuple on result_queue:
      - On success: ("ok", success, message, errors, data)
      - On exception: ("exception", message, traceback_str)
    """
    try:
        from src.application.bootstrap import initialize_services
        from src.utils.paths import get_database_path

        if db_path is None:
            db_path = str(get_database_path("ez"))

        container = initialize_services(
            db_path=db_path,
            progress_tracker=None,
            clear_runtime_tables=False,
        )
        try:
            facade = container.facade
            facade.current_project_id = project_id
            result = facade.execute_block(block_id)
            result_queue.put((
                "ok",
                result.success,
                result.message or "",
                result.errors or [],
                getattr(result, "data", None),
            ))
        finally:
            container.cleanup()
    except Exception as e:
        Log.error(f"Block execution worker failed: {e}")
        result_queue.put(("exception", str(e), traceback.format_exc()))


def run_persistent_worker(request_queue: Any, result_queue: Any) -> None:
    """
    Long-lived worker: bootstrap once, then run execute_block for each request.

    Reads from request_queue: (project_id, block_id, db_path) or None to exit.
    Puts on result_queue: ("ok", success, message, errors, data) or
    ("exception", message, traceback_str).
    """
    container = None
    current_db_path = None

    def ensure_container(db_path: str):
        nonlocal container, current_db_path
        if container is not None and current_db_path == db_path:
            return container
        if container is not None:
            try:
                container.cleanup()
            except Exception as e:
                Log.warning(f"Worker cleanup during reinit: {e}")
            container = None
        from src.application.bootstrap import initialize_services
        Log.info("Block execution worker: initializing services (once per worker)")
        container = initialize_services(
            db_path=db_path,
            progress_tracker=None,
            clear_runtime_tables=False,
        )
        current_db_path = db_path
        return container

    try:
        while True:
            request = request_queue.get()
            if request is None:
                Log.info("Block execution worker: shutdown requested")
                break
            project_id, block_id, db_path = request
            if db_path is None:
                from src.utils.paths import get_database_path
                db_path = str(get_database_path("ez"))
            try:
                ensure_container(db_path)
                facade = container.facade
                facade.current_project_id = project_id
                result = facade.execute_block(block_id)
                result_queue.put((
                    "ok",
                    result.success,
                    result.message or "",
                    result.errors or [],
                    getattr(result, "data", None),
                ))
            except Exception as e:
                Log.error(f"Block execution worker failed: {e}")
                result_queue.put(("exception", str(e), traceback.format_exc()))
    finally:
        if container is not None:
            try:
                container.cleanup()
            except Exception as e:
                Log.warning(f"Block execution worker cleanup: {e}")
