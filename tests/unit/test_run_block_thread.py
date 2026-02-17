"""
Unit tests for RunBlockThread.

RunBlockThread runs facade.execute_block(block_id) in a background QThread so the
UI stays responsive. Progress and errors stay in-process; no subprocess is used.
"""
import pytest
from unittest.mock import MagicMock

from src.application.api.result_types import CommandResult


@pytest.fixture
def qapp():
    """Ensure QApplication exists for QThread and signals."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_facade_success():
    """Facade that returns success from execute_block."""
    facade = MagicMock()
    facade.current_project_id = "test_project"
    facade.execute_block = MagicMock(
        return_value=CommandResult.success_result("OK", data={"block": None, "outputs": {}})
    )
    return facade


@pytest.fixture
def mock_facade_failure():
    """Facade that returns failure from execute_block."""
    facade = MagicMock()
    facade.current_project_id = "test_project"
    facade.execute_block = MagicMock(
        return_value=CommandResult.error_result("Block failed", errors=["detail1", "detail2"])
    )
    return facade


@pytest.fixture
def mock_facade_raises():
    """Facade that raises from execute_block."""
    facade = MagicMock()
    facade.current_project_id = "test_project"
    facade.execute_block = MagicMock(side_effect=RuntimeError("execute_block raised"))
    return facade


class TestRunBlockThread:
    """Tests for RunBlockThread signal emission and in-process execution."""

    def test_run_block_thread_emits_complete_on_success(self, qapp, mock_facade_success):
        """Thread emits execution_complete(True) when facade returns success."""
        from ui.qt_gui.core.run_block_thread import RunBlockThread

        results = []
        thread = RunBlockThread(mock_facade_success, "block_1", parent=None)
        thread.execution_complete.connect(lambda ok: results.append(("complete", ok)))
        thread.execution_failed.connect(lambda msg, errs: results.append(("failed", msg, errs)))
        thread.start()
        thread.wait(5000)
        qapp.processEvents()

        assert thread.isFinished()
        assert results == [("complete", True)]
        mock_facade_success.execute_block.assert_called_once_with("block_1")

    def test_run_block_thread_emits_failed_on_error_result(self, qapp, mock_facade_failure):
        """Thread emits execution_failed when facade returns error result."""
        from ui.qt_gui.core.run_block_thread import RunBlockThread

        results = []
        thread = RunBlockThread(mock_facade_failure, "block_2", parent=None)
        thread.execution_complete.connect(lambda ok: results.append(("complete", ok)))
        thread.execution_failed.connect(lambda msg, errs: results.append(("failed", msg, errs)))
        thread.start()
        thread.wait(5000)
        qapp.processEvents()

        assert thread.isFinished()
        assert len(results) == 1
        assert results[0][0] == "failed"
        assert results[0][1] == "Block failed"
        assert results[0][2] == ["detail1", "detail2"]

    def test_run_block_thread_emits_failed_on_exception(self, qapp, mock_facade_raises):
        """Thread emits execution_failed when facade raises."""
        from ui.qt_gui.core.run_block_thread import RunBlockThread

        results = []
        thread = RunBlockThread(mock_facade_raises, "block_3", parent=None)
        thread.execution_complete.connect(lambda ok: results.append(("complete", ok)))
        thread.execution_failed.connect(lambda msg, errs: results.append(("failed", msg, errs)))
        thread.start()
        thread.wait(5000)
        qapp.processEvents()

        assert thread.isFinished()
        assert len(results) == 1
        assert results[0][0] == "failed"
        assert "execute_block raised" in results[0][1]
        assert "Traceback" in results[0][2][1]

    def test_run_block_thread_requires_block_id(self):
        """RunBlockThread raises ValueError if block_id is empty."""
        from ui.qt_gui.core.run_block_thread import RunBlockThread

        facade = MagicMock()
        with pytest.raises(ValueError, match="block_id is required"):
            RunBlockThread(facade, "", parent=None)
