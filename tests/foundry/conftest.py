from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import _pytest.pathlib
import _pytest.tmpdir
import pytest


_BASE_TMP_PARENT = Path(__file__).resolve().parents[2] / ".foundry-test-tmp"


def pytest_configure(config) -> None:
    _BASE_TMP_PARENT.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(_BASE_TMP_PARENT / uuid4().hex)
    _pytest.pathlib.cleanup_dead_symlinks = lambda root: None
    _pytest.tmpdir.cleanup_dead_symlinks = lambda root: None


@pytest.fixture
def tmp_path() -> Path:
    path = _BASE_TMP_PARENT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
