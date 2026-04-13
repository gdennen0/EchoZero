from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


_TESTING_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_testing")


@pytest.fixture
def tmp_path():
    root = _TESTING_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)
