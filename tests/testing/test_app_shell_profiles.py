from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.ui.qt.app_shell import AppRuntimeProfile, AppShellRuntime, build_app_shell

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_shell_profiles")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_build_app_shell_profiles_route_to_expected_runtime():
    temp_root = _repo_local_temp_root()
    production = build_app_shell(profile=AppRuntimeProfile.PRODUCTION, working_dir_root=temp_root / "prod")
    test_runtime = build_app_shell(profile=AppRuntimeProfile.TEST, working_dir_root=temp_root / "test")

    try:
        assert isinstance(production, AppShellRuntime)
        assert isinstance(test_runtime, AppShellRuntime)
    finally:
        production.shutdown()
        test_runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
