from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path("/Users/march/Documents/GitHub/EchoZero")


def _load_script_module(name: str):
    script_path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_architecture_boundary_check_rejects_ui_import_in_guarded_layer(tmp_path: Path):
    module = _load_script_module("check_architecture_boundaries")
    guarded_dir = tmp_path / "echozero" / "application"
    guarded_dir.mkdir(parents=True)
    (guarded_dir / "bad_module.py").write_text(
        '"""Bad module.\nExists for test coverage.\nConnects nothing valid.\n"""\n'
        "from PyQt6.QtWidgets import QWidget\n",
        encoding="utf-8",
    )

    result = module.main([str(tmp_path)])

    assert result == 1


def test_architecture_boundary_check_accepts_clean_guarded_layer(tmp_path: Path):
    module = _load_script_module("check_architecture_boundaries")
    guarded_dir = tmp_path / "echozero" / "services"
    guarded_dir.mkdir(parents=True)
    (guarded_dir / "good_module.py").write_text(
        '"""Good module.\nExists for test coverage.\nConnects nothing invalid.\n"""\n'
        "from echozero.domain import models\n",
        encoding="utf-8",
    )

    result = module.main([str(tmp_path)])

    assert result == 0


def test_canonical_launcher_check_rejects_legacy_launcher_file(tmp_path: Path):
    module = _load_script_module("check_canonical_launcher")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (tmp_path / "run_echozero.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('legacy')\n", encoding="utf-8")
    (scripts_dir / "build-test-release.ps1").write_text("run_echozero.py\n", encoding="utf-8")
    (scripts_dir / "smoke-test-release.ps1").write_text("Write-Host 'ok'\n", encoding="utf-8")

    result = module.main([str(tmp_path)])

    assert result == 1


def test_pr_requirements_require_decision_and_contract_mapping(tmp_path: Path, monkeypatch):
    module = _load_script_module("check_pr_requirements")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "## Verification\n- appflow\n",
                    "base": {"sha": "base"},
                    "head": {"sha": "head"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: ["echozero/ui/qt/timeline/widget.py"],
    )

    result = module.main()

    assert result == 1


def test_pr_requirements_accept_valid_sync_contract_mapping(tmp_path: Path, monkeypatch):
    module = _load_script_module("check_pr_requirements")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Implements D30 and FP7.",
                            "- [x] Sync boundary",
                            "- [x] Timeline truth model",
                            "Verification: appflow-sync appflow-protocol",
                        ]
                    ),
                    "base": {"sha": "base"},
                    "head": {"sha": "head"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: ["echozero/application/sync/service.py"],
    )

    result = module.main()

    assert result == 0


def test_timeline_feel_guardrails_reject_numeric_default_in_changed_widget_file(tmp_path: Path, monkeypatch):
    module = _load_script_module("check_timeline_feel_guardrails")
    guarded_file = tmp_path / "echozero" / "ui" / "qt" / "timeline" / "widget.py"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_text(
        '"""Widget.\nExists for test coverage.\nConnects FEEL guardrails.\n"""\n'
        "def build_widget(*, width: int = 1440):\n"
        "    return width\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: [guarded_file],
    )

    result = module.main(["base", "head"])

    assert result == 1


def test_timeline_feel_guardrails_accept_feel_constant_in_changed_widget_file(tmp_path: Path, monkeypatch):
    module = _load_script_module("check_timeline_feel_guardrails")
    guarded_file = tmp_path / "echozero" / "ui" / "qt" / "timeline" / "widget.py"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_text(
        '"""Widget.\nExists for test coverage.\nConnects FEEL guardrails.\n"""\n'
        "from echozero.ui.FEEL import LAYER_HEADER_WIDTH_PX\n"
        "def build_widget(*, width: int = LAYER_HEADER_WIDTH_PX):\n"
        "    return width\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: [guarded_file],
    )

    result = module.main(["base", "head"])

    assert result == 0


def test_lane_ownership_check_accepts_non_lane_branch(monkeypatch):
    module = _load_script_module("check_lane_ownership")
    monkeypatch.setattr(module, "_git_current_branch", lambda repo_root: "feature/test")

    result = module.main([])

    assert result == 0


def test_lane_ownership_uses_github_head_ref_when_present(monkeypatch):
    module = _load_script_module("check_lane_ownership")
    monkeypatch.setenv("GITHUB_HEAD_REF", "lane/ez/contract-work")
    monkeypatch.setattr(module, "_git_current_branch", lambda repo_root: "HEAD")
    monkeypatch.setattr(module, "_git_changed_files", lambda repo_root, *, base_ref: [])

    result = module.main([])

    assert result == 0


def test_changed_python_style_ignores_preexisting_legacy_size_and_docstring_violations(
    tmp_path: Path, monkeypatch
):
    """Legacy touched files should not fail for unchanged historical violations."""

    module = _load_script_module("check_changed_python_style")
    guarded_file = tmp_path / "echozero" / "ui" / "qt" / "timeline" / "widget.py"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_text(
        '"""Legacy widget file.\nExists for test coverage.\nConnects incremental style rules.\n"""\n'
        + "\n".join(["x = 1"] * 505)
        + "\n"
        + "def build_widget():\n"
        + "    return 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: [guarded_file],
    )
    monkeypatch.setattr(
        module,
        "_git_show_file",
        lambda repo_root, *, revision, relative_path: '"""Legacy widget file.\nExists for test coverage.\nConnects incremental style rules.\n"""\n'
        + "\n".join(["x = 1"] * 505)
        + "\n"
        + "def build_widget():\n"
        + "    return 0\n",
    )

    result = module.main(["base", "head"])

    assert result == 0


def test_changed_python_style_rejects_new_public_symbol_without_docstring(tmp_path: Path, monkeypatch):
    """New public symbols must still carry docstrings under incremental enforcement."""

    module = _load_script_module("check_changed_python_style")
    guarded_file = tmp_path / "echozero" / "application" / "sync" / "service.py"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_text(
        '"""Sync service.\nExists for test coverage.\nConnects incremental style rules.\n"""\n'
        "def new_public_symbol():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: [guarded_file],
    )
    monkeypatch.setattr(
        module,
        "_git_show_file",
        lambda repo_root, *, revision, relative_path: '"""Sync service.\nExists for test coverage.\nConnects incremental style rules.\n"""\n',
    )

    result = module.main(["base", "head"])

    assert result == 1


def test_changed_python_style_allows_new_test_functions_without_docstrings(tmp_path: Path, monkeypatch):
    """Test modules should not need per-test docstrings to satisfy the guardrail."""

    module = _load_script_module("check_changed_python_style")
    guarded_file = tmp_path / "tests" / "ui" / "test_widget.py"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_text(
        '"""Widget tests.\nExists for test coverage.\nConnects incremental style rules.\n"""\n'
        "def test_widget_draws():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_changed_files",
        lambda repo_root, *, base, head: [guarded_file],
    )
    monkeypatch.setattr(
        module,
        "_git_show_file",
        lambda repo_root, *, revision, relative_path: '"""Widget tests.\nExists for test coverage.\nConnects incremental style rules.\n"""\n',
    )

    result = module.main(["base", "head"])

    assert result == 0
