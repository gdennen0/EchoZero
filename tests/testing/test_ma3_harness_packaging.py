from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile


def _load_dev_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "MA3" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ma3_harness_transfer_package_contains_install_payload_and_manifest(
    tmp_path: Path,
) -> None:
    module = _load_dev_module("package_ma3_harness")
    archive_path = tmp_path / "ma3-harness-transfer.zip"
    staging_dir = tmp_path / "staging"

    result = module.build_transfer_package(
        archive_path,
        repo_root=Path(__file__).resolve().parents[2],
        keep_staging_dir=staging_dir,
    )

    assert result.output_path == archive_path.resolve()
    assert result.archive_root == "echozero-ma3-harness"
    assert archive_path.is_file()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "echozero-ma3-harness/INSTALL.md" in names
        assert "echozero-ma3-harness/manifest.json" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/Ez#2.xml" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/TC22.xml" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/EZ/ez_core.lua" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/EZ/ez_osc.lua" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/EZ/ez_sequence.lua" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/HitMaker/main.lua" in names
        assert "echozero-ma3-harness/grandMA3/datapools/plugins/TC22/autosave_showfile.lua" in names
        assert "echozero-ma3-harness/source/MA3/dev/ma3_harness_cli.py" in names
        assert "echozero-ma3-harness/source/MA3/plugins/echozero.lua" in names
        assert not any("__pycache__" in name for name in names)
        assert not any(name.endswith(".pyc") for name in names)
        assert not any(name.endswith(".DS_Store") for name in names)

        manifest = json.loads(archive.read("echozero-ma3-harness/manifest.json"))

    assert manifest["schema"] == "echozero.ma3-harness-transfer.v1"
    assert manifest["install"]["copy_payload_from"] == "grandMA3/datapools/plugins/"
    assert manifest["install"]["reload_command"] == "RP"
    assert manifest["install"]["validation_command"] == (
        "python MA3/dev/ma3_harness_cli.py --json smoke"
    )
    manifest_paths = {item["path"] for item in manifest["contents"]["files"]}
    assert "grandMA3/datapools/plugins/Ez#2.xml" in manifest_paths
    assert "grandMA3/datapools/plugins/TC22.xml" in manifest_paths
    assert "grandMA3/datapools/plugins/EZ/ez_core.lua" in manifest_paths
    assert "source/MA3/dev/ma3_harness_cli.py" in manifest_paths
    assert len(manifest_paths) == manifest["contents"]["file_count"]


def test_ma3_harness_transfer_package_cli_emits_json(capsys, tmp_path: Path) -> None:
    module = _load_dev_module("package_ma3_harness")
    archive_path = tmp_path / "ma3-harness-transfer.zip"

    result = module.main(
        [
            "--json",
            "--output",
            str(archive_path),
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["archive_path"] == str(archive_path.resolve())
    assert payload["archive_root"] == "echozero-ma3-harness"
    assert payload["file_count"] > 0
    assert archive_path.is_file()
