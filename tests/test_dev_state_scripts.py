"""
Dev-state script tests: prove explicit cross-machine state export/import works.
Exists to keep laptop handoff paths deterministic instead of tribal.
Connects CLI helper modules to canonical settings/models payload behavior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile


def _load_script_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_export_dev_state_writes_manifest_and_selected_payloads(tmp_path: Path) -> None:
    export_module = _load_script_module("export_dev_state")
    settings_path = tmp_path / "config" / "app-settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text('{"audio_output": {"sample_rate": 48000}}\n', encoding="utf-8")
    models_dir = tmp_path / "models"
    bundle_dir = models_dir / "snare-v1"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "snare.manifest.json").write_text('{"classes":["snare","other"]}\n', encoding="utf-8")
    (bundle_dir / "weights.pt").write_text("weights\n", encoding="utf-8")

    archive_path = tmp_path / "handoff" / "echozero-dev-state.zip"
    result = export_module.export_dev_state(
        archive_path,
        selection=export_module.ExportSelection(
            settings_path=settings_path,
            models_dir=models_dir,
        ),
        force=False,
    )

    assert result["exported_settings"] is True
    assert result["exported_models"] is True
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "settings/app-settings.json" in names
        assert "models/snare-v1/snare.manifest.json" in names
        assert "models/snare-v1/weights.pt" in names


def test_import_dev_state_restores_settings_and_models_into_requested_paths(tmp_path: Path) -> None:
    export_module = _load_script_module("export_dev_state")
    import_module = _load_script_module("import_dev_state")

    source_settings = tmp_path / "source-config" / "app-settings.json"
    source_settings.parent.mkdir(parents=True, exist_ok=True)
    source_settings.write_text('{"ma3_osc": {"receive": {"port": 7001}}}\n', encoding="utf-8")
    source_models = tmp_path / "source-models"
    source_bundle = source_models / "kick-v1"
    source_bundle.mkdir(parents=True)
    (source_bundle / "kick.manifest.json").write_text('{"classes":["kick","other"]}\n', encoding="utf-8")
    (source_bundle / "weights.pt").write_text("kick\n", encoding="utf-8")

    archive_path = tmp_path / "export" / "echozero-dev-state.zip"
    export_module.export_dev_state(
        archive_path,
        selection=export_module.ExportSelection(
            settings_path=source_settings,
            models_dir=source_models,
        ),
        force=False,
    )

    target_settings = tmp_path / "target-config" / "app-settings.json"
    target_models = tmp_path / "target-models"
    result = import_module.import_dev_state(
        archive_path,
        settings_path=target_settings,
        models_dir=target_models,
        import_settings=True,
        import_models=True,
        force=False,
    )

    assert result["imported_settings"] is True
    assert result["imported_models"] is True
    assert target_settings.read_text(encoding="utf-8") == source_settings.read_text(encoding="utf-8")
    assert (target_models / "kick-v1" / "kick.manifest.json").read_text(encoding="utf-8") == (
        source_bundle / "kick.manifest.json"
    ).read_text(encoding="utf-8")
    assert (target_models / "kick-v1" / "weights.pt").read_text(encoding="utf-8") == "kick\n"
