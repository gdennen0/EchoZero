"""
Tests for the macOS release artifact verifier.
Exists so release gates reject bad zips without needing a real signed app in unit tests.
Connects zip integrity, bundle mutation, UUID, smoke, and asset parity checks to scripts.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_verifier_module():
    script_path = REPO_ROOT / "scripts" / "verify_macos_release_artifact.py"
    spec = importlib.util.spec_from_file_location("verify_macos_release_artifact", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_release_zip(zip_path: Path, *, extra_files: dict[str, bytes] | None = None) -> None:
    files = {
        "EchoZero.app/Contents/MacOS/EchoZero": b"fake-mach-o",
        "EchoZero.app/Contents/Info.plist": b"plist",
        "EchoZero.app/Contents/Resources/icon.icns": b"icon",
    }
    files.update(extra_files or {})
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def _stub_native_checks(
    monkeypatch, module, *, uuid: str = "AC4A6A20-9E69-F717-9BBE-F9025FA69EB7"
) -> None:
    monkeypatch.setattr(module, "extract_binary_uuids", lambda _path: [uuid])
    monkeypatch.setattr(module, "verify_codesign_strict", lambda _path: "valid on disk")
    monkeypatch.setattr(
        module,
        "run_packaged_smoke",
        lambda *_args, **_kwargs: {"status": "passed", "exit_code": 0},
    )


def test_release_verifier_accepts_zip_with_expected_sha_uuid_and_codesign(tmp_path, monkeypatch):
    module = _load_verifier_module()
    archive = tmp_path / "EchoZero-macOS.zip"
    _write_release_zip(archive)
    _stub_native_checks(monkeypatch, module)

    report = module.verify_macos_release_artifact(
        module.VerificationOptions(
            archive=archive,
            expected_sha256=module.compute_sha256(archive),
            expected_binary_uuid="AC4A6A20-9E69-F717-9BBE-F9025FA69EB7",
        )
    )

    assert report.status == "passed"
    assert report.checks["sha256"]["actual"] == module.compute_sha256(archive)
    assert report.checks["codesign_strict"]["passed"] is True


def test_release_verifier_rejects_wrong_zip_sha256(tmp_path, monkeypatch):
    module = _load_verifier_module()
    archive = tmp_path / "EchoZero-macOS.zip"
    _write_release_zip(archive)
    _stub_native_checks(monkeypatch, module)

    report = module.verify_macos_release_artifact(
        module.VerificationOptions(archive=archive, expected_sha256="0" * 64)
    )

    assert report.status == "failed"
    assert [failure.check for failure in report.failures] == ["sha256"]
    assert "redownload" in report.failures[0].action


def test_release_verifier_rejects_runtime_config_file_before_smoke(tmp_path, monkeypatch):
    module = _load_verifier_module()
    archive = tmp_path / "EchoZero-macOS.zip"
    _write_release_zip(
        archive,
        extra_files={"EchoZero.app/Contents/MacOS/config/app-settings.json": b"{}"},
    )
    _stub_native_checks(monkeypatch, module)

    report = module.verify_macos_release_artifact(module.VerificationOptions(archive=archive))

    assert report.status == "failed"
    assert "runtime_config_pre_smoke" in {failure.check for failure in report.failures}
    assert report.checks["runtime_config_pre_smoke"]["files"] == [
        "Contents/MacOS/config/app-settings.json"
    ]


def test_release_verifier_rejects_runtime_config_created_by_smoke(tmp_path, monkeypatch):
    module = _load_verifier_module()
    archive = tmp_path / "EchoZero-macOS.zip"
    _write_release_zip(archive)
    _stub_native_checks(monkeypatch, module)

    def create_config(app_bundle, **_kwargs):
        path = app_bundle / "Contents" / "MacOS" / "config" / "app-settings.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        return {"status": "passed", "exit_code": 0}

    monkeypatch.setattr(module, "run_packaged_smoke", create_config)
    report = module.verify_macos_release_artifact(module.VerificationOptions(archive=archive))

    assert report.status == "failed"
    assert "runtime_config_post_smoke" in {failure.check for failure in report.failures}


def test_release_verifier_compares_named_zip_asset_equivalence(tmp_path, monkeypatch):
    module = _load_verifier_module()
    archive = tmp_path / "EchoZero-macOS.zip"
    renamed = tmp_path / "EchoZero-v1.0.0-macos-arm64.zip"
    _write_release_zip(archive)
    _write_release_zip(
        renamed,
        extra_files={"EchoZero.app/Contents/Resources/icon.icns": b"stale"},
    )
    _stub_native_checks(monkeypatch, module)

    report = module.verify_macos_release_artifact(
        module.VerificationOptions(archive=archive, compare_zip=renamed)
    )

    assert report.status == "failed"
    assert report.checks["asset_equivalence"]["matched"] is False
    assert report.checks["asset_equivalence"]["changed"] == ["Contents/Resources/icon.icns"]
