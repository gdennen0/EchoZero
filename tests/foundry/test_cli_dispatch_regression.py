from __future__ import annotations

from pathlib import Path

import pytest

from echozero.foundry.cli import main


class _DispatchReached(RuntimeError):
    pass


def _raise_dispatch_reached(command: str) -> None:
    raise _DispatchReached(command)


class _DispatchProbeDatasets:
    def create_dataset(self, *_args, **_kwargs):
        _raise_dispatch_reached("train-folder")


class _DispatchProbeRuntimeBundles:
    def install_binary_drum_artifact(self, *_args, **_kwargs):
        _raise_dispatch_reached("install-runtime-bundle")


class _DispatchProbeFoundryApp:
    def __init__(self, _root: Path) -> None:
        self.datasets = _DispatchProbeDatasets()
        self.runtime_bundles = _DispatchProbeRuntimeBundles()

    def create_run(self, *_args, **_kwargs):
        _raise_dispatch_reached("create-run")

    def start_run(self, *_args, **_kwargs):
        _raise_dispatch_reached("start-run")

    def validate_artifact(self, *_args, **_kwargs):
        _raise_dispatch_reached("validate-artifact")


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    [
        (["train-folder", "Dispatch Drums", "unused-folder"], "train-folder"),
        (["create-run", "dsv_dispatch", "{}"], "create-run"),
        (["start-run", "run_dispatch"], "start-run"),
        (["install-runtime-bundle", "art_dispatch"], "install-runtime-bundle"),
        (["validate-artifact", "art_dispatch"], "validate-artifact"),
        (["ui"], "ui"),
    ],
)
def test_cli_commands_after_plan_version_are_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_command: str
) -> None:
    def fake_run_foundry_ui(_root: Path) -> int:
        _raise_dispatch_reached("ui")

    monkeypatch.setattr("echozero.foundry.cli.FoundryApp", _DispatchProbeFoundryApp)
    monkeypatch.setattr("echozero.foundry.cli.run_foundry_ui", fake_run_foundry_ui)

    with pytest.raises(_DispatchReached, match=expected_command):
        main(["--root", str(tmp_path), *argv])
