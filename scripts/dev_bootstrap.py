#!/usr/bin/env python3
"""
Developer bootstrap: create the local venv, install EchoZero, and run a smoke lane.
Exists so humans and agents can provision the same workspace with one command.
Connects repo setup to the canonical app path and proof lanes in docs/TESTING.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _run(command: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a local EchoZero dev environment.")
    parser.add_argument("--venv", default=".venv", help="Virtualenv directory to create/use.")
    parser.add_argument(
        "--extras",
        nargs="*",
        default=[],
        choices=["dev", "ml", "packaging"],
        help="Optional dependency groups to install in addition to the default dev set.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the default smoke verification after install.",
    )
    parser.add_argument(
        "--smoke-lane",
        default="appflow",
        help="Testing lane to run when smoke verification is enabled.",
    )
    parsed = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    venv_root = (repo_root / parsed.venv).resolve()
    host_python = Path(sys.executable).resolve()

    if not venv_root.exists():
        _run([str(host_python), "-m", "venv", str(venv_root)], cwd=repo_root)

    venv_python = _venv_python(venv_root)
    extras = {"dev", *parsed.extras}
    extras_suffix = f"[{','.join(sorted(extras))}]"

    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo_root)
    _run([str(venv_python), "-m", "pip", "install", "-e", f".{extras_suffix}"], cwd=repo_root)
    _run([str(venv_python), "scripts/check_repo_hygiene.py"], cwd=repo_root)

    if not parsed.skip_smoke:
        _run([str(venv_python), "-m", "echozero.testing.run", "--lane", parsed.smoke_lane], cwd=repo_root)

    print("")
    print("Bootstrap complete.")
    print(f"Activate: source {parsed.venv}/bin/activate")
    print(f"App: {venv_python} run_echozero.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
