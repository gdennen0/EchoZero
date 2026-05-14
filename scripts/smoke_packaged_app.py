#!/usr/bin/env python3
"""
Packaged app smoke launcher for EchoZero release artifacts.
Exists so macOS .app and extracted one-folder builds get the same launch gate.
Connects release packaging to the canonical run_echozero.py smoke-exit contract.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_SMOKE_EXIT_SECONDS = 6.0


def resolve_packaged_executable(path: Path) -> Path:
    """Resolve an EchoZero app bundle, release folder, or executable to a binary path."""
    candidate = path.resolve()
    if candidate.is_file():
        return candidate
    if candidate.suffix == ".app":
        executable = candidate / "Contents" / "MacOS" / candidate.stem
        if executable.is_file():
            return executable
        raise FileNotFoundError(f"macOS app executable not found: {executable}")

    mac_app = candidate / "EchoZero.app"
    if mac_app.is_dir():
        return resolve_packaged_executable(mac_app)

    executable_names = ("EchoZero", "EchoZero.exe", "EchoZeroTest.exe")
    for name in executable_names:
        direct = candidate / name
        if direct.is_file():
            return direct
        nested = candidate / "EchoZero" / name
        if nested.is_file():
            return nested

    raise FileNotFoundError(f"packaged EchoZero executable not found under: {candidate}")


def build_smoke_command(
    executable: Path,
    *,
    smoke_exit_seconds: float,
    working_dir_root: Path,
    log_dir: Path | None,
) -> list[str]:
    """Build the packaged launch command using the canonical smoke-exit flags."""
    command = [
        str(executable),
        "--smoke-exit-seconds",
        str(smoke_exit_seconds),
        "--working-dir-root",
        str(working_dir_root),
    ]
    if log_dir is not None:
        command.extend(["--log-dir", str(log_dir)])
    return command


def run_packaged_smoke(
    packaged_path: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    smoke_exit_seconds: float = DEFAULT_SMOKE_EXIT_SECONDS,
    playback_start_timeout_seconds: float = 2.0,
    working_dir_root: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, object]:
    """Launch the packaged app, wait for smoke shutdown, and return a report."""
    executable = resolve_packaged_executable(packaged_path)
    smoke_working_dir = working_dir_root or Path(tempfile.mkdtemp(prefix="echozero-smoke-"))
    command = build_smoke_command(
        executable,
        smoke_exit_seconds=smoke_exit_seconds,
        working_dir_root=smoke_working_dir,
        log_dir=log_dir,
    )
    started = time.monotonic()
    env = dict(os.environ)
    env["ECHOZERO_PLAYBACK_START_TIMEOUT_SECONDS"] = str(playback_start_timeout_seconds)
    process = subprocess.Popen(command, env=env)
    status = "failed"
    exit_code: int | None = None
    reason = ""
    try:
        exit_code = process.wait(timeout=timeout_seconds)
        status = "passed" if exit_code == 0 else "failed"
        if exit_code != 0:
            reason = "non_zero_exit"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        status = "timeout"
        reason = "timeout"

    return {
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "packaged_path": str(packaged_path),
        "executable": str(executable),
        "working_dir_root": str(smoke_working_dir),
        "timeout_seconds": timeout_seconds,
        "smoke_exit_seconds": smoke_exit_seconds,
        "playback_start_timeout_seconds": playback_start_timeout_seconds,
        "platform": platform.platform(),
        "reason": reason,
    }


def write_report(report: dict[str, object], report_path: Path) -> None:
    """Write a smoke report as stable JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the packaged smoke command from CLI arguments."""
    parser = argparse.ArgumentParser(description="Smoke-test a packaged EchoZero app.")
    parser.add_argument(
        "packaged_path",
        nargs="?",
        type=Path,
        default=Path("dist") / "EchoZero.app",
        help="Path to EchoZero.app, an executable, or an extracted release folder.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--smoke-exit-seconds", type=float, default=DEFAULT_SMOKE_EXIT_SECONDS)
    parser.add_argument("--playback-start-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--working-dir-root", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("dist") / "packaged-smoke-report.json",
    )
    parsed = parser.parse_args(argv)

    try:
        report = run_packaged_smoke(
            parsed.packaged_path,
            timeout_seconds=parsed.timeout_seconds,
            smoke_exit_seconds=parsed.smoke_exit_seconds,
            playback_start_timeout_seconds=parsed.playback_start_timeout_seconds,
            working_dir_root=parsed.working_dir_root,
            log_dir=parsed.log_dir,
        )
    except Exception as exc:
        report = {
            "status": "failed",
            "exit_code": None,
            "duration_seconds": 0.0,
            "packaged_path": str(parsed.packaged_path),
            "reason": f"{type(exc).__name__}: {exc}",
            "platform": platform.platform(),
        }
    write_report(report, parsed.report_path)
    print(f"report={parsed.report_path}")
    print(f"status={report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
