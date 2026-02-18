"""
EchoZero environment smoke checks.

Usage:
    python scripts/verify_env.py
    python scripts/verify_env.py --quiet
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata


EXPECTED_PYQT6 = "6.4.2"
EXPECTED_QT6 = "6.4.3"
EXPECTED_SIP = "13.4.1"


def _print(msg: str, quiet: bool) -> None:
    """Print message unless quiet mode is enabled."""
    if not quiet:
        print(msg)


def _get_version(package_name: str) -> str | None:
    """Return installed package version or None when missing."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    """Run environment checks and return process exit code."""
    parser = argparse.ArgumentParser(description="Verify EchoZero Python environment")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures and final status",
    )
    args = parser.parse_args()
    quiet = args.quiet

    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10+ is required.")
        print(f"Found: {sys.version.split()[0]}")
        return 1

    _print(f"Python: {sys.version.split()[0]}", quiet)

    failures: list[str] = []

    v_pyqt6 = _get_version("PyQt6")
    v_qt6 = _get_version("PyQt6-Qt6")
    v_sip = _get_version("PyQt6-sip")

    if v_pyqt6 is None:
        failures.append("PyQt6 is not installed.")
    elif v_pyqt6 != EXPECTED_PYQT6:
        failures.append(f"PyQt6 version mismatch: expected {EXPECTED_PYQT6}, found {v_pyqt6}.")

    if v_qt6 is None:
        failures.append("PyQt6-Qt6 is not installed.")
    elif v_qt6 != EXPECTED_QT6:
        failures.append(f"PyQt6-Qt6 version mismatch: expected {EXPECTED_QT6}, found {v_qt6}.")

    if v_sip is None:
        failures.append("PyQt6-sip is not installed.")
    elif v_sip != EXPECTED_SIP:
        failures.append(f"PyQt6-sip version mismatch: expected {EXPECTED_SIP}, found {v_sip}.")

    try:
        from PyQt6 import QtCore  # type: ignore
    except Exception as exc:
        failures.append(f"PyQt6.QtCore import failed: {exc}")
    else:
        _print(f"Qt runtime: {QtCore.QT_VERSION_STR}", quiet)

    if failures:
        print("FAIL: environment verification failed.")
        for item in failures:
            print(f" - {item}")
        print(
            "Fix with: python -m pip install "
            '"PyQt6==6.4.2" "PyQt6-Qt6==6.4.3" "PyQt6-sip==13.4.1"'
        )
        return 1

    _print("PASS: environment verification passed.", quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
