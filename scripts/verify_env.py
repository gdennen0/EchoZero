"""
EchoZero environment smoke checks.

Usage:
    python scripts/verify_env.py
    python scripts/verify_env.py --quiet
    python scripts/verify_env.py --build
"""
from __future__ import annotations

import argparse
import importlib
import sys
from importlib import metadata


BASE_IMPORTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"),
)

BUILD_IMPORTS: tuple[tuple[str, str], ...] = (
    ("PyInstaller", "PyInstaller"),
    ("dotenv", "python-dotenv"),
    ("httpx", "httpx"),
)


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


def _import_module(import_name: str) -> str | None:
    """Attempt to import a module and return an error string on failure."""
    try:
        importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - surfaced in CLI output
        return str(exc)
    return None


def _check_imports(
    *,
    imports: tuple[tuple[str, str], ...],
    failures: list[str],
    quiet: bool,
) -> None:
    """Validate importability of a list of modules and print discovered versions."""
    for import_name, package_name in imports:
        version = _get_version(package_name)
        if version is None:
            failures.append(f"{package_name} is not installed.")
            continue
        error = _import_module(import_name)
        if error is not None:
            failures.append(f"{package_name} import failed: {error}")
            continue
        _print(f"{package_name}: {version}", quiet)


def main() -> int:
    """Run environment checks and return process exit code."""
    parser = argparse.ArgumentParser(description="Verify EchoZero Python environment")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures and final status",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also verify optional packaging/build dependencies.",
    )
    args = parser.parse_args()
    quiet = args.quiet

    if sys.version_info < (3, 11):
        print("FAIL: Python 3.11+ is required.")
        print(f"Found: {sys.version.split()[0]}")
        return 1

    _print(f"Python: {sys.version.split()[0]}", quiet)

    failures: list[str] = []
    _check_imports(imports=BASE_IMPORTS, failures=failures, quiet=quiet)

    pyqt_version = _get_version("PyQt6")
    if pyqt_version is None:
        failures.append("PyQt6 is not installed.")
    else:
        _print(f"PyQt6: {pyqt_version}", quiet)

    try:
        from PyQt6 import QtCore  # type: ignore
    except Exception as exc:
        failures.append(f"PyQt6.QtCore import failed: {exc}")
    else:
        _print(f"Qt runtime: {QtCore.QT_VERSION_STR}", quiet)

    if args.build:
        _check_imports(imports=BUILD_IMPORTS, failures=failures, quiet=quiet)

    if failures:
        print("FAIL: environment verification failed.")
        for item in failures:
            print(f" - {item}")
        print('Fix base env with: python -m pip install -e .')
        if args.build:
            print('Fix build extras with: python -m pip install -e ".[packaging]"')
        return 1

    _print("PASS: environment verification passed.", quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
