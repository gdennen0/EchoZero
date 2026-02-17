"""
Build EchoZero as a standalone application (PyInstaller).

Usage (from project root, with venv activated):
    python scripts/build_app.py [--clean]

For a complete package with no user configuration (zero-config ship), set auth
vars in the environment before building; they are embedded in the bundle:
    MEMBERSTACK_APP_SECRET=your_secret python scripts/build_app.py
    MEMBERSTACK_VERIFY_URL=https://...  (optional; has default)

Reads packaging_config.json for app name, version, and bundle identifier.
Output: dist/EchoZero/ (all platforms); dist/EchoZero.app on macOS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Env vars that can be embedded for zero-config shipping
# MEMBERSTACK_APP_SECRET must come from environment at build time (never in config)
# MEMBERSTACK_VERIFY_URL can come from env or from packaging_config.json bundled_runtime_defaults
BUNDLED_ENV_VARS = ("MEMBERSTACK_APP_SECRET", "MEMBERSTACK_VERIFY_URL")


def _load_bundled_defaults(project_root: Path) -> dict[str, str]:
    """Load non-secret defaults from packaging_config.json (e.g. MEMBERSTACK_VERIFY_URL)."""
    config_path = project_root / "packaging_config.json"
    if not config_path.is_file():
        return {}
    try:
        import json
        data = json.loads(config_path.read_text(encoding="utf-8"))
        defaults = data.get("bundled_runtime_defaults") or {}
        return {k: str(v).strip() for k, v in defaults.items() if v}
    except Exception:
        return {}


def _load_local_env_values(project_root: Path) -> dict[str, str]:
    """Load build-time fallback values from local .env if present."""
    env_path = project_root / ".env"
    if not env_path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
        values = dotenv_values(str(env_path))
        return {k: str(v).strip() for k, v in values.items() if v}
    except Exception:
        return {}


def _check_build_deps() -> bool:
    """Ensure required packages are available so the frozen app has them."""
    missing = []
    try:
        import dotenv  # noqa: F401
    except ImportError:
        missing.append("python-dotenv")
    try:
        import httpx  # noqa: F401
    except ImportError:
        missing.append("httpx")
    if missing:
        print(
            f"Error: missing build dependencies: {', '.join(missing)}. "
            "Run: pip install -r requirements.txt and rebuild."
        )
        return False
    return True


def _write_bundled_config(project_root: Path) -> None:
    """Write build/bundled_config.env from env + packaging_config defaults for zero-config shipping."""
    build_dir = project_root / "build"
    out = build_dir / "bundled_config.env"
    defaults = _load_bundled_defaults(project_root)
    local_env = _load_local_env_values(project_root)
    lines = []
    for key in BUNDLED_ENV_VARS:
        val = (
            os.environ.get(key, "").strip()
            or local_env.get(key, "")
            or defaults.get(key, "")
        )
        if val:
            lines.append(f"{key}={val}")
    if not lines:
        if out.exists():
            out.unlink()
        return
    build_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bundled config: {len(lines)} var(s) -> {out.name} (zero-config build)")
    if not any(line.startswith("MEMBERSTACK_APP_SECRET=") for line in lines):
        print(
            "Warning: MEMBERSTACK_APP_SECRET not bundled. "
            "If your worker enforces APP_SECRET, packaged login will fail with Unauthorized."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EchoZero with PyInstaller")
    parser.add_argument("--clean", action="store_true", help="Clean PyInstaller cache before build")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    spec_path = project_root / "echozero.spec"
    if not spec_path.is_file():
        print(f"Error: spec file not found: {spec_path}")
        return 1

    if not _check_build_deps():
        return 1
    _write_bundled_config(project_root)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if args.clean:
        cmd.append("--clean")
    cmd.append(str(spec_path))

    print("Building EchoZero...")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode == 0:
        dist = project_root / "dist"
        print(f"Build complete. Output: {dist}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
