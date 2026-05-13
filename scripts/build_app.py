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
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Env vars that can be embedded for zero-config shipping
# MEMBERSTACK_APP_SECRET must come from environment at build time (never in config)
# MEMBERSTACK_VERIFY_URL can come from env or from packaging_config.json bundled_runtime_defaults
BUNDLED_ENV_VARS = ("MEMBERSTACK_APP_SECRET", "MEMBERSTACK_VERIFY_URL")
MAX_BUNDLED_MODEL_WEIGHT_BYTES = 1_000_000


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
        import PyInstaller  # noqa: F401
    except ImportError:
        missing.append("PyInstaller")
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


def _run_env_smoke_check(project_root: Path) -> bool:
    """Run environment verification and fail early on known Qt DLL issues."""
    verify_script = project_root / "scripts" / "verify_env.py"
    if not verify_script.is_file():
        print("Warning: scripts/verify_env.py not found; skipping environment smoke check.")
        return True

    cmd = [sys.executable, str(verify_script), "--quiet", "--build"]
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(
            "Error: environment verification failed. "
            "Fix the Python environment before building."
        )
        return False
    return True


def _write_bundled_config(project_root: Path) -> None:
    """Write build/bundled_config.env for zero-config shipping."""
    build_dir = project_root / "build"
    out = build_dir / "bundled_config.env"
    defaults = _load_bundled_defaults(project_root)
    local_env = _load_local_env_values(project_root)
    lines = []
    for key in BUNDLED_ENV_VARS:
        val = os.environ.get(key, "").strip() or local_env.get(key, "") or defaults.get(key, "")
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


def _validate_production_template(project_root: Path) -> bool:
    """Warn (but don't fail) if the production template is missing."""
    template_path = project_root / "data" / "production_template.ez"
    if not template_path.is_file():
        print(
            "Warning: data/production_template.ez not found. "
            "Production mode will fall back to developer behaviour. "
            "Use File > 'Export as Production Template' to create one."
        )
        return True  # Non-fatal; build continues
    print(f"Production template found: {template_path} ({template_path.stat().st_size} bytes)")
    return True


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_packaging_identity(project_root: Path) -> dict[str, str]:
    config_path = project_root / "packaging_config.json"
    if not config_path.is_file():
        return {"app_name": "EchoZero", "version": "0.0.0"}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {"app_name": "EchoZero", "version": "0.0.0"}
    return {
        "app_name": str(data.get("app_name") or "EchoZero"),
        "version": str(data.get("version") or "0.0.0"),
    }


def _assert_no_large_bundled_model_weights(dist_dir: Path) -> bool:
    if not dist_dir.exists():
        return True
    matches = [
        path
        for path in dist_dir.rglob("*.pth")
        if path.is_file() and path.stat().st_size > MAX_BUNDLED_MODEL_WEIGHT_BYTES
    ]
    if not matches:
        return True
    print(
        "Error: packaged app includes large .pth model weights; "
        "v1-alpha models must install separately."
    )
    for path in matches:
        print(f"  - {path} ({path.stat().st_size} bytes)")
    return False


def _write_build_metadata(project_root: Path, dist_dir: Path, command: list[str]) -> None:
    identity = _load_packaging_identity(project_root)
    metadata = {
        "schema": "echozero.build_metadata.v1",
        "app_name": identity["app_name"],
        "version": identity["version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "command": " ".join(command),
        "models_bundled_by_default": False,
    }
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EchoZero with PyInstaller")
    parser.add_argument(
        "--clean", action="store_true", help="Clean PyInstaller cache before build"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    spec_path = project_root / "echozero.spec"
    if not spec_path.is_file():
        print(f"Error: spec file not found: {spec_path}")
        return 1

    if not _check_build_deps():
        return 1
    if not _run_env_smoke_check(project_root):
        return 1
    _validate_production_template(project_root)
    _write_bundled_config(project_root)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if args.clean:
        cmd.append("--clean")
    cmd.append(str(spec_path))

    print("Building EZ...")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode == 0:
        dist = project_root / "dist"
        if not _assert_no_large_bundled_model_weights(dist):
            return 1
        _write_build_metadata(project_root, dist, cmd)
        print(f"Build complete. Output: {dist}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
