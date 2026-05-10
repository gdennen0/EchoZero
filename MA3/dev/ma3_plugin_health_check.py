#!/usr/bin/env python3
"""Check MA3-loaded EZ/HitMaker plugin health and compare against local plugin files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_harness_common import build_bridge, resolve_target  # noqa: E402

DEFAULT_LIVE_PLUGIN_ROOT = Path("/Users/march/MALightingTechnology/gma3_library/datapools/plugins")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query MA3 EZ.GetPluginHealth() and compare with local plugin build/version markers.",
    )
    parser.add_argument("--ma3-host", default=None)
    parser.add_argument("--ma3-port", type=int, default=None)
    parser.add_argument("--command-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--settings-path", type=Path, default=None)
    parser.add_argument(
        "--expected-root",
        type=Path,
        default=None,
        help=(
            "Directory containing expected plugin files. "
            "Defaults to the live MA3 datapools plugin root when available, else repo MA3/plugins."
        ),
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Only print MA3-reported health. Do not fail on local-vs-MA3 mismatches.",
    )
    return parser


def _extract_marker(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if match is None:
        return None
    return str(match.group(1)).strip() or None


def _resolve_expected_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    if DEFAULT_LIVE_PLUGIN_ROOT.exists():
        return DEFAULT_LIVE_PLUGIN_ROOT
    return REPO_ROOT / "MA3/plugins"


def _expected_local_markers(expected_root: Path) -> dict[str, str | None]:
    expected_root = expected_root.resolve()
    ez_candidates = [
        expected_root / "EZ/ez_core.lua",
        expected_root / "echozero.lua",
    ]
    hitmaker_candidates = [
        expected_root / "HitMaker/main.lua",
    ]

    def first_match(paths: list[Path], pattern: str) -> str | None:
        for path in paths:
            value = _extract_marker(path, pattern)
            if value is not None:
                return value
        return None

    return {
        "ez_version": first_match(ez_candidates, r'EZ\._version\s*=\s*"([^"]+)"'),
        "ez_build": (
            first_match(ez_candidates, r'EZ\._build\s*=\s*EZ\._build\s*or\s*"([^"]+)"')
            or first_match(ez_candidates, r'EZ\._build\s*=\s*"([^"]+)"')
        ),
        "hitmaker_version": first_match(
            hitmaker_candidates,
            r'HitMaker\._version\s*=\s*HitMaker\._version\s*or\s*"([^"]+)"',
        ),
        "hitmaker_build": (
            first_match(
                hitmaker_candidates,
                r'HitMaker\._build\s*=\s*HitMaker\._build\s*or\s*"([^"]+)"',
            )
            or first_match(
                hitmaker_candidates,
                r'HitMaker\._build\s*=\s*"([^"]+)"',
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ma3_host, ma3_port, command_path, settings_path = resolve_target(
        ma3_host=args.ma3_host,
        ma3_port=args.ma3_port,
        command_path=args.command_path,
        settings_path=args.settings_path,
    )
    expected_root = _resolve_expected_root(args.expected_root)
    expected = _expected_local_markers(expected_root)

    print(
        f"TARGET {ma3_host}:{ma3_port} {command_path} (settings: {settings_path})",
        flush=True,
    )
    print(f"EXPECTED_ROOT {expected_root}", flush=True)

    bridge, _target = build_bridge(
        ma3_host=ma3_host,
        ma3_port=ma3_port,
        command_path=command_path,
        settings_path=args.settings_path,
        listen_host=str(args.listen_host or "0.0.0.0"),
        listen_port=int(args.listen_port),
        timeout=float(args.timeout),
    )
    try:
        try:
            health = bridge.get_plugin_health()
        except TimeoutError:
            print(
                "RESULT FAIL",
                flush=True,
            )
            print(
                "  - Timed out waiting for EZ.GetPluginHealth() reply. "
                "MA3 likely has older EZ plugin code loaded or plugins did not reload.",
                flush=True,
            )
            return 2
    finally:
        bridge.shutdown()

    print("MA3 plugin health:", flush=True)
    for key in sorted(health):
        print(f"  {key}={health[key]!r}", flush=True)

    if args.no_compare:
        return 0

    failures: list[str] = []
    for key in ("ez_version", "ez_build", "hitmaker_version", "hitmaker_build"):
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        actual_value = str(health.get(key) or "")
        if actual_value != expected_value:
            failures.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")

    if not bool(health.get("hitmaker_loaded", False)):
        failures.append("hitmaker_loaded: expected True, got False")
    if not bool(health.get("hitmaker_supports_event_type_create", False)):
        failures.append("hitmaker_supports_event_type_create: expected True, got False")

    if failures:
        print("RESULT FAIL", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1

    print("RESULT PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
