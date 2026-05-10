#!/usr/bin/env python3
"""ma3-datapool-documenter: Capture custom MA3 browse hierarchy over OSC and prepare terminal-proof bundles.
Exists because MA native inspection must be proven from the MA terminal/CLI surface, while OSC is reserved for our custom Lua service layer.
Connects the EchoZero OSC browse layer to a documentation bundle that tells agents what still needs terminal capture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_harness_common import build_bridge  # noqa: E402
from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge  # noqa: E402


def _normalize_path(path: object) -> str:
    return str(path or "").strip().strip("/")


def _path_depth(path: str) -> int:
    if not path:
        return 0
    return len([token for token in path.split("/") if token])


def _path_label(path: str) -> str:
    return path or "DataPool"


def capture_datapool_snapshot(
    bridge: MA3OSCBridge,
    *,
    root_path: str | None = None,
    max_objects: int | None = None,
) -> list[dict[str, object]]:
    """Walk the MA3 DataPool tree and return a custom-OSC hierarchy snapshot."""

    pending_paths = [_normalize_path(root_path)]
    seen_paths: set[str] = set()
    objects: list[dict[str, object]] = []

    while pending_paths:
        path = pending_paths.pop()
        if path in seen_paths:
            continue
        seen_paths.add(path)

        raw_object = bridge.describe_datapool_object(path or None)
        if not raw_object:
            continue
        raw_object["path"] = _normalize_path(raw_object.get("path"))
        objects.append(raw_object)
        if max_objects is not None and len(objects) >= max(0, int(max_objects)):
            break

        children = bridge.list_datapool_objects(path or None)
        child_paths = [
            _normalize_path(child.get("path"))
            for child in children
            if isinstance(child, dict) and _normalize_path(child.get("path")) not in seen_paths
        ]
        pending_paths.extend(reversed(child_paths))

    objects.sort(
        key=lambda item: (
            _path_depth(_normalize_path(item.get("path"))),
            _normalize_path(item.get("path")),
        )
    )
    return objects


def _render_hierarchy_markdown(
    snapshot: list[dict[str, object]], *, target: dict[str, object]
) -> str:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# MA3 DataPool Hierarchy",
        "",
        f"- Captured at: `{captured_at}`",
        f"- Target: `{target['ma3_host']}:{target['ma3_port']}{target['command_path']}`",
        f"- Object count: `{len(snapshot)}`",
        "- Source: EchoZero custom OSC browse layer only. Not authoritative for native MA attributes.",
        "",
    ]
    for raw_object in snapshot:
        path = _normalize_path(raw_object.get("path"))
        indent = "  " * _path_depth(path)
        class_name = str(raw_object.get("class") or "Unknown")
        child_count = raw_object.get("child_count")
        descriptor = f"`{_path_label(path)}` [{class_name}]"
        metadata = [f"children={child_count if child_count is not None else 0}"]
        object_no = raw_object.get("no")
        if object_no is not None:
            metadata.append(f"no={object_no}")
        lines.append(f"{indent}- {descriptor} ({', '.join(metadata)})")
    lines.append("")
    return "\n".join(lines)


def _terminal_capture_targets(snapshot: list[dict[str, object]]) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for raw_object in snapshot:
        path = _normalize_path(raw_object.get("path"))
        targets.append(
            {
                "path": path,
                "label": _path_label(path),
                "class": str(raw_object.get("class") or "Unknown"),
                "priority": "root" if not path else "object",
                "requires_terminal_native_capture": True,
                "required_evidence": [
                    "terminal_list_output",
                    "terminal_property_inventory",
                    "terminal_dump_output",
                ],
                "notes": (
                    "Use the MA terminal/CLI surface for native inspection. "
                    "Do not route Dump(), PropertyCount(), PropertyName(), "
                    "PropertyType(), or PropertyInfo() through OSC."
                ),
            }
        )
    return targets


def _render_terminal_capture_markdown(
    snapshot: list[dict[str, object]],
    *,
    target: dict[str, object],
) -> str:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# MA3 Terminal Capture Plan",
        "",
        f"- Captured at: `{captured_at}`",
        f"- Target: `{target['ma3_host']}:{target['ma3_port']}{target['command_path']}`",
        "- Rule: native MA introspection must be captured from the MA terminal/CLI interface.",
        "- OSC browse is allowed only for hierarchy discovery and path planning.",
        "",
        "| Path | Class | Required Terminal Proof |",
        "|------|-------|-------------------------|",
    ]
    for raw_object in snapshot:
        path = _normalize_path(raw_object.get("path"))
        class_name = str(raw_object.get("class") or "Unknown")
        lines.append(
            f"| `{_path_label(path)}` | `{class_name}` | `List`, property inventory, `Dump()` in terminal |"
        )
    lines.append("")
    return "\n".join(lines)


def write_datapool_bundle(
    output_dir: Path,
    *,
    snapshot: list[dict[str, object]],
    target: dict[str, object],
    root_path: str | None,
) -> dict[str, Path]:
    """Write a DataPool planning bundle and return the created file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "snapshot.json"
    hierarchy_path = output_dir / "hierarchy.md"
    capture_plan_path = output_dir / "terminal_capture_plan.md"
    capture_targets_path = output_dir / "terminal_capture_targets.json"
    readme_path = output_dir / "README.md"
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    snapshot_payload = {
        "generated_at": generated_at,
        "root_path": _normalize_path(root_path),
        "target": dict(target),
        "object_count": len(snapshot),
        "source_of_truth": {
            "custom_api": "osc_lua_service_layer",
            "raw_ma_authority": "ma_terminal_cli",
        },
        "objects": snapshot,
    }
    snapshot_path.write_text(
        json.dumps(snapshot_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    hierarchy_path.write_text(
        _render_hierarchy_markdown(snapshot, target=target), encoding="utf-8"
    )
    capture_plan_path.write_text(
        _render_terminal_capture_markdown(snapshot, target=target),
        encoding="utf-8",
    )
    capture_targets_path.write_text(
        json.dumps(_terminal_capture_targets(snapshot), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                "# MA3 DataPool Documentation Bundle",
                "",
                f"- Generated at: `{generated_at}`",
                f"- Root path: `{_path_label(_normalize_path(root_path))}`",
                f"- Object count: `{len(snapshot)}`",
                "",
                "- `snapshot.json`: custom OSC/Lua hierarchy snapshot only.",
                "- `hierarchy.md`: human-readable path tree from OSC browse.",
                "- `terminal_capture_plan.md`: terminal-first proof checklist per object.",
                "- `terminal_capture_targets.json`: machine-readable terminal capture manifest.",
                "",
                "Native MA attributes and dump text must be captured from the MA terminal/CLI surface.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "snapshot": snapshot_path,
        "hierarchy": hierarchy_path,
        "terminal_capture_plan": capture_plan_path,
        "terminal_capture_targets": capture_targets_path,
        "readme": readme_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture MA3 hierarchy and prepare terminal-proof docs."
    )
    parser.add_argument("--ma3-host", default=None)
    parser.add_argument("--ma3-port", type=int, default=None)
    parser.add_argument("--command-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--settings-path", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--root-path", default=None, help="Optional DataPool subpath to capture.")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-datapool" / "latest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Capture an OSC hierarchy bundle and print a compact JSON summary."""

    args = _build_parser().parse_args(argv)
    bridge, target = build_bridge(
        ma3_host=args.ma3_host,
        ma3_port=args.ma3_port,
        command_path=args.command_path,
        settings_path=args.settings_path,
        listen_host=str(args.listen_host or "0.0.0.0"),
        listen_port=int(args.listen_port),
        timeout=float(args.timeout),
    )
    try:
        snapshot = capture_datapool_snapshot(
            bridge,
            root_path=args.root_path,
            max_objects=args.max_objects,
        )
        bundle = write_datapool_bundle(
            args.output_dir,
            snapshot=snapshot,
            target=target,
            root_path=args.root_path,
        )
    finally:
        bridge.shutdown()

    result = {
        "target": target,
        "root_path": _normalize_path(args.root_path),
        "object_count": len(snapshot),
        "output_dir": str(args.output_dir),
        "source_of_truth": {
            "custom_api": "osc_lua_service_layer",
            "raw_ma_authority": "ma_terminal_cli",
        },
        "files": {key: str(path) for key, path in bundle.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
