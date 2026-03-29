"""
DatasetViewerProcessor: Scans an exported dataset directory and produces a summary.
Exists because dataset quality validation is essential before training —
you need to see class distributions, sample counts, and outliers.
Used by ExecutionEngine when running blocks of type 'DatasetViewer'.

This is a read-only processor — it scans a directory structure and returns
metadata about the dataset. No audio processing. The UI renders the summary.

The injectable function pattern allows testing without file system access.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# Scan function signature for DI
# ---------------------------------------------------------------------------

ScanDatasetFn = Callable[
    [
        str,  # dataset_dir
        tuple[str, ...],  # audio_extensions to look for
    ],
    dict[str, Any],  # scan results
]


def _default_scan(
    dataset_dir: str,
    audio_extensions: tuple[str, ...],
) -> dict[str, Any]:
    """Scan dataset directory and return class-level stats. Production default."""
    if not os.path.isdir(dataset_dir):
        raise ExecutionError(f"Dataset directory not found: {dataset_dir}")

    classes: dict[str, dict[str, Any]] = {}
    total_files = 0

    for entry in os.scandir(dataset_dir):
        if entry.is_dir():
            # Subdirectory = class label
            class_name = entry.name
            class_files = []
            for f in os.scandir(entry.path):
                if f.is_file() and f.name.lower().endswith(audio_extensions):
                    class_files.append({
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                    })
            classes[class_name] = {
                "count": len(class_files),
                "total_bytes": sum(f["size_bytes"] for f in class_files),
                "files": class_files,
            }
            total_files += len(class_files)
        elif entry.is_file() and entry.name.lower().endswith(audio_extensions):
            # Root-level files go into "unclassified"
            if "unclassified" not in classes:
                classes["unclassified"] = {"count": 0, "total_bytes": 0, "files": []}
            classes["unclassified"]["files"].append({
                "name": entry.name,
                "size_bytes": entry.stat().st_size,
            })
            classes["unclassified"]["count"] += 1
            classes["unclassified"]["total_bytes"] += entry.stat().st_size
            total_files += 1

    return {
        "dataset_dir": dataset_dir,
        "total_files": total_files,
        "total_classes": len(classes),
        "classes": {
            name: {
                "count": info["count"],
                "total_bytes": info["total_bytes"],
            }
            for name, info in sorted(classes.items())
        },
    }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")


class DatasetViewerProcessor:
    """Scans an exported dataset directory and returns a metadata summary."""

    def __init__(self, scan_fn: ScanDatasetFn | None = None) -> None:
        self._scan_fn = scan_fn or _default_scan

    def execute(self, block_id: str, context: ExecutionContext) -> Result[dict[str, Any]]:
        """Scan a dataset directory and return class distribution metadata.

        Returns a dict with: {dataset_dir, total_files, total_classes, classes}.
        """
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="dataset_viewer",
                percent=0.0,
                message="Scanning dataset directory",
            )
        )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        dataset_dir = settings.get("dataset_dir")

        if not dataset_dir:
            return err(ValidationError(
                f"Block '{block_id}' is missing required setting 'dataset_dir'"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="dataset_viewer",
                percent=0.3,
                message=f"Scanning {dataset_dir}",
            )
        )

        # Scan
        try:
            results = self._scan_fn(dataset_dir, AUDIO_EXTENSIONS)
        except (ExecutionError, ValidationError) as exc:
            return err(exc)
        except Exception as exc:
            return err(ExecutionError(
                f"Dataset scan failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="dataset_viewer",
                percent=1.0,
                message=(
                    f"Scan complete — {results['total_files']} files "
                    f"in {results['total_classes']} classes"
                ),
            )
        )

        return ok(results)


