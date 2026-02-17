#!/usr/bin/env python3
"""
Inspect a .pth model from PyTorch Audio Trainer: classes, config, and dataset stats.
Use to debug "everything classified as other" or verify training setup.

Usage:
  python scripts/inspect_pth_model.py "/path/to/model.pth"
"""
import sys
from pathlib import Path

# Allow running from repo root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.application.blocks.training.checkpointing import load_training_metadata


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_pth_model.py <path_to_model.pth>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    meta = load_training_metadata(str(path))
    if not meta:
        print("Could not load checkpoint (wrong format or missing file).")
        sys.exit(1)

    print("=== Classes (order = model output index) ===")
    classes = meta.get("classes", [])
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")

    print("\n=== Classification mode ===")
    print(f"  {meta.get('classification_mode', '?')}")

    print("\n=== Dataset stats (training-time distribution) ===")
    ds = meta.get("dataset_stats") or {}
    if isinstance(ds, dict):
        dist = ds.get("class_distribution") or {}
        total = ds.get("total_samples") or 0
        for name, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = (100.0 * count / total) if total else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
        print(f"  total_samples: {total}")
        if ds.get("balance_strategy"):
            print(f"  balance_strategy: {ds['balance_strategy']}")
    else:
        print("  (no dataset_stats)")

    print("\n=== Inference preprocessing (must match at inference) ===")
    prep = meta.get("inference_preprocessing")
    if prep:
        for k, v in prep.items():
            print(f"  {k}: {v}")
    else:
        print("  (none; older model - classifier will use config)")

    print("\n=== Binary threshold (binary mode only) ===")
    opt = meta.get("optimal_threshold")
    print(f"  optimal_threshold: {opt}")

    print("\n=== Config (relevant) ===")
    config = meta.get("config") or {}
    print(f"  classification_mode: {config.get('classification_mode')}")
    print(f"  positive_classes: {config.get('positive_classes')}")
    print(f"  use_class_weights: {config.get('use_class_weights')}")
    print(f"  balance_strategy: {config.get('balance_strategy', 'none')}")


if __name__ == "__main__":
    main()
