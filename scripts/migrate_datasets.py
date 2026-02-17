#!/usr/bin/env python3
"""
Migrate legacy repo-local datasets into managed user data storage.

This script performs a non-destructive one-time migration:
- Source:   <app_install_dir>/data/datasets
- Dest:     <user_data_dir>/datasets
"""
from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.datasets import migrate_legacy_repo_datasets

    result = migrate_legacy_repo_datasets()
    print(result.message)
    if result.source is not None:
        print(f"source: {result.source}")
    print(f"destination: {result.destination}")
    print(f"migrated: {result.migrated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
