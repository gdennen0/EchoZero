#!/usr/bin/env python3
"""
Reset EchoZero authentication state.

Deletes auth_data.json so the app treats the next launch as a fresh login.
Run this script, then restart EchoZero to see the login dialog.
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import get_user_data_dir

AUTH_FILE = "auth_data.json"


def main() -> int:
    path = get_user_data_dir() / AUTH_FILE
    if path.exists():
        path.unlink()
        print(f"Cleared auth state: {path}")
        return 0
    else:
        print(f"No auth file found at {path} (already fresh)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
