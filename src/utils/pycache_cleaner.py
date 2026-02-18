"""
Python Cache Cleaner Utility

Clears __pycache__ directories and .pyc files to prevent stale bytecode issues.

This is run on application startup when the clear_pycache_on_startup setting is enabled.
"""
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional


def get_clear_pycache_setting() -> bool:
    """
    Read the clear_pycache_on_startup setting directly from the database.
    
    This is done before full app initialization to allow early cache clearing.
    Returns True by default if setting is not found.
    """
    # Get the database path (same logic as in persistence layer)
    if os.name == 'nt':  # Windows
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        db_dir = Path(app_data) / "EchoZero"
    else:  # macOS / Linux
        db_dir = Path.home() / "Library" / "Application Support" / "EchoZero"
    
    db_path = db_dir / "ez.db"
    
    if not db_path.exists():
        return True  # Default to True if no database yet
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Query the preferences table for this setting
        cursor.execute(
            "SELECT value FROM preferences WHERE namespace = ? AND key = ?",
            ("app", "clear_pycache_on_startup")
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return True  # Default to True if setting not found
        
        # The value is stored as a string
        return result[0].lower() in ('true', '1', 'yes')
        
    except Exception:
        return True  # Default to True on any error


def clear_pycache(project_root: Optional[Path] = None, verbose: bool = False) -> int:
    """
    Clear all __pycache__ directories and .pyc files in project source directories.
    
    Only cleans: src/, ui/, tests/ - excludes .venv, node_modules, etc.
    
    Args:
        project_root: Root directory to clean. Defaults to EchoZero project root.
        verbose: If True, print what's being deleted.
        
    Returns:
        Number of directories/files removed.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent  # Go up from src/utils
    
    # Only clean source directories (not .venv, node_modules, etc.)
    source_dirs = ['src', 'ui', 'tests', 'AgentAssets']
    
    removed_count = 0
    
    for source_dir in source_dirs:
        dir_path = project_root / source_dir
        if not dir_path.exists():
            continue
            
        # Remove __pycache__ directories
        for pycache_dir in dir_path.rglob("__pycache__"):
            try:
                if pycache_dir.is_dir():
                    shutil.rmtree(pycache_dir)
                    removed_count += 1
                    if verbose:
                        print(f"[pycache_cleaner] Removed: {pycache_dir}")
            except Exception as e:
                if verbose:
                    print(f"[pycache_cleaner] Failed to remove {pycache_dir}: {e}")
        
        # Remove any stray .pyc files (shouldn't exist outside __pycache__ in Python 3)
        for pyc_file in dir_path.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                removed_count += 1
                if verbose:
                    print(f"[pycache_cleaner] Removed: {pyc_file}")
            except Exception as e:
                if verbose:
                    print(f"[pycache_cleaner] Failed to remove {pyc_file}: {e}")
    
    return removed_count


def clear_pycache_if_enabled(verbose: bool = False) -> bool:
    """
    Clear pycache if the setting is enabled.
    
    This is the main function to call from main_qt.py.
    
    Args:
        verbose: If True, print status messages.
        
    Returns:
        True if cache was cleared, False if skipped.
    """
    if not get_clear_pycache_setting():
        if verbose:
            print("[pycache_cleaner] Skipped - disabled in settings")
        return False
    
    removed = clear_pycache(verbose=verbose)
    if verbose:
        print(f"[pycache_cleaner] Cleared {removed} cache entries")
    
    return True


if __name__ == "__main__":
    # Allow running directly for testing
    print("Clearing pycache...")
    count = clear_pycache(verbose=True)
    print(f"Done. Removed {count} cache entries.")
