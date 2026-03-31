"""
Rename script for EchoZero naming convention audit.

Applies approved renames to all Python files in echozero/ and tests/.

Rename order matters: longer/more-specific names first to avoid partial matches.

Step 1: ProjectSettings -> ProjectSettingsRecord
Step 2: SongVersion -> SongVersionRecord  
Step 3: PipelineConfig -> PipelineConfigRecord
Step 4: ProjectSession -> ProjectStorage
Step 5: \bProject\b -> ProjectRecord  (won't match ProjectSettingsRecord, ProjectStorage, ProjectRepository)
Step 6: \bSong\b -> SongRecord         (won't match SongVersionRecord, SongRepository)

Word boundaries (\b) ensure we only rename standalone class names, not substrings.
"""
import re
from pathlib import Path

BASE = Path(__file__).parent
EXCLUDE_DIRS = {'.venv', 'src', 'ui', '.cursor', '__pycache__', '.git'}

RENAMES = [
    # Longer names first (more specific -> avoid partial replacement issues)
    ('ProjectSettings', 'ProjectSettingsRecord'),
    ('SongVersion',     'SongVersionRecord'),
    ('PipelineConfig',  'PipelineConfigRecord'),
    ('ProjectSession',  'ProjectStorage'),
    # Standalone names second (word boundaries prevent matching the above results)
    ('Project',         'ProjectRecord'),   # \bProject\b won't match ProjectSettingsRecord etc.
    ('Song',            'SongRecord'),      # \bSong\b won't match SongVersionRecord etc.
]


def apply_renames(content: str) -> str:
    for old, new in RENAMES:
        content = re.sub(r'\b' + re.escape(old) + r'\b', new, content)
    return content


def get_python_files():
    files = []
    for root_dir in ['echozero', 'tests']:
        root_path = BASE / root_dir
        if not root_path.exists():
            continue
        for path in root_path.rglob('*.py'):
            parts = set(path.parts)
            if parts & EXCLUDE_DIRS:
                continue
            files.append(path)
    return sorted(files)


def main():
    files = get_python_files()
    changed = []

    for f in files:
        original = f.read_text(encoding='utf-8')
        modified = apply_renames(original)
        if modified != original:
            f.write_text(modified, encoding='utf-8')
            changed.append(f)
            print(f"  Updated: {f.relative_to(BASE)}")

    print(f"\nTotal files changed: {len(changed)}")
    return changed


if __name__ == '__main__':
    main()
