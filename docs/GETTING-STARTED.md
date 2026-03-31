# EchoZero 2 — Getting Started

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | latest |
| ffmpeg | any recent (for audio decoding) |

Optional (for ML processors):
- `demucs` — stem separation
- `torch` / `torchaudio` — PyTorch audio classify
- `essentia` — audio feature extraction

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url> EchoZero
cd EchoZero
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

If you don't need ML processors (faster install):
```bash
pip install -e ".[core]"
```

---

## Run the tests

```bash
pytest tests/ -x --timeout=30
```

Expected baseline: **1523 passed, 1 skipped**.

To run a specific module:
```bash
pytest tests/test_project.py -v
```

---

## Using EchoZero Programmatically

### Create a new project

```python
from echozero.main import create_project

# Creates a project with all real processors wired up
project = create_project(name="My Show")

# Or use as context manager (auto-closes on exit)
with create_project(name="My Show") as project:
    print(project.name)  # "My Show"
```

### Import a song

```python
from pathlib import Path

song, version = project.import_song(
    title="Song Title",
    audio_source=Path("path/to/song.wav"),
    artist="Artist Name",
    label="Original",
)

print(f"Imported: {song.title} ({version.duration_seconds:.1f}s)")
```

### Run analysis

```python
# List available pipeline templates
from echozero.pipelines.registry import get_registry

registry = get_registry()
for template in registry.list():
    print(f"  {template.id}: {template.name}")

# Run analysis with a template
result = project.analyze(
    song_version_id=version.id,
    template_id="onset_detection",
    knob_overrides={"threshold": 0.4},
)

if result.is_ok():
    analysis = result.unwrap()
    print(f"Analysis complete in {analysis.duration_ms:.0f}ms")
    print(f"Layers: {analysis.layer_ids}")
    print(f"Takes:  {analysis.take_ids}")
```

### Open an existing project

```python
from pathlib import Path
from echozero.main import open_project

with open_project(Path("my_show.ez")) as project:
    songs = project.songs.list_by_project(project.storage.project.id)
    for song in songs:
        print(f"  {song.title}")
```

### Save and export

```python
# Save to working directory (auto-committed every 30s anyway)
project.save()

# Export as .ez archive
project.save_as(Path("my_show.ez"))

# Always close when done
project.close()
```

---

## Mutating the Graph

Graph mutations go through `project.dispatch()`. Never mutate the graph directly.

```python
from echozero.editor.commands import AddBlockCommand, AddConnectionCommand
import uuid

# Add a block
result = project.dispatch(AddBlockCommand(
    block_id=uuid.uuid4().hex,
    name="Load Audio",
    block_type="LoadAudio",
    category="SOURCE",
    input_ports=[],
    output_ports=[("audio_out", "AUDIO", "OUTPUT")],
    control_ports=[],
    settings_entries={"file_path": "audio/song.wav"},
))

if result.is_ok():
    block_id = result.unwrap()
    print(f"Added block: {block_id}")
```

---

## Running the Pipeline

```python
# Run all blocks
result = project.run()
if result.is_ok():
    execution_id = result.unwrap()
    print(f"Execution complete: {execution_id}")

# Run in background
handle_result = project.run_async()
if handle_result.is_ok():
    handle = handle_result.unwrap()
    # ... do other work ...
    result = handle.result(timeout=60.0)
```

---

## Project Working Directory

EchoZero uses a **working directory** pattern:

- Location: `~/.echozero/working/<hash>/`
- Contains `project.db` (SQLite) and imported audio files
- **NOT deleted on close** — enables crash recovery
- On next open, EchoZero detects the stale dir and can recover

Check for recovery:
```python
from pathlib import Path
from echozero.persistence.session import ProjectStorage

if ProjectStorage.check_recovery(Path("my_show.ez")):
    # Offer recovery dialog
    project_storage = ProjectStorage.recover(Path("my_show.ez"))
```

---

## Development Tips

- **Test isolation:** Use `Project.create()` directly with mock executors — don't use `main.py` in tests.
- **No Qt needed:** All core functionality works without PyQt6 installed.
- **Pipeline templates:** Register templates via `@pipeline_template` decorator in `pipelines/templates/`.
- **Logging:** Set `ECHOZERO_LOG_LEVEL=DEBUG` to see execution traces.
