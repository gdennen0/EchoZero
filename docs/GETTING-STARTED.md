# EchoZero 2 — Getting Started

Status: reference
Last reviewed: 2026-04-30


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
python3 --version
python3 -m venv .venv
```

EchoZero requires Python 3.11+. If your machine reports an older version,
install Python 3.11 or 3.12 and create the venv with that interpreter instead.
On macOS, the quickest fix is usually `brew install python@3.11`.

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

If you only want the minimal EZ2 app shell:
```bash
pip install -e .
```

If you want the full local environment, including optional ML processors and packaging tools:
```bash
pip install -r requirements.txt
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

## Installed Models

EchoZero now uses a canonical local models folder:

```text
~/.echozero/models
```

This folder is part of the app-managed install surface.
It is the canonical place for:

- runtime classification bundles the app should see
- future in-app model service downloads and installs
- manually dropped Foundry export bundles for local runtime use

Recommended contents:

- Foundry export folders containing `model.pth` plus `*.manifest.json`
- standalone `*.manifest.json` files beside their referenced `model.pth`
- raw `model.pth` files only when you intentionally bypass artifact-style packaging

The classify-drum-events file picker opens to this folder by default.

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

- **Test isolation:** Use `Project.create()` directly with mock executors — don't boot the desktop launcher path when you only need core/application tests.
- **Headless core work:** You can exercise many core APIs without launching the Qt shell, but the default app install path includes PyQt6.
- **Pipeline templates:** Register templates via `@pipeline_template` decorator in `pipelines/templates/`.
- **Logging:** Set `ECHOZERO_LOG_LEVEL=DEBUG` to see execution traces.
