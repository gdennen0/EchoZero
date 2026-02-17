# EchoZero

A modular audio processing engine with a node-based architecture. Build audio processing pipelines by connecting blocks (processing nodes) in a visual graph.

## Features

- **Node-based workflow**: Visual block graph for audio processing pipelines
- **Extensible block system**: Easy to add new processing blocks
- **Multiple interfaces**: CLI for scripting, Qt GUI for visual editing
- **Audio source separation**: Demucs integration for stem separation
- **Note transcription**: Basic-pitch and librosa-based note extraction
- **Event visualization**: Plot and inspect audio events
- **Project files**: Save and share workflows as .ez files

## Quick Start

### Installation

**Important:** Always use a virtual environment to ensure Demucs and other dependencies are properly installed. See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

```bash
# Clone the repository
git clone https://github.com/yourusername/EchoZero.git
cd EchoZero

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (includes Demucs)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Launch GUI

**Always activate the virtual environment first:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main_qt.py
```

### Launch CLI

**Always activate the virtual environment first:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
# or
echozero
```

**Note:** If you see "Demucs is not installed or not in PATH" errors, ensure the virtual environment is activated. See [INSTALLATION.md](INSTALLATION.md) for troubleshooting.

Type `help` at the `EZ>` prompt for available commands.

## Usage Example

### CLI Workflow

```bash
EZ> new MyProject
EZ> add_block LoadAudio audio1
EZ> add_block DetectOnsets onsets1
EZ> connect audio1 audio onsets1 audio
EZ> execute
EZ> save
```

### GUI Workflow

1. Launch with `python main_qt.py`
2. Create a new project (Ctrl+N)
3. Add blocks from the toolbar
4. Connect blocks by dragging from ports
5. Configure blocks via double-click
6. Execute with F5 or the Execute button
7. Save your project (Ctrl+S)

## Available Blocks

EchoZero includes 17+ block types organized by category:

### Input Blocks
- **LoadAudio** - Load audio files (WAV, MP3, FLAC, etc.)
- **SetlistAudioInput** - Audio input for setlist batch processing

### Analysis Blocks
- **DetectOnsets** - Detect onset times in audio using librosa
- **TranscribeNote** (NoteExtractorBasicPitch) - ML-based note extraction
- **TranscribeLib** (NoteExtractorLibrosa) - Librosa-based note extraction

### Classification Blocks
- **TensorFlowClassify** - Classify events using TensorFlow/Keras models
- **PyTorchClassify** - Classify events using PyTorch models
- **PyTorchAudioClassify** - Audio classification with PyTorch (uses models from PyTorchAudioTrainer)

### Training Blocks
- **PyTorchAudioTrainer** - Train PyTorch models for audio classification

### Processing Blocks
- **Separator** - Demucs source separation (vocals, drums, bass, other)
- **SeparatorCloud** - Cloud-based Demucs separation (AWS)

### Editor & Visualization
- **Editor** - Interactive audio editor with timeline visualization
- **PlotEvents** - Create timeline visualizations of events

### Output Blocks
- **ExportAudio** - Export audio to file
- **ExportClipsByClass** - Export audio clips organized by classification

### Integration Blocks
- **ShowManager** - GrandMA3 lighting console integration and sync

## Project Structure

```
EchoZero/
├── src/
│   ├── features/           # Feature modules (vertical slices)
│   │   ├── blocks/         # Block entities and services
│   │   ├── connections/    # Block connections
│   │   ├── execution/      # Execution engine
│   │   ├── projects/       # Project management
│   │   ├── setlists/       # Setlist batch processing
│   │   ├── show_manager/   # Show-based workflows & MA3 sync
│   │   └── ma3/            # GrandMA3 integration
│   ├── application/        # Application services
│   │   ├── blocks/         # Block processors (execution logic)
│   │   ├── commands/       # Command pattern (undo/redo)
│   │   ├── settings/       # Settings system
│   │   └── api/            # ApplicationFacade
│   ├── shared/            # Cross-cutting concerns
│   │   ├── application/    # Events, registry, validation
│   │   ├── domain/         # Shared entities
│   │   └── infrastructure/ # Base repositories
│   └── infrastructure/    # Persistence layer
├── ui/
│   └── qt_gui/            # PyQt6 interface
│       ├── block_panels/   # Block configuration UI
│       ├── node_editor/    # Visual graph editor
│       └── widgets/       # Reusable widgets
├── tests/                 # Test suite
├── docs/                  # Documentation (including web interface)
├── ma3_plugins/           # GrandMA3 Lua plugins
├── main.py                # CLI entry point
└── main_qt.py             # GUI entry point
```

## Documentation

### Web Documentation Interface

Open `docs/index.html` in your browser for an interactive documentation interface with:
- Complete block reference
- Architecture overview
- Feature module documentation
- Search functionality

### Documentation Files

- [Architecture](docs/ARCHITECTURE.md) - System architecture overview
- [Feature Modules](docs/README.md) - Feature module documentation index
- [MA3 Integration](docs/show_manager_sync_system.md) - Show Manager and MA3 sync system
- [MA3 Sync Cases](docs/MA3_SYNC_CASES.md) - MA3 integration use cases

### Feature Module READMEs

Each feature module includes its own README:
- `src/features/blocks/README.md` - Block system
- `src/features/connections/README.md` - Connection management
- `src/features/execution/README.md` - Execution engine
- `src/features/projects/README.md` - Project management
- `src/features/setlists/README.md` - Setlist processing
- `src/features/show_manager/README.md` - Show Manager
- `src/features/ma3/README.md` - MA3 integration

## Requirements

- Python 3.10+
- PyQt6 (for GUI)
- librosa, soundfile (audio processing)
- torch, torchaudio (ML features)
- demucs (source separation)

See `requirements.txt` for complete list.

## User Data

EchoZero stores user data in standard locations:

- **macOS**: `~/Library/Application Support/EchoZero/`
- **Linux**: `~/.local/share/echozero/`
- **Windows**: `%APPDATA%/EchoZero/`

## Packaging (PyInstaller)

Build is driven by `echozero.spec` and `packaging_config.json` (version, bundle identifier, company name).

```bash
# From project root, with venv activated
pip install pyinstaller
python scripts/build_app.py
# or: pyinstaller echozero.spec
# Optional: python scripts/build_app.py --clean  (clean cache first)
```

- **Output**: `dist/EchoZero/` (one-folder). On macOS, `dist/EchoZero.app` is also created.
- **Run**: `dist/EchoZero/EchoZero` (or `EchoZero.exe` on Windows); on macOS you can run the .app.
- **Zero-config shipping**: To ship a build that needs no user configuration, set auth at build time: `MEMBERSTACK_APP_SECRET=your_secret python scripts/build_app.py`. The secret is embedded in the bundle; end users just run the app. See `docs/PACKAGING.md`.
- **Config (no embed)**: Otherwise set `MEMBERSTACK_APP_SECRET` via a `.env` file next to the executable or in the environment (copy `.env.example` to `.env`).
- **Releases**: Update `version` in `setup.py` and `packaging_config.json` together.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read the development documentation and follow the existing code patterns.
