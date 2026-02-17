# EchoZero UI Module

Separate UI implementations for EchoZero, completely decoupled from the core application.

## Architecture

```
ui/
├── base/              # UI abstraction layer (protocols)
│   └── ui_bridge.py   # UIBridge and BlockUIProvider protocols
│
├── qt_gui/            # Qt implementation
│   ├── qt_application.py    # Main Qt app
│   ├── main_window.py       # Main window with menus
│   │
│   ├── core/                # Shared Qt components
│   │   └── properties_panel.py
│   │
│   ├── node_editor/         # Block graph editor
│   │   ├── node_editor_widget.py
│   │   ├── node_graphics_view.py
│   │   ├── node_scene.py
│   │   ├── block_item.py
│   │   └── connection_item.py
│   │
│   ├── timeline/            # DAW-style timeline
│   │   ├── timeline_widget.py
│   │   ├── timeline_view.py
│   │   ├── event_item.py
│   │   ├── playhead.py
│   │   └── audio_player.py
│   │
│   └── block_uis/           # Block-specific UIs (future)
│
└── web_ui/            # Future: web implementation
```

## Design Principles

### 1. Complete Separation
- Core EchoZero (`src/`) never imports from `ui/`
- UI gets `ApplicationFacade` reference, not the other way around
- Multiple UI implementations can coexist

### 2. Protocol-Based
- `UIBridge` protocol defines UI contract
- `BlockUIProvider` protocol for custom block UIs
- Allows different frameworks (Qt, web, etc.)

### 3. Event-Driven
- UI subscribes to `EventBus` for updates
- Core publishes events, UI reacts
- No tight coupling

## Qt GUI Features

### Node Editor
- Visual block graph editing
- Drag blocks to arrange
- Click-drag to connect blocks
- Context menus for block operations
- Auto-layout support
- Zoom and pan

### Timeline
- DAW-style event display
- Multiple layers (tracks)
- Drag events between layers
- Grid snapping
- Zoom controls
- Audio playback synchronization
- Playhead indicator
- Event inspection

### Properties Panel
- Displays selected item details
- Block parameters
- Event metadata
- Connection info

### Main Window
- Menu bar (File, Edit, Project, View, Help)
- Toolbar (quick actions)
- Dockable panels
- Status bar
- Dark theme

## Usage

### Launch Qt GUI

```bash
python main_qt.py
```

### Creating Custom Block UIs

Implement `BlockUIProvider` protocol:

```python
from ui.base.ui_bridge import BlockUIProvider
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSpinBox

class MyBlockUI(BlockUIProvider):
    def get_block_type(self) -> str:
        return "MyBlock"
    
    def create_editor_widget(self, block_data):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Add custom controls
        spinbox = QSpinBox()
        layout.addWidget(spinbox)
        
        widget.setLayout(layout)
        return widget
    
    def get_parameter_values(self):
        return {"my_param": self.spinbox.value()}
    
    def set_parameter_values(self, values):
        if "my_param" in values:
            self.spinbox.setValue(values["my_param"])
```

## Dependencies

### Qt GUI Requirements
- PyQt6 >= 6.4.0
- PyQt6-Multimedia >= 6.4.0 (for audio playback)

Install with:
```bash
pip install PyQt6 PyQt6-Multimedia
```

## Performance

### Node Editor
- Handles 100+ blocks efficiently
- Scene graph optimization
- Incremental updates

### Timeline
- Viewport culling (only visible events rendered)
- Efficient event item pooling
- 60 FPS playback updates
- Smooth zoom and pan

### Audio Playback
- Qt Multimedia backend
- Hardware-accelerated when available
- Low-latency position updates

## Keyboard Shortcuts

### Global
- `Ctrl+N`: New project
- `Ctrl+O`: Open project
- `Ctrl+S`: Save project
- `Ctrl+Shift+S`: Save project as
- `F5`: Execute project
- `Ctrl+Q`: Quit

### Node Editor
- `Middle Mouse + Drag`: Pan view
- `Ctrl + Mouse Wheel`: Zoom
- `Delete`: Delete selected block

### Timeline
- `Space`: Play/pause
- `Ctrl + Mouse Wheel`: Zoom
- `Shift + Click`: Multi-select events

## Extending

### Adding New UI Implementation

1. Create directory: `ui/my_ui/`
2. Implement `UIBridge` protocol
3. Create entry point: `main_my_ui.py`
4. Launch independently

Example:
```python
from src.application.bootstrap import initialize_services
from ui.my_ui.my_application import MyUIApp

container = initialize_services()
app = MyUIApp()
app.initialize(container.facade)
app.run()
```

### Adding Block-Specific UIs

1. Create file in `ui/qt_gui/block_uis/`
2. Implement `BlockUIProvider` protocol
3. Register with block UI registry
4. UI automatically appears in properties panel

## Future Plans

### Near-term
- Block-specific parameter editors
- Undo/redo support
- Keyboard shortcut customization
- UI state persistence
- Waveform display in timeline

### Medium-term
- Web-based UI (`ui/web_ui/`)
- Touch-optimized layouts
- Plugin UI system
- Theme customization

### Long-term
- Collaborative editing
- Remote UI (headless core + web UI)
- Mobile companion app
- VR/AR experimentation

## Testing

UI testing is primarily manual due to Qt's event-driven nature.

### Test Checklist

**Node Editor:**
- [ ] Create blocks
- [ ] Delete blocks
- [ ] Connect blocks
- [ ] Disconnect blocks
- [ ] Drag blocks
- [ ] Auto-layout
- [ ] Zoom and pan

**Timeline:**
- [ ] Display events
- [ ] Select events
- [ ] Drag events
- [ ] Change layers
- [ ] Grid snapping
- [ ] Zoom timeline
- [ ] Audio playback
- [ ] Playhead sync

**Integration:**
- [ ] Load project
- [ ] Save project
- [ ] Execute project
- [ ] View results
- [ ] Properties panel updates

## Troubleshooting

### Qt Not Found
```
ModuleNotFoundError: No module named 'PyQt6'
```
**Solution:** `pip install PyQt6 PyQt6-Multimedia`

### Audio Playback Issues
- Ensure Qt Multimedia is installed
- Check audio file format (WAV, MP3 supported)
- Verify audio output device availability

### Performance Issues
- Reduce number of events displayed
- Increase timeline grid size
- Disable anti-aliasing in settings

### Dark Theme Not Working
- Qt Fusion style required
- Some platforms may override theme
- Check Qt platform theme settings

## Contributing

When adding UI features:

1. Keep core separation intact
2. Use `ApplicationFacade` for all operations
3. Subscribe to `EventBus` for updates
4. Follow existing patterns
5. Add to test checklist
6. Update this README

## Questions?

See `AgentAssets/PROJECT_OVERVIEW.md` for architecture details.
See `docs/HANDBOOK.md` for core EchoZero documentation.

