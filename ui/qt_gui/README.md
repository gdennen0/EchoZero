# Qt GUI Layer

PyQt6-based user interface implementation.

## Overview

The Qt GUI provides the graphical interface for EchoZero:
- Node editor for block graph visualization
- Block panels for configuration
- Timeline for event editing
- Dialogs and widgets

## Architecture

```
qt_gui/
├── main_window.py        # Main application window
├── qt_application.py     # Qt app initialization
├── design_system.py      # Visual design tokens
├── theme_registry.py     # Theme management
├── node_editor/          # Block graph visualization
├── block_panels/         # Block configuration UI
├── widgets/              # Reusable widgets
│   └── timeline/         # Timeline components
├── views/                # View controllers
├── dialogs/              # Modal dialogs
└── core/                 # Core UI components
```

## Key Components

- **MainWindow** - Application shell
- **NodeEditor** - Visual block graph
- **BlockPanels** - Per-block configuration
- **Timeline** - Event editing

## Design System

See [DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) for visual design guidelines.

## Related

- [Node Editor](./node_editor/README.md)
- [Block Panels](./block_panels/README.md)
- [Timeline](./widgets/timeline/README.md)
