# Block Processors Package

## Overview

This package contains all block processor implementations. Processors are automatically registered with the execution engine when the application starts.

## Auto-Registration System

Processors are automatically registered when:
1. Their module is imported (via `register_processor_class()`)
2. `register_all_processors()` is called (which happens in bootstrap)

### How It Works

Each processor module:
1. Defines a `BlockProcessor` subclass
2. Calls `register_processor_class()` at module level

Example:
```python
from src.application.processing.block_processor import BlockProcessor
from src.application.blocks import register_processor_class

class MyBlockProcessor(BlockProcessor):
    def get_block_type(self) -> str:
        return "MyBlockType"
    # ... implementation ...

# Auto-register
register_processor_class(MyBlockProcessor)
```

When bootstrap runs, it imports all processor modules and calls `register_all_processors()`, which:
1. Imports all processor modules (triggering their registration)
2. Instantiates each registered processor class
3. Registers the instance with the execution engine

## Adding a New Processor

1. Create a new file: `src/application/blocks/my_block.py`
2. Implement `BlockProcessor` subclass
3. Call `register_processor_class()` at module level
4. Add import to `src/application/blocks/__init__.py` in `register_all_processors()`

That's it! No manual registration needed elsewhere.

## Processor Files

- `load_audio_block.py` - LoadAudio processor ✅ Implemented
- `detect_onsets_block.py` - DetectOnsets processor ✅ Implemented
- `separator_block.py` - Separator processor ✅ Implemented
- `editor_block.py` - Editor/EditorV2 processor ✅ Implemented
- `export_audio_block.py` - ExportAudio processor ✅ Implemented
- `tensorflow_classify_block.py` - TensorFlowClassify processor ✅ Implemented
- `pytorch_classify_block.py` - PyTorchClassify processor ✅ Implemented
- `pytorch_drum_trainer_block.py` - PyTorchDrumTrainer processor ✅ Implemented

## Benefits

- **No Manual Registration** - Just create the processor file
- **Automatic Discovery** - Bootstrap finds all processors
- **Simple Pattern** - One line per processor (`register_processor_class()`)
- **Clear Location** - All processors in one place

