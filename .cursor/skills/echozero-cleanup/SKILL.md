---
name: echozero-cleanup
description: Resource cleanup patterns in EchoZero blocks. Use when implementing cleanup(), fixing memory leaks, block resource management, or when the user asks about @cleanup, block cleanup, or resource disposal.
---

# Cleanup Patterns

## When Cleanup Is Needed

Blocks that use:
- Timers
- Media players
- UI windows
- File handles
- Network connections
- Cached data

## BlockProcessor.cleanup()

Override in BlockProcessor. Called when:
- Block is removed
- Project is unloaded

```python
def cleanup(self, block: Block) -> None:
    """Clean up resources for this block."""
    if hasattr(self, '_timer') and self._timer:
        self._timer.stop()
        self._timer.deleteLater()
        self._timer = None
    if hasattr(self, '_player') and self._player:
        self._player.stop()
        self._player.deleteLater()
        self._player = None
```

## Qt Objects

Use `deleteLater()` for Qt objects - never delete directly:

```python
widget.deleteLater()
timer.deleteLater()
```

## Pattern: Capture Then Clear

```python
def cleanup(self, block: Block) -> None:
    timer = getattr(self, '_timer', None)
    if timer:
        timer.stop()
        timer.deleteLater()
        delattr(self, '_timer')
```

## Common Resources

| Resource | Cleanup |
|----------|---------|
| QTimer | stop(), deleteLater() |
| QMediaPlayer | stop(), deleteLater() |
| QWidget (owned) | deleteLater() |
| File handle | close() |
| Socket | close() |
| Cached dict/list | clear() or = {} |

## Checklist

- [ ] Identify all resources in block
- [ ] Implement cleanup() override
- [ ] Call stop/close before deleteLater for Qt
- [ ] Clear references to prevent leaks
- [ ] Test: add block, remove block, verify no leaks

## Reference

- SUMMARY: `AgentAssets/modules/commands/cleanup/SUMMARY.md`
- Block implementation: echozero-block-implementation skill
