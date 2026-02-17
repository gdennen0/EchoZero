# Quick Actions Input Dialogs - Reference Guide

## Purpose

This document provides a reference for creating input dialogs in quick actions, ensuring consistency and preventing common errors.

## Number Input Dialogs

### PyQt6 API Signature

**CRITICAL:** `QInputDialog.getDouble()` has a specific parameter order that MUST be followed:

```python
QInputDialog.getDouble(
    parent,              # QWidget
    title,               # str
    label,               # str
    value,               # float (default value)
    min,                 # float (minimum)
    max,                 # float (maximum)
    decimals,            # int (decimal places)
    flags,               # Qt.WindowType (REQUIRED - do not omit!)
    step                 # float (step size for arrows)
)
```

**Common Error:** Omitting the `flags` parameter or placing `step` before `flags` will cause:
```
argument 8 has unexpected type 'float'
```

### Required Parameters for Number Inputs

When returning a number input request from a quick action, **always include**:

```python
{
    "needs_input": True,
    "input_type": "number",
    "min": 0.0,                    # REQUIRED: Minimum value
    "max": 1.0,                    # REQUIRED: Maximum value
    "default": current_value,      # REQUIRED: Current value from database (single source of truth)
    "decimals": 2,                 # REQUIRED: Decimal places
    "increment_jump": 0.05,        # REQUIRED: Step size for increment/decrement arrows
    "title": "Dialog Title"
}
```

### Parameter Details

- **min/max**: Always specify explicit bounds. Prevents invalid input and provides clear constraints.
- **decimals**: Controls precision. Common values: 0 (integers), 2 (percentages), 3 (time in seconds).
- **increment_jump**: Step size when user clicks up/down arrows. Should match typical adjustment size.
  - Preferred name: `increment_jump`
  - Fallback name: `step` (supported for backwards compatibility)
- **default**: Always read from settings manager (single source of truth), never hardcode.

### Example: Threshold Input

```python
@quick_action("DetectOnsets", "Tune Sensitivity", ...)
def action_tune_sensitivity(facade, block_id: str, value: float = None, **kwargs):
    from src.application.settings.detect_onsets_settings import DetectOnsetsSettingsManager
    
    if value is not None:
        # Write path
        settings_manager = DetectOnsetsSettingsManager(facade, block_id)
        settings_manager.onset_threshold = float(value)
        settings_manager.force_save()
        return {"success": True, "message": f"Sensitivity set to {value:.2f}"}
    
    # Read path: get current value
    try:
        settings_manager = DetectOnsetsSettingsManager(facade, block_id)
        current_value = settings_manager.onset_threshold
    except Exception:
        current_value = 0.5  # Fallback only
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 0.0,
        "max": 1.0,
        "default": current_value,      #  From database
        "decimals": 2,
        "increment_jump": 0.05,         #  Step size for arrows
        "title": "Onset Sensitivity (0.0-1.0)"
    }
```

## Choice Input Dialogs

For dropdown/choice inputs:

```python
{
    "needs_input": True,
    "input_type": "choice",
    "choices": ["option1", "option2", "option3"],
    "default": current_value,      #  From database (single source of truth)
    "title": "Select Option"
}
```

## Directory Input Dialogs

For directory selection:

```python
{
    "needs_input": True,
    "input_type": "directory",
    "title": "Select Directory"
}
```

## Text Input Dialogs

For text input:

```python
{
    "needs_input": True,
    "input_type": "text",
    "default": current_value,      #  From database
    "title": "Enter Value"
}
```

## Checklist for New Number Inputs

When creating a new number input quick action:

- [ ] Read current value from settings manager (not hardcoded)
- [ ] Specify `min` parameter
- [ ] Specify `max` parameter
- [ ] Specify `decimals` parameter
- [ ] Specify `increment_jump` parameter (or `step` as fallback)
- [ ] Use `force_save()` after setting value
- [ ] Test that dialog opens without errors
- [ ] Test that increment/decrement arrows work correctly
- [ ] Test that min/max constraints are enforced

## Prevention

**To prevent the `argument 8 has unexpected type 'float'` error:**

1. **Always check PyQt6 API documentation** when using dialog functions
2. **Use named parameters** when possible (though getDouble doesn't support this)
3. **Add inline comments** showing parameter order in code
4. **Reference this document** when creating new number inputs
5. **Test dialogs immediately** after creation

## Implementation Location

The input dialog handler is located in:
- `ui/qt_gui/core/actions_panel.py` - `_handle_input_request()` method
- Contains inline documentation explaining the correct parameter order

---

**Last Updated:** After fixing QInputDialog.getDouble() parameter order issue
**Related:** `AgentAssets/SETTINGS_ABSTRACTION_PRESET.md`
