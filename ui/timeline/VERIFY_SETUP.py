#!/usr/bin/env python
"""
VERIFY_SETUP.py — Quick sanity check for the EchoZero 2 Timeline Prototype
Run this to confirm all imports and basic functionality work.
"""
import sys

def check_python_version():
    """Ensure Python 3.7+"""
    if sys.version_info < (3, 7):
        print(f"ERROR: Python 3.7+ required, you have {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"OK - Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_pyqt6():
    """Check PyQt6 is installed"""
    try:
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QApplication, QWidget
        from PyQt6.QtCore import Qt
        print("OK - PyQt6 imports work")
        return True
    except ImportError as e:
        print(f"ERROR - PyQt6 not found: {e}")
        print("  Install with: pip install PyQt6")
        return False


def check_model():
    """Verify model.py works"""
    try:
        from model import (
            TimelineEvent, TimelineLayer, TimelineState, ViewportRect,
            visible_events, generate_fake_data
        )
        
        # Test data generation
        state = generate_fake_data(num_events=10, num_layers=3, duration=60.0)
        assert len(state.events) == 10, f"Expected 10 events, got {len(state.events)}"
        assert len(state.layers) == 3, f"Expected 3 layers, got {len(state.layers)}"
        
        # Test visible_events
        visible = visible_events(state.events, 0.0, 30.0)
        assert len(visible) <= 10, "visible_events returned more than total events"
        
        # Test that events are sorted
        times = [e.time for e in state.events]
        assert times == sorted(times), "Events not sorted by time"
        
        print(f"OK - model.py: generated {len(state.events)} events, {len(state.layers)} layers")
        return True
    except Exception as e:
        print(f"ERROR - model.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_feel():
    """Verify FEEL.py exports all required constants"""
    try:
        import FEEL
        
        required = [
            'BG_COLOR', 'GRID_MINOR_COLOR', 'GRID_MAJOR_COLOR',
            'RULER_HEIGHT', 'RULER_BG_COLOR', 'PLAYHEAD_COLOR',
            'LAYERS_PANEL_WIDTH', 'EVENT_HEIGHT', 'LAYER_ROW_HEIGHT',
            'ZOOM_MIN', 'ZOOM_MAX', 'ZOOM_STEP',
        ]
        
        missing = [c for c in required if not hasattr(FEEL, c)]
        if missing:
            print(f"ERROR - FEEL.py missing: {missing}")
            return False
        
        print(f"OK - FEEL.py: {len(required)} required constants present")
        return True
    except Exception as e:
        print(f"ERROR - FEEL.py: {e}")
        return False


def check_canvas():
    """Verify canvas.py imports (doesn't instantiate GUI)"""
    try:
        from canvas import TimelineCanvas
        print("OK - canvas.py imports")
        return True
    except Exception as e:
        print(f"ERROR - canvas.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ruler():
    """Verify ruler.py imports"""
    try:
        from ruler import TimeRuler
        print("OK - ruler.py imports")
        return True
    except Exception as e:
        print(f"ERROR - ruler.py: {e}")
        return False


def check_layers_panel():
    """Verify layers_panel.py imports"""
    try:
        from layers_panel import LayersPanel
        print("OK - layers_panel.py imports")
        return True
    except Exception as e:
        print(f"ERROR - layers_panel.py: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("EchoZero 2 Timeline Prototype — Setup Verification")
    print("="*60 + "\n")
    
    checks = [
        ("Python version", check_python_version),
        ("PyQt6 installation", check_pyqt6),
        ("FEEL.py constants", check_feel),
        ("model.py functionality", check_model),
        ("canvas.py imports", check_canvas),
        ("ruler.py imports", check_ruler),
        ("layers_panel.py imports", check_layers_panel),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n{passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("\nSUCCESS! Ready to run: python prototype.py\n")
        return 0
    else:
        print("\nFAILURE! Fix errors above, then retry.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
