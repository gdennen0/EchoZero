#!/usr/bin/env python3
"""
Script to remove #region agent log blocks from Python files.

These debug logging blocks follow the pattern:
    # #region agent log
    try:
        ...
    except: pass
    # #endregion

Or sometimes with Exception:
    # #region agent log
    try:
        ...
    except Exception:
        pass
    # #endregion
"""

import re
import sys
from pathlib import Path


def remove_agent_log_blocks(content: str) -> str:
    """Remove all #region agent log blocks from content."""
    
    # Pattern to match the full block including indentation
    # Handles both "except: pass" and "except Exception:\n    pass"
    pattern = r'''
        ^(?P<indent>[ \t]*)          # Capture leading indent
        \#[ ]*\#region[ ]+agent[ ]+log\n   # Opening region
        (?:.*?\n)*?                   # Non-greedy match of content lines
        (?P=indent)\#[ ]*\#endregion\n?    # Closing endregion with same indent
    '''
    
    # Try multiline pattern
    result = re.sub(pattern, '', content, flags=re.MULTILINE | re.VERBOSE)
    
    # Also handle inline single-line debug blocks (rare but possible)
    result = re.sub(r'[ \t]*# #region agent log.*?# #endregion\n?', '', result, flags=re.DOTALL)
    
    return result


def process_file(filepath: Path) -> bool:
    """Process a single file, removing debug blocks. Returns True if modified."""
    try:
        original = filepath.read_text()
        cleaned = remove_agent_log_blocks(original)
        
        if cleaned != original:
            filepath.write_text(cleaned)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main entry point."""
    # Files to clean (those that will be kept after refactor)
    files_to_clean = [
        "ui/qt_gui/block_panels/show_manager_panel.py",
        "src/application/settings/show_manager_settings.py",
        "src/features/show_manager/application/commands/add_synced_ma3_track_command.py",
        "ui/qt_gui/widgets/timeline/core/widget.py",
        "ui/qt_gui/block_panels/editor_panel.py",
        "src/features/blocks/application/editor_api.py",
        "src/application/commands/editor_commands.py",
        "src/shared/infrastructure/persistence/data_item_repository_impl.py",
        "src/features/ma3/application/ma3_communication_service.py",
        "src/features/show_manager/application/sync_system_manager.py",
        "src/features/show_manager/application/commands/add_synced_editor_layer_command.py",
        "src/features/show_manager/application/commands/remove_synced_entity_command.py",
        "src/application/commands/block_commands.py",
        "src/features/show_manager/application/ma3_event_handler.py",
        "ui/qt_gui/dialogs/add_editor_layer_dialog.py",
        "src/application/blocks/editor_block.py",
        "src/application/settings/block_settings.py",
        "src/application/api/application_facade.py",
        "src/application/services/layer_order_service.py",
        "ui/qt_gui/widgets/timeline/events/layer_manager.py",
        "src/application/commands/data_item_commands.py",
        "src/application/blocks/show_manager_block.py",
        "src/application/services/sync_port.py",
        "ui/qt_gui/node_editor/block_item.py",
        "ui/qt_gui/views/setlist_view.py",
        "ui/qt_gui/dialogs/setlist_processing_dialog.py",
        "ui/qt_gui/node_editor/node_editor_window.py",
        "ui/qt_gui/block_panels/detect_onsets_panel.py",
        "ui/qt_gui/core/setlist_processing_thread.py",
        "src/application/commands/command_bus.py",
        "src/features/setlists/application/setlist_service.py",
        "src/features/execution/application/progress_tracker.py",
        "src/features/blocks/application/block_service.py",
        "src/features/blocks/application/block_status_service.py",
    ]
    
    project_root = Path(__file__).parent.parent
    modified_count = 0
    
    for rel_path in files_to_clean:
        filepath = project_root / rel_path
        if filepath.exists():
            if process_file(filepath):
                print(f"Cleaned: {rel_path}")
                modified_count += 1
            else:
                print(f"No changes: {rel_path}")
        else:
            print(f"Not found: {rel_path}")
    
    print(f"\nTotal files modified: {modified_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
