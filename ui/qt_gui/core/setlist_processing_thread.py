"""
Setlist Processing Thread

Background thread for processing setlists without blocking the UI.
Uses Qt signals for thread-safe communication back to main thread.
"""
from typing import Optional, Dict, Any, List, Callable
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.message import Log


class SetlistProcessingThread(QThread):
    """
    Background thread for setlist processing.
    
    Keeps UI responsive while setlist processing runs (including long Demucs operations).
    Uses Qt signals for thread-safe communication back to main thread.
    
    Signals:
        processing_started: Emitted when processing begins
        song_progress: Emitted for song-level progress (song_id, status)
        action_progress: Emitted for action-level progress (song_id, action_index, total, name, status)
        processing_complete: Emitted when all processing is done (results dict)
        processing_failed: Emitted on fatal error (error message, details)
        error_occurred: Emitted for per-song errors (song_path, error_message)
    """
    
    # Thread-safe signals for UI updates
    processing_started = pyqtSignal()
    song_progress = pyqtSignal(str, str)  # song_id, status
    action_progress = pyqtSignal(str, int, int, str, str)  # song_id, action_index, total_actions, action_name, status
    processing_complete = pyqtSignal(bool, dict)  # success, results dict {song_id: bool}
    processing_failed = pyqtSignal(str, list)  # error message, detailed errors
    error_occurred = pyqtSignal(str, str)  # song_path, error_message
    
    def __init__(self, facade, setlist_id: str, parent=None):
        """
        Initialize setlist processing thread.
        
        Args:
            facade: ApplicationFacade instance
            setlist_id: Setlist ID to process
            parent: Parent QObject
        """
        super().__init__(parent)
        self.facade = facade
        self.setlist_id = setlist_id
        self._should_cancel = False
        self._errors: List[Dict[str, str]] = []
        
        Log.info(f"SetlistProcessingThread: Created for setlist {setlist_id}")
    
    def run(self):
        """
        Execute setlist processing in background thread.
        
        This method runs in a separate thread, keeping the UI responsive.
        All GUI updates must happen via signals, not direct calls.
        """
        try:
            Log.info(f"SetlistProcessingThread: Starting processing for setlist {self.setlist_id}")
            Log.info(f"SetlistProcessingThread: Facade event_bus id: {id(self.facade.event_bus) if self.facade.event_bus else 'None'}")
            self.processing_started.emit()
            
            # Create thread-safe callbacks that emit signals
            def error_callback(song_path: str, error_message: str):
                """Emit error signal for per-song errors"""
                self._errors.append({"song": song_path, "error": error_message})
                self.error_occurred.emit(song_path, error_message)
            
            def action_progress_callback(song_id: str, action_index: int, total_actions: int, action_name: str, status: str):
                """Emit action progress signal"""
                Log.info(f"SetlistProcessingThread: action_progress_callback called - song={song_id}, action={action_index}/{total_actions}, name={action_name}, status={status}")
                
                # When first action starts for a song, emit song processing status
                if action_index == 0 and status == "running":
                    self.song_progress.emit(song_id, "processing")
                
                self.action_progress.emit(song_id, action_index, total_actions, action_name, status)
            
            # Process setlist (this is the long-running operation)
            result = self.facade.process_setlist(
                setlist_id=self.setlist_id,
                error_callback=error_callback,
                action_progress_callback=action_progress_callback,
                cancel_check=self.should_cancel
            )
            
            if result.success:
                Log.info(f"SetlistProcessingThread: Processing completed successfully")
                # Emit final song statuses
                if result.data:
                    for song_id, success in result.data.items():
                        status = "completed" if success else "failed"
                        self.song_progress.emit(song_id, status)
                
                self.processing_complete.emit(True, result.data or {})
            else:
                Log.error(f"SetlistProcessingThread: Processing failed: {result.message}")
                self.processing_failed.emit(result.message, result.errors or [])
                
        except Exception as e:
            Log.error(f"SetlistProcessingThread: Exception during processing: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            self.processing_failed.emit(str(e), [str(e), f"Traceback:\n{error_traceback}"])
    
    def request_cancel(self):
        """
        Request cancellation of processing.
        
        Sets the cancellation flag which is checked via should_cancel() callback.
        """
        Log.info("SetlistProcessingThread: Cancellation requested")
        self._should_cancel = True
    
    def should_cancel(self) -> bool:
        """Check if cancellation has been requested."""
        return self._should_cancel
    
    def get_errors(self) -> List[Dict[str, str]]:
        """Get list of errors that occurred during processing"""
        return self._errors.copy()

