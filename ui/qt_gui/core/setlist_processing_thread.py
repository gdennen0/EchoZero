"""
Setlist Processing Thread

Background thread for processing setlists without blocking the UI.
Uses Qt signals for thread-safe communication back to main thread.
Progress is reported via ProgressStore (polled by the UI).
"""
from typing import Dict, List
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.message import Log


class SetlistProcessingThread(QThread):
    """
    Background thread for setlist processing.

    Keeps UI responsive while setlist processing runs (including long Demucs operations).
    Progress is written to ProgressStore by the processing service. The UI
    reads it via polling -- no callback chain passes through this thread.

    Signals:
        processing_started: Emitted when processing begins
        song_progress: Emitted for song-level progress (song_id, status)
        processing_complete: Emitted when all processing is done (results dict)
        processing_failed: Emitted on fatal error (error message, details)
        error_occurred: Emitted for per-song errors (song_path, error_message)
    """

    processing_started = pyqtSignal()
    song_progress = pyqtSignal(str, str)  # song_id, status
    processing_complete = pyqtSignal(bool, dict)  # success, results dict {song_id: bool}
    processing_failed = pyqtSignal(str, list)  # error message, detailed errors
    error_occurred = pyqtSignal(str, str)  # song_path, error_message

    def __init__(self, facade, setlist_id: str, song_id: str = None, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.setlist_id = setlist_id
        self.song_id = song_id
        self._should_cancel = False
        self._errors: List[Dict[str, str]] = []
        mode = f"song {song_id}" if song_id else "all songs"
        Log.info(f"SetlistProcessingThread: Created for setlist {setlist_id} ({mode})")

    def run(self):
        """Execute setlist processing in background thread."""
        try:
            self.processing_started.emit()

            if self.song_id:
                self._run_single_song()
            else:
                self._run_all_songs()

        except Exception as e:
            Log.error(f"SetlistProcessingThread: Exception during processing: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            self.processing_failed.emit(str(e), [str(e), f"Traceback:\n{error_traceback}"])

    def _run_all_songs(self):
        """Process all songs in the setlist."""
        Log.info(f"SetlistProcessingThread: Starting processing for setlist {self.setlist_id}")

        def error_callback(song_path: str, error_message: str):
            self._errors.append({"song": song_path, "error": error_message})
            self.error_occurred.emit(song_path, error_message)

        result = self.facade.process_setlist(
            setlist_id=self.setlist_id,
            error_callback=error_callback,
            cancel_check=self.should_cancel
        )

        if result.success:
            Log.info(f"SetlistProcessingThread: Processing completed successfully")
            if result.data:
                for sid, success in result.data.items():
                    status = "completed" if success else "failed"
                    self.song_progress.emit(sid, status)
            self.processing_complete.emit(True, result.data or {})
        else:
            Log.error(f"SetlistProcessingThread: Processing failed: {result.message}")
            self.processing_failed.emit(result.message, result.errors or [])

    def _run_single_song(self):
        """Process a single song."""
        Log.info(f"SetlistProcessingThread: Processing single song {self.song_id}")

        result = self.facade.process_song(
            setlist_id=self.setlist_id,
            song_id=self.song_id,
        )

        if result.success:
            Log.info(f"SetlistProcessingThread: Song {self.song_id} processed successfully")
            self.song_progress.emit(self.song_id, "completed")
            self.processing_complete.emit(True, {self.song_id: True})
        else:
            error_msg = result.message or "Processing failed"
            Log.error(f"SetlistProcessingThread: Song {self.song_id} failed: {error_msg}")
            self.song_progress.emit(self.song_id, "failed")
            self.error_occurred.emit(self.song_id, error_msg)
            self.processing_complete.emit(True, {self.song_id: False})

    def request_cancel(self):
        """Request cancellation of processing."""
        Log.info("SetlistProcessingThread: Cancellation requested")
        self._should_cancel = True

    def should_cancel(self) -> bool:
        """Check if cancellation has been requested."""
        return self._should_cancel

    def get_errors(self) -> List[Dict[str, str]]:
        """Get list of errors that occurred during processing."""
        return self._errors.copy()

