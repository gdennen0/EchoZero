"""
Playback Controller

Coordinates audio playback with timeline visualization.
Updates playhead at 60 FPS during playback.
"""

from typing import Optional, Callable
from PyQt6.QtCore import QObject, QTimer, QElapsedTimer, pyqtSignal

# Local imports (makes package standalone)
from ..logging import TimelineLog as Log

from ..constants import PLAYHEAD_UPDATE_INTERVAL_MS
from ..interfaces import PlaybackInterface


class PlaybackController(QObject):
    """
    Coordinates audio playback with timeline playhead.
    
    Features:
    - 60 FPS playhead updates during playback
    - Smooth interpolation between audio position updates
    - Transport controls (play, pause, stop, seek)
    - Position change notifications
    
    Signals:
        position_changed(seconds): Current position changed
        playback_started(): Playback started
        playback_paused(): Playback paused
        playback_stopped(): Playback stopped
    """
    
    position_changed = pyqtSignal(float)  # Current position in seconds
    playback_started = pyqtSignal()
    playback_paused = pyqtSignal()
    playback_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._audio_backend: Optional[PlaybackInterface] = None
        self._is_playing = False
        self._current_position = 0.0
        self._duration = 0.0
        
        # Timer for 60 FPS updates
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(PLAYHEAD_UPDATE_INTERVAL_MS)
        self._update_timer.timeout.connect(self._on_update_tick)
        
        # For smooth interpolation
        self._elapsed_timer = QElapsedTimer()
        self._last_known_position = 0.0
        
        # Position change callback (for external sync)
        self._position_callback: Optional[Callable[[float], None]] = None
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing"""
        return self._is_playing
    
    @property
    def position(self) -> float:
        """Get current position in seconds"""
        return self._current_position
    
    @property
    def duration(self) -> float:
        """Get total duration"""
        return self._duration
    
    def set_audio_backend(self, backend: Optional[PlaybackInterface]):
        """
        Set the audio backend for playback.
        
        Args:
            backend: Object implementing PlaybackInterface, or None to disconnect
        """
        # Stop any current playback
        if self._is_playing:
            self.stop()
        
        self._audio_backend = backend
        
        if backend:
            self._duration = backend.get_duration()
            backend_pos = backend.get_position()
            self._current_position = backend_pos
            self._last_known_position = backend_pos
            Log.info(f"PlaybackController: Audio backend connected, duration={self._duration:.2f}s")
        else:
            Log.info("PlaybackController: Audio backend disconnected")
    
    def set_position_callback(self, callback: Optional[Callable[[float], None]]):
        """
        Set callback for position changes.
        
        Args:
            callback: Function to call with position in seconds
        """
        self._position_callback = callback
    
    def play(self):
        """Start or resume playback"""
        if self._is_playing:
            return
        
        if self._audio_backend:
            # Sync position before starting - QMediaPlayer can lose position when stopped
            self._audio_backend.set_position(self._current_position)
            self._audio_backend.play()
        
        self._is_playing = True
        self._last_known_position = self._current_position
        self._elapsed_timer.start()
        self._update_timer.start()
        
        self.playback_started.emit()
        Log.debug(f"PlaybackController: Play from {self._current_position:.3f}s")
    
    def pause(self):
        """Pause playback"""
        if not self._is_playing:
            return
        
        if self._audio_backend:
            self._audio_backend.pause()
        
        self._is_playing = False
        self._update_timer.stop()
        
        # Get final position
        self._sync_position_from_backend()
        
        self.playback_paused.emit()
        Log.debug(f"PlaybackController: Pause at {self._current_position:.3f}s")
    
    def stop(self):
        """Stop playback and reset to start"""
        was_playing = self._is_playing
        
        if self._audio_backend:
            self._audio_backend.stop()
        
        self._is_playing = False
        self._update_timer.stop()
        self._current_position = 0.0
        self._last_known_position = 0.0
        
        self._emit_position()
        
        if was_playing:
            self.playback_stopped.emit()
        Log.debug("PlaybackController: Stop")
    
    def toggle_playback(self):
        """Toggle between play and pause"""
        if self._is_playing:
            self.pause()
        else:
            self.play()
    
    def seek(self, seconds: float):
        """
        Seek to a specific position.
        
        Args:
            seconds: Target position in seconds
        """
        # Only clamp to duration if duration is valid (> 0)
        # If duration is 0, audio might not be loaded yet, so allow seeking to any position
        if self._duration > 0:
            seconds = max(0, min(seconds, self._duration))
        else:
            seconds = max(0, seconds)
        
        if self._audio_backend:
            self._audio_backend.set_position(seconds)
        
        self._current_position = seconds
        self._last_known_position = seconds
        
        if self._is_playing:
            self._elapsed_timer.restart()
        
        self._emit_position()
        Log.debug(f"PlaybackController: Seek to {seconds:.3f}s")
    
    def set_duration(self, seconds: float):
        """
        Set total duration (when no audio backend).
        
        Args:
            seconds: Duration in seconds
        """
        self._duration = max(0, seconds)
    
    def _on_update_tick(self):
        """Handle timer tick for playhead updates"""
        if not self._is_playing:
            return
        
        if self._audio_backend:
            # Get actual position from audio backend
            backend_position = self._audio_backend.get_position()
            backend_is_playing = self._audio_backend.is_playing()
            
            # Interpolate for smoother updates
            # The audio backend might update less frequently than our 60 FPS timer
            elapsed_ms = self._elapsed_timer.elapsed()
            
            # CRITICAL FIX: Always use backend position as ground truth when it changes
            # Compare backend to last_known to detect actual backend updates
            position_diff_from_last_known = abs(backend_position - self._last_known_position)
            
            if position_diff_from_last_known > 0.001:  # Backend position changed
                # Backend updated - sync immediately to prevent jitter
                self._current_position = backend_position
                self._last_known_position = backend_position
                self._elapsed_timer.restart()
            else:
                # Backend position hasn't changed, use smooth interpolation
                estimated_position = self._last_known_position + (elapsed_ms / 1000.0)
                self._current_position = estimated_position
            
            # Check for end of playback
            if not backend_is_playing:
                self.stop()
                return
        else:
            # No audio backend - just increment time
            elapsed_ms = self._elapsed_timer.elapsed()
            self._current_position = self._last_known_position + (elapsed_ms / 1000.0)
            
            # Check for end
            if self._current_position >= self._duration:
                self._current_position = self._duration
                self.stop()
                return
        
        self._emit_position()
    
    def _sync_position_from_backend(self):
        """Sync current position from audio backend"""
        if self._audio_backend:
            self._current_position = self._audio_backend.get_position()
            self._last_known_position = self._current_position
    
    def _emit_position(self):
        """Emit position changed signal and call callback"""
        self.position_changed.emit(self._current_position)
        
        if self._position_callback:
            self._position_callback(self._current_position)
    
    def cleanup(self):
        """Clean up resources"""
        self._update_timer.stop()
        self._audio_backend = None
        Log.debug("PlaybackController: Cleanup complete")


class SimpleAudioPlayer:
    """
    Simple audio player implementation using Qt Multimedia.
    
    Implements PlaybackInterface for integration with PlaybackController.
    This is a minimal implementation - replace with a more sophisticated
    audio engine for production use.
    """
    
    def __init__(self):
        self._player = None
        self._audio_output = None
        self._file_path: Optional[str] = None
        self._duration = 0.0
        
        self._init_player()
    
    def _init_player(self):
        """Initialize Qt Multimedia player"""
        self._play_when_ready = False
        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtCore import QUrl

            self._player = QMediaPlayer()
            self._audio_output = QAudioOutput()
            self._audio_output.setVolume(1.0)
            self._player.setAudioOutput(self._audio_output)

            self._player.durationChanged.connect(self._on_duration_changed)
            self._player.mediaStatusChanged.connect(self._on_media_status)

            Log.info("SimpleAudioPlayer: Qt Multimedia initialized")
        except ImportError:
            Log.warning("SimpleAudioPlayer: Qt Multimedia not available")

    def _on_media_status(self, status):
        """When media is loaded, start playback if play_when_ready was set."""
        from PyQt6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if getattr(self, "_play_when_ready", False):
                self._play_when_ready = False
                if self._player:
                    self._player.play()

    def _on_duration_changed(self, duration_ms: int):
        """Handle duration change"""
        self._duration = duration_ms / 1000.0

    def load(self, file_path: str, play_when_ready: bool = False) -> bool:
        """
        Load an audio file.
        
        Args:
            file_path: Path to audio file (will be resolved to absolute for QUrl)
            
        Returns:
            True if loaded successfully
        """
        if not self._player:
            return False
        
        try:
            from PyQt6.QtCore import QUrl
            import os
            
            self._player.stop()
            self._player.setPosition(0)
            self._file_path = os.path.abspath(os.path.normpath(file_path))
            url = QUrl.fromLocalFile(self._file_path)
            self._play_when_ready = bool(play_when_ready)
            self._player.setSource(url)

            Log.info(f"SimpleAudioPlayer: Loaded {self._file_path}")
            return True
        except Exception as e:
            Log.error(f"SimpleAudioPlayer: Failed to load {file_path}: {e}")
            return False
    
    # PlaybackInterface implementation
    
    def get_position(self) -> float:
        """Get current position in seconds"""
        if self._player:
            return self._player.position() / 1000.0
        return 0.0
    
    def set_position(self, seconds: float) -> None:
        """Seek to position"""
        if self._player:
            self._player.setPosition(int(seconds * 1000))
    
    def play(self) -> None:
        """Start playback"""
        if self._player:
            self._player.play()
    
    def pause(self) -> None:
        """Pause playback"""
        if self._player:
            self._player.pause()
    
    def stop(self) -> None:
        """Stop playback"""
        if self._player:
            self._player.stop()
            self._player.setPosition(0)
    
    def is_playing(self) -> bool:
        """Check if playing"""
        if self._player:
            from PyQt6.QtMultimedia import QMediaPlayer
            state = self._player.playbackState()
            return state == QMediaPlayer.PlaybackState.PlayingState
        return False
    
    def get_duration(self) -> float:
        """Get duration in seconds"""
        return self._duration
    
    def cleanup(self):
        """Clean up resources - non-blocking"""
        if self._player:
            # CRITICAL: Disconnect signals FIRST to prevent blocking callbacks
            try:
                # Disconnect all signals to prevent blocking callbacks
                self._player.durationChanged.disconnect()
                # Disconnect any other potential signal connections
                try:
                    self._player.errorOccurred.disconnect()
                except:
                    pass
                try:
                    self._player.mediaStatusChanged.disconnect()
                except:
                    pass
            except Exception as e:
                Log.debug(f"SimpleAudioPlayer: Error disconnecting signals: {e}")
            
            # CRITICAL FIX: Don't call stop() during cleanup - it can block indefinitely
            # Use pause() instead (non-blocking), then clear references
            try:
                from PyQt6.QtMultimedia import QMediaPlayer
                # Pause if playing (non-blocking, unlike stop())
                if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    try:
                        self._player.pause()  # Non-blocking pause
                    except:
                        pass
                # Don't call stop() - it can block if QMediaPlayer is in a bad state
                # Signals are already disconnected above, so no callbacks will fire
                # Qt will clean up the player when references are cleared
            except Exception as e:
                Log.debug(f"SimpleAudioPlayer: Error during cleanup: {e}")
            
            # CRITICAL: Use deleteLater() for async deletion to prevent blocking
            # Setting to None triggers destructor which can block
            # Schedule for async deletion instead of immediate destruction
            # This prevents blocking during close
            try:
                self._player.deleteLater()
                if self._audio_output:
                    self._audio_output.deleteLater()
            except Exception as e:
                Log.debug(f"SimpleAudioPlayer: Error scheduling deletion: {e}")
            
            # Clear references - Qt will delete objects asynchronously
            self._player = None
            self._audio_output = None

