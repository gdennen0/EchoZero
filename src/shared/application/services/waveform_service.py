"""
Waveform Service

Computes and stores waveform envelopes for AudioDataItems.
Waveforms are peak envelopes stored as 1D numpy arrays in a file system cache.

Design:
- One waveform per audio item at a single resolution (DEFAULT_RESOLUTION).
- Deterministic cache paths: {cache_dir}/{audio_id}_r{resolution}.npy
- Metadata is minimal: {resolution, points_count} only.
- Strict validation: non-1D data is rejected, never silently fixed.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.utils.message import Log
from src.utils.paths import get_user_cache_dir

if TYPE_CHECKING:
    from src.shared.domain.entities import AudioDataItem

try:
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    Log.warning("librosa not available - waveform generation disabled")


class WaveformService:
    """
    Service for computing and managing waveform envelopes for AudioDataItems.
    
    Waveforms are peak envelopes (1D float32 arrays) stored in file system cache.
    One waveform per audio item at a single resolution. UI resamples for display.
    """
    
    DEFAULT_RESOLUTION = 50  # Points per second
    
    def __init__(self, cache_dir: Optional[Path] = None, default_resolution: Optional[int] = None):
        """
        Initialize waveform service.
        
        Args:
            cache_dir: Optional cache directory (defaults to user cache dir)
            default_resolution: Optional default resolution (defaults to DEFAULT_RESOLUTION)
        """
        if cache_dir is None:
            cache_dir = get_user_cache_dir()
        
        self._cache_dir = Path(cache_dir) / "waveforms"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_resolution = default_resolution or self.DEFAULT_RESOLUTION
    
    def get_waveform_path(self, audio_item_id: str, resolution: int) -> Path:
        """
        Get deterministic cache path for a waveform file.
        
        Path is: {cache_dir}/{audio_item_id}_r{resolution}.npy
        
        This is the single source of truth for waveform file locations.
        No metadata lookup needed -- the path is deterministic from the audio item ID.
        
        Args:
            audio_item_id: AudioDataItem ID
            resolution: Points per second
            
        Returns:
            Path to waveform cache file
        """
        return self._cache_dir / f"{audio_item_id}_r{resolution}.npy"
    
    def _find_best_waveform_path(self, audio_item_id: str, resolution: Optional[int] = None) -> Optional[Path]:
        """
        Find the waveform file for an audio item using deterministic path.
        
        Args:
            audio_item_id: AudioDataItem ID
            resolution: Target resolution (None = use default)
            
        Returns:
            Path to waveform file, or None if no file exists
        """
        res = resolution if resolution is not None else self._default_resolution
        path = self.get_waveform_path(audio_item_id, res)
        if path.exists():
            return path
        
        # If exact resolution not found and it differs from default, try default
        if res != self._default_resolution:
            default_path = self.get_waveform_path(audio_item_id, self._default_resolution)
            if default_path.exists():
                return default_path
        
        return None
    
    def compute_waveform(
        self,
        audio_item: 'AudioDataItem',  # type: ignore
        resolution_per_second: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute peak envelope waveform for an AudioDataItem.
        
        Uses peak envelope algorithm: for each window, take the max absolute
        amplitude. This is what professional DAWs use for waveform display.
        
        Args:
            audio_item: AudioDataItem to compute waveform for
            resolution_per_second: Target resolution (defaults to DEFAULT_RESOLUTION)
            
        Returns:
            Normalized 1D peak envelope array (shape: (N,), dtype: float32) or None
        """
        if not HAS_AUDIO_LIBS:
            Log.warning("WaveformService: Audio libraries not available")
            return None
        
        if resolution_per_second is None:
            resolution_per_second = self._default_resolution
        
        try:
            # Get audio data
            audio_data = None
            sample_rate = None
            
            # Try in-memory data first
            if hasattr(audio_item, 'get_audio_data'):
                audio_data = audio_item.get_audio_data()
                if audio_data is not None:
                    sample_rate = audio_item.sample_rate
                    # Convert multi-channel to mono for waveform computation
                    if audio_data.ndim == 2:
                        audio_data = np.mean(audio_data, axis=0)
            
            # Fallback: load from file at low sample rate (sufficient for waveform)
            WAVEFORM_SAMPLE_RATE = 8000
            
            if audio_data is None and audio_item.file_path and os.path.exists(audio_item.file_path):
                audio_data, sample_rate = librosa.load(
                    audio_item.file_path,
                    sr=WAVEFORM_SAMPLE_RATE,
                    mono=True
                )
                Log.debug(f"WaveformService: Loaded audio at {WAVEFORM_SAMPLE_RATE}Hz for waveform generation")
            
            if audio_data is None or len(audio_data) == 0:
                Log.warning(f"WaveformService: No audio data available for {audio_item.name}")
                return None
            
            if sample_rate is None:
                sample_rate = WAVEFORM_SAMPLE_RATE
            
            # Get duration
            duration = audio_item.length_ms / 1000.0 if audio_item.length_ms else len(audio_data) / sample_rate
            if duration <= 0:
                Log.warning(f"WaveformService: Zero duration for {audio_item.name}")
                return None
            
            # Calculate target points
            target_points = int(duration * resolution_per_second)
            target_points = max(10, min(target_points, 50000))
            
            # Peak envelope computation
            total_samples = len(audio_data)
            if total_samples <= target_points:
                # Fewer samples than target points -- use absolute values directly
                peaks = np.abs(audio_data).astype(np.float32)
            else:
                samples_per_point = total_samples / target_points
                peaks = np.empty(target_points, dtype=np.float32)
                for i in range(target_points):
                    start = int(i * samples_per_point)
                    end = int((i + 1) * samples_per_point)
                    end = min(end, total_samples)
                    if start < end:
                        peaks[i] = np.max(np.abs(audio_data[start:end]))
                    else:
                        peaks[i] = 0.0
            
            # Normalize to [0, 1]
            max_val = np.max(peaks)
            if max_val > 0:
                peaks = peaks / max_val
            
            Log.debug(
                f"WaveformService: Computed waveform for {audio_item.name} "
                f"({len(peaks)} points, {duration:.2f}s, {resolution_per_second} pts/sec)"
            )
            
            return peaks
            
        except Exception as e:
            Log.error(f"WaveformService: Failed to compute waveform for {audio_item.name}: {e}")
            return None
    
    def store_waveform(
        self,
        audio_item: 'AudioDataItem',  # type: ignore
        waveform_data: np.ndarray,
        resolution: int
    ) -> Optional[str]:
        """
        Store waveform data to file system cache.
        
        Sets minimal metadata on audio_item: {resolution, points_count} only.
        
        Args:
            audio_item: AudioDataItem to store waveform for
            waveform_data: 1D peak envelope array
            resolution: Points per second used
            
        Returns:
            File path to stored waveform, or None if storage fails
        """
        try:
            # Strict validation: must be 1D
            if waveform_data.ndim != 1:
                Log.error(
                    f"WaveformService: Refusing to store non-1D waveform for {audio_item.name} "
                    f"(shape: {waveform_data.shape}). This is raw audio data, not a waveform envelope."
                )
                return None
            
            waveform_file = self.get_waveform_path(audio_item.id, resolution)
            np.save(str(waveform_file), waveform_data)
            
            # Minimal metadata -- no file_path, no resolutions dict
            audio_item.metadata['waveform'] = {
                'resolution': resolution,
                'points_count': len(waveform_data),
            }
            
            Log.debug(f"WaveformService: Stored waveform for {audio_item.name} at r{resolution} ({len(waveform_data)} pts)")
            return str(waveform_file)
            
        except Exception as e:
            Log.error(f"WaveformService: Failed to store waveform for {audio_item.name}: {e}")
            return None
    
    def compute_and_store(
        self,
        audio_item: 'AudioDataItem',  # type: ignore
        resolution_per_second: Optional[int] = None
    ) -> bool:
        """
        Compute and store waveform for an AudioDataItem.
        
        If resolution_per_second is None, uses timeline setting or DEFAULT_RESOLUTION.
        
        Args:
            audio_item: AudioDataItem to process
            resolution_per_second: Target resolution (None = auto)
            
        Returns:
            True if successful, False otherwise
        """
        if resolution_per_second is None:
            try:
                from ui.qt_gui.widgets.timeline.settings.storage import get_timeline_settings_manager
                settings_mgr = get_timeline_settings_manager()
                if settings_mgr:
                    resolution_per_second = settings_mgr.waveform_resolution
            except Exception:
                pass
            if resolution_per_second is None:
                resolution_per_second = self._default_resolution
        
        # Check if waveform already exists at this resolution
        if self.get_waveform_path(audio_item.id, resolution_per_second).exists():
            Log.debug(f"WaveformService: Waveform already exists at r{resolution_per_second} for {audio_item.name}")
            return True
        
        waveform_data = self.compute_waveform(audio_item, resolution_per_second)
        if waveform_data is None:
            return False
        
        file_path = self.store_waveform(audio_item, waveform_data, resolution_per_second)
        if file_path is None:
            return False
        
        Log.info(f"WaveformService: Generated waveform at r{resolution_per_second} for {audio_item.name}")
        return True
    
    def _validate_waveform_data(self, waveform_data: np.ndarray, audio_item: 'AudioDataItem', file_path: str) -> Optional[np.ndarray]:
        """
        Validate loaded waveform data. Returns data if 1D, None otherwise.
        
        Strict: non-1D data is rejected with Log.error. No conversion.
        """
        if waveform_data.ndim == 1:
            return waveform_data
        
        Log.error(
            f"WaveformService: Corrupted waveform for {audio_item.name} "
            f"(shape: {waveform_data.shape}, expected 1D). "
            f"Rejecting. Delete the file and re-run the pipeline to regenerate."
        )
        return None
    
    def load_waveform(
        self, 
        audio_item: 'AudioDataItem',  # type: ignore
        resolution: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Load waveform from cache using deterministic path.
        
        Args:
            audio_item: AudioDataItem to load waveform for
            resolution: Specific resolution (None = find best available)
            
        Returns:
            Waveform data array or None if not found
        """
        file_path = self._find_best_waveform_path(audio_item.id, resolution)
        if not file_path:
            return None
        
        try:
            waveform_data = np.load(str(file_path))
            waveform_data = self._validate_waveform_data(waveform_data, audio_item, str(file_path))
            if waveform_data is not None:
                Log.debug(f"WaveformService: Loaded waveform for {audio_item.name} from {file_path}")
            return waveform_data
        except Exception as e:
            Log.warning(f"WaveformService: Failed to load waveform from {file_path}: {e}")
            return None
    
    def load_waveform_mmap(
        self,
        audio_item: 'AudioDataItem',  # type: ignore
        resolution: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Load waveform using memory-mapped access.
        
        If resolution is None, uses timeline setting or best available.
        
        Args:
            audio_item: AudioDataItem to load waveform for
            resolution: Specific resolution (None = auto)
            
        Returns:
            Memory-mapped waveform data array or None
        """
        if resolution is None:
            try:
                from ui.qt_gui.widgets.timeline.settings.storage import get_timeline_settings_manager
                settings_mgr = get_timeline_settings_manager()
                if settings_mgr:
                    resolution = settings_mgr.waveform_resolution
            except Exception:
                pass
        
        file_path = self._find_best_waveform_path(audio_item.id, resolution)
        if not file_path:
            return None
        
        try:
            waveform_data = np.load(str(file_path), mmap_mode='r')
            validated = self._validate_waveform_data(waveform_data, audio_item, str(file_path))
            if validated is not None:
                Log.debug(f"WaveformService: Loaded waveform (mmap) for {audio_item.name} from {file_path}")
            return validated
        except Exception as e:
            Log.warning(f"WaveformService: Failed to load waveform (mmap) from {file_path}: {e}")
            return None
    
    def has_waveform(
        self, 
        audio_item: 'AudioDataItem',  # type: ignore
        resolution: Optional[int] = None
    ) -> bool:
        """
        Check if AudioDataItem has stored waveform using deterministic path.
        
        Args:
            audio_item: AudioDataItem to check
            resolution: Specific resolution (None = check default)
            
        Returns:
            True if waveform file exists
        """
        return self._find_best_waveform_path(audio_item.id, resolution) is not None
    
    def get_available_resolutions(self, audio_item: 'AudioDataItem') -> list[int]:  # type: ignore
        """
        Get list of available resolutions for an AudioDataItem.
        
        Args:
            audio_item: AudioDataItem to check
            
        Returns:
            Sorted list of available resolutions
        """
        available = []
        # Check default resolution
        if self.get_waveform_path(audio_item.id, self._default_resolution).exists():
            available.append(self._default_resolution)
        return sorted(available)
    
    def get_waveform_slice(
        self,
        audio_item: 'AudioDataItem',  # type: ignore
        start_time: float,
        end_time: float,
        target_resolution: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Extract time slice from stored base waveform.
        
        Args:
            audio_item: AudioDataItem
            start_time: Start time in seconds
            end_time: End time in seconds
            target_resolution: Optional target points for display resampling
            
        Returns:
            Waveform slice array or None
        """
        duration = audio_item.length_ms / 1000.0 if audio_item.length_ms else 0.0
        if duration <= 0:
            return None
        
        full_waveform = self.load_waveform_mmap(audio_item)
        if full_waveform is None:
            return None
        
        waveform_len = len(full_waveform)
        
        # Convert time to indices
        start_idx = int((start_time / duration) * waveform_len)
        end_idx = int((end_time / duration) * waveform_len)
        
        start_idx = max(0, min(start_idx, waveform_len - 1))
        end_idx = max(start_idx + 1, min(end_idx, waveform_len))
        
        slice_data = np.array(full_waveform[start_idx:end_idx], dtype=np.float32)
        
        if len(slice_data) == 0:
            if waveform_len > 0:
                slice_data = np.array([
                    float(full_waveform[start_idx]),
                    float(full_waveform[min(end_idx, waveform_len - 1)])
                ], dtype=np.float32)
            else:
                return None
        
        # Resample for display if target_resolution specified
        if target_resolution and target_resolution > 0:
            target_points = max(2, min(10000, target_resolution))
            current_len = len(slice_data)
            
            if current_len == target_points:
                return slice_data
            elif current_len == 1:
                slice_data = np.full(target_points, slice_data[0], dtype=np.float32)
            elif current_len > target_points:
                # Downsample via indexing
                step = current_len / target_points
                indices = (np.arange(target_points) * step).astype(np.int32)
                indices = np.clip(indices, 0, current_len - 1)
                slice_data = slice_data[indices]
            else:
                # Upsample via interpolation
                indices = np.linspace(0, current_len - 1, target_points, dtype=np.float32)
                slice_data = np.interp(indices, np.arange(current_len, dtype=np.float32), slice_data).astype(np.float32)
        elif len(slice_data) < 2:
            if len(slice_data) == 1:
                slice_data = np.array([slice_data[0], slice_data[0]], dtype=np.float32)
            else:
                slice_data = np.array([0.0, 0.0], dtype=np.float32)
        
        # Safety: cap at 10K points
        if len(slice_data) > 10000:
            step = len(slice_data) / 10000
            indices = (np.arange(10000) * step).astype(np.int32)
            indices = np.clip(indices, 0, len(slice_data) - 1)
            slice_data = slice_data[indices].astype(np.float32)
        
        return slice_data
    
    def get_waveform_resolution(self, audio_item: 'AudioDataItem') -> Optional[int]:  # type: ignore
        """
        Get the waveform resolution for an AudioDataItem.
        
        Args:
            audio_item: AudioDataItem
            
        Returns:
            Resolution in points per second, or None
        """
        if self.get_waveform_path(audio_item.id, self._default_resolution).exists():
            return self._default_resolution
        return None
    
    def delete_waveform(self, audio_item: 'AudioDataItem') -> bool:  # type: ignore
        """
        Delete stored waveform file(s) for an AudioDataItem.
        
        Args:
            audio_item: AudioDataItem to delete waveform for
            
        Returns:
            True if cleaned up
        """
        deleted_count = 0
        
        try:
            # Delete at default resolution
            path = self.get_waveform_path(audio_item.id, self._default_resolution)
            if path.exists():
                try:
                    path.unlink()
                    deleted_count += 1
                except Exception as e:
                    Log.warning(f"WaveformService: Failed to delete waveform file {path}: {e}")
            
            # Clear metadata
            if 'waveform' in audio_item.metadata:
                del audio_item.metadata['waveform']
            
            if deleted_count > 0:
                Log.debug(f"WaveformService: Deleted waveform for {audio_item.name}")
            
            return True
                
        except Exception as e:
            Log.warning(f"WaveformService: Failed to delete waveform for {audio_item.name}: {e}")
            return False
    
    def regenerate_all_waveforms_at_resolution(self, resolution_per_second: int) -> int:
        """
        Regenerate all waveforms at a new resolution.
        
        Deletes old waveform files and regenerates at the specified resolution.
        
        Args:
            resolution_per_second: Target resolution in points per second
            
        Returns:
            Number of waveforms regenerated
        """
        try:
            from src.shared.domain.entities import AudioDataItem
            from src.shared.application.services.data_state_service import get_data_state_service
        except ImportError:
            Log.warning("WaveformService: Cannot regenerate waveforms - required modules not available")
            return 0
        
        Log.info(f"WaveformService: Regenerating all waveforms at r{resolution_per_second}")
        
        data_service = get_data_state_service()
        if not data_service:
            Log.warning("WaveformService: Cannot regenerate waveforms - data state service not available")
            return 0
        
        audio_items = []
        try:
            for item in data_service.get_all_items():
                if isinstance(item, AudioDataItem):
                    audio_items.append(item)
        except Exception as e:
            Log.warning(f"WaveformService: Failed to get audio items: {e}")
            return 0
        
        if not audio_items:
            Log.info("WaveformService: No audio items found to regenerate")
            return 0
        
        success_count = 0
        for audio_item in audio_items:
            try:
                self.delete_waveform(audio_item)
                if self.compute_and_store(audio_item, resolution_per_second):
                    success_count += 1
            except Exception as e:
                Log.warning(f"WaveformService: Failed to regenerate waveform for {audio_item.name}: {e}")
        
        Log.info(f"WaveformService: Regenerated {success_count}/{len(audio_items)} waveforms at r{resolution_per_second}")
        return success_count
    
    def clear_waveforms_for_project(self, project_id: str) -> int:
        """
        Clear all waveforms for audio items in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of waveforms cleared
        """
        try:
            from src.shared.domain.entities import AudioDataItem
            from src.shared.application.services.data_state_service import get_data_state_service
        except ImportError:
            Log.warning("WaveformService: Cannot clear waveforms - required modules not available")
            return 0
        
        data_service = get_data_state_service()
        if not data_service:
            return 0
        
        audio_items = []
        try:
            for item in data_service.get_all_items():
                if isinstance(item, AudioDataItem):
                    audio_items.append(item)
        except Exception as e:
            Log.warning(f"WaveformService: Failed to get audio items: {e}")
            return 0
        
        cleared_count = 0
        for audio_item in audio_items:
            try:
                if self.delete_waveform(audio_item):
                    cleared_count += 1
            except Exception as e:
                Log.warning(f"WaveformService: Failed to clear waveform for {audio_item.name}: {e}")
        
        Log.info(f"WaveformService: Cleared {cleared_count}/{len(audio_items)} waveforms")
        return cleared_count


# Global instance
_waveform_service_instance: Optional[WaveformService] = None


def get_waveform_service() -> WaveformService:
    """Get WaveformService singleton instance."""
    global _waveform_service_instance
    if _waveform_service_instance is None:
        _waveform_service_instance = WaveformService()
    return _waveform_service_instance


def set_waveform_service(service: WaveformService) -> None:
    """Set WaveformService instance (for testing)."""
    global _waveform_service_instance
    _waveform_service_instance = service
