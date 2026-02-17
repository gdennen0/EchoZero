"""
Clip End Detector Module

Dedicated module for detecting clip end times from energy data.
Separates detection logic from duration constraints.
"""
from typing import Optional, Tuple
import numpy as np
from src.utils.message import Log
from src.application.settings.detect_onsets_settings import DetectOnsetsBlockSettings


class ClipEndDetector:
    """
    Detects clip end times from energy data.
    
    This module handles Phase 2 of onset detection: determining where detected
    events end. It is separate from Phase 1 (onset detection) which finds where
    events start.
    
    Responsibility: Given an onset time, determine when the sound stops.
    
    Supports two detection modes:
    - Peak Decay mode: Cuts immediately when energy drops to % of peak
    - Sustained Silence mode: Waits for multiple consecutive frames below threshold
    """
    
    def detect_clip_end(
        self,
        onset_time: float,
        energy_frames: np.ndarray,
        energy_times: np.ndarray,
        onset_frame_idx: int,
        next_onset_time: Optional[float],
        settings: DetectOnsetsBlockSettings,
        sample_rate: int,
        audio_duration: float,
        default_end_time: float
    ) -> Tuple[float, dict]:
        """
        Detect clip end time from energy data.
        
        Args:
            onset_time: Onset time in seconds
            energy_frames: RMS energy frames array
            energy_times: Time array corresponding to energy_frames
            onset_frame_idx: Frame index of the onset
            next_onset_time: Next onset time (if exists), None otherwise
            settings: DetectOnsetsBlockSettings instance
            sample_rate: Audio sample rate
            audio_duration: Total audio duration in seconds
            default_end_time: Default end time (next onset or audio end)
            
        Returns:
            Tuple of (detected_end_time, metadata_dict)
            metadata_dict contains: peak_energy, peak_energy_time, threshold_energy, 
            threshold_type, clip_scope_energy_for_avg
        """
        # Initialize metadata
        metadata = {
            "peak_energy": 0.0,
            "peak_energy_time": onset_time,
            "threshold_energy": 0.0,
            "threshold_type": "absolute",
            "clip_scope_energy_for_avg": None
        }
        
        # Default end time (will be used if detection fails)
        end_time = default_end_time
        decay_found = False
        
        # Extract settings
        early_cut_mode = settings.early_cut_mode
        peak_decay_ratio = settings.peak_decay_ratio
        decay_detection_method = settings.decay_detection_method
        energy_decay_threshold = settings.energy_decay_threshold
        adaptive_threshold_enabled = settings.adaptive_threshold_enabled
        adaptive_threshold_factor = settings.adaptive_threshold_factor
        onset_lookahead_time = settings.onset_lookahead_time
        energy_rise_threshold = settings.energy_rise_threshold
        min_separation_time = settings.min_separation_time
        energy_hop_length = settings.energy_hop_length
        
        # Extend search region to include lookahead for next onset detection
        lookahead_end_time = min(
            default_end_time + onset_lookahead_time,
            audio_duration if audio_duration > 0 else default_end_time + 0.5
        )
        end_frame_idx = min(
            np.searchsorted(energy_times, lookahead_end_time, side='right'),
            len(energy_frames)
        )
        
        if onset_frame_idx >= len(energy_frames):
            return end_time, metadata
        
        # Get energy values for this clip region (with lookahead)
        clip_energy = energy_frames[onset_frame_idx:end_frame_idx]
        clip_times = energy_times[onset_frame_idx:end_frame_idx]
        
        if len(clip_energy) == 0:
            return end_time, metadata
        
        Log.debug(
            f"ClipEndDetector: Processing onset at {onset_time:.3f}s: "
            f"onset_frame_idx={onset_frame_idx}, end_frame_idx={end_frame_idx}, "
            f"clip_energy_length={len(clip_energy)}, default_end_time={default_end_time:.3f}s"
        )
        
        # Determine max search index first (before next onset)
        temp_max_search_idx = len(clip_energy)
        if next_onset_time is not None:
            next_onset_clip_idx = np.searchsorted(
                clip_times, next_onset_time - min_separation_time, side='left'
            )
            temp_max_search_idx = min(temp_max_search_idx, next_onset_clip_idx)
        
        # Define clip scope: actual clip region, not extended lookahead
        decay_window_frames = int(0.2 * sample_rate / energy_hop_length)  # 200ms decay window
        clip_scope_end = min(temp_max_search_idx, len(clip_energy))
        
        # Calculate peak energy from clip scope only
        clip_scope_energy = clip_energy[:clip_scope_end]
        peak_energy = np.max(clip_scope_energy) if len(clip_scope_energy) > 0 else np.max(clip_energy)
        
        # Find peak position within clip scope
        if len(clip_scope_energy) > 0:
            peak_idx = np.argmax(clip_scope_energy)
        else:
            peak_idx = np.argmax(clip_energy)
            peak_idx = min(peak_idx, temp_max_search_idx - 1) if temp_max_search_idx > 0 else 0
        
        # Convert numpy array to Python list for JSON serialization
        # This is used internally for avg_energy calculation, not stored in final event metadata
        metadata["clip_scope_energy_for_avg"] = clip_scope_energy.tolist() if len(clip_scope_energy) > 0 else None
        metadata["peak_energy"] = float(peak_energy)
        
        if peak_energy <= 0:
            return end_time, metadata
        
        # Calculate threshold
        if adaptive_threshold_enabled:
            # Get baseline energy before onset (background level)
            baseline_start = max(0, onset_frame_idx - 10)  # 10 frames before
            baseline_end = onset_frame_idx
            if baseline_start < baseline_end:
                baseline_energy = np.mean(energy_frames[baseline_start:baseline_end])
            else:
                baseline_energy = 0.0
            
            # Calculate adaptive threshold
            peak_threshold = peak_energy * energy_decay_threshold
            baseline_threshold = baseline_energy + (peak_energy - baseline_energy) * adaptive_threshold_factor
            
            # Use the higher of the two thresholds (more conservative)
            threshold_energy = max(peak_threshold, baseline_threshold)
            metadata["threshold_type"] = "adaptive"
            
            Log.debug(
                f"ClipEndDetector: Adaptive threshold for onset at {onset_time:.3f}s: "
                f"peak={peak_energy:.4f}, baseline={baseline_energy:.4f}, "
                f"threshold={threshold_energy:.4f}, clip_scope_end={clip_scope_end}"
            )
        else:
            # Standard threshold (peak-relative only)
            threshold_energy = peak_energy * energy_decay_threshold
            metadata["threshold_type"] = "absolute"
            
            Log.debug(
                f"ClipEndDetector: Standard threshold for onset at {onset_time:.3f}s: "
                f"peak={peak_energy:.4f}, threshold_factor={energy_decay_threshold}, "
                f"threshold={threshold_energy:.4f}, clip_scope_end={clip_scope_end}"
            )
        
        metadata["threshold_energy"] = float(threshold_energy)
        
        # Calculate peak time
        peak_time = clip_times[peak_idx] if peak_idx < len(clip_times) else onset_time
        metadata["peak_energy_time"] = float(peak_time)
        
        Log.debug(
            f"ClipEndDetector: Peak energy found at index {peak_idx} "
            f"(time={peak_time:.3f}s, energy={peak_energy:.4f})"
        )
        
        # Detect next onset via energy rise (optional enhancement)
        next_onset_detected = False
        if next_onset_time is not None and next_onset_time <= onset_time + onset_lookahead_time:
            next_onset_frame_idx = np.searchsorted(energy_times, next_onset_time, side='left')
            if next_onset_frame_idx < len(energy_frames):
                next_onset_clip_idx = next_onset_frame_idx - onset_frame_idx
                
                if 0 <= next_onset_clip_idx < len(clip_energy):
                    # Check energy rise before next onset
                    check_start = max(peak_idx + 1, next_onset_clip_idx - 5)
                    check_end = min(next_onset_clip_idx + 2, len(clip_energy))
                    
                    if check_start < check_end and check_start >= 0:
                        energy_before_onset = np.mean(clip_energy[check_start:check_end])
                        if peak_idx + 3 < len(clip_energy):
                            energy_at_decay = np.mean(clip_energy[peak_idx:peak_idx+3])
                            if energy_before_onset > energy_at_decay * energy_rise_threshold:
                                next_onset_detected = True
                                Log.debug(
                                    f"ClipEndDetector: Next onset detected at {next_onset_time:.3f}s "
                                    f"(energy rise: {energy_at_decay:.4f} -> {energy_before_onset:.4f})"
                                )
        
        # Determine maximum search index (ALWAYS limit to before next onset)
        max_search_idx = len(clip_energy)
        if next_onset_time is not None:
            next_onset_clip_idx = np.searchsorted(
                clip_times, next_onset_time - min_separation_time, side='left'
            )
            max_search_idx = min(max_search_idx, next_onset_clip_idx)
            Log.debug(
                f"ClipEndDetector: Limiting search to before next onset at {next_onset_time:.3f}s "
                f"(max_search_idx={max_search_idx}, cut at {next_onset_time - min_separation_time:.3f}s)"
            )
        
        # Choose decay detection strategy based on early_cut_mode
        if early_cut_mode:
            # Peak decay mode: Multiple detection strategies available
            # Uses peak_decay_ratio (energy_decay_threshold is ignored in this mode)
            peak_decay_threshold = peak_energy * peak_decay_ratio
            
            # Start searching from after the peak
            search_start_idx = peak_idx + 1
            
            # Only search if we have a valid range
            if search_start_idx < max_search_idx:
                Log.debug(
                    f"ClipEndDetector: Early cut mode - searching for peak decay from "
                    f"search_start_idx={search_start_idx} (peak_idx={peak_idx}) to max_search_idx={max_search_idx}, "
                    f"peak_energy={peak_energy:.4f}, peak_decay_ratio={peak_decay_ratio}, "
                    f"peak_decay_threshold={peak_decay_threshold:.4f}, method={decay_detection_method}"
                )
                
                end_time, decay_found = self._detect_early_cut(
                    clip_energy=clip_energy,
                    clip_times=clip_times,
                    search_start_idx=search_start_idx,
                    max_search_idx=max_search_idx,
                    peak_decay_threshold=peak_decay_threshold,
                    peak_energy=peak_energy,
                    peak_decay_ratio=peak_decay_ratio,
                    decay_detection_method=decay_detection_method,
                    sample_rate=sample_rate,
                    energy_hop_length=energy_hop_length
                )
        else:
            # Sustained silence mode: Wait for multiple consecutive frames below threshold
            relative_threshold = peak_energy * 0.25
            effective_threshold = min(threshold_energy, relative_threshold)
            
            silence_frames_required = max(3, int(0.01 * sample_rate / energy_hop_length))
            
            Log.debug(
                f"ClipEndDetector: Sustained silence mode - searching from peak_idx={peak_idx} "
                f"to max_search_idx={max_search_idx}, "
                f"silence_frames_required={silence_frames_required}, "
                f"absolute_threshold={threshold_energy:.4f}, relative_threshold={relative_threshold:.4f}, "
                f"effective_threshold={effective_threshold:.4f}"
            )
            
            end_time, decay_found = self._detect_sustained_silence(
                clip_energy=clip_energy,
                clip_times=clip_times,
                peak_idx=peak_idx,
                max_search_idx=max_search_idx,
                effective_threshold=effective_threshold,
                silence_frames_required=silence_frames_required
            )
        
        # If next onset was detected via energy rise, ensure we cut before it
        if next_onset_detected and next_onset_time is not None:
            cut_time = next_onset_time - min_separation_time
            if cut_time > onset_time:
                # Use the earlier of: detection result or next onset cut
                if not decay_found or cut_time < end_time:
                    end_time = cut_time
                    decay_found = True
                    Log.debug(
                        f"ClipEndDetector: Cutting before next onset at {end_time:.3f}s "
                        f"(min_separation={min_separation_time:.3f}s before {next_onset_time:.3f}s)"
                    )
        
        # Fallback: Always cut before next onset if it exists (even if energy rise detection failed)
        if not decay_found and next_onset_time is not None:
            cut_time = next_onset_time - min_separation_time
            if cut_time > onset_time:
                end_time = cut_time
                decay_found = True
                Log.debug(
                    f"ClipEndDetector: Fallback cut before next onset at {end_time:.3f}s "
                    f"(min_separation={min_separation_time:.3f}s before {next_onset_time:.3f}s, "
                    f"energy rise detection failed)"
                )
        
        if decay_found:
            Log.debug(
                f"ClipEndDetector: Onset at {onset_time:.3f}s: "
                f"decay detected at {end_time:.3f}s, duration={end_time - onset_time:.3f}s "
                f"(peak={peak_energy:.4f}, threshold={threshold_energy:.4f})"
            )
        else:
            Log.debug(
                f"ClipEndDetector: Onset at {onset_time:.3f}s: "
                f"no decay found, using default end time {end_time:.3f}s, "
                f"duration={end_time - onset_time:.3f}s "
                f"(peak={peak_energy:.4f}, threshold={threshold_energy:.4f})"
            )
        
        return end_time, metadata
    
    def _detect_early_cut(
        self,
        clip_energy: np.ndarray,
        clip_times: np.ndarray,
        search_start_idx: int,
        max_search_idx: int,
        peak_decay_threshold: float,
        peak_energy: float,
        peak_decay_ratio: float,
        decay_detection_method: str,
        sample_rate: int,
        energy_hop_length: int
    ) -> Tuple[float, bool]:
        """
        Detect clip end using early cut mode (peak decay detection).
        
        Returns:
            Tuple of (end_time, decay_found)
        """
        decay_found = False
        end_time = clip_times[-1] if len(clip_times) > 0 else 0.0
        
        if decay_detection_method == "all":
            # Try all methods and use the EARLIEST cut found
            candidate_cuts = []
            
            # Method 1: Simple threshold
            for j in range(search_start_idx, max_search_idx):
                if clip_energy[j] < peak_decay_threshold:
                    candidate_cuts.append((clip_times[j], "threshold"))
                    break
            
            # Method 2: Rate of change
            if max_search_idx - search_start_idx >= 2:
                # Scale min_drop_rate with peak_decay_ratio for consistent sensitivity
                # Lower ratio (tighter cuts) = lower drop rate requirement
                min_drop_rate = peak_energy * peak_decay_ratio * 0.1
                for j in range(search_start_idx, max_search_idx - 1):
                    energy_change = clip_energy[j + 1] - clip_energy[j]
                    if energy_change < -min_drop_rate and clip_energy[j] < peak_decay_threshold:
                        candidate_cuts.append((clip_times[j], "rate_of_change"))
                        break
            
            # Method 3: Slope-based
            window_size = min(3, max_search_idx - search_start_idx)
            if window_size >= 2:
                # Range adjusted: need j + window_size < max_search_idx, so j < max_search_idx - window_size
                for j in range(search_start_idx, max_search_idx - window_size):
                    # Use energy at j and j+window_size (both after peak) to calculate slope
                    energy_start = clip_energy[j]
                    energy_end = clip_energy[j + window_size]
                    slope = (energy_end - energy_start) / window_size
                    if slope < 0 and clip_energy[j] < peak_decay_threshold:
                        candidate_cuts.append((clip_times[j], "slope"))
                        break
            
            # Method 4: Confirmed threshold
            confirmation_frames = max(2, min(3, int(0.005 * sample_rate / energy_hop_length)))
            consecutive_below = 0
            for j in range(search_start_idx, max_search_idx):
                if clip_energy[j] < peak_decay_threshold:
                    consecutive_below += 1
                    if consecutive_below >= confirmation_frames:
                        candidate_cuts.append((clip_times[j - confirmation_frames + 1], "confirmed_threshold"))
                        break
                else:
                    consecutive_below = 0
            
            # Use the EARLIEST cut found by any method
            if candidate_cuts:
                candidate_cuts.sort(key=lambda x: x[0])
                earliest_cut_time, method_used = candidate_cuts[0]
                decay_found = True
                end_time = earliest_cut_time
                Log.debug(
                    f"ClipEndDetector: Early cut (all methods) - earliest cut at {end_time:.3f}s "
                    f"(method={method_used}, tried {len(candidate_cuts)} methods, "
                    f"threshold={peak_decay_threshold:.4f})"
                )
        
        elif decay_detection_method == "threshold":
            # Simple threshold: Cut on first frame below threshold
            for j in range(search_start_idx, max_search_idx):
                if clip_energy[j] < peak_decay_threshold:
                    decay_found = True
                    end_time = clip_times[j]
                    Log.debug(
                        f"ClipEndDetector: Early cut (threshold) - decay detected at {end_time:.3f}s "
                        f"(energy={clip_energy[j]:.4f} < threshold={peak_decay_threshold:.4f})"
                    )
                    break
        
        elif decay_detection_method == "rate_of_change":
            # Rate of change: Detect when energy is decreasing rapidly
            # Scale min_drop_rate with peak_decay_ratio for consistent sensitivity
            # Lower ratio (tighter cuts) = lower drop rate requirement
            min_drop_rate = peak_energy * peak_decay_ratio * 0.1
            if max_search_idx - search_start_idx >= 2:
                for j in range(search_start_idx, max_search_idx - 1):
                    energy_change = clip_energy[j + 1] - clip_energy[j]
                    if energy_change < -min_drop_rate and clip_energy[j] < peak_decay_threshold:
                        decay_found = True
                        end_time = clip_times[j]
                        Log.debug(
                            f"ClipEndDetector: Early cut (rate_of_change) - decay detected at {end_time:.3f}s "
                            f"(energy={clip_energy[j]:.4f}, drop_rate={energy_change:.4f}, "
                            f"min_drop_rate={min_drop_rate:.4f}, threshold={peak_decay_threshold:.4f})"
                        )
                        break
        
        elif decay_detection_method == "slope":
            # Slope-based: Detect consistent negative slope after peak
            # Calculate slope using only energy AFTER the peak (not including peak)
            window_size = min(3, max_search_idx - search_start_idx)
            if window_size >= 2:
                # Range adjusted: need j + window_size < max_search_idx, so j < max_search_idx - window_size
                for j in range(search_start_idx, max_search_idx - window_size):
                    # Use energy at j and j+window_size (both after peak) to calculate slope
                    energy_start = clip_energy[j]
                    energy_end = clip_energy[j + window_size]
                    slope = (energy_end - energy_start) / window_size
                    if slope < 0 and clip_energy[j] < peak_decay_threshold:
                        decay_found = True
                        end_time = clip_times[j]
                        Log.debug(
                            f"ClipEndDetector: Early cut (slope) - decay detected at {end_time:.3f}s "
                            f"(energy={clip_energy[j]:.4f}, slope={slope:.4f}, threshold={peak_decay_threshold:.4f})"
                        )
                        break
            else:
                # Fallback to threshold if window too small
                for j in range(search_start_idx, max_search_idx):
                    if clip_energy[j] < peak_decay_threshold:
                        decay_found = True
                        end_time = clip_times[j]
                        break
        
        elif decay_detection_method == "confirmed_threshold":
            # Confirmed threshold: Require 2-3 consecutive frames below threshold
            confirmation_frames = max(2, min(3, int(0.005 * sample_rate / energy_hop_length)))
            consecutive_below = 0
            for j in range(search_start_idx, max_search_idx):
                if clip_energy[j] < peak_decay_threshold:
                    consecutive_below += 1
                    if consecutive_below >= confirmation_frames:
                        decay_found = True
                        end_time = clip_times[j - confirmation_frames + 1]
                        Log.debug(
                            f"ClipEndDetector: Early cut (confirmed_threshold) - decay detected at {end_time:.3f}s "
                            f"(energy={clip_energy[j]:.4f}, {consecutive_below} consecutive frames below threshold={peak_decay_threshold:.4f})"
                        )
                        break
                else:
                    consecutive_below = 0
        
        else:
            # Unknown method, fallback to threshold
            Log.warning(f"ClipEndDetector: Unknown decay_detection_method '{decay_detection_method}', falling back to 'threshold'")
            for j in range(search_start_idx, max_search_idx):
                if clip_energy[j] < peak_decay_threshold:
                    decay_found = True
                    end_time = clip_times[j]
                    break
        
        return end_time, decay_found
    
    def _detect_sustained_silence(
        self,
        clip_energy: np.ndarray,
        clip_times: np.ndarray,
        peak_idx: int,
        max_search_idx: int,
        effective_threshold: float,
        silence_frames_required: int
    ) -> Tuple[float, bool]:
        """
        Detect clip end using sustained silence mode.
        
        Returns:
            Tuple of (end_time, decay_found)
        """
        decay_found = False
        end_time = clip_times[-1] if len(clip_times) > 0 else 0.0
        
        silence_start_idx = None
        consecutive_silence = 0
        
        for j in range(peak_idx, max_search_idx):
            if clip_energy[j] < effective_threshold:
                # Energy is below threshold
                if silence_start_idx is None:
                    silence_start_idx = j
                consecutive_silence += 1
                
                # Check if we've found sustained silence
                if consecutive_silence >= silence_frames_required:
                    # Found sustained silence - cut at the start of silence
                    decay_found = True
                    end_time = clip_times[silence_start_idx]
                    Log.debug(
                        f"ClipEndDetector: Found sustained silence at {end_time:.3f}s "
                        f"(start_idx={silence_start_idx}, consecutive_frames={consecutive_silence})"
                    )
                    break
            else:
                # Energy rose again - reset silence counter
                if consecutive_silence > 0:
                    Log.debug(
                        f"ClipEndDetector: Silence interrupted at j={j} "
                        f"(energy={clip_energy[j]:.4f} >= effective_threshold={effective_threshold:.4f})"
                    )
                silence_start_idx = None
                consecutive_silence = 0
        
        return end_time, decay_found










