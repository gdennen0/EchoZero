"""
Audio Filter Block Processor

Applies audio filters to incoming audio data using scipy signal processing.

Supported filter types:
- Low-pass: Remove frequencies above cutoff
- High-pass: Remove frequencies below cutoff
- Band-pass: Keep only a frequency range
- Band-stop (Notch): Remove a frequency range
- Low-shelf: Boost/cut frequencies below cutoff
- High-shelf: Boost/cut frequencies above cutoff
- Peak (Bell): Boost/cut around a center frequency

All filters use second-order sections (SOS) for numerical stability.
"""

import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem, DataItem, EventDataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Available filter types with descriptions
FILTER_TYPES = {
    "lowpass": {
        "name": "Low-Pass",
        "description": "Remove frequencies above the cutoff frequency",
        "uses_cutoff": True,
        "uses_cutoff_high": False,
        "uses_gain": False,
        "uses_q": False,
    },
    "highpass": {
        "name": "High-Pass",
        "description": "Remove frequencies below the cutoff frequency",
        "uses_cutoff": True,
        "uses_cutoff_high": False,
        "uses_gain": False,
        "uses_q": False,
    },
    "bandpass": {
        "name": "Band-Pass",
        "description": "Keep only frequencies within a range",
        "uses_cutoff": True,
        "uses_cutoff_high": True,
        "uses_gain": False,
        "uses_q": False,
    },
    "bandstop": {
        "name": "Band-Stop (Notch)",
        "description": "Remove frequencies within a range",
        "uses_cutoff": True,
        "uses_cutoff_high": True,
        "uses_gain": False,
        "uses_q": False,
    },
    "lowshelf": {
        "name": "Low-Shelf",
        "description": "Boost or cut frequencies below the cutoff",
        "uses_cutoff": True,
        "uses_cutoff_high": False,
        "uses_gain": True,
        "uses_q": True,
    },
    "highshelf": {
        "name": "High-Shelf",
        "description": "Boost or cut frequencies above the cutoff",
        "uses_cutoff": True,
        "uses_cutoff_high": False,
        "uses_gain": True,
        "uses_q": True,
    },
    "peak": {
        "name": "Peak (Bell)",
        "description": "Boost or cut around a center frequency",
        "uses_cutoff": True,
        "uses_cutoff_high": False,
        "uses_gain": True,
        "uses_q": True,
    },
}


def _design_biquad_shelf_peak(
    filter_type: str,
    freq: float,
    sample_rate: int,
    gain_db: float = 0.0,
    q_factor: float = 0.707,
) -> np.ndarray:
    """
    Design a biquad filter for shelf and peak types.

    Returns second-order sections (SOS) array for use with sosfilt.

    Args:
        filter_type: "lowshelf", "highshelf", or "peak"
        freq: Center/cutoff frequency in Hz
        sample_rate: Audio sample rate in Hz
        gain_db: Gain in decibels
        q_factor: Q factor (bandwidth control)

    Returns:
        SOS array (shape: 1x6)
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q_factor)

    if filter_type == "lowshelf":
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "highshelf":
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peak":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    else:
        raise ValueError(f"Unknown biquad filter type: {filter_type}")

    # Normalize coefficients
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return sos


def apply_audio_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    filter_type: str,
    cutoff_freq: float,
    cutoff_freq_high: float = 8000.0,
    order: int = 4,
    gain_db: float = 0.0,
    q_factor: float = 0.707,
) -> np.ndarray:
    """
    Apply an audio filter to the given audio data.

    Args:
        audio_data: Audio samples as numpy array (1D mono or 2D multi-channel)
        sample_rate: Sample rate in Hz
        filter_type: One of the FILTER_TYPES keys
        cutoff_freq: Primary cutoff/center frequency in Hz
        cutoff_freq_high: Upper cutoff for bandpass/bandstop filters in Hz
        order: Filter order for butterworth filters (1-8)
        gain_db: Gain in dB for shelf/peak filters (-24 to +24)
        q_factor: Q factor for shelf/peak filters (0.1 to 10.0)

    Returns:
        Filtered audio data as numpy array (same shape as input)
    """
    from scipy.signal import butter, sosfilt

    nyquist = sample_rate / 2.0

    # Clamp frequencies to valid range
    cutoff_freq = max(20.0, min(cutoff_freq, nyquist - 1.0))

    if filter_type in ("lowpass", "highpass"):
        wn = cutoff_freq / nyquist
        wn = max(0.001, min(wn, 0.999))
        btype = "low" if filter_type == "lowpass" else "high"
        sos = butter(order, wn, btype=btype, output="sos")

    elif filter_type in ("bandpass", "bandstop"):
        cutoff_freq_high = max(cutoff_freq + 1.0, min(cutoff_freq_high, nyquist - 1.0))
        wn_low = cutoff_freq / nyquist
        wn_high = cutoff_freq_high / nyquist
        wn_low = max(0.001, min(wn_low, 0.998))
        wn_high = max(wn_low + 0.001, min(wn_high, 0.999))
        btype = "band" if filter_type == "bandpass" else "bandstop"
        sos = butter(order, [wn_low, wn_high], btype=btype, output="sos")

    elif filter_type in ("lowshelf", "highshelf", "peak"):
        sos = _design_biquad_shelf_peak(
            filter_type, cutoff_freq, sample_rate, gain_db, q_factor
        )

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply filter to each channel
    if len(audio_data.shape) == 1:
        # Mono
        filtered = sosfilt(sos, audio_data)
    else:
        # Multi-channel: filter each channel independently
        filtered = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[0]):
            filtered[ch] = sosfilt(sos, audio_data[ch])

    return filtered


def _lookup_audio_from_events(
    events_input: Any,
    metadata: Optional[Dict[str, Any]],
) -> List[AudioDataItem]:
    """
    Resolve source AudioDataItems from event data (event metadata audio_id or item-level source).

    Events from DetectOnsets / LearnedOnsetDetector store "audio_id" in event metadata.
    EventDataItem metadata may also have "_source_audio_id". We collect these and fetch
    the corresponding AudioDataItems from the data item repository.

    Args:
        events_input: EventDataItem or list of EventDataItems
        metadata: Execution metadata (must contain "data_item_repo")

    Returns:
        List of AudioDataItems (may be empty if no references or repo unavailable)
    """
    if not metadata:
        return []
    data_item_repo = metadata.get("data_item_repo")
    if not data_item_repo:
        return []

    audio_ids = set()
    items = events_input if isinstance(events_input, list) else [events_input]
    for item in items:
        if not isinstance(item, EventDataItem):
            continue
        # Item-level source (e.g. from quick_actions / some exporters)
        item_audio_id = item.metadata.get("_source_audio_id") if item.metadata else None
        if item_audio_id:
            audio_ids.add(item_audio_id)
        for event in item.get_events():
            eid = event.metadata.get("audio_id") if event.metadata else None
            if eid:
                audio_ids.add(eid)

    if not audio_ids:
        return []

    audio_items: List[AudioDataItem] = []
    for audio_id in audio_ids:
        try:
            audio_item = data_item_repo.get(audio_id)
            if audio_item and isinstance(audio_item, AudioDataItem):
                if audio_item.get_audio_data() is not None:
                    audio_items.append(audio_item)
                    Log.info(
                        f"AudioFilterBlockProcessor: Resolved source audio "
                        f"'{audio_item.name}' from event data"
                    )
                else:
                    Log.warning(
                        f"AudioFilterBlockProcessor: Source audio '{audio_id}' "
                        f"has no audio data (file may be missing)"
                    )
        except Exception as e:
            Log.warning(
                f"AudioFilterBlockProcessor: Failed to look up audio '{audio_id}': {e}"
            )
    return audio_items


class AudioFilterBlockProcessor(BlockProcessor):
    """Processor for AudioFilter block type."""

    def can_process(self, block: Block) -> bool:
        """Check if this processor handles the given block."""
        return block.type == "AudioFilter"

    def get_block_type(self) -> str:
        """Get the block type identifier."""
        return "AudioFilter"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for AudioFilter block.

        Status levels:
        - Error (0): Audio input not connected
        - Stale (1): Data is stale
        - Ready (2): All requirements met
        """
        from src.features.blocks.domain import BlockStatusLevel
        from src.shared.domain.data_state import DataState

        def check_audio_or_events_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio or events input is connected (at least one required)."""
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id]
            has_audio = any(c.target_input_name == "audio" for c in incoming)
            has_events = any(c.target_input_name == "events" for c in incoming)
            return has_audio or has_events

        def check_data_fresh(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if block data is fresh (not stale)."""
            if not hasattr(f, "data_state_service") or not f.data_state_service:
                return True
            try:
                project_id = (
                    getattr(f, "current_project_id", None)
                    if hasattr(f, "current_project_id")
                    else None
                )
                data_state = f.data_state_service.get_block_data_state(blk.id, project_id)
                return data_state != DataState.STALE
            except Exception:
                return True

        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_or_events_input],
            ),
            BlockStatusLevel(
                priority=1,
                name="stale",
                display_name="Stale",
                color="#ffa94d",
                conditions=[check_data_fresh],
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[],
            ),
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process AudioFilter block: apply the configured filter to input audio.

        Handles both single AudioDataItem and list of AudioDataItems (e.g., from
        a Separator block that outputs multiple stems on one port). The output
        structure mirrors the input: single in -> single out, list in -> list out.

        The engine handles persistence -- we return DataItem(s) and the engine
        calls data_item_repo.create() to save them and updates local state so
        downstream blocks can pull.
        """
        from src.application.processing.output_name_helpers import make_output_name

        # -- Resolve audio input: direct audio or from events -----------------
        audio_input = inputs.get("audio")
        events_input = inputs.get("events")

        if audio_input is not None:
            # Direct audio: same as before
            if isinstance(audio_input, list):
                audio_items = audio_input
                input_is_list = True
            else:
                audio_items = [audio_input]
                input_is_list = False
        elif events_input is not None:
            # Resolve source audio from event data (audio_id / _source_audio_id)
            audio_items = _lookup_audio_from_events(events_input, metadata)
            if not audio_items:
                raise ProcessingError(
                    "AudioFilter block received event data but could not resolve source audio. "
                    "Events must reference audio (e.g. from DetectOnsets or LearnedOnsetDetector with audio_id).",
                    block_id=block.id,
                    block_name=block.name,
                )
            input_is_list = len(audio_items) > 1
        else:
            raise ProcessingError(
                "AudioFilter block requires either audio or event data. "
                "Connect an audio source or an events source (e.g. DetectOnsets) and pull data first.",
                block_id=block.id,
                block_name=block.name,
            )

        # Validate every item is AudioDataItem
        for item in audio_items:
            if not isinstance(item, AudioDataItem):
                raise ProcessingError(
                    f"Input contains non-audio item: {type(item).__name__}",
                    block_id=block.id,
                    block_name=block.name,
                )

        # -- Read filter settings from block metadata -----------------------
        filter_type = block.metadata.get("filter_type", "lowpass")
        cutoff_freq = float(block.metadata.get("cutoff_freq", 1000.0))
        cutoff_freq_high = float(block.metadata.get("cutoff_freq_high", 8000.0))
        order = int(block.metadata.get("order", 4))
        gain_db = float(block.metadata.get("gain_db", 0.0))
        q_factor = float(block.metadata.get("q_factor", 0.707))

        if filter_type not in FILTER_TYPES:
            raise ProcessingError(
                f"Unknown filter type: '{filter_type}'. "
                f"Valid types: {', '.join(FILTER_TYPES.keys())}",
                block_id=block.id,
                block_name=block.name,
            )

        # -- Progress -------------------------------------------------------
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        total_items = len(audio_items)
        filter_info = FILTER_TYPES[filter_type]
        if progress_tracker:
            progress_tracker.start(
                f"Applying {filter_info['name']} filter to {total_items} item(s)...",
                total=total_items,
            )

        Log.info(
            f"AudioFilterBlockProcessor: Applying {filter_type} filter to "
            f"{total_items} item(s) (cutoff={cutoff_freq}Hz, order={order}, "
            f"gain={gain_db}dB, Q={q_factor})"
        )

        # -- Build a shared output directory --------------------------------
        # Use the first item's location as the base, or fall back to temp
        first_path = audio_items[0].file_path if audio_items[0].file_path else None
        if first_path:
            output_dir = Path(first_path).parent / f"{block.name}_filtered"
        else:
            output_dir = Path(tempfile.gettempdir()) / "echozero_filters"
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- Process each audio item ----------------------------------------
        output_items: List[AudioDataItem] = []

        for idx, audio_item in enumerate(audio_items):
            audio_data = audio_item.get_audio_data()
            if audio_data is None:
                raise ProcessingError(
                    f"Could not load audio data from item '{audio_item.name}'",
                    block_id=block.id,
                    block_name=block.name,
                )

            sample_rate = audio_item.sample_rate or 44100

            try:
                filtered_data = apply_audio_filter(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    filter_type=filter_type,
                    cutoff_freq=cutoff_freq,
                    cutoff_freq_high=cutoff_freq_high,
                    order=order,
                    gain_db=gain_db,
                    q_factor=q_factor,
                )
            except Exception as e:
                raise ProcessingError(
                    f"Filter processing failed on '{audio_item.name}': {e}",
                    block_id=block.id,
                    block_name=block.name,
                ) from e

            # Determine a unique output filename
            if audio_item.file_path:
                stem_name = Path(audio_item.file_path).stem
            else:
                stem_name = audio_item.name or f"item_{idx}"
            output_path = output_dir / f"{stem_name}_{filter_type}.wav"

            # Create the output AudioDataItem
            output_item = AudioDataItem(
                id="",
                block_id=block.id,
                name=f"{block.name}_{stem_name}",
                type="Audio",
                created_at=datetime.utcnow(),
                file_path=str(output_path),
            )

            # Set filtered audio data and persist to disk
            output_item.set_audio_data(filtered_data, sample_rate)
            output_item.save_audio(str(output_path))

            # Preserve the upstream output_name so downstream filters still
            # see the same semantic name (e.g. "audio:vocals" stays "audio:vocals").
            # If the upstream item had no output_name, fall back to a generic one.
            upstream_output_name = audio_item.metadata.get("output_name")
            if upstream_output_name:
                output_item.metadata["output_name"] = upstream_output_name
            else:
                output_item.metadata["output_name"] = make_output_name(
                    "audio", stem_name
                )

            # Generate waveform for timeline display
            try:
                from src.shared.application.services.waveform_service import (
                    get_waveform_service,
                )

                waveform_service = get_waveform_service()
                waveform_service.compute_and_store(output_item)
            except Exception as e:
                Log.warning(
                    f"AudioFilterBlockProcessor: Failed to generate waveform "
                    f"for '{stem_name}': {e}"
                )

            output_items.append(output_item)

            if progress_tracker:
                progress_tracker.update(
                    idx + 1,
                    total_items,
                    f"Filtered {idx + 1}/{total_items}: {stem_name}",
                )

            Log.info(
                f"AudioFilterBlockProcessor: Saved filtered '{stem_name}' -> {output_path}"
            )

        # -- Complete -------------------------------------------------------
        if progress_tracker:
            progress_tracker.complete(
                f"Applied {filter_info['name']} filter to {total_items} item(s)"
            )

        Log.info(
            f"AudioFilterBlockProcessor: Returning {len(output_items)} "
            f"filtered audio item(s) on 'audio' port"
        )

        # #region agent log
        try:
            import json, time
            _out_names = [i.metadata.get("output_name", "?") for i in output_items]
            with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H-A,H-B", "location": "audio_filter_block.py:process-exit", "message": "AudioFilter process() output", "data": {"num_output_items": len(output_items), "input_is_list": input_is_list, "output_names": _out_names}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion

        # Mirror the input structure: single item or list (audio only; events input does not pass through)
        if input_is_list:
            return {"audio": output_items}
        return {"audio": output_items[0]}

    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        Get expected output names for the audio filter block.

        Since this is a pass-through transform block, the expected outputs
        depend dynamically on upstream connections and filter_selections.
        ExpectedOutputsService handles this via CONNECTION_BASED_OUTPUTS.

        This static fallback is used only when the service is unavailable
        (e.g., no connections, unconnected state).
        """
        from src.application.processing.output_name_helpers import make_output_name

        # #region agent log
        try:
            import json, time
            with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H-C", "location": "audio_filter_block.py:get_expected_outputs", "message": "get_expected_outputs called (static fallback)", "data": {"block_name": block.name}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion

        return {"audio": [make_output_name("audio", "main")]}

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        """Validate AudioFilter block configuration."""
        errors = []
        filter_type = block.metadata.get("filter_type", "lowpass")
        if filter_type not in FILTER_TYPES:
            errors.append(f"Invalid filter type: '{filter_type}'")

        cutoff = block.metadata.get("cutoff_freq", 1000.0)
        if cutoff < 20 or cutoff > 20000:
            errors.append(f"Cutoff frequency {cutoff}Hz is outside valid range (20-20000 Hz)")

        order = block.metadata.get("order", 4)
        if order < 1 or order > 8:
            errors.append(f"Filter order {order} is outside valid range (1-8)")

        return errors


register_processor_class(AudioFilterBlockProcessor)
