"""
EQ Bands Block Processor

Multi-band parametric equalizer that applies gain to configurable frequency ranges.
Each band is defined by a low frequency, high frequency, and gain in dB.
Bands are applied in series using bandpass filters combined with gain mixing.

The approach for each band:
1. Isolate the frequency range with a bandpass filter
2. Scale the isolated signal by (linear_gain - 1)
3. Add back to the running signal

This ensures frequencies outside all bands pass through unchanged, and
overlapping bands accumulate their effects.
"""

import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem, DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


# Default bands for new blocks
DEFAULT_BANDS = [
    {"freq_low": 60.0, "freq_high": 250.0, "gain_db": 0.0},
    {"freq_low": 250.0, "freq_high": 2000.0, "gain_db": 0.0},
    {"freq_low": 2000.0, "freq_high": 8000.0, "gain_db": 0.0},
]


def apply_eq_bands(
    audio_data: np.ndarray,
    sample_rate: int,
    bands: List[Dict[str, float]],
    order: int = 4,
) -> np.ndarray:
    """
    Apply multi-band EQ gain to audio data.

    For each band, a bandpass filter isolates the frequency range, then the
    isolated signal is scaled and mixed back into the original to achieve
    the desired gain boost or cut.

    Args:
        audio_data: Audio samples as numpy array (1D mono or 2D multi-channel)
        sample_rate: Sample rate in Hz
        bands: List of dicts, each with "freq_low", "freq_high", "gain_db"
        order: Butterworth filter order (1-8)

    Returns:
        Processed audio data (same shape as input)
    """
    from scipy.signal import butter, sosfilt

    if not bands:
        return audio_data.copy()

    nyquist = sample_rate / 2.0
    result = audio_data.astype(np.float64).copy()

    for band in bands:
        freq_low = float(band.get("freq_low", 20.0))
        freq_high = float(band.get("freq_high", 20000.0))
        gain_db = float(band.get("gain_db", 0.0))

        # Skip bands with zero gain (no change)
        if abs(gain_db) < 0.01:
            continue

        # Clamp frequencies to valid range
        freq_low = max(20.0, min(freq_low, nyquist - 2.0))
        freq_high = max(freq_low + 1.0, min(freq_high, nyquist - 1.0))

        # Normalized frequencies
        wn_low = freq_low / nyquist
        wn_high = freq_high / nyquist
        wn_low = max(0.001, min(wn_low, 0.998))
        wn_high = max(wn_low + 0.001, min(wn_high, 0.999))

        # Design bandpass filter for this range
        try:
            sos = butter(order, [wn_low, wn_high], btype="band", output="sos")
        except Exception as e:
            Log.warning(
                f"EQBands: Could not design filter for band "
                f"{freq_low}-{freq_high} Hz: {e}"
            )
            continue

        # Linear gain factor: how much to scale the isolated band
        # gain_db > 0 => boost, gain_db < 0 => cut
        linear_gain = 10.0 ** (gain_db / 20.0)
        # We mix (linear_gain - 1) * bandpassed back into the signal
        # This means: output = original + (gain - 1) * bandpassed
        # For boost: adds energy in that range
        # For cut: subtracts energy in that range
        mix_factor = linear_gain - 1.0

        if len(audio_data.shape) == 1:
            # Mono
            bandpassed = sosfilt(sos, result)
            result += mix_factor * bandpassed
        else:
            # Multi-channel
            for ch in range(audio_data.shape[0]):
                bandpassed = sosfilt(sos, result[ch])
                result[ch] += mix_factor * bandpassed

    return result


class EQBandsBlockProcessor(BlockProcessor):
    """Processor for EQBands block type - multi-band parametric EQ."""

    def can_process(self, block: Block) -> bool:
        """Check if this processor handles the given block."""
        return block.type == "EQBands"

    def get_block_type(self) -> str:
        """Get the block type identifier."""
        return "EQBands"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for EQBands block.

        Status levels:
        - Error (0): Audio input not connected
        - Stale (1): Data is stale
        - Ready (2): All requirements met
        """
        from src.features.blocks.domain import BlockStatusLevel
        from src.shared.domain.data_state import DataState

        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio input is connected."""
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [
                c
                for c in connections
                if c.target_block_id == blk.id and c.target_input_name == "audio"
            ]
            return len(incoming) > 0

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
                conditions=[check_audio_input],
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
        Process EQBands block: apply multi-band EQ to input audio.

        Handles both single AudioDataItem and list of AudioDataItems.
        The output structure mirrors the input.
        """
        from src.application.processing.output_name_helpers import make_output_name

        # -- Validate input -------------------------------------------------
        audio_input = inputs.get("audio")
        if audio_input is None:
            raise ProcessingError(
                "EQBands block requires audio data from upstream block. "
                "Connect an audio source and pull data first.",
                block_id=block.id,
                block_name=block.name,
            )

        # Normalise to a list
        if isinstance(audio_input, list):
            audio_items = audio_input
            input_is_list = True
        else:
            audio_items = [audio_input]
            input_is_list = False

        # Validate every item is AudioDataItem
        for item in audio_items:
            if not isinstance(item, AudioDataItem):
                raise ProcessingError(
                    f"Input 'audio' contains non-audio item: {type(item).__name__}",
                    block_id=block.id,
                    block_name=block.name,
                )

        # -- Read EQ settings from block metadata ---------------------------
        bands = block.metadata.get("bands", DEFAULT_BANDS)
        order = int(block.metadata.get("order", 4))

        if not isinstance(bands, list):
            bands = DEFAULT_BANDS

        # Filter out invalid bands
        valid_bands = []
        for band in bands:
            if isinstance(band, dict) and "freq_low" in band and "freq_high" in band:
                valid_bands.append(band)
        bands = valid_bands

        # Count active bands (non-zero gain)
        active_bands = [b for b in bands if abs(float(b.get("gain_db", 0.0))) >= 0.01]

        # -- Progress -------------------------------------------------------
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        total_items = len(audio_items)
        if progress_tracker:
            progress_tracker.start(
                f"Applying {len(active_bands)} EQ band(s) to {total_items} item(s)...",
                total=total_items,
            )

        Log.info(
            f"EQBandsBlockProcessor: Applying {len(bands)} band(s) "
            f"({len(active_bands)} active) to {total_items} item(s), order={order}"
        )

        # -- Build output directory -----------------------------------------
        first_path = audio_items[0].file_path if audio_items[0].file_path else None
        if first_path:
            output_dir = Path(first_path).parent / f"{block.name}_eq"
        else:
            output_dir = Path(tempfile.gettempdir()) / "echozero_eq_bands"
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
                processed_data = apply_eq_bands(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    bands=bands,
                    order=order,
                )
            except Exception as e:
                raise ProcessingError(
                    f"EQ processing failed on '{audio_item.name}': {e}",
                    block_id=block.id,
                    block_name=block.name,
                ) from e

            # Output filename
            if audio_item.file_path:
                stem_name = Path(audio_item.file_path).stem
            else:
                stem_name = audio_item.name or f"item_{idx}"
            output_path = output_dir / f"{stem_name}_eq.wav"

            # Create output AudioDataItem
            output_item = AudioDataItem(
                id="",
                block_id=block.id,
                name=f"{block.name}_{stem_name}",
                type="Audio",
                created_at=datetime.utcnow(),
                file_path=str(output_path),
            )

            output_item.set_audio_data(processed_data, sample_rate)
            output_item.save_audio(str(output_path))

            # Preserve upstream output_name
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
                    f"EQBandsBlockProcessor: Failed to generate waveform "
                    f"for '{stem_name}': {e}"
                )

            output_items.append(output_item)

            if progress_tracker:
                progress_tracker.update(
                    idx + 1,
                    total_items,
                    f"EQ applied {idx + 1}/{total_items}: {stem_name}",
                )

            Log.info(
                f"EQBandsBlockProcessor: Saved EQ'd '{stem_name}' -> {output_path}"
            )

        # -- Complete -------------------------------------------------------
        if progress_tracker:
            progress_tracker.complete(
                f"Applied {len(active_bands)} EQ band(s) to {total_items} item(s)"
            )

        Log.info(
            f"EQBandsBlockProcessor: Returning {len(output_items)} "
            f"processed audio item(s) on 'audio' port"
        )

        # Mirror the input structure
        if input_is_list:
            return {"audio": output_items}
        else:
            return {"audio": output_items[0]}

    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """Get expected output names for the EQ bands block."""
        from src.application.processing.output_name_helpers import make_output_name
        return {"audio": [make_output_name("audio", "main")]}

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        """Validate EQBands block configuration."""
        errors = []
        bands = block.metadata.get("bands", [])

        if not isinstance(bands, list):
            errors.append("Bands configuration must be a list")
            return errors

        for i, band in enumerate(bands):
            if not isinstance(band, dict):
                errors.append(f"Band {i + 1} is not a valid configuration")
                continue

            freq_low = band.get("freq_low", 0)
            freq_high = band.get("freq_high", 0)
            gain_db = band.get("gain_db", 0)

            if freq_low < 20 or freq_low > 20000:
                errors.append(
                    f"Band {i + 1}: Low frequency {freq_low} Hz outside valid range (20-20000)"
                )
            if freq_high < 20 or freq_high > 20000:
                errors.append(
                    f"Band {i + 1}: High frequency {freq_high} Hz outside valid range (20-20000)"
                )
            if freq_low >= freq_high:
                errors.append(
                    f"Band {i + 1}: Low frequency must be less than high frequency"
                )
            if gain_db < -24 or gain_db > 24:
                errors.append(
                    f"Band {i + 1}: Gain {gain_db} dB outside valid range (-24 to +24)"
                )

        order = block.metadata.get("order", 4)
        if order < 1 or order > 8:
            errors.append(f"Filter order {order} is outside valid range (1-8)")

        return errors


register_processor_class(EQBandsBlockProcessor)
