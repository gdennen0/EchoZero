"""
Separator Block Processor

Processes Separator blocks - separates audio sources using Demucs.

Note: This block requires Demucs to be installed and configured separately.
The application does not manage Demucs installation or model downloads.

Optimization Options:
- device: "auto", "cpu", "cuda", "mps" (GPU acceleration - 30-50x faster)
- model: Model choice affects speed significantly:
  * "htdemucs" - Single model, faster than ensemble models
  * "mdx_extra" - Bag of 4 models (slower but higher quality)
  * "htdemucs_ft" - Single model, best quality, slower
- shifts: Number of random shifts for quality (default: 1)
  * 0 = Fastest (no shifts, lowest quality)
  * 1 = Default (good balance)
  * 10 = Paper recommendation (best quality, much slower)
- output_format: "wav" (default), "mp3" (faster writes, smaller files)
- mp3_bitrate: "320" (default), "192", "128"
- two_stems: None (default 4-stem), "vocals", "drums", "bass", "other"
  * Outputs only 2 stems (selected + everything else combined)
  * Note: Still processes all stems internally (no processing time savings)
  * Ensemble models (mdx_extra) still run all 4 models in the bag
  * Benefits: Less disk space, cleaner output, only get the stems you need
"""

# Available Demucs models with descriptions
DEMUCS_MODELS = {
    "htdemucs": {
        "description": "Hybrid Transformer Demucs (fast, good quality)",
        "quality": "Good",
        "speed": "Fast",
        "stems": 4
    },
    "htdemucs_ft": {
        "description": "Hybrid Transformer Demucs Fine-Tuned (best quality)",
        "quality": "Best",
        "speed": "Slower",
        "stems": 4
    },
    "htdemucs_6s": {
        "description": "Hybrid Transformer Demucs 6-stem (vocals, drums, bass, other, guitar, piano)",
        "quality": "Best",
        "speed": "Slowest",
        "stems": 6
    },
    "mdx_extra": {
        "description": "MDX Extra Quality",
        "quality": "Very Good",
        "speed": "Medium",
        "stems": 4
    },
    "mdx_extra_q": {
        "description": "MDX Extra Quality Quantized (faster)",
        "quality": "Very Good",
        "speed": "Fast",
        "stems": 4
    }
}


def get_demucs_models_info() -> str:
    """Get formatted string of available Demucs models."""
    lines = ["Available Demucs Models:", ""]
    for model_name, info in DEMUCS_MODELS.items():
        lines.append(f"  {model_name}")
        lines.append(f"    {info['description']}")
        lines.append(f"    Quality: {info['quality']} | Speed: {info['speed']} | Stems: {info['stems']}")
        lines.append("")
    lines.append("Usage: <block_name> set_model <model_name>")
    return "\n".join(lines)
import subprocess
import os
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities import DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Try to import certifi for SSL certificate bundle
try:
    import certifi
    CERTIFI_CA_BUNDLE = certifi.where()
except ImportError:
    CERTIFI_CA_BUNDLE = None


def detect_best_device() -> str:
    """
    Auto-detect the best available device for Demucs processing.
    
    Returns:
        "cuda" for NVIDIA GPU, "cpu" as fallback
    
    Note: MPS (Apple Silicon) is not used because PyTorch MPS backend doesn't
    support FFT operations yet, which are required by Demucs for spectral analysis.
    This is a known PyTorch limitation (https://github.com/pytorch/pytorch/issues/77764).
    """
    # Check for NVIDIA CUDA (fully supported)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    # Use CPU for Apple Silicon and other systems
    # MPS is not used because Demucs requires FFT operations (aten::_fft_r2c)
    # which are not yet implemented in PyTorch's MPS backend
    return "cpu"


class SeparatorBlockProcessor(BlockProcessor):
    """Processor for Separator block type"""
    
    def can_process(self, block: Block) -> bool:
        return block.type == "Separator"
    
    def get_block_type(self) -> str:
        return "Separator"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for Separator block.
        
        Status levels:
        - Error (0): Audio input not connected
        - Warning (1): Demucs not available
        - Ready (2): All requirements met
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio input is connected."""
            if not hasattr(f, 'connection_service'):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id and c.target_input_name == "audio"]
            return len(incoming) > 0
        
        def check_demucs_available(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if Demucs is available."""
            try:
                import demucs
                return True
            except ImportError:
                return False
        
        from src.shared.domain.data_state import DataState
        
        def check_data_fresh(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if block data is fresh (not stale)."""
            if not hasattr(f, 'data_state_service') or not f.data_state_service:
                return True  # If no data state service, assume fresh
            try:
                project_id = getattr(f, 'current_project_id', None) if hasattr(f, 'current_project_id') else None
                data_state = f.data_state_service.get_block_data_state(blk.id, project_id)
                return data_state != DataState.STALE
            except Exception:
                return True  # On error, assume fresh
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_input]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_demucs_available]
            ),
            BlockStatusLevel(
                priority=2,
                name="stale",
                display_name="Stale",
                color="#ffa94d",
                conditions=[check_data_fresh]
            ),
            BlockStatusLevel(
                priority=3,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        Get expected output names for separator block.
        
        Returns different outputs based on two_stems configuration:
        - 4-stem mode: ["audio:vocals", "audio:drums", "audio:bass", "audio:other"]
        - 2-stem mode: ["audio:{selected_stem}", "audio:other"]
        """
        from src.application.processing.output_name_helpers import make_output_name
        
        two_stems = block.metadata.get("two_stems")
        if two_stems:
            # 2-stem mode: selected stem + "other" combined
            return {
                "audio": [
                    make_output_name("audio", two_stems),
                    make_output_name("audio", "other")
                ]
            }
        else:
            # 4-stem mode
            return {
                "audio": [
                    make_output_name("audio", "vocals"),
                    make_output_name("audio", "drums"),
                    make_output_name("audio", "bass"),
                    make_output_name("audio", "other")
                ]
            }
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process Separator block using Demucs.
        
        Requires:
        - Demucs installed: pip install demucs
        - Models will be downloaded automatically on first use (if not already cached)
        """
        audio_item = inputs.get("audio")
        if not audio_item or not audio_item.file_path:
            raise ProcessingError(
                "Separator block requires audio data from upstream block",
                block_id=block.id,
                block_name=block.name
            )

        # Check if Demucs CLI is available
        import shutil
        demucs_cmd = shutil.which("demucs")
        if not demucs_cmd:
            raise ProcessingError(
                "Demucs is not installed or not in PATH.\n"
                "Install with: pip install demucs\n"
                "Then ensure 'demucs' command is available in your PATH.",
                block_id=block.id,
                block_name=block.name
            )

        # Get settings from block metadata with optimized defaults
        separator_settings = block.metadata.get("separator_settings", {})
        model = separator_settings.get("model", block.metadata.get("model", "htdemucs"))
        device_setting = block.metadata.get("device", "auto")
        output_format = block.metadata.get("output_format", "wav")
        mp3_bitrate = block.metadata.get("mp3_bitrate", "320")
        two_stems = block.metadata.get("two_stems")  # None, "vocals", "drums", "bass", or "other"
        shifts = block.metadata.get("shifts", 1)  # Number of random shifts (default: 1, paper used 10)
        
        # Auto-detect best device
        if device_setting == "auto":
            device = detect_best_device()
            Log.info(f"SeparatorBlockProcessor: Auto-detected device: {device}")
        else:
            device = device_setting
            # Warn if user manually selected MPS (not supported by Demucs)
            if device == "mps":
                Log.warning(
                    "MPS device requested but PyTorch MPS doesn't support FFT operations "
                    "required by Demucs. Falling back to CPU. "
                    "See: https://github.com/pytorch/pytorch/issues/77764"
                )
                device = "cpu"
        
        output_dir_str = block.metadata.get("output_dir")
        
        if output_dir_str:
            output_dir = Path(output_dir_str).expanduser()
        else:
            output_dir = Path(audio_item.file_path).parent / f"{block.name}_stems"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        input_file = Path(audio_item.file_path)
        if not input_file.exists():
            raise ProcessingError(
                f"Audio file not found: {input_file}",
                block_id=block.id,
                block_name=block.name
            )

        # Build optimization info message
        opts = [f"model={model}", f"device={device}"]
        if two_stems:
            opts.append(f"two_stems={two_stems}")
        if output_format == "mp3":
            opts.append(f"mp3@{mp3_bitrate}kbps")
        Log.info(f"SeparatorBlockProcessor: Separating {input_file} ({', '.join(opts)})")

        # Clean previous outputs for this model to prevent bleed-over from previous runs
        # This ensures when switching from 4-stem to 2-stem mode, old stems don't get returned
        model_output_dir = output_dir / model
        track_name = input_file.stem
        track_output_dir = model_output_dir / track_name
        if track_output_dir.exists():
            import shutil
            Log.info(f"SeparatorBlockProcessor: Cleaning previous outputs in {track_output_dir}")
            shutil.rmtree(track_output_dir)

        # Prepare environment for subprocess with SSL certificate support
        env = os.environ.copy()
        if CERTIFI_CA_BUNDLE:
            env["SSL_CERT_FILE"] = CERTIFI_CA_BUNDLE
            env["REQUESTS_CA_BUNDLE"] = CERTIFI_CA_BUNDLE

        # Build Demucs command with optimizations
        # Demucs uses -n/--name for model name, not --model
        # Demucs will download models automatically if needed (to ~/.cache/torch/hub/checkpoints/)
        # Use the resolved path from shutil.which() for reliable cross-platform execution
        cmd = [demucs_cmd, "-n", model, "-o", str(output_dir)]
        
        # Add device selection (GPU acceleration)
        if device != "cpu":
            cmd.extend(["-d", device])
        
        # Add shifts parameter (controls multiple passes for quality vs speed trade-off)
        if shifts != 1:  # Only add if non-default
            cmd.extend(["--shifts", str(shifts)])
        
        # Add two-stems mode to limit output files
        # Note: This still processes all stems internally, only controls output
        if two_stems:
            cmd.extend(["--two-stems", two_stems])
        
        # Add output format options
        if output_format == "mp3":
            cmd.append("--mp3")
            cmd.extend(["--mp3-bitrate", mp3_bitrate])
        
        # Add input file
        cmd.append(str(input_file))
        
        Log.info(f"SeparatorBlockProcessor: Running command: {' '.join(cmd)}")

        # Initialize progress tracker
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        Log.info(f"SeparatorBlockProcessor: progress_tracker available: {progress_tracker is not None}")
        if progress_tracker:
            Log.info(f"SeparatorBlockProcessor: progress_tracker.is_available() = {progress_tracker.is_available()}")
            progress_tracker.start("Starting audio separation...", total=None)

        # Run Demucs CLI with real-time output streaming
        # Use iter(readline) for safer line-by-line reading that avoids blocking
        try:
            # Force unbuffered output from demucs if it's Python-based
            # This ensures progress updates appear in real-time
            demucs_env = env.copy() if env else {}
            demucs_env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered Python output
            demucs_env['PYTHONDONTWRITEBYTECODE'] = '1'  # Disable .pyc files for faster startup
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                encoding='utf-8',
                errors='replace',  # Handle non-UTF-8 bytes gracefully on Windows
                bufsize=1,  # Line buffered (enables real-time streaming)
                env=demucs_env
            )
            
            # Stream output in real-time using iter(readline) which is safer than 'for line in stdout'
            # This reads line-by-line without blocking on buffer fills
            output_lines = []
            try:
                # Use iter() with readline for safer non-blocking reads
                # Empty string '' is sentinel - stops when stdout closes
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip()
                    if line:
                        Log.debug(f"Demucs: {line}")
                        output_lines.append(line)
                        
                        # Parse demucs progress bar output and publish immediately
                        # Format: "25%|██▌       | 1/4 [00:15<00:45, 15.2s/it]"
                        self._parse_and_publish_progress(line, block, metadata)
            finally:
                # Ensure stdout is closed (process might have finished)
                process.stdout.close()
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            if return_code != 0:
                error_msg = "\n".join(output_lines) if output_lines else "Unknown error"
                raise subprocess.CalledProcessError(return_code, process.args, output=error_msg)
                
        except subprocess.CalledProcessError as e:
            error_msg = e.output or str(e)
            
            # Provide helpful error messages for common issues
            if "SSL" in error_msg or "certificate" in error_msg.lower():
                error_msg = (
                    "SSL certificate verification failed. Demucs needs to download models.\n"
                    "To fix:\n"
                    "1. macOS: Run '/Applications/Python 3.*/Install Certificates.command'\n"
                    "2. Windows: pip install --upgrade certifi, or download cacert.pem and set SSL_CERT_FILE\n"
                    "3. Linux: Ensure ca-certificates package is installed\n"
                    "4. Or set SSL_CERT_FILE environment variable to your certificate bundle\n"
                    "5. Or manually download models (models cache in your torch hub directory)\n"
                    f"Original error: {error_msg}"
                )
            elif "not found" in error_msg.lower() or "No such file" in error_msg:
                error_msg = (
                    f"Demucs command failed: {error_msg}\n"
                    "Ensure Demucs is properly installed: pip install demucs"
                )
            
            raise ProcessingError(
                f"Demucs separation failed: {error_msg}",
                block_id=block.id,
                block_name=block.name
            ) from e
        except FileNotFoundError:
            raise ProcessingError(
                "Demucs command not found. Install with: pip install demucs",
                block_id=block.id,
                block_name=block.name
            )

        # Find output stems from the model we just ran
        # Demucs structure: output_dir/model_name/track_name/stem.wav (or .mp3)
        file_extension = "mp3" if output_format == "mp3" else "wav"
        stem_files = []
        
        # Only look in the specific model and track directory we just executed
        # model_output_dir and track_output_dir were calculated earlier
        if track_output_dir.exists():
            for stem_file in track_output_dir.glob(f"*.{file_extension}"):
                stem_files.append(stem_file)

        if not stem_files:
            raise ProcessingError(
                f"Demucs produced no stems in {output_dir}\n"
                "Check that the input file is valid and Demucs completed successfully.",
                block_id=block.id,
                block_name=block.name
            )

        # Create AudioDataItem for each stem - return as list on single port
        from src.application.processing.output_name_helpers import make_output_name
        
        audio_items = []
        for stem_path in stem_files:
            stem_name = stem_path.stem  # e.g., "vocals", "drums", "bass", "other"
            stem_item = AudioDataItem(
                id="",
                block_id=block.id,
                name=f"{block.name}_{stem_name}",
                type="Audio",
                created_at=datetime.utcnow(),
                file_path=str(stem_path)
            )
            
            # Set semantic output name for filtering
            stem_item.metadata['output_name'] = make_output_name("audio", stem_name)
            
            # Load audio metadata (needed for waveform generation)
            stem_item.load_audio(str(stem_path))
            
            # Generate and store waveform
            try:
                from src.shared.application.services.waveform_service import get_waveform_service
                waveform_service = get_waveform_service()
                waveform_service.compute_and_store(stem_item)
            except Exception as e:
                Log.warning(f"SeparatorBlockProcessor: Failed to generate waveform for stem '{stem_name}': {e}")
                # Continue without waveform (backward compatible)
            
            audio_items.append(stem_item)
            Log.info(f"SeparatorBlockProcessor: Created stem '{stem_name}' at {stem_path}")

        Log.info(f"SeparatorBlockProcessor: Returning {len(audio_items)} audio item(s) on 'audio' port")
        
        # Complete progress tracking
        if progress_tracker:
            progress_tracker.complete("Audio separation complete")
        
        return {"audio": audio_items}
    
    def _parse_and_publish_progress(self, line: str, block: Block, metadata: Optional[Dict[str, Any]]):
        """
        Parse demucs progress output and update progress tracker.
        
        Demucs progress format examples:
        - "  0%|          | 0/4 [00:00<?, ?it/s]"
        - " 25%|██▌       | 1/4 [00:15<00:45, 15.2s/it]"
        - " 50%|█████     | 2/4 [00:30<00:30, 15.1s/it]"
        - "100%|██████████| 4/4 [01:00<00:00, 15.0s/it]"
        """
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        if not progress_tracker:
            Log.debug(f"SeparatorBlock: No progress_tracker in metadata, skipping progress update")
            return
        
        # Check if progress tracker is available (has event_bus)
        if not progress_tracker.is_available():
            Log.debug(f"SeparatorBlock: Progress tracker not available (no event_bus), skipping progress update")
            return
        
        # Try to match percentage progress: "XX%|"
        match = re.search(r'(\d+)%\|', line)
        if match:
            percentage = int(match.group(1))
            
            # Update progress tracker using the DEMUCS PERCENTAGE directly
            # (Don't use the X/Y values as they are floating point seconds and cause incorrect calculations)
            message = f"Separating audio... {percentage}%"
            # Use INFO level so progress updates are visible even with DEBUG filtering
            Log.info(f"SeparatorBlock: Progress {percentage}%: {message}")
            progress_tracker.update(
                current=percentage,
                total=100,
                message=message
            )


    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate Separator block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # SeparatorBlock doesn't have specific validation requirements
        return []


register_processor_class(SeparatorBlockProcessor)

