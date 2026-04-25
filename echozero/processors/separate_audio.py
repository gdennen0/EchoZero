"""Demucs-backed source separation processor.
Exists to split mixed audio into isolated stems for downstream analysis workflows.
Connects execution-engine block runs to model-backed stem separation results.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echozero.domain.types import AudioData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# Available Demucs models
# ---------------------------------------------------------------------------

DEMUCS_MODELS: dict[str, dict[str, Any]] = {
    "htdemucs": {
        "description": "Hybrid Transformer Demucs (fast, good quality)",
        "quality": "good",
        "speed": "fast",
        "stems": ("drums", "bass", "other", "vocals"),
    },
    "htdemucs_ft": {
        "description": "Hybrid Transformer Demucs Fine-Tuned (best quality)",
        "quality": "best",
        "speed": "slower",
        "stems": ("drums", "bass", "other", "vocals"),
    },
    "htdemucs_6s": {
        "description": "Hybrid Transformer Demucs 6-stem",
        "quality": "best",
        "speed": "slowest",
        "stems": ("drums", "bass", "other", "vocals", "guitar", "piano"),
    },
    "mdx_extra": {
        "description": "MDX Extra Quality",
        "quality": "very_good",
        "speed": "medium",
        "stems": ("drums", "bass", "other", "vocals"),
    },
    "mdx_extra_q": {
        "description": "MDX Extra Quality Quantized (faster)",
        "quality": "very_good",
        "speed": "fast",
        "stems": ("drums", "bass", "other", "vocals"),
    },
}

DEMUCS_MODEL_ALIASES: dict[str, str] = {
    "latest_model": "htdemucs_ft",
}

DEFAULT_DEMUCS_MODEL = "latest_model"

VALID_DEVICES = {"auto", "cpu", "cuda"}
VALID_TWO_STEMS = {"vocals", "drums", "bass", "other"}
VALID_OUTPUT_FORMATS = {"wav", "mp3"}
VALID_MP3_BITRATES = {128, 192, 320}


def resolve_demucs_model_name(model_name: str) -> str:
    """Resolve user-facing stem-model aliases to concrete Demucs model ids."""
    return DEMUCS_MODEL_ALIASES.get(model_name, model_name)


def cleanup_stem_temp_dirs(working_dir: Path) -> int:
    """Remove temporary stem directories. Returns count of dirs removed."""
    tmp_stems = working_dir / "tmp" / "stems"
    if not tmp_stems.exists():
        return 0
    import shutil
    count = 0
    for d in tmp_stems.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Stem result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StemResult:
    """One separated stem with its name and file path."""

    name: str
    file_path: str
    sample_rate: int
    duration: float
    channel_count: int


# ---------------------------------------------------------------------------
# Separation function signature
# ---------------------------------------------------------------------------

# The injectable function. Default uses Demucs Python API.
# Returns a list of StemResults (one per output stem).
SeparateFn = Callable[
    [
        str,   # input_file_path
        str,   # model name
        str,   # device (cpu/cuda)
        int,   # shifts
        str | None,  # two_stems (None = all stems)
        str,   # output_dir
        str,   # output_format (wav/mp3)
        int,   # mp3_bitrate
    ],
    list[StemResult],
]


def _detect_device(requested: str) -> str:
    """Resolve 'auto' to the best available device."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _default_separate(
    input_file: str,
    model_name: str,
    device: str,
    shifts: int,
    two_stems: str | None,
    output_dir: str,
    output_format: str,
    mp3_bitrate: int,
) -> list[StemResult]:
    """Run Demucs separation (API when available, CLI fallback for Demucs v4)."""
    try:
        import demucs.api  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        return _separate_via_native_demucs(
            input_file=input_file,
            model_name=model_name,
            device=device,
            shifts=shifts,
            two_stems=two_stems,
            output_dir=output_dir,
            output_format=output_format,
            mp3_bitrate=mp3_bitrate,
        )
    except ImportError:
        raise ExecutionError("Demucs is not installed. Install with: pip install demucs")

    # Legacy API path (for builds exposing demucs.api.Separator)
    separator = demucs.api.Separator(
        model=model_name,
        device=device,
        shifts=shifts,
        segment=None,
    )
    _, separated = separator.separate_audio_file(Path(input_file))

    model_info = DEMUCS_MODELS.get(model_name, DEMUCS_MODELS["htdemucs"])
    output_stems = [two_stems] if two_stems else list(model_info["stems"])

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if output_format == "mp3":
        import logging
        logging.getLogger(__name__).warning(
            "MP3 output format requested but not yet supported in API path. Writing WAV instead."
        )
    ext = "wav"
    sample_rate = separator.samplerate
    results: list[StemResult] = []

    if two_stems:
        if two_stems in separated:
            stem_tensor = separated[two_stems]
            stem_file = out_path / f"{two_stems}.{ext}"
            _write_audio(stem_tensor, sample_rate, str(stem_file), output_format, mp3_bitrate)
            duration = stem_tensor.shape[-1] / sample_rate
            results.append(
                StemResult(
                    name=two_stems,
                    file_path=str(stem_file),
                    sample_rate=sample_rate,
                    duration=duration,
                    channel_count=stem_tensor.shape[0],
                )
            )

        import torch
        other_tensor = torch.zeros_like(next(iter(separated.values())))
        for name, tensor in separated.items():
            if name != two_stems:
                other_tensor += tensor
        other_file = out_path / f"no_{two_stems}.{ext}"
        _write_audio(other_tensor, sample_rate, str(other_file), output_format, mp3_bitrate)
        duration = other_tensor.shape[-1] / sample_rate
        results.append(
            StemResult(
                name=f"no_{two_stems}",
                file_path=str(other_file),
                sample_rate=sample_rate,
                duration=duration,
                channel_count=other_tensor.shape[0],
            )
        )
    else:
        for stem_name in output_stems:
            if stem_name not in separated:
                continue
            stem_tensor = separated[stem_name]
            stem_file = out_path / f"{stem_name}.{ext}"
            _write_audio(stem_tensor, sample_rate, str(stem_file), output_format, mp3_bitrate)
            duration = stem_tensor.shape[-1] / sample_rate
            results.append(
                StemResult(
                    name=stem_name,
                    file_path=str(stem_file),
                    sample_rate=sample_rate,
                    duration=duration,
                    channel_count=stem_tensor.shape[0],
                )
            )

    return results


def _separate_via_native_demucs(
    input_file: str,
    model_name: str,
    device: str,
    shifts: int,
    two_stems: str | None,
    output_dir: str,
    output_format: str,
    mp3_bitrate: int,
) -> list[StemResult]:
    """Demucs v4 fallback path without relying on torchaudio save/TorchCodec."""
    try:
        import torch as th
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.separate import load_track
        import soundfile as sf
    except ImportError as exc:
        raise ExecutionError(f"Demucs native fallback unavailable: {exc}")

    model = get_model(model_name)
    model.cpu()
    model.eval()

    wav = load_track(Path(input_file), model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    wav = wav - ref.mean()
    wav = wav / ref.std()

    sources = apply_model(
        model,
        wav[None],
        device=device,
        shifts=shifts,
        split=True,
        overlap=0.25,
        progress=False,
        num_workers=0,
        segment=None,
    )[0]
    sources = (sources * ref.std()) + ref.mean()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if output_format == "mp3":
        import logging
        logging.getLogger(__name__).warning(
            "MP3 output requested; fallback path writes WAV to avoid TorchCodec dependency."
        )

    names = list(model.sources)
    tensors = list(sources)

    if two_stems:
        if two_stems not in names:
            raise ExecutionError(f"Requested two_stems '{two_stems}' not in model sources: {names}")
        selected = tensors[names.index(two_stems)]
        remainder = th.zeros_like(selected)
        for name, tensor in zip(names, tensors):
            if name != two_stems:
                remainder += tensor
        names = [two_stems, f"no_{two_stems}"]
        tensors = [selected, remainder]

    results: list[StemResult] = []
    for stem_name, tensor in zip(names, tensors):
        stem_file = out / f"{stem_name}.wav"
        audio_np = tensor.cpu().numpy().T
        sf.write(str(stem_file), audio_np, model.samplerate)
        duration = float(audio_np.shape[0]) / float(model.samplerate)
        channels = int(audio_np.shape[1]) if audio_np.ndim > 1 else 1
        results.append(
            StemResult(
                name=stem_name,
                file_path=str(stem_file),
                sample_rate=int(model.samplerate),
                duration=duration,
                channel_count=channels,
            )
        )

    return results


def _audio_file_info(path: Path) -> tuple[int, float, int]:
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate else 0.0
        return int(info.sample_rate), duration, int(info.num_channels)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            n = int(wf.getnframes())
            ch = int(wf.getnchannels())
        return sr, (float(n) / float(sr) if sr else 0.0), ch


def _write_audio(
    tensor: Any,  # torch.Tensor — (channels, samples)
    sample_rate: int,
    file_path: str,
    output_format: str,
    mp3_bitrate: int,
) -> None:
    """Write a torch tensor to an audio file."""
    import numpy as np
    import soundfile as sf

    # Convert torch tensor to numpy: (channels, samples) → (samples, channels)
    audio_np = tensor.cpu().numpy().T

    # Note: output_format and mp3_bitrate are accepted for future use.
    # V1 always writes WAV. MP3 encoding requires pydub or ffmpeg.
    sf.write(file_path, audio_np, sample_rate)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class SeparateAudioProcessor:
    """Separates audio into stems and returns multiple AudioData outputs."""

    def __init__(self, separate_fn: SeparateFn | None = None) -> None:
        self._separate_fn = separate_fn or _default_separate

    def execute(self, block_id: str, context: ExecutionContext) -> Result[dict[str, AudioData]]:
        """Read upstream audio, separate into stems, return dict of AudioData per stem.

        Stems are written to a temp directory. The returned AudioData file_paths point
        into this temp dir. The caller (or downstream pipeline) owns cleanup after the
        files are consumed (e.g. imported into content-addressed storage).
        """
        # Report start
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="separate_audio",
                percent=0.0,
                message="Starting audio separation",
            )
        )

        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        model = str(settings.get("model", DEFAULT_DEMUCS_MODEL))
        resolved_model = resolve_demucs_model_name(model)
        device_setting = settings.get("device", "auto")
        shifts = settings.get("shifts", 1)
        two_stems = settings.get("two_stems", None)
        if isinstance(two_stems, str):
            normalized_two_stems = two_stems.strip().lower()
            if not normalized_two_stems or normalized_two_stems == "none":
                two_stems = None
            else:
                two_stems = normalized_two_stems
        include_bass_stem_layer = bool(settings.get("include_bass_stem_layer", False))
        include_vocals_stem_layer = bool(settings.get("include_vocals_stem_layer", False))
        include_other_stem_layer = bool(settings.get("include_other_stem_layer", False))
        if include_bass_stem_layer or include_vocals_stem_layer or include_other_stem_layer:
            # Full stem outputs are required whenever a non-drums stem is requested.
            two_stems = None
        output_format = settings.get("output_format", "wav")
        mp3_bitrate = settings.get("mp3_bitrate", 320)

        # Validate settings
        if resolved_model not in DEMUCS_MODELS:
            valid_models = tuple(sorted({*DEMUCS_MODELS.keys(), *DEMUCS_MODEL_ALIASES.keys()}))
            return err(ValidationError(
                f"Unknown model '{model}'. Valid: {', '.join(valid_models)}"
            ))
        if device_setting not in VALID_DEVICES:
            return err(ValidationError(
                f"Unknown device '{device_setting}'. Valid: {', '.join(VALID_DEVICES)}"
            ))
        if two_stems is not None and two_stems not in VALID_TWO_STEMS:
            return err(ValidationError(
                f"Unknown two_stems '{two_stems}'. Valid: {', '.join(VALID_TWO_STEMS)}"
            ))
        if not isinstance(shifts, int) or shifts < 0:
            return err(ValidationError(f"shifts must be a non-negative integer, got {shifts}"))
        if output_format not in VALID_OUTPUT_FORMATS:
            return err(ValidationError(
                f"Unknown output_format '{output_format}'. Valid: {', '.join(VALID_OUTPUT_FORMATS)}"
            ))
        if mp3_bitrate not in VALID_MP3_BITRATES:
            return err(ValidationError(
                f"Invalid mp3_bitrate {mp3_bitrate}. Valid: {VALID_MP3_BITRATES}"
            ))

        # Resolve device
        device = _detect_device(device_setting)

        # Create temp output directory for stems.
        # Use working_dir/tmp/stems if available (allows batch cleanup), else system temp.
        working_dir = settings.get("working_dir")
        if working_dir:
            tmp_base = Path(working_dir) / "tmp" / "stems"
            tmp_base.mkdir(parents=True, exist_ok=True)
            output_dir = str(tmp_base / f"{block_id}_{uuid.uuid4().hex[:8]}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = tempfile.mkdtemp(prefix=f"ez_stems_{block_id}_")

        # Report progress
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="separate_audio",
                percent=0.1,
                message=f"Separating with {resolved_model} on {device}",
            )
        )

        # Run separation
        try:
            stem_results = self._separate_fn(
                input_file=audio.file_path,
                model_name=resolved_model,
                device=device,
                shifts=shifts,
                two_stems=two_stems,
                output_dir=output_dir,
                output_format=output_format,
                mp3_bitrate=mp3_bitrate,
            )
        except Exception as exc:
            return err(
                ExecutionError(
                    f"Separation failed for block '{block_id}': "
                    f"{type(exc).__name__}: {exc}"
                )
            )

        if not stem_results:
            return err(ExecutionError(
                f"Separation produced no stems for block '{block_id}'"
            ))

        # Report near-complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="separate_audio",
                percent=0.9,
                message="Building output data",
            )
        )

        # Build output dict: one AudioData per stem, keyed by port name
        outputs: dict[str, AudioData] = {}
        for stem in stem_results:
            port_name = f"{stem.name}_out"
            outputs[port_name] = AudioData(
                sample_rate=stem.sample_rate,
                duration=stem.duration,
                file_path=stem.file_path,
                channel_count=stem.channel_count,
            )

        # Report complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="separate_audio",
                percent=1.0,
                message=f"Separation complete — {len(outputs)} stems",
            )
        )

        return ok(outputs)
