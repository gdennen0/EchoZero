"""
LearnedOnsetDetector Block Processor

Proof-of-concept learned onset detector that supports:
1) CNN-based frame prediction from a PyTorch checkpoint
2) Fallback spectral-flux envelope when model is unavailable
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.application.blocks import register_processor_class
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.application.settings.learned_onset_detector_settings import (
    LearnedOnsetDetectorBlockSettings,
)
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem, DataItem, Event, EventDataItem, EventLayer
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.features.blocks.domain import BlockStatusLevel

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    import torch.nn as nn

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


class TinyOnsetCNN(nn.Module):
    """Small CNN head for frame-level onset probability prediction."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.temporal_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, n_mels, n_frames]
        feats = self.conv2d(x)
        feats = feats.mean(dim=2)  # [batch, channels, n_frames]
        logits = self.temporal_head(feats).squeeze(1)  # [batch, n_frames]
        return logits


@dataclass
class _ModelCache:
    model_path: str
    model_mtime: float
    model: Any
    device: str
    config: Dict[str, Any]


@register_processor_class
class LearnedOnsetDetectorBlockProcessor(BlockProcessor):
    """Detect drum onsets using a learned probability curve + peak picking."""

    def __init__(self):
        self._cache: Optional[_ModelCache] = None

    def can_process(self, block: Block) -> bool:
        return block.type == "LearnedOnsetDetector"

    def get_block_type(self) -> str:
        return "LearnedOnsetDetector"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        from src.features.blocks.domain import BlockStatusLevel

        def has_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            return any(
                c.target_block_id == blk.id and c.target_input_name == "audio"
                for c in connections
            )

        def has_required_libs(blk: Block, f: "ApplicationFacade") -> bool:
            return HAS_LIBROSA

        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[has_audio_input],
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[has_required_libs],
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[],
            ),
        ]

    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        # Dynamic output names are handled by ExpectedOutputsService.
        return {}

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataItem]:
        if not HAS_LIBROSA:
            raise ProcessingError(
                "librosa is required for LearnedOnsetDetector. Install with: pip install librosa",
                block_id=block.id,
                block_name=block.name,
            )

        audio_input = inputs.get("audio")
        if not audio_input:
            raise ProcessingError(
                "Audio input required for LearnedOnsetDetector",
                block_id=block.id,
                block_name=block.name,
            )

        if isinstance(audio_input, list):
            audio_items = audio_input
        elif isinstance(audio_input, AudioDataItem):
            audio_items = [audio_input]
        else:
            raise ProcessingError(
                f"Audio input must be AudioDataItem or list of AudioDataItems, got {type(audio_input)}",
                block_id=block.id,
                block_name=block.name,
            )

        settings = LearnedOnsetDetectorBlockSettings.from_dict(block.metadata or {})
        from src.features.execution.application.progress_helpers import get_progress_tracker, track_progress

        progress_tracker = get_progress_tracker(metadata)
        output_items: List[EventDataItem] = []
        details: List[Dict[str, Any]] = []

        for audio_item in track_progress(audio_items, progress_tracker, "Detecting learned onsets"):
            if not isinstance(audio_item, AudioDataItem):
                continue

            y = audio_item.get_audio_data()
            if y is None and audio_item.file_path and os.path.exists(audio_item.file_path):
                if audio_item.load_audio(audio_item.file_path):
                    y = audio_item.get_audio_data()

            if y is None:
                Log.warning(f"LearnedOnsetDetector: Skipping '{audio_item.name}' (no audio data)")
                continue

            sr = audio_item.sample_rate
            if sr is None:
                Log.warning(f"LearnedOnsetDetector: Skipping '{audio_item.name}' (missing sample rate)")
                continue

            if y.ndim > 1:
                y = np.mean(y, axis=0)

            onset_curve, model_used, hop_length_used = self._predict_onset_curve(y, sr, settings, block)
            onset_times = self._peak_pick(onset_curve, sr, settings, hop_length_used)
            event_item = self._build_event_item(
                block,
                audio_item,
                onset_times,
                onset_curve,
                sr,
                hop_length_used,
            )
            output_items.append(event_item)
            details.append(
                {
                    "audio_name": audio_item.name,
                    "audio_id": audio_item.id if hasattr(audio_item, "id") else None,
                    "onset_count": len(onset_times),
                    "model_used": model_used,
                }
            )

        if not output_items:
            raise ProcessingError(
                "No events created. Check audio input and model settings.",
                block_id=block.id,
                block_name=block.name,
            )

        summary = {
            "last_execution": {
                "timestamp": datetime.now().isoformat(),
                "audio_items_processed": len(details),
                "total_onsets_detected": sum(d["onset_count"] for d in details),
                "details": details,
            }
        }
        if block.metadata:
            block.metadata.update(summary)
        else:
            block.metadata = summary

        if len(output_items) == 1:
            return {"events": output_items[0]}
        return {"events": output_items}

    def _predict_onset_curve(
        self,
        y: np.ndarray,
        sr: int,
        settings: LearnedOnsetDetectorBlockSettings,
        block: Block,
    ) -> tuple[np.ndarray, str, int]:
        """
        Returns normalized onset probability curve and model usage label.
        """
        model_path = settings.model_path
        can_try_model = bool(
            HAS_PYTORCH and model_path and os.path.exists(model_path)
        )

        if can_try_model:
            try:
                curve, hop_length = self._predict_with_model(y, sr, settings, model_path)
                return curve, "pytorch_cnn", hop_length
            except Exception as exc:
                if not settings.fallback_to_spectral_flux:
                    raise ProcessingError(
                        f"Model inference failed and fallback is disabled: {exc}",
                        block_id=block.id,
                        block_name=block.name,
                    )
                Log.warning(f"LearnedOnsetDetector: Model inference failed, using fallback. Error: {exc}")

        if not settings.fallback_to_spectral_flux:
            raise ProcessingError(
                "No valid model available and fallback_to_spectral_flux is disabled",
                block_id=block.id,
                block_name=block.name,
            )

        curve = librosa.onset.onset_strength(y=y, sr=sr, hop_length=settings.hop_length)
        curve = self._normalize_curve(curve)
        return curve, "spectral_flux_fallback", settings.hop_length

    def _predict_with_model(
        self,
        y: np.ndarray,
        sr: int,
        settings: LearnedOnsetDetectorBlockSettings,
        model_path: str,
    ) -> tuple[np.ndarray, int]:
        cache = self._load_cached_model(model_path, settings.device)
        model = cache.model
        config = cache.config or {}

        n_mels = int(config.get("n_mels", settings.n_mels))
        hop_length = int(config.get("hop_length", settings.hop_length))
        fmax = int(config.get("fmax", 11025))

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            fmax=fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        x = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(cache.device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

        return self._normalize_curve(np.asarray(probs, dtype=np.float32)), hop_length

    def _load_cached_model(self, model_path: str, device: str) -> _ModelCache:
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch is not installed")

        mtime = os.path.getmtime(model_path)
        if (
            self._cache
            and self._cache.model_path == model_path
            and self._cache.model_mtime == mtime
            and self._cache.device == device
        ):
            return self._cache

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("Expected checkpoint with key 'model_state_dict'")

        model = TinyOnsetCNN()
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        self._cache = _ModelCache(
            model_path=model_path,
            model_mtime=mtime,
            model=model,
            device=device,
            config=checkpoint.get("config", {}),
        )
        return self._cache

    def cleanup(self, block: Block) -> None:
        """
        Release cached model and GPU/MPS memory when block is removed or project unloaded.
        """
        cache = getattr(self, "_cache", None)
        if cache is not None:
            if HAS_PYTORCH and hasattr(cache, "model"):
                try:
                    cache.model.cpu()
                except Exception:
                    pass
            self._cache = None
        if HAS_PYTORCH:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and getattr(torch.mps, "empty_cache", None):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    def _peak_pick(
        self,
        onset_curve: np.ndarray,
        sr: int,
        settings: LearnedOnsetDetectorBlockSettings,
        hop_length: int,
    ) -> List[float]:
        wait_frames = max(1, int(settings.min_silence * sr / hop_length))
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_curve,
            sr=sr,
            hop_length=hop_length,
            units="frames",
            backtrack=settings.use_backtrack,
            delta=settings.threshold,
            wait=wait_frames,
        )
        return librosa.frames_to_time(
            onset_frames,
            sr=sr,
            hop_length=hop_length,
        ).tolist()

    def _build_event_item(
        self,
        block: Block,
        audio_item: AudioDataItem,
        onset_times: List[float],
        onset_curve: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> EventDataItem:
        from src.application.processing.output_name_helpers import make_default_output_name, make_output_name, parse_output_name
        from ui.qt_gui.widgets.timeline.types import MIN_EVENT_DURATION

        layer_name = f"{audio_item.name}-onsets"
        audio_output_name = audio_item.metadata.get("output_name") if audio_item.metadata else None
        if audio_output_name:
            try:
                _, item_name = parse_output_name(audio_output_name)
                event_output_name = make_output_name("events", item_name)
            except ValueError:
                event_output_name = make_default_output_name("events")
        else:
            event_output_name = make_default_output_name("events")

        events: List[Event] = []
        for t in onset_times:
            frame_idx = int((t * sr) / max(1, hop_length))
            prob = float(onset_curve[min(frame_idx, len(onset_curve) - 1)]) if len(onset_curve) else 0.0
            metadata = {
                "source": "LearnedOnsetDetector",
                "render_as_marker": True,
                "audio_name": str(audio_item.name),
                "audio_id": str(audio_item.id) if hasattr(audio_item, "id") and audio_item.id else None,
                "clip_start_time": float(t),
                "clip_end_time": float(t + MIN_EVENT_DURATION),
                "onset_probability": prob,
                "sample_rate": int(sr),
                "_original_source_block_id": str(block.id),
            }
            events.append(
                Event(
                    time=float(t),
                    classification="onset",
                    duration=float(MIN_EVENT_DURATION),
                    metadata=metadata,
                )
            )

        layer = EventLayer(
            name=layer_name,
            events=events,
            metadata={"source": "LearnedOnsetDetector", "audio_name": audio_item.name},
        )
        return EventDataItem(
            id="",
            block_id=block.id,
            name=f"{block.name}_{audio_item.name}_events",
            type="Event",
            metadata={"output_name": event_output_name},
            layers=[layer],
        )

    @staticmethod
    def _normalize_curve(curve: np.ndarray) -> np.ndarray:
        if curve.size == 0:
            return curve
        cmin = float(np.min(curve))
        cmax = float(np.max(curve))
        if cmax - cmin < 1e-8:
            return np.zeros_like(curve, dtype=np.float32)
        return ((curve - cmin) / (cmax - cmin)).astype(np.float32)
