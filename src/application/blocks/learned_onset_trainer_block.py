"""
LearnedOnsetTrainer Block Processor

PoC trainer for frame-level onset detection using IDMT-style XML annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from src.application.blocks import register_processor_class
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.application.settings.learned_onset_trainer_settings import LearnedOnsetTrainerBlockSettings
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities.data_item import AudioDataItem as ModelDataItem
from src.utils.datasets import resolve_dataset_path
from src.utils.message import Log
from src.utils.paths import get_models_dir

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.features.blocks.domain import BlockStatusLevel

try:
    import librosa
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False


class TinyOnsetCNN(nn.Module):
    """Same architecture used by LearnedOnsetDetector for checkpoint compatibility."""

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
        feats = self.conv2d(x)
        feats = feats.mean(dim=2)
        logits = self.temporal_head(feats).squeeze(1)
        return logits


@dataclass
class TrainingExample:
    """A single windowed training example."""

    audio_path: Path
    center_time: float
    onset_times: np.ndarray


class WindowedOnsetDataset(Dataset):
    """Creates fixed windows with frame-level onset targets."""

    def __init__(
        self,
        examples: List[TrainingExample],
        sample_rate: int,
        n_mels: int,
        mel_hop_length: int,
        window_seconds: float,
        positive_radius_ms: float,
    ):
        self.examples = examples
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_hop_length = mel_hop_length
        self.window_seconds = window_seconds
        self.window_samples = int(sample_rate * window_seconds)
        self.positive_radius_sec = positive_radius_ms / 1000.0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        start_time = max(0.0, ex.center_time - self.window_seconds / 2.0)
        y, _ = librosa.load(
            str(ex.audio_path),
            sr=self.sample_rate,
            mono=True,
            offset=start_time,
            duration=self.window_seconds,
        )

        if len(y) < self.window_samples:
            y = np.pad(y, (0, self.window_samples - len(y)))
        elif len(y) > self.window_samples:
            y = y[: self.window_samples]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.mel_hop_length,
            fmax=min(11025, self.sample_rate // 2),
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        n_frames = mel_norm.shape[1]
        frame_times = start_time + (np.arange(n_frames) * self.mel_hop_length / self.sample_rate)
        labels = np.zeros(n_frames, dtype=np.float32)
        if ex.onset_times.size > 0:
            for i, t in enumerate(frame_times):
                if np.any(np.abs(ex.onset_times - t) <= self.positive_radius_sec):
                    labels[i] = 1.0

        x = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        y_target = torch.tensor(labels, dtype=torch.float32)
        return x, y_target


@register_processor_class
class LearnedOnsetTrainerBlockProcessor(BlockProcessor):
    """Train and export a PoC learned onset model."""

    def can_process(self, block: Block) -> bool:
        return block.type == "LearnedOnsetTrainer"

    def get_block_type(self) -> str:
        return "LearnedOnsetTrainer"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        from src.features.blocks.domain import BlockStatusLevel

        def has_dataset(blk: Block, f: "ApplicationFacade") -> bool:
            dataset_root = resolve_dataset_path((blk.metadata or {}).get("dataset_root", ""))
            return bool(dataset_root and os.path.isdir(dataset_root))

        def deps_ready(blk: Block, f: "ApplicationFacade") -> bool:
            return HAS_TRAINING_DEPS

        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[has_dataset],
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[deps_ready],
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
    ) -> Dict[str, DataItem]:
        if not HAS_TRAINING_DEPS:
            raise ProcessingError(
                "PyTorch and librosa are required for onset training. Install torch and librosa.",
                block_id=block.id,
                block_name=block.name,
            )

        settings = LearnedOnsetTrainerBlockSettings.from_dict(block.metadata or {})
        settings.dataset_root = resolve_dataset_path(settings.dataset_root) or ""
        self._validate_dataset_root(settings.dataset_root, block)
        self._seed_everything(settings.seed)

        records = self._collect_records(settings)
        if not records:
            raise ProcessingError(
                "No valid audio/annotation pairs found in dataset_root.",
                block_id=block.id,
                block_name=block.name,
            )

        train_examples, val_examples = self._build_examples(records, settings)
        if not train_examples or not val_examples:
            raise ProcessingError(
                "Insufficient training examples after dataset preparation.",
                block_id=block.id,
                block_name=block.name,
            )

        train_ds = WindowedOnsetDataset(
            examples=train_examples,
            sample_rate=settings.sample_rate,
            n_mels=settings.n_mels,
            mel_hop_length=settings.mel_hop_length,
            window_seconds=settings.window_seconds,
            positive_radius_ms=settings.positive_radius_ms,
        )
        val_ds = WindowedOnsetDataset(
            examples=val_examples,
            sample_rate=settings.sample_rate,
            n_mels=settings.n_mels,
            mel_hop_length=settings.mel_hop_length,
            window_seconds=settings.window_seconds,
            positive_radius_ms=settings.positive_radius_ms,
        )

        train_loader = DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)

        device = self._resolve_device(settings.device)
        model = TinyOnsetCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None

        from src.features.execution.application.progress_helpers import get_progress_tracker
        progress_tracker = get_progress_tracker(metadata)
        if progress_tracker:
            progress_tracker.start("Training onset model", total=settings.epochs)

        for epoch in range(settings.epochs):
            model.train()
            train_loss_acc = 0.0
            train_batches = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_acc += float(loss.item())
                train_batches += 1

            model.eval()
            val_loss_acc = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss_acc += float(loss.item())
                    val_batches += 1

            avg_train = train_loss_acc / max(1, train_batches)
            avg_val = val_loss_acc / max(1, val_batches)
            Log.info(
                f"LearnedOnsetTrainer: epoch {epoch + 1}/{settings.epochs} "
                f"train_loss={avg_train:.4f} val_loss={avg_val:.4f}"
            )
            if progress_tracker:
                progress_tracker.update(epoch + 1, settings.epochs, f"Epoch {epoch + 1}/{settings.epochs}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if progress_tracker:
            progress_tracker.complete("Onset model training complete")

        if best_state is None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        output_path = self._resolve_output_path(settings.output_model_path)
        checkpoint = {
            "model_state_dict": best_state,
            "config": {
                "task": "onset_detection",
                "sample_rate": settings.sample_rate,
                "n_mels": settings.n_mels,
                "hop_length": settings.mel_hop_length,
                "window_seconds": settings.window_seconds,
                "positive_radius_ms": settings.positive_radius_ms,
            },
            "metrics": {
                "best_val_loss": best_val_loss,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "epochs": settings.epochs,
            },
            "created_at": datetime.now().isoformat(),
        }
        torch.save(checkpoint, output_path)

        block.metadata = block.metadata or {}
        block.metadata["last_training"] = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(output_path),
            "best_val_loss": best_val_loss,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
        }

        output_item = ModelDataItem(
            id="",
            block_id=block.id,
            name="learned_onset_cnn",
            type="Model",
            created_at=datetime.now(),
            file_path=str(output_path),
            metadata={
                "model_path": str(output_path),
                "task": "onset_detection",
                "best_val_loss": best_val_loss,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
            },
        )
        return {"model": output_item}

    def _validate_dataset_root(self, dataset_root: str, block: Block) -> None:
        if not dataset_root:
            raise ProcessingError(
                "dataset_root is required for LearnedOnsetTrainer.",
                block_id=block.id,
                block_name=block.name,
            )
        root = Path(dataset_root)
        if not root.exists() or not root.is_dir():
            raise ProcessingError(
                f"Dataset root not found: {dataset_root}",
                block_id=block.id,
                block_name=block.name,
            )
        if not (root / "audio").exists() or not (root / "annotation_xml").exists():
            raise ProcessingError(
                "Dataset root must contain 'audio' and 'annotation_xml' directories.",
                block_id=block.id,
                block_name=block.name,
            )

    def _collect_records(self, settings: LearnedOnsetTrainerBlockSettings) -> List[Tuple[Path, np.ndarray]]:
        root = Path(settings.dataset_root)
        audio_dir = root / "audio"
        xml_dir = root / "annotation_xml"

        xml_files = sorted(xml_dir.glob("*.xml"))
        if settings.max_files and settings.max_files > 0:
            xml_files = xml_files[: settings.max_files]

        records: List[Tuple[Path, np.ndarray]] = []
        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
                root_node = tree.getroot()
                audio_name_node = root_node.find("./globalParameter/audioFileName")
                if audio_name_node is None or not audio_name_node.text:
                    continue
                audio_path = audio_dir / audio_name_node.text
                if not audio_path.exists():
                    continue
                onset_nodes = root_node.findall("./transcription/event/onsetSec")
                onset_times = []
                for node in onset_nodes:
                    try:
                        onset_times.append(float(node.text))
                    except (TypeError, ValueError):
                        continue
                if onset_times:
                    records.append((audio_path, np.array(sorted(onset_times), dtype=np.float32)))
            except Exception as exc:
                Log.warning(f"LearnedOnsetTrainer: skipping malformed xml {xml_path.name}: {exc}")
        return records

    def _build_examples(
        self,
        records: List[Tuple[Path, np.ndarray]],
        settings: LearnedOnsetTrainerBlockSettings,
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        rng = np.random.default_rng(settings.seed)
        radius_sec = settings.positive_radius_ms / 1000.0
        all_examples: List[TrainingExample] = []

        for audio_path, onset_times in records:
            pos_centers = onset_times.tolist()
            neg_count = int(len(pos_centers) * settings.negative_ratio)
            if neg_count <= 0:
                neg_count = len(pos_centers)

            duration = librosa.get_duration(path=str(audio_path))
            if duration <= 0:
                continue

            neg_centers = []
            attempts = 0
            max_attempts = max(200, neg_count * 30)
            while len(neg_centers) < neg_count and attempts < max_attempts:
                t = float(rng.uniform(0.0, duration))
                attempts += 1
                if np.any(np.abs(onset_times - t) <= radius_sec):
                    continue
                neg_centers.append(t)

            for t in pos_centers:
                all_examples.append(TrainingExample(audio_path=audio_path, center_time=float(t), onset_times=onset_times))
            for t in neg_centers:
                all_examples.append(TrainingExample(audio_path=audio_path, center_time=float(t), onset_times=onset_times))

        random.Random(settings.seed).shuffle(all_examples)
        split_idx = int(len(all_examples) * (1.0 - settings.validation_split))
        split_idx = max(1, min(split_idx, len(all_examples) - 1))
        return all_examples[:split_idx], all_examples[split_idx:]

    def _resolve_output_path(self, configured: Optional[str]) -> Path:
        if configured:
            out = Path(configured)
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        models_dir = get_models_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return models_dir / f"learned_onset_cnn_{ts}.pth"

    def _resolve_device(self, configured: str):
        if configured == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if configured == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TRAINING_DEPS:
            torch.manual_seed(seed)

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        errors: List[str] = []
        if not HAS_TRAINING_DEPS:
            errors.append("Missing dependencies: torch and librosa are required.")
            return errors

        dataset_root = (block.metadata or {}).get("dataset_root", "")
        dataset_root = resolve_dataset_path(dataset_root) or ""
        if not dataset_root:
            errors.append("dataset_root is required.")
            return errors

        root = Path(dataset_root)
        if not root.exists() or not root.is_dir():
            errors.append(f"Dataset root not found: {dataset_root}")
            return errors

        if not (root / "audio").exists():
            errors.append("dataset_root missing 'audio' folder")
        if not (root / "annotation_xml").exists():
            errors.append("dataset_root missing 'annotation_xml' folder")
        return errors
