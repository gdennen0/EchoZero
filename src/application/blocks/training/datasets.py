"""
Audio Classification Dataset

Unified dataset class supporting multi-class, binary (one-vs-all), and
positive_vs_other classification. Handles multi-format audio, stratified
splitting, feature caching, per-dataset normalization, weighted sampling,
and dataset statistics.

For binary mode: target/positive_classes -> positive, all others -> negative.
For positive_vs_other: selected classes keep distinct labels; all others -> "other".
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import functools
import hashlib
import json
import math
import random
import sys
import threading
import time
import warnings

import numpy as np

from src.utils.message import Log
from src.utils.paths import get_user_cache_dir

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Single supported path: soundfile only (WAV, FLAC, OGG, AIFF). No MP3, no deprecated backends.
SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
AUDIO_EXTENSIONS = SOUNDFILE_EXTENSIONS


def _load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load audio with soundfile and resample to target_sr if needed. One path, no fallbacks.
    Raises on unsupported format, missing file, or if soundfile is not installed.
    """
    if not HAS_SOUNDFILE:
        raise RuntimeError(
            "Audio loading requires soundfile. Install with: pip install soundfile. "
            "Supported formats: WAV, FLAC, OGG, AIFF (no MP3)."
        )
    if path.suffix.lower() not in SOUNDFILE_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Use one of: {', '.join(sorted(SOUNDFILE_EXTENSIONS))}. MP3 is not supported."
        )
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr and HAS_LIBROSA:
        # Training path prefers faster resampling for throughput.
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
    elif sr != target_sr:
        raise RuntimeError(
            f"File {path} has sample rate {sr}, target is {target_sr}. "
            "Resampling requires librosa. Install librosa or use files at the target sample rate."
        )
    return np.ascontiguousarray(data, dtype=np.float32), target_sr


def _safe_n_fft(n_fft: int, max_length: int) -> int:
    """
    Return an n_fft valid for signals of length max_length (power of 2, <= min(n_fft, max_length)).
    Avoids librosa warning "n_fft is too large for input signal".
    """
    cap = min(n_fft, max_length)
    if cap < 2:
        return 256
    p = 2 ** max(0, int(math.floor(math.log2(cap))))
    return max(256, min(p, cap))


def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Seeds numpy and random for a DataLoader worker. Must be module-level for pickling."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _find_audio_files(directory: Path, formats: List[str]) -> List[Path]:
    """Find all audio files in a directory recursively, including subdirectories."""
    allowed = {f".{fmt.lstrip('.')}" for fmt in formats}
    files = []
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in allowed:
            files.append(f)
    return files


def _compute_file_hash(path: Path) -> str:
    """Compute a fast hash of a file for caching purposes."""
    stat = path.stat()
    key = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Dataset Statistics
# ---------------------------------------------------------------------------

class DatasetStats:
    """Container for dataset statistics, saved with the model."""

    def __init__(
        self,
        total_samples: int,
        class_distribution: Dict[str, int],
        audio_formats_used: List[str],
        sample_rate: int,
        max_length: int,
        classification_mode: str,
        target_class: Optional[str] = None,
        balance_strategy: Optional[str] = None,
        pre_balance_distribution: Optional[Dict[str, int]] = None,
    ):
        self.total_samples = total_samples
        self.class_distribution = class_distribution
        self.audio_formats_used = audio_formats_used
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.classification_mode = classification_mode
        self.target_class = target_class
        self.balance_strategy = balance_strategy
        self.pre_balance_distribution = pre_balance_distribution

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_samples": self.total_samples,
            "class_distribution": self.class_distribution,
            "audio_formats_used": self.audio_formats_used,
            "sample_rate": self.sample_rate,
            "max_length": self.max_length,
            "classification_mode": self.classification_mode,
            "target_class": self.target_class,
        }
        if self.balance_strategy and self.balance_strategy != "none":
            d["balance_strategy"] = self.balance_strategy
            if self.pre_balance_distribution:
                d["pre_balance_distribution"] = self.pre_balance_distribution
        return d

    def summary(self) -> str:
        lines = [f"Total samples: {self.total_samples}"]
        for cls, count in sorted(self.class_distribution.items()):
            pct = 100 * count / max(self.total_samples, 1)
            lines.append(f"  {cls}: {count} ({pct:.1f}%)")
        lines.append(f"Audio formats: {', '.join(self.audio_formats_used)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature Cache
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Disk-based cache for pre-computed spectrograms.

    Caches are keyed by file hash + spectrogram config, so changing
    parameters automatically invalidates the cache.
    """

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        if enabled:
            config_key = hashlib.md5(
                json.dumps({
                    "sr": config.get("sample_rate", 22050),
                    "max_len": config.get("max_length", 22050),
                    "n_mels": config.get("n_mels", 128),
                    "hop": config.get("hop_length", 512),
                    "fmax": config.get("fmax", 8000),
                    "n_fft": config.get("n_fft", 2048),
                }, sort_keys=True).encode()
            ).hexdigest()[:8]
            self.cache_dir = get_user_cache_dir() / "feature_cache" / config_key
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def get(self, file_path: Path) -> Optional[np.ndarray]:
        """Load cached spectrogram if available."""
        if not self.enabled:
            return None
        cache_path = self._cache_path(file_path)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                return None
        return None

    def put(self, file_path: Path, spectrogram: np.ndarray) -> None:
        """Save spectrogram to cache."""
        if not self.enabled:
            return
        cache_path = self._cache_path(file_path)
        try:
            np.save(cache_path, spectrogram)
        except Exception as e:
            Log.debug(f"Failed to cache spectrogram: {e}")

    def _cache_path(self, file_path: Path) -> Path:
        file_hash = _compute_file_hash(file_path)
        return self.cache_dir / f"{file_hash}.npy"


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------

class AudioClassificationDataset(Dataset):
    """
    Unified audio classification dataset.

    Supports both multi-class and binary classification modes.
    Loads audio from folder structure, computes mel spectrograms,
    applies augmentation, and caches features to disk.

    Folder structure:
        data_dir/
            class_a/
                sample1.wav
            class_b/
                sample2.wav

    For binary mode with target_class="class_a":
        - class_a/ -> label 1 (positive)
        - class_b/, class_c/, ... -> label 0 (negative)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dict (from TrainingConfig.to_dict())
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for AudioClassificationDataset")

        self.config = config
        self.sample_rate = config.get("sample_rate", 22050)
        self.max_length = config.get("max_length", 22050)
        self.n_mels = config.get("n_mels", 128)
        self.hop_length = config.get("hop_length", 512)
        # Clamp fmax to Nyquist (sr/2) - physical limit of digital audio; above that, content is undefined
        requested_fmax = config.get("fmax", 8000)
        nyquist = self.sample_rate // 2
        self.fmax = min(requested_fmax, nyquist)
        if requested_fmax != self.fmax:
            Log.info(
                f"Spectrogram fmax clamped to Nyquist: {requested_fmax} -> {self.fmax} (sr={self.sample_rate}). "
                f"For fmax above {nyquist} Hz (e.g. snare crack 12-16 kHz), set Sample Rate to 44100 Hz or higher."
            )
        # Ensure n_fft <= max_length (power of 2) to avoid "n_fft too large for input signal"
        requested_n_fft = config.get("n_fft", 2048)
        self.n_fft = _safe_n_fft(requested_n_fft, self.max_length)
        if self.n_fft != requested_n_fft:
            Log.info(
                f"Spectrogram n_fft adjusted for max_length: {requested_n_fft} -> {self.n_fft} (max_length={self.max_length})"
            )

        self.is_binary = config.get("classification_mode", "multiclass") == "binary"
        self.is_positive_vs_other = config.get("classification_mode", "multiclass") == "positive_vs_other"
        # Support 1+ positive classes; backward compat from target_class
        self.positive_classes = list(config.get("positive_classes") or [])
        if not self.positive_classes and config.get("target_class"):
            self.positive_classes = [config.get("target_class")]
        self.target_class = (self.positive_classes[0] if self.positive_classes else None) or config.get("target_class")

        # Audio formats to load (soundfile only: WAV, FLAC, OGG, AIFF; no MP3, no deprecated backends)
        formats = config.get("audio_formats", ["wav", "flac", "ogg", "aiff"])
        allowed_exts = {f".{f.strip().lstrip('.').lower()}" for f in formats}
        unsupported = allowed_exts - SOUNDFILE_EXTENSIONS
        if unsupported:
            raise ValueError(
                f"Unsupported audio format(s): {', '.join(sorted(unsupported))}. "
                f"Use only: {', '.join(sorted(SOUNDFILE_EXTENSIONS))}. "
                "MP3 is not supported; convert to WAV or FLAC."
            )
        if not HAS_SOUNDFILE:
            raise RuntimeError(
                "Audio loading requires soundfile. Install with: pip install soundfile"
            )

        # Setup augmentation pipelines
        from .augmentation import (
            AudioAugmentationPipeline,
            SpectrogramAugmentationPipeline,
            apply_positive_class_filter,
        )
        self.audio_augmenter = AudioAugmentationPipeline(config)
        self.spec_augmenter = SpectrogramAugmentationPipeline(config)
        self._apply_positive_class_filter_fn = apply_positive_class_filter

        # Optional filter applied to all positive-class samples (binary mode only)
        self._positive_filter_type = config.get("positive_filter_type") or None
        self._positive_filter_enabled = (
            self.is_binary
            and self._positive_filter_type in ("lowpass", "highpass", "bandpass")
        )
        self._positive_filter_cutoff_hz = config.get("positive_filter_cutoff_hz", 1000.0)
        self._positive_filter_cutoff_high_hz = config.get(
            "positive_filter_cutoff_high_hz", 4000.0
        )
        self._positive_filter_order = config.get("positive_filter_order", 4)

        # Feature cache
        self.cache = FeatureCache(config, enabled=config.get("use_feature_cache", True))

        # DOSE-inspired features
        self.use_transient_emphasis = config.get("use_transient_emphasis", False)
        self.use_multi_scale = config.get("use_multi_scale_features", False)
        self.use_onset_weighting = config.get("use_onset_weighting", False)

        # Scan data directory
        data_dir = Path(config["data_dir"])
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Discover classes and samples
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[Path, int]] = []
        self._formats_used: set = set()
        self.onset_strengths: List[float] = []

        self._scan_directory(data_dir, formats)

        if not self.samples:
            raise ValueError(
                f"No audio files found in {data_dir}. "
                f"Expected subdirectories with audio files ({', '.join(formats)})."
            )

        # Handle binary mode negative sampling / balancing (not for positive_vs_other)
        if self.is_binary:
            self._apply_binary_balancing(config, data_dir, formats)

        # Apply dataset balancing strategy (non-destructive sample selection)
        balance_strategy = config.get("balance_strategy", "none")
        if balance_strategy != "none":
            self._apply_balance_strategy(config)

        # Compute per-dataset normalization statistics
        self.normalize_per_dataset = config.get("normalize_per_dataset", True)
        self._global_mean: Optional[float] = None
        self._global_std: Optional[float] = None

        # Use provided normalization if available (e.g., from a loaded model)
        norm_mean = config.get("normalization_mean")
        norm_std = config.get("normalization_std")
        if norm_mean is not None and norm_std is not None:
            self._global_mean = np.array(norm_mean)
            self._global_std = np.array(norm_std)

        # Build stats
        class_dist = {}
        for _, label in self.samples:
            cls_name = self.classes[label] if label < len(self.classes) else "unknown"
            class_dist[cls_name] = class_dist.get(cls_name, 0) + 1

        # Capture pre-balance distribution if balancing was applied
        pre_balance_dist = None
        bal_strategy = config.get("balance_strategy", "none")
        if hasattr(self, "_balance_result") and self._balance_result:
            pre_balance_dist = {
                self.classes[label] if label < len(self.classes) else str(label): count
                for label, count in self._balance_result.original_distribution.items()
            }

        mode = "binary" if self.is_binary else ("positive_vs_other" if self.is_positive_vs_other else "multiclass")
        self.stats = DatasetStats(
            total_samples=len(self.samples),
            class_distribution=class_dist,
            audio_formats_used=sorted(self._formats_used),
            sample_rate=self.sample_rate,
            max_length=self.max_length,
            classification_mode=mode,
            target_class=(self.positive_classes[0] if self.positive_classes else None) or self.target_class,
            balance_strategy=bal_strategy if bal_strategy != "none" else None,
            pre_balance_distribution=pre_balance_dist,
        )

        Log.info(f"Dataset: {self.stats.summary()}")

    def _scan_directory(self, data_dir: Path, formats: List[str]) -> None:
        """Scan directory structure to discover classes and samples."""
        positive_set = set(self.positive_classes) if self.positive_classes else set()
        if self.is_binary and positive_set:
            # Binary mode: 1 or 2 folders in positive_classes -> 1, everything else -> 0
            self.classes = ["negative", "positive"]
            self.class_to_idx = {"negative": 0, "positive": 1}

            for class_dir in sorted(data_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                is_positive = class_dir.name in positive_set
                label = 1 if is_positive else 0

                files = _find_audio_files(class_dir, formats)
                for f in files:
                    self.samples.append((f, label))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)  # Default

            # Also check for explicit positive/negative folders
            pos_dir = data_dir / "positive"
            neg_dir = data_dir / "negative"
            if pos_dir.exists():
                for f in _find_audio_files(pos_dir, formats):
                    self.samples.append((f, 1))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)
            if neg_dir.exists():
                for f in _find_audio_files(neg_dir, formats):
                    self.samples.append((f, 0))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)
        elif self.is_positive_vs_other and positive_set:
            # Positive vs other: selected classes get indices 0..K-1, all others -> "other" at index K
            self.classes = list(self.positive_classes) + ["other"]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            for class_dir in sorted(data_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                label = self.class_to_idx.get(class_name, self.class_to_idx["other"])

                files = _find_audio_files(class_dir, formats)
                for f in files:
                    self.samples.append((f, label))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)

            other_idx = self.class_to_idx["other"]
            pos_count = sum(1 for _, lab in self.samples if lab != other_idx)
            if pos_count == 0:
                names = ", ".join(self.positive_classes) if self.positive_classes else "?"
                raise ValueError(
                    f"No samples found for positive class(es) '{names}'. "
                    f"Ensure at least one of these folders exists in {data_dir}."
                )
        else:
            # Multi-class mode
            for class_dir in sorted(data_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                idx = len(self.classes)
                self.classes.append(class_name)
                self.class_to_idx[class_name] = idx

                files = _find_audio_files(class_dir, formats)
                for f in files:
                    self.samples.append((f, idx))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)

    def _apply_binary_balancing(self, config: Dict[str, Any], data_dir: Path, formats: List[str]) -> None:
        """Balance positive/negative samples for binary classification."""
        pos_count = sum(1 for _, label in self.samples if label == 1)
        neg_count = sum(1 for _, label in self.samples if label == 0)

        if pos_count == 0:
            names = ", ".join(self.positive_classes) if self.positive_classes else (self.target_class or "?")
            raise ValueError(
                f"No positive samples found for positive class(es) '{names}'. "
                f"Make sure folder(s) exist in {data_dir}."
            )
        if neg_count == 0:
            raise ValueError(
                f"No negative samples found. Need at least one folder that is not in positive classes in {data_dir}."
            )

        # Load hard negatives if specified
        hard_neg_dir = config.get("hard_negative_dir")
        if hard_neg_dir:
            hard_neg_path = Path(hard_neg_dir)
            if hard_neg_path.exists():
                hard_neg_files = _find_audio_files(hard_neg_path, formats)
                for f in hard_neg_files:
                    self.samples.append((f, 0))
                    self._formats_used.add(f.suffix.lower().lstrip("."))
                    self.onset_strengths.append(0.5)
                neg_count += len(hard_neg_files)
                Log.info(f"Loaded {len(hard_neg_files)} hard negatives from {hard_neg_dir}")

        # Apply negative_ratio balancing
        target_ratio = config.get("negative_ratio", 1.0)
        desired_neg = int(pos_count * target_ratio)

        if desired_neg < neg_count:
            # Under-sample negatives
            neg_indices = [i for i, (_, label) in enumerate(self.samples) if label == 0]
            keep_indices = set(np.random.choice(neg_indices, size=desired_neg, replace=False))
            pos_samples = [(p, l) for i, (p, l) in enumerate(self.samples) if l == 1]
            neg_samples = [(p, l) for i, (p, l) in enumerate(self.samples) if i in keep_indices]
            self.samples = pos_samples + neg_samples
            self.onset_strengths = [0.5] * len(self.samples)
            Log.info(f"Under-sampled negatives: {neg_count} -> {desired_neg}")
        elif desired_neg > neg_count:
            # Over-sample negatives by duplication
            neg_samples = [(p, l) for p, l in self.samples if l == 0]
            extra_needed = desired_neg - neg_count
            extra = [neg_samples[i % len(neg_samples)] for i in range(extra_needed)]
            self.samples.extend(extra)
            self.onset_strengths.extend([0.5] * len(extra))
            Log.info(f"Over-sampled negatives: {neg_count} -> {desired_neg}")

        Log.info(
            f"Binary dataset: {sum(1 for _, l in self.samples if l == 1)} positive, "
            f"{sum(1 for _, l in self.samples if l == 0)} negative"
        )

    def _apply_balance_strategy(self, config: Dict[str, Any]) -> None:
        """
        Apply a non-destructive balancing strategy to the current sample list.

        Works for both binary (labels 0/1) and multiclass (labels 0..N-1): the
        same strategies (undersample_min, oversample_max, hybrid, etc.) apply
        to the current set of labels. Replaces self.samples and self.onset_strengths
        with the balanced set. Stores augmentation flags for oversampled duplicates
        so the augmentation pipeline can apply extra variation to those samples.
        """
        from .balancing import balance_dataset

        strategy = config.get("balance_strategy", "none")
        target_count = config.get("balance_target_count")
        seed = config.get("seed", 42)

        result = balance_dataset(
            samples=self.samples,
            strategy=strategy,
            target_count=target_count,
            seed=seed,
        )

        self.samples = result.samples
        self.onset_strengths = [0.5] * len(self.samples)
        self._augment_flags = result.augment_flags
        self._balance_result = result

        Log.info(f"Dataset balancing applied:\n{result.summary()}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_start = time.perf_counter()
        file_path = None
        try:
            file_path, label = self.samples[idx]

            # Try cache first (only when augmentation off: augmented samples vary so we can't reuse cached mel).
            # Skip cache for positive samples when positive-class filter is enabled (mel must be from filtered audio).
            use_cache = (
                not (self._positive_filter_enabled and label == 1)
                and not self.audio_augmenter.enabled
            )
            cached = self.cache.get(file_path) if use_cache else None
            if cached is not None:
                mel_spec_norm = cached
            else:
                # Load audio: single path via soundfile; resample with librosa if needed.
                audio, sr = _load_audio(file_path, self.sample_rate)
                audio = np.asarray(audio, dtype=np.float32)

                # Apply deterministic filter to positive-class samples (binary mode only)
                if self._positive_filter_enabled and label == 1:
                    audio = self._apply_positive_class_filter_fn(
                        audio,
                        self.sample_rate,
                        self._positive_filter_type,
                        self._positive_filter_cutoff_hz,
                        self._positive_filter_cutoff_high_hz,
                        self._positive_filter_order,
                    )

                # Apply audio augmentation
                if self.audio_augmenter.enabled:
                    audio = self.audio_augmenter(audio, self.sample_rate)
                    # Oversampled duplicates (from balancing) get a second pass for extra variation
                    augment_flags = getattr(self, "_augment_flags", None)
                    if augment_flags is not None and idx < len(augment_flags) and augment_flags[idx]:
                        audio = self.audio_augmenter(audio, self.sample_rate)

                # Pad or truncate to fixed max_length so all samples have same length
                if len(audio) < self.max_length:
                    audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
                elif len(audio) > self.max_length:
                    if self.audio_augmenter.enabled:
                        start = np.random.randint(0, len(audio) - self.max_length)
                        audio = audio[start:start + self.max_length]
                    else:
                        audio = audio[:self.max_length]
                # Defensive: guarantee length (handles any augmenter that might change length)
                if len(audio) != self.max_length:
                    if len(audio) < self.max_length:
                        audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
                    else:
                        audio = audio[:self.max_length]

                # Apply transient emphasis if enabled (DOSE-inspired)
                if self.use_transient_emphasis:
                    audio = librosa.effects.preemphasis(audio, coef=0.97)

                # Force length to at least max(max_length, n_fft) so librosa.stft never sees n_fft > len(y)
                min_len = max(self.max_length, self.n_fft)
                if len(audio) < min_len:
                    audio = np.pad(
                        audio, (0, min_len - len(audio)),
                        mode="constant", constant_values=0,
                    )
                y_for_mel = np.ascontiguousarray(audio)

                # Compute mel spectrogram (suppress librosa stft "n_fft too large" if it still fires)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*n_fft.*too large for input signal.*", category=UserWarning
                    )
                    mel_spec = librosa.feature.melspectrogram(
                        y=y_for_mel,
                        sr=self.sample_rate,
                        n_mels=self.n_mels,
                        hop_length=self.hop_length,
                        fmax=self.fmax,
                        n_fft=self.n_fft,
                    )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Normalize
                if self._global_mean is not None and self._global_std is not None:
                    mel_spec_norm = (mel_spec_db - self._global_mean) / (self._global_std + 1e-10)
                else:
                    # Per-sample normalization (fallback)
                    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
                        mel_spec_db.max() - mel_spec_db.min() + 1e-10
                    )

                # Cache (only if no augmentation and not positive with filter; augmented/filtered vary)
                if use_cache:
                    self.cache.put(file_path, mel_spec_norm)

            # Apply spectrogram augmentation
            if self.spec_augmenter.enabled:
                mel_spec_norm = self.spec_augmenter(mel_spec_norm)

            # Add channel dimension: (1, freq, time)
            mel_spec_rgb = np.expand_dims(mel_spec_norm, axis=0).astype(np.float32)

            audio_tensor = torch.tensor(mel_spec_rgb, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Return onset strength if needed
            if self.use_onset_weighting and idx < len(self.onset_strengths):
                onset_tensor = torch.tensor(self.onset_strengths[idx], dtype=torch.float32)
                return audio_tensor, label_tensor, onset_tensor

            elapsed = time.perf_counter() - sample_start
            if elapsed > 1.0:
                Log.debug(f"Slow sample preprocessing ({elapsed:.2f}s): {file_path}")
            return audio_tensor, label_tensor
        except Exception as e:
            target = str(file_path) if file_path is not None else f"sample_idx={idx}"
            raise RuntimeError(f"Failed to prepare training sample '{target}': {e}") from e

    def compute_normalization_stats(self, max_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dataset-wide mean and standard deviation for normalization.

        Samples up to max_samples spectrograms to estimate statistics efficiently.

        Returns:
            Tuple of (mean, std) arrays
        """
        if not HAS_LIBROSA:
            return np.array([0.0]), np.array([1.0])

        indices = np.random.choice(
            len(self.samples), size=min(max_samples, len(self.samples)), replace=False
        )

        all_specs = []
        for idx in indices:
            file_path, _ = self.samples[idx]
            try:
                audio, _ = _load_audio(file_path, self.sample_rate)
                audio = np.asarray(audio, dtype=np.float32)
                if len(audio) < self.max_length:
                    audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")
                else:
                    audio = audio[:self.max_length]
                min_len_norm = max(self.max_length, self.n_fft)
                if len(audio) < min_len_norm:
                    audio = np.pad(
                        audio, (0, min_len_norm - len(audio)),
                        mode="constant", constant_values=0,
                    )
                y_norm = np.ascontiguousarray(audio)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*n_fft.*too large for input signal.*", category=UserWarning
                    )
                    mel_spec = librosa.feature.melspectrogram(
                        y=y_norm, sr=self.sample_rate, n_mels=self.n_mels,
                        hop_length=self.hop_length, fmax=self.fmax, n_fft=self.n_fft,
                    )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                all_specs.append(mel_spec_db)
            except Exception as e:
                raise RuntimeError(f"Failed to load or process {file_path} for normalization: {e}") from e

        if not all_specs:
            return np.array([0.0]), np.array([1.0])

        stacked = np.stack(all_specs)
        mean = stacked.mean()
        std = stacked.std()

        self._global_mean = mean
        self._global_std = std

        Log.info(f"Dataset normalization: mean={mean:.4f}, std={std:.4f}")
        return np.array([mean]), np.array([std])

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the loss function."""
        from .losses import compute_class_weights
        return compute_class_weights(self.stats.class_distribution)

    def get_binary_pos_weight(self) -> float:
        """Compute pos_weight for binary BCE loss."""
        from .losses import compute_binary_pos_weight
        pos = sum(1 for _, l in self.samples if l == 1)
        neg = sum(1 for _, l in self.samples if l == 0)
        return compute_binary_pos_weight(pos, neg)

    def create_weighted_sampler(self) -> "WeightedRandomSampler":
        """Create a WeightedRandomSampler for handling class imbalance in the DataLoader."""
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(self.samples)
        class_weight = {
            label: total / (len(label_counts) * count)
            for label, count in label_counts.items()
        }

        sample_weights = [class_weight[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# Data Splitting + Loader Creation
# ---------------------------------------------------------------------------

def create_data_splits(
    dataset: AudioClassificationDataset,
    config: Dict[str, Any],
) -> Tuple["Dataset", "Dataset", Optional["Dataset"]]:
    """
    Split dataset into train/val/test sets.

    Uses stratified splitting to maintain class proportions. Falls back
    to random splitting if sklearn is not available.

    Args:
        dataset: The full dataset
        config: Training configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        test_dataset is None if test_split=0
    """
    val_split = config.get("validation_split", 0.15)
    test_split = config.get("test_split", 0.10)
    stratified = config.get("stratified_split", True)
    seed = config.get("seed", 42)

    n = len(dataset)
    labels = np.array([label for _, label in dataset.samples])

    if stratified and HAS_SKLEARN and len(set(labels)) > 1:
        # Stratified splitting
        indices = np.arange(n)

        # First split: separate test set
        if test_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
            train_val_idx, test_idx = next(sss.split(indices, labels))
        else:
            train_val_idx = indices
            test_idx = np.array([], dtype=int)

        # Second split: separate validation from training
        train_val_labels = labels[train_val_idx]
        relative_val = val_split / (1.0 - test_split) if test_split < 1.0 else val_split
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
        train_idx, val_idx = next(sss2.split(train_val_idx, train_val_labels))
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
    else:
        # Random splitting
        if stratified and not HAS_SKLEARN:
            Log.warning("Stratified split requested but sklearn not available. Using random split.")

        generator = torch.Generator().manual_seed(seed)
        test_n = int(n * test_split)
        val_n = int(n * val_split)
        train_n = n - val_n - test_n

        from torch.utils.data import random_split
        splits = random_split(dataset, [train_n, val_n, test_n], generator=generator)

        test_ds = splits[2] if test_n > 0 else None
        return splits[0], splits[1], test_ds

    # Create subsets
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist()) if len(test_idx) > 0 else None

    Log.info(
        f"Data split: train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds) if test_ds else 0}"
    )

    return train_ds, val_ds, test_ds


def create_data_loaders(
    dataset: AudioClassificationDataset,
    config: Dict[str, Any],
) -> Tuple["DataLoader", "DataLoader", Optional["DataLoader"]]:
    """
    Create train/val/test DataLoaders with proper configuration.

    Applies weighted sampling if configured, uses appropriate batch sizes,
    and sets up parallel data loading.

    Args:
        dataset: The full AudioClassificationDataset
        config: Training configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config.get("batch_size", 32)
    use_weighted = config.get("use_weighted_sampling", False)
    num_workers = config.get("num_workers", 0)
    use_augmentation = config.get("use_augmentation", False)
    seed = config.get("seed", 42)

    # With augmentation on, load+augment+mel run per sample (no cache). Use workers so this runs in parallel with training.
    if use_augmentation and num_workers == 0:
        num_workers = 2
        Log.info(
            "Data augmentation is on: using 2 DataLoader workers so loading and augmentation run in parallel with training. "
            "Set num_workers in Training settings to change (0 = single-threaded, often slow with augmentation)."
        )

    # macOS + Qt background thread + DataLoader workers is a common deadlock/hang path.
    # If execution is not on the main thread, force a safe single-threaded loader mode.
    if sys.platform == "darwin" and num_workers > 0 and threading.current_thread() is not threading.main_thread():
        Log.warning(
            "PyTorch DataLoader workers were disabled for this run because training is executing "
            "from a background thread on macOS. This avoids worker spawn hangs. "
            "Training will be slower but reliable in this mode."
        )
        num_workers = 0

    train_ds, val_ds, test_ds = create_data_splits(dataset, config)

    # Reproducibility: seed workers (module-level fn so DataLoader can pickle it) and shuffle generator
    worker_init_fn = functools.partial(_worker_init_fn, seed=seed) if num_workers > 0 else None

    train_generator = None
    if HAS_PYTORCH and (not use_weighted):
        train_generator = torch.Generator().manual_seed(seed)

    # Train loader with optional weighted sampling
    train_sampler = None
    train_shuffle = True
    if use_weighted:
        train_sampler = dataset.create_weighted_sampler()
        train_shuffle = False  # Sampler handles ordering

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config.get("device", "cpu") != "cpu",
        worker_init_fn=worker_init_fn,
        generator=train_generator,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get("device", "cpu") != "cpu",
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.get("device", "cpu") != "cpu",
        )

    return train_loader, val_loader, test_loader
