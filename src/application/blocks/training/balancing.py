"""
Dataset Balancing Strategies

Non-destructive dataset balancing for audio classification. Operates on
sample manifests (lists of file paths + labels) without modifying or
copying original files.

Works for both configurations:
  - Binary: exactly two labels (0=negative, 1=positive). Strategies balance
    the positive vs negative counts (e.g. undersample_min caps both to the
    smaller count; oversample_max brings the smaller to the larger).
  - Multiclass: N labels (0..N-1). Strategies balance across all N classes
    (e.g. undersample_median caps each class to the median count).

Strategies:
    - none: No balancing, use all samples as-is
    - undersample_min: Cap every class at the smallest class count
    - undersample_median: Cap every class at the median class count
    - undersample_target: Cap every class at a user-specified count
    - oversample_max: Duplicate minority samples to match the largest class
    - oversample_target: Duplicate minority samples to reach a target count
    - smart_undersample: Cluster-based selection that preserves diversity
    - hybrid: Smart undersample majority + augmentation-flagged oversample minority
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

from src.utils.message import Log


# Type alias: (file_path, label)
Sample = Tuple[Path, int]


@dataclass
class BalanceResult:
    """Result of a balancing operation with before/after statistics."""

    samples: List[Sample]
    strategy: str
    original_distribution: Dict[int, int]
    balanced_distribution: Dict[int, int]
    augment_flags: List[bool]  # True = sample should receive extra augmentation

    @property
    def total_original(self) -> int:
        return sum(self.original_distribution.values())

    @property
    def total_balanced(self) -> int:
        return sum(self.balanced_distribution.values())

    def summary(self) -> str:
        lines = [f"Balance strategy: {self.strategy}"]
        lines.append(f"  Before: {self.total_original} samples")
        for label, count in sorted(self.original_distribution.items()):
            lines.append(f"    class {label}: {count}")
        lines.append(f"  After:  {self.total_balanced} samples")
        for label, count in sorted(self.balanced_distribution.items()):
            lines.append(f"    class {label}: {count}")
        aug_count = sum(1 for f in self.augment_flags if f)
        if aug_count > 0:
            lines.append(f"  Augmentation-flagged: {aug_count} samples")
        return "\n".join(lines)


def _class_distribution(samples: List[Sample]) -> Dict[int, int]:
    """Count samples per class label."""
    dist: Dict[int, int] = {}
    for _, label in samples:
        dist[label] = dist.get(label, 0) + 1
    return dist


def _samples_by_class(samples: List[Sample]) -> Dict[int, List[Sample]]:
    """Group samples by class label."""
    groups: Dict[int, List[Sample]] = {}
    for sample in samples:
        label = sample[1]
        if label not in groups:
            groups[label] = []
        groups[label].append(sample)
    return groups


def balance_dataset(
    samples: List[Sample],
    strategy: str = "none",
    target_count: Optional[int] = None,
    seed: int = 42,
) -> BalanceResult:
    """
    Apply a balancing strategy to a sample manifest.

    Works for both binary (labels 0 and 1) and multiclass (labels 0..N-1).
    In binary mode, label 0 is typically "negative" and 1 "positive"; the
    same strategies apply (e.g. undersample_min equalizes the two sides).

    Non-destructive: returns a new list of (path, label) tuples without
    modifying the original files. Oversampled entries are duplicates of
    existing paths (flagged for augmentation so the model sees unique
    variations, not identical copies).

    Args:
        samples: Original sample list [(file_path, label), ...]
        strategy: Balancing strategy name (see module docstring)
        target_count: Target count per class (for target-based strategies)
        seed: Random seed for reproducibility

    Returns:
        BalanceResult with balanced samples and statistics
    """
    rng = np.random.RandomState(seed)
    original_dist = _class_distribution(samples)
    groups = _samples_by_class(samples)

    if strategy == "none" or not samples:
        return BalanceResult(
            samples=list(samples),
            strategy="none",
            original_distribution=original_dist,
            balanced_distribution=_class_distribution(samples),
            augment_flags=[False] * len(samples),
        )

    if strategy == "undersample_min":
        target = min(original_dist.values())
        return _undersample_random(groups, target, original_dist, strategy, rng)

    if strategy == "undersample_median":
        counts = sorted(original_dist.values())
        target = int(np.median(counts))
        return _undersample_random(groups, target, original_dist, strategy, rng)

    if strategy == "undersample_target":
        if target_count is None:
            raise ValueError("target_count is required for undersample_target strategy")
        return _undersample_random(groups, target_count, original_dist, strategy, rng)

    if strategy == "oversample_max":
        target = max(original_dist.values())
        return _oversample(groups, target, original_dist, strategy, rng)

    if strategy == "oversample_target":
        if target_count is None:
            raise ValueError("target_count is required for oversample_target strategy")
        return _oversample(groups, target_count, original_dist, strategy, rng)

    if strategy == "smart_undersample":
        target = min(original_dist.values())
        return _smart_undersample(groups, target, original_dist, strategy, rng)

    if strategy == "hybrid":
        return _hybrid_balance(groups, original_dist, rng, target_count)

    raise ValueError(
        f"Unknown balancing strategy: '{strategy}'. "
        f"Valid options: none, undersample_min, undersample_median, "
        f"undersample_target, oversample_max, oversample_target, "
        f"smart_undersample, hybrid"
    )


def _undersample_random(
    groups: Dict[int, List[Sample]],
    target: int,
    original_dist: Dict[int, int],
    strategy: str,
    rng: np.random.RandomState,
) -> BalanceResult:
    """Random undersampling: pick target samples per class (or keep all if below target)."""
    balanced: List[Sample] = []
    for label, class_samples in sorted(groups.items()):
        if len(class_samples) > target:
            indices = rng.choice(len(class_samples), size=target, replace=False)
            selected = [class_samples[i] for i in sorted(indices)]
        else:
            selected = list(class_samples)
        balanced.extend(selected)

    return BalanceResult(
        samples=balanced,
        strategy=strategy,
        original_distribution=original_dist,
        balanced_distribution=_class_distribution(balanced),
        augment_flags=[False] * len(balanced),
    )


def _oversample(
    groups: Dict[int, List[Sample]],
    target: int,
    original_dist: Dict[int, int],
    strategy: str,
    rng: np.random.RandomState,
) -> BalanceResult:
    """Oversample minority classes by duplicating samples (flagged for augmentation)."""
    balanced: List[Sample] = []
    augment_flags: List[bool] = []

    for label, class_samples in sorted(groups.items()):
        # Add all originals
        balanced.extend(class_samples)
        augment_flags.extend([False] * len(class_samples))

        # Duplicate if below target
        deficit = target - len(class_samples)
        if deficit > 0:
            # Cycle through existing samples
            extras = [class_samples[i % len(class_samples)] for i in range(deficit)]
            balanced.extend(extras)
            # Flag all duplicates for extra augmentation
            augment_flags.extend([True] * deficit)

    return BalanceResult(
        samples=balanced,
        strategy=strategy,
        original_distribution=original_dist,
        balanced_distribution=_class_distribution(balanced),
        augment_flags=augment_flags,
    )


def _smart_undersample(
    groups: Dict[int, List[Sample]],
    target: int,
    original_dist: Dict[int, int],
    strategy: str,
    rng: np.random.RandomState,
) -> BalanceResult:
    """
    Cluster-based undersampling that preserves diversity.

    For classes above the target count, loads lightweight audio features
    (RMS energy, zero-crossing rate, spectral centroid), clusters with
    k-means, and picks the sample closest to each centroid. This ensures
    the selected subset covers the full range of variation in the class.

    Falls back to random undersampling if feature extraction fails.
    """
    balanced: List[Sample] = []

    for label, class_samples in sorted(groups.items()):
        if len(class_samples) <= target:
            balanced.extend(class_samples)
            continue

        # Try cluster-based selection
        try:
            selected = _cluster_select(class_samples, target, rng)
            balanced.extend(selected)
            Log.info(
                f"Smart undersample class {label}: {len(class_samples)} -> {len(selected)} "
                f"(cluster-based)"
            )
        except Exception as e:
            Log.warning(
                f"Smart undersample fallback for class {label}: {e}. Using random selection."
            )
            indices = rng.choice(len(class_samples), size=target, replace=False)
            balanced.extend([class_samples[i] for i in sorted(indices)])

    return BalanceResult(
        samples=balanced,
        strategy=strategy,
        original_distribution=original_dist,
        balanced_distribution=_class_distribution(balanced),
        augment_flags=[False] * len(balanced),
    )


def _cluster_select(
    samples: List[Sample],
    n_select: int,
    rng: np.random.RandomState,
) -> List[Sample]:
    """
    Select n_select samples using k-means clustering on lightweight features.

    Features extracted per sample (fast, no GPU needed):
    - RMS energy (overall loudness)
    - Zero-crossing rate (noisiness)
    - Spectral centroid (brightness)
    - Duration/length

    These four features are enough to capture meaningful diversity
    in audio samples without loading full spectrograms.
    """
    import librosa
    from sklearn.cluster import MiniBatchKMeans

    features = []
    valid_indices = []

    for i, (path, _) in enumerate(samples):
        try:
            # Load a short clip (fast), mono, native sr
            y, sr = librosa.load(str(path), sr=None, mono=True, duration=2.0)
            if len(y) < 512:
                continue

            rms = float(np.sqrt(np.mean(y ** 2)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            length = len(y) / sr

            features.append([rms, zcr, centroid, length])
            valid_indices.append(i)
        except Exception:
            continue

    if len(features) < n_select:
        # Not enough valid features; fall back to random
        indices = rng.choice(len(samples), size=n_select, replace=False)
        return [samples[i] for i in sorted(indices)]

    features_arr = np.array(features, dtype=np.float32)

    # Normalize features to [0, 1] range
    mins = features_arr.min(axis=0)
    maxs = features_arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    features_norm = (features_arr - mins) / ranges

    # Cluster
    n_clusters = min(n_select, len(features_norm))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(rng.randint(0, 2**31)),
        n_init=3,
        batch_size=min(256, len(features_norm)),
    )
    kmeans.fit(features_norm)

    # Pick the sample closest to each centroid
    selected_indices = set()
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(features_norm - centroid, axis=1)
        # Mask already-selected to avoid duplicates
        for idx in selected_indices:
            pos = valid_indices.index(idx) if idx in valid_indices else -1
            if pos >= 0:
                distances[pos] = np.inf
        closest = np.argmin(distances)
        selected_indices.add(valid_indices[closest])

    # If we need more (rounding), fill with random unselected
    while len(selected_indices) < n_select:
        remaining = [i for i in valid_indices if i not in selected_indices]
        if not remaining:
            break
        pick = rng.choice(remaining)
        selected_indices.add(pick)

    return [samples[i] for i in sorted(selected_indices)]


def _hybrid_balance(
    groups: Dict[int, List[Sample]],
    original_dist: Dict[int, int],
    rng: np.random.RandomState,
    target_count: Optional[int] = None,
) -> BalanceResult:
    """
    Hybrid strategy: smart undersample majority classes, oversample minority.

    Target is the median class count (or user-specified target_count).
    - Classes above target: smart undersampled for diversity
    - Classes at target: kept as-is
    - Classes below target: oversampled with augmentation flags
    """
    counts = sorted(original_dist.values())
    target = target_count if target_count is not None else int(np.median(counts))

    balanced: List[Sample] = []
    augment_flags: List[bool] = []

    for label, class_samples in sorted(groups.items()):
        if len(class_samples) > target:
            # Smart undersample
            try:
                selected = _cluster_select(class_samples, target, rng)
            except Exception:
                indices = rng.choice(len(class_samples), size=target, replace=False)
                selected = [class_samples[i] for i in sorted(indices)]
            balanced.extend(selected)
            augment_flags.extend([False] * len(selected))
        elif len(class_samples) < target:
            # Keep originals + oversample with augmentation flag
            balanced.extend(class_samples)
            augment_flags.extend([False] * len(class_samples))
            deficit = target - len(class_samples)
            extras = [class_samples[i % len(class_samples)] for i in range(deficit)]
            balanced.extend(extras)
            augment_flags.extend([True] * deficit)
        else:
            balanced.extend(class_samples)
            augment_flags.extend([False] * len(class_samples))

    return BalanceResult(
        samples=balanced,
        strategy="hybrid",
        original_distribution=original_dist,
        balanced_distribution=_class_distribution(balanced),
        augment_flags=augment_flags,
    )


def preview_balance(
    samples: List[Sample],
    strategy: str,
    target_count: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Preview what a balancing strategy would do without actually loading audio.

    Works for both binary (2 classes) and multiclass (N classes). Returns a
    dictionary with before/after counts suitable for UI display. For strategies
    that require audio loading (smart_undersample, hybrid), shows the expected
    counts without performing the actual clustering.

    Args:
        samples: Current sample manifest [(path, label), ...]
        strategy: Strategy name
        target_count: Target count (for target-based strategies)
        class_names: Optional mapping of label -> class name for display
        seed: Random seed

    Returns:
        Dictionary with 'before', 'after', 'strategy', 'changes' keys
    """
    original_dist = _class_distribution(samples)
    counts = sorted(original_dist.values())

    # Calculate target for each strategy
    if strategy == "none":
        target = None
    elif strategy == "undersample_min":
        target = min(counts) if counts else 0
    elif strategy == "undersample_median":
        target = int(np.median(counts)) if counts else 0
    elif strategy == "undersample_target":
        target = target_count
    elif strategy == "oversample_max":
        target = max(counts) if counts else 0
    elif strategy == "oversample_target":
        target = target_count
    elif strategy in ("smart_undersample",):
        target = min(counts) if counts else 0
    elif strategy == "hybrid":
        target = target_count if target_count else (int(np.median(counts)) if counts else 0)
    else:
        target = None

    # Build before/after
    before = {}
    after = {}
    changes = {}

    for label, count in sorted(original_dist.items()):
        name = class_names.get(label, str(label)) if class_names else str(label)
        before[name] = count

        if strategy == "none" or target is None:
            after[name] = count
            changes[name] = 0
        elif strategy in ("undersample_min", "undersample_median", "undersample_target", "smart_undersample"):
            new_count = min(count, target)
            after[name] = new_count
            changes[name] = new_count - count
        elif strategy in ("oversample_max", "oversample_target"):
            new_count = max(count, target)
            after[name] = new_count
            changes[name] = new_count - count
        elif strategy == "hybrid":
            if count > target:
                after[name] = target
            elif count < target:
                after[name] = target
            else:
                after[name] = count
            changes[name] = after[name] - count

    return {
        "strategy": strategy,
        "target_per_class": target,
        "before": before,
        "after": after,
        "changes": changes,
        "total_before": sum(before.values()),
        "total_after": sum(after.values()),
    }


# Available strategies for UI population
BALANCE_STRATEGIES = {
    "none": "No Balancing",
    "undersample_min": "Undersample to Smallest Class",
    "undersample_median": "Undersample to Median Class",
    "undersample_target": "Undersample to Target Count",
    "oversample_max": "Oversample to Largest Class",
    "oversample_target": "Oversample to Target Count",
    "smart_undersample": "Smart Undersample (Cluster-Based)",
    "hybrid": "Hybrid (Smart Under + Augmented Over)",
}
