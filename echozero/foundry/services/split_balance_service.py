from __future__ import annotations

import random

from echozero.foundry.domain import DatasetVersion


class SplitBalanceService:
    def plan_splits(
        self,
        version: DatasetVersion,
        *,
        validation_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
    ) -> dict:
        if validation_split < 0 or test_split < 0 or (validation_split + test_split) >= 1.0:
            raise ValueError("Invalid split percentages")

        rng = random.Random(seed)
        by_label: dict[str, list[str]] = {}
        for sample in version.samples:
            by_label.setdefault(sample.label, []).append(sample.sample_id)

        train: list[str] = []
        val: list[str] = []
        test: list[str] = []

        for label, sample_ids in by_label.items():
            ids = list(sample_ids)
            rng.shuffle(ids)
            n = len(ids)
            n_test = int(round(n * test_split))
            n_val = int(round(n * validation_split))

            test.extend(ids[:n_test])
            val.extend(ids[n_test:n_test + n_val])
            train.extend(ids[n_test + n_val:])

        return {
            "validation_split": validation_split,
            "test_split": test_split,
            "seed": seed,
            "train_ids": sorted(train),
            "val_ids": sorted(val),
            "test_ids": sorted(test),
        }

    def plan_balance(
        self,
        version: DatasetVersion,
        *,
        strategy: str = "none",
    ) -> dict:
        class_counts: dict[str, int] = {}
        for sample in version.samples:
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1

        if strategy == "none":
            return {
                "strategy": "none",
                "target_counts": class_counts,
                "deltas": {k: 0 for k in class_counts},
            }

        if not class_counts:
            return {"strategy": strategy, "target_counts": {}, "deltas": {}}

        if strategy in {"undersample_min", "smart_undersample"}:
            target = min(class_counts.values())
        elif strategy in {"oversample_max", "hybrid"}:
            target = max(class_counts.values())
        else:
            raise ValueError(f"Unsupported balance strategy for v1: {strategy}")

        target_counts = {label: target for label in class_counts}
        deltas = {label: target - count for label, count in class_counts.items()}
        return {
            "strategy": strategy,
            "target_counts": target_counts,
            "deltas": deltas,
        }
