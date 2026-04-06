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
        grouped_samples: dict[str, list] = {}
        class_counts: dict[str, int] = {}
        for sample in version.samples:
            group_key = sample.content_hash or sample.sample_id
            grouped_samples.setdefault(group_key, []).append(sample)
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1

        train: list[str] = []
        val: list[str] = []
        test: list[str] = []
        assignments: dict[str, str] = {}
        groups = [(group_key, sorted(samples, key=lambda sample: sample.sample_id)) for group_key, samples in grouped_samples.items()]
        rng.shuffle(groups)
        target_test = {label: int(round(count * test_split)) for label, count in class_counts.items()}
        target_val = {label: int(round(count * validation_split)) for label, count in class_counts.items()}
        assigned_test = {label: 0 for label in class_counts}
        assigned_val = {label: 0 for label in class_counts}

        for index, (_, samples) in enumerate(groups):
            sample_ids = [sample.sample_id for sample in samples]
            label_counts: dict[str, int] = {}
            for sample in samples:
                label_counts[sample.label] = label_counts.get(sample.label, 0) + 1

            remaining_groups = len(groups) - index
            split_name = "train"
            if remaining_groups > 1 and self._should_assign(label_counts, assigned_test, target_test):
                split_name = "test"
                for label, count in label_counts.items():
                    assigned_test[label] += count
            elif remaining_groups > 1 and self._should_assign(label_counts, assigned_val, target_val):
                split_name = "val"
                for label, count in label_counts.items():
                    assigned_val[label] += count

            self._assign_samples(sample_ids, split_name, train=train, val=val, test=test, assignments=assignments)

        leakage = self._build_leakage_report(version, assignments)
        label_distribution = self._build_label_distribution(version, assignments)

        return {
            "validation_split": validation_split,
            "test_split": test_split,
            "seed": seed,
            "planner": "content_hash_grouped_v1",
            "train_ids": sorted(train),
            "val_ids": sorted(val),
            "test_ids": sorted(test),
            "assignments": assignments,
            "label_distribution": label_distribution,
            "content_hash_groups": {
                group_key: sorted(sample.sample_id for sample in samples)
                for group_key, samples in grouped_samples.items()
            },
            "leakage": leakage,
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

    @staticmethod
    def _assign_samples(
        sample_ids: list[str],
        split_name: str,
        *,
        train: list[str],
        val: list[str],
        test: list[str],
        assignments: dict[str, str],
    ) -> None:
        target = train if split_name == "train" else val if split_name == "val" else test
        target.extend(sample_ids)
        for sample_id in sample_ids:
            assignments[sample_id] = split_name

    @staticmethod
    def _build_label_distribution(version: DatasetVersion, assignments: dict[str, str]) -> dict:
        distribution: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        for sample in version.samples:
            split_name = assignments.get(sample.sample_id, "unassigned")
            if split_name not in distribution:
                distribution[split_name] = {}
            distribution[split_name][sample.label] = distribution[split_name].get(sample.label, 0) + 1
        return distribution

    @staticmethod
    def _build_leakage_report(version: DatasetVersion, assignments: dict[str, str]) -> dict:
        hash_splits: dict[str, set[str]] = {}
        for sample in version.samples:
            if not sample.content_hash:
                continue
            split_name = assignments.get(sample.sample_id)
            if split_name is None:
                continue
            hash_splits.setdefault(sample.content_hash, set()).add(split_name)

        duplicate_hashes = sorted(content_hash for content_hash, splits in hash_splits.items() if len(splits) > 1)
        return {
            "guard": "content_hash",
            "duplicate_hashes_across_splits": duplicate_hashes,
            "ok": not duplicate_hashes,
        }

    @staticmethod
    def _should_assign(label_counts: dict[str, int], assigned: dict[str, int], targets: dict[str, int]) -> bool:
        return any(assigned.get(label, 0) < targets.get(label, 0) for label in label_counts)
