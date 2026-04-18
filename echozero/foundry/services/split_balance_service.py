from __future__ import annotations

import hashlib
import json
import random

from echozero.foundry.domain import DatasetSample, DatasetVersion


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
        grouped_samples: dict[str, list[DatasetSample]] = {}
        class_counts: dict[str, int] = {}
        for sample in version.samples:
            group_key = self.resolve_group_id(sample)
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
        group_distribution = self._build_group_distribution(version, assignments)
        reproducibility = self._build_reproducibility_report(version, assignments, seed=seed)

        return {
            "validation_split": validation_split,
            "test_split": test_split,
            "seed": seed,
            "planner": "content_hash_grouped_v1",
            "policy": "grouped_anti_leakage_v2",
            "grouping_strategy": "source_provenance.group_id|synthetic_provenance.source_sample_id|content_hash|sample_id",
            "dataset_version_id": version.id,
            "dataset_manifest_hash": version.manifest_hash,
            "dataset_sample_count": len(version.samples),
            "train_ids": sorted(train),
            "val_ids": sorted(val),
            "test_ids": sorted(test),
            "assignments": assignments,
            "label_distribution": label_distribution,
            "group_distribution": group_distribution,
            "content_hash_groups": {
                group_key: sorted(sample.sample_id for sample in samples)
                for group_key, samples in grouped_samples.items()
            },
            "reproducibility": reproducibility,
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

        if not class_counts:
            return {
                "strategy": strategy,
                "target_counts": {},
                "deltas": {},
                "class_counts": {},
                "majority_to_minority_ratio": None,
                "is_skewed": False,
                "warnings": [],
                "recommended_training_overrides": {
                    "classWeighting": "none",
                    "rebalanceStrategy": "none",
                },
            }

        ratio = self._majority_to_minority_ratio(class_counts)
        is_skewed = ratio >= 1.5
        warnings: list[str] = []
        recommended_training_overrides = {
            "classWeighting": "balanced" if is_skewed else "none",
            "rebalanceStrategy": "oversample" if ratio >= 2.0 else "none",
        }

        if strategy == "none":
            target_counts = class_counts
            deltas = {k: 0 for k in class_counts}
            if is_skewed:
                warnings.append(
                    "Class distribution is skewed; default plan leaves counts unchanged. "
                    "Consider classWeighting=balanced and rebalanceStrategy=oversample for training."
                )
            return {
                "strategy": "none",
                "target_counts": target_counts,
                "deltas": deltas,
                "class_counts": class_counts,
                "majority_to_minority_ratio": ratio,
                "is_skewed": is_skewed,
                "warnings": warnings,
                "recommended_training_overrides": recommended_training_overrides,
            }

        if strategy in {"undersample_min", "smart_undersample"}:
            target = min(class_counts.values())
        elif strategy in {"oversample_max", "hybrid"}:
            target = max(class_counts.values())
        else:
            raise ValueError(f"Unsupported balance strategy for v1: {strategy}")

        target_counts = {label: target for label in class_counts}
        deltas = {label: target - count for label, count in class_counts.items()}
        if is_skewed:
            warnings.append(
                f"Detected skew ratio {ratio:.2f}:1 across classes; applying requested balance strategy '{strategy}'."
            )
        return {
            "strategy": strategy,
            "target_counts": target_counts,
            "deltas": deltas,
            "class_counts": class_counts,
            "majority_to_minority_ratio": ratio,
            "is_skewed": is_skewed,
            "warnings": warnings,
            "recommended_training_overrides": recommended_training_overrides,
        }

    @classmethod
    def validate_split_plan(cls, version: DatasetVersion, split_plan: dict) -> dict:
        sample_ids = {sample.sample_id for sample in version.samples}
        assignments = split_plan.get("assignments", {})
        errors: list[str] = []
        warnings: list[str] = []

        if split_plan.get("dataset_version_id") not in {None, version.id}:
            errors.append("split plan dataset_version_id does not match dataset version")
        if split_plan.get("dataset_manifest_hash") not in {None, version.manifest_hash}:
            errors.append("split plan dataset_manifest_hash does not match dataset version manifest_hash")
        if split_plan.get("dataset_sample_count") not in {None, len(version.samples)}:
            errors.append("split plan dataset_sample_count does not match dataset version sample count")

        referenced_ids = set(assignments)
        unknown_ids = sorted(referenced_ids - sample_ids)
        missing_ids = sorted(sample_ids - referenced_ids)
        if unknown_ids:
            errors.append("split plan references unknown sample ids")
        if missing_ids:
            errors.append("split plan is missing sample assignments")

        for split_name in ("train", "val", "test"):
            split_ids = split_plan.get(f"{split_name}_ids", [])
            invalid = [sample_id for sample_id in split_ids if assignments.get(sample_id) != split_name]
            if invalid:
                errors.append(f"split plan {split_name}_ids diverge from assignments")

        leakage = cls._build_leakage_report(version, assignments)
        if not leakage["ok"]:
            errors.append("split plan violates grouped anti-leakage policy")

        reproducibility = split_plan.get("reproducibility", {})
        recomputed = cls._build_reproducibility_report(version, assignments, seed=int(split_plan.get("seed", 42)))
        for key in ("group_fingerprint", "assignment_fingerprint"):
            expected = reproducibility.get(key)
            if expected is not None and expected != recomputed[key]:
                errors.append(f"split plan reproducibility.{key} does not match current dataset state")

        if split_plan.get("planner") == "content_hash_grouped_v1" and split_plan.get("policy") is None:
            warnings.append("split plan predates grouped_anti_leakage_v2 metadata")

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "leakage": leakage,
            "reproducibility": recomputed,
        }

    @staticmethod
    def resolve_group_id(sample: DatasetSample) -> str:
        source_group_id = str(sample.source_provenance.get("group_id", "")).strip()
        if source_group_id:
            return source_group_id
        synthetic_source_id = str(sample.synthetic_provenance.get("source_sample_id", "")).strip()
        if synthetic_source_id:
            return f"synthetic:{synthetic_source_id}"
        if sample.content_hash:
            return f"content:{sample.content_hash}"
        return f"sample:{sample.sample_id}"

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

    @classmethod
    def _build_group_distribution(cls, version: DatasetVersion, assignments: dict[str, str]) -> dict[str, int]:
        distribution: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for sample in version.samples:
            split_name = assignments.get(sample.sample_id)
            if split_name not in distribution:
                continue
            distribution[split_name].add(cls.resolve_group_id(sample))
        return {split_name: len(group_ids) for split_name, group_ids in distribution.items()}

    @classmethod
    def _build_leakage_report(cls, version: DatasetVersion, assignments: dict[str, str]) -> dict:
        hash_splits: dict[str, set[str]] = {}
        group_splits: dict[str, set[str]] = {}
        for sample in version.samples:
            split_name = assignments.get(sample.sample_id)
            if split_name is None:
                continue
            if sample.content_hash:
                hash_splits.setdefault(sample.content_hash, set()).add(split_name)
            group_splits.setdefault(cls.resolve_group_id(sample), set()).add(split_name)

        duplicate_hashes = sorted(content_hash for content_hash, splits in hash_splits.items() if len(splits) > 1)
        duplicate_groups = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
        return {
            "guard": "group_id+content_hash",
            "duplicate_hashes_across_splits": duplicate_hashes,
            "duplicate_groups_across_splits": duplicate_groups,
            "ok": not duplicate_hashes and not duplicate_groups,
        }

    @classmethod
    def _build_reproducibility_report(cls, version: DatasetVersion, assignments: dict[str, str], *, seed: int) -> dict:
        grouped_sample_ids: dict[str, list[str]] = {}
        for sample in version.samples:
            group_key = cls.resolve_group_id(sample)
            grouped_sample_ids.setdefault(group_key, []).append(sample.sample_id)

        group_rows = sorted((group_id, sorted(sample_ids)) for group_id, sample_ids in grouped_sample_ids.items())
        group_fingerprint = hashlib.sha256(json.dumps(group_rows, sort_keys=True).encode("utf-8")).hexdigest()
        assignment_rows = sorted(assignments.items())
        assignment_fingerprint = hashlib.sha256(json.dumps(assignment_rows, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "seed": seed,
            "group_count": len(group_rows),
            "group_fingerprint": group_fingerprint,
            "assignment_fingerprint": assignment_fingerprint,
        }

    @staticmethod
    def _should_assign(label_counts: dict[str, int], assigned: dict[str, int], targets: dict[str, int]) -> bool:
        return any(assigned.get(label, 0) < targets.get(label, 0) for label in label_counts)

    @staticmethod
    def _majority_to_minority_ratio(class_counts: dict[str, int]) -> float:
        if not class_counts:
            return 1.0
        counts = [count for count in class_counts.values() if count > 0]
        if not counts:
            return 1.0
        return max(counts) / min(counts)
