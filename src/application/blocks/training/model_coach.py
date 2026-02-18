"""
Model coach helpers for beginner-friendly training/inference feedback.

This module converts raw model metrics into plain-language findings and
recommended next actions. It also stores a compact per-run history (JSONL)
to provide basic trend feedback between runs.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.message import Log
from src.utils.paths import get_logs_dir


_COACH_HISTORY_FILE = "model_coach_training_history.jsonl"


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_path() -> Path:
    return get_logs_dir() / _COACH_HISTORY_FILE


def _read_history_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception as e:
        Log.warning(f"Model coach: failed to read history file: {e}")
    return rows


def _append_history(entry: Dict[str, Any]) -> None:
    path = _history_path()
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception as e:
        Log.warning(f"Model coach: failed to append history: {e}")


def _compute_trend(block_id: str, current_accuracy: Optional[float]) -> Dict[str, Any]:
    if current_accuracy is None:
        return {"label": "unknown", "message": "No comparable accuracy metric for trend analysis."}

    rows = _read_history_lines(_history_path())
    previous = None
    for row in reversed(rows):
        if row.get("block_id") == block_id:
            prev_acc = _to_float(row.get("primary_accuracy"))
            if prev_acc is not None:
                previous = prev_acc
                break

    if previous is None:
        return {"label": "baseline", "message": "Baseline run captured. Run again to see trend."}

    delta = current_accuracy - previous
    if delta >= 2.0:
        label = "improving"
    elif delta <= -2.0:
        label = "regressing"
    else:
        label = "stable"
    return {
        "label": label,
        "delta_accuracy": float(delta),
        "message": f"Accuracy trend vs previous run: {delta:+.2f} points.",
    }


def build_training_feedback(
    *,
    block_id: str,
    model_path: str,
    config: Dict[str, Any],
    training_stats: Dict[str, Any],
    dataset_stats: Dict[str, Any],
    validation_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    threshold_metrics: Optional[Dict[str, Any]],
    excluded_bad_file_count: int = 0,
) -> Dict[str, Any]:
    """
    Build plain-language training feedback and persist a history row.
    """
    findings: List[str] = []
    actions: List[str] = []
    score = 100

    primary_metrics = test_metrics or validation_metrics or {}
    primary_accuracy = _to_float(primary_metrics.get("accuracy"))
    if primary_accuracy is None:
        primary_accuracy = _to_float(training_stats.get("val_accuracy", [None])[-1] if training_stats.get("val_accuracy") else None)

    train_acc_last = _to_float(training_stats.get("train_accuracy", [None])[-1] if training_stats.get("train_accuracy") else None)
    val_acc_last = _to_float(training_stats.get("val_accuracy", [None])[-1] if training_stats.get("val_accuracy") else None)
    overfit_gap = None
    if train_acc_last is not None and val_acc_last is not None:
        overfit_gap = train_acc_last - val_acc_last

    total_samples = int(dataset_stats.get("total_samples") or 0)
    class_distribution = dataset_stats.get("class_distribution") or {}
    imbalance_ratio = None
    if isinstance(class_distribution, dict) and class_distribution:
        values = [int(v) for v in class_distribution.values() if int(v) > 0]
        if values:
            imbalance_ratio = max(values) / max(min(values), 1)

    if primary_accuracy is None:
        score -= 25
        findings.append("No primary accuracy metric was available for this run.")
        actions.append("Enable validation/test split so model quality can be measured.")
    elif primary_accuracy < 70:
        score -= 35
        findings.append(f"Primary accuracy is low ({primary_accuracy:.1f}%).")
        actions.append("Collect more clean labeled samples before tuning hyperparameters.")
    elif primary_accuracy < 80:
        score -= 20
        findings.append(f"Primary accuracy is moderate ({primary_accuracy:.1f}%).")
        actions.append("Try more epochs and verify class balance for weak classes.")
    else:
        findings.append(f"Primary accuracy is strong ({primary_accuracy:.1f}%).")

    if overfit_gap is not None:
        if overfit_gap >= 12.0:
            score -= 25
            findings.append(f"Overfitting detected (train/val gap {overfit_gap:.1f} points).")
            actions.append("Reduce model complexity or increase regularization (dropout/weight_decay).")
        elif overfit_gap >= 8.0:
            score -= 15
            findings.append(f"Potential overfitting (train/val gap {overfit_gap:.1f} points).")
            actions.append("Use stronger augmentation or earlier stopping to improve generalization.")

    if imbalance_ratio is not None:
        if imbalance_ratio >= 4.0:
            score -= 18
            findings.append(f"Severe class imbalance detected (largest class is {imbalance_ratio:.1f}x smallest).")
            actions.append("Balance dataset (or adjust class weighting) before the next run.")
        elif imbalance_ratio >= 2.0:
            score -= 8
            findings.append(f"Moderate class imbalance detected ({imbalance_ratio:.1f}x).")

    if excluded_bad_file_count > 0:
        penalty = 12 if excluded_bad_file_count >= 20 else 6
        score -= penalty
        findings.append(f"{excluded_bad_file_count} bad files were excluded during dataset preparation.")
        actions.append("Audit excluded files and replace/remove corrupted samples.")

    binary_f1 = _to_float(primary_metrics.get("f1"))
    if binary_f1 is not None and binary_f1 < 0.75:
        score -= 15
        findings.append(f"Binary F1 is low ({binary_f1:.2f}), indicating weak precision/recall balance.")
        actions.append("Tune threshold and inspect false positives/false negatives.")

    tuned_threshold = _to_float((threshold_metrics or {}).get("threshold"))
    if tuned_threshold is not None:
        findings.append(f"Auto-tuned decision threshold: {tuned_threshold:.3f}.")

    score = max(0, min(100, score))
    if score >= 80:
        verdict = "good"
    elif score >= 60:
        verdict = "needs_work"
    else:
        verdict = "unreliable"

    unique_actions: List[str] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    if not unique_actions:
        unique_actions = [
            "Run inference on a holdout clip set and check confidence consistency.",
            "Promote this model if it matches your real-world use-case quality.",
        ]

    trend = _compute_trend(block_id, primary_accuracy)

    feedback = {
        "verdict": verdict,
        "score": score,
        "summary": (
            f"Model quality verdict: {verdict.replace('_', ' ')} "
            f"(score {score}/100)."
        ),
        "findings": findings[:6],
        "next_actions": unique_actions[:3],
        "key_metrics": {
            "primary_accuracy": primary_accuracy,
            "train_accuracy_last": train_acc_last,
            "val_accuracy_last": val_acc_last,
            "overfit_gap": overfit_gap,
            "total_samples": total_samples,
            "excluded_bad_file_count": int(excluded_bad_file_count),
            "class_imbalance_ratio": imbalance_ratio,
            "binary_f1": binary_f1,
        },
        "trend": trend,
        "generated_at": datetime.now().isoformat(),
    }

    _append_history(
        {
            "timestamp": feedback["generated_at"],
            "block_id": block_id,
            "model_path": model_path,
            "verdict": verdict,
            "score": score,
            "primary_accuracy": primary_accuracy,
            "overfit_gap": overfit_gap,
            "total_samples": total_samples,
        }
    )

    return feedback


def build_inference_feedback(
    *,
    execution_summary: Dict[str, Any],
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build plain-language inference feedback from classifier execution summary.
    """
    total_events = int(execution_summary.get("total_events_input") or 0)
    total_classified = int(execution_summary.get("total_classified") or 0)
    skipped = int(execution_summary.get("total_skipped") or 0)
    events_per_second = _to_float(execution_summary.get("events_per_second"))
    confidence_stats = execution_summary.get("confidence_stats") or {}
    avg_conf = _to_float(confidence_stats.get("avg"))

    low_conf_cutoff = 0.60
    if confidence_threshold is not None:
        low_conf_cutoff = min(0.80, max(0.50, float(confidence_threshold) - 0.10))

    findings: List[str] = []
    actions: List[str] = []
    score = 100

    if total_events <= 0:
        score -= 40
        findings.append("No events were provided to classify.")
        actions.append("Run DetectOnsets/Editor first so classifier has events to process.")

    if skipped > 0 and total_events > 0:
        skipped_pct = 100.0 * skipped / max(total_events, 1)
        if skipped_pct >= 20.0:
            score -= 20
            findings.append(f"A large portion of events were skipped ({skipped_pct:.1f}%).")
            actions.append("Check event clip metadata (clip_start_time/clip_end_time) and source audio links.")
        else:
            findings.append(f"Some events were skipped ({skipped_pct:.1f}%).")

    if avg_conf is None:
        score -= 10
        findings.append("Average confidence was unavailable.")
    elif avg_conf < low_conf_cutoff:
        score -= 25
        findings.append(f"Average confidence is low ({avg_conf:.2f}).")
        actions.append("Improve training data quality or retrain with more representative samples.")
    elif avg_conf < 0.75:
        score -= 10
        findings.append(f"Average confidence is moderate ({avg_conf:.2f}).")
        actions.append("Review borderline predictions and consider threshold tuning.")
    else:
        findings.append(f"Average confidence is strong ({avg_conf:.2f}).")

    if events_per_second is not None and events_per_second < 20:
        findings.append(f"Inference throughput is low ({events_per_second:.1f} events/sec).")
        actions.append("Increase classifier batch size if memory allows.")

    score = max(0, min(100, score))
    if score >= 80:
        verdict = "good"
    elif score >= 60:
        verdict = "needs_work"
    else:
        verdict = "unreliable"

    unique_actions: List[str] = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)
    if not unique_actions:
        unique_actions = ["Use this model on a larger holdout set to confirm stability."]

    return {
        "verdict": verdict,
        "score": score,
        "summary": f"Inference quality verdict: {verdict.replace('_', ' ')} (score {score}/100).",
        "findings": findings[:5],
        "next_actions": unique_actions[:3],
        "key_metrics": {
            "total_events": total_events,
            "classified_events": total_classified,
            "skipped_events": skipped,
            "events_per_second": events_per_second,
            "avg_confidence": avg_conf,
        },
        "generated_at": datetime.now().isoformat(),
    }
