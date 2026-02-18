"""
EchoZero Production Training Pipeline Benchmark

Tests the ACTUAL production path through EchoZero's block processor system:
  Block entity -> PyTorchAudioTrainerBlockProcessor.process() -> full pipeline

This is NOT a unit test of individual components. It exercises the real code path
that runs when a user clicks "Train" in the GUI, with timing instrumentation at
each stage to identify correctness issues and performance regressions.

Run from the repo root:
    python tests/benchmarks/benchmark_ez_training_pipeline.py \
        --data-dir /path/to/OrginizedSamples

Compare results against benchmark_training_bottleneck.py to see if the EZ
orchestration layer adds overhead vs. raw dataset/engine usage.
"""
import argparse
import os
import sys
import time
import tempfile
import threading
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def check_environment():
    """Print environment info."""
    _section("ENVIRONMENT")

    print(f"Platform:       {sys.platform}")
    print(f"Python:         {sys.version.split()[0]}")
    print(f"Main thread:    {threading.current_thread() is threading.main_thread()}")

    import torch
    print(f"PyTorch:        {torch.__version__}")

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available:  {mps_available}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU cores:      {os.cpu_count() or 1}")


def build_block(metadata: dict) -> "Block":
    """Construct a real Block entity with the given metadata."""
    from src.features.blocks.domain.block import Block
    return Block(
        id="bench-trainer-001",
        project_id="bench-project-001",
        name="BenchmarkTrainer",
        type="PyTorchAudioTrainer",
        metadata=metadata,
    )


def benchmark_ez_pipeline(
    data_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "auto",
):
    """
    Run the full EZ training pipeline through the block processor and time every stage.
    """
    _section("EZ PIPELINE: Full Block Processor Execution")

    from src.application.blocks.pytorch_audio_trainer_block import (
        PyTorchAudioTrainerBlockProcessor,
    )

    with tempfile.TemporaryDirectory(prefix="ez_bench_model_") as tmp_dir:
        metadata = {
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "num_workers": 0,
            "use_feature_cache": True,
            "use_augmentation": False,
            "use_early_stopping": False,
            "use_ema": False,
            "use_swa": False,
            "use_tensorboard": False,
            "export_onnx": False,
            "export_quantized": False,
            "use_class_weights": True,
            "normalize_per_dataset": True,
            "model_type": "cnn",
            "output_model_path": os.path.join(tmp_dir, "bench_model.pth"),
            "classification_mode": "multiclass",
        }

        block = build_block(metadata)
        processor = PyTorchAudioTrainerBlockProcessor()

        print(f"Block ID:       {block.id}")
        print(f"Block type:     {block.type}")
        print(f"Epochs:         {epochs}")
        print(f"Batch size:     {batch_size}")
        print(f"Device:         {device}")
        print(f"num_workers:    0 (simulating GUI / QThread constraint)")
        print(f"Output dir:     {tmp_dir}")

        t_start = time.perf_counter()
        try:
            result = processor.process(block, inputs={}, metadata={})
            t_end = time.perf_counter()
        except Exception as e:
            t_end = time.perf_counter()
            print(f"\n  FAILED after {t_end - t_start:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return None

        total_time = t_end - t_start
        print(f"\n  Total pipeline time: {total_time:.2f}s")

        output_item = result.get("model")
        if output_item:
            print(f"  Output item:   {output_item.name}")
            print(f"  Model path:    {output_item.file_path}")

            last_training = block.metadata.get("last_training", {})
            stats = last_training.get("stats", {})
            best_acc = last_training.get("best_accuracy")
            classes = last_training.get("classes", [])

            print(f"  Classes:       {len(classes)} ({', '.join(classes[:5])}{'...' if len(classes) > 5 else ''})")
            if best_acc is not None:
                print(f"  Best val acc:  {best_acc:.2f}%")

            coach = last_training.get("coach_feedback")
            if coach:
                print(f"  Coach grade:   {coach.get('grade', 'N/A')}")

        return {"total_time": total_time, "result": result, "block": block}


def benchmark_ez_pipeline_staged(
    data_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "auto",
):
    """
    Run the EZ pipeline but instrument each internal stage separately.

    This manually replicates the steps in PyTorchAudioTrainerBlockProcessor.process()
    so we can measure each stage's wall time. The code path is identical to production.
    """
    _section("EZ PIPELINE: Staged Timing Breakdown")

    from src.application.blocks.training import (
        TrainingConfig,
        AudioClassificationDataset,
        create_data_loaders,
        create_classifier,
        create_loss_function,
        TrainingEngine,
        evaluate_model,
        save_final_model,
        seed_everything,
    )
    import torch

    timings = {}

    with tempfile.TemporaryDirectory(prefix="ez_bench_staged_") as tmp_dir:
        metadata = {
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "num_workers": 0,
            "use_feature_cache": True,
            "use_augmentation": False,
            "use_early_stopping": False,
            "use_ema": False,
            "use_swa": False,
            "use_tensorboard": False,
            "export_onnx": False,
            "export_quantized": False,
            "use_class_weights": True,
            "normalize_per_dataset": True,
            "model_type": "cnn",
            "output_model_path": os.path.join(tmp_dir, "bench_staged_model.pth"),
            "classification_mode": "multiclass",
        }

        # --- Stage 1: Config ---
        t0 = time.perf_counter()
        config = TrainingConfig.from_block_metadata(metadata)
        timings["1_config"] = time.perf_counter() - t0

        seed_everything(config.seed, deterministic=config.deterministic_training)

        # Resolve device
        requested = (config.device or "auto").strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                runtime_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                runtime_device = torch.device("mps")
            else:
                runtime_device = torch.device("cpu")
        else:
            runtime_device = torch.device(requested)

        print(f"Device resolved: {runtime_device}")

        # --- Stage 2: Dataset creation (includes preloading) ---
        t0 = time.perf_counter()
        config_dict = config.to_dict()
        dataset = AudioClassificationDataset(config_dict)
        timings["2_dataset_init"] = time.perf_counter() - t0

        config_dict["sample_rate"] = int(dataset.sample_rate)
        config_dict["max_length"] = int(dataset.max_length)
        config_dict["n_fft"] = int(dataset.n_fft)
        config_dict["n_mels"] = int(dataset.n_mels)
        config_dict["hop_length"] = int(dataset.hop_length)
        config_dict["fmax"] = int(dataset.fmax)

        print(f"Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        print(f"Features preloaded: {dataset._features_preloaded}")

        # --- Stage 3: Normalization ---
        t0 = time.perf_counter()
        if config.normalize_per_dataset:
            norm_mean, norm_std = dataset.compute_normalization_stats()
        else:
            norm_mean, norm_std = None, None
        timings["3_normalization"] = time.perf_counter() - t0

        # --- Stage 4: DataLoader creation ---
        t0 = time.perf_counter()
        train_loader, val_loader, test_loader = create_data_loaders(dataset, config_dict)
        timings["4_dataloaders"] = time.perf_counter() - t0

        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # --- Stage 5: Model creation ---
        t0 = time.perf_counter()
        num_classes = len(dataset.classes)
        model = create_classifier(num_classes, config_dict)
        model.to(runtime_device)
        timings["5_model_creation"] = time.perf_counter() - t0

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model: {config.model_type} ({param_count:,} params) on {runtime_device}")

        # --- Stage 6: Loss function ---
        t0 = time.perf_counter()
        class_weights = None
        if config.use_class_weights:
            class_weights = dataset.get_class_weights().to(runtime_device)
        criterion = create_loss_function(config_dict, class_weights, None)
        timings["6_loss_setup"] = time.perf_counter() - t0

        # --- Stage 7: Training ---
        t0 = time.perf_counter()
        engine = TrainingEngine(config_dict, device=str(runtime_device))
        result = engine.train(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=dataset.classes,
            progress_tracker=None,
        )
        timings["7_training"] = time.perf_counter() - t0

        # --- Stage 8: Evaluation ---
        t0 = time.perf_counter()
        eval_model = result.ema_model if result.ema_model else result.model
        eval_model.eval()
        eval_config = config_dict.copy()
        eval_config["_classes"] = dataset.classes
        val_metrics = evaluate_model(eval_model, val_loader, eval_config, str(runtime_device))

        test_metrics = None
        if test_loader:
            test_metrics = evaluate_model(eval_model, test_loader, eval_config, str(runtime_device))
        timings["8_evaluation"] = time.perf_counter() - t0

        # --- Stage 9: Save model ---
        t0 = time.perf_counter()
        normalization = None
        if norm_mean is not None:
            normalization = {"mean": norm_mean.tolist(), "std": norm_std.tolist()}

        model_path = save_final_model(
            model=eval_model,
            classes=dataset.classes,
            config=config_dict,
            training_stats=result.stats,
            test_metrics=test_metrics or val_metrics,
            dataset_stats=dataset.stats.to_dict(),
            normalization=normalization,
            optimal_threshold=None,
            ema_model=result.ema_model,
            output_path=config.output_model_path,
        )
        timings["9_save_model"] = time.perf_counter() - t0

    # --- Print timing breakdown ---
    _subsection("TIMING BREAKDOWN")
    total = sum(timings.values())
    for stage, elapsed in timings.items():
        pct = 100 * elapsed / total if total > 0 else 0
        print(f"  {stage:25s}: {elapsed:7.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {total:7.2f}s")

    # Highlight bottleneck
    slowest = max(timings, key=timings.get)
    print(f"\n  Bottleneck: {slowest} ({timings[slowest]:.2f}s / {100 * timings[slowest] / total:.0f}%)")

    if timings["7_training"] / total > 0.7:
        print("  HEALTHY: Training is the dominant stage (as expected)")
    elif timings["2_dataset_init"] / total > 0.4:
        print("  WARNING: Dataset init is dominating -- check cache warming / disk I/O")
    elif timings["3_normalization"] / total > 0.2:
        print("  WARNING: Normalization is slow -- check dataset.compute_normalization_stats()")
    elif timings["8_evaluation"] / total > 0.2:
        print("  WARNING: Evaluation is slow relative to training")

    return timings, result


def benchmark_qthread_simulation(
    data_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "auto",
):
    """
    Simulate the actual GUI execution context: run the block processor
    from a background thread (like QThread) to verify it works.
    """
    _section("EZ PIPELINE: QThread Simulation (Background Thread)")

    from src.application.blocks.pytorch_audio_trainer_block import (
        PyTorchAudioTrainerBlockProcessor,
    )

    result_holder = {"result": None, "error": None, "time": 0}

    with tempfile.TemporaryDirectory(prefix="ez_bench_qthread_") as tmp_dir:
        metadata = {
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "num_workers": 0,
            "use_feature_cache": True,
            "use_augmentation": False,
            "use_early_stopping": False,
            "use_ema": False,
            "use_swa": False,
            "use_tensorboard": False,
            "export_onnx": False,
            "export_quantized": False,
            "use_class_weights": True,
            "normalize_per_dataset": True,
            "model_type": "cnn",
            "output_model_path": os.path.join(tmp_dir, "bench_qthread_model.pth"),
            "classification_mode": "multiclass",
        }

        block = build_block(metadata)
        processor = PyTorchAudioTrainerBlockProcessor()

        def _run_in_thread():
            is_main = threading.current_thread() is threading.main_thread()
            print(f"  Running on main thread: {is_main}")
            t0 = time.perf_counter()
            try:
                result_holder["result"] = processor.process(block, inputs={}, metadata={})
            except Exception as e:
                result_holder["error"] = e
            result_holder["time"] = time.perf_counter() - t0

        thread = threading.Thread(target=_run_in_thread, name="SimQThread")
        thread.start()
        thread.join()

        if result_holder["error"]:
            print(f"  FAILED after {result_holder['time']:.2f}s: {result_holder['error']}")
            import traceback
            traceback.print_exception(
                type(result_holder["error"]),
                result_holder["error"],
                result_holder["error"].__traceback__,
            )
        else:
            print(f"  Completed in {result_holder['time']:.2f}s (background thread)")
            output = result_holder["result"].get("model")
            if output:
                last_training = block.metadata.get("last_training", {})
                best_acc = last_training.get("best_accuracy")
                print(f"  Best val acc:  {best_acc:.2f}%" if best_acc else "  Best val acc: N/A")

    return result_holder


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the real EchoZero training pipeline"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to organized audio samples (class-per-folder layout)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda")
    parser.add_argument(
        "--skip-qthread", action="store_true",
        help="Skip the background thread simulation"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    check_environment()

    # Test 1: Full pipeline through block processor (main thread)
    print("\n" + "=" * 60)
    print("  TEST 1: Block Processor on Main Thread")
    print("=" * 60)
    pipeline_result = benchmark_ez_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Test 2: Staged timing breakdown (main thread)
    print("\n" + "=" * 60)
    print("  TEST 2: Staged Timing Breakdown")
    print("=" * 60)
    staged_timings, staged_result = benchmark_ez_pipeline_staged(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Test 3: Background thread simulation (mimics GUI)
    if not args.skip_qthread:
        print("\n" + "=" * 60)
        print("  TEST 3: Background Thread (QThread Simulation)")
        print("=" * 60)
        qthread_result = benchmark_qthread_simulation(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Summary comparison
    _section("COMPARISON SUMMARY")

    if pipeline_result:
        main_time = pipeline_result["total_time"]
        print(f"  Main thread (block processor):    {main_time:.2f}s")
    if staged_timings:
        staged_total = sum(staged_timings.values())
        training_pct = 100 * staged_timings.get("7_training", 0) / staged_total if staged_total else 0
        print(f"  Staged total:                     {staged_total:.2f}s")
        print(f"  Training stage alone:              {staged_timings.get('7_training', 0):.2f}s ({training_pct:.0f}%)")
        overhead = staged_total - staged_timings.get("7_training", 0)
        print(f"  Non-training overhead:             {overhead:.2f}s ({100 - training_pct:.0f}%)")
    if not args.skip_qthread:
        bg_time = qthread_result["time"]
        print(f"  Background thread (QThread sim):  {bg_time:.2f}s")
        if pipeline_result:
            ratio = bg_time / main_time if main_time > 0 else 0
            print(f"  BG/Main ratio:                    {ratio:.2f}x")
            if ratio > 1.5:
                print("  WARNING: Background thread is significantly slower than main thread")
            else:
                print("  OK: Background thread performance is comparable to main thread")

    if pipeline_result and staged_timings:
        _subsection("HEALTH CHECK")
        training_time = staged_timings.get("7_training", 0)
        dataset_time = staged_timings.get("2_dataset_init", 0)
        norm_time = staged_timings.get("3_normalization", 0)
        eval_time = staged_timings.get("8_evaluation", 0)
        total = sum(staged_timings.values())

        checks = []
        if training_time / total > 0.6:
            checks.append(("PASS", "Training is the dominant stage"))
        else:
            checks.append(("FAIL", f"Training is only {100 * training_time / total:.0f}% of total"))

        if dataset_time < 20:
            checks.append(("PASS", f"Dataset init: {dataset_time:.1f}s (reasonable)"))
        else:
            checks.append(("WARN", f"Dataset init: {dataset_time:.1f}s (slow -- check caching)"))

        if norm_time < 5:
            checks.append(("PASS", f"Normalization: {norm_time:.1f}s"))
        else:
            checks.append(("WARN", f"Normalization: {norm_time:.1f}s (slow)"))

        if eval_time < training_time * 0.3:
            checks.append(("PASS", f"Evaluation: {eval_time:.1f}s"))
        else:
            checks.append(("WARN", f"Evaluation: {eval_time:.1f}s (high relative to training)"))

        for status, msg in checks:
            print(f"  [{status}] {msg}")


if __name__ == "__main__":
    main()
