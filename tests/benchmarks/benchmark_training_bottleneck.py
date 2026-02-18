"""
Training Bottleneck Diagnostic Benchmark

Proves whether the speed bottleneck is caused by:
  A) macOS DataLoader workers=0 (data starvation) -- MOST LIKELY
  B) macOS MPS device overhead
  C) Something else (CPU compute, disk I/O, etc.)

Run from the repo root (main thread, NOT a QThread):
    python tests/benchmarks/benchmark_training_bottleneck.py --data-dir /path/to/dataset

This script runs directly on the main thread so DataLoader workers > 0 are
allowed. If training is dramatically faster here than in the GUI, the cause
is confirmed: forced num_workers=0 from the QThread is the bottleneck.
"""

import argparse
import os
import sys
import time
import threading
import random
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Module-level worker init so multiprocessing can pickle it
def _worker_init_fn(worker_id: int) -> None:
    """Seed RNG per worker for reproducibility. Must be module-level for pickling."""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def check_environment():
    """Print environment info relevant to the bottleneck."""
    _section("ENVIRONMENT")

    print(f"Platform:         {sys.platform}")
    print(f"Python:           {sys.version.split()[0]}")
    print(f"Main thread:      {threading.current_thread() is threading.main_thread()}")

    import torch
    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available:    {mps_available}")

    mp_method = torch.multiprocessing.get_start_method(allow_none=True)
    print(f"MP start method:  {mp_method or '(not set)'}")

    import librosa
    print(f"librosa:          {librosa.__version__}")

    try:
        import soundfile as sf
        print(f"soundfile:        {sf.__version__}")
    except ImportError:
        print("soundfile:        NOT INSTALLED")

    cpu_count = os.cpu_count() or 1
    print(f"CPU cores:        {cpu_count}")
    print(f"Recommended workers: {min(4, cpu_count)}")


def benchmark_data_loading(dataset, config, num_workers_list, num_batches=20):
    """
    Benchmark DataLoader throughput with different num_workers settings.
    This is the core test: if workers > 0 is dramatically faster, the
    bottleneck is confirmed as data starvation from num_workers=0.
    """
    _section("BENCHMARK: DataLoader Throughput (data loading only)")

    from torch.utils.data import DataLoader

    batch_size = config.get("batch_size", 32)

    results = {}

    for nw in num_workers_list:
        print(f"\n--- num_workers={nw}, batch_size={batch_size} ---")

        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=False,
                worker_init_fn=_worker_init_fn if nw > 0 else None,
                persistent_workers=nw > 0,
                multiprocessing_context="fork" if nw > 0 else None,
            )

            # Warmup: load 2 batches to prime caches / spawn workers
            loader_iter = iter(loader)
            for _ in range(min(2, len(loader))):
                try:
                    next(loader_iter)
                except StopIteration:
                    break

            # Timed run
            loader_iter = iter(loader)
            times = []
            for i in range(num_batches):
                t0 = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                if i < 3 or (i + 1) % 5 == 0:
                    print(f"  Batch {i+1}: {elapsed:.4f}s")

            avg = sum(times) / len(times)
            total = sum(times)
            max_t = max(times)
            min_t = min(times)
            print(f"  RESULT: avg={avg:.4f}s, min={min_t:.4f}s, max={max_t:.4f}s, "
                  f"total={total:.2f}s for {num_batches} batches")
            results[nw] = {"avg": avg, "total": total, "max": max_t, "min": min_t}
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[nw] = {"avg": float("inf"), "total": 0, "max": 0, "min": 0, "error": str(e)}

    # Print comparison
    if 0 in results and "error" not in results[0] and any(
        nw > 0 and "error" not in results.get(nw, {}) for nw in results
    ):
        baseline = results[0]["avg"]
        print(f"\n--- COMPARISON (baseline: num_workers=0 avg={baseline:.4f}s) ---")
        for nw, r in sorted(results.items()):
            if "error" in r:
                print(f"  num_workers={nw}: FAILED ({r['error'][:60]})")
            else:
                speedup = baseline / r["avg"] if r["avg"] > 0 else 0
                print(f"  num_workers={nw}: avg={r['avg']:.4f}s  ({speedup:.1f}x speedup)")

    return results


def benchmark_training_loop(dataset, config, device_name, num_workers, num_epochs=3):
    """
    Run a short training loop and measure epoch times.
    Compares data wait time vs compute time.
    """
    _section(f"BENCHMARK: Training Loop (device={device_name}, workers={num_workers}, epochs={num_epochs})")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.application.blocks.training.datasets import create_data_splits
    from src.application.blocks.training.architectures import create_classifier
    from src.application.blocks.training.losses import create_loss_function

    device = torch.device(device_name)
    batch_size = config.get("batch_size", 32)

    try:
        train_ds, val_ds, _ = create_data_splits(dataset, config)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device_name != "cpu",
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )
    except Exception as e:
        print(f"  FAILED to create DataLoader: {type(e).__name__}: {e}")
        return

    is_binary = config.get("classification_mode", "multiclass") == "binary"
    num_classes = 1 if is_binary else len(dataset.classes)
    model = create_classifier(num_classes, config)
    model.to(device)

    criterion = create_loss_function(config, None, None)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.get('model_type', 'cnn')} ({param_count:,} params)")
    print(f"Device: {device}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"num_workers: {num_workers}")

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        data_wait_total = 0.0
        compute_total = 0.0
        batch_count = 0

        data_start = time.perf_counter()
        for batch_data in train_loader:
            data_wait = time.perf_counter() - data_start
            data_wait_total += data_wait

            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data

            compute_start = time.perf_counter()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if is_binary:
                loss = criterion(outputs.squeeze(-1), labels.float())
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Sync MPS to get accurate timing
            if device_name == "mps":
                torch.mps.synchronize()

            compute_total += time.perf_counter() - compute_start
            batch_count += 1
            data_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start
        data_pct = 100 * data_wait_total / epoch_time if epoch_time > 0 else 0
        compute_pct = 100 * compute_total / epoch_time if epoch_time > 0 else 0

        print(
            f"  Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s total | "
            f"Data: {data_wait_total:.2f}s ({data_pct:.0f}%) | "
            f"Compute: {compute_total:.2f}s ({compute_pct:.0f}%) | "
            f"Batches: {batch_count}"
        )

        if data_pct > 60:
            print(f"  ** DATA-STARVED: {data_pct:.0f}% of epoch spent waiting for data **")
        elif compute_pct > 60:
            print(f"  ** COMPUTE-BOUND: {compute_pct:.0f}% of epoch spent on forward/backward **")


def benchmark_sample_preprocessing(dataset, num_samples=100):
    """Benchmark individual sample preprocessing time with random access."""
    _section("BENCHMARK: Per-Sample Preprocessing Time (random access)")

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    times = []
    for idx in indices:
        t0 = time.perf_counter()
        _ = dataset[int(idx)]
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg = sum(times) / len(times)
    max_t = max(times)
    min_t = min(times)
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]

    print(f"Samples tested:   {len(times)} (random indices)")
    print(f"Avg time/sample:  {avg*1000:.1f}ms")
    print(f"P50 time/sample:  {p50*1000:.1f}ms")
    print(f"P95 time/sample:  {p95*1000:.1f}ms")
    print(f"Min time/sample:  {min_t*1000:.1f}ms")
    print(f"Max time/sample:  {max_t*1000:.1f}ms")
    print(f"Batch of 32:      ~{avg*32*1000:.0f}ms (serial, num_workers=0)")
    print(f"Batch of 128:     ~{avg*128*1000:.0f}ms (serial, num_workers=0)")

    batch_time_32 = avg * 32
    batches_per_epoch = len(dataset) // 32
    epoch_data_time = batch_time_32 * batches_per_epoch
    print(f"\nProjected data loading time per epoch ({batches_per_epoch} batches): ~{epoch_data_time:.1f}s")

    if avg > 0.05:
        print(f"\n  ** SLOW: {avg*1000:.0f}ms/sample means the DataLoader is "
              f"the bottleneck with num_workers=0 **")
        print(f"  With 4 workers, effective time would be ~{batch_time_32/4:.3f}s per batch of 32")

    return avg


def main():
    parser = argparse.ArgumentParser(description="Training bottleneck diagnostic")
    parser.add_argument("--data-dir", required=True, help="Path to training dataset folder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--classification-mode", default="multiclass",
                        choices=["multiclass", "binary"])
    parser.add_argument("--target-class", default=None,
                        help="Target class for binary mode")
    parser.add_argument("--augmentation", action="store_true",
                        help="Enable audio augmentation (makes bottleneck worse)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable feature cache (simulates first run)")
    args = parser.parse_args()

    check_environment()

    import torch

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = args.device

    print(f"\nSelected device: {device_name}")

    # Build config
    config = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "classification_mode": args.classification_mode,
        "target_class": args.target_class,
        "positive_classes": [args.target_class] if args.target_class else [],
        "model_type": "cnn",
        "epochs": 3,
        "learning_rate": 0.001,
        "sample_rate": 22050,
        "max_length": 22050,
        "n_mels": 128,
        "hop_length": 512,
        "fmax": 8000,
        "n_fft": 2048,
        "use_feature_cache": not args.no_cache,
        "use_augmentation": args.augmentation,
        "num_workers": 0,
        "seed": 42,
        "validation_split": 0.15,
        "test_split": 0.10,
        "stratified_split": True,
        "device": device_name,
        "normalize_per_dataset": False,
        "exclude_bad_files": True,
    }

    _section("LOADING DATASET")
    from src.application.blocks.training.datasets import AudioClassificationDataset

    t0 = time.perf_counter()
    dataset = AudioClassificationDataset(config)
    load_time = time.perf_counter() - t0
    print(f"Dataset loaded in {load_time:.2f}s ({len(dataset)} samples, {len(dataset.classes)} classes)")
    print(f"Classes: {dataset.classes}")

    # Test 1: Per-sample preprocessing (random access to avoid OS cache bias)
    avg_sample_time = benchmark_sample_preprocessing(dataset)

    # Test 2: DataLoader throughput with varying workers
    cpu_count = os.cpu_count() or 1
    worker_counts = [0, 2, min(4, cpu_count)]
    if cpu_count >= 8:
        worker_counts.append(min(8, cpu_count))
    worker_counts = sorted(set(worker_counts))

    benchmark_data_loading(dataset, config, worker_counts)

    # Test 3: Full training loop -- workers=0 (simulates GUI)
    benchmark_training_loop(dataset, config, device_name, num_workers=0, num_epochs=3)

    # Test 4: Full training loop -- workers>0 (what SHOULD happen)
    best_workers = min(4, cpu_count)
    if best_workers > 0:
        benchmark_training_loop(dataset, config, device_name, num_workers=best_workers, num_epochs=3)

    # Test 5: CPU vs MPS comparison (if MPS available)
    if device_name == "mps":
        _section("BONUS: CPU vs MPS Comparison (num_workers=0)")
        print("Testing if MPS autocast fallbacks cause hidden overhead...")
        benchmark_training_loop(dataset, config, "cpu", num_workers=0, num_epochs=2)
        benchmark_training_loop(dataset, config, "mps", num_workers=0, num_epochs=2)

    # Summary
    _section("DIAGNOSIS SUMMARY")
    print("If data loading is 60%+ of epoch time with num_workers=0:")
    print("  -> The bottleneck is DATA STARVATION (not macOS throttling)")
    print("  -> The GPU/MPS sits idle waiting for CPU to prepare batches")
    print()
    print("If adding workers (2-4) makes training 2-5x faster:")
    print("  -> CONFIRMED: the macOS QThread + num_workers=0 restriction is the cause")
    print()
    print("The GUI forces num_workers=0 because training runs in a QThread,")
    print("and Python multiprocessing workers deadlock when forked from a")
    print("non-main thread on macOS. This is a known limitation.")
    print()
    print("Solutions:")
    print("  1. Run training via subprocess (run_block_cli.py) where workers work")
    print("  2. Pre-compute all spectrograms into memory before the training loop")
    print("  3. Use torch.multiprocessing with 'fork' from main thread only")
    print("  4. Allow main-thread execution for training blocks (UI freezes but fast)")


if __name__ == "__main__":
    main()
