#!/usr/bin/env python3
"""
Interactive Model Upload Script

Uploads a trained .pth model to the EchoZero Cloudflare R2 bucket and
updates the manifest.json so the model becomes available in the Model Store.

Metadata (classes, architecture, classification mode, etc.) is auto-extracted
from the PyTorch checkpoint.  You only need to provide a friendly name,
description, and version number.

Requirements (dev-only):
    pip install boto3 torch python-dotenv

R2 credentials (set in .env or environment):
    R2_ACCOUNT_ID        -- Cloudflare account ID
    R2_ACCESS_KEY_ID     -- R2 API token access key
    R2_SECRET_ACCESS_KEY -- R2 API token secret key
    R2_BUCKET_NAME       -- R2 bucket name (default: echozero-models)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env files (project root, then user data dir)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        print(
            "\nError: boto3 is required for uploads.\n"
            "Install it with:  pip install boto3\n"
            "(This is a dev-only dependency, not needed by the app.)\n"
        )
        sys.exit(1)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print(
            "\nError: torch is required to read .pth checkpoints.\n"
            "Install it with:  pip install torch\n"
        )
        sys.exit(1)


def _get_r2_config() -> Dict[str, str]:
    """Read R2 credentials from environment variables."""
    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.getenv("R2_BUCKET_NAME", "").strip() or "echozero-models"

    missing = []
    if not account_id:
        missing.append("R2_ACCOUNT_ID")
    if not access_key:
        missing.append("R2_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("R2_SECRET_ACCESS_KEY")

    if missing:
        print(f"\nError: Missing required environment variables: {', '.join(missing)}")
        print("Set them in your .env file or export them before running this script.")
        print(
            "\nExample .env:\n"
            "  R2_ACCOUNT_ID=your_account_id\n"
            "  R2_ACCESS_KEY_ID=your_access_key\n"
            "  R2_SECRET_ACCESS_KEY=your_secret_key\n"
            "  R2_BUCKET_NAME=echozero-models\n"
        )
        sys.exit(1)

    return {
        "account_id": account_id,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
    }


def _create_s3_client(config: Dict[str, str]):
    """Create a boto3 S3 client configured for Cloudflare R2."""
    boto3 = _require_boto3()
    endpoint = f"https://{config['account_id']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name="auto",
    )


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hash with progress output for large files."""
    size = path.stat().st_size
    sha = hashlib.sha256()
    processed = 0
    start = time.time()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            processed += len(chunk)
            if size > 50 * 1024 * 1024:
                pct = processed * 100 // size
                print(f"\r  Computing SHA256... {pct}%", end="", flush=True)

    elapsed = time.time() - start
    if size > 50 * 1024 * 1024:
        print(f"\r  Computing SHA256... done ({elapsed:.1f}s)")
    else:
        print(f"  SHA256 computed ({elapsed:.1f}s)")

    return sha.hexdigest()


def _extract_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """Load a .pth checkpoint and extract model metadata."""
    torch = _require_torch()

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    required_keys = {"classes", "config", "model_state_dict"}
    if not required_keys.issubset(checkpoint.keys()):
        missing = required_keys - checkpoint.keys()
        print(f"\nError: Not a valid PyTorch Audio Trainer checkpoint.")
        print(f"Missing keys: {', '.join(missing)}")
        sys.exit(1)

    config = checkpoint.get("config", {})

    classes = checkpoint.get("classes", [])
    architecture = config.get("model_type", "cnn")
    classification_mode = config.get("classification_mode", "multiclass")
    training_date = checkpoint.get("training_date", "")

    test_metrics = checkpoint.get("test_metrics", {})
    test_accuracy = None
    if isinstance(test_metrics, dict):
        test_accuracy = test_metrics.get("accuracy")

    return {
        "classes": classes,
        "architecture": architecture,
        "classification_mode": classification_mode,
        "training_date": training_date,
        "test_accuracy": test_accuracy,
    }


def _prompt(label: str, default: str = "") -> str:
    """Prompt for input with an optional default."""
    if default:
        raw = input(f"  {label} [{default}]: ").strip()
        return raw or default
    while True:
        raw = input(f"  {label}: ").strip()
        if raw:
            return raw
        print("    (required)")


def _upload_with_progress(s3, bucket: str, key: str, path: Path):
    """Upload a file to S3/R2 with a progress indicator."""
    size = path.stat().st_size
    uploaded = 0
    start = time.time()

    def callback(bytes_transferred):
        nonlocal uploaded
        uploaded += bytes_transferred
        pct = uploaded * 100 // size if size else 100
        mb = uploaded / (1024 * 1024)
        total_mb = size / (1024 * 1024)
        print(f"\r  Uploading... {pct}% ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)

    s3.upload_file(
        str(path),
        bucket,
        key,
        Callback=callback,
    )
    elapsed = time.time() - start
    print(f"\r  Uploading... 100% ({size / (1024*1024):.1f} MB, {elapsed:.1f}s)")


def _get_current_manifest(s3, bucket: str) -> Dict[str, Any]:
    """Download the current manifest.json from R2, or return a new empty one."""
    try:
        response = s3.get_object(Bucket=bucket, Key="manifest.json")
        data = json.loads(response["Body"].read().decode("utf-8"))
        print(f"  Loaded existing manifest ({len(data.get('models', []))} model(s))")
        return data
    except s3.exceptions.NoSuchKey:
        print("  No existing manifest found, creating new one")
        return {"version": 1, "models": []}
    except Exception as e:
        print(f"  Warning: Could not read manifest ({e}), creating new one")
        return {"version": 1, "models": []}


def _update_manifest(s3, bucket: str, manifest: Dict[str, Any]):
    """Upload the updated manifest.json to R2."""
    body = json.dumps(manifest, indent=2, ensure_ascii=False)
    s3.put_object(
        Bucket=bucket,
        Key="manifest.json",
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    print("  Manifest updated")


def main():
    print("\n=== EchoZero Model Upload ===\n")

    # -- Get file path --
    raw_path = input("  Model file path: ").strip().strip("'\"")
    model_path = Path(raw_path).expanduser().resolve()

    if not model_path.exists():
        print(f"\n  Error: File not found: {model_path}")
        sys.exit(1)
    if model_path.suffix.lower() != ".pth":
        print(f"\n  Error: Expected a .pth file, got: {model_path.suffix}")
        sys.exit(1)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    # -- Extract metadata from checkpoint --
    print("\n  Loading checkpoint...")
    meta = _extract_checkpoint_metadata(model_path)
    print("  Valid PyTorch Audio Trainer checkpoint\n")

    print("  Auto-extracted from checkpoint:")
    print(f"    Architecture:        {meta['architecture'].upper()}")
    print(f"    Classification mode: {meta['classification_mode']}")
    print(f"    Classes:             {', '.join(meta['classes'])}")
    if meta["training_date"]:
        date_display = meta["training_date"][:19].replace("T", " ")
        print(f"    Training date:       {date_display}")
    if meta["test_accuracy"] is not None:
        print(f"    Test accuracy:       {meta['test_accuracy']:.4f}")
    print()

    # -- Prompt for user-provided fields --
    default_name = model_path.stem
    name = _prompt("Model name", default=default_name)
    description = _prompt("Description")
    version = _prompt("Version", default="1.0.0")

    # -- Build model ID from name --
    model_id = name.lower().replace(" ", "-").replace("_", "-")
    model_id = "".join(c for c in model_id if c.isalnum() or c == "-")
    filename = model_path.name

    # -- Confirmation --
    print(f"\n  Summary:")
    print(f"    ID:       {model_id}")
    print(f"    Name:     {name}")
    print(f"    File:     {filename}")
    print(f"    Size:     {size_mb:.1f} MB")
    print(f"    Version:  {version}")
    print()

    confirm = input("  Proceed with upload? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("\n  Cancelled.")
        sys.exit(0)

    print()

    # -- Compute hash --
    sha256 = _compute_sha256(model_path)

    # -- Connect to R2 --
    r2_config = _get_r2_config()
    s3 = _create_s3_client(r2_config)
    bucket = r2_config["bucket"]

    # -- Upload model file --
    r2_key = f"models/{filename}"
    _upload_with_progress(s3, bucket, r2_key, model_path)

    # -- Update manifest --
    print("\n  Updating manifest...")
    manifest = _get_current_manifest(s3, bucket)

    new_entry = {
        "id": model_id,
        "name": name,
        "description": description,
        "filename": filename,
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256,
        "version": version,
        "classification_mode": meta["classification_mode"],
        "classes": meta["classes"],
        "architecture": meta["architecture"],
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }

    # Replace existing entry with same ID, or append
    models = manifest.get("models", [])
    replaced = False
    for i, existing in enumerate(models):
        if existing.get("id") == model_id:
            models[i] = new_entry
            replaced = True
            print(f"  Replaced existing entry for '{model_id}'")
            break
    if not replaced:
        models.append(new_entry)

    manifest["models"] = models
    _update_manifest(s3, bucket, manifest)

    print(f'\n  Model "{name}" is now available for download.\n')


if __name__ == "__main__":
    main()
