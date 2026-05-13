"""Central model distribution and install management for EchoZero.
Exists to keep app model downloads independent from packaged app binaries.
Connects registry manifests, staged downloads, and runtime bundle resolution.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from echozero.models.paths import ensure_installed_models_dir
from echozero.models.provider_shared import sha256_file, validate_model_id
from echozero.models.runtime_bundle_index import (
    IndexedBinaryDrumBundle,
    load_binary_drum_bundle_index,
    save_binary_drum_bundle_index,
)

_INSTALL_INDEX_FILENAME = "installed_model_registry.json"
_INSTALL_INDEX_SCHEMA = "echozero.installed_model_registry.v1"
_MODEL_RECORD_FILENAME = "echozero_model.json"
_REGISTRY_SOURCE_FILENAME = "model_registry_source.txt"
_STAGING_DIRNAME = ".staging"


class ModelInstallState(Enum):
    """Operator-visible state for one centrally distributed model."""

    MISSING = "missing"
    READY = "ready"
    OUTDATED = "outdated"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RegistryModelFile:
    """One downloadable file from a central model registry entry."""

    path: str
    url: str
    sha256: str
    size_bytes: int | None = None
    role: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryModelEntry:
    """One model entry advertised by the central model registry."""

    model_id: str
    model_type: str
    label: str
    version: str
    files: tuple[RegistryModelFile, ...]
    min_app_version: str | None = None
    classes: tuple[str, ...] = ()
    runtime_consumer: str | None = None
    compatibility_fingerprint: str | None = None
    channel: str = "alpha"
    description: str = ""

    @property
    def safe_id(self) -> str:
        return validate_model_id(self.model_id)


@dataclass(frozen=True, slots=True)
class InstalledModelRecord:
    """Local install record for a promoted central registry model."""

    model_id: str
    model_type: str
    label: str
    version: str
    bundle_dir: str
    manifest_file: str | None = None
    weights_file: str | None = None
    classes: tuple[str, ...] = ()
    runtime_consumer: str | None = None
    compatibility_fingerprint: str | None = None
    source_manifest: str | None = None

    @property
    def safe_id(self) -> str:
        return validate_model_id(self.model_id)


@dataclass(frozen=True, slots=True)
class RegistryModelListing:
    """One central registry entry annotated with local install state."""

    entry: RegistryModelEntry
    state: ModelInstallState


def load_registry_manifest(source: str | Path) -> tuple[RegistryModelEntry, ...]:
    """Load model entries from a local path, file URL, or HTTPS manifest."""
    source_text = str(source)
    payload = _read_json_source(source_text)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("Model registry manifest must contain a 'models' list.")
    base_uri = _base_uri_for_source(source_text)
    entries = tuple(_parse_registry_entry(raw, base_uri=base_uri) for raw in raw_models)
    if not entries:
        raise ValueError("Model registry manifest does not advertise any models.")
    return entries


def default_registry_manifest_source(models_dir: Path | None = None) -> str | None:
    """Return the configured central registry manifest source, if any."""
    env_source = os.environ.get("ECHOZERO_MODEL_REGISTRY_URL", "").strip()
    if env_source:
        return env_source
    root = models_dir or ensure_installed_models_dir()
    source_path = root / _REGISTRY_SOURCE_FILENAME
    if not source_path.exists():
        return None
    try:
        text = source_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def save_registry_manifest_source(
    source: str,
    *,
    models_dir: Path | None = None,
) -> Path:
    """Persist the central registry manifest source for app-managed discovery."""
    if not source.strip():
        raise ValueError("Registry manifest source cannot be blank.")
    root = models_dir or ensure_installed_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / _REGISTRY_SOURCE_FILENAME
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return path


def discover_registry_models(
    *,
    manifest_source: str | Path | None = None,
    models_dir: Path | None = None,
) -> tuple[RegistryModelListing, ...]:
    """List central registry models with their local missing/ready/update state."""
    source = str(manifest_source or default_registry_manifest_source(models_dir) or "").strip()
    if not source:
        return ()
    entries = load_registry_manifest(source)
    return tuple(
        RegistryModelListing(
            entry=entry,
            state=model_state_for_entry(entry, models_dir=models_dir),
        )
        for entry in entries
    )


def install_model_from_registry(
    *,
    model_id: str,
    manifest_source: str | Path,
    models_dir: Path | None = None,
) -> InstalledModelRecord:
    """Install one model from a central manifest through staging and validation."""
    entries = load_registry_manifest(manifest_source)
    entry = next((candidate for candidate in entries if candidate.model_id == model_id), None)
    if entry is None:
        raise KeyError(f"Model '{model_id}' is not present in {manifest_source}.")
    return install_registry_entry(
        entry=entry,
        manifest_source=str(manifest_source),
        models_dir=models_dir,
    )


def install_registry_entry(
    *,
    entry: RegistryModelEntry,
    manifest_source: str | None = None,
    models_dir: Path | None = None,
) -> InstalledModelRecord:
    """Download, verify, and atomically promote one registry model entry."""
    root = models_dir or ensure_installed_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    bundle_name = _bundle_dir_name(entry)
    staging_root = root / _STAGING_DIRNAME
    staging_dir = staging_root / bundle_name
    final_dir = root / bundle_name
    prior_dir = root / f"{bundle_name}.previous"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for model_file in entry.files:
            target = _safe_child_path(staging_dir, model_file.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _download_to_path(model_file.url, target)
            _verify_downloaded_file(target, model_file)
        _write_model_record(
            staging_dir / _MODEL_RECORD_FILENAME,
            entry=entry,
            bundle_dir=bundle_name,
            manifest_source=manifest_source,
        )
        promoted_record = _record_for_entry(
            entry,
            bundle_dir=bundle_name,
            manifest_source=manifest_source,
        )
        if prior_dir.exists():
            shutil.rmtree(prior_dir)
        if final_dir.exists():
            os.replace(str(final_dir), str(prior_dir))
        os.replace(str(staging_dir), str(final_dir))
        _save_install_index(root, promoted_record)
        _update_runtime_bundle_index(root, promoted_record)
        if prior_dir.exists():
            shutil.rmtree(prior_dir)
        return promoted_record
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def import_local_model_bundle(
    *,
    bundle_path: Path,
    model_id: str,
    model_type: str,
    label: str,
    version: str,
    classes: tuple[str, ...] = (),
    runtime_consumer: str | None = None,
    compatibility_fingerprint: str | None = None,
    models_dir: Path | None = None,
) -> InstalledModelRecord:
    """Import a local model bundle into the central model store layout."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    root = models_dir or ensure_installed_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    safe_id = validate_model_id(model_id)
    bundle_dir_name = f"{safe_id}-{validate_model_id(version)}"
    final_dir = root / bundle_dir_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    if bundle_path.is_dir():
        shutil.copytree(bundle_path, final_dir)
    else:
        final_dir.mkdir(parents=True)
        shutil.copy2(bundle_path, final_dir / bundle_path.name)

    manifest_file = _first_relative_match(final_dir, "*.manifest.json")
    weights_file = _first_relative_match(final_dir, "*.pth")
    record = InstalledModelRecord(
        model_id=model_id,
        model_type=model_type,
        label=label,
        version=version,
        bundle_dir=bundle_dir_name,
        manifest_file=manifest_file,
        weights_file=weights_file,
        classes=classes,
        runtime_consumer=runtime_consumer,
        compatibility_fingerprint=compatibility_fingerprint,
        source_manifest=str(bundle_path),
    )
    _write_installed_model_record(final_dir / _MODEL_RECORD_FILENAME, record)
    _save_install_index(root, record)
    _update_runtime_bundle_index(root, record)
    return record


def list_installed_models(models_dir: Path | None = None) -> tuple[InstalledModelRecord, ...]:
    """Return locally installed central-registry model records."""
    root = models_dir or ensure_installed_models_dir()
    index_path = root / _INSTALL_INDEX_FILENAME
    if not index_path.exists():
        return ()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return ()
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        return ()
    records: list[InstalledModelRecord] = []
    for raw_record in raw_models.values():
        record = _installed_record_from_payload(raw_record)
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda item: (item.model_id, item.version)))


def model_state_for_entry(
    entry: RegistryModelEntry,
    *,
    models_dir: Path | None = None,
) -> ModelInstallState:
    """Resolve missing/ready/outdated/invalid state for one remote registry entry."""
    root = models_dir or ensure_installed_models_dir()
    records = [
        record for record in list_installed_models(root) if record.model_id == entry.model_id
    ]
    if not records:
        return ModelInstallState.MISSING
    matching = next((record for record in records if record.version == entry.version), None)
    if matching is None:
        return ModelInstallState.OUTDATED
    if validate_installed_model(matching, models_dir=root):
        return ModelInstallState.READY
    return ModelInstallState.INVALID


def validate_installed_model(
    record: InstalledModelRecord,
    *,
    models_dir: Path | None = None,
) -> bool:
    """Check that a local installed model still has its indexed runtime files."""
    root = models_dir or ensure_installed_models_dir()
    bundle_dir = root / record.bundle_dir
    if not bundle_dir.is_dir():
        return False
    if record.manifest_file and not (bundle_dir / record.manifest_file).is_file():
        return False
    if record.weights_file and not (bundle_dir / record.weights_file).is_file():
        return False
    return True


def _parse_registry_entry(raw: object, *, base_uri: str | None) -> RegistryModelEntry:
    if not isinstance(raw, dict):
        raise ValueError("Each model registry entry must be an object.")
    files_raw = raw.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise ValueError("Each model registry entry must include non-empty 'files'.")
    files = tuple(_parse_registry_file(value, base_uri=base_uri) for value in files_raw)
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    return RegistryModelEntry(
        model_id=_required_str(raw, "model_id"),
        model_type=_required_str(raw, "type"),
        label=_required_str(raw, "label"),
        version=_required_str(raw, "version"),
        files=files,
        min_app_version=_optional_str(raw.get("min_app_version")),
        classes=tuple(
            str(value).strip().lower()
            for value in raw.get("classes", ())
            if str(value).strip()
        )
        if isinstance(raw.get("classes"), list)
        else (),
        runtime_consumer=_optional_str(runtime.get("consumer")),
        compatibility_fingerprint=_optional_str(raw.get("compatibility_fingerprint")),
        channel=_optional_str(raw.get("channel")) or "alpha",
        description=_optional_str(raw.get("description")) or "",
    )


def _parse_registry_file(raw: object, *, base_uri: str | None) -> RegistryModelFile:
    if not isinstance(raw, dict):
        raise ValueError("Each model file entry must be an object.")
    path = _required_str(raw, "path")
    url = _required_str(raw, "url")
    if base_uri is not None:
        url = urllib.parse.urljoin(base_uri, url)
    size_value = raw.get("size_bytes")
    return RegistryModelFile(
        path=path,
        url=url,
        sha256=_required_str(raw, "sha256").lower(),
        size_bytes=int(size_value) if isinstance(size_value, int) else None,
        role=_optional_str(raw.get("role")),
    )


def _read_json_source(source: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(source, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    elif parsed.scheme == "file":
        payload = json.loads(
            Path(urllib.request.url2pathname(parsed.path)).read_text(encoding="utf-8")
        )
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model registry manifest must be a JSON object.")
    return payload


def _base_uri_for_source(source: str) -> str | None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        return source.rsplit("/", 1)[0] + "/"
    path = Path(source)
    if path.exists() or path.parent.exists():
        return path.parent.resolve().as_uri() + "/"
    return None


def _download_to_path(url: str, target: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(url, timeout=60) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return
    if parsed.scheme == "file":
        shutil.copy2(Path(urllib.request.url2pathname(parsed.path)), target)
        return
    shutil.copy2(Path(url), target)


def _verify_downloaded_file(path: Path, model_file: RegistryModelFile) -> None:
    if model_file.size_bytes is not None and path.stat().st_size != model_file.size_bytes:
        raise ValueError(
            f"Downloaded model file size mismatch for {model_file.path}: "
            f"expected {model_file.size_bytes}, got {path.stat().st_size}."
        )
    actual_hash = sha256_file(path)
    if actual_hash.lower() != model_file.sha256.lower():
        raise ValueError(
            f"Downloaded model file hash mismatch for {model_file.path}: "
            f"expected {model_file.sha256}, got {actual_hash}."
        )


def _record_for_entry(
    entry: RegistryModelEntry,
    *,
    bundle_dir: str,
    manifest_source: str | None,
) -> InstalledModelRecord:
    manifest_file = _first_file_with_role_or_suffix(
        entry,
        role="manifest",
        suffix=".manifest.json",
    )
    weights_file = _first_file_with_role_or_suffix(entry, role="weights", suffix=".pth")
    return InstalledModelRecord(
        model_id=entry.model_id,
        model_type=entry.model_type,
        label=entry.label,
        version=entry.version,
        bundle_dir=bundle_dir,
        manifest_file=manifest_file,
        weights_file=weights_file,
        classes=entry.classes,
        runtime_consumer=entry.runtime_consumer,
        compatibility_fingerprint=entry.compatibility_fingerprint,
        source_manifest=manifest_source,
    )


def _write_model_record(
    path: Path,
    *,
    entry: RegistryModelEntry,
    bundle_dir: str,
    manifest_source: str | None,
) -> None:
    _write_installed_model_record(
        path,
        _record_for_entry(entry, bundle_dir=bundle_dir, manifest_source=manifest_source),
    )


def _write_installed_model_record(path: Path, record: InstalledModelRecord) -> None:
    path.write_text(
        json.dumps(_installed_record_payload(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_install_index(root: Path, record: InstalledModelRecord) -> None:
    records = {item.model_id: item for item in list_installed_models(root)}
    records[record.model_id] = record
    payload = {
        "schema": _INSTALL_INDEX_SCHEMA,
        "models": {
            key: _installed_record_payload(records[key])
            for key in sorted(records)
        },
    }
    path = root / _INSTALL_INDEX_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _update_runtime_bundle_index(root: Path, record: InstalledModelRecord) -> None:
    if not record.manifest_file or not record.weights_file:
        return
    labels = tuple(label for label in record.classes if label != "other")
    if (
        record.runtime_consumer != "BinaryDrumClassify"
        and not record.model_type.startswith("binary")
    ):
        return
    if not labels:
        return
    records = load_binary_drum_bundle_index(root)
    for label in labels:
        records[label] = IndexedBinaryDrumBundle(
            label=label,
            bundle_dir=record.bundle_dir,
            manifest_file=record.manifest_file,
            weights_file=record.weights_file,
            artifact_id=record.model_id,
            source_manifest_path=record.source_manifest,
        )
    save_binary_drum_bundle_index(root, records)


def _installed_record_payload(record: InstalledModelRecord) -> dict[str, object]:
    return {
        "modelId": record.model_id,
        "type": record.model_type,
        "label": record.label,
        "version": record.version,
        "bundleDir": record.bundle_dir,
        "manifestFile": record.manifest_file,
        "weightsFile": record.weights_file,
        "classes": list(record.classes),
        "runtimeConsumer": record.runtime_consumer,
        "compatibilityFingerprint": record.compatibility_fingerprint,
        "sourceManifest": record.source_manifest,
    }


def _installed_record_from_payload(payload: object) -> InstalledModelRecord | None:
    if not isinstance(payload, dict):
        return None
    required = (
        payload.get("modelId"),
        payload.get("type"),
        payload.get("label"),
        payload.get("version"),
        payload.get("bundleDir"),
    )
    if not all(isinstance(value, str) and value.strip() for value in required):
        return None
    classes = payload.get("classes")
    return InstalledModelRecord(
        model_id=str(payload["modelId"]),
        model_type=str(payload["type"]),
        label=str(payload["label"]),
        version=str(payload["version"]),
        bundle_dir=str(payload["bundleDir"]),
        manifest_file=_optional_str(payload.get("manifestFile")),
        weights_file=_optional_str(payload.get("weightsFile")),
        classes=tuple(str(value).strip().lower() for value in classes if str(value).strip())
        if isinstance(classes, list)
        else (),
        runtime_consumer=_optional_str(payload.get("runtimeConsumer")),
        compatibility_fingerprint=_optional_str(payload.get("compatibilityFingerprint")),
        source_manifest=_optional_str(payload.get("sourceManifest")),
    )


def _bundle_dir_name(entry: RegistryModelEntry) -> str:
    return f"{entry.safe_id}-{validate_model_id(entry.version)}"


def _safe_child_path(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    base = root.resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Model file path escapes bundle directory: {relative_path!r}")
    return target


def _first_file_with_role_or_suffix(
    entry: RegistryModelEntry,
    *,
    role: str,
    suffix: str,
) -> str | None:
    role_match = next((item.path for item in entry.files if item.role == role), None)
    if role_match is not None:
        return role_match
    return next((item.path for item in entry.files if item.path.endswith(suffix)), None)


def _first_relative_match(root: Path, pattern: str) -> str | None:
    match = next(iter(sorted(root.rglob(pattern))), None)
    if match is None:
        return None
    return str(match.relative_to(root))


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model registry entry is missing required string '{key}'.")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()
