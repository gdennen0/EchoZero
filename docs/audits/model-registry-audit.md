# EchoZero Model Registry & Provider — Ship-Readiness Audit

**Date:** 2026-03-29  
**Auditor:** Chonch (automated subagent)  
**Scope:** `echozero/models/__init__.py`, `registry.py`, `provider.py`, `tests/test_model_registry.py`, `tests/test_model_provider.py`  
**Verdict:** ⛔ NOT SHIP-READY — multiple critical security and correctness issues must be resolved first.

---

## Severity Legend

- 🔴 **Must fix before ship** — correctness bugs, security vulnerabilities, data loss risk
- 🟡 **Should fix** — behavioral gaps, missing hardening, footguns
- 🟢 **Nice to have** — polish, test coverage, API ergonomics

---

## 1. Correctness Issues

### 🔴 C-1: `save()` is not atomic — manifest corruption on partial write

**File:** `registry.py`, `ModelRegistry.save()`

```python
manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

`write_text` truncates the file then writes. If the process crashes mid-write (power loss, OOM, SIGKILL), the manifest is left as a partial/empty JSON file. The next `load()` will call `json.loads()` on garbage and raise an unhandled `json.JSONDecodeError`, making the entire registry unreadable.

**Fix:** Write to a `.tmp` sibling file, then `os.replace()` (atomic on POSIX and Windows-NTFS):
```python
tmp = manifest_path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
tmp.replace(manifest_path)
```

---

### 🔴 C-2: `load()` has no error handling — corrupted manifest crashes the app

**File:** `registry.py`, `ModelRegistry.load()`

```python
data = json.loads(manifest_path.read_text(encoding="utf-8"))
```

If `models.json` is corrupted (truncated write, manual edit gone wrong, filesystem error), this raises `json.JSONDecodeError` which propagates uncaught. There is no recovery path. The user's entire model catalog becomes inaccessible.

**Fix:** Wrap in try/except, log a warning, and fall back to an empty registry (or a backup copy). Also validate the structure, not just the JSON syntax.

---

### 🔴 C-3: `load()` silently ignores unknown `ModelType` / `ModelSource` values

**File:** `registry.py`, `ModelRegistry.load()` and `ModelCard.from_dict()`

```python
d["model_type"] = ModelType(d["model_type"])
d["source"] = ModelSource(d["source"])
```

If the manifest contains a value not in the enum (e.g., from a newer version of EchoZero adding a new type, or a typo), `ModelType(value)` raises `ValueError`. This crashes `load()` and makes the entire registry inaccessible, not just the offending entry. This is a forward-compatibility time bomb.

**Fix:** Wrap per-entry deserialization in try/except and skip/log bad entries rather than aborting the whole load.

---

### 🟡 C-4: `unregister()` default promotion is non-deterministic

**File:** `registry.py`, `ModelRegistry.unregister()`

```python
others = [c for c in self._cards.values() if c.model_type == card.model_type]
if others:
    self._defaults[card.model_type] = others[0].id
```

`self._cards` is a plain `dict`. Python 3.7+ dicts maintain insertion order, but which model gets promoted as default is still arbitrary from the user's perspective — it's whichever was registered first, not the highest version. If the user had two v2 and v3 models and removes v3, the default should logically become v2, but it might become an older one.

**Fix:** Promote by highest semantic version, not insertion order.

---

### 🟡 C-5: `check_updates()` uses string comparison for version ordering

**File:** `provider.py`, `ModelProvider.check_updates()` and `HuggingFaceSource.get_latest_version()`

```python
if current_ver is None or update.available_version > current_ver:
```

```python
versions = sorted(updates, key=lambda u: u.available_version, reverse=True)
```

String comparison on version strings is broken:
- `"9.0.0" > "10.0.0"` → **True** (wrong — `"9"` > `"1"` lexicographically)
- `"1.9" > "1.10"` → **True** (wrong)
- `"2.0.0" > "2.0.0-rc1"` → **False** (wrong)

This will cause EchoZero to skip valid updates or, worse, "downgrade" users to older models.

**Fix:** Use `packaging.version.Version` for comparison:
```python
from packaging.version import Version, InvalidVersion

def _ver(s: str) -> Version:
    try:
        return Version(s)
    except InvalidVersion:
        return Version("0")

if current_ver is None or _ver(update.available_version) > _ver(current_ver):
    ...
```

---

### 🟡 C-6: `install()` re-emits "downloading" progress that the source already emitted

**File:** `provider.py`, `ModelProvider.install()`

```python
if on_progress:
    on_progress(DownloadProgress(..., status="downloading"))

downloaded_path = self._source.download(model_id, target_dir, on_progress)
```

The provider fires a `downloading` event, then calls `source.download()` which also fires `downloading` (see `HuggingFaceSource.download()`). The UI will receive duplicate events. After `source.download()` completes with `status="complete"`, the provider then emits another `complete`. So the sequence is: `downloading, downloading, complete, complete`.

**Fix:** Don't emit progress in `install()` — let `source.download()` own the progress lifecycle.

---

### 🟢 C-7: `resolve()` ignores `model_type` when `model_id` is given

**File:** `registry.py`, `ModelRegistry.resolve()`

```python
if model_id is not None:
    return self._cards.get(model_id)
```

If the caller passes `model_id="onset-v1"` with `model_type=ModelType.CLASSIFICATION`, they get back an ONSET_DETECTION model without any warning. This is a silent type mismatch that could load the wrong model into a processor.

**Fix:** When `model_id` is provided alongside `model_type`, validate that the returned card's type matches.

---

## 2. What-If: Adversarial Scenarios

### 🔴 W-1: Concurrent installs of the same model — registry corruption

**Scenario:** Two threads/processes call `provider.install("echozero/onset-v2", ...)` simultaneously.

**What happens:**
1. Both compute `safe_id = "echozero_onset-v2"` and `target_dir = models_dir / "echozero_onset-v2"`
2. Both call `target_dir.mkdir(parents=True, exist_ok=True)` — this is fine
3. Both call `source.download()` writing to the same directory — files get clobbered mid-write
4. Both call `registry.register(card)` — second write stomps the first, but that's fine
5. Both call `registry.save()` — one's write partially overwrites the other's since there's no lock, producing a torn manifest (see C-1)

**No locking anywhere in the codebase.** The registry comment says "Thread-safe for reads. Writes should happen from one thread" but provides zero enforcement — no `threading.Lock`, no file lock, no `DOWNLOADING` guard in the registry.

**Fix:**
- Add a `threading.RLock` to `ModelRegistry` protecting all state mutations and `save()`
- Add an in-progress guard: check `ModelStatus.DOWNLOADING` before starting a second install of the same ID
- Use atomic file writes (C-1)

---

### 🔴 W-2: Download fails halfway — partial files left, registry possibly updated

**Scenario:** `source.download()` raises an exception mid-download (network drop, timeout, HuggingFace 500).

**What happens in `ModelProvider.install()`:**
```python
downloaded_path = self._source.download(...)  # raises here
# --- Everything below this line never runs ---
relative_path = str(...)
card = ModelCard(...)
self._registry.register(card)   # ← skipped
self._registry.save()           # ← skipped
```

The registry is NOT corrupted (good), but **the partial files remain on disk** forever. There's no cleanup of `target_dir`. On a retry, `target_dir.mkdir(exist_ok=True)` silently succeeds and `snapshot_download` may or may not resume depending on HF's caching behavior — but the old partial files are still mixed in.

**Fix:** Wrap `install()` in try/except, clean up `target_dir` on failure:
```python
try:
    downloaded_path = self._source.download(...)
except Exception:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    raise
```

---

### 🔴 W-3: Corrupted manifest — unhandled `JSONDecodeError` crashes startup

**Scenario:** `models.json` gets corrupted (partial write, disk error, user edited it).

**What happens:** `json.loads()` raises `JSONDecodeError`. This propagates out of `registry.load()` with no catch. If `load()` is called at startup, the entire app fails to initialize.

**This is covered under C-2 above.** Repeating here because the impact is severe: users lose access to ALL their locally installed models.

---

### 🔴 W-4: Path traversal via `model_id` — arbitrary file writes

**Scenario:** An attacker-controlled `model_id` contains path traversal characters.

**In `ModelProvider.install()`:**
```python
safe_id = model_id.replace("/", "_")
target_dir = self._registry.models_dir / safe_id
```

The `/` → `_` replacement is **insufficient**. Consider:
- `model_id = "../../etc/cron.d/evil"` → `safe_id = ".._.._etc_cron.d_evil"` → writes to `models_dir/.._.._etc_cron.d_evil` — OK on this path, but...
- `model_id = ".."` → `safe_id = ".."` → `target_dir = models_dir / ".."` — writes model files to the **parent of models_dir**!
- `model_id = "../../../home/user/.bashrc"` on Linux: slash-replaced is `".._.._.._.._home_user_.bashrc"` — still creates a bad path-like name

**In `ModelCard.relative_path`:**
```python
relative_path = str(downloaded_path.relative_to(self._registry.models_dir))
```
Then later:
```python
def model_path(self, card: ModelCard) -> Path:
    return self._models_dir / card.relative_path
```

If `relative_path` in the manifest is `../../etc/passwd` (manually or via a compromised HF model's metadata), `model_path()` returns a path **outside** the models directory entirely. No validation.

**In `ModelRegistry.load()`:** Relative paths from the manifest are trusted wholesale.

**Fix:**
```python
# Validate model_id
import re
if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*(/[a-zA-Z0-9][a-zA-Z0-9_\-\.]*)?$', model_id):
    raise ValueError(f"Invalid model_id: {model_id!r}")

# Validate relative_path doesn't escape models_dir
def model_path(self, card: ModelCard) -> Path:
    path = (self._models_dir / card.relative_path).resolve()
    if not str(path).startswith(str(self._models_dir.resolve())):
        raise SecurityError(f"Model path escapes models directory: {card.relative_path!r}")
    return path
```

---

### 🔴 W-5: Disk full during download — silent partial write, no cleanup

**Scenario:** Disk fills up during `snapshot_download()` or `shutil.copy2()`.

**What happens:**
- `snapshot_download` will raise an `OSError: [Errno 28] No space left on device`
- This is caught by `HuggingFaceSource.download()`'s broad `except Exception` and re-raised as `RuntimeError("Failed to download model ...")`
- `install()` propagates this up (no cleanup — see W-2)
- The partial download directory remains, consuming whatever disk was written before the failure

**Worse:** If `shutil.copy2` fails mid-copy in `LocalFileSource.download()` or `import_local()`, Python does NOT clean up the destination file. You get a truncated model file at the destination path. On a subsequent `check_status()` call, it shows `AVAILABLE` because the file exists — but it's corrupt.

**Fix:**
- Cleanup on exception (W-2 fix covers this)
- After download, validate the model file (size > 0, format check if feasible)

---

### 🔴 W-6: HuggingFace unreachable — broad exception swallows useful diagnostics

**Scenario:** No internet, HF is down, corporate firewall, DNS failure.

**What happens in `HuggingFaceSource.download()`:**
```python
except Exception as e:
    ...
    raise RuntimeError(f"Failed to download model '{model_id}': {e}") from e
```

The exception message preserves `e`, which is OK. But:
1. `check_available()` silently returns `[]` on any error — including network errors. The caller can't distinguish "no models available" from "network is down".
2. The `RuntimeError` wrapper loses the original exception type (e.g., `ConnectionError`, `TimeoutError`), making programmatic error handling impossible for callers.

**Fix:**
- Raise specific exception types: `NetworkError`, `ModelNotFoundError`
- Or at minimum, re-raise the original exception type, not a generic `RuntimeError`
- `check_available()` should raise on network errors, not silently return `[]`

---

### 🔴 W-7: Uninstall while model is in use — no reference counting, silent deletion

**Scenario:** A pipeline has loaded `onset-v2.pt` into memory (PyTorch `torch.load()`). User triggers uninstall.

**What happens:**
```python
shutil.rmtree(model_path)  # deletes files immediately
```

On Linux: the process still has a file descriptor open; the inode stays alive until it's closed. Model inference continues.  
On Windows: `shutil.rmtree` will **fail with `PermissionError`** if the file is open, leaving the registry entry already removed (because `unregister()` runs before the file deletion). Now the registry has no entry, but the file exists — inconsistent state.

Even on Linux: after the file is deleted, any attempt to reload the model or save checkpoints to that path will fail silently or with confusing errors.

**Fix:**
- Add a reference counter or "in-use" flag to `ModelRegistry`
- `uninstall()` should refuse (or warn) if `ref_count > 0`
- Processors should call `registry.acquire(model_id)` / `registry.release(model_id)`
- At minimum, catch the `PermissionError` on Windows and do NOT remove the registry entry if file deletion fails

---

### 🟡 W-8: Version comparison string sorting — "9.0" > "10.0"

Covered under **C-5** above. Repeated here because it affects `check_updates()` directly: a user on `v9.0.0` would never be offered `v10.0.0` as an update.

---

### 🔴 W-9: Models directory deleted while app is running — no detection

**Scenario:** User or another process deletes the models directory while EchoZero is running.

**What happens:**
- `check_status()` returns `MISSING` — this is correct behavior
- `model_path(card)` returns a path that no longer exists — caller gets `FileNotFoundError` when they try to use it
- `save()` calls `mkdir(parents=True, exist_ok=True)` — this recreates the directory, but writes an empty manifest, **silently discarding all in-memory registrations** that haven't been reloaded

The last point is the killer: if `save()` is called after the directory is deleted, it recreates the dir and writes the current in-memory state. But if the registry was loaded before the deletion and new models were registered in-memory but not saved, those registrations survive. If `load()` is called again (e.g., on a "refresh"), all unsaved in-memory state is lost — there's no dirty flag or notification.

**Fix:**
- Watch the manifest file for external modification (using `watchdog` or filesystem events)
- Add a `dirty` flag to track unsaved in-memory changes
- Raise on `save()` if directory was deleted and recreating it (data integrity concern)

---

### 🟡 W-10: Model file exists but is corrupted (zero bytes, wrong format)

**Scenario:** A model file was partially written, zeroed out, or is the wrong format.

**What happens:**
- `check_status()` returns `AVAILABLE` — because it only checks `path.exists()`, not file validity
- The processor calls `torch.load(path)` and gets a `RuntimeError` or `EOFError` deep inside PyTorch
- The error has no context connecting it back to the registry model ID

**No integrity checking anywhere.** Zero-byte files, truncated checkpoints, and wrong-format files all appear as "AVAILABLE" to the registry.

**Fix:**
- Add `ModelStatus.CORRUPT` status
- `check_status()` should check file size > 0 at minimum
- Optional: store SHA-256 hash in `ModelCard.metadata` at registration time and verify on status check

---

## 3. Security Concerns

### 🔴 S-1: Path traversal in `relative_path` — no bounds checking

**File:** `registry.py`, `ModelRegistry.model_path()`

`card.relative_path` is loaded from the JSON manifest without any validation. A crafted manifest (or a compromised HuggingFace repo that influences how the path is stored) can escape the models directory. No bounds check is performed when resolving:

```python
return self._models_dir / card.relative_path
```

Any code that uses `model_path()` to write files (e.g., future checkpoint saving) becomes an arbitrary file write vulnerability. Currently read-only paths returned here will silently resolve outside the sandbox.

**Fix:** See W-4 fix above — validate that resolved path is inside `models_dir`.

---

### 🔴 S-2: `safe_id` sanitization is insufficient against directory traversal

**File:** `provider.py`, `ModelProvider.install()`

```python
safe_id = model_id.replace("/", "_")
target_dir = self._registry.models_dir / safe_id
```

- `".."` passes through unchanged → `target_dir = models_dir / ".."` → parent directory
- `"..\\..\\windows\\system32"` (Windows) → only forward slashes are replaced
- `"com1"`, `"nul"`, `"con"` (Windows reserved names) → accepted, may cause OS errors
- Null bytes → not stripped, may truncate paths on some systems

**Fix:** Use an allowlist regex; reject on any non-match. See W-4 fix.

---

### 🔴 S-3: `LocalFileSource.download()` — arbitrary file copy from user-supplied path

**File:** `provider.py`, `LocalFileSource.download()`

```python
source_path = self._source_dir / model_id
```

`model_id` is directly appended to `source_dir`. If `model_id = "../../etc/shadow"`, the source path escapes the source directory. While `LocalFileSource` is ostensibly for testing, `import_local()` doesn't use it — but if someone wires this up with user input in `model_id`, it becomes a file read/copy vulnerability.

**Fix:** Validate that `source_path.resolve()` is inside `source_dir.resolve()`.

---

### 🟡 S-4: No verification of downloaded model integrity

**File:** `provider.py`, `HuggingFaceSource.download()`

Models are downloaded from HuggingFace with no checksum verification. HuggingFace Hub itself does provide SHA256 hashes for files — these are not used. A compromised CDN, MITM, or a poisoned HF repo could deliver a malicious model file.

PyTorch's `torch.load()` is [known to execute arbitrary code](https://github.com/pytorch/pytorch/issues/52596) when loading pickled checkpoints. A malicious `.pt` file can achieve RCE.

**This is not hypothetical.** ML supply chain attacks (e.g., poisoned Hugging Face models) are a documented attack vector.

**Fix:**
- Store expected SHA256 in `ModelCard.metadata` at install time
- Verify hash before registering
- Document that `ModelStatus.AVAILABLE` does NOT imply integrity
- Consider using `.safetensors` format which is safer than pickled PyTorch

---

### 🟡 S-5: HuggingFace Hub API — no authentication, rate limits, or token handling

**File:** `provider.py`, `HuggingFaceSource`

`HfApi()` is instantiated with no token. This:
- Limits to public model access only (may be intentional)
- Rate-limits more aggressively without auth
- If a token were added later, there's no safe place to store/inject it

No handling for 401 (token expired), 429 (rate limited), or 403 (private model).

**Fix:** Accept optional `hf_token` in `HuggingFaceSource.__init__()`, pass through to `HfApi(token=...)`. Handle HTTP errors distinctly.

---

### 🟢 S-6: `import_local()` accepts symlinks, which could escape models directory

**File:** `provider.py`, `ModelProvider.import_local()`

`shutil.copytree()` with default settings follows symlinks. A model directory containing symlinks to sensitive files (e.g., `/etc/passwd`) will copy those file contents into the models directory. This is a low-severity concern for `import_local()` since the user is explicitly choosing the source.

**Fix:** Pass `symlinks=False` (already the default for `copytree`) and consider `follow_symlinks=False` on `copy2`.

---

## 4. Missing Test Coverage

### 🔴 T-1: No test for corrupted manifest recovery

`load()` on a corrupted `models.json` is not tested. This is a crash path in production.

### 🔴 T-2: No test for path traversal in `model_id` or `relative_path`

No security tests at all. A malicious `model_id = "../../../etc"` or `relative_path = "../../evil"` is never exercised.

### 🔴 T-3: No test for concurrent access / double-install

No threading tests. The "thread-safe for reads" claim in the docstring is untested.

### 🟡 T-4: No test for partial download cleanup

The failure path of `source.download()` raising mid-install is not tested. No verification that `target_dir` is cleaned up.

### 🟡 T-5: No test for version comparison edge cases

No test for `"9.0" vs "10.0"`, `"1.10" vs "1.9"`, or `"2.0.0-rc1" vs "2.0.0"`. The broken string comparison (C-5) goes undetected.

### 🟡 T-6: No test for `check_status()` on zero-byte or corrupt file

`test_available_when_file_exists` creates `b"fake weights"` which is non-empty. A zero-byte file is never tested.

### 🟡 T-7: No test for uninstall with `delete_files=True` on a directory-type model

`test_uninstall_deletes_files` only tests file-based models. The `shutil.rmtree` path for directory models is not covered.

### 🟡 T-8: No test for `save()` atomicity / recovery after interrupted write

No test simulates a crash during `save()` and verifies the manifest is not corrupted.

### 🟡 T-9: No test for `HuggingFaceSource` when `huggingface_hub` is not installed

The graceful-degradation path (`available = False`) is not tested. Only `LocalFileSource` and `FakeRemoteSource` are used.

### 🟢 T-10: No test for `ModelCard.from_dict()` with unknown enum values

Forward-compatibility is untested. `ModelType("unknown_future_type")` will raise, not skip.

### 🟢 T-11: `test_unregister_default_promotes_next` doesn't verify correct version promotion

The test only checks that *some* model becomes default — not that it's the highest-versioned one.

### 🟢 T-12: No integration test covering full lifecycle

No test that covers: install → resolve → use path → uninstall → verify gone. Each operation is tested in isolation.

---

## 5. API Design Issues

### 🟡 A-1: `resolve()` returns `None` for "not found" — callers must check, but many won't

```python
card = registry.resolve(ModelType.ONSET_DETECTION)
# Forgot to check None...
path = registry.model_path(card)  # AttributeError: 'NoneType' has no attribute 'relative_path'
```

Returning `None` for "not found" is a footgun. Every caller must check, but there's no enforcement. Consider raising `ModelNotFoundError` by default and providing a `resolve_or_none()` variant.

---

### 🟡 A-2: `ModelRegistry` is not thread-safe but claims to be

The docstring says:
> Thread-safe for reads. Writes should happen from one thread (app init, user action).

But there is no locking. Concurrent reads during a write are not safe either — a dict being mutated mid-read in Python can raise `RuntimeError: dictionary changed size during iteration`. The "should happen from one thread" is not enforced — it's advisory only.

---

### 🟡 A-3: `register()` silently overwrites — no warning on ID collision

```python
def register(self, card: ModelCard) -> None:
    self._cards[card.id] = card  # silent overwrite
```

If you register two different models with the same ID (e.g., a CLASSIFICATION and an ONSET_DETECTION model both named `"model-v1"`), the second silently wins. No exception, no warning, no return value indicating overwrite occurred.

---

### 🟡 A-4: `uninstall()` deletes the model file but not the parent container directory

In `import_local()`, files are placed in `models_dir / card_id / filename`. When uninstalling:

```python
model_path = self._registry.model_path(card)
# model_path = models_dir / card_id / filename
if model_path.is_dir():
    shutil.rmtree(model_path)
else:
    model_path.unlink()
# models_dir / card_id/ still exists — empty directory left behind
```

The file is deleted but the `card_id` subdirectory is left as an empty orphan. Over time this leaves junk directories in the models folder.

---

### 🟡 A-5: `ModelProvider.install()` constructs `safe_id` differently than `import_local()` constructs `card_id`

- `install()`: `safe_id = model_id.replace("/", "_")`
- `import_local()`: `card_id = name.lower().replace(" ", "-").replace("/", "-")`

Two different ID generation strategies in the same class. If a user installs via HF and then imports the same model locally, the IDs will be different. No documentation of this asymmetry.

---

### 🟢 A-6: `DownloadProgress.fraction` returns `None` when `bytes_total == 0` — but `complete` status also has `bytes_total=0`

In `HuggingFaceSource.download()`:
```python
on_progress(DownloadProgress(
    model_id=model_id,
    bytes_downloaded=0,
    bytes_total=0,
    status="complete",
))
```

When `status="complete"` and `bytes_total=0`, `fraction` returns `None`. A UI showing a progress bar would show "unknown progress" at completion instead of 100%. This is confusing.

**Fix:** When `status="complete"`, return `fraction = 1.0` regardless.

---

### 🟢 A-7: `check_updates()` only compares against the *default* model of each type

```python
local = self._registry.resolve(mt)  # only returns the default
current_ver = local.version if local else None
```

If the user has `onset-v1` (default) and `onset-v2` installed (non-default), and `onset-v3` is available, `check_updates()` compares against `v1` and reports `v3` as an update — even though `v2` is already installed. It would offer `v3` as a new download even if `v2` was already the user's preferred version.

---

### 🟢 A-8: `ModelRegistry.defaults` property returns a copy but is named misleadingly

```python
@property
def defaults(self) -> dict[ModelType, str]:
    """Current default model IDs by type. Read-only copy."""
    return dict(self._defaults)
```

The docstring says "Read-only copy" but the property name `defaults` sounds mutable. Callers do `reg.defaults[ModelType.ONSET_DETECTION] = "x"` and wonder why it doesn't persist. Consider naming it `get_defaults()` or documenting this more prominently.

---

## Summary Table

| ID   | Severity | Category    | Issue |
|------|----------|-------------|-------|
| C-1  | 🔴       | Correctness | Non-atomic manifest write — corruption on crash |
| C-2  | 🔴       | Correctness | No error handling on `load()` — corrupted manifest crashes app |
| C-3  | 🔴       | Correctness | Unknown enum values crash `load()` — forward compat broken |
| C-4  | 🟡       | Correctness | Default promotion on unregister is non-deterministic |
| C-5  | 🟡       | Correctness | String-based version comparison breaks on double-digit versions |
| C-6  | 🟡       | Correctness | Duplicate progress events emitted on install |
| C-7  | 🟢       | Correctness | `resolve(model_type, model_id=...)` ignores type mismatch |
| W-1  | 🔴       | What-if     | Concurrent installs → manifest corruption, no locking |
| W-2  | 🔴       | What-if     | Failed download leaves partial files on disk, no cleanup |
| W-3  | 🔴       | What-if     | Corrupted manifest → unhandled crash (see C-2) |
| W-4  | 🔴       | What-if     | Path traversal in `model_id` → writes outside models dir |
| W-5  | 🔴       | What-if     | Disk full → partial file left, appears AVAILABLE |
| W-6  | 🔴       | What-if     | HF unreachable → silent empty list, unhelpful errors |
| W-7  | 🔴       | What-if     | Uninstall while in-use → Windows PermissionError + inconsistent state |
| W-8  | 🟡       | What-if     | Version string sort: "9.0" > "10.0" (see C-5) |
| W-9  | 🔴       | What-if     | Models dir deleted → `save()` silently recreates empty registry |
| W-10 | 🟡       | What-if     | Zero-byte/corrupt file appears AVAILABLE |
| S-1  | 🔴       | Security    | `relative_path` not bounds-checked → escape models dir |
| S-2  | 🔴       | Security    | `safe_id` sanitization bypassed by `..`, `\\`, Windows reserved names |
| S-3  | 🔴       | Security    | `LocalFileSource` path traversal via `model_id` |
| S-4  | 🟡       | Security    | No checksum verification — RCE risk via malicious `.pt` |
| S-5  | 🟡       | Security    | No HF auth, no HTTP error handling |
| S-6  | 🟢       | Security    | `import_local()` may follow symlinks out of source dir |
| T-1  | 🔴       | Tests       | Corrupted manifest crash path untested |
| T-2  | 🔴       | Tests       | Path traversal attacks untested |
| T-3  | 🔴       | Tests       | Concurrent access untested |
| T-4  | 🟡       | Tests       | Partial download cleanup untested |
| T-5  | 🟡       | Tests       | Version comparison edge cases untested |
| T-6  | 🟡       | Tests       | Zero-byte file status untested |
| T-7  | 🟡       | Tests       | Directory-type model uninstall untested |
| T-8  | 🟡       | Tests       | Interrupted `save()` recovery untested |
| T-9  | 🟡       | Tests       | Missing `huggingface_hub` graceful degradation untested |
| T-10 | 🟢       | Tests       | Unknown enum values in manifest untested |
| T-11 | 🟢       | Tests       | Default promotion correctness untested |
| T-12 | 🟢       | Tests       | No full lifecycle integration test |
| A-1  | 🟡       | API         | `resolve()` returns `None` — footgun for callers |
| A-2  | 🟡       | API         | Registry not actually thread-safe despite docstring claim |
| A-3  | 🟡       | API         | `register()` silently overwrites on ID collision |
| A-4  | 🟡       | API         | Uninstall leaves empty parent directory behind |
| A-5  | 🟡       | API         | Inconsistent ID generation between `install()` and `import_local()` |
| A-6  | 🟢       | API         | `fraction` is `None` at completion status (should be 1.0) |
| A-7  | 🟢       | API         | `check_updates()` compares default only, ignores non-default installs |
| A-8  | 🟢       | API         | `defaults` property name implies mutability |

**🔴 Critical (must fix):** 14 issues  
**🟡 Should fix:** 16 issues  
**🟢 Nice to have:** 10 issues  

---

## Recommended Fix Priority Order

1. **S-1, S-2, S-3, W-4** — Path traversal / arbitrary file writes. Ship this and you have an RCE vector.
2. **C-1, C-2** — Atomic writes and corrupted manifest recovery. These are data loss bugs at startup.
3. **W-1** — Add a lock. Concurrent installs corrupt data.
4. **W-2, W-5** — Cleanup partial downloads on failure.
5. **C-3** — Forward-compatibility: unknown enums should skip, not crash.
6. **S-4** — Checksum verification. PyTorch pickle RCE is real.
7. **W-7** — In-use reference counting before deletion.
8. **C-5, W-8** — Fix version comparison (`packaging.version.Version`).
9. **T-1, T-2, T-3** — Add the missing critical tests.
10. Everything else — important but lower blast radius.

---

*End of audit. Generated 2026-03-29.*
