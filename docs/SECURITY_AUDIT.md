# EchoZero Security Audit

This document summarizes a security audit of the EchoZero codebase performed in February 2026. It covers authentication, secrets handling, process execution, persistence, and UI data flow.

---

## Summary

| Severity | Count | Notes |
|----------|--------|--------|
| High     | 0     | None identified |
| Medium   | 1     | shell=True with file-driven commands (refactoring tracker) |
| Low      | 2     | HTML injection in model details; CORS * on localhost |
| Fixed    | 2     | Hardcoded app secret (now required from env); GCP code removed |

---

## 1. Authentication and Secrets

### 1.1 App secret (Fixed)

**Location:** `main_qt.py`

The app no longer uses a hardcoded default for `MEMBERSTACK_APP_SECRET`. The secret must be set in the environment or in a `.env` file (see `.env.example`). If unset, startup fails with a clear error. The secret should be shared with developers via a password manager or other secure channel; `.env` is in `.gitignore` and must not be committed.

---

### 1.2 Credential storage

**Location:** `src/infrastructure/auth/token_storage.py`

TokenStorage uses the `keyring` library and OS keychain (Windows Credential Manager, macOS Keychain, Linux Secret Service). Credentials are not stored in plain config files. No credentials are logged (only success/failure messages).

**Status:** Acceptable.

---

### 1.3 Local auth callback server

**Location:** `src/infrastructure/auth/local_auth_server.py`

- Binds to `127.0.0.1` only (no external exposure).
- One-time nonce prevents replay/CSRF.
- Only POST `/callback` with valid nonce is accepted; GET callback is rejected.

**CORS:** `Access-Control-Allow-Origin: *` is set for the callback. This is acceptable for a localhost-only callback server used by the desktop app’s own login page.

**Status:** Acceptable.

---

## 2. Process Execution and Injection

### 2.1 Refactoring tracker – shell=True with file-driven commands (Medium)

**Location:** `AgentAssets/scripts/refactoring_tracker.py` (lines 529–537)

Tasks are loaded from `AgentAssets/data/refactoring_progress.json` via `Task.from_dict()`. The `verification_commands` list is executed with:

```python
result = subprocess.run(cmd, shell=True, ...)
```

**Issue:** If the JSON file is modified (e.g. by a contributor or tool), arbitrary commands can be run when the refactoring tracker executes verification. Risk is limited to developers or automation that run this script and use a modified progress file.

**Recommendation:** Prefer `shell=False` and pass a list of arguments (e.g. `["python3", "-c", "..."])` so the command is not parsed by a shell. If shell is required, restrict who can modify `refactoring_progress.json` and consider validating/sanitizing commands (e.g. allowlist of commands or patterns).

---

### 2.2 Other subprocess usage

- **separator_block.py:** Demucs is invoked via `subprocess.Popen(cmd, ...)` with `cmd` as a list; no `shell=True`. Arguments (paths, model name) come from block metadata but are passed as separate list elements, so shell injection is not applicable.

**Status:** Acceptable aside from refactoring tracker (above). (GCP cloud block was removed from the codebase.)

---

## 3. Temp Files and Credentials

(GCP cloud block and its temp credential file handling were removed from the codebase.)

---

## 4. Persistence and Injection

### 4.1 SQL

- **database.py:** `clear_runtime_tables()` uses `cursor.execute(f"DELETE FROM {table}")` where `table` is from a fixed tuple of table names. No user input.
- **base_repository.py:** Uses f-strings for `table_name` and `id_column`; these are class attributes set by subclasses, not user input.
- Other repositories use parameterized queries (`?` placeholders). No SQL injection identified from user-controlled input.

**Status:** Acceptable.

---

### 4.2 JSON deserialization

Usage of `json.loads()` on project data, preferences, and API responses is present throughout. JSON does not expose arbitrary code execution. No use of `pickle.loads` or `yaml.load` (unsafe) on untrusted data was found.

**Status:** Acceptable.

---

## 5. UI and HTML

### 5.1 Model details HTML – possible XSS (Low)

**Location:** `ui/qt_gui/block_panels/pytorch_audio_classify_panel.py` – `_build_model_details_html()`

Checkpoint data (e.g. `classes`, `config`, `training_date`, `target_class`) is interpolated into HTML via `row()` and similar helpers and then shown with `setHtml()`. Values are not HTML-escaped.

**Issue:** A maliciously crafted checkpoint file (e.g. `classes = ["<script>...</script>"]`) could inject script into the model details view when the user loads that model.

**Recommendation:** Escape all user- or file-derived values before inserting into HTML (e.g. `html.escape(str(value))` or a small helper). Restrict HTML to a minimal subset if you need any formatting.

---

### 5.2 Add action dialog

**Location:** `ui/qt_gui/views/add_action_dialog.py`

`setHtml()` is used with block name, type, and action name/description. These come from the block registry and action definitions (code/config), not arbitrary end-user content. Risk is low unless the registry can be populated from untrusted input.

**Status:** Acceptable with current data source; apply escaping if sources change.

---

## 6. Paths and File Access

### 6.1 LoadAudio block

**Location:** `src/application/blocks/load_audio_block.py`

`audio_path` is taken from block metadata and passed to `AudioDataItem.load_audio(file_path)`. There is no path normalization or check that the path stays under a project or allowed directory.

**Issue:** A project file (or UI) could set `audio_path` to something like `../../../etc/passwd` or another sensitive path, and the application would attempt to read it as audio. Impact is limited to what the process can read and to misrepresentation of non-audio files as audio.

**Recommendation:** Normalize the path and enforce that it lies under a project root or an explicit allowlist of directories; reject or sanitize path traversal sequences.

---

## 7. Dependencies

**Location:** `requirements.txt`

- Versions use minimum bounds (e.g. `>=`) or pins (e.g. `tensorflow==2.15.0`, `protobuf==3.20.3`) where needed for compatibility.
- No obviously vulnerable patterns were identified in the listed packages; a dedicated dependency scan (e.g. `pip audit`, Dependabot, or Snyk) should be run regularly.

**Recommendation:** Run `pip audit` (or equivalent) in CI and address reported vulnerabilities; consider pinning critical dependencies to exact versions and reviewing updates.

---

## 8. Other Notes

- **AuthService** (`src/shared/application/services/auth_service.py`): Sends login credentials over HTTP when `ECHOZERO_BACKEND_URL` points to an http:// URL. Ensure production uses HTTPS and that the backend URL is trusted.
- **Memberstack verify URL:** Default is `https://echozero-auth.speeoflight.workers.dev`; verification is done over HTTPS.
- **Logging:** No logging of tokens, passwords, or credential payloads was found; only high-level success/failure and error messages.

---

## 9. Recommended Actions (Priority Order)

1. **Medium:** Remove or restrict the hardcoded `MEMBERSTACK_APP_SECRET` fallback; require explicit config in production.
2. **Medium:** Refactoring tracker: avoid `shell=True`; use list-form `subprocess` and/or restrict and validate `verification_commands` (and who can change the progress file).
3. **Medium:** GCP block: Harden temp credential file cleanup (specific exception, logging, optional retry); avoid bare `except`.
4. **Low:** Escape checkpoint/model-derived strings before inserting into HTML in the PyTorch classify panel.
5. **Low:** LoadAudio: Validate/normalize `audio_path` and restrict to project or allowlisted directories.
6. **Ongoing:** Run dependency checks (e.g. `pip audit`) in CI and before releases.

---

*Audit date: February 2026. Re-run after major auth, execution, or persistence changes.*
