# EchoZero Full Codebase Audit — Part 2 (Deep Dive)

**Date:** 2026-03-30
**Auditor:** Chonch
**Scope:** Processors, pipelines, repositories, templates, params — everything not covered in Part 1
**Extends:** `codebase-audit-2026-03-30.md`

---

## 9. Processors

### 🔴 PR1: `_default_classify` uses `torch.load(weights_only=False)` — arbitrary code execution

**File:** `processors/pytorch_audio_classify.py:78`

```python
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
```

`weights_only=False` allows pickle deserialization, which can execute arbitrary Python code. A malicious `.pth` file (from a user import, shared project, or compromised model download) achieves full RCE. PyTorch explicitly warns against this. The model registry audit (S-4) already flagged this at the download level — this is the load-side of the same vulnerability.

**Fix:** Use `weights_only=True` (default in PyTorch 2.6+). For models that require pickle, use `torch.load(..., weights_only=True)` with explicit exception handling for `UnpicklingError` and a "this model requires legacy format" warning.

---

### 🔴 PR2: `SeparateAudioProcessor` temp directory never cleaned up

**File:** `processors/separate_audio.py:223`

```python
output_dir = tempfile.mkdtemp(prefix=f"ez_stems_{block_id}_")
```

The comment says "The caller owns cleanup after the files are consumed." But the Orchestrator's `_handle_persist_as_song_version()` is a no-op stub that logs and returns empty. No caller ever cleans up. Each separation creates 4+ stem files (100MB+ for a 4-min song). After 10 analyses, that's 1GB+ of orphaned temp files.

**Fix:** Add cleanup to the Orchestrator after output persistence. Or: use a working-dir-scoped temp directory (e.g., `session.working_dir / "tmp"`) that gets cleaned on session close.

---

### 🟡 PR3: `_default_classify` classification logic is hardcoded dummy

**File:** `processors/pytorch_audio_classify.py:131-138`

```python
def _predict_event_class(event, audio_file, model, config, device):
    time = event.time
    if time < 1.0: return "kick"
    elif time < 3.0: return "snare"
    else: return "hihat"
```

The model is loaded but never actually used for inference. Every event is classified by a time-based rule. The `model.eval()` and `model.to(device)` calls burn GPU memory for no reason.

**Fix:** Either implement actual inference (forward pass with audio features), or remove the model loading and document this as a stub. The injectable `classify_fn` pattern works — just make the default honest about being a stub.

---

### 🟡 PR4: `_default_separate` always writes WAV regardless of `output_format` setting

**File:** `processors/separate_audio.py:166`

```python
# Note: output_format and mp3_bitrate are accepted for future use.
# V1 always writes WAV. MP3 encoding requires pydub or ffmpeg.
sf.write(file_path, audio_np, sample_rate)
```

The user can set `output_format="mp3"` and it's validated as valid, but WAV is always written. The file extension will be `.mp3` (from `ext = "mp3" if output_format == "mp3" else "wav"`) but the content is WAV. A `.mp3` file containing WAV data will confuse any downstream consumer.

**Fix:** Either raise an error when `output_format="mp3"` ("MP3 export not yet supported"), or force `ext = "wav"` regardless of setting.

---

### 🟡 PR5: `SeparateAudioProcessor` two-stems mode constructs `other_tensor` on CPU even if model ran on CUDA

**File:** `processors/separate_audio.py:146`

```python
other_tensor = torch.zeros_like(next(iter(separated.values())))
for name, tensor in separated.items():
    if name != two_stems:
        other_tensor += tensor
```

If separation ran on CUDA, `separated` tensors are on CUDA. `zeros_like` creates a CUDA tensor, `+=` is a CUDA operation. Then `tensor.cpu().numpy()` in `_write_audio` moves it to CPU. This is correct but the intermediate CUDA tensor is unnecessary — we could just move to CPU first and sum in numpy.

Not a bug, just unnecessary GPU memory usage.

---

### 🟡 PR6: All processors catch `Exception` broadly around the main work

Every processor wraps its core logic in `try/except Exception as exc: return err(ExecutionError(...))`. This is the intended pattern, BUT it means `KeyboardInterrupt` and `SystemExit` are not caught (they inherit from `BaseException`, not `Exception`), which is correct. However, the `from exc` chaining is inconsistent — some processors chain (`from e`), others don't. The E1 fix in the engine doesn't help if processors themselves lose the traceback.

**Fix:** Ensure all processor `except Exception` blocks use `from exc` chaining.

---

## 10. Pipeline Registry & Templates

### 🟡 REG1: `PipelineRegistry` is a global singleton with no reset mechanism

**File:** `pipelines/registry.py:120`

```python
_registry = PipelineRegistry()
```

The global `_registry` is populated by `@pipeline_template` decorators at import time. There's no `reset()` or `clear()` method. In tests, importing any template module permanently registers it in the global registry. Tests that check `get_registry().list()` may see different results depending on import order.

**Fix:** Add `PipelineRegistry.clear()` and use it in test fixtures. Or: make `get_registry()` return a thread-local instance in tests.

---

### 🟡 REG2: `build_pipeline()` wraps legacy Graph returns by reaching into Pipeline._graph

**File:** `pipelines/registry.py:80`

```python
p = EnginePipeline(id=self.id, name=self.name)
p._graph = result
return p
```

Same private access pattern flagged in S2. If Pipeline's constructor is updated (per the S2 fix) to accept `graph=`, this should use the public API.

**Fix:** Use `Pipeline(id=self.id, name=self.name, graph=result)` after the S2 fix lands.

---

### 🟡 REG3: `build()` and `build_pipeline()` duplicate 80% of their logic

**File:** `pipelines/registry.py:33-82`

Both methods build kwargs from knobs + bindings, call `self.builder(**kwargs)`, and handle Pipeline vs Graph returns. The only difference is what they return (Graph vs Pipeline).

**Fix:** Extract shared logic into a private `_build()` method that returns the raw builder result. `build()` extracts the graph, `build_pipeline()` wraps if needed.

---

## 11. Domain & Params

### 🟡 PAR1: `Knob.default` has no type constraint — can be any object

**File:** `pipelines/params.py:68`

```python
default: Any
```

A `Knob(default={"nested": "dict"})` is valid. `validate_bindings()` checks `type(pdef.default)` for type matching, which means it compares against `dict` — so any dict passes. But `BlockSettings.__hash__` would crash if this dict ends up in settings (S4 from Part 1).

**Fix:** Document that Knob defaults should be primitive types (str, int, float, bool) or tuple. Add a warning in `knob()` for complex types.

---

### 🟡 PAR2: `validate_bindings()` doesn't validate `MULTI_SELECT` default values

**File:** `pipelines/params.py:215-220`

```python
if pdef.widget == KnobWidget.MULTI_SELECT and pdef.options is not None:
    if isinstance(value, (list, tuple)):
        for item in value:
            if item not in pdef.options:
```

Only validates items if the value is a list/tuple. But the Knob's `default` could be a list like `["a", "b"]` — and `_validate()` at construction time doesn't check that default items are in options for MULTI_SELECT.

**Fix:** Add default validation for MULTI_SELECT in `_validate()`:
```python
if widget == KnobWidget.MULTI_SELECT and options is not None:
    if isinstance(default, (list, tuple)):
        for item in default:
            if item not in options:
                errors.append(f"default item '{item}' not in options")
```

---

## 12. Repositories

### 🟡 REPO1: No SQL injection risk BUT no parameterized LIKE queries

All repositories use parameterized queries (`?` placeholders), which is correct. No SQL injection risk.

However, none of the repositories support search/filter by name — when the UI needs to search songs by title, it'll need a LIKE query. When that's added, ensure the `%` and `_` wildcard characters in user input are escaped.

Not a current bug — just a note for future work.

---

### 🟡 REPO2: `SongRepository.delete()` cascades via FK but doesn't clean up audio files

**File:** `persistence/repositories/song.py:52`

```python
def delete(self, song_id: str) -> None:
    self._execute("DELETE FROM songs WHERE id = ?", (song_id,))
```

FK cascade deletes song_versions, layers, and takes from the DB. But the audio files in the working directory (`audio/<hash>.wav`) remain. Content-addressed storage means the file might be shared with other songs (same audio file imported twice). A naive cleanup would delete files still referenced by other songs.

**Fix:** After delete, run an orphan audio cleanup: find files in `audio/` not referenced by any remaining song_version. This is a service-layer concern, not repository.

---

### 🟡 REPO3: `LayerRepository.reorder()` doesn't validate layer_ids belong to song_version_id

**File:** `persistence/repositories/layer.py:93`

```python
def reorder(self, song_version_id: str, layer_ids: list[str]) -> None:
    for i, layer_id in enumerate(layer_ids):
        self._execute(
            'UPDATE layers SET "order" = ? WHERE id = ? AND song_version_id = ?',
            (i, layer_id, song_version_id),
        )
```

The WHERE clause includes `song_version_id` so a mismatched layer_id just updates 0 rows (harmless). But if `layer_ids` is missing some layers, those layers keep their old order values, potentially creating duplicate order values.

**Fix:** Validate that `layer_ids` contains all layer IDs for the song_version before reordering.

---

## 13. Session / Lifecycle

### 🟡 SESS1: `ProjectSession.import_song()` doesn't validate audio file format

**File:** `persistence/session.py:220`

No check that `audio_source` is actually an audio file. If someone passes a `.txt` file, `import_audio()` copies it into the audio directory, and `scan_audio_metadata()` raises `RuntimeError`. The file is already copied by then — orphaned non-audio file in the content store.

**Fix:** Check file extension or try scanning BEFORE copying:
```python
# Validate audio before import
scan_audio_metadata(audio_source, scan_fn=scan_fn)  # raises if not valid
audio_rel_path, audio_hash = import_audio(audio_source, self.working_dir)
```

---

### 🟡 SESS2: `add_song_version()` copies PipelineConfigs but doesn't update audio_file in graph

**File:** `persistence/session.py:285`

```python
new_config = _replace(
    config,
    id=uuid.uuid4().hex,
    song_version_id=version.id,
    created_at=now,
    updated_at=now,
)
self.pipeline_configs.create(new_config)
```

The config is copied with a new ID and version_id, but `graph_json` still contains the OLD audio file path from the source version. When `Orchestrator.execute()` runs, it overrides the LoadAudio block's `file_path` with the new version's audio — so this is safe at execution time. But if anyone inspects the stored `graph_json` (e.g., for UI display of settings), they'll see the old path.

**Fix:** Either update the audio path in the copied config's graph_json, or document that graph_json paths are overridden at execution time and shouldn't be read for display.

---

## 14. Misc

### 🟡 MISC1: `editor/pipeline.py` and `pipelines/pipeline.py` — naming collision

Two classes both named `Pipeline` in different modules:
- `echozero.editor.pipeline.Pipeline` — command dispatcher for the editor (mutations)
- `echozero.pipelines.pipeline.Pipeline` — engine-level pipeline with graph + outputs

The Coordinator imports `from echozero.editor.pipeline import Pipeline`. The Orchestrator uses `from echozero.pipelines.pipeline import Pipeline`. This works but is confusing. The codebase already uses `EnginePipeline` as an alias in some places.

**Fix:** Rename `editor/pipeline.py`'s class to `DocumentPipeline` or `PipelineEditor` to eliminate the ambiguity.

---

### 🟡 MISC2: `RuntimeBus.unsubscribe()` uses `list.remove()` — O(n) and crashes on double-remove

**File:** `progress.py:73`

```python
def unsubscribe(self, callback):
    self._subscribers.remove(callback)
```

If the same callback is unsubscribed twice, `list.remove()` raises `ValueError`. EventBus handles this (via try/except in DirtyTracker._unsubscribe), but RuntimeBus doesn't.

**Fix:** Use `discard`-like behavior:
```python
try:
    self._subscribers.remove(callback)
except ValueError:
    pass
```

---

## Summary — Part 2 Additional Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| PR1 | 🔴 | Classifier | `torch.load(weights_only=False)` — RCE via malicious model |
| PR2 | 🔴 | Separator | Temp stem files never cleaned up — disk leak |
| PR3 | 🟡 | Classifier | Dummy classification ignores loaded model |
| PR4 | 🟡 | Separator | MP3 format writes WAV with .mp3 extension |
| PR5 | 🟡 | Separator | Unnecessary GPU memory in two-stems mode |
| PR6 | 🟡 | Processors | Inconsistent exception chaining across processors |
| REG1 | 🟡 | Registry | Global singleton with no test reset |
| REG2 | 🟡 | Registry | Private _graph access in build_pipeline |
| REG3 | 🟡 | Registry | build/build_pipeline logic duplication |
| PAR1 | 🟡 | Params | Knob.default accepts arbitrary complex types |
| PAR2 | 🟡 | Params | MULTI_SELECT default items not validated against options |
| REPO2 | 🟡 | Repos | Song delete doesn't clean up audio files |
| REPO3 | 🟡 | Repos | reorder() doesn't validate all IDs present |
| SESS1 | 🟡 | Session | No audio format validation before import |
| SESS2 | 🟡 | Session | Copied PipelineConfigs retain old audio paths |
| MISC1 | 🟡 | Naming | Two classes named Pipeline — confusing |
| MISC2 | 🟡 | RuntimeBus | Double-unsubscribe crashes |

**🔴 Must-fix count: 2**
**🟡 Should-fix count: 15**

---

## Combined Priority (Part 1 + Part 2)

### Batch 3 — After Batches 1+2 land (~30 min)
PR1 (torch.load security), PR2 (temp cleanup), PR4 (MP3 extension lie), S4 (BlockSettings hash crash)

### Batch 4 — Robustness (~30 min)
P4 (autosave threading), P5 (WAL checkpoint), P8 (NULL data_json), ED4 (cancel wait), MISC2 (double-unsubscribe), REG2 (private access after S2 fix)

### Batch 5 — Quality/Polish (~20 min)
O3 (dedup analyze/execute), REG3 (build dedup), SESS1 (audio validation), P6 (migration guard), A2+A3 (archive completeness), PR6 (exception chaining)

---

*End of Part 2. Generated 2026-03-30.*
