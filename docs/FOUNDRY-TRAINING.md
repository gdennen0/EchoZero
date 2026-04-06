# Foundry End-to-End Training

This flow matches the current `ez-foundry` CLI and produces a dataset version, split plan, train run, exported model, and compatibility validation report.

## Windows desktop run path

From the repo root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m echozero.foundry.cli --root . ui
```

Optional helper script from the repo root:

```powershell
.\scripts\run-foundry-ui.ps1
```

What Griff should expect in the desktop UI:

- `Dataset` tab: choose a folder, click `Create + Ingest`, then `Plan Split / Balance`.
- `Dataset` tab: use the new `Dataset` and `Version` selectors when the workspace contains multiple imported datasets or multiple dataset versions.
- `Run` tab: `Start Run` and `Create + Start` now run in the background, keep the window responsive, and stream live run-stage updates into the activity log and status line.
- `Run` tab: use the new open actions to jump straight to the selected run's exports folder, `metrics.json`, and `run_summary.json`.
- `Artifacts` tab and right-side workspace panel: inspect the latest run, eval report, artifact manifest, and validation status. Use `Open Artifact Manifest` to open the selected manifest directly, or `Open Latest Artifact Package` to jump to the most recent persisted export package in one click.
- Right-side `Background Queue`: shows queued, active, and recent runs with created/updated timestamps. The current active run is marked `ACTIVE`.
- Right-side `Background Queue`: select a run and use `Cancel Run` to cancel queued or in-flight work. The current background worker exits cooperatively and the UI stays responsive while status updates stream back.
- Right-side `Background Queue`: use `Retry / Requeue` on `failed` or `canceled` runs to move the same run id back to `queued` for a minimal safe retry.
- `Refresh` queues a non-blocking state reload from `foundry\state\` and `foundry\runs\`.
- If an export or manifest file is missing, the UI keeps the window responsive and reports the missing path in the status/error area instead of crashing.

## Prerequisites

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## One-command train-folder workflow

Default baseline profile:

```powershell
python -m echozero.foundry.cli --root . train-folder "Drum OneShots" ".\data\drum-oneshots" --val 0.2 --test 0.2 --epochs 4
```

Improved opt-in profile:

```powershell
python -m echozero.foundry.cli --root . train-folder "Drum OneShots" ".\data\drum-oneshots" --val 0.2 --test 0.2 --epochs 8 --next-level --trainer-profile stronger_v1 --optimizer sgd_optimal --average-weights --early-stopping-patience 3 --min-epochs 3 --regularization-alpha 0.00005
```

Synthetic mix options:

```powershell
python -m echozero.foundry.cli --root . train-folder "Drum OneShots" ".\data\drum-oneshots" --epochs 8 --trainer-profile stronger_v1 --synthetic-mix-enabled --synthetic-mix-ratio 0.25 --synthetic-mix-cap 200
```

Promotion gates with reference comparison:

```powershell
python -m echozero.foundry.cli --root . train-folder "Drum OneShots" ".\data\drum-oneshots" --epochs 8 --trainer-profile stronger_v1 --synthetic-mix-enabled --synthetic-mix-ratio 0.25 --gate-macro-f1-floor 0.80 --gate-max-regression-vs-reference 0.03 --gate-max-real-vs-synth-gap 0.05 --gate-per-class-recall-floor kick=0.75 --gate-per-class-recall-floor snare=0.75 --reference-run-id RUN_ID
```

The command prints JSON containing:

- `dataset_id`
- `dataset_version_id`
- `run_id`
- `artifact_ids`
- `exports_dir`

When promotion settings are present, Foundry also writes the gate result and any reference comparison summary into `metrics.json`, `run_summary.json`, and the artifact manifest.

## Windows convenience script

Baseline run:

```powershell
.\scripts\foundry-train-folder.ps1 -Name "Drum OneShots" -Folder ".\data\drum-oneshots"
```

Improved run with synthetic mix:

```powershell
.\scripts\foundry-train-folder.ps1 -Name "Drum OneShots" -Folder ".\data\drum-oneshots" -Epochs 8 -NextLevel -TrainerProfile stronger_v1 -Optimizer sgd_optimal -AverageWeights -EarlyStoppingPatience 3 -MinEpochs 3 -RegularizationAlpha 0.00005 -SyntheticMixEnabled -SyntheticMixRatio 0.25 -SyntheticMixCap 200
```

The script runs `train-folder`, then validates every returned artifact id with `validate-artifact`.

Desktop UI helper:

```powershell
.\scripts\run-foundry-ui.ps1
```

## Explicit multi-step workflow

```powershell
python -m echozero.foundry.cli --root . create-dataset "Drum OneShots"
python -m echozero.foundry.cli --root . ingest-folder DATASET_ID ".\data\drum-oneshots"
python -m echozero.foundry.cli --root . plan-version DATASET_VERSION_ID --val 0.2 --test 0.2 --seed 42 --balance none
python -m echozero.foundry.cli --root . create-run DATASET_VERSION_ID "{\"schema\":\"foundry.train_run_spec.v1\",\"classificationMode\":\"multiclass\",\"data\":{\"datasetVersionId\":\"DATASET_VERSION_ID\",\"sampleRate\":22050,\"maxLength\":22050,\"nFft\":2048,\"hopLength\":512,\"nMels\":128,\"fmax\":8000},\"training\":{\"epochs\":8,\"batchSize\":4,\"learningRate\":0.01,\"seed\":42,\"trainerProfile\":\"stronger_v1\",\"optimizer\":\"sgd_optimal\",\"averageWeights\":true,\"earlyStoppingPatience\":3,\"minEpochs\":3,\"regularizationAlpha\":0.00005,\"classWeighting\":\"balanced\",\"rebalanceStrategy\":\"oversample\",\"augmentTrain\":true,\"augmentNoiseStd\":0.03,\"augmentGainJitter\":0.15,\"augmentCopies\":2,\"syntheticMix\":{\"enabled\":true,\"ratio\":0.25,\"cap\":200}}}"
python -m echozero.foundry.cli --root . start-run RUN_ID
python -m echozero.foundry.cli --root . validate-artifact ARTIFACT_ID
```

## Artifact validation

Validate the exported artifact from the `train-folder` output:

```powershell
python -m echozero.foundry.cli --root . validate-artifact ARTIFACT_ID
```

Expected success shape:

```json
{
  "ok": true,
  "errors": [],
  "warnings": []
}
```

The exported files are under `foundry\runs\RUN_ID\exports\`:

- `model.pth`
- `metrics.json`
- `run_summary.json`
- `art_*.manifest.json`

## Reproducibility notes

- Use the same `--root`, dataset folder, and `--seed` to keep split planning and training deterministic.
- `--next-level` only changes data balancing and augmentation defaults.
- The stronger optimization path is only enabled when `--trainer-profile stronger_v1` is passed.
