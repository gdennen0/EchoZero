param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$Folder,

    [string]$Root = ".",
    [double]$Val = 0.15,
    [double]$Test = 0.10,
    [int]$Seed = 42,
    [string]$Balance = "none",
    [int]$Epochs = 4,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.01,
    [int]$SampleRate = 22050,
    [int]$MaxLength = 22050,
    [int]$NFft = 2048,
    [int]$HopLength = 512,
    [int]$NMels = 128,
    [int]$Fmax = 8000,
    [ValidateSet("none", "balanced")]
    [string]$ClassWeighting = "none",
    [ValidateSet("none", "oversample")]
    [string]$Rebalance = "none",
    [switch]$AugmentTrain,
    [double]$AugmentNoiseStd = 0.02,
    [double]$AugmentGainJitter = 0.10,
    [int]$AugmentCopies = 1,
    [switch]$NextLevel,
    [ValidateSet("baseline_v1", "stronger_v1")]
    [string]$TrainerProfile = "baseline_v1",
    [ValidateSet("sgd_constant", "sgd_optimal")]
    [string]$Optimizer = "sgd_constant",
    [double]$RegularizationAlpha = 0.0001,
    [switch]$AverageWeights,
    [Nullable[int]]$EarlyStoppingPatience = $null,
    [Nullable[int]]$MinEpochs = $null,
    [switch]$SyntheticMixEnabled,
    [double]$SyntheticMixRatio = 0.0,
    [Nullable[int]]$SyntheticMixCap = $null
)

$arguments = @(
    "-m", "echozero.foundry.cli",
    "--root", $Root,
    "train-folder",
    $Name,
    $Folder,
    "--val", $Val.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--test", $Test.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--seed", $Seed,
    "--balance", $Balance,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--learning-rate", $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--sample-rate", $SampleRate,
    "--max-length", $MaxLength,
    "--n-fft", $NFft,
    "--hop-length", $HopLength,
    "--n-mels", $NMels,
    "--fmax", $Fmax,
    "--class-weighting", $ClassWeighting,
    "--rebalance", $Rebalance,
    "--augment-noise-std", $AugmentNoiseStd.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--augment-gain-jitter", $AugmentGainJitter.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--augment-copies", $AugmentCopies,
    "--trainer-profile", $TrainerProfile,
    "--optimizer", $Optimizer,
    "--regularization-alpha", $RegularizationAlpha.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

if ($AugmentTrain) { $arguments += "--augment-train" }
if ($NextLevel) { $arguments += "--next-level" }
if ($AverageWeights) { $arguments += "--average-weights" }
if ($EarlyStoppingPatience -ne $null) { $arguments += @("--early-stopping-patience", $EarlyStoppingPatience) }
if ($MinEpochs -ne $null) { $arguments += @("--min-epochs", $MinEpochs) }
if ($SyntheticMixEnabled) { $arguments += "--synthetic-mix-enabled" }
if ($SyntheticMixRatio -gt 0) {
    $arguments += @("--synthetic-mix-ratio", $SyntheticMixRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($SyntheticMixCap -ne $null) { $arguments += @("--synthetic-mix-cap", $SyntheticMixCap) }

$raw = & python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$payload = $raw | ConvertFrom-Json
$validation = @()
foreach ($artifactId in $payload.artifact_ids) {
    $result = & python -m echozero.foundry.cli --root $Root validate-artifact $artifactId
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $validation += ($result | ConvertFrom-Json)
}

[pscustomobject]@{
    train = $payload
    validation = $validation
} | ConvertTo-Json -Depth 8
