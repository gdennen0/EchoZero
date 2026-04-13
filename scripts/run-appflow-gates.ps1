[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv/Scripts/python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$releaseRoot = Join-Path $repoRoot "artifacts/releases/test"
$fallbackReleaseRoot = Join-Path $repoRoot "build/test-release/releases/test"
$releaseFolder = $null
$reportPath = $null
$failedStep = $null

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Label"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        $script:failedStep = $Label
        throw "Step failed: $Label (exit_code=$LASTEXITCODE)"
    }
}

function Get-LatestReleaseFolder {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Roots
    )

    foreach ($root in $Roots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $latest = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($null -ne $latest) {
            return $latest.FullName
        }
    }

    return $null
}

Write-Host "Using python: $pythonExe"

try {
    Invoke-Step "appflow lane" { & $pythonExe -m echozero.testing.run --lane appflow }
    Invoke-Step "appflow-sync lane" { & $pythonExe -m echozero.testing.run --lane appflow-sync }
    Invoke-Step "appflow-osc lane" { & $pythonExe -m echozero.testing.run --lane appflow-osc }
    Invoke-Step "appflow-protocol lane" { & $pythonExe -m echozero.testing.run --lane appflow-protocol }
    Invoke-Step "appflow-all lane" { & $pythonExe -m echozero.testing.run --lane appflow-all }
    Invoke-Step "build test release" { & powershell -File (Join-Path $PSScriptRoot "build-test-release.ps1") }
    Invoke-Step "smoke test release" { & powershell -File (Join-Path $PSScriptRoot "smoke-test-release.ps1") }

    $releaseFolder = Get-LatestReleaseFolder -Roots @($releaseRoot, $fallbackReleaseRoot)
    if ($releaseFolder) {
        $reportPath = Join-Path $releaseFolder "smoke-report.json"
    }

    Write-Host ""
    Write-Host "APPFLOW GATES PASS"
    if ($releaseFolder) {
        Write-Host "release_folder=$releaseFolder"
    }
    if ($reportPath -and (Test-Path $reportPath)) {
        Write-Host "report=$reportPath"
    }
    exit 0
}
catch {
    $failedStep = if ($failedStep) { $failedStep } else { $_.Exception.Message }
    $releaseFolder = Get-LatestReleaseFolder -Roots @($releaseRoot, $fallbackReleaseRoot)
    if ($releaseFolder) {
        $reportPath = Join-Path $releaseFolder "smoke-report.json"
    }

    Write-Host ""
    Write-Host "APPFLOW GATES FAIL"
    Write-Host "reason=$failedStep"
    if ($releaseFolder) {
        Write-Host "release_folder=$releaseFolder"
    }
    if ($reportPath -and (Test-Path $reportPath)) {
        Write-Host "report=$reportPath"
    }
    exit 1
}
