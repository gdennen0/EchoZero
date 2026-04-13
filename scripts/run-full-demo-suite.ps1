[CmdletBinding()]
param(
    [string]$AudioPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_OPENGL = "software"

$arguments = @(
    "-m",
    "echozero.testing.demo_suite",
    "--output-root",
    "artifacts/demo-suite",
    "--record"
)

if ($AudioPath) {
    $arguments += @("--audio-path", $AudioPath)
}

$output = & $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "demo suite runner failed with exit code $LASTEXITCODE"
}

$repoRunFolder = $null
$manifestPath = $null
foreach ($line in $output) {
    Write-Host $line
    if ($line -like "run_folder=*") {
        $repoRunFolder = $line.Substring("run_folder=".Length)
    }
    elseif ($line -like "manifest=*") {
        $manifestPath = $line.Substring("manifest=".Length)
    }
}

if (-not $repoRunFolder) {
    throw "demo suite runner did not print run_folder"
}

$stagedRunFolder = ""
$repoTmpRoot = Join-Path $repoRoot ".openclaw\workspace\tmp"
if (Test-Path $repoTmpRoot) {
    $stagedRoot = Join-Path $repoTmpRoot "demo-suite"
    New-Item -ItemType Directory -Path $stagedRoot -Force | Out-Null
    $runId = Split-Path $repoRunFolder -Leaf
    $stagedRunFolder = Join-Path $stagedRoot $runId
    if (Test-Path $stagedRunFolder) {
        Remove-Item -Recurse -Force $stagedRunFolder
    }
    Copy-Item -Recurse -Force $repoRunFolder $stagedRunFolder
}

Write-Host "repo_run_folder=$repoRunFolder"
Write-Host "staged_run_folder=$stagedRunFolder"
Write-Host "manifest=$manifestPath"
