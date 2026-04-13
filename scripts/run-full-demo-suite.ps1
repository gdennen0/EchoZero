[CmdletBinding()]
param(
    [string]$AudioPath,
    [string]$Scenarios
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "python executable not found at $pythonExe"
}
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

if ($Scenarios) {
    foreach ($scenario in ($Scenarios -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $arguments += @("--scenario", $scenario)
    }
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
$workspaceTmpRoot = Join-Path $HOME ".openclaw\workspace\tmp"
if (Test-Path $workspaceTmpRoot) {
    $stagedRoot = Join-Path $workspaceTmpRoot "demo-suite"
    try {
        New-Item -ItemType Directory -Path $stagedRoot -Force | Out-Null
        $runId = Split-Path $repoRunFolder -Leaf
        $stagedRunFolder = Join-Path $stagedRoot $runId
        if (Test-Path $stagedRunFolder) {
            Remove-Item -Recurse -Force $stagedRunFolder
        }
        Copy-Item -Recurse -Force $repoRunFolder $stagedRunFolder
    }
    catch {
        $stagedRunFolder = ""
        Write-Warning ("unable to mirror demo suite run into {0}: {1}" -f $stagedRoot, $_.Exception.Message)
    }
}

Write-Host "repo_run_folder=$repoRunFolder"
Write-Host "staged_run_folder=$stagedRunFolder"
Write-Host "manifest=$manifestPath"
