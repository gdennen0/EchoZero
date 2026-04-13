[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Scenario,
    [string]$AudioPath
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
    "--record",
    "--scenario",
    $Scenario
)

if ($AudioPath) {
    $arguments += @("--audio-path", $AudioPath)
}

& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "demo scenario runner failed with exit code $LASTEXITCODE"
}
