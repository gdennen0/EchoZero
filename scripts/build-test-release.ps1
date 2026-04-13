[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "C:/Users/griff/EchoZero/.venv/Scripts/python.exe"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$buildRoot = Join-Path $repoRoot "build/test-release"
$distRoot = Join-Path $repoRoot "dist/test-release"
$artifactsRoot = Join-Path $repoRoot "build/test-release/releases/test"
$releaseRoot = Join-Path $artifactsRoot $timestamp
$releaseAppDir = Join-Path $releaseRoot "EchoZeroTest"
$zipPath = "$releaseRoot.zip"
$timelineFixtureDir = Join-Path $repoRoot "echozero/ui/qt/timeline/fixtures"

New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $distRoot -Force | Out-Null
New-Item -ItemType Directory -Path $artifactsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

$pyInstallerArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--windowed",
    "--name",
    "EchoZeroTest",
    "--workpath",
    $buildRoot,
    "--specpath",
    $buildRoot,
    "--distpath",
    $distRoot,
    "--add-data",
    "${timelineFixtureDir};echozero/ui/qt/timeline/fixtures",
    "run_echozero.py"
)

if ($Clean) {
    $pyInstallerArgs = @("-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", "EchoZeroTest", "--workpath", $buildRoot, "--specpath", $buildRoot, "--distpath", $distRoot, "--add-data", "${timelineFixtureDir};echozero/ui/qt/timeline/fixtures", "run_echozero.py")
}

Write-Host "Building EchoZero test release..."
Write-Host "$pythonExe $($pyInstallerArgs -join ' ')"
& $pythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$builtAppDir = Join-Path $distRoot "EchoZeroTest"
if (-not (Test-Path $builtAppDir)) {
    throw "Built app folder not found: $builtAppDir"
}

if (Test-Path $releaseAppDir) {
    Remove-Item -Recurse -Force $releaseAppDir
}
Copy-Item -Recurse -Force $builtAppDir $releaseAppDir

$commit = (git -c safe.directory=C:/Users/griff/EchoZero rev-parse HEAD).Trim()
$branch = (git -c safe.directory=C:/Users/griff/EchoZero rev-parse --abbrev-ref HEAD).Trim()
$commandSummary = "$pythonExe $($pyInstallerArgs -join ' ')"
$metadataPath = Join-Path $releaseRoot "build-metadata.json"
$metadata = [ordered]@{
    timestamp = $timestamp
    git_commit = $commit
    git_branch = $branch
    command_summary = $commandSummary
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $metadataPath

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path $releaseRoot -DestinationPath $zipPath -Force

Write-Host "BUILD TEST RELEASE COMPLETE"
Write-Host "release_folder=$releaseRoot"
Write-Host "app_folder=$releaseAppDir"
Write-Host "metadata=$metadataPath"
Write-Host "zip=$zipPath"
