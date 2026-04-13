[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv/Scripts/python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$buildRoot = Join-Path $repoRoot "build/test-release"
$distRoot = Join-Path $repoRoot "dist/test-release"
$preferredArtifactsRoot = Join-Path $repoRoot "artifacts/releases/test"
$fallbackArtifactsRoot = Join-Path $repoRoot "build/test-release/releases/test"
$artifactsRoot = $preferredArtifactsRoot
$releaseRoot = Join-Path $artifactsRoot $timestamp
$releaseAppDir = Join-Path $releaseRoot "EchoZeroTest"
$zipPath = "$releaseRoot.zip"
$timelineFixtureDir = Join-Path $repoRoot "echozero/ui/qt/timeline/fixtures"

function Select-ArtifactsRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Preferred,
        [Parameter(Mandatory = $true)]
        [string]$Fallback
    )

    try {
        New-Item -ItemType Directory -Path $Preferred -Force | Out-Null
        return $Preferred
    }
    catch {
        Write-Warning "Preferred artifacts root unavailable: $Preferred"
        Write-Warning "Falling back to: $Fallback"
        New-Item -ItemType Directory -Path $Fallback -Force | Out-Null
        return $Fallback
    }
}

$artifactsRoot = Select-ArtifactsRoot -Preferred $preferredArtifactsRoot -Fallback $fallbackArtifactsRoot
$releaseRoot = Join-Path $artifactsRoot $timestamp
$releaseAppDir = Join-Path $releaseRoot "EchoZeroTest"
$zipPath = "$releaseRoot.zip"

New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $distRoot -Force | Out-Null
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
Write-Host "artifacts_root=$artifactsRoot"
Write-Host "release_folder=$releaseRoot"
Write-Host "app_folder=$releaseAppDir"
Write-Host "metadata=$metadataPath"
Write-Host "zip=$zipPath"
