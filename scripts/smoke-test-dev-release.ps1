[CmdletBinding()]
param(
    [string]$ReleaseFolder,
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$releasesRoot = Join-Path $repoRoot "artifacts/releases/dev"
$fallbackReleasesRoot = Join-Path $repoRoot "build/dev-release/releases/dev"

if (-not $ReleaseFolder) {
    if (-not (Test-Path $releasesRoot) -and -not (Test-Path $fallbackReleasesRoot)) {
        throw "Release root not found: $releasesRoot or $fallbackReleasesRoot"
    }
    $searchRoot = if (Test-Path $releasesRoot) { $releasesRoot } else { $fallbackReleasesRoot }
    $latest = Get-ChildItem -Path $searchRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $latest) {
        throw "No development release folders found under $searchRoot"
    }
    $ReleaseFolder = $latest.FullName
}

$appDir = Join-Path $ReleaseFolder "EchoZeroDev"
$exePath = Join-Path $appDir "EchoZeroDev.exe"
$reportPath = Join-Path $ReleaseFolder "smoke-report.json"
$smokeWorkingRoot = Join-Path $ReleaseFolder "smoke-working"
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$status = "failed"
$exitCode = $null
$durationSeconds = 0.0

if (-not (Test-Path $exePath)) {
    $report = [ordered]@{
        status = "failed"
        exit_code = $null
        duration_seconds = 0.0
        timestamp = $timestamp
        release_folder = $ReleaseFolder
        exe_path = $exePath
        reason = "missing_executable"
    }
    $report | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $reportPath
    Write-Host "SMOKE FAIL: executable not found at $exePath"
    exit 1
}

New-Item -ItemType Directory -Path $smokeWorkingRoot -Force | Out-Null

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $exePath -ArgumentList "--smoke-exit-seconds", "6", "--working-dir-root", $smokeWorkingRoot -PassThru

try {
    $exited = $process.WaitForExit($TimeoutSeconds * 1000)
    $stopwatch.Stop()
    $durationSeconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)

    if (-not $exited) {
        try {
            Stop-Process -Id $process.Id -Force
        } catch {
        }
        $status = "timeout"
        $exitCode = $null
        Write-Host "SMOKE FAIL: process did not exit within $TimeoutSeconds seconds"
        $scriptExit = 1
    } else {
        $exitCode = $process.ExitCode
        if ($exitCode -eq 0) {
            $status = "passed"
            Write-Host "SMOKE PASS: process exited with code 0 in $durationSeconds seconds"
            $scriptExit = 0
        } else {
            $status = "failed"
            Write-Host "SMOKE FAIL: process exited with code $exitCode in $durationSeconds seconds"
            $scriptExit = 1
        }
    }
}
finally {
    if ($stopwatch.IsRunning) {
        $stopwatch.Stop()
        $durationSeconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    }
}

$report = [ordered]@{
    status = $status
    exit_code = $exitCode
    duration_seconds = $durationSeconds
    timestamp = $timestamp
    release_folder = $ReleaseFolder
    exe_path = $exePath
    working_dir_root = $smokeWorkingRoot
    timeout_seconds = $TimeoutSeconds
}
$report | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $reportPath

Write-Host "report=$reportPath"
exit $scriptExit
