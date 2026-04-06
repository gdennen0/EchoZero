param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python
)

$repoRoot = (Resolve-Path $Root).Path

if (-not $Python) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
    }
    else {
        $Python = "python"
    }
}

Write-Host "Launching EchoZero Foundry UI from $repoRoot"
& $Python -m echozero.foundry.cli --root $repoRoot ui
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Error "Foundry UI exited with code $exitCode"
}

exit $exitCode
