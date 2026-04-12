param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('foundry', 'ez')]
    [string]$Lane,

    [Parameter(Mandatory = $false)]
    [ValidateSet('staged', 'worktree', 'range')]
    [string]$Mode = 'range',

    [Parameter(Mandatory = $false)]
    [string]$BaseRef = 'origin/main',

    [switch]$AllowShared
)

$ErrorActionPreference = 'Stop'

function Get-ChangedFiles {
    param(
        [string]$Mode,
        [string]$BaseRef
    )

    switch ($Mode) {
        'staged' {
            return @(git diff --name-only --cached)
        }
        'worktree' {
            return @(git diff --name-only)
        }
        'range' {
            return @(git diff --name-only "$BaseRef...HEAD")
        }
    }
}

function StartsWithAny {
    param(
        [string]$Value,
        [string[]]$Prefixes
    )

    foreach ($prefix in $Prefixes) {
        if ($Value.StartsWith($prefix)) {
            return $true
        }
    }
    return $false
}

$foundryOwned = @(
    'echozero/foundry/',
    'tests/foundry/'
)

$ezOwned = @(
    'echozero/application/',
    'echozero/ui/',
    'tests/application/',
    'tests/ui/'
)

$shared = @(
    'echozero/inference_eval/',
    'tests/inference_eval/',
    'tests/processors/test_pytorch_audio_classify_preflight.py'
)

$changed = Get-ChangedFiles -Mode $Mode -BaseRef $BaseRef |
    Where-Object { $_ -and $_.Trim().Length -gt 0 } |
    ForEach-Object { $_.Replace('\\', '/') }

if (-not $changed -or $changed.Count -eq 0) {
    Write-Output "OK: no changed files for mode '$Mode'."
    exit 0
}

$violations = New-Object System.Collections.Generic.List[string]
$sharedHits = New-Object System.Collections.Generic.List[string]

foreach ($path in $changed) {
    $isFoundryOwned = StartsWithAny -Value $path -Prefixes $foundryOwned
    $isEzOwned = StartsWithAny -Value $path -Prefixes $ezOwned
    $isShared = StartsWithAny -Value $path -Prefixes $shared

    if ($Lane -eq 'foundry') {
        if ($isEzOwned) {
            $violations.Add($path)
            continue
        }
        if ($isShared -and -not $AllowShared) {
            $sharedHits.Add($path)
            continue
        }
        continue
    }

    if ($Lane -eq 'ez') {
        if ($isFoundryOwned) {
            $violations.Add($path)
            continue
        }
        if ($isShared -and -not $AllowShared) {
            $sharedHits.Add($path)
            continue
        }
        continue
    }
}

if ($violations.Count -gt 0) {
    Write-Error (
        "Lane ownership violation for lane '$Lane'. Disallowed paths:`n - " +
        ($violations -join "`n - ")
    )
}

if ($sharedHits.Count -gt 0) {
    Write-Error (
        "Shared-zone paths detected for lane '$Lane' without -AllowShared. Use integration gate.`n - " +
        ($sharedHits -join "`n - ")
    )
}

Write-Output "OK: lane '$Lane' path ownership validation passed for mode '$Mode'."
exit 0
