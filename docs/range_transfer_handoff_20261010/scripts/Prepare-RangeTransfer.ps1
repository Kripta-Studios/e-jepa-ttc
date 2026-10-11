#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Base = 'da068ebdeb2d967ee10b2afa348e1317b8720ad0'
$Branch = 'scientific-recovery-v14-range-transfer'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$Source = (Resolve-Path -LiteralPath $SourceRepo).Path
$Target = [System.IO.Path]::GetFullPath($Destination)

# Hash all payloads before creating or writing a worktree.
$Manifest = Join-Path $PackageRoot 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw 'Missing SHA256SUMS.txt' }
$Seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($Line in Get-Content -LiteralPath $Manifest -Encoding utf8) {
    if ([string]::IsNullOrWhiteSpace($Line)) { continue }
    if ($Line -notmatch '^([0-9a-f]{64})  (.+)$') { throw "Invalid manifest entry: $Line" }
    $Expected = $Matches[1]; $Relative = $Matches[2]
    if ($Relative.StartsWith('/') -or $Relative.Contains('\') -or $Relative.Split('/') -contains '..') { throw "Unsafe payload path: $Relative" }
    if (-not $Seen.Add($Relative)) { throw "Duplicate path: $Relative" }
    $File = Join-Path $PackageRoot $Relative
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing payload: $File" }
    $Actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) { throw "Payload hash mismatch: $Relative" }
}

& git -C $Source rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'SourceRepo is not an accessible Git worktree.' }
if (Test-Path -LiteralPath $Target) { throw 'Destination already exists. Preserve it; do not overwrite or reset.' }
& git -C $Source show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) { throw "Branch already exists: $Branch. Inspect its existing worktree instead of duplicating it." }

& git -C $Source cat-file -e "${Base}^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowFetch) { throw 'Pinned V13 commit unavailable; rerun with -AllowFetch only to fetch the V13 branch.' }
    & git -C $Source fetch origin scientific-recovery-v13-rgb-port
    if ($LASTEXITCODE -ne 0) { throw 'Fetch failed; no substitution by main is allowed.' }
    & git -C $Source cat-file -e "${Base}^{commit}"
    if ($LASTEXITCODE -ne 0) { throw 'Pinned V13 commit still unavailable after fetch.' }
}

& git -C $Source worktree add -b $Branch $Target $Base
if ($LASTEXITCODE -ne 0) { throw 'git worktree add failed; no reset/clean attempted.' }
$CopyTarget = Join-Path $Target 'docs/range_transfer_handoff_20261010'
if (Test-Path -LiteralPath $CopyTarget) { throw 'Unexpected existing handoff in new worktree.' }
New-Item -ItemType Directory -Path $CopyTarget | Out-Null
Get-ChildItem -LiteralPath $PackageRoot -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $CopyTarget -Recurse
}
Write-Host "Prepared separate worktree: $Target"
Write-Host "No patch, training, commit, push or change to the original worktree was executed."
Write-Host "Next: codex --cd `"$Target`""
Write-Host 'Read docs/range_transfer_handoff_20261010/PROMPT_CODEX_START.md in that session.'
