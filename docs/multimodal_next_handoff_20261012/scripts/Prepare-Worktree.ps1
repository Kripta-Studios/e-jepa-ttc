# Run under PowerShell 7. This script prepares a separate worktree; it does not train.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$Python = 'python',
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Base = '62cf4dbe914ea077c68e42983f869cd5422ab71c'
$Branch = 'scientific-recovery-v16-multimodal-validation'
$PackageRoot = Split-Path -Parent $PSScriptRoot
& $Python -B (Join-Path $PSScriptRoot 'verify_package.py') $PackageRoot
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }
$Repo = (Resolve-Path -LiteralPath $SourceRepo).Path
if (Test-Path -LiteralPath $Destination) {
    throw 'Destination exists. Preserve any running work; do not overwrite it.'
}
& git -C $Repo rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'Source is not a Git worktree.' }
if ($AllowFetch) {
    & git -C $Repo fetch origin scientific-recovery-v13-rgb-port
    if ($LASTEXITCODE -ne 0) { throw 'Fetch failed; no reset or replacement is permitted.' }
}
& git -C $Repo cat-file -e ($Base + '^{commit}')
if ($LASTEXITCODE -ne 0) { throw 'Pinned base commit is unavailable.' }
& git -C $Repo show-ref --verify --quiet ('refs/heads/' + $Branch)
if ($LASTEXITCODE -eq 0) { throw 'Branch already exists. Locate and resume it; do not recreate it.' }
& git -C $Repo worktree add -b $Branch $Destination $Base
if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed.' }
$Target = Join-Path $Destination 'docs/multimodal_next_handoff_20261012'
New-Item -ItemType Directory -Path $Target | Out-Null
Get-ChildItem -LiteralPath $PackageRoot -Force | Copy-Item -Destination $Target -Recurse
Write-Host "Prepared: $Destination"
Write-Host "Read: $Target/PROMPT_CODEX_START.md"
Write-Host 'No model code has been patched and no training has been started.'
