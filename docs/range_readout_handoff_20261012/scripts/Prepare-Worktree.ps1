# Requires PowerShell 7. Does not train, reset, clean, merge, commit or push.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$PythonExecutable = 'python',
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Base = '1c251781ac845a5d661523e607e0f7c4a8d3af8a'
$Branch = 'scientific-recovery-v15-range-readout'
$Handoff = Split-Path -Parent $PSScriptRoot

& $PythonExecutable (Join-Path $PSScriptRoot 'verify_package.py') --root $Handoff
if ($LASTEXITCODE -ne 0) { throw 'Handoff verification failed.' }
& git -C $SourceRepo rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'SourceRepo is not an existing worktree.' }
if (Test-Path -LiteralPath $Destination) { throw 'Destination exists; do not overwrite it.' }

$BranchExists = & git -C $SourceRepo branch --list $Branch
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect branches.' }
if ($BranchExists) { throw 'Target branch exists; use its worktree instead of recreating it.' }

& git -C $SourceRepo cat-file -e "$Base`^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowFetch) { throw 'Base commit absent. Explicit -AllowFetch is required.' }
    & git -C $SourceRepo fetch --no-tags origin scientific-recovery-v14-range-transfer
    if ($LASTEXITCODE -ne 0) { throw 'Fetch failed; existing worktrees remain unchanged.' }
    & git -C $SourceRepo cat-file -e "$Base`^{commit}"
    if ($LASTEXITCODE -ne 0) { throw 'Pinned base is not available after fetch.' }
}

& git -C $SourceRepo worktree add -b $Branch $Destination $Base
if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed; do not repair with reset/clean.' }
$Target = Join-Path $Destination 'docs/range_readout_handoff_20261012'
if (Test-Path -LiteralPath $Target) { throw 'Target handoff path already exists.' }
New-Item -ItemType Directory -Path $Target -Force | Out-Null
Get-ChildItem -LiteralPath $Handoff -Force | Copy-Item -Destination $Target -Recurse
& $PythonExecutable (Join-Path $Target 'scripts/verify_package.py') --root $Target
if ($LASTEXITCODE -ne 0) { throw 'Copied handoff did not verify.' }
Write-Host 'Worktree prepared. No scientific code was patched or trained.'
Write-Host ('Open Codex in: ' + $Destination)
Write-Host ('Read: ' + (Join-Path $Target 'PROMPT_CODEX_START.md'))
