#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceRepo,
    [Parameter(Mandatory)][string]$Destination,
    [string]$PythonPath = 'python',
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$BaseCommit = 'b44443ae331b5e84b94e940801468449b2858de4'
$Branch = 'scientific-recovery-v13-rgb-port'
$PackageRoot = Split-Path -Parent $PSScriptRoot

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) { throw "$Operation failed (exit $LASTEXITCODE)." }
}

$null = Get-Command git -ErrorAction Stop
$null = Get-Command $PythonPath -ErrorAction Stop
& $PythonPath -B (Join-Path $PSScriptRoot 'verify_package.py') --root $PackageRoot
Assert-NativeSuccess 'Handoff verification'

$SourceRepo = (Resolve-Path -LiteralPath $SourceRepo).Path
$Top = & git -C $SourceRepo rev-parse --show-toplevel
Assert-NativeSuccess 'Source worktree lookup'
$SourceRepo = $Top.Trim()
$Remote = & git -C $SourceRepo remote get-url origin
Assert-NativeSuccess 'Origin lookup'
if ($Remote.Trim() -notmatch 'github\.com[:/]Kripta-Studios/e-jepa-ttc(?:\.git)?/?$') {
    throw "The source origin is not the expected repository: $Remote"
}
if (Test-Path -LiteralPath $Destination) {
    throw 'Destination already exists. Resume the existing worktree or choose another path; do not overwrite it.'
}
& git -C $SourceRepo show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) { throw 'The target branch exists. Inspect/resume it; no automatic reset is permitted.' }
if ($LASTEXITCODE -ne 1) { throw 'Cannot check target branch.' }

& git -C $SourceRepo cat-file -e "$BaseCommit`^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowFetch) { throw 'Pinned commit unavailable locally. Re-run with -AllowFetch to fetch only that commit.' }
    & git -C $SourceRepo fetch --no-tags origin $BaseCommit
    Assert-NativeSuccess 'Pinned commit fetch'
    & git -C $SourceRepo cat-file -e "$BaseCommit`^{commit}"
    Assert-NativeSuccess 'Pinned commit verification'
}
$SourceHead = & git -C $SourceRepo rev-parse HEAD
Assert-NativeSuccess 'Source HEAD lookup'
$SourceStatus = & git -C $SourceRepo status --porcelain=v1
Assert-NativeSuccess 'Source status snapshot'
$Destination = [System.IO.Path]::GetFullPath($Destination)
& git -C $SourceRepo worktree add -b $Branch $Destination $BaseCommit
Assert-NativeSuccess 'Worktree creation'
$TargetHead = & git -C $Destination rev-parse HEAD
Assert-NativeSuccess 'Target HEAD lookup'
if ($TargetHead.Trim() -ne $BaseCommit) { throw 'New worktree does not match the pinned base.' }

$TargetHandoff = Join-Path $Destination 'docs/rgb_port_handoff_20261008'
New-Item -ItemType Directory -Path $TargetHandoff -ErrorAction Stop | Out-Null
Get-ChildItem -LiteralPath $PackageRoot -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $TargetHandoff -Recurse -ErrorAction Stop
}
& $PythonPath -B (Join-Path $TargetHandoff 'scripts/verify_package.py') --root $TargetHandoff
Assert-NativeSuccess 'Copied handoff verification'
$Record = [ordered]@{
    schema = 'rgb_port_worktree_preparation_v1'
    base_commit = $BaseCommit
    branch = $Branch
    source_head_observed = $SourceHead.Trim()
    source_status_observed = @($SourceStatus)
    destination = $Destination
    historical_files_changed = $false
    training_executed = $false
    created_utc = [DateTime]::UtcNow.ToString('o')
}
$Record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Destination 'docs/RGB_PORT_SETUP.json') -Encoding utf8NoBOM
Write-Host "Prepared: $Destination"
Write-Host 'No models have been trained. Start Codex in this worktree and explicitly send the new campaign prompt.'
