#Requires -Version 7.0
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$SourceRepo,
  [string]$Destination,
  [string]$Branch = 'scientific-recovery-v12-efficient-context',
  [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$Commit = 'fd16d8102914537653622d3abf298b753120ea43'
$Tag = 'simplex-t-local-results-20261004-complete'
$null = Get-Command git -ErrorAction Stop

# Verify every handoff payload before creating anything in the repository.
foreach ($Line in [IO.File]::ReadAllLines((Join-Path $PackageRoot 'SHA256SUMS.txt'))) {
  if ([string]::IsNullOrWhiteSpace($Line)) { continue }
  $Parts = $Line -split '  ', 2
  if ($Parts.Length -ne 2 -or $Parts[0] -notmatch '^[0-9a-f]{64}$') { throw 'Invalid handoff manifest.' }
  $Relative = $Parts[1]
  if ($Relative -match '(^[/\\]|:|(^|/)\.\.(/|$))') { throw 'Unsafe manifest path.' }
  $Path = Join-Path $PackageRoot $Relative
  if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Parts[0]) {
    throw "Handoff hash mismatch: $Relative"
  }
}
$SourceRepo = (Resolve-Path -LiteralPath $SourceRepo).Path
& git -C $SourceRepo rev-parse --is-inside-work-tree
if ($LASTEXITCODE -ne 0) { throw 'SourceRepo is not a Git worktree.' }
& git -C $SourceRepo cat-file -e "$Commit^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
  if (-not $AllowFetch) { throw 'Pinned commit absent. Re-run with -AllowFetch after authorizing network fetch.' }
  & git -C $SourceRepo fetch --no-tags origin "refs/tags/$Tag"
  if ($LASTEXITCODE -ne 0) { throw 'Fetch failed; source worktree preserved.' }
  $Fetched = (& git -C $SourceRepo rev-parse 'FETCH_HEAD^{commit}').Trim()
  if ($Fetched -ne $Commit) { throw 'Remote tag no longer matches the inspected commit.' }
}
if ([string]::IsNullOrWhiteSpace($Destination)) {
  $Destination = Join-Path (Split-Path -Parent $SourceRepo) 'e-jepa-ttc-v12-efficient-context'
}
$Destination = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $Destination) {
  & git -C $Destination rev-parse --is-inside-work-tree
  if ($LASTEXITCODE -ne 0) { throw 'Destination exists but is not a worktree; nothing overwritten.' }
  & git -C $Destination merge-base --is-ancestor $Commit HEAD
  if ($LASTEXITCODE -ne 0) { throw 'Destination is not based on the inspected commit.' }
  Write-Host 'Existing compatible worktree preserved. No reset or checkout performed.'
} else {
  & git -C $SourceRepo worktree add -b $Branch $Destination $Commit
  if ($LASTEXITCODE -ne 0) { throw 'Could not create worktree; inspect branch/path rather than forcing cleanup.' }
}
$Target = Join-Path $Destination 'docs/efficient_context_handoff_20261004'
if (Test-Path -LiteralPath $Target) {
  $A = (Get-FileHash -LiteralPath (Join-Path $Target 'SHA256SUMS.txt')).Hash
  $B = (Get-FileHash -LiteralPath (Join-Path $PackageRoot 'SHA256SUMS.txt')).Hash
  if ($A -ne $B) { throw 'A different handoff already exists; it was preserved.' }
  foreach ($Line in [IO.File]::ReadAllLines((Join-Path $Target 'SHA256SUMS.txt'))) {
    if ([string]::IsNullOrWhiteSpace($Line)) { continue }
    $Parts = $Line -split '  ', 2
    if ((Get-FileHash -LiteralPath (Join-Path $Target $Parts[1])).Hash.ToLowerInvariant() -ne $Parts[0]) {
      throw 'Existing handoff was modified; preserved for agent review.'
    }
  }
} else {
  $null = New-Item -ItemType Directory -Path $Target
  Copy-Item -Path (Join-Path $PackageRoot '*') -Destination $Target -Recurse
}
Write-Host "Worktree: $Destination"
Write-Host "Handoff:  $Target"
Write-Host "Open a NEW Codex session here: codex --cd `"$Destination`""
Write-Host 'Paste PROMPT_CODEX_START.md and point the agent to the handoff directory.'
Write-Host 'No training, package installation, push, source reset or protected data read was performed.'
