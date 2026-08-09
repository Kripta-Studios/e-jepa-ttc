[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Force,
    [switch]$BuildCache,
    [switch]$DryRun,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [Parameter(Mandatory = $true)][string]$Cache,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Config,
    [string]$LogRoot,
    [string]$SourceParquet,
    [string]$EventRoot,
    [string]$Stage2Dir,
    [string]$Stage2Output,
    [string]$Stage2Seed7,
    [string]$Stage2Seed13,
    [string]$Stage2Seed23
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Config) {
    $Config = Join-Path $Root 'configs/experiment/e_jepa_garl_object_event_operator_audit_v4_31.yaml'
}
if (-not $LogRoot) {
    $LogRoot = Join-Path $Root 'artifacts/logs'
}

$Mode = if ($Full) { 'full' } else { 'diagnostic' }
$Runner = Join-Path $PSScriptRoot 'run_object_event_v4_31_pipeline.py'
$RunnerArgs = @(
    'run', '--no-sync', 'python', $Runner, $Mode,
    '--device', $Device,
    '--config', $Config,
    '--cache', $Cache,
    '--output-dir', $OutputDir,
    '--log-root', $LogRoot
)
if ($BuildCache) { $RunnerArgs += '--build-cache' }
if ($SourceParquet) { $RunnerArgs += @('--source-parquet', $SourceParquet) }
if ($EventRoot) { $RunnerArgs += @('--event-root', $EventRoot) }
if ($Force) { $RunnerArgs += '--force' }
if ($DryRun) { $RunnerArgs += '--dry-run' }
if ($Stage2Dir) { $RunnerArgs += @('--stage2-dir', $Stage2Dir) }

$SeedSources = @{
    7 = $Stage2Seed7
    13 = $Stage2Seed13
    23 = $Stage2Seed23
}
$AnySeed = @($SeedSources.Values | Where-Object { $_ }).Count -gt 0
if ($AnySeed -and @($SeedSources.Values | Where-Object { -not $_ }).Count -gt 0) {
    throw 'Stage2 sources must contain all of Stage2Seed7, Stage2Seed13, and Stage2Seed23.'
}
if ($AnySeed) {
    foreach ($Seed in 7, 13, 23) {
        $RunnerArgs += @('--stage2-source', "$Seed=$($SeedSources[$Seed])")
    }
    if (-not $Stage2Output) {
        $OutputLeaf = Split-Path -Leaf $OutputDir
        $Stage2Output = Join-Path (Split-Path -Parent $OutputDir) "$OutputLeaf.stage2"
    }
    $RunnerArgs += @('--stage2-output', $Stage2Output)
}
if ($Full -and -not $Stage2Dir -and -not $AnySeed) {
    throw 'Full mode requires -Stage2Dir or all three Stage2Seed parameters.'
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Transcript = Join-Path $LogRoot "object_event_v4_31_wrapper_$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')).log"
Start-Transcript -Path $Transcript -Force | Out-Null
try {
    & uv @RunnerArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    Stop-Transcript | Out-Null
}
exit $ExitCode
