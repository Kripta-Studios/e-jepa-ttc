[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'OuterTrainSmoke', 'OOF')]
    [string]$Mode = 'DryRun',
    [ValidateSet('X0-A5-REPLAY', 'X0-PAIR-U', 'X0-BASE-U', 'X0-DYN-U', 'X0-DYN-W')]
    [string[]]$Arms = @('X0-A5-REPLAY', 'X0-PAIR-U', 'X0-BASE-U', 'X0-DYN-U'),
    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,
    [Parameter(Mandatory = $true)]
    [string]$ReferenceRoot,
    [string]$OutputRoot = 'artifacts/scientific_recovery_v9_eclock/runs',
    [string]$Device = 'cpu',
    [ValidateSet(0, 1, 2)]
    [int]$Fold = 0,
    [switch]$ExecuteAuthorizedOuterTrainSmoke,
    [switch]$ExecuteAuthorizedOOF
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
$TrainScript = Join-Path $PSScriptRoot 'train_scientific_recovery_v9_eclock.py'
$ConfigRoot = Join-Path $RepoRoot 'configs/experiment/scientific_recovery_v9_eclock'
$ResolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path $RepoRoot $OutputRoot
}
$ModeValue = switch ($Mode) {
    'DryRun' { 'dry-run' }
    'OuterTrainSmoke' { 'outer-train-smoke' }
    'OOF' { 'oof' }
}
$ConfigNames = @{
    'X0-A5-REPLAY' = 'x0_a5_replay.yaml'
    'X0-PAIR-U' = 'x0_pair_u.yaml'
    'X0-BASE-U' = 'x0_base_u.yaml'
    'X0-DYN-U' = 'x0_dyn_u.yaml'
    'X0-DYN-W' = 'x0_dyn_w.yaml'
}

foreach ($Arm in $Arms) {
    if ($Mode -eq 'OOF' -and $Arm -eq 'X0-DYN-W') {
        throw 'X0-DYN-W is always forbidden as scientific_oof.'
    }
    if ($Mode -eq 'OuterTrainSmoke' -and $Arm -notin @('X0-PAIR-U', 'X0-BASE-U', 'X0-DYN-U')) {
        throw "Outer-train smoke is unavailable for $Arm."
    }
    $Config = Join-Path $ConfigRoot $ConfigNames[$Arm]
    $ArmOutput = Join-Path $ResolvedOutputRoot $Arm
    $Arguments = @(
        $TrainScript,
        '--config', $Config,
        '--mode', $ModeValue,
        '--cache-root', $CacheRoot,
        '--reference-root', $ReferenceRoot,
        '--output-root', $ArmOutput,
        '--device', $Device
    )
    if ($ExecuteAuthorizedOOF) {
        $Arguments += '--execute-authorized-oof'
    }
    if ($Mode -eq 'OuterTrainSmoke') {
        $Arguments += @('--fold', "$Fold")
        if ($ExecuteAuthorizedOuterTrainSmoke) {
            $Arguments += '--execute-authorized-outer-train-smoke'
        }
    }
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "E-Clock command failed for $Arm."
    }
}
