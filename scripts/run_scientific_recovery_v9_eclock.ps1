[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'SyntheticSmoke', 'OOF')]
    [string]$Mode = 'DryRun',
    [ValidateSet('X0-A5-REPLAY', 'X0-PAIR-U', 'X0-BASE-U', 'X0-DYN-U')]
    [string[]]$Arms = @('X0-A5-REPLAY', 'X0-PAIR-U', 'X0-BASE-U', 'X0-DYN-U'),
    [ValidateSet(7)]
    [int]$Seed = 7,
    [ValidateSet(0, 1, 2)]
    [int[]]$Folds = @(0, 1, 2),
    [switch]$ExecuteAuthorizedOOF
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
$TrainScript = Join-Path $PSScriptRoot 'train_scientific_recovery_v9_eclock.py'
$ConfigRoot = Join-Path $RepoRoot 'configs/experiment/scientific_recovery_v9_eclock'
$ModeValue = switch ($Mode) {
    'DryRun' { 'dry-run' }
    'SyntheticSmoke' { 'synthetic-smoke' }
    'OOF' { 'oof' }
}
$ConfigNames = @{
    'X0-A5-REPLAY' = 'x0_a5_replay.yaml'
    'X0-PAIR-U' = 'x0_pair_u.yaml'
    'X0-BASE-U' = 'x0_base_u.yaml'
    'X0-DYN-U' = 'x0_dyn_u.yaml'
}

foreach ($Arm in $Arms) {
    $Config = Join-Path $ConfigRoot $ConfigNames[$Arm]
    if ($Mode -eq 'OOF') {
        foreach ($Fold in $Folds) {
            $Arguments = @(
                $TrainScript,
                '--config', $Config, '--mode', $ModeValue, '--seed', "$Seed", '--fold', "$Fold"
            )
            if ($ExecuteAuthorizedOOF) { $Arguments += '--execute-authorized-oof' }
            & python @Arguments
            if ($LASTEXITCODE -ne 0) { throw "E-Clock OOF command failed for $Arm fold $Fold" }
        }
    } else {
        & python $TrainScript `
            --config $Config --mode $ModeValue --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "E-Clock command failed for $Arm" }
    }
}
