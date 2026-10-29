param(
    [string]$PlanPath = '',
    [string]$PythonPath = '',
    [switch]$AllowScreenOff = $true
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $PlanPath) {
    $PlanPath = Join-Path $repositoryRoot 'artifacts/final_refit_20261019_scientific_oof/SCIENTIFIC_OOF_QUEUE_PLAN.json'
}
if (-not $PythonPath) {
    $PythonPath = Join-Path $repositoryRoot '../e-jepa-ttc/.venv/Scripts/python.exe'
}
$resolvedPlan = (Resolve-Path -LiteralPath $PlanPath).Path
$resolvedPython = (Resolve-Path -LiteralPath $PythonPath).Path
$outputDirectory = Split-Path -Parent $resolvedPlan
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = "$repositoryRoot\src;$repositoryRoot"
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$arguments = @('-m', 'operational.final_refit.scientific_oof_queue', 'run', '--plan', $resolvedPlan)
$argumentText = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$process = Start-Process -FilePath $resolvedPython -ArgumentList $argumentText `
    -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputDirectory "queue_$stamp.out.log") `
    -RedirectStandardError (Join-Path $outputDirectory "queue_$stamp.err.log") -PassThru
$awakeArguments = @('-m', 'operational.oof_awake', '--run', $outputDirectory, '--queue-pid', $process.Id)
if (-not $AllowScreenOff) { $awakeArguments += '--display' }
$awakeText = ($awakeArguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$awakeProcess = Start-Process -FilePath $resolvedPython -ArgumentList $awakeText `
    -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputDirectory "awake_$stamp.out.log") `
    -RedirectStandardError (Join-Path $outputDirectory "awake_$stamp.err.log") -PassThru
[pscustomobject]@{
    wrapper_pid = $process.Id
    awake_wrapper_pid = $awakeProcess.Id
    awake_status = (Join-Path $outputDirectory 'AWAKE_STATUS.json')
    plan = $resolvedPlan
    status = (Join-Path $outputDirectory 'SCIENTIFIC_OOF_QUEUE_STATE.json')
    scope = 'Scientific OOF rotations, final TRAIN40 chain, export, local test12 ZIP'
} | ConvertTo-Json
