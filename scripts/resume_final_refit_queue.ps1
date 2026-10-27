param(
    [string]$PlanPath = '',
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $PlanPath) {
    $PlanPath = Join-Path $repositoryRoot 'artifacts/final_refit_20261019_queue/QUEUE_PLAN.json'
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
$arguments = @('-m', 'operational.final_refit.continuation_queue', 'run', '--plan', $resolvedPlan)
$argumentText = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$process = Start-Process -FilePath $resolvedPython -ArgumentList $argumentText `
    -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputDirectory "queue_$stamp.out.log") `
    -RedirectStandardError (Join-Path $outputDirectory "queue_$stamp.err.log") -PassThru
# The Python executor owns an OS lock. A duplicate invocation cannot run another queue.
[pscustomobject]@{
    wrapper_pid = $process.Id
    plan = $resolvedPlan
    status = (Join-Path $outputDirectory 'QUEUE_STATUS.json')
    scope = 'Wait for RGB, verify OOF roles, report missing scientific implementation'
} | ConvertTo-Json
