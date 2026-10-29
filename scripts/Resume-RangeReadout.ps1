# Resume only the closed V15 queue. No V13 mutation, push, reset, or new experiments.
[CmdletBinding()]
param([switch]$WaitActivePExtraction, [switch]$Help)
$ErrorActionPreference = 'Stop'
$TaskRepo = Split-Path -Parent $PSScriptRoot
$TaskPython = Join-Path (Split-Path -Parent $TaskRepo) 'e-jepa-ttc\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) { throw "Existing Python missing: $TaskPython" }
$env:PYTHONPATH = (Join-Path $TaskRepo 'src') + ';' + $TaskRepo
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$TaskArgs = @('-B', '-m', 'operational.range_readout.campaign')
if ($WaitActivePExtraction) { $TaskArgs += '--wait-active-p-extraction' }
if ($Help) { $TaskArgs += '--help' }
Push-Location -LiteralPath $TaskRepo
try {
    & $TaskPython @TaskArgs
    if ($LASTEXITCODE -ne 0) { throw "V15 stopped with exit code $LASTEXITCODE; inspect artifacts/range_readout_20261012 logs before resuming." }
}
finally { Pop-Location }
