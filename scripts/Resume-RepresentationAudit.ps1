param([ValidateSet('train','evaluate','all')][string]$Stage = 'all')
$ErrorActionPreference = 'Stop'
$ResearchRoot = Split-Path -Parent $PSScriptRoot
$ResearchPython = Join-Path (Split-Path -Parent $ResearchRoot) 'e-jepa-ttc\.venv\Scripts\python.exe'
$ResearchLock = Join-Path $ResearchRoot 'artifacts\representation_audit_20261012\PROCESS.lock'
if (Test-Path -LiteralPath $ResearchLock) {
    $ResearchPid = [int](Get-Content -LiteralPath $ResearchLock -Raw)
    if (Get-Process -Id $ResearchPid -ErrorAction SilentlyContinue) {
        throw "V17 process $ResearchPid is alive; do not duplicate it."
    }
    Remove-Item -LiteralPath $ResearchLock
}
$PriorPythonPath = $env:PYTHONPATH
$PriorOmp = $env:OMP_NUM_THREADS
$PriorMkl = $env:MKL_NUM_THREADS
Push-Location -LiteralPath $ResearchRoot
try {
    $env:PYTHONPATH = 'src;.'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    if ($Stage -in @('train','all')) {
        & $ResearchPython -u -m operational.representation_audit.run train
        if ($LASTEXITCODE -ne 0) { throw 'V17 train failed; preserve journal and logs.' }
    }
    if ($Stage -in @('evaluate','all')) {
        & $ResearchPython -u -m operational.representation_audit.run evaluate
        if ($LASTEXITCODE -ne 0) { throw 'V17 evaluation failed or queue not frozen.' }
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $PriorPythonPath
    $env:OMP_NUM_THREADS = $PriorOmp
    $env:MKL_NUM_THREADS = $PriorMkl
}
