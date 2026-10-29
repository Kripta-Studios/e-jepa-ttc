#Requires -Version 7.0
[CmdletBinding()]
param(
    [int]$MaxUpdatesThisCall = 25000,
    [switch]$DiagnosticsOnly,
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$rangeRepo = Split-Path -Parent $PSScriptRoot
$rangeParent = Split-Path -Parent $rangeRepo
$rangePython = Join-Path $rangeParent 'e-jepa-ttc/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $rangePython)) { throw "Missing project Python: $rangePython" }
$env:PYTHONPATH = (Join-Path $rangeRepo 'src') + [IO.Path]::PathSeparator + $rangeRepo
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$rangeArguments = @('-B','-m','operational.range_transfer.run',
    '--owner-run',(Join-Path $rangeParent 'e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008'),
    '--v12-root',(Join-Path $rangeParent 'e-jepa-ttc-v12-efficient-context'),
    '--run',(Join-Path $rangeRepo 'artifacts/range_transfer_20261010'),
    '--max-updates-this-call',"$MaxUpdatesThisCall")
if ($DiagnosticsOnly) { $rangeArguments += '--diagnostics-only' }
if ($SelfTest) { $rangeArguments += '--self-test' }
Push-Location $rangeRepo
try { & $rangePython @rangeArguments; $rangeExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $rangeExitCode
