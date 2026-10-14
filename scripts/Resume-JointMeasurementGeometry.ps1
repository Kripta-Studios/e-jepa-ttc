param([switch]$Once)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path (Split-Path -Parent $Repo) 'e-jepa-ttc/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'No se encuentra el entorno compartido.' }
Push-Location -LiteralPath $Repo
try {
    $env:PYTHONPATH = 'src;.'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $CapturedRegistration = Join-Path $Repo 'artifacts/joint_measurement_20261012/runtime/geometry_captured_training_v1/REGISTRATION.json'
    $Module = if (Test-Path -LiteralPath $CapturedRegistration) {
        'operational.joint_measurement.geometry_capture_run'
    } else {
        'operational.joint_measurement.geometry_run'
    }
    $Arguments = @('-u', '-B', '-m', $Module)
    if ($Once) { $Arguments += '--once' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "V18 C1 terminó con código $LASTEXITCODE." }
}
finally { Pop-Location }
