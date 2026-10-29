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
    $RuntimeAdmission = Join-Path $Repo 'artifacts/joint_measurement_20261012/runtime/rgb_resident_v1/REGISTRATION.json'
    $DriverModule = if (Test-Path -LiteralPath $RuntimeAdmission) {
        'operational.joint_measurement.rgb_cached_run'
    } else {
        'operational.joint_measurement.rgb_run'
    }
    $Arguments = @('-u', '-B', '-m', $DriverModule)
    if ($Once) { $Arguments += '--once' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "V18 RGB terminó con código $LASTEXITCODE." }
}
finally { Pop-Location }
