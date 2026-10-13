param([switch]$Once)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path (Split-Path -Parent $Repo) 'e-jepa-ttc/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'No se encuentra el entorno compartido documentado.'
}
Push-Location -LiteralPath $Repo
try {
    $env:PYTHONPATH = 'src;.'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $RuntimeBinding = Join-Path $Repo 'artifacts/joint_measurement_20261012/runtime/optimization_20261013/RUNTIME_BINDING.json'
    $Module = if (Test-Path -LiteralPath $RuntimeBinding) {
        'operational.joint_measurement.fast_run'
    } else {
        'operational.joint_measurement.run'
    }
    $Arguments = @('-u', '-B', '-m', $Module)
    if ($Once) { $Arguments += '--once' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "V18 terminó con código $LASTEXITCODE." }
}
finally { Pop-Location }
