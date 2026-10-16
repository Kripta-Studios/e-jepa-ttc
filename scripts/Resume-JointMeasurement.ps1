param(
    [ValidateSet('Auto', 'Primary', 'B1Training', 'B1Evaluation')]
    [string]$Stage = 'Auto',
    [switch]$Once,
    [switch]$ShowCommand
)
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
    $Campaign = Join-Path $Repo 'artifacts/joint_measurement_20261012'
    $B1 = Join-Path $Campaign 'adaptive/B1_correspondence'
    if ($Stage -eq 'Auto') {
        if (Test-Path -LiteralPath (Join-Path $B1 'evidence/COMPLETE.json')) {
            throw 'B1 ya tiene evidencia completa. Interpreta sus resultados y NEXT_DECISION.json antes de elegir otra rama.'
        } elseif (Test-Path -LiteralPath (Join-Path $B1 'ENDPOINTS.json')) {
            $Stage = 'B1Evaluation'
        } elseif (Test-Path -LiteralPath (Join-Path $B1 'REGISTRATION.json')) {
            $Stage = 'B1Training'
        } elseif (Test-Path -LiteralPath (Join-Path $Campaign 'primary_evidence/COMPLETE.json')) {
            throw 'La matriz primaria está cerrada. Consulta NEXT_DECISION.json para la rama registrada pendiente.'
        } else {
            $Stage = 'Primary'
        }
    }
    if ($Once -and $Stage -ne 'Primary') {
        throw '-Once sólo corresponde al runner primario; B1 conserva sus segmentos durables registrados.'
    }
    $Module = switch ($Stage) {
        'B1Training' { 'operational.joint_measurement.correspondence_isolated' }
        'B1Evaluation' { 'operational.joint_measurement.correspondence_evaluate' }
        'Primary' {
            $RuntimeBinding = Join-Path $Campaign 'runtime/optimization_20261013/RUNTIME_BINDING.json'
            if (Test-Path -LiteralPath $RuntimeBinding) {
                'operational.joint_measurement.fast_run'
            } else {
                'operational.joint_measurement.run'
            }
        }
    }
    $Arguments = @('-u', '-B', '-m', $Module)
    if ($Once) { $Arguments += '--once' }
    if ($ShowCommand) {
        [pscustomobject]@{ Stage = $Stage; Python = $Python; Arguments = $Arguments; WorkingDirectory = $Repo }
        return
    }
    # A conservative preflight across project modules also catches the gap between
    # GPU children. The actual runtimes retain their own exclusive locks/admission.
    $Writers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match 'operational\.joint_measurement\.' -and
        $_.CommandLine -notmatch 'operational\.joint_measurement\.hourly_observe'
    })
    if ($Writers.Count -gt 0) {
        throw "Hay procesos V18 activos (PID $($Writers.ProcessId -join ', ')). No se inicia otro escritor."
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "V18 terminó con código $LASTEXITCODE." }
}
finally { Pop-Location }
