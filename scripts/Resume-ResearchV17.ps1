param(
    [ValidateSet('all', 'A', 'B', 'D', 'E', 'F', 'G')]
    [string]$Queue = 'all'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Run = Join-Path $Repo 'artifacts/representation_audit_20261012'
$Python = Join-Path (Split-Path -Parent $Repo) 'e-jepa-ttc/.venv/Scripts/python.exe'
$Lock = Join-Path $Run 'PROCESS.lock'
if (Test-Path -LiteralPath $Lock) {
    $OwnerPid = [int](Get-Content -LiteralPath $Lock -Raw)
    if (Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue) {
        throw "V17 ya tiene un worker vivo: PID $OwnerPid. No iniciar otro."
    }
    # This is a single verified stale file in the campaign, never recursive cleanup.
    Remove-Item -LiteralPath $Lock
}
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$env:PYTHONPATH = 'src;.'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
function Invoke-Stage([string]$Module, [string]$Stage) {
    $Name = $Module.Split('.')[-1]
    $Log = Join-Path $Run "RESUME_${Stamp}_${Name}_${Stage}.log"
    & $Python -u -m $Module $Stage *> $Log
    if ($LASTEXITCODE -ne 0) { throw "Fallo en $Name/$Stage. Evidencia: $Log" }
}
Push-Location $Repo
try {
    if ($Queue -in @('all', 'A')) {
        Invoke-Stage 'operational.representation_audit.run' 'train'
        Invoke-Stage 'operational.representation_audit.run' 'evaluate'
    }
    if ($Queue -in @('all', 'B')) {
        Invoke-Stage 'operational.representation_audit.precision' 'train'
        Invoke-Stage 'operational.representation_audit.precision' 'evaluate'
    }
    if ($Queue -in @('all', 'D')) {
        Invoke-Stage 'operational.representation_audit.directional' 'qa'
        Invoke-Stage 'operational.representation_audit.directional' 'train'
        Invoke-Stage 'operational.representation_audit.directional' 'evaluate'
    }
    foreach ($Item in @(
        @{ Key='E'; Module='geometry'; Prefix='GEOMETRY' },
        @{ Key='F'; Module='temporal_geometry'; Prefix='TEMP_GEOMETRY' }
    )) {
        if ($Queue -notin @('all', $Item.Key)) { continue }
        $Module = 'operational.representation_audit.' + $Item.Module
        Invoke-Stage $Module 'qa'
        Invoke-Stage $Module 'train'
        Invoke-Stage $Module 'evaluate'
        $Decision = Get-Content -LiteralPath (Join-Path $Run ($Item.Prefix + '_SCREEN_DECISION.json')) -Raw | ConvertFrom-Json
        if ($Decision.status -eq 'REPLICATE') {
            Invoke-Stage $Module 'replicate'
            Invoke-Stage $Module 'evaluate-replicas'
        }
    }
    if ($Queue -in @('all', 'G')) {
        Invoke-Stage 'operational.representation_audit.spatial_cache' 'qa'
        Invoke-Stage 'operational.representation_audit.spatial_cache' 'P'
        Invoke-Stage 'operational.representation_audit.spatial_cache' 'H'
        & $Python -u -m operational.representation_audit.spatial_model *> (Join-Path $Run "RESUME_${Stamp}_SPATIAL_CALIBRATION.log")
        if ($LASTEXITCODE -ne 0) { throw 'Fallo del ajuste cerrado espacial.' }
        Invoke-Stage 'operational.representation_audit.spatial_geometry' 'qa'
        Invoke-Stage 'operational.representation_audit.spatial_geometry' 'train'
        Invoke-Stage 'operational.representation_audit.spatial_geometry' 'evaluate'
        $Decision = Get-Content -LiteralPath (Join-Path $Run 'SPATIAL_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
        if ($Decision.status -eq 'REPLICATE') {
            Invoke-Stage 'operational.representation_audit.spatial_geometry' 'replicate'
            Invoke-Stage 'operational.representation_audit.spatial_geometry' 'evaluate-replicas'
        }
    }
    & $Python scripts/report_v17_queues.py *> (Join-Path $Run "RESUME_${Stamp}_REPORT.log")
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo verificar/regenerar el informe.' }
}
finally { Pop-Location }
