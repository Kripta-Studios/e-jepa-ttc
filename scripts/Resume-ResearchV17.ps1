param(
    [ValidateSet('all', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'I')]
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
function Assert-Fits([string[]]$Names) {
    foreach ($Name in $Names) {
        $ReceiptPath = Join-Path $Run "fits/$Name/CHECKPOINT_RECEIPT.json"
        if (-not (Test-Path -LiteralPath $ReceiptPath)) {
            throw "Fit pendiente: $Name. Reanuda esta cola; no se abrirá la evaluación."
        }
        $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
        if ($Receipt.status -ne 'COMPLETE') {
            throw "Fit $Name pausado en $($Receipt.completed_updates). Reanuda esta cola; no se abrirá la evaluación."
        }
    }
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
        Assert-Fits @('SP0-S7', 'SP1-S7')
        Invoke-Stage 'operational.representation_audit.spatial_geometry' 'evaluate'
        $Decision = Get-Content -LiteralPath (Join-Path $Run 'SPATIAL_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
        if ($Decision.status -eq 'REPLICATE') {
            Invoke-Stage 'operational.representation_audit.spatial_geometry' 'replicate'
            Invoke-Stage 'operational.representation_audit.spatial_geometry' 'evaluate-replicas'
        }
    }
    # Preserve the prospectively registered conditional-family order G then H.
    if ($Queue -in @('all', 'H')) {
        $Module = 'operational.representation_audit.unclipped'
        Invoke-Stage $Module 'qa'
        Invoke-Stage $Module 'train'
        Assert-Fits @('N0-S7')
        Invoke-Stage $Module 'evaluate'
        $Decision = Get-Content -LiteralPath (Join-Path $Run 'NOCLIP_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
        if ($Decision.status -eq 'REPLICATE') {
            Invoke-Stage $Module 'replicate'
            Invoke-Stage $Module 'evaluate-replicas'
        }
    }
    if ($Queue -in @('all', 'I')) {
        $G = Get-Content -LiteralPath (Join-Path $Run 'SPATIAL_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
        $H = Get-Content -LiteralPath (Join-Path $Run 'NOCLIP_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
        if ($G.status -eq 'NO_CANDIDATE' -and $H.status -eq 'NO_CANDIDATE') {
            $Module = 'operational.representation_audit.nonlinear'
            Invoke-Stage $Module 'qa-geometry'
            Invoke-Stage $Module 'geometry'
            Assert-Fits @('NLH-S7')
            Invoke-Stage $Module 'qa-scorer'
            Invoke-Stage $Module 'scorer'
            Assert-Fits @('NL1-S7')
            Invoke-Stage $Module 'evaluate'
            $Decision = Get-Content -LiteralPath (Join-Path $Run 'NONLINEAR_SCREEN_DECISION.json') -Raw | ConvertFrom-Json
            if ($Decision.status -eq 'REPLICATE') {
                Invoke-Stage $Module 'replicate'
                Assert-Fits @('NLH-S13', 'NLH-S23', 'NL0-S13', 'NL0-S23', 'NL1-S13', 'NL1-S23')
                Invoke-Stage $Module 'evaluate-replicas'
            }
        }
        elseif ($Queue -eq 'I') { throw 'I sólo se activa tras el rechazo de G y H.' }
    }
    & $Python scripts/report_v17_queues.py *> (Join-Path $Run "RESUME_${Stamp}_REPORT.log")
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo verificar/regenerar el informe.' }
}
finally { Pop-Location }
