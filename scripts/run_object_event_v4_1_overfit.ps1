param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$Config = "configs\experiment\e_jepa_garl_object_event_overfit_v4_1.yaml",
    [string]$OutputDir = "artifacts\debug\object_event_v4_1_overfit",
    [string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CacheManifest)) {
    throw "No existe el manifest de caché v4: $CacheManifest"
}

uv run --no-sync python scripts\preflight_object_event_v4_1.py `
    --cache-manifest $CacheManifest
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.1" }

$Arguments = @(
    "run", "--no-sync", "python",
    "scripts\diagnose_object_event_v4_1.py",
    "--cache-manifest", $CacheManifest,
    "--config", $Config,
    "--output-dir", $OutputDir,
    "--device", $Device
)
if ($Force) { $Arguments += "--force" }

& uv @Arguments
if ($LASTEXITCODE -ne 0) { throw "Falló operacionalmente el diagnóstico v4.1" }

$Summary = Get-Content -LiteralPath (Join-Path $OutputDir "summary.json") -Raw |
    ConvertFrom-Json

Write-Host ""
Write-Host "Resultado v4.1: $($Summary.status)" -ForegroundColor Cyan
Write-Host "Overfit superado: $($Summary.overfit_passed)"
Write-Host "Screen superado:  $($Summary.screen_passed)"
Write-Host "Best step:        $($Summary.best_step)"
Write-Host "Train Pearson:    $($Summary.train_metrics.branches.event.pearson)"
Write-Host "Validation Pearson: $($Summary.validation_metrics.branches.event.pearson)"

if (-not $Summary.overfit_passed) {
    Write-Warning "La rama event-only no puede memorizar 64 muestras. No avanzar a entrenamiento completo."
} elseif (-not $Summary.screen_passed) {
    Write-Warning "Memoriza train pero no generaliza ni depende suficientemente de eventos. No avanzar a full."
} else {
    Write-Host "Los gates v4.1 pasan. Ya se justifica diseñar el screen event-only completo." -ForegroundColor Green
}
