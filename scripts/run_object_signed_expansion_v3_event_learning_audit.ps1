param(
    [string]$ScratchDir =
        "artifacts\runs\e_jepa_garl_object_signed_expansion_screen_v3\scratch\seed-7",
    [string]$TransferDir =
        "artifacts\runs\e_jepa_garl_object_signed_expansion_screen_v3\level-transfer\seed-7",
    [string]$Manifest =
        "artifacts\cache\garl_object_lhr_screen_v2\manifest.json",
    [string]$OutputDir =
        "artifacts\debug\object_signed_expansion_v3_event_learning",
    [int]$SampleCount = 256,
    [int]$ProbeTrainCount = 256,
    [int]$ProbeValidationCount = 256
)

$ErrorActionPreference = "Stop"

# Keep this audit on CPU so it can run beside a CUDA training process.
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

uv run --no-sync python `
  artifacts\debug\audit_object_signed_expansion_v3_event_learning.py `
  --scratch-dir $ScratchDir `
  --transfer-dir $TransferDir `
  --manifest $Manifest `
  --output-dir $OutputDir `
  --device cpu `
  --sample-count $SampleCount `
  --probe-train-count $ProbeTrainCount `
  --probe-validation-count $ProbeValidationCount `
  --gradient-samples 4 `
  --batch-size 4

if ($LASTEXITCODE -ne 0) {
    throw "Falló la auditoría de aprendizaje de eventos v3"
}

Write-Host ""
Write-Host "Sube estos archivos:" -ForegroundColor Green
Write-Host "$OutputDir\event_learning_audit.json" -ForegroundColor Cyan
Write-Host "$OutputDir\event_learning_summary.csv" -ForegroundColor Cyan
