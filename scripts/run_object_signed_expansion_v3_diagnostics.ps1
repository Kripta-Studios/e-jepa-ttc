param(
    [string]$ScratchDir =
        "artifacts\runs\e_jepa_garl_object_signed_expansion_screen_v3\scratch\seed-7",
    [string]$TransferDir =
        "artifacts\runs\e_jepa_garl_object_signed_expansion_screen_v3\level-transfer\seed-7",
    [string]$Manifest =
        "artifacts\cache\garl_object_lhr_screen_v2\manifest.json",
    [string]$OutputDir =
        "artifacts\debug\object_signed_expansion_v3_perturbations",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

uv run --no-sync python `
  artifacts\debug\diagnose_object_signed_expansion_v3.py `
  --scratch-dir $ScratchDir `
  --transfer-dir $TransferDir `
  --manifest $Manifest `
  --output-dir $OutputDir `
  --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Falló el diagnóstico por perturbaciones v3"
}

Write-Host ""
Write-Host "Sube estos archivos:" -ForegroundColor Green
Write-Host "$OutputDir\perturbation_diagnostics.json" -ForegroundColor Cyan
Write-Host "$OutputDir\perturbation_summary.csv" -ForegroundColor Cyan
