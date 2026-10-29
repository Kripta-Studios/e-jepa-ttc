param(
    [string]$ScratchDir =
        "artifacts\runs\e_jepa_garl_object_expansion_screen_v2\scratch\seed-7",

    [string]$TransferDir =
        "artifacts\runs\e_jepa_garl_object_expansion_screen_v2\level-transfer\seed-7",

    [string]$Manifest =
        "artifacts\cache\garl_object_lhr_screen_v2\manifest.json",

    [string]$OutputDir =
        "artifacts\debug\object_expansion_v2_diagnostics",

    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

uv run --no-sync python `
  artifacts\debug\diagnose_object_expansion_v2.py `
  --scratch-dir $ScratchDir `
  --transfer-dir $TransferDir `
  --manifest $Manifest `
  --output-dir $OutputDir `
  --device $Device `
  --probe-steps 600

if ($LASTEXITCODE -ne 0) {
    throw "Falló el diagnóstico Object-Expansion v2"
}

Write-Host ""
Write-Host "Sube estos dos archivos:" -ForegroundColor Green
Write-Host "$OutputDir\diagnostics.json" -ForegroundColor Cyan
Write-Host "$OutputDir\diagnostics_summary.csv" -ForegroundColor Cyan
