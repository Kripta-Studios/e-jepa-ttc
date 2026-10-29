param(
    [string]$EapRoot = "E:\eAP_dataset",
    [string]$GarlRoot = "E:\GarlTTC_dataset"
)

$ErrorActionPreference = "Stop"

uv run --no-sync python `
  artifacts\debug\audit_eap_garl_objectcentric_patch.py `
  --repo-root . `
  --eap-root $EapRoot `
  --garlttc-root $GarlRoot `
  --output-dir artifacts\debug\eap_garl_objectcentric_audit

if ($LASTEXITCODE -ne 0) {
    throw "La auditoría falló con código $LASTEXITCODE"
}

Write-Host ""
Write-Host "Sube este archivo:" -ForegroundColor Green
Write-Host "artifacts\debug\eap_garl_objectcentric_audit.zip" -ForegroundColor Cyan
