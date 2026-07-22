param()

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Output "=== PHASE 1: EvTTC Matrix ==="
powershell -ExecutionPolicy Bypass -File scripts\run_recovery_multiseed.ps1
if ($LASTEXITCODE -ne 0) { throw "EvTTC matrix failed" }

Write-Output "=== PHASE 2: eAP Matrix ==="
powershell -ExecutionPolicy Bypass -File scripts\run_eap_matrix.ps1
if ($LASTEXITCODE -ne 0) { throw "eAP matrix failed" }

Write-Output "=== ALL MATRICES COMPLETED SUCCESSFULLY ==="
