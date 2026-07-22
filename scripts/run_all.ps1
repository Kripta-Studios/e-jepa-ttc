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

Write-Output "=== PHASE 3: ONNX Validation ==="
$onnxCheckpoint = "artifacts/runs/recovery_downstream_ssl7_seed7_post_fix_v3_cache_verified/tiny_cnn_best.pt"
if (Test-Path $onnxCheckpoint) {
    & $python scripts/export_onnx.py --checkpoint $onnxCheckpoint --output artifacts/runs/ttc_model.onnx --model-type event-tubelet-transformer
    if ($LASTEXITCODE -ne 0) { throw "ONNX validation failed" }
} else {
    Write-Output "Checkpoint not found, skipping ONNX validation"
}

Write-Output "=== ALL MATRICES COMPLETED SUCCESSFULLY ==="
