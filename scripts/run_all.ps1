param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if ($Smoke) {
    if (Test-Path "artifacts/smoke/current") {
        Remove-Item -Recurse -Force "artifacts/smoke/current"
    }
    New-Item -ItemType Directory -Force "artifacts/smoke/current" | Out-Null
}

function Invoke-Python {
    uv run --no-sync python @args
}

Write-Output "=== PHASE 1: EvTTC Matrix ==="
$evttcCommand = "powershell -ExecutionPolicy Bypass -File scripts\run_recovery_multiseed.ps1"
if ($Smoke) { $evttcCommand += " -Smoke" }
Invoke-Expression $evttcCommand
if ($LASTEXITCODE -ne 0) { throw "EvTTC matrix failed" }

Write-Output "=== PHASE 2: eAP Matrix ==="
$eapCommand = "powershell -ExecutionPolicy Bypass -File scripts\run_eap_matrix.ps1"
if ($Smoke) { $eapCommand += " -Smoke" }
Invoke-Expression $eapCommand
if ($LASTEXITCODE -ne 0) { throw "eAP matrix failed" }

Write-Output "=== PHASE 3: Checkpoint Selection ==="
$runsDir = if ($Smoke) { "artifacts/smoke/current/evttc" } else { "artifacts/runs" }
$cmd = "uv run --no-sync python scripts/select_best_onnx_candidate.py --runs-dir $runsDir"
$onnxCheckpoint = Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) { throw "Checkpoint selection failed" }

Write-Output "=== PHASE 4: ONNX Validation ==="
if (-not (Test-Path $onnxCheckpoint)) {
    throw "Validation-selected ONNX checkpoint is missing: $onnxCheckpoint"
}

$onnxOut = if ($Smoke) { "artifacts/smoke/current/ttc_model.onnx" } else { "artifacts/runs/ttc_model.onnx" }
$cmd = "uv run --no-sync python scripts/export_onnx.py --checkpoint $onnxCheckpoint --output $onnxOut"
Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) { throw "ONNX validation failed" }

Write-Output "=== PHASE 5: Final Validation Gate ==="
function Assert-CompletionGate {
    $eapSplitStats = if ($Smoke) { "artifacts/smoke/current/eap/matrix/eap_split_statistics.json" } else { "artifacts/data_audit/eap_split_statistics.json" }
    $manifestPath = if ($Smoke) { "artifacts/features/smoke_nav_voxel.summary.json" } else { "artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.summary.json" }
    
    $requiredFiles = @(
        $manifestPath,
        $onnxOut,
        $eapSplitStats
    )
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            throw "Completion Gate Failed: Missing required artifact $file"
        }
    }
}
Assert-CompletionGate

Write-Output "=== ALL MATRICES COMPLETED SUCCESSFULLY ==="
