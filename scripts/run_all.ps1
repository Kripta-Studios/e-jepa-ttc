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
$currentCommit = (git rev-parse HEAD).Trim()
$cmd = "uv run --no-sync python scripts/select_best_onnx_candidate.py --runs-dir $runsDir --require-full-label --require-commit $currentCommit"
$onnxCheckpoint = Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) { throw "Checkpoint selection failed" }

Write-Output "=== PHASE 4: ONNX Validation ==="
if (-not (Test-Path $onnxCheckpoint)) {
    throw "Validation-selected ONNX checkpoint is missing: $onnxCheckpoint"
}

$onnxOut = if ($Smoke) { "artifacts/smoke/current/onnx/model.onnx" } else { "artifacts/onnx/final/model.onnx" }
$onnxOutDir = Split-Path $onnxOut -Parent
New-Item -ItemType Directory -Force -Path $onnxOutDir | Out-Null
$cmd = "uv run --no-sync python scripts/export_onnx.py --checkpoint $onnxCheckpoint --output $onnxOut --cache artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --validation-split validation --sample-count 32"
Invoke-Expression $cmd
if ($LASTEXITCODE -ne 0) { throw "ONNX validation failed" }

Write-Output "=== PHASE 5: Final Validation Gate ==="
function Assert-CompletionGate {
    if ($Smoke) {
        Write-Output "Running robust smoke completion verification..."
        Invoke-Expression "uv run --no-sync python scripts/verify_smoke_completion.py --smoke-dir artifacts/smoke/current"
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke Completion Gate Failed!"
        }
    } else {
        Write-Output "Running full matrix completion verification..."
        Invoke-Expression "uv run --no-sync python scripts/verify_full_completion.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Full Matrix Completion Gate Failed!"
        }
    }
}
Assert-CompletionGate
