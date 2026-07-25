param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
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
$evttcArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\run_recovery_multiseed.ps1")
if ($Smoke) { $evttcArgs += "-Smoke" }
& powershell @evttcArgs
if ($LASTEXITCODE -ne 0) { throw "EvTTC matrix failed" }

$baseDir = if ($Smoke) { "artifacts/smoke/current" } else { "artifacts/runs" }
$dateStr = (Get-Date).ToString("o")
@{ artifact_type = "stage_record_v3"; phase = "evttc_matrix"; status = "passed"; timestamp = $dateStr } | ConvertTo-Json | Out-File -FilePath "$baseDir/phase_1_evttc.json" -Encoding utf8

Write-Output "=== PHASE 2: eAP Matrix ==="
$eapArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\run_eap_matrix.ps1")
if ($Smoke) { $eapArgs += "-Smoke" }
& powershell @eapArgs
if ($LASTEXITCODE -ne 0) { throw "eAP matrix failed" }

@{ artifact_type = "stage_record_v3"; phase = "eap_matrix"; status = "passed"; timestamp = $dateStr } | ConvertTo-Json | Out-File -FilePath "$baseDir/phase_2_eap.json" -Encoding utf8

Write-Output "=== PHASE 3: Checkpoint Selection ==="
$runsDir = if ($Smoke) { "artifacts/smoke/current/evttc" } else { "artifacts/runs" }
$currentCommit = (& git rev-parse HEAD).Trim()
$chkArgs = @("scripts/select_best_onnx_candidate.py", "--runs-dir", $runsDir, "--require-full-label", "--require-commit", $currentCommit)
$selectionJson = Invoke-Python @chkArgs
if ($LASTEXITCODE -ne 0) { throw "Checkpoint selection failed" }

$selectionFile = if ($Smoke) { "artifacts/smoke/current/onnx_selection.json" } else { "artifacts/onnx_selection.json" }
$selectionJson | Out-File -FilePath $selectionFile -Encoding utf8

Write-Output "=== PHASE 4: ONNX Validation ==="
if (-not (Test-Path $selectionFile)) {
    throw "Selection record missing: $selectionFile"
}

$onnxOut = if ($Smoke) { "artifacts/smoke/current/onnx/model.onnx" } else { "artifacts/onnx/final/model.onnx" }
$onnxOutDir = Split-Path $onnxOut -Parent
New-Item -ItemType Directory -Force -Path $onnxOutDir | Out-Null
$exportArgs = @("scripts/export_onnx.py", "--selection-record", $selectionFile, "--output", $onnxOut, "--sample-count", "32")
Invoke-Python @exportArgs
if ($LASTEXITCODE -ne 0) { throw "ONNX validation failed" }

@{ artifact_type = "stage_record_v3"; phase = "onnx_export"; status = "passed"; timestamp = $dateStr } | ConvertTo-Json | Out-File -FilePath "$baseDir/phase_4_onnx.json" -Encoding utf8

Write-Output "=== PHASE 5: Final Validation Gate ==="
function Assert-CompletionGate {
    if ($Smoke) {
        Write-Output "Running robust smoke completion verification..."
        $smokeArgs = @("scripts/verify_smoke_completion.py", "--smoke-dir", "artifacts/smoke/current")
        Invoke-Python @smokeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke Completion Gate Failed!"
        }
    } else {
        Write-Output "Running full matrix completion verification..."
        Invoke-Python "scripts/verify_full_completion.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Full Matrix Completion Gate Failed!"
        }
    }
}
Assert-CompletionGate
