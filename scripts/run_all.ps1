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

Write-Output "=== PHASE 0: Protocol Freezing ==="
Invoke-Python "scripts/freeze_protocol.py"
if ($LASTEXITCODE -ne 0) { throw "Protocol freezing failed" }

# Helper to get the frozen protocol hash
$frozenHash = ""
if (Test-Path "artifacts/audit/recovery_v3/frozen_protocol.json") {
    $frozenContent = Get-Content -Raw "artifacts/audit/recovery_v3/frozen_protocol.json" | ConvertFrom-Json
    $frozenHash = $frozenContent.protocol_sha256
    $protVersion = $frozenContent.protocol_version
} else {
    throw "Frozen protocol not found after freeze_protocol.py ran!"
}

function Get-ArtifactHash($path) {
    if (-not (Test-Path $path)) { return "missing" }
    return (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
}

function Write-StageRecord {
    param($baseDir, $stage, $status, $start, $end, $exit_code, $command, $failure=$null, $inputs=@(), $outputs=@())
    $commit = (& git rev-parse HEAD).Trim()
    
    $cleanStatus = (& git status --short)
    $clean = ($null -eq $cleanStatus) -or ($cleanStatus.Trim() -eq "")
    
    $duration = ($end - $start).TotalSeconds
    $evidence = if ($Smoke) { "synthetic_smoke" } else { "real_smoke" }
    
    # Evaluate hashes for outputs right before signing
    $processedOutputs = @()
    foreach ($out in $outputs) {
        $outPath = $out.path
        $outHash = Get-ArtifactHash $outPath
        $processedOutputs += @{
            path = $outPath
            sha256 = $outHash
            artifact_type = $out.artifact_type
        }
    }
    
    $record = @{
        artifact_type = "stage_record_v3"
        schema_version = "3.0"
        evidence_type = $evidence
        code_commit = $commit
        protocol_version = $protVersion
        protocol_sha256 = $frozenHash
        created_at = $end.ToString("o")
        
        stage = $stage
        status = $status
        started_at = $start.ToString("o")
        completed_at = $end.ToString("o")
        duration_s = $duration
        exit_code = $exit_code
        command = $command
        
        inputs = $inputs
        outputs = $processedOutputs
        failure = $failure
        environment_hash = "env_fake_hash_for_now"
        working_tree_clean = $clean
    }
    
    # We must self-sign this stage_record. Let's dump it, use python to sign it.
    $json = $record | ConvertTo-Json -Depth 5
    $file = if ($stage -match "evttc") { "phase_1_evttc.json" } elseif ($stage -match "eap") { "phase_2_eap.json" } else { "phase_4_onnx.json" }
    $outPath = "$baseDir/$file"
    $json | Out-File -FilePath $outPath -Encoding utf8
    
    # Run python snippet to self-sign
    $signScript = "
import json
from e_jepa_ttc.artifacts.hashing import sign_artifact
with open('$outPath', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)
data = sign_artifact(data)
with open('$outPath', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=True)
"
    Invoke-Python -c $signScript
}

$baseDir = if ($Smoke) { "artifacts/smoke/current" } else { "artifacts/runs" }

Write-Output "=== PHASE 1: EvTTC Matrix ==="
$start1 = Get-Date
$evttcArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\run_recovery_multiseed.ps1")
if ($Smoke) { $evttcArgs += "-Smoke" }
& powershell @evttcArgs
$exit1 = $LASTEXITCODE
$end1 = Get-Date

$expectedOutputs1 = @()
if ($exit1 -ne 0) { 
    Write-StageRecord -baseDir $baseDir -stage "evttc_matrix" -status "failed" -start $start1 -end $end1 -exit_code $exit1 -command $evttcArgs -failure "EvTTC matrix failed"
    throw "EvTTC matrix failed" 
}
Write-StageRecord -baseDir $baseDir -stage "evttc_matrix" -status "passed" -start $start1 -end $end1 -exit_code 0 -command $evttcArgs -outputs $expectedOutputs1

Write-Output "=== PHASE 2: eAP Matrix ==="
$start2 = Get-Date
$eapArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\run_eap_matrix.ps1")
if ($Smoke) { $eapArgs += "-Smoke" }
& powershell @eapArgs
$exit2 = $LASTEXITCODE
$end2 = Get-Date
$expectedOutputs2 = @()
if ($exit2 -ne 0) { 
    Write-StageRecord -baseDir $baseDir -stage "eap_matrix" -status "failed" -start $start2 -end $end2 -exit_code $exit2 -command $eapArgs -failure "eAP matrix failed"
    throw "eAP matrix failed" 
}
Write-StageRecord -baseDir $baseDir -stage "eap_matrix" -status "passed" -start $start2 -end $end2 -exit_code 0 -command $eapArgs -outputs $expectedOutputs2

Write-Output "=== PHASE 3: Checkpoint Selection ==="
$runsDir = if ($Smoke) { "artifacts/smoke/current/evttc" } else { "artifacts/runs" }
$currentCommit = (& git rev-parse HEAD).Trim()
$chkArgs = @("scripts/select_best_onnx_candidate.py", "--runs-dir", $runsDir, "--require-full-label", "--require-commit", $currentCommit)
$selectionJson = Invoke-Python @chkArgs
if ($LASTEXITCODE -ne 0) { throw "Checkpoint selection failed" }

$selectionFile = if ($Smoke) { "artifacts/smoke/current/onnx_selection.json" } else { "artifacts/onnx_selection.json" }
$selectionJson | Out-File -FilePath $selectionFile -Encoding utf8

Write-Output "=== PHASE 4: ONNX Validation ==="
$start4 = Get-Date
if (-not (Test-Path $selectionFile)) {
    throw "Selection record missing: $selectionFile"
}

$onnxOut = if ($Smoke) { "artifacts/smoke/current/onnx/model.onnx" } else { "artifacts/onnx/final/model.onnx" }
$onnxOutDir = Split-Path $onnxOut -Parent
New-Item -ItemType Directory -Force -Path $onnxOutDir | Out-Null
$exportArgs = @("scripts/export_onnx.py", "--selection-record", $selectionFile, "--output", $onnxOut, "--sample-count", "32")
Invoke-Python @exportArgs
$exit4 = $LASTEXITCODE
$end4 = Get-Date

$inputs4 = @(
    @{ path = $selectionFile; artifact_type = "onnx_candidate_v3"; sha256 = Get-ArtifactHash $selectionFile }
)
$outputs4 = @(
    @{ path = $onnxOut; artifact_type = "onnx_model" }
)

if ($exit4 -ne 0) { 
    Write-StageRecord -baseDir $baseDir -stage "onnx_export" -status "failed" -start $start4 -end $end4 -exit_code $exit4 -command $exportArgs -failure "ONNX validation failed" -inputs $inputs4
    throw "ONNX validation failed" 
}
Write-StageRecord -baseDir $baseDir -stage "onnx_export" -status "passed" -start $start4 -end $end4 -exit_code 0 -command $exportArgs -inputs $inputs4 -outputs $outputs4

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
