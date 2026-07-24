param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Invoke-Python {
    uv run --no-sync python @args
}

$eapRoot = "datasets/eap-50"
$cacheOut = if ($Smoke) { "artifacts/smoke/current/eap/cache" } else { "artifacts/features/eap50_object_cache" }
$cacheManifest = "$cacheOut/manifest.json"

if (-not (Test-Path $cacheManifest)) {
    Write-Output "Building eAP object cache..."
    if ($Smoke) {
        $argsArray = @(
            "-m", "e_jepa_ttc", "cache", "eap-object",
            "--eap-root", $eapRoot,
            "--output-dir", $cacheOut,
            "--sequence-split", "2cyv0Oedzg=train",
            "--sequence-split", "mHGFBekt7X=validation",
            "--sequence-split", "pBqGOb2vYq=calibration",
            "--max-windows-per-sequence", "50"
        )
    } else {
        $argsArray = @(
            "-m", "e_jepa_ttc", "cache", "eap-object",
            "--eap-root", $eapRoot,
            "--output-dir", $cacheOut,
            "--sequence-split", "2cyv0Oedzg=train",
            "--sequence-split", "6h5yRW2LGc=train",
            "--sequence-split", "DGqicHUGWb=train",
            "--sequence-split", "OBneIVg4Cw=train",
            "--sequence-split", "mHGFBekt7X=validation",
            "--sequence-split", "pBqGOb2vYq=calibration",
            "--sequence-split", "qGsgzl4Q8B=test",
            "--sequence-split", "qoohcdtLDH=test"
        )
    }
    Invoke-Python @argsArray `
        --history-frames 5 `
        --prediction-horizons-ms 50 100 250 500 `
        --event-window-ms 50 `
        --roi-width 64 `
        --roi-height 64 `
        --roi-expansion 1.2 `
        --event-bins 5 `
        --workers 4
    if ($LASTEXITCODE -ne 0) { throw "eAP cache generation failed" }
} else {
    Write-Output "eAP object cache already exists at $cacheManifest"
}

$baseDir = if ($Smoke) { "artifacts/smoke/current" } else { "artifacts/runs" }
$dateStr = (Get-Date).ToString("o")
@{ phase = "eap_cache"; status = "passed"; timestamp = $dateStr } | ConvertTo-Json | Out-File -FilePath "$baseDir/phase_eap_cache.json" -Encoding utf8

$matrixOut = if ($Smoke) { "artifacts/smoke/current/eap/matrix" } else { "artifacts/runs/eap_object_jepa_matrix" }
New-Item -ItemType Directory -Force -Path $matrixOut | Out-Null

Write-Output "Generating eAP Split Statistics..."
$eapSplitStats = "$matrixOut/eap_split_statistics.json"
Invoke-Python scripts/generate_split_statistics.py --manifest $cacheManifest --output $eapSplitStats
if ($LASTEXITCODE -ne 0) { throw "eAP split statistics generation failed (possible data corruption)" }

$seedsStr = if ($Smoke) { "7" } else { "7 13 21" }
$epochsPre = if ($Smoke) { 2 } else { 30 }
$epochsFine = if ($Smoke) { 2 } else { 40 }

Write-Output "Running eAP Object-JEPA Matrix..."
$cmdArgs = @(
    "scripts/run_object_jepa_matrix.py",
    "--cache-manifest", $cacheManifest,
    "--output-dir", $matrixOut,
    "--seeds"
) + ($seedsStr -split " ") + @(
    "--label-fractions", "1.0", "0.10", "0.05",
    "--pretrain-epochs", "$epochsPre",
    "--finetune-epochs", "$epochsFine",
    "--batch-size", "32",
    "--device", "auto"
)

$cmdArgs += "--report-splits"
$cmdArgs += "validation"
$cmdArgs += "calibration"

Invoke-Python @cmdArgs
if ($LASTEXITCODE -ne 0) { throw "eAP matrix execution failed" }

@{ phase = "eap_matrix_inner"; status = "passed"; timestamp = $dateStr } | ConvertTo-Json | Out-File -FilePath "$baseDir/phase_eap_matrix_inner.json" -Encoding utf8

Write-Output "eAP matrix and ablations completed successfully!"
