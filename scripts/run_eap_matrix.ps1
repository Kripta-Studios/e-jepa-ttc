param()

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = ".venv/Scripts/python.exe"
$eapRoot = "datasets/eap-50"
$cacheOut = "artifacts/features/eap50_object_cache"
$cacheManifest = "artifacts/features/eap50_object_cache/manifest.json"

if (-not (Test-Path $cacheManifest)) {
    Write-Output "Building eAP object cache..."
    & $python -m e_jepa_ttc cache eap-object `
        --eap-root $eapRoot `
        --output-dir $cacheOut `
        --sequence-split 2cyv0Oedzg=train `
        --sequence-split 6h5yRW2LGc=train `
        --sequence-split DGqicHUGWb=train `
        --sequence-split OBneIVg4Cw=train `
        --sequence-split mHGFBekt7X=validation `
        --sequence-split pBqGOb2vYq=validation `
        --sequence-split qGsgzl4Q8B=test `
        --sequence-split qoohcdtLDH=test `
        --history-frames 5 `
        --prediction-horizons-ms 20 60 100 240 500 `
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

$matrixOut = "artifacts/runs/eap_object_jepa_matrix"

Write-Output "Running eAP Object-JEPA Matrix..."
& $python scripts/run_object_jepa_matrix.py `
    --cache-manifest $cacheManifest `
    --output-dir $matrixOut `
    --seeds 7 13 21 `
    --label-fractions 1.0 0.1 0.01 `
    --pretrain-epochs 30 `
    --finetune-epochs 40 `
    --batch-size 32 `
    --device auto
if ($LASTEXITCODE -ne 0) { throw "eAP matrix execution failed" }

$ablationOut = "artifacts/runs/eap_object_jepa_ablation_no_ego"
Write-Output "Running eAP Ego-Action Ablation..."
& $python scripts/run_object_jepa_matrix.py `
    --cache-manifest $cacheManifest `
    --output-dir $ablationOut `
    --seeds 7 `
    --label-fractions 1.0 `
    --pretrain-epochs 30 `
    --finetune-epochs 40 `
    --batch-size 32 `
    --no-ego-actions `
    --device auto
if ($LASTEXITCODE -ne 0) { throw "eAP ego-action ablation execution failed" }

Write-Output "eAP matrix and ablations completed successfully!"
