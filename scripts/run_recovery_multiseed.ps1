param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$dirty = git status --porcelain
if ($dirty -and -not $Smoke) {
    throw "Recovery runs require a clean committed worktree. Use -Smoke only for smoke_only diagnostics."
}
$baselineCommit = git rev-parse HEAD
if (-not (Test-Path ".venv/Scripts/python.exe")) {
    throw "Missing .venv/Scripts/python.exe"
}

$env:PYTHONPATH = (Join-Path $repo "src")
$python = ".venv/Scripts/python.exe"
$cache = "artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz"
$seeds = if ($Smoke) { @(7) } else { @(7, 13, 21) }
$sslEpochs = if ($Smoke) { 2 } else { 30 }
$downstreamEpochs = if ($Smoke) { 2 } else { 80 }
$downstreamSeeds = if ($Smoke) { @(7) } else { @(7, 13, 21) }
$suffix = if ($Smoke) { "smoke_only" } else { "post_fix_v3_cache_verified" }

foreach ($seed in $seeds) {
    $sslOut = "artifacts/runs/recovery_jepa_tubeletmask_transformer_seed${seed}_${suffix}"
    $sslStarted = Get-Date -Format o
    $sslCommand = "$python -m e_jepa_ttc pretrain jepa --cache $cache --output-dir $sslOut --epochs $sslEpochs --batch-size 24 --learning-rate 0.0003 --seed $seed --device auto --model event-tubelet-transformer --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --mask-ratio 0.45 --block-count 4 --mask-mode tubelet --ema-momentum 0.99 --variance-weight 1.0 --min-std 0.05 --dense-predictor transformer"
    if (Test-Path "$sslOut/jepa_encoder_best.pt") {
        Write-Output "SSL pretraining for seed $seed already exists. Skipping pretraining."
        $sslCompleted = Get-Date -Format o
    } else {
        & $python -m e_jepa_ttc pretrain jepa `
            --cache $cache `
            --output-dir $sslOut `
            --epochs $sslEpochs `
            --batch-size 24 `
            --learning-rate 0.0003 `
            --seed $seed `
            --device auto `
            --model event-tubelet-transformer `
            --pretrain-splits train `
            --validation-splits validation `
            --temporal-horizons-ms 20 60 100 240 500 `
            --max-target-slop-ms 10 `
            --mask-ratio 0.45 `
            --block-count 4 `
            --mask-mode tubelet `
            --ema-momentum 0.99 `
            --variance-weight 1.0 `
            --min-std 0.05 `
            --dense-predictor transformer
        if ($LASTEXITCODE -ne 0) { throw "SSL seed $seed failed" }
        $sslCompleted = Get-Date -Format o
    }
    $registerArgs = @(
        "scripts/register_recovery_run.py", "--run-id", "recovery-ssl-${seed}-${suffix}",
        "--stage", "ssl_pretrain", "--run-dir", $sslOut, "--pretrain-seed", $seed,
        "--requested-backbone", "event-tubelet-transformer", "--command", $sslCommand,
        "--started-at", $sslStarted, "--completed-at", $sslCompleted,
        "--expected-commit", $baselineCommit
    )
    if ($Smoke) { $registerArgs += "--smoke" }
    & $python @registerArgs
    if ($LASTEXITCODE -ne 0) { throw "SSL seed $seed registry append failed" }

    foreach ($downstreamSeed in $downstreamSeeds) {
        $downstreamOut = "artifacts/runs/recovery_downstream_ssl${seed}_seed${downstreamSeed}_${suffix}"
        $downstreamStarted = Get-Date -Format o
        $downstreamCommand = "$python -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $downstreamOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --pretrained-encoder $sslOut/jepa_encoder_best.pt --train-splits train --validation-splits validation --evaluation-splits train validation"
        if (Test-Path "$downstreamOut/tiny_cnn_best.pt") {
            Write-Output "Downstream training for pretrain seed $seed and downstream seed $downstreamSeed already exists. Skipping downstream training."
            $downstreamCompleted = Get-Date -Format o
        } else {
            & $python -m e_jepa_ttc train tiny-cnn `
                --cache $cache `
                --output-dir $downstreamOut `
                --epochs $downstreamEpochs `
                --batch-size 32 `
                --learning-rate 0.00003 `
                --seed $downstreamSeed `
                --device auto `
                --model event-tubelet-transformer `
                --pretrained-encoder "$sslOut/jepa_encoder_best.pt" `
                --train-splits train `
                --validation-splits validation `
                --evaluation-splits train validation
            if ($LASTEXITCODE -ne 0) { throw "Downstream SSL $seed seed $downstreamSeed failed" }
            $downstreamCompleted = Get-Date -Format o
        }
        $registerArgs = @(
            "scripts/register_recovery_run.py", "--run-id", "recovery-downstream-ssl${seed}-seed${downstreamSeed}-${suffix}",
            "--stage", "downstream_ttc", "--run-dir", $downstreamOut,
            "--pretrain-seed", $seed, "--downstream-seed", $downstreamSeed,
            "--requested-backbone", "event-tubelet-transformer", "--command", $downstreamCommand,
            "--started-at", $downstreamStarted, "--completed-at", $downstreamCompleted,
            "--expected-commit", $baselineCommit
        )
        if ($Smoke) { $registerArgs += "--smoke" }
        & $python @registerArgs
        if ($LASTEXITCODE -ne 0) { throw "Downstream SSL $seed seed $downstreamSeed registry append failed" }

        # Low-label finetuning
        foreach ($frac in @(0.1, 0.01)) {
            $fracStr = "frac$($frac * 100)"
            $lowLabelOut = "artifacts/runs/recovery_downstream_ssl${seed}_seed${downstreamSeed}_${fracStr}_${suffix}"
            $lowLabelStarted = Get-Date -Format o
            $lowLabelCommand = "$python -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $lowLabelOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --pretrained-encoder $sslOut/jepa_encoder_best.pt --train-splits train --validation-splits validation --evaluation-splits train validation --train-fraction $frac"
            if (Test-Path "$lowLabelOut/tiny_cnn_best.pt") {
                Write-Output "Low-label training ($fracStr) for pretrain seed $seed and downstream seed $downstreamSeed already exists. Skipping."
                $lowLabelCompleted = Get-Date -Format o
            } else {
                & $python -m e_jepa_ttc train tiny-cnn `
                    --cache $cache `
                    --output-dir $lowLabelOut `
                    --epochs $downstreamEpochs `
                    --batch-size 32 `
                    --learning-rate 0.00003 `
                    --seed $downstreamSeed `
                    --device auto `
                    --model event-tubelet-transformer `
                    --pretrained-encoder "$sslOut/jepa_encoder_best.pt" `
                    --train-splits train `
                    --validation-splits validation `
                    --evaluation-splits train validation `
                    --train-fraction $frac
                if ($LASTEXITCODE -ne 0) { throw "Low-label SSL $seed seed $downstreamSeed frac $frac failed" }
                $lowLabelCompleted = Get-Date -Format o
            }
            $registerArgs = @(
                "scripts/register_recovery_run.py", "--run-id", "recovery-lowlabel-ssl${seed}-seed${downstreamSeed}-${fracStr}-${suffix}",
                "--stage", "downstream_ttc", "--run-dir", $lowLabelOut,
                "--pretrain-seed", $seed, "--downstream-seed", $downstreamSeed,
                "--requested-backbone", "event-tubelet-transformer", "--command", $lowLabelCommand,
                "--started-at", $lowLabelStarted, "--completed-at", $lowLabelCompleted,
                "--expected-commit", $baselineCommit
            )
            if ($Smoke) { $registerArgs += "--smoke" }
            & $python @registerArgs
            if ($LASTEXITCODE -ne 0) { throw "Low-label SSL $seed seed $downstreamSeed frac $frac registry append failed" }
        }
    }
}

foreach ($downstreamSeed in $downstreamSeeds) {
    $scratchOut = "artifacts/runs/recovery_scratch_seed${downstreamSeed}_${suffix}"
    $scratchStarted = Get-Date -Format o
    $scratchCommand = "$python -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $scratchOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --train-splits train --validation-splits validation --evaluation-splits train validation"
    if (Test-Path "$scratchOut/tiny_cnn_best.pt") {
        Write-Output "Scratch training for seed $downstreamSeed already exists. Skipping."
        $scratchCompleted = Get-Date -Format o
    } else {
        & $python -m e_jepa_ttc train tiny-cnn `
            --cache $cache `
            --output-dir $scratchOut `
            --epochs $downstreamEpochs `
            --batch-size 32 `
            --learning-rate 0.00003 `
            --seed $downstreamSeed `
            --device auto `
            --model event-tubelet-transformer `
            --train-splits train `
            --validation-splits validation `
            --evaluation-splits train validation
        if ($LASTEXITCODE -ne 0) { throw "Scratch seed $downstreamSeed failed" }
        $scratchCompleted = Get-Date -Format o
    }
    $registerArgs = @(
        "scripts/register_recovery_run.py", "--run-id", "recovery-scratch-seed${downstreamSeed}-${suffix}",
        "--stage", "scratch_ttc", "--run-dir", $scratchOut,
        "--downstream-seed", $downstreamSeed,
        "--requested-backbone", "event-tubelet-transformer", "--command", $scratchCommand,
        "--started-at", $scratchStarted, "--completed-at", $scratchCompleted,
        "--expected-commit", $baselineCommit
    )
    if ($Smoke) { $registerArgs += "--smoke" }
    & $python @registerArgs
    if ($LASTEXITCODE -ne 0) { throw "Scratch seed $downstreamSeed registry append failed" }
}

Write-Output "All recovery runs and registrations complete."

& $python scripts/validate_artifact_registry.py
if ($LASTEXITCODE -ne 0) { throw "Final registry validation failed" }
Write-Output "Completed 3x3 validation-only recovery runs. CPLA-high was not opened."
