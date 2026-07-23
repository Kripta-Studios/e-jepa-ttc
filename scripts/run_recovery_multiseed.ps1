param(
    [switch]$Smoke
)

if ($Smoke) {
    Write-Output "Running in Smoke Test Mode"
}

# Ensure execution policy is set correctly (only needed if running standalone, but good practice)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Invoke-Python {
    uv run --no-sync python @args
}
$pythonCmdStr = "uv run --no-sync python"

$baselineCommit = & git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Not in a git repository or no commits." }

$cache = "artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz"
$seeds = if ($Smoke) { @(7) } else { @(7, 13, 21) }
$sslEpochs = if ($Smoke) { 2 } else { 30 }
$downstreamEpochs = if ($Smoke) { 2 } else { 80 }
$downstreamSeeds = if ($Smoke) { @(7) } else { @(7, 13, 21) }
$suffix = if ($Smoke) { "smoke_only" } else { "post_fix_v3_cache_verified" }
$fractions = if ($Smoke) { @(0.10) } else { @(0.10, 0.05) }
$navModes = if ($Smoke) { @("enabled") } else { @("enabled", "disabled") }

$outRoot = if ($Smoke) { "artifacts/smoke/current/evttc" } else { "artifacts/runs" }

foreach ($navMode in $navModes) {
    # 1. Scratch Training
    foreach ($downstreamSeed in $downstreamSeeds) {
        $scratchOut = "$outRoot/recovery_scratch_nav${navMode}_seed${downstreamSeed}_${suffix}"
        $scratchStarted = Get-Date -Format o
        $scratchCommand = "$pythonCmdStr -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $scratchOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --navigation-mode $navMode --train-splits train --validation-splits validation --evaluation-splits train validation"
        if (-not $Smoke -and (Test-Path "$scratchOut/tiny_cnn_best.pt")) {
            Write-Output "Scratch full training ($navMode) for seed $downstreamSeed already exists. Skipping."
            $scratchCompleted = Get-Date -Format o
        } else {
            Invoke-Python -m e_jepa_ttc train tiny-cnn `
                --cache $cache `
                --output-dir $scratchOut `
                --epochs $downstreamEpochs `
                --batch-size 32 `
                --learning-rate 0.00003 `
                --seed $downstreamSeed `
                --device auto `
                --model event-tubelet-transformer `
                --navigation-mode $navMode `
                --train-splits train `
                --validation-splits validation `
                --evaluation-splits train validation
            if ($LASTEXITCODE -ne 0) { throw "Scratch seed $downstreamSeed failed" }
            $scratchCompleted = Get-Date -Format o
        }
        $registerArgs = @(
            "scripts/register_recovery_run.py", "--run-id", "recovery-scratch-nav${navMode}-seed${downstreamSeed}-${suffix}",
            "--stage", "scratch_ttc", "--run-dir", $scratchOut,
            "--downstream-seed", $downstreamSeed,
            "--requested-backbone", "event-tubelet-transformer", "--command", $scratchCommand,
            "--started-at", $scratchStarted, "--completed-at", $scratchCompleted,
            "--expected-commit", $baselineCommit
        )
        if ($Smoke) { $registerArgs += "--smoke" }
        Invoke-Python @registerArgs
        if ($LASTEXITCODE -ne 0) { throw "Scratch nav ${navMode} seed $downstreamSeed registry append failed" }

        # Low-label Scratch finetuning
        foreach ($frac in $fractions) {
            $fracStr = "frac$([math]::Round($frac * 100))"
            $manifestPath = "artifacts/subsets/evttc_${fracStr}_seed${downstreamSeed}.json"
            $lowLabelScratchOut = "$outRoot/recovery_scratch_nav${navMode}_seed${downstreamSeed}_${fracStr}_${suffix}"
            $lowLabelScratchStarted = Get-Date -Format o
            $lowLabelScratchCommand = "$pythonCmdStr -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $lowLabelScratchOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --navigation-mode $navMode --train-splits train --validation-splits validation --evaluation-splits train validation --train-fraction $frac --subset-manifest-path $manifestPath"
            
            if (-not $Smoke -and (Test-Path "$lowLabelScratchOut/tiny_cnn_best.pt")) {
                Write-Output "Scratch low-label training ($navMode, $fracStr) for seed $downstreamSeed already exists. Skipping."
                $lowLabelScratchCompleted = Get-Date -Format o
            } else {
                Invoke-Python -m e_jepa_ttc train tiny-cnn `
                    --cache $cache `
                    --output-dir $lowLabelScratchOut `
                    --epochs $downstreamEpochs `
                    --batch-size 32 `
                    --learning-rate 0.00003 `
                    --seed $downstreamSeed `
                    --device auto `
                    --model event-tubelet-transformer `
                    --navigation-mode $navMode `
                    --train-splits train `
                    --validation-splits validation `
                    --evaluation-splits train validation `
                    --train-fraction $frac `
                    --subset-manifest-path $manifestPath
                if ($LASTEXITCODE -ne 0) { throw "Low-label Scratch nav ${navMode} seed $downstreamSeed frac $frac failed" }
                $lowLabelScratchCompleted = Get-Date -Format o
            }
            $registerArgs = @(
                "scripts/register_recovery_run.py", "--run-id", "recovery-lowlabel-scratch-nav${navMode}-seed${downstreamSeed}-${fracStr}-${suffix}",
                "--stage", "scratch_ttc", "--run-dir", $lowLabelScratchOut,
                "--downstream-seed", $downstreamSeed,
                "--requested-backbone", "event-tubelet-transformer", "--command", $lowLabelScratchCommand,
                "--started-at", $lowLabelScratchStarted, "--completed-at", $lowLabelScratchCompleted,
                "--expected-commit", $baselineCommit
            )
            if ($Smoke) { $registerArgs += "--smoke" }
            Invoke-Python @registerArgs
            if ($LASTEXITCODE -ne 0) { throw "Low-label Scratch nav ${navMode} seed $downstreamSeed frac $frac registry append failed" }
        }
    }

    # 2. JEPA Pretraining and Fine-tuning
    foreach ($seed in $seeds) {
        $sslOut = "$outRoot/recovery_jepa_nav${navMode}_seed${seed}_${suffix}"
        $sslStarted = Get-Date -Format o
        $sslCommand = "$pythonCmdStr -m e_jepa_ttc pretrain jepa --cache $cache --output-dir $sslOut --epochs $sslEpochs --batch-size 128 --learning-rate 0.0005 --seed $seed --device auto --model event-tubelet-transformer --navigation-mode $navMode --pretrain-splits train --validation-splits validation"
        
        if (-not $Smoke -and (Test-Path "$sslOut/jepa_encoder_best.pt")) {
            Write-Output "SSL pretraining ($navMode) for seed $seed already exists. Skipping."
            $sslCompleted = Get-Date -Format o
        } else {
            Invoke-Python -m e_jepa_ttc pretrain jepa `
                --cache $cache `
                --output-dir $sslOut `
                --epochs $sslEpochs `
                --batch-size 24 `
                --learning-rate 0.0003 `
                --seed $seed `
                --device auto `
                --model event-tubelet-transformer `
                --navigation-mode $navMode `
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
            if ($LASTEXITCODE -ne 0) { throw "SSL nav ${navMode} seed $seed failed" }
            $sslCompleted = Get-Date -Format o
        }
        $registerArgs = @(
            "scripts/register_recovery_run.py", "--run-id", "recovery-ssl-nav${navMode}-${seed}-${suffix}",
            "--stage", "ssl_pretrain", "--run-dir", $sslOut, "--pretrain-seed", $seed,
            "--requested-backbone", "event-tubelet-transformer", "--command", $sslCommand,
            "--started-at", $sslStarted, "--completed-at", $sslCompleted,
            "--expected-commit", $baselineCommit
        )
        if ($Smoke) { $registerArgs += "--smoke" }
        Invoke-Python @registerArgs
        if ($LASTEXITCODE -ne 0) { throw "SSL nav ${navMode} seed $seed registry append failed" }

        foreach ($downstreamSeed in $downstreamSeeds) {
            $downstreamOut = "$outRoot/recovery_downstream_ssl${seed}_nav${navMode}_seed${downstreamSeed}_${suffix}"
            $downstreamStarted = Get-Date -Format o
            $downstreamCommand = "$pythonCmdStr -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $downstreamOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --navigation-mode $navMode --pretrained-encoder $sslOut/jepa_encoder_best.pt --train-splits train --validation-splits validation --evaluation-splits train validation"

            if (-not $Smoke -and (Test-Path "$downstreamOut/tiny_cnn_best.pt")) {
                Write-Output "Downstream full training ($navMode) for pretrain seed $seed and downstream seed $downstreamSeed already exists. Skipping."
                $downstreamCompleted = Get-Date -Format o
            } else {
                Invoke-Python -m e_jepa_ttc train tiny-cnn `
                    --cache $cache `
                    --output-dir $downstreamOut `
                    --epochs $downstreamEpochs `
                    --batch-size 32 `
                    --learning-rate 0.00003 `
                    --seed $downstreamSeed `
                    --device auto `
                    --model event-tubelet-transformer `
                    --navigation-mode $navMode `
                    --pretrained-encoder "$sslOut/jepa_encoder_best.pt" `
                    --train-splits train `
                    --validation-splits validation `
                    --evaluation-splits train validation
                if ($LASTEXITCODE -ne 0) { throw "Downstream SSL $seed nav ${navMode} seed $downstreamSeed failed" }
                $downstreamCompleted = Get-Date -Format o
            }
            $registerArgs = @(
                "scripts/register_recovery_run.py", "--run-id", "recovery-downstream-ssl${seed}-nav${navMode}-seed${downstreamSeed}-${suffix}",
                "--stage", "downstream_ttc", "--run-dir", $downstreamOut,
                "--pretrain-seed", $seed, "--downstream-seed", $downstreamSeed,
                "--pretrained-checkpoint", "$sslOut/jepa_encoder_best.pt",
                "--requested-backbone", "event-tubelet-transformer", "--command", $downstreamCommand,
                "--started-at", $downstreamStarted, "--completed-at", $downstreamCompleted,
                "--expected-commit", $baselineCommit
            )
            if ($Smoke) { $registerArgs += "--smoke" }
            Invoke-Python @registerArgs
            if ($LASTEXITCODE -ne 0) { throw "Downstream SSL $seed nav ${navMode} seed $downstreamSeed registry append failed" }

            # Low-label finetuning
            foreach ($frac in $fractions) {
                $fracStr = "frac$([math]::Round($frac * 100))"
                $manifestPath = "artifacts/subsets/evttc_${fracStr}_seed${downstreamSeed}.json"
                $lowLabelOut = "$outRoot/recovery_downstream_ssl${seed}_nav${navMode}_seed${downstreamSeed}_${fracStr}_${suffix}"
                $lowLabelStarted = Get-Date -Format o
                $lowLabelCommand = "$pythonCmdStr -m e_jepa_ttc train tiny-cnn --cache $cache --output-dir $lowLabelOut --epochs $downstreamEpochs --batch-size 32 --learning-rate 0.00003 --seed $downstreamSeed --device auto --model event-tubelet-transformer --navigation-mode $navMode --pretrained-encoder $sslOut/jepa_encoder_best.pt --train-splits train --validation-splits validation --evaluation-splits train validation --train-fraction $frac --subset-manifest-path $manifestPath"

                if (-not $Smoke -and (Test-Path "$lowLabelOut/tiny_cnn_best.pt")) {
                    Write-Output "Downstream low-label ($navMode, $fracStr) for pretrain seed $seed and downstream seed $downstreamSeed already exists. Skipping."
                    $lowLabelCompleted = Get-Date -Format o
                } else {
                    Invoke-Python -m e_jepa_ttc train tiny-cnn `
                        --cache $cache `
                        --output-dir $lowLabelOut `
                        --epochs $downstreamEpochs `
                        --batch-size 32 `
                        --learning-rate 0.00003 `
                        --seed $downstreamSeed `
                        --device auto `
                        --model event-tubelet-transformer `
                        --navigation-mode $navMode `
                        --pretrained-encoder "$sslOut/jepa_encoder_best.pt" `
                        --train-splits train `
                        --validation-splits validation `
                        --evaluation-splits train validation `
                        --train-fraction $frac `
                        --subset-manifest-path $manifestPath
                    if ($LASTEXITCODE -ne 0) { throw "Low-label SSL $seed nav ${navMode} seed $downstreamSeed frac $frac failed" }
                    $lowLabelCompleted = Get-Date -Format o
                }
                $registerArgs = @(
                    "scripts/register_recovery_run.py", "--run-id", "recovery-lowlabel-ssl${seed}-nav${navMode}-seed${downstreamSeed}-${fracStr}-${suffix}",
                    "--stage", "downstream_ttc", "--run-dir", $lowLabelOut,
                    "--pretrain-seed", $seed, "--downstream-seed", $downstreamSeed,
                    "--pretrained-checkpoint", "$sslOut/jepa_encoder_best.pt",
                    "--requested-backbone", "event-tubelet-transformer", "--command", $lowLabelCommand,
                    "--started-at", $lowLabelStarted, "--completed-at", $lowLabelCompleted,
                    "--expected-commit", $baselineCommit
                )
                if ($Smoke) { $registerArgs += "--smoke" }
                Invoke-Python @registerArgs
                if ($LASTEXITCODE -ne 0) { throw "Low-label SSL $seed nav ${navMode} seed $downstreamSeed frac $frac registry append failed" }
            }
        }
    }
}

Write-Output "All recovery runs and registrations complete."

Invoke-Python scripts/validate_artifact_registry.py
if ($LASTEXITCODE -ne 0) { throw "Final registry validation failed" }
Write-Output "Completed recovery runs matrix."
