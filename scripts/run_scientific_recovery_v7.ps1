param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$SkipBaselines,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigRoot = Join-Path $Root "configs\experiment\scientific_recovery_v7_fold_chain"
$RunRoot = Join-Path $Root "artifacts\runs"
$ResultRoot = Join-Path $Root "artifacts\scientific_recovery_v7"
$env:PYTHONPATH = "src;.."

function Assert-SignedJson {
    param([Parameter(Mandatory = $true)][string]$Path)
    uv run --no-sync python -c "import json,sys; from e_jepa_ttc.artifacts.hashing import verify_artifact_hash; p=json.load(open(sys.argv[1],encoding='utf-8')); assert verify_artifact_hash(p), 'invalid artifact signature'" $Path
    if ($LASTEXITCODE -ne 0) {
        throw "Artifact signature validation failed: $Path"
    }
}

function Assert-ConfigHash {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][object]$Manifest
    )
    $Key = [System.IO.Path]::GetFileNameWithoutExtension($Config)
    $Expected = $Manifest.configs.$Key.sha256
    if (-not $Expected) {
        throw "Frozen manifest has no record for $Key"
    }
    $Observed = (Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected) {
        throw "Config hash mismatch for $Config"
    }
}

Set-Location $Root
$Protocol = Join-Path $Root "configs\protocol\scientific_recovery_v7_balanced_oof.json"
$ManifestPath = Join-Path $ConfigRoot "frozen_manifest.json"
Assert-SignedJson -Path $Protocol
Assert-SignedJson -Path $ManifestPath
$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json

if (-not $SkipBaselines) {
    uv run --no-sync python scripts/reevaluate_v7_baselines.py --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "V7 baseline re-evaluation failed"
    }
    Assert-SignedJson -Path (Join-Path $ResultRoot "baselines\manifest.json")
}

if (-not $SkipTraining) {
    foreach ($Arm in @("soft", "c2f", "t20", "cap_s")) {
        foreach ($Fold in 0..2) {
            $Config = Join-Path $ConfigRoot "v7_${Arm}_fold${Fold}_seed7.yaml"
            Assert-ConfigHash -Config $Config -Manifest $Manifest
            $RunName = "scientific_recovery_v7_${Arm}_fold${Fold}_seed7"
            $RunDir = Join-Path $RunRoot $RunName
            $Summary = Join-Path $RunDir "summary.json"
            if (Test-Path -LiteralPath $Summary) {
                Assert-SignedJson -Path $Summary
                continue
            }
            uv run --no-sync python scripts/train_causal_scale_eap_screen.py `
                --config $Config `
                --output-dir $RunDir `
                --device $Device `
                --resume
            if ($LASTEXITCODE -ne 0) {
                throw "V7 run failed or became corrupt: $RunName"
            }
            Assert-SignedJson -Path $Summary
        }
    }
}

Write-Output "V7 fold chain completed without opening validation, test, or CodaBench."
