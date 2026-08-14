param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateRange(1, 2)]
    [int]$MaximumParallel = 2,
    [ValidateRange(60, 3600)]
    [int]$PollSeconds = 900
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigRoot = Join-Path $Root "configs\experiment\scientific_recovery_v7_fold_chain"
$RunRoot = Join-Path $Root "artifacts\runs"
$ResultRoot = Join-Path $Root "artifacts\scientific_recovery_v7"
$RunnerRoot = Join-Path $ResultRoot "runner\soft_partial_freeze"
$ManifestPath = Join-Path $ConfigRoot "soft_partial_freeze_manifest.json"
$Protocol = Join-Path $Root "configs\protocol\scientific_recovery_v7_balanced_oof.json"
$Baseline = Join-Path $ResultRoot "baselines\a5_oof_predictions.csv"
$StatusPath = Join-Path $ResultRoot "SOFT_PARTIAL_FREEZE_STATUS.md"
$EventLog = Join-Path $RunnerRoot "supervisor.events.log"
$env:PYTHONPATH = "src;.."
New-Item -ItemType Directory -Force -Path $RunnerRoot | Out-Null
Set-Location $Root

function Test-SignedJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    uv run --no-sync python -c "import json,sys; from e_jepa_ttc.artifacts.hashing import verify_artifact_hash; p=json.load(open(sys.argv[1],encoding='utf-8')); raise SystemExit(0 if verify_artifact_hash(p) else 1)" $Path
    return $LASTEXITCODE -eq 0
}

if (-not (Test-SignedJson $ManifestPath)) {
    throw "Partial-freeze manifest is absent or invalid"
}
if (-not (Test-SignedJson $Protocol)) {
    throw "V7 protocol is absent or invalid"
}
$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$Queue = @(0..2 | ForEach-Object {
    [pscustomobject]@{
        Fold = $_
        Name = "scientific_recovery_v7_soft_partial_freeze_fold${_}_seed7"
    }
})
$Active = @{}

function Write-Event([string]$Message) {
    Add-Content -LiteralPath $EventLog -Value "$(Get-Date -Format o) $Message" -Encoding utf8
}

function Get-Config([object]$Item) {
    $path = Join-Path $ConfigRoot "v7_soft_partial_freeze_fold$($Item.Fold)_seed7.yaml"
    $key = [System.IO.Path]::GetFileNameWithoutExtension($path)
    $expected = $Manifest.configs.$key.sha256
    $observed = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $expected -or $observed -ne $expected) {
        throw "Frozen config hash mismatch: $key"
    }
    return $path
}

function Start-ControlRun([object]$Item) {
    $config = Get-Config $Item
    $runDir = Join-Path $RunRoot $Item.Name
    $arguments = "run --no-sync python scripts/train_causal_scale_eap_screen.py --config `"$config`" --output-dir `"$runDir`" --device $Device"
    $last = Join-Path $runDir "state\last.pt"
    if (Test-Path -LiteralPath $last) { $arguments += " --resume" }
    $stdout = Join-Path $RunnerRoot "$($Item.Name).stdout.log"
    $stderr = Join-Path $RunnerRoot "$($Item.Name).stderr.log"
    $process = Start-Process -FilePath (Get-Command uv).Source -ArgumentList $arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $Active[$Item.Name] = [pscustomobject]@{ Item = $Item; Pid = $process.Id }
    Write-Event "START $($Item.Name) pid=$($process.Id) resume=$(Test-Path -LiteralPath $last)"
}

function Write-ControlStatus {
    $lines = @(
        "# V7 SOFT partial-freeze: estado operativo",
        "",
        "Generado: $(Get-Date -Format o). Configuración: ``encoder.features[0:3]`` congelado; pérdidas SOFT sin cambios.",
        "",
        "| Fold | Estado | Epoch | Best | MiD provisional |",
        "|---:|---|---:|---:|---:|"
    )
    foreach ($item in $Queue) {
        $runDir = Join-Path $RunRoot $item.Name
        $summary = Join-Path $runDir "summary.json"
        $progress = Join-Path $runDir "state\progress.json"
        if (Test-SignedJson $summary) {
            $json = Get-Content -Raw -LiteralPath $summary | ConvertFrom-Json
            $lines += "| $($item.Fold) | firmado | $($json.history[-1].epoch) | $($json.selection.best_epoch) | $([math]::Round($json.dev_metrics.sequence_macro.sequence_macro_paper_MiD_overall,3)) |"
        } elseif (Test-Path -LiteralPath $progress) {
            $json = Get-Content -Raw -LiteralPath $progress | ConvertFrom-Json
            $mid = if ($json.latest_selection) { [math]::Round($json.latest_selection.sequence_macro_MiD,3) } else { "-" }
            $lines += "| $($item.Fold) | $($json.status) | $($json.epoch) | $($json.best_epoch) | $mid |"
        } elseif ($Active.ContainsKey($item.Name)) {
            $lines += "| $($item.Fold) | cargando | 0 | 0 | - |"
        } else {
            $lines += "| $($item.Fold) | en cola | 0 | 0 | - |"
        }
    }
    Set-Content -LiteralPath $StatusPath -Value $lines -Encoding utf8
}

while ($true) {
    foreach ($name in @($Active.Keys)) {
        $entry = $Active[$name]
        $summary = Join-Path $RunRoot "$name\summary.json"
        if (Test-SignedJson $summary) {
            Write-Event "COMPLETE $name artifact_signed=true"
            $Active.Remove($name)
        } elseif (-not (Get-Process -Id $entry.Pid -ErrorAction SilentlyContinue)) {
            Write-Event "HALT $name process_ended_without_signed_summary"
            Write-ControlStatus
            exit 2
        }
    }
    while ($Active.Count -lt $MaximumParallel) {
        $next = $Queue | Where-Object {
            -not $Active.ContainsKey($_.Name) -and
            -not (Test-SignedJson (Join-Path $RunRoot "$($_.Name)\summary.json"))
        } | Select-Object -First 1
        if ($null -eq $next) { break }
        Start-ControlRun $next
    }
    Write-ControlStatus
    $complete = @($Queue | Where-Object {
        Test-SignedJson (Join-Path $RunRoot "$($_.Name)\summary.json")
    }).Count
    if ($complete -eq 3 -and $Active.Count -eq 0) { break }
    Start-Sleep -Seconds $PollSeconds
}

$Summaries = @(0..2 | ForEach-Object { Join-Path $RunRoot "scientific_recovery_v7_soft_partial_freeze_fold${_}_seed7\summary.json" })
$References = @(0..2 | ForEach-Object { Join-Path $RunRoot "scientific_recovery_v5_a4_parent_grouped_fold${_}_seed7\summary.json" })
$Predictions = @(0..2 | ForEach-Object { Join-Path $RunRoot "scientific_recovery_v7_soft_partial_freeze_fold${_}_seed7\dev_predictions.csv" })
$Audit = Join-Path $ResultRoot "audit\soft_partial_freeze_geometry.json"
$Aggregate = Join-Path $ResultRoot "results\soft_partial_freeze_seed7_oof.json"
uv run --no-sync python scripts/audit_v7_fold_geometry.py --candidate-summaries $Summaries --reference-summaries $References --output $Audit
if ($LASTEXITCODE -ne 0) { throw "Partial-freeze geometry audit failed" }
uv run --no-sync python scripts/aggregate_v7_fold_results.py --arm soft_partial_freeze --predictions $Predictions --a5-baseline $Baseline --protocol $Protocol --geometry-audit $Audit --output $Aggregate
if ($LASTEXITCODE -ne 0 -or -not (Test-SignedJson $Aggregate)) {
    throw "Partial-freeze aggregation failed"
}
Write-Event "DONE runs=3 aggregate=$Aggregate"
Write-ControlStatus
