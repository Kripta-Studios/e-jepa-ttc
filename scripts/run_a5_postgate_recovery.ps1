param(
    [string]$Device = "cuda:0",
    [switch]$Force,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Protocol = "configs/experiment/e_jepa_garl_event_causal_scale_a5_postgate_recovery_v1.yaml"
$Frozen = "artifacts/configs/a5_postgate_recovery_v1"
$Audit = "artifacts/audit/a5_postgate_recovery_v1"
$LogDir = "artifacts/logs/a5_postgate_recovery_v1"
$BundleDir = "artifacts/audit/a5_postgate_recovery_results_v1"
$BundleZip = "artifacts/audit/a5_postgate_recovery_results_v1.zip"
$script:Status = "STARTED"
$script:Steps = @()

function Reset-Path([string]$Path) {
    if (Test-Path $Path) {
        if (-not $Force) { throw "Path already exists: $Path. Re-run with -Force to replace recovery outputs." }
        Remove-Item $Path -Recurse -Force
    }
}
function Invoke-Python([string]$Name,[string[]]$Args,[int[]]$Allowed=@(0)) {
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    $log = Join-Path $LogDir "$Name.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host ("python " + ($Args -join " "))
    $started=Get-Date
    & python @Args 2>&1 | Tee-Object -FilePath $log
    $code=$LASTEXITCODE
    if (-not (Test-Path $log)) { New-Item -ItemType File -Force $log | Out-Null }
    $script:Steps += [PSCustomObject]@{name=$Name;exit_code=$code;elapsed_seconds=((Get-Date)-$started).TotalSeconds;log=$log}
    if ($Allowed -notcontains $code) { throw "Step $Name failed with exit code $code" }
    return $code
}
function Copy-IfExists([string]$Source,[string]$DestRoot) {
    if (Test-Path $Source) {
        $dest=Join-Path $DestRoot $Source
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $Source $dest -Force
    }
}
function Finalize-Bundle {
    try {
        Remove-Item $BundleDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $BundleZip -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force $BundleDir | Out-Null
        foreach ($p in @($Protocol,$Frozen,$Audit,$LogDir,
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/validation_predictions.csv",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13/validation_predictions.csv",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23/validation_predictions.csv",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed7/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed7/validation_predictions.csv",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed13/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed13/validation_predictions.csv",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed23/summary.json",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed23/validation_predictions.csv",
          "artifacts/metrics/a5_transport_preflight_v3_confirm/a5_transport_preflight_v3_confirm.json")) {
            if (Test-Path $p) {
                if ((Get-Item $p) -is [System.IO.DirectoryInfo]) {
                    $dest = Join-Path $BundleDir $p
                    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
                    Copy-Item $p $dest -Recurse -Force
                } else { Copy-IfExists $p $BundleDir }
            }
        }
        $prov=Join-Path $BundleDir "provenance"; New-Item -ItemType Directory -Force $prov | Out-Null
        git rev-parse HEAD | Out-File (Join-Path $prov "git_head.txt") -Encoding utf8
        git status --short | Out-File (Join-Path $prov "git_status.txt") -Encoding utf8
        git log -25 --oneline --decorate | Out-File (Join-Path $prov "git_log_25.txt") -Encoding utf8
        $manifest=[ordered]@{artifact_type="a5_postgate_recovery_results_bundle_v1";created_at=(Get-Date).ToUniversalTime().ToString("o");status=$script:Status;steps=$script:Steps;private_test_opened=$false}
        $manifest | ConvertTo-Json -Depth 8 | Out-File (Join-Path $BundleDir "suite_manifest.json") -Encoding utf8
        $checkpointRows=@()
        foreach($pt in @(
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/model_best.pt",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13/model_best.pt",
          "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23/model_best.pt",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed7/model_best.pt",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed13/model_best.pt",
          "artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed23/model_best.pt")) {
            if(Test-Path $pt){$i=Get-Item $pt;$checkpointRows += [PSCustomObject]@{path=$pt;bytes=$i.Length;sha256=(Get-FileHash $pt -Algorithm SHA256).Hash.ToLower()}}
        }
        $checkpointRows | Export-Csv (Join-Path $BundleDir "checkpoint_inventory.csv") -NoTypeInformation -Encoding utf8
        $inventoryPath=Join-Path $BundleDir "inventory.csv"
        $rows=Get-ChildItem $BundleDir -Recurse -File | Where-Object {$_.FullName -ne $inventoryPath} | ForEach-Object {[PSCustomObject]@{path=$_.FullName.Substring((Resolve-Path $BundleDir).Path.Length+1);bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()}}
        $rows | Export-Csv $inventoryPath -NoTypeInformation -Encoding utf8
        Compress-Archive -Force -Path "$BundleDir\*" -DestinationPath $BundleZip
        Write-Host "`nA5 post-gate recovery bundle: $(Resolve-Path $BundleZip)" -ForegroundColor Green
        Write-Host "SHA256: $((Get-FileHash $BundleZip -Algorithm SHA256).Hash.ToLower())"
    } catch { Write-Warning "Could not finalize recovery bundle: $_" }
}

try {
    Reset-Path $Audit; Reset-Path $LogDir; New-Item -ItemType Directory -Force $Audit | Out-Null
    if (-not $SkipTests) {
        Invoke-Python "00_py_compile" @("-m","py_compile","src/e_jepa_ttc/training/causal_scale_eap.py","scripts/train_causal_scale_eap_screen.py","scripts/freeze_a5_postgate_recovery_configs.py","scripts/summarize_a5_signal_replication.py","scripts/summarize_a5_anchor_replication.py") | Out-Null
        Invoke-Python "01_pytest" @("-m","pytest","-q","tests/unit/test_a5_anchor_initialization.py","tests/unit/test_a5_train_preflight_contract.py","tests/unit/test_a5_local_transport.py","tests/unit/test_causal_scale_ttc.py") | Out-Null
    }
    Invoke-Python "02_refreeze_v3_runtime" @("scripts/freeze_a5_suite_configs.py","--output-dir","artifacts/configs/a5_transport_suite_v3","--preflight-selection","artifacts/metrics/a5_transport_preflight_v3_confirm/a5_transport_preflight_v3_confirm.json") | Out-Null
    Reset-Path $Frozen
    Invoke-Python "03_freeze_recovery_configs" @("scripts/freeze_a5_postgate_recovery_configs.py","--base-config-dir","artifacts/configs/a5_transport_suite_v3","--output-dir",$Frozen) | Out-Null

    foreach($seed in @(13,23)){
        $run="artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed$seed"; Reset-Path $run
        Invoke-Python "10_train_diagnostic_seed$seed" @("scripts/train_causal_scale_eap_screen.py","--config","$Frozen/diagnostic_seed$seed.yaml","--output-dir",$run,"--device",$Device) | Out-Null
    }
    $rep=Invoke-Python "12_diagnostic_replication" @("scripts/summarize_a5_signal_replication.py","--protocol",$Protocol,"--summary","artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/summary.json","--summary","artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13/summary.json","--summary","artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23/summary.json","--output","$Audit/diagnostic_replication.json") @(0,6)
    if($rep -ne 0){$script:Status="STOPPED_DIAGNOSTIC_REPLICATION_NOT_CONFIRMED"; throw "STOP"}

    $anchor7="artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed7"; Reset-Path $anchor7
    Invoke-Python "20_train_anchor_seed7" @("scripts/train_causal_scale_eap_screen.py","--config","$Frozen/anchor_seed7.yaml","--output-dir",$anchor7,"--device",$Device) | Out-Null
    $g7=Invoke-Python "21_gate_anchor_seed7" @("scripts/summarize_a5_anchor_replication.py","--protocol",$Protocol,"--summary","$anchor7/summary.json","--required-passes","1","--output","$Audit/anchor_seed7_gate.json") @(0,7)
    if($g7 -ne 0){$script:Status="STOPPED_ANCHOR_SEED7_GATE"; throw "STOP"}

    foreach($seed in @(13,23)){
        $run="artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed$seed"; Reset-Path $run
        Invoke-Python "30_train_anchor_seed$seed" @("scripts/train_causal_scale_eap_screen.py","--config","$Frozen/anchor_seed$seed.yaml","--output-dir",$run,"--device",$Device) | Out-Null
    }
    $areg=Invoke-Python "32_anchor_replication" @("scripts/summarize_a5_anchor_replication.py","--protocol",$Protocol,"--summary","artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed7/summary.json","--summary","artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed13/summary.json","--summary","artifacts/runs/causal_scale_eap_screen_a5_anchor_v1_seed23/summary.json","--output","$Audit/anchor_replication.json") @(0,7)
    if($areg -ne 0){$script:Status="STOPPED_ANCHOR_REPLICATION_GATE"; throw "STOP"}
    $script:Status="COMPLETE_ANCHOR_REPLICATION_PASS_ANALYZE_BEFORE_CAPACITY"
} catch {
    if($script:Status -eq "STARTED"){$script:Status="FAILED: $($_.Exception.Message)"; Write-Error $_}
    elseif($_.Exception.Message -ne "STOP"){Write-Warning $_}
} finally { Finalize-Bundle }
if($script:Status -like "FAILED*"){exit 1}
if($script:Status -like "STOPPED*"){exit 2}
exit 0
