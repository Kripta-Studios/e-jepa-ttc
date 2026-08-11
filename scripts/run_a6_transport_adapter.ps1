param(
    [string]$Device = "cuda:0",
    [switch]$Force,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$AuditDir = "artifacts\audit\a6_transport_adapter_v1"
$LogDir = "artifacts\logs\a6_transport_adapter_v1"
$ConfigDir = "artifacts\configs\a6_transport_adapter_v1"
$BundleStage = "artifacts\audit\a6_transport_adapter_results_v1"
$BundleZip = "artifacts\audit\a6_transport_adapter_results_v1.zip"
$Status = "RUNNING"

if ($Force) {
    Remove-Item $AuditDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $LogDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $ConfigDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $BundleStage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $BundleZip -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $AuditDir,$LogDir,$ConfigDir | Out-Null

function Invoke-Python {
    param([string]$Name, [string[]]$PythonArgs)
    Write-Host "`n=== $Name ==="
    Write-Host ("python " + ($PythonArgs -join " "))
    $log = Join-Path $LogDir ("$Name.log")
    New-Item -ItemType File -Force $log | Out-Null
    & python @PythonArgs 2>&1 | Tee-Object -FilePath $log | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    return [int]$code
}

function Copy-IfExists {
    param([string]$Source,[string]$DestinationRoot)
    if (Test-Path $Source) {
        $dest = Join-Path $DestinationRoot $Source
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $Source $dest -Force
    }
}

function Finalize-Bundle {
    Remove-Item $BundleStage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $BundleStage | Out-Null
    @(
        "configs\experiment\e_jepa_garl_event_causal_scale_a6_transport_adapter_v1.yaml",
        $AuditDir,
        $LogDir,
        $ConfigDir,
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed7\summary.json",
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed7\validation_predictions.csv",
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed13\summary.json",
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed13\validation_predictions.csv",
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed23\summary.json",
        "artifacts\runs\causal_scale_eap_screen_a6_transport_adapter_v1_seed23\validation_predictions.csv"
    ) | ForEach-Object { Copy-IfExists $_ $BundleStage }
    New-Item -ItemType Directory -Force (Join-Path $BundleStage "provenance") | Out-Null
    git rev-parse HEAD | Out-File (Join-Path $BundleStage "provenance\git_head.txt") -Encoding utf8
    git status --short | Out-File (Join-Path $BundleStage "provenance\git_status.txt") -Encoding utf8
    git log -25 --oneline --decorate | Out-File (Join-Path $BundleStage "provenance\git_log_25.txt") -Encoding utf8
    $manifest = [ordered]@{ artifact_type="a6_transport_adapter_suite_bundle_v1"; status=$Status; private_test_opened=$false }
    $manifest | ConvertTo-Json -Depth 5 | Out-File (Join-Path $BundleStage "suite_manifest.json") -Encoding utf8
    $inventory = Get-ChildItem $BundleStage -Recurse -File | Where-Object { $_.Name -ne "inventory.csv" } | ForEach-Object {
        [PSCustomObject]@{ Path=$_.FullName.Substring((Resolve-Path $BundleStage).Path.Length + 1); Bytes=$_.Length; SHA256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() }
    }
    $inventory | Export-Csv (Join-Path $BundleStage "inventory.csv") -NoTypeInformation -Encoding utf8
    Remove-Item $BundleZip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Force -Path "$BundleStage\*" -DestinationPath $BundleZip
    Write-Host "`nA6 transport-adapter bundle: $((Resolve-Path $BundleZip).Path)"
    Write-Host "SHA256: $((Get-FileHash $BundleZip -Algorithm SHA256).Hash.ToLower())"
}

try {
    $compile = @(
        "-m","py_compile",
        "src/e_jepa_ttc/models/causal_scale_ttc.py",
        "scripts/train_causal_scale_eap_screen.py",
        "scripts/freeze_a6_transport_adapter_configs.py",
        "scripts/summarize_a6_transport_adapter.py"
    )
    if ((Invoke-Python "00_py_compile" $compile) -ne 0) { throw "py_compile failed" }

    if (-not $SkipTests) {
        $tests = @("-m","pytest","-q","tests/unit/test_a6_transport_adapter.py","tests/unit/test_a5_anchor_initialization.py","tests/unit/test_a5_train_preflight_contract.py","tests/unit/test_a5_local_transport.py","tests/unit/test_causal_scale_ttc.py")
        if ((Invoke-Python "01_pytest" $tests) -ne 0) { throw "A6/A5 unit tests failed" }
    }

    $freeze = @("scripts/freeze_a6_transport_adapter_configs.py","--output-dir",$ConfigDir)
    if ((Invoke-Python "02_freeze_configs" $freeze) -ne 0) { throw "A6 config freeze failed" }

    $run7 = "artifacts/runs/causal_scale_eap_screen_a6_transport_adapter_v1_seed7"
    if ((Invoke-Python "10_train_seed7" @("scripts/train_causal_scale_eap_screen.py","--config","$ConfigDir/seed7.yaml","--output-dir",$run7,"--device",$Device)) -ne 0) { throw "A6 seed7 training failed" }
    $gate7 = "$AuditDir/seed7_gate.json"
    $gate7Code = Invoke-Python "11_gate_seed7" @("scripts/summarize_a6_transport_adapter.py","--summary","$run7/summary.json","--required-passes","1","--output",$gate7)
    if ($gate7Code -ne 0) {
        $Status = "STOPPED_A6_SEED7_GATE"
        Write-Warning $Status
        Finalize-Bundle
        exit 0
    }

    foreach ($seed in @(13,23)) {
        $run = "artifacts/runs/causal_scale_eap_screen_a6_transport_adapter_v1_seed$seed"
        if ((Invoke-Python "20_train_seed$seed" @("scripts/train_causal_scale_eap_screen.py","--config","$ConfigDir/seed$seed.yaml","--output-dir",$run,"--device",$Device)) -ne 0) { throw "A6 seed$seed training failed" }
    }
    $repArgs = @(
        "scripts/summarize_a6_transport_adapter.py",
        "--summary","artifacts/runs/causal_scale_eap_screen_a6_transport_adapter_v1_seed7/summary.json",
        "--summary","artifacts/runs/causal_scale_eap_screen_a6_transport_adapter_v1_seed13/summary.json",
        "--summary","artifacts/runs/causal_scale_eap_screen_a6_transport_adapter_v1_seed23/summary.json",
        "--required-passes","2",
        "--output","$AuditDir/replication_gate.json"
    )
    $repCode = Invoke-Python "22_replication_gate" $repArgs
    $Status = if ($repCode -eq 0) { "COMPLETE_A6_REPLICATION_PASS" } else { "COMPLETE_A6_REPLICATION_FAIL" }
    Finalize-Bundle
    exit 0
}
catch {
    $Status = "FAILED: $($_.Exception.Message)"
    try { Finalize-Bundle } catch { Write-Warning "Could not finalize A6 bundle: $_" }
    Write-Error $_
    exit 1
}
