param(
    [string]$Device = "cuda:0",
    [string]$GarlRepo = "E:\Garl-TTC",
    [string]$GarlDataset = "E:\GarlTTC_dataset",
    [string]$EapDataset = "E:\eAP_dataset",
    [int]$NumWorkers = 10,
    [int]$PrefetchFactor = 3,
    [int]$BootstrapResamples = 5000,
    [int]$HardwarePollSeconds = 5,
    [int]$MinFreeVramForParallelMiB = 6000,
    [switch]$Force,
    [switch]$SkipTests,
    [switch]$DisableParallelReplications,
    [switch]$SkipBudgetMatchedGarl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-NativeExitCode {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$false)][string[]]$Arguments = @(),
        [switch]$Quiet
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = [bool]$Quiet
    $psi.RedirectStandardError = [bool]$Quiet
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add([string]$arg) }
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    if ($Quiet) {
        $null = $proc.StandardOutput.ReadToEnd()
        $null = $proc.StandardError.ReadToEnd()
    }
    $proc.WaitForExit()
    return [int]$proc.ExitCode
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$ExpectedBase = "954fa52"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Master = Join-Path $Root "artifacts\scientific_recovery_master_v1"
$Logs = Join-Path $Master "logs"
$Audit = Join-Path $Master "audit"
$Configs = Join-Path $Master "configs"
$StatusPath = Join-Path $Master "master_status.json"
$HardwareCsv = Join-Path $Master "hardware\hardware_timeseries.csv"
$HardwareStop = Join-Path $Master "hardware\STOP"
$OutputZip = Join-Path $Root "artifacts\audit\E_JEPA_SCIENTIFIC_RECOVERY_MASTER_RESULTS_$Timestamp.zip"

$A4Base2k = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_dinov3_relational_rgb_v2_seed7\summary.json"
$A4L8RunBase = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7"
$A4L8Summary = Join-Path $A4L8RunBase "summary.json"
$A4L8Checkpoint = Join-Path $A4L8RunBase "model_best.pt"
$TrainManifest = Join-Path $Root "artifacts\cache\garl_object_event_common_roi_train8192_v1\manifest.json"
$ValManifest = Join-Path $Root "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$TeacherManifest = Join-Path $Root "artifacts\cache\dinov3_convnext_large_relational_a4_train8192_rgb_v1\manifest.json"
$PreflightV2 = Join-Path $Root "artifacts\metrics\a5_transport_preflight_v2\a5_transport_preflight_v2.json"
$PreflightV3 = Join-Path $Root "artifacts\metrics\a5_transport_preflight_v3_confirm\a5_transport_preflight_v3_confirm.json"

$script:Rows = New-Object System.Collections.Generic.List[object]
$script:HardwareProcess = $null
$script:LegacyWinner = $null
$script:CausalWinner = $null
$script:FinalCandidateRun = $null
$script:FinalCandidateMode = "legacy"
$script:ClaimsBlocked = $false
$script:TransportBlocked = $false
$script:CausalBlocked = $false

function Write-Status {
    $payload = [ordered]@{
        artifact_type = "scientific_recovery_master_status_v1"
        updated_at = (Get-Date).ToString("o")
        git_head = ((& git rev-parse HEAD) | Select-Object -First 1).Trim()
        git_dirty = [bool]((& git status --porcelain | Measure-Object).Count)
        legacy_winner = $script:LegacyWinner
        causal_winner = $script:CausalWinner
        final_candidate_run = $script:FinalCandidateRun
        final_candidate_mode = $script:FinalCandidateMode
        claims_blocked = $script:ClaimsBlocked
        transport_blocked = $script:TransportBlocked
        causal_blocked = $script:CausalBlocked
        private_test_opened = $false
        steps = @($script:Rows)
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StatusPath) | Out-Null
    $payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Add-StepStatus {
    param([string]$Name,[string]$Category,[string]$Status,[int]$ExitCode,[string]$Message,[string]$LogDir)
    $script:Rows.Add([ordered]@{
        name=$Name; category=$Category; status=$Status; exit_code=$ExitCode; message=$Message;
        log_dir=$LogDir; at=(Get-Date).ToString("o")
    })
    Write-Status
}

function Quote-DisplayArg {
    param([string]$Value)
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"','\"') + '"' }
    return $Value
}

function Invoke-LoggedProcess {
    param(
        [string]$Name,
        [string]$Category,
        [string]$FilePath,
        [string[]]$Arguments,
        [switch]$AllowFailure
    )
    $stepDir = Join-Path $Logs $Name
    New-Item -ItemType Directory -Force -Path $stepDir | Out-Null
    $stdout = Join-Path $stepDir "stdout.log"
    $stderr = Join-Path $stepDir "stderr.log"
    $command = ((Quote-DisplayArg $FilePath) + " " + (($Arguments | ForEach-Object { Quote-DisplayArg $_ }) -join " ")).Trim()
    $command | Set-Content -LiteralPath (Join-Path $stepDir "command.txt") -Encoding utf8
    (Get-Date).ToString("o") | Set-Content -LiteralPath (Join-Path $stepDir "started_at.txt") -Encoding utf8
    (& git rev-parse HEAD) | Set-Content -LiteralPath (Join-Path $stepDir "git_head.txt") -Encoding utf8
    (& git status --short) | Set-Content -LiteralPath (Join-Path $stepDir "git_status.txt") -Encoding utf8
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host $command
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add([string]$arg) }
    $proc = [System.Diagnostics.Process]::new(); $proc.StartInfo=$psi
    [void]$proc.Start()
    $outTask=$proc.StandardOutput.ReadToEndAsync(); $errTask=$proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit(); $out=$outTask.Result; $err=$errTask.Result
    $out | Set-Content -LiteralPath $stdout -Encoding utf8
    $err | Set-Content -LiteralPath $stderr -Encoding utf8
    if ($out) { Write-Host $out.TrimEnd() }
    if ($err) { Write-Host $err.TrimEnd() -ForegroundColor DarkYellow }
    $code=[int]$proc.ExitCode
    $code | Set-Content -LiteralPath (Join-Path $stepDir "exit_code.txt") -Encoding ascii
    (Get-Date).ToString("o") | Set-Content -LiteralPath (Join-Path $stepDir "finished_at.txt") -Encoding utf8
    if ($code -eq 0) { Add-StepStatus $Name $Category "PASS" $code "completed" $stepDir }
    else { Add-StepStatus $Name $Category "FAIL" $code "process exit $code" $stepDir; if (-not $AllowFailure) { throw "$Name failed (exit $code)" } }
    return $code
}

function Invoke-Python {
    param([string]$Name,[string]$Category,[string[]]$Arguments,[switch]$AllowFailure)
    return Invoke-LoggedProcess -Name $Name -Category $Category -FilePath ((Get-Command python).Source) -Arguments $Arguments -AllowFailure:$AllowFailure
}

function Test-JsonDecision {
    param([string]$Path,[string]$Expected)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try { return ((Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json).decision -eq $Expected) } catch { return $false }
}
function Test-JsonStatus {
    param([string]$Path,[string]$Expected)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try { return ((Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json).status -eq $Expected) } catch { return $false }
}

function Invoke-Train {
    param([string]$Name,[string]$Config,[string]$RunDir,[switch]$AllowFailure)
    if ((Test-Path (Join-Path $RunDir "summary.json")) -and -not $Force) {
        Add-StepStatus $Name "training" "REUSED" 0 "existing summary reused" $RunDir
        return 0
    }
    if ($Force -and (Test-Path $RunDir)) { Remove-Item -Recurse -Force -LiteralPath $RunDir }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    return Invoke-Python $Name "training" @("scripts/train_causal_scale_eap_screen.py","--config",$Config,"--output-dir",$RunDir,"--device",$Device) -AllowFailure:$AllowFailure
}

function Get-FreeVramMiB {
    try {
        $raw = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($raw) { return [int]([double]$raw.Trim()) }
    } catch {}
    return -1
}

function Invoke-ParallelTrainPair {
    param(
        [string]$Prefix,
        [string]$ConfigA,[string]$RunA,
        [string]$ConfigB,[string]$RunB
    )
    $freeVram = Get-FreeVramMiB
    $parallelAllowed = (-not $DisableParallelReplications) -and ($freeVram -lt 0 -or $freeVram -ge $MinFreeVramForParallelMiB)
    if (-not $parallelAllowed) {
        $reason = if ($DisableParallelReplications) { "disabled by switch" } else { "free VRAM ${freeVram} MiB < ${MinFreeVramForParallelMiB} MiB" }
        Add-StepStatus "${Prefix}_parallel_policy" "hardware" "SEQUENTIAL" 0 $reason $Logs
        $a=Invoke-Train "${Prefix}_seed13" $ConfigA $RunA -AllowFailure
        $b=Invoke-Train "${Prefix}_seed23" $ConfigB $RunB -AllowFailure
        return @($a,$b)
    }
    Add-StepStatus "${Prefix}_parallel_policy" "hardware" "PARALLEL" 0 "free VRAM ${freeVram} MiB; running independent replications concurrently" $Logs
    $jobs=@()
    foreach ($spec in @(@(13,$ConfigA,$RunA),@(23,$ConfigB,$RunB))) {
        $seed=$spec[0]; $cfg=$spec[1]; $run=$spec[2]; $name="${Prefix}_seed$seed"
        if ((Test-Path (Join-Path $run "summary.json")) -and -not $Force) {
            Add-StepStatus $name "training" "REUSED" 0 "existing summary reused" $run
            continue
        }
        if ($Force -and (Test-Path $run)) { Remove-Item -Recurse -Force -LiteralPath $run }
        New-Item -ItemType Directory -Force -Path $run | Out-Null
        $stepDir=Join-Path $Logs $name; New-Item -ItemType Directory -Force -Path $stepDir | Out-Null
        $args=@("scripts/train_causal_scale_eap_screen.py","--config",$cfg,"--output-dir",$run,"--device",$Device)
        (($args | ForEach-Object { Quote-DisplayArg $_ }) -join " ") | Set-Content (Join-Path $stepDir "command.txt") -Encoding utf8
        (Get-Date).ToString("o") | Set-Content (Join-Path $stepDir "started_at.txt") -Encoding utf8
        $psi=[System.Diagnostics.ProcessStartInfo]::new(); $psi.FileName=(Get-Command python).Source; $psi.WorkingDirectory=$Root; $psi.UseShellExecute=$false; $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
        foreach($arg in $args){[void]$psi.ArgumentList.Add([string]$arg)}
        $p=[System.Diagnostics.Process]::new();$p.StartInfo=$psi;[void]$p.Start()
        $jobs += [pscustomobject]@{Name=$name;Run=$run;StepDir=$stepDir;Proc=$p;Out=$p.StandardOutput.ReadToEndAsync();Err=$p.StandardError.ReadToEndAsync()}
    }
    $codes=@()
    foreach($j in $jobs){
        $j.Proc.WaitForExit(); $out=$j.Out.Result; $err=$j.Err.Result
        $out | Set-Content (Join-Path $j.StepDir "stdout.log") -Encoding utf8; $err | Set-Content (Join-Path $j.StepDir "stderr.log") -Encoding utf8
        $code=[int]$j.Proc.ExitCode; $codes += $code; $code | Set-Content (Join-Path $j.StepDir "exit_code.txt") -Encoding ascii; (Get-Date).ToString("o") | Set-Content (Join-Path $j.StepDir "finished_at.txt") -Encoding utf8
        if($code -eq 0){Add-StepStatus $j.Name "training" "PASS" 0 "parallel replication completed" $j.StepDir}else{Add-StepStatus $j.Name "training" "FAIL" $code "parallel replication failed" $j.StepDir}
    }
    return $codes
}

function Start-HardwareMonitor {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $HardwareCsv) | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue $HardwareStop
    $args=@("-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $Root "scripts\monitor_hardware.ps1"),"-OutputCsv",$HardwareCsv,"-StopFile",$HardwareStop,"-IntervalSeconds",[string]$HardwarePollSeconds)
    try { $script:HardwareProcess=Start-Process -FilePath "pwsh" -ArgumentList $args -PassThru -WindowStyle Hidden } catch { Write-Warning "hardware monitor could not start: $_" }
}
function Stop-HardwareMonitor {
    try { New-Item -ItemType File -Force -Path $HardwareStop | Out-Null } catch {}
    if($script:HardwareProcess){try{$script:HardwareProcess.WaitForExit(15000)|Out-Null}catch{}}
}

function Read-GateDecision([string]$Path) {
    try { return [string](Get-Content -Raw $Path | ConvertFrom-Json).decision } catch { return "INVALID" }
}

function Package-Results {
    $runDirs = @(
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda4_seed7",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed13",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed23",
        "artifacts/runs/scientific_recovery_a5_s1_seed7","artifacts/runs/scientific_recovery_a5_s1_seed13","artifacts/runs/scientific_recovery_a5_s1_seed23",
        "artifacts/runs/scientific_recovery_a6_s1_seed7","artifacts/runs/scientific_recovery_a6_s1_seed13","artifacts/runs/scientific_recovery_a6_s1_seed23",
        "artifacts/runs/scientific_recovery_a7_s1_seed7","artifacts/runs/scientific_recovery_a7_s1_seed13","artifacts/runs/scientific_recovery_a7_s1_seed23",
        "artifacts/runs/scientific_recovery_a4_causal_left_seed7","artifacts/runs/scientific_recovery_a4_causal_left_seed13","artifacts/runs/scientific_recovery_a4_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a5_causal_left_seed7","artifacts/runs/scientific_recovery_a5_causal_left_seed13","artifacts/runs/scientific_recovery_a5_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a6_causal_left_seed7","artifacts/runs/scientific_recovery_a6_causal_left_seed13","artifacts/runs/scientific_recovery_a6_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a7_causal_left_seed7","artifacts/runs/scientific_recovery_a7_causal_left_seed13","artifacts/runs/scientific_recovery_a7_causal_left_seed23",
        "artifacts/runs/garl_budget_matched_s1_8192_seed7"
    ) | ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path $_ }
    $args=@("scripts/package_scientific_recovery_results.py","--master-root",$Master,"--repo-root",$Root,"--output-zip",$OutputZip)
    foreach($r in $runDirs){$args += @("--run-dir",$r)}
    Invoke-Python "99_package_results" "packaging" $args -AllowFailure | Out-Null
}

# Conservative allocator behavior; no TF32/compile/batch-size changes are introduced.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUNBUFFERED = "1"
# With two 10-worker DataLoaders in parallel this uses ~20 loader processes while
# preventing BLAS/OpenMP oversubscription across the 32 logical Ryzen threads.
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

if ($Force -and (Test-Path $Master)) { Remove-Item -Recurse -Force $Master }
New-Item -ItemType Directory -Force -Path $Master,$Logs,$Audit,$Configs | Out-Null
Start-HardwareMonitor

try {
    $head=(& git rev-parse HEAD).Trim()
    $mergeBaseCode = Invoke-NativeExitCode -FilePath ((Get-Command git).Source) -Arguments @("merge-base","--is-ancestor",$ExpectedBase,$head) -Quiet
    if($mergeBaseCode -ne 0){throw "HEAD $head is not a descendant of expected hardening base $ExpectedBase"}
    (& git status --short) | Set-Content (Join-Path $Master "git_status_start.txt") -Encoding utf8
    (& git log -30 --oneline --decorate) | Set-Content (Join-Path $Master "git_log_30.txt") -Encoding utf8
    (& nvidia-smi) | Set-Content (Join-Path $Master "nvidia_smi_start.txt") -Encoding utf8
    (python --version) | Set-Content (Join-Path $Master "python_version.txt") -Encoding utf8
    $cpuInfo = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    $osInfo = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $gpuInfo = (& nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    [ordered]@{
        logical_processors = if($cpuInfo){$cpuInfo.NumberOfLogicalProcessors}else{$null}
        physical_memory_gb = if($cpuInfo){[math]::Round([double]$cpuInfo.TotalPhysicalMemory/1GB,2)}else{$null}
        free_memory_gb = if($osInfo){[math]::Round(([double]$osInfo.FreePhysicalMemory*1KB)/1GB,2)}else{$null}
        gpu_snapshot = $gpuInfo
        num_workers_per_training = $NumWorkers
        prefetch_factor = $PrefetchFactor
        parallel_replications = (-not $DisableParallelReplications)
        min_free_vram_for_parallel_mib = $MinFreeVramForParallelMiB
        omp_threads_per_process = 2
        private_test_opened = $false
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Master "hardware_profile.json") -Encoding utf8

    # 00: code integrity. These tests do not open validation/test data.
    $compileArgs=@("-m","py_compile","src/e_jepa_ttc/models/causal_scale_ttc.py","src/e_jepa_ttc/training/causal_scale_eap.py","scripts/train_causal_scale_eap_screen.py","scripts/freeze_a4_s1_runtime_configs.py","scripts/freeze_a5_suite_configs.py","scripts/freeze_scientific_recovery_s1_configs.py","scripts/freeze_causal_hardening_configs.py","scripts/audit_scientific_recovery_contracts.py","scripts/audit_prefix_causality.py","scripts/evaluate_oracle_roi_stress.py","scripts/classify_scientific_recovery_gate.py","scripts/summarize_scientific_recovery_replication.py","scripts/paired_cluster_bootstrap.py","scripts/build_garl_budget_matched_subset.py","scripts/build_scientific_claim_readiness.py","scripts/package_scientific_recovery_results.py")
    if((Invoke-Python "00_py_compile" "integrity" $compileArgs -AllowFailure) -ne 0){ throw "scientific-recovery code does not compile" }
    if(-not $SkipTests){
        $testArgs=@("-m","pytest","-q","tests/unit/test_scientific_recovery_causality.py","tests/unit/test_a4_s1_lambda8_contract.py","tests/unit/test_a5_local_transport.py","tests/unit/test_a6_transport_adapter.py","tests/unit/test_causal_scale_ttc.py","tests/unit/test_garl_matched_cached_training.py")
        if((Invoke-Python "01_pytest" "integrity" $testArgs -AllowFailure) -ne 0){ throw "integrity tests failed; training blocked" }
    }

    # 01: claim/casuality audits are diagnostic. Garl audit failure blocks claims, not E-JEPA training.
    $contractOut=Join-Path $Audit "scientific_contracts.json"
    if((Invoke-Python "02_contract_audit" "audit" @("scripts/audit_scientific_recovery_contracts.py","--garl-repo",$GarlRepo,"--output",$contractOut) -AllowFailure) -ne 0){$script:ClaimsBlocked=$true}
    $prefixOut=Join-Path $Audit "prefix_causality.json"
    if((Invoke-Python "03_prefix_audit" "audit" @("scripts/audit_prefix_causality.py","--output",$prefixOut) -AllowFailure) -ne 0){$script:CausalBlocked=$true}

    # 02: immutable A4-S1 data/teacher contract. Failure blocks S1-family training.
    foreach($p in @($TrainManifest,$ValManifest,$TeacherManifest)){if(-not(Test-Path $p)){throw "required S1 artifact missing: $p"}}
    if((Invoke-Python "04_a4_s1_prerequisites" "integrity" @("scripts/verify_a4_s1_lambda8_prerequisites.py","--config","configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml") -AllowFailure) -ne 0){throw "A4-S1 preregistration/data/teacher gate failed"}
    $a4CfgDir=Join-Path $Configs "a4_runtime"
    if((Invoke-Python "05_freeze_a4_runtime" "freeze" @("scripts/freeze_a4_s1_runtime_configs.py","--output-dir",$a4CfgDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure) -ne 0){throw "A4 runtime freeze failed"}

    # 10: execute the preregistered lambda=4 attribution control. Its failure does not block A5.
    $a4l4run=Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda4_seed7"
    [void](Invoke-Train "10_a4_s1_lambda4_seed7" (Join-Path $a4CfgDir "a4_s1_lambda4_seed7.yaml") $a4l4run -AllowFailure)

    # Ensure primary A4-S1 lambda8 seed7 exists; usually reused from the completed branch.
    if(-not(Test-Path $A4L8Summary) -or -not(Test-Path $A4L8Checkpoint)){
        [void](Invoke-Train "11_a4_s1_lambda8_seed7" (Join-Path $a4CfgDir "a4_s1_lambda8_seed7.yaml") $A4L8RunBase -AllowFailure)
    } else { Add-StepStatus "11_a4_s1_lambda8_seed7" "training" "REUSED" 0 "existing A4-S1 seed7 result/checkpoint reused" $A4L8RunBase }
    if(-not(Test-Path $A4L8Summary) -or -not(Test-Path $A4L8Checkpoint)){throw "A4-S1 lambda8 seed7 comparator/checkpoint unavailable"}

    # Replicate A4-S1 lambda8. Failures are recorded but do not stop seed7 transport discovery.
    [void](Invoke-ParallelTrainPair "12_a4_s1_lambda8" (Join-Path $a4CfgDir "a4_s1_lambda8_seed13.yaml") (Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed13") (Join-Path $a4CfgDir "a4_s1_lambda8_seed23.yaml") (Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed23"))

    if((Test-Path $A4Base2k) -and (Test-Path (Join-Path $a4l4run "summary.json"))){
        [void](Invoke-Python "13_a4_attribution" "analysis" @("scripts/summarize_a4_s1_attribution.py","--a4-2k-lambda4",$A4Base2k,"--a4-8k-lambda4",(Join-Path $a4l4run "summary.json"),"--a4-8k-lambda8",$A4L8Summary,"--output",(Join-Path $Audit "a4_s1_attribution.json")) -AllowFailure)
    } else { Add-StepStatus "13_a4_attribution" "analysis" "SKIPPED" 0 "one attribution summary missing" $Audit }

    # 20: train-only A5 transport selection. Reuse V3 if valid; otherwise rerun V2/V3 only (zero optimizer steps).
    $selection=$null
    if(Test-Path $PreflightV3){
        try{$j=Get-Content -Raw $PreflightV3|ConvertFrom-Json;if($j.decision.a5_corr_authorized -eq $true){$selection=$PreflightV3}}catch{}
    }
    if(-not $selection){
        [void](Invoke-Python "20_a5_preflight_v2" "train_only_preflight" @("scripts/diagnose_a5_transport_preflight_v2.py","--output-dir","artifacts/metrics/a5_transport_preflight_v2","--device",$Device,"--samples","512","--batch-size","8") -AllowFailure)
        if(Test-Path $PreflightV2){
            $v2=Get-Content -Raw $PreflightV2|ConvertFrom-Json
            if($v2.decision.a5_corr_authorized -eq $true){$selection=$PreflightV2}
            else {
                [void](Invoke-Python "21_a5_preflight_v3" "train_only_preflight" @("scripts/diagnose_a5_transport_preflight_v3_confirm.py","--v2-artifact",$PreflightV2,"--output-dir","artifacts/metrics/a5_transport_preflight_v3_confirm","--device",$Device,"--batch-size","8") -AllowFailure)
                if(Test-Path $PreflightV3){$v3=Get-Content -Raw $PreflightV3|ConvertFrom-Json;if($v3.decision.a5_corr_authorized -eq $true){$selection=$PreflightV3}}
            }
        }
    } else { Add-StepStatus "20_a5_preflight" "train_only_preflight" "REUSED" 0 "authorized train-only selection reused" $selection }

    if(-not $selection){
        $script:TransportBlocked=$true
        Add-StepStatus "22_transport_branch" "decision" "STOP" 0 "A5 train-only transport preflight did not authorize a candidate" $Audit
    } else {
        $a5CfgDir=Join-Path $Configs "a5_s1"
        $freezeCode=Invoke-Python "22_freeze_a5_s1" "freeze" @("scripts/freeze_a5_suite_configs.py","--output-dir",$a5CfgDir,"--preflight-selection",$selection,"--include-scale","--scale-dino-lambda","8","--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure
        $a5cfg7=Join-Path $a5CfgDir "scale_8192_seed7.yaml"
        if($freezeCode -ne 0 -or -not(Test-Path $a5cfg7)){
            $script:TransportBlocked=$true
            Add-StepStatus "23_a5_s1" "decision" "STOP_INFRASTRUCTURE" 0 "A5-S1 config could not be frozen; diagnostics continue" $Audit
        } else {
            $a5runs=@{}
            foreach($seed in @(7,13,23)){$a5runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a5_s1_seed$seed"}
            $a5code=Invoke-Train "23_a5_s1_seed7" $a5cfg7 $a5runs[7] -AllowFailure
            if($a5code -eq 0 -and (Test-Path (Join-Path $a5runs[7] "summary.json"))){
                $a5Gate=Join-Path $Audit "a5_s1_seed7_gate.json"
                [void](Invoke-Python "24_gate_a5_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a5","--base-summary",$A4L8Summary,"--candidate-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a5Gate) -AllowFailure)
                $decision=Read-GateDecision $a5Gate
                if($decision -eq "REPLICATE_A5"){
                    [void](Invoke-ParallelTrainPair "25_a5_s1" (Join-Path $a5CfgDir "scale_8192_seed13.yaml") $a5runs[13] (Join-Path $a5CfgDir "scale_8192_seed23.yaml") $a5runs[23])
                    if((Test-Path (Join-Path $a5runs[13] "summary.json")) -and (Test-Path (Join-Path $a5runs[23] "summary.json"))){
                        $rep=Join-Path $Audit "a5_s1_replication.json"
                        [void](Invoke-Python "26_replicate_a5" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a5","--base-summary",$A4L8Summary,"--summary",(Join-Path $a5runs[7] "summary.json"),"--summary",(Join-Path $a5runs[13] "summary.json"),"--summary",(Join-Path $a5runs[23] "summary.json"),"--required-passes","2","--output",$rep) -AllowFailure)
                        if(Test-JsonStatus $rep "PASS"){$script:LegacyWinner="a5"}
                    }
                } elseif($decision -eq "RUN_A6") {
                    $transportCfgDir=Join-Path $Configs "a6_a7_s1"
                    $fc=Invoke-Python "27_freeze_a6_a7_s1" "freeze" @("scripts/freeze_scientific_recovery_s1_configs.py","--a5-s1-config",$a5cfg7,"--a4-s1-checkpoint",$A4L8Checkpoint,"--output-dir",$transportCfgDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure
                    if($fc -eq 0){
                        $a6runs=@{};foreach($seed in @(7,13,23)){$a6runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a6_s1_seed$seed"}
                        $a6code=Invoke-Train "30_a6_s1_seed7" (Join-Path $transportCfgDir "a6_s1_seed7.yaml") $a6runs[7] -AllowFailure
                        if($a6code -eq 0 -and (Test-Path (Join-Path $a6runs[7] "summary.json"))){
                            $a6Gate=Join-Path $Audit "a6_s1_seed7_gate.json"
                            [void](Invoke-Python "31_gate_a6_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a6","--base-summary",$A4L8Summary,"--candidate-summary",(Join-Path $a6runs[7] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a6Gate) -AllowFailure)
                            $d6=Read-GateDecision $a6Gate
                            if($d6 -eq "REPLICATE_A6"){
                                [void](Invoke-ParallelTrainPair "32_a6_s1" (Join-Path $transportCfgDir "a6_s1_seed13.yaml") $a6runs[13] (Join-Path $transportCfgDir "a6_s1_seed23.yaml") $a6runs[23])
                                if((Test-Path (Join-Path $a6runs[13] "summary.json")) -and (Test-Path (Join-Path $a6runs[23] "summary.json"))){
                                    $rep6=Join-Path $Audit "a6_s1_replication.json"; [void](Invoke-Python "33_replicate_a6" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a6","--base-summary",$A4L8Summary,"--summary",(Join-Path $a6runs[7] "summary.json"),"--summary",(Join-Path $a6runs[13] "summary.json"),"--summary",(Join-Path $a6runs[23] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--required-passes","2","--output",$rep6) -AllowFailure); if(Test-JsonStatus $rep6 "PASS"){$script:LegacyWinner="a6"}
                                }
                            } elseif($d6 -eq "RUN_A7") {
                                $a7runs=@{};foreach($seed in @(7,13,23)){$a7runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a7_s1_seed$seed"}
                                $a7code=Invoke-Train "34_a7_s1_seed7" (Join-Path $transportCfgDir "a7_s1_seed7.yaml") $a7runs[7] -AllowFailure
                                if($a7code -eq 0 -and (Test-Path (Join-Path $a7runs[7] "summary.json"))){
                                    $a7Gate=Join-Path $Audit "a7_s1_seed7_gate.json"; [void](Invoke-Python "35_gate_a7_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a7","--base-summary",$A4L8Summary,"--candidate-summary",(Join-Path $a7runs[7] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a7Gate) -AllowFailure)
                                    if((Read-GateDecision $a7Gate) -eq "REPLICATE_A7"){
                                        [void](Invoke-ParallelTrainPair "36_a7_s1" (Join-Path $transportCfgDir "a7_s1_seed13.yaml") $a7runs[13] (Join-Path $transportCfgDir "a7_s1_seed23.yaml") $a7runs[23])
                                        if((Test-Path (Join-Path $a7runs[13] "summary.json")) -and (Test-Path (Join-Path $a7runs[23] "summary.json"))){
                                            $rep7=Join-Path $Audit "a7_s1_replication.json"; [void](Invoke-Python "37_replicate_a7" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a7","--base-summary",$A4L8Summary,"--summary",(Join-Path $a7runs[7] "summary.json"),"--summary",(Join-Path $a7runs[13] "summary.json"),"--summary",(Join-Path $a7runs[23] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--required-passes","2","--output",$rep7) -AllowFailure); if(Test-JsonStatus $rep7 "PASS"){$script:LegacyWinner="a7"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    $script:TransportBlocked=$true
                    Add-StepStatus "27_transport_decision" "decision" "STOP" 0 "A5 gate=$decision; A6/A7 not scientifically justified" $Audit
                }
            } else {
                $script:TransportBlocked=$true
                Add-StepStatus "24_transport_decision" "decision" "STOP_INFRASTRUCTURE" 0 "A5 seed7 failed; dependent A6/A7 blocked, independent diagnostics continue" $Audit
            }
        }
    }

    # 40: strict model-prefix causal hardening. Only a replicated legacy winner is promoted.
    if($script:LegacyWinner -and -not $script:CausalBlocked){
        $stage=[string]$script:LegacyWinner
        $legacyCfgDir = if($stage -eq "a5") { Join-Path $Configs "a5_s1" } else { Join-Path $Configs "a6_a7_s1" }
        $winnerCfg = if($stage -eq "a5") { Join-Path $legacyCfgDir "scale_8192_seed7.yaml" } else { Join-Path $legacyCfgDir "${stage}_s1_seed7.yaml" }
        $causalDir=Join-Path $Configs "causal_$stage"
        [void](Invoke-Python "40_freeze_causal_${stage}_pre" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config","configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml","--winner-source-config",$winnerCfg,"--winner-stage",$stage,"--output-dir",$causalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
        # A4 causal seeds are matched comparators for strict-causal replication.
        $a4cr=@{};foreach($seed in @(7,13,23)){$a4cr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a4_causal_left_seed$seed"}
        $c4=Invoke-Train "41_a4_causal_seed7" (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed7.yaml") $a4cr[7] -AllowFailure
        if($c4 -eq 0 -and (Test-Path (Join-Path $a4cr[7] "model_best.pt"))){
            [void](Invoke-ParallelTrainPair "42_a4_causal" (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed13.yaml") $a4cr[13] (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed23.yaml") $a4cr[23])
            # Re-freeze to cryptographically bind A6/A7 initialization to causal A4 seed7.
            [void](Invoke-Python "43_refreeze_causal_${stage}" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config","configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml","--winner-source-config",$winnerCfg,"--winner-stage",$stage,"--causal-a4-checkpoint",(Join-Path $a4cr[7] "model_best.pt"),"--output-dir",$causalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
            # For A6/A7 recovery fractions, train the causal A5 reference on matched seeds.
            $a5CausalDir=$null; $a5cr=@{}
            if($stage -in @("a6","a7")){
                $a5CausalDir=Join-Path $Configs "causal_a5_reference"
                $a5source=Join-Path (Join-Path $Configs "a5_s1") "scale_8192_seed7.yaml"
                [void](Invoke-Python "44_freeze_causal_a5_reference" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config","configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml","--winner-source-config",$a5source,"--winner-stage","a5","--output-dir",$a5CausalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
                foreach($seed in @(7,13,23)){$a5cr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a5_causal_left_seed$seed"}
                [void](Invoke-Train "45_a5_causal_seed7" (Join-Path $a5CausalDir "a5_s1_causal_left_seed7.yaml") $a5cr[7] -AllowFailure)
                [void](Invoke-ParallelTrainPair "46_a5_causal" (Join-Path $a5CausalDir "a5_s1_causal_left_seed13.yaml") $a5cr[13] (Join-Path $a5CausalDir "a5_s1_causal_left_seed23.yaml") $a5cr[23])
            }
            $wcr=@{};foreach($seed in @(7,13,23)){$wcr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_${stage}_causal_left_seed$seed"}
            $wc=Invoke-Train "47_${stage}_causal_seed7" (Join-Path $causalDir "${stage}_s1_causal_left_seed7.yaml") $wcr[7] -AllowFailure
            if($wc -eq 0){[void](Invoke-ParallelTrainPair "48_${stage}_causal" (Join-Path $causalDir "${stage}_s1_causal_left_seed13.yaml") $wcr[13] (Join-Path $causalDir "${stage}_s1_causal_left_seed23.yaml") $wcr[23])}
            $allCausal = @(7,13,23) | ForEach-Object { Test-Path (Join-Path $wcr[$_] "summary.json") }
            $allBase = @(7,13,23) | ForEach-Object { Test-Path (Join-Path $a4cr[$_] "summary.json") }
            if(($allCausal -notcontains $false) -and ($allBase -notcontains $false)){
                $repArgs=@("scripts/summarize_scientific_recovery_replication.py","--stage",$stage)
                foreach($seed in @(7,13,23)){$repArgs+=@("--base-summary",(Join-Path $a4cr[$seed] "summary.json"))}
                foreach($seed in @(7,13,23)){$repArgs+=@("--summary",(Join-Path $wcr[$seed] "summary.json"))}
                if($stage -in @("a6","a7")){foreach($seed in @(7,13,23)){$repArgs+=@("--a5-summary",(Join-Path $a5cr[$seed] "summary.json"))}}
                $causalRep=Join-Path $Audit "${stage}_causal_replication.json"; $repArgs+=@("--required-passes","2","--output",$causalRep)
                [void](Invoke-Python "49_${stage}_causal_replication" "gate" $repArgs -AllowFailure)
                if(Test-JsonStatus $causalRep "PASS"){$script:CausalWinner=$stage;$script:FinalCandidateRun=$wcr[7];$script:FinalCandidateMode="causal_left"}
            }
        }
    }

    # If no strict-causal transport winner exists, retain the best honest public-only diagnostic candidate.
    if(-not $script:FinalCandidateRun){
        if($script:LegacyWinner){
            $script:FinalCandidateRun=Join-Path $Root "artifacts\runs\scientific_recovery_$($script:LegacyWinner)_s1_seed7"; $script:FinalCandidateMode="legacy"
        } else {$script:FinalCandidateRun=$A4L8RunBase;$script:FinalCandidateMode="legacy"}
    }

    # 60: ROI stress is independent of transport gates and always useful when a checkpoint exists.
    $candidateCheckpoint=Join-Path $script:FinalCandidateRun "model_best.pt"
    $candidateSummary=Join-Path $script:FinalCandidateRun "summary.json"
    $candidatePred=Join-Path $script:FinalCandidateRun "validation_predictions.csv"
    if((Test-Path $candidateCheckpoint) -and (Test-Path $ValManifest)){
        [void](Invoke-Python "60_oracle_roi_stress" "audit" @("scripts/evaluate_oracle_roi_stress.py","--checkpoint",$candidateCheckpoint,"--validation-manifest",$ValManifest,"--output",(Join-Path $Audit "oracle_roi_stress.json"),"--device",$Device,"--batch-size","64","--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
    }

    # 70: budget-matched Garl 8192 comparator. Failure blocks SOTA claim only, never E-JEPA evidence.
    $garlSummary=$null; $paired=$null
    if(-not $SkipBudgetMatchedGarl){
        $publicData=Join-Path $GarlDataset "data\train.parquet"; $publicLabels=Join-Path $GarlDataset "annotations\train.parquet"
        if((Test-Path $publicData) -and (Test-Path $publicLabels) -and (Test-Path $TrainManifest) -and (Test-Path $ValManifest)){
            $subsetDir=Join-Path $Master "garl_budget_subset"
            $sub=Invoke-Python "70_build_garl_8192_subset" "garl_compare" @("scripts/build_garl_budget_matched_subset.py","--train-cache-manifest",$TrainManifest,"--validation-cache-manifest",$ValManifest,"--public-data-parquet",$publicData,"--public-labels-parquet",$publicLabels,"--output-dir",$subsetDir) -AllowFailure
            $cacheDir=Join-Path $Root "artifacts\cache\garl_budget_matched_s1_8192_v1"
            if($sub -eq 0){
                $bc=Invoke-Python "71_build_garl_8192_cache" "garl_compare" @("scripts/build_garl_matched_preprocessing_cache.py","--release-root",$GarlRepo,"--subset-manifest",(Join-Path $subsetDir "manifest.json"),"--eap-root",$EapDataset,"--output-dir",$cacheDir,"--batch-size","32","--num-workers",[string]$NumWorkers,"--shard-size","64","--seed","7") -AllowFailure
                if($bc -eq 0){
                    $garlRun=Join-Path $Root "artifacts\runs\garl_budget_matched_s1_8192_seed7"
                    $gt=Invoke-Python "72_train_garl_8192" "garl_compare" @("scripts/train_garl_matched_from_cache.py","--release-root",$GarlRepo,"--cache-manifest",(Join-Path $cacheDir "manifest.json"),"--output-dir",$garlRun,"--device",$Device,"--seed","7","--epochs","18","--batch-size","32","--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor,"--expected-train-rows","8192","--expected-validation-rows","2048","--minimum-epochs","8","--early-stopping-patience","5","--maximum-runtime-hours","8") -AllowFailure
                    $garlSummary=Join-Path $garlRun "summary.json"
                    $garlPred=Join-Path $garlRun "validation_predictions.parquet"
                    if($gt -eq 0 -and (Test-Path $candidatePred) -and (Test-Path $garlPred)){
                        $paired=Join-Path $Audit "paired_ejepa_vs_garl_8192.json"
                        [void](Invoke-Python "73_paired_bootstrap" "garl_compare" @("scripts/paired_cluster_bootstrap.py","--ejepa-predictions",$candidatePred,"--garl-predictions",$garlPred,"--resamples",[string]$BootstrapResamples,"--seed","20260811","--output",$paired) -AllowFailure)
                    }
                }
            }
        } else { $script:ClaimsBlocked=$true; Add-StepStatus "70_garl_budget_matched" "garl_compare" "SKIPPED" 0 "public Garl train/labels or E-JEPA manifests missing" $Audit }
    } else {Add-StepStatus "70_garl_budget_matched" "garl_compare" "SKIPPED" 0 "-SkipBudgetMatchedGarl requested" $Audit}

    # 80: claim boundary. This can only authorize readiness for a future one-shot sealed test; never SOTA itself.
    if((Test-Path $candidateSummary) -and (Test-Path $contractOut) -and (Test-Path $prefixOut)){
        $claimArgs=@("scripts/build_scientific_claim_readiness.py","--contract-audit",$contractOut,"--prefix-audit",$prefixOut,"--candidate-summary",$candidateSummary,"--candidate-mode",$script:FinalCandidateMode,"--output",(Join-Path $Audit "claim_readiness.json"))
        if($garlSummary -and (Test-Path $garlSummary)){$claimArgs+=@("--garl-budget-summary",$garlSummary)}
        if($paired -and (Test-Path $paired)){$claimArgs+=@("--paired-bootstrap",$paired)}
        [void](Invoke-Python "80_claim_readiness" "claim" $claimArgs -AllowFailure)
    }
}
catch {
    Add-StepStatus "MASTER_EXCEPTION" "master" "FAIL" 1 $_.Exception.Message $Master
    Write-Warning "Master encountered a global integrity/infrastructure stop: $($_.Exception.Message). Independent evidence collected so far will still be packaged."
}
finally {
    Stop-HardwareMonitor
    Write-Status
    try { Package-Results } catch { Write-Warning "Packaging failed: $_" }
    Write-Host "`n=== SCIENTIFIC RECOVERY MASTER COMPLETE ===" -ForegroundColor Green
    Write-Host "Master evidence: $Master"
    if(Test-Path $OutputZip){
        Write-Host "ZIP: $OutputZip" -ForegroundColor Green
        Write-Host "SHA256: $((Get-FileHash $OutputZip -Algorithm SHA256).Hash.ToLower())" -ForegroundColor Green
    }
    Write-Host "Private/test evaluation was NOT invoked by this script."
}
