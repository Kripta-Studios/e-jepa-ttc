[CmdletBinding()]
param(
    [string]$RepoPath = "C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc",
    [string]$EapRoot = "E:\eAP_dataset",
    [string]$GarlTtcRoot = "E:\GarlTTC_dataset",
    [string]$SplitPath = "data\splits\eap_pilot12_v1.json",
    [string]$EvTtcValidationManifest = "",
    [string]$PatchPath = "",
    [int]$Workers = 10,
    [int]$CacheSamplesPerSplit = 4096,
    [int]$Epochs = 8,
    [int]$BatchSize = 8,
    [int[]]$Seeds = @(7, 13, 23),
    [switch]$IncludeRgb,
    [switch]$RunFullTests,
    [switch]$SkipGeo2,
    [switch]$AllowDatasetVersionChange
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ExpectedCommit = "16feff38a1bf5746c32d84935ace6c3bad225b5c"
$PatchFileName = "eap_geometry_v2_lhr_jepa_ttc_16feff38_v2.patch"

if ([string]::IsNullOrWhiteSpace($PatchPath)) {
    $Candidate = Join-Path $PSScriptRoot $PatchFileName
    if (Test-Path $Candidate) {
        $PatchPath = $Candidate
    }
    else {
        $PatchPath = Join-Path $RepoPath $PatchFileName
    }
}

$RepoPath = (Resolve-Path $RepoPath).Path
Set-Location $RepoPath

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $RepoPath "artifacts\runs\eap_lhr_v2_unattended_$Timestamp"
$Logs = Join-Path $RunRoot "logs"
$Metrics = Join-Path $RunRoot "metrics"
$Experiments = Join-Path $RunRoot "experiments"
$OfficialCache = Join-Path $RunRoot "official_cache"
$SmokeCache = Join-Path $RunRoot "smoke_cache"
$Geo2Run = Join-Path $RunRoot "geo2_pilot_seed42"
New-Item -ItemType Directory -Force -Path $RunRoot, $Logs, $Metrics, $Experiments | Out-Null

$TranscriptPath = Join-Path $Logs "transcript.log"
Start-Transcript -Path $TranscriptPath -Force | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Awake {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = 0x80000000
$ES_SYSTEM_REQUIRED = 0x00000001
$ES_AWAYMODE_REQUIRED = 0x00000040
[Awake]::SetThreadExecutionState(
    $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
) | Out-Null

function Write-Stage {
    param([string]$Message)
    $Line = "=" * 88
    Write-Host ""
    Write-Host $Line
    Write-Host $Message
    Write-Host $Line
}

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $LogPath = Join-Path $Logs "$Name.log"
    Write-Host "[$Name] $Executable $($Arguments -join ' ')"
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "Step '$Name' failed with exit code $Code. See $LogPath"
    }
}

function Test-NativeSuccess {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments *> $null
    return ($LASTEXITCODE -eq 0)
}

try {
    Write-Stage "1/10 - Repository and patch"

    $Head = (& git rev-parse HEAD).Trim()
    if ($Head -ne $ExpectedCommit) {
        throw "Expected HEAD $ExpectedCommit, found $Head. Use a clean branch based on the requested commit."
    }

    if (-not (Test-Path $PatchPath)) {
        throw "Patch not found: $PatchPath"
    }
    $PatchPath = (Resolve-Path $PatchPath).Path

    $TrackedDirty = (
        (-not (Test-NativeSuccess git @("diff", "--quiet"))) -or
        (-not (Test-NativeSuccess git @("diff", "--cached", "--quiet")))
    )

    $AlreadyApplied = Test-NativeSuccess git @(
        "apply", "--reverse", "--check", $PatchPath
    )

    if (-not $AlreadyApplied) {
        if ($TrackedDirty) {
            & git status --short
            throw "Tracked working-tree changes exist before applying the patch. Save them first."
        }
        Invoke-NativeLogged "git_apply_check" git @("apply", "--check", $PatchPath)
        Invoke-NativeLogged "git_apply" git @("apply", $PatchPath)
    }
    else {
        Write-Host "Patch is already applied; continuing."
    }

    & git diff --binary | Set-Content -Encoding UTF8 (Join-Path $RunRoot "git_diff.patch")
    & git status --short | Set-Content -Encoding UTF8 (Join-Path $RunRoot "git_status.txt")
    & git rev-parse HEAD | Set-Content -Encoding UTF8 (Join-Path $RunRoot "base_commit.txt")

    Write-Stage "2/10 - Environment preflight"

    $Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Python virtual environment not found: $Python"
    }
    foreach ($Required in @(
        $EapRoot,
        $GarlTtcRoot,
        (Join-Path $GarlTtcRoot "data\train.parquet"),
        (Join-Path $GarlTtcRoot "annotations\train.parquet"),
        (Join-Path $RepoPath $SplitPath)
    )) {
        if (-not (Test-Path $Required)) {
            throw "Required path not found: $Required"
        }
    }

    $DriveName = (Split-Path -Qualifier $RunRoot).TrimEnd(":")
    $Drive = Get-PSDrive -Name $DriveName
    $Preflight = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        expected_commit = $ExpectedCommit
        actual_commit = $Head
        repo_path = $RepoPath
        eap_root = $EapRoot
        garlttc_root = $GarlTtcRoot
        split_path = $SplitPath
        patch_path = $PatchPath
        free_disk_gib = [math]::Round($Drive.Free / 1GB, 2)
        workers = $Workers
        cache_samples_per_split = $CacheSamplesPerSplit
        epochs = $Epochs
        batch_size = $BatchSize
        seeds = $Seeds
        include_rgb = [bool]$IncludeRgb
    }
    $Preflight | ConvertTo-Json -Depth 10 |
        Set-Content -Encoding UTF8 (Join-Path $RunRoot "run_manifest.json")

    Invoke-NativeLogged "python_version" $Python @("--version")
    Invoke-NativeLogged "cuda_preflight" $Python @(
        "-c",
        "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'vram_gib':round(torch.cuda.get_device_properties(0).total_memory/2**30,2) if torch.cuda.is_available() else None},indent=2))"
    )

    Write-Stage "3/10 - Compile and tests"

    $CompileFiles = @(
        "src\e_jepa_ttc\data\eap.py",
        "src\e_jepa_ttc\data\eap_cache.py",
        "src\e_jepa_ttc\data\eap_geometry_v2.py",
        "src\e_jepa_ttc\data\garlttc_lhr_cache.py",
        "src\e_jepa_ttc\models\eap_lhr_jepa_ttc.py",
        "src\e_jepa_ttc\training\eap_jepa.py",
        "src\e_jepa_ttc\training\eap_lhr_jepa_ttc.py",
        "scripts\pretrain_eap_jepa.py",
        "scripts\build_eap_lhr_cache.py",
        "scripts\train_eap_lhr_jepa_ttc.py",
        "scripts\evaluate_eap_lhr_zero_shot.py",
        "scripts\audit_garlttc_lhr_v2.py",
        "scripts\analyze_eap_geometry_v2.py",
        "scripts\summarize_eap_lhr_v2.py"
    )
    Invoke-NativeLogged "py_compile" $Python (@("-m", "py_compile") + $CompileFiles)
    Invoke-NativeLogged "pytest_targeted" $Python @(
        "-m", "pytest",
        "tests\unit\test_eap_geometry_v2.py",
        "tests\unit\test_eap_geo2_config.py",
        "tests\unit\test_eap_lhr_jepa_ttc.py",
        "tests\unit\test_garlttc_lhr_cache_v2.py",
        "-q"
    )
    Invoke-NativeLogged "pytest_unit" $Python @("-m", "pytest", "tests\unit", "-q")
    if ($RunFullTests) {
        Invoke-NativeLogged "pytest_full" $Python @("-m", "pytest", "-q")
    }

    Write-Stage "4/10 - Geo2 initialization"

    $GeoCheckpoint = ""
    if (-not $SkipGeo2) {
        Invoke-NativeLogged "geo2_smoke" $Python @(
            "scripts\pretrain_eap_jepa.py",
            "--objective", "geo2",
            "--profile", "smoke",
            "--root", $EapRoot,
            "--inventory", "data\manifests\eap_train40_inventory_v1.json",
            "--split", $SplitPath,
            "--output", (Join-Path $RunRoot "geo2_smoke_seed42"),
            "--workers", "0",
            "--seed", "42"
        )
        Invoke-NativeLogged "geo2_pilot" $Python @(
            "scripts\pretrain_eap_jepa.py",
            "--objective", "geo2",
            "--profile", "pilot",
            "--root", $EapRoot,
            "--inventory", "data\manifests\eap_train40_inventory_v1.json",
            "--split", $SplitPath,
            "--output", $Geo2Run,
            "--workers", "$Workers",
            "--batch-size", "24",
            "--gradient-accumulation", "2",
            "--seed", "42",
            "--resume"
        )
        $GeoCheckpoint = Join-Path $Geo2Run "eap_jepa_encoder_best.pt"
        if (-not (Test-Path $GeoCheckpoint)) {
            throw "Geo2 checkpoint not produced: $GeoCheckpoint"
        }
    }

    Write-Stage "5/10 - Official-label smoke cache and audit"

    $CacheBaseArgs = @(
        "scripts\build_eap_lhr_cache.py",
        "--eap-root", $EapRoot,
        "--garlttc-root", $GarlTtcRoot,
        "--split", $SplitPath,
        "--shard-size", "128"
    )
    if ($IncludeRgb) {
        $CacheBaseArgs += "--include-rgb"
    }
    if ($AllowDatasetVersionChange) {
        $CacheBaseArgs += "--allow-dataset-version-change"
    }

    Invoke-NativeLogged "build_smoke_cache" $Python (
        $CacheBaseArgs + @(
            "--output", $SmokeCache,
            "--max-samples-per-split", "32"
        )
    )
    $SmokeManifest = Join-Path $SmokeCache "manifest.json"
    Invoke-NativeLogged "audit_smoke_cache" $Python @(
        "scripts\audit_garlttc_lhr_v2.py",
        "--manifest", $SmokeManifest,
        "--output", (Join-Path $RunRoot "smoke_cache_audit.json"),
        "--max-samples", "32"
    )

    $SmokeRun = Join-Path $RunRoot "smoke_train"
    $SmokeArgs = @(
        "scripts\train_eap_lhr_jepa_ttc.py",
        "--manifest", $SmokeManifest,
        "--output", $SmokeRun,
        "--epochs", "1",
        "--minimum-epochs", "1",
        "--early-stopping-patience", "0",
        "--batch-size", "2",
        "--workers", "0",
        "--precision", "fp32",
        "--seed", "7",
        "--height-weight", "0.5",
        "--ratio-weight", "1.0",
        "--ttc-weight", "0.25",
        "--jepa-weight", "0.1",
        "--geometry-weight", "0.1",
        "--category-weight", "0.05",
        "--foreground-weight", "0.0"
    )
    if ($GeoCheckpoint) {
        $SmokeArgs += @("--geo-checkpoint", $GeoCheckpoint)
    }
    if ($IncludeRgb) {
        $SmokeArgs += "--use-rgb"
    }
    Invoke-NativeLogged "train_smoke" $Python $SmokeArgs

    Write-Stage "6/10 - Official-label pilot cache"

    Invoke-NativeLogged "build_official_cache" $Python (
        $CacheBaseArgs + @(
            "--output", $OfficialCache,
            "--max-samples-per-split", "$CacheSamplesPerSplit"
        )
    )
    $OfficialManifest = Join-Path $OfficialCache "manifest.json"
    Invoke-NativeLogged "audit_official_cache" $Python @(
        "scripts\audit_garlttc_lhr_v2.py",
        "--manifest", $OfficialManifest,
        "--output", (Join-Path $RunRoot "cache_audit.json"),
        "--max-samples", "256"
    )
    Invoke-NativeLogged "analyze_official_cache" $Python @(
        "scripts\analyze_eap_geometry_v2.py",
        "--manifest", $OfficialManifest,
        "--splits", "train",
        "--output", (Join-Path $Metrics "official_cache_balance.json")
    )

    Write-Stage "7/10 - L0/L1/L2/L3 ablations"

    $Arms = @(
        [ordered]@{
            Name = "L0_LHR_ONLY"
            Residual = "0"
            TTC = "0"
            JEPA = "0"
            Geometry = "0"
            Category = "0"
        },
        [ordered]@{
            Name = "L1_LHR_TTC"
            Residual = "0.25"
            TTC = "0.25"
            JEPA = "0"
            Geometry = "0"
            Category = "0"
        },
        [ordered]@{
            Name = "L2_LHR_TTC_GEO2"
            Residual = "0.25"
            TTC = "0.25"
            JEPA = "0"
            Geometry = "0.1"
            Category = "0.05"
        },
        [ordered]@{
            Name = "L3_LHR_TTC_GEO2_JEPA"
            Residual = "0.25"
            TTC = "0.25"
            JEPA = "0.1"
            Geometry = "0.1"
            Category = "0.05"
        }
    )

    foreach ($Arm in $Arms) {
        foreach ($Seed in $Seeds) {
            $Output = Join-Path $Experiments "$($Arm.Name)\seed-$Seed"
            $Arguments = @(
                "scripts\train_eap_lhr_jepa_ttc.py",
                "--manifest", $OfficialManifest,
                "--output", $Output,
                "--epochs", "$Epochs",
                "--minimum-epochs", "3",
                "--early-stopping-patience", "2",
                "--batch-size", "$BatchSize",
                "--workers", "$Workers",
                "--learning-rate", "0.0001",
                "--precision", "bf16",
                "--seed", "$Seed",
                "--height-weight", "0.5",
                "--ratio-weight", "1.0",
                "--ttc-weight", "$($Arm.TTC)",
                "--jepa-weight", "$($Arm.JEPA)",
                "--geometry-weight", "$($Arm.Geometry)",
                "--category-weight", "$($Arm.Category)",
                "--foreground-weight", "0.0",
                "--ttc-residual-scale-s", "$($Arm.Residual)",
                "--resume"
            )
            if ($GeoCheckpoint) {
                $Arguments += @("--geo-checkpoint", $GeoCheckpoint)
            }
            if ($IncludeRgb) {
                $Arguments += "--use-rgb"
            }
            Invoke-NativeLogged "train_$($Arm.Name)_seed$Seed" $Python $Arguments
        }
    }

    Write-Stage "8/10 - Zero-shot validation, when a compatible target cache is supplied"

    if (-not [string]::IsNullOrWhiteSpace($EvTtcValidationManifest)) {
        if (-not (Test-Path $EvTtcValidationManifest)) {
            throw "EvTTC validation manifest does not exist: $EvTtcValidationManifest"
        }
        $ZeroShotFailures = @()
        foreach ($Arm in $Arms) {
            foreach ($Seed in $Seeds) {
                $Checkpoint = Join-Path $Experiments "$($Arm.Name)\seed-$Seed\weights_only.pt"
                $MetricOutput = Join-Path $Metrics "$($Arm.Name)_seed${Seed}_zero_shot.json"
                try {
                    Invoke-NativeLogged "zero_shot_$($Arm.Name)_seed$Seed" $Python @(
                        "scripts\evaluate_eap_lhr_zero_shot.py",
                        "--checkpoint", $Checkpoint,
                        "--manifest", $EvTtcValidationManifest,
                        "--splits", "validation",
                        "--output", $MetricOutput,
                        "--batch-size", "$BatchSize",
                        "--workers", "$Workers"
                    )
                }
                catch {
                    $ZeroShotFailures += [ordered]@{
                        arm = $Arm.Name
                        seed = $Seed
                        checkpoint = $Checkpoint
                        manifest = $EvTtcValidationManifest
                        message = $_.Exception.Message
                    }
                    Write-Warning "Zero-shot failed for $($Arm.Name), seed $Seed; continuing."
                }
            }
        }
        if ($ZeroShotFailures.Count -gt 0) {
            $ZeroShotFailures | ConvertTo-Json -Depth 10 |
                Set-Content -Encoding UTF8 (Join-Path $RunRoot "zero_shot_failures.json")
        }
    }
    else {
        "EvTTC zero-shot skipped: no -EvTtcValidationManifest was supplied." |
            Set-Content -Encoding UTF8 (Join-Path $Logs "zero_shot_skipped.log")
    }

    Write-Stage "9/10 - Handoff summary"

    Invoke-NativeLogged "summarize" $Python @(
        "scripts\summarize_eap_lhr_v2.py",
        "--root", $RunRoot,
        "--output", (Join-Path $RunRoot "pipeline_summary.json")
    )

    $ShareList = @(
        "Upload these files/directories when you return:",
        (Join-Path $RunRoot "run_manifest.json"),
        (Join-Path $RunRoot "pipeline_summary.json"),
        (Join-Path $RunRoot "cache_audit.json"),
        (Join-Path $RunRoot "zero_shot_failures.json"),
        (Join-Path $OfficialCache "manifest.json"),
        (Join-Path $RunRoot "git_diff.patch"),
        (Join-Path $RunRoot "git_status.txt"),
        (Join-Path $RunRoot "logs"),
        (Join-Path $RunRoot "metrics"),
        (Join-Path $RunRoot "experiments")
    )
    $ShareList | Set-Content -Encoding UTF8 (Join-Path $RunRoot "FILES_TO_SHARE.txt")

    $HandoffDir = Join-Path $RunRoot "handoff_bundle"
    New-Item -ItemType Directory -Force -Path $HandoffDir | Out-Null
    foreach ($Name in @(
        "run_manifest.json",
        "pipeline_summary.json",
        "cache_audit.json",
        "zero_shot_failures.json",
        "git_diff.patch",
        "git_status.txt",
        "FILES_TO_SHARE.txt"
    )) {
        $Source = Join-Path $RunRoot $Name
        if (Test-Path $Source) {
            Copy-Item $Source (Join-Path $HandoffDir $Name) -Force
        }
    }
    if (Test-Path $Logs) {
        Copy-Item $Logs (Join-Path $HandoffDir "logs") -Recurse -Force
    }
    if (Test-Path $Metrics) {
        Copy-Item $Metrics (Join-Path $HandoffDir "metrics") -Recurse -Force
    }
    if (Test-Path (Join-Path $OfficialCache "manifest.json")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $HandoffDir "official_cache") |
            Out-Null
        Copy-Item (Join-Path $OfficialCache "manifest.json") `
            (Join-Path $HandoffDir "official_cache\manifest.json") -Force
    }
    foreach ($SummaryFile in (
        Get-ChildItem $Experiments -Recurse -File |
            Where-Object { $_.Name -in @("summary.json", "history.jsonl") }
    )) {
        $Relative = $SummaryFile.FullName.Substring($Experiments.Length).TrimStart("\")
        $Destination = Join-Path (Join-Path $HandoffDir "experiments") $Relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) |
            Out-Null
        Copy-Item $SummaryFile.FullName $Destination -Force
    }

    $SmallHandoff = Join-Path $RunRoot "handoff_without_checkpoints.zip"
    Compress-Archive -Path (Join-Path $HandoffDir "*") `
        -DestinationPath $SmallHandoff -Force

    Write-Stage "10/10 - Complete"
    Write-Host "Run root: $RunRoot"
    Write-Host "Small handoff ZIP: $SmallHandoff"
    Write-Host "Read: $(Join-Path $RunRoot 'FILES_TO_SHARE.txt')"
}
catch {
    $Failure = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        message = $_.Exception.Message
        script_stack = $_.ScriptStackTrace
        run_root = $RunRoot
    }
    $FailurePath = Join-Path $RunRoot "FAILURE.json"
    $Failure | ConvertTo-Json -Depth 10 |
        Set-Content -Encoding UTF8 $FailurePath

    try {
        $FailureBundle = Join-Path $RunRoot "failure_handoff"
        New-Item -ItemType Directory -Force -Path $FailureBundle | Out-Null
        foreach ($Name in @(
            "FAILURE.json",
            "run_manifest.json",
            "pipeline_summary.json",
            "cache_audit.json",
            "smoke_cache_audit.json",
            "git_diff.patch",
            "git_status.txt"
        )) {
            $Source = Join-Path $RunRoot $Name
            if (Test-Path $Source) {
                Copy-Item $Source (Join-Path $FailureBundle $Name) -Force
            }
        }
        if (Test-Path $Logs) {
            Copy-Item $Logs (Join-Path $FailureBundle "logs") -Recurse -Force
        }
        if (Test-Path $Metrics) {
            Copy-Item $Metrics (Join-Path $FailureBundle "metrics") -Recurse -Force
        }
        if (Test-Path $Experiments) {
            foreach ($ResultFile in (
                Get-ChildItem $Experiments -Recurse -File |
                    Where-Object { $_.Name -in @("summary.json", "history.jsonl") }
            )) {
                $Relative = $ResultFile.FullName.Substring($Experiments.Length).TrimStart("\")
                $Destination = Join-Path (Join-Path $FailureBundle "experiments") $Relative
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) |
                    Out-Null
                Copy-Item $ResultFile.FullName $Destination -Force
            }
        }
        Compress-Archive -Path (Join-Path $FailureBundle "*") `
            -DestinationPath (Join-Path $RunRoot "failure_handoff.zip") -Force
    }
    catch {
        Write-Warning "Could not create failure handoff ZIP: $($_.Exception.Message)"
    }

    Write-Error $Failure.message
    exit 1
}
finally {
    [Awake]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    try { Stop-Transcript | Out-Null } catch {}
}
