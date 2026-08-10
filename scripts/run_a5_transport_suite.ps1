param(
    [string]$Python = "python",
    [string]$Device = "cuda:0",
    [Nullable[double]]$ScaleDinoLambda = $null,
    [string]$A4S1Summary = "",
    [bool]$RunReplication = $true,
    [bool]$RunCapacity = $true,
    [bool]$RunScale = $true,
    [switch]$Force,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$BaseHead = "2e2db3533081bf56b67cccbdc01e52c29f02bad0"
$Logs = Join-Path $Root "artifacts\logs\a5_transport_suite_v1"
$Metrics = Join-Path $Root "artifacts\metrics"
$Frozen = Join-Path $Root "artifacts\configs\a5_transport_suite_v1"
$Preflight = Join-Path $Root "artifacts\metrics\a5_transport_preflight_v1"
$Audit = Join-Path $Root "artifacts\audit\a5_transport_suite_v1"
$BundleStage = Join-Path $Root "artifacts\audit\a5_transport_suite_results_v1"
$BundleZip = Join-Path $Root "artifacts\audit\a5_transport_suite_results_v1.zip"

$RunSeed7 = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a5_corr_v1_seed7"
$RunSeed13 = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a5_corr_v1_seed13"
$RunSeed23 = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a5_corr_v1_seed23"
$RunCapS = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a5_cap_s_v1_seed7"
$RunCapM = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a5_cap_m_v1_seed7"
$Run8k = Join-Path $Root "artifacts\runs\causal_scale_eap_a5_corr_v1_train8192_seed7"
$Run16k = Join-Path $Root "artifacts\runs\causal_scale_eap_a5_corr_v1_train16384_seed7"

$script:Steps = [System.Collections.Generic.List[object]]::new()
$script:SuiteStatus = "STARTED"
$script:ResolvedScaleLambda = $null
$script:ResolvedA4S1Summary = $null

function Ensure-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Reset-Path([string]$Path) {
    if (Test-Path $Path) {
        if (-not $Force) {
            throw "Output already exists: $Path. Re-run with -Force to replace A5-suite outputs."
        }
        Remove-Item $Path -Recurse -Force
    }
}

function Invoke-LoggedNative {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )
    Ensure-Directory $Logs
    $LogPath = Join-Path $Logs ("{0}.log" -f $Name)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host "$Executable $($Arguments -join ' ')"
    $started = Get-Date
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $code = $LASTEXITCODE
    $elapsed = ((Get-Date) - $started).TotalSeconds
    $ok = $AllowedExitCodes -contains $code
    $script:Steps.Add([pscustomobject]@{
        name = $Name
        exit_code = $code
        allowed = $ok
        elapsed_seconds = $elapsed
        log = (Resolve-Path $LogPath).Path
    })
    if (-not $ok) {
        throw "Step '$Name' failed with exit code $code. See $LogPath"
    }
    return $code
}

function Invoke-Python {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )
    return Invoke-LoggedNative -Name $Name -Executable $Python -Arguments $Arguments -AllowedExitCodes $AllowedExitCodes
}

function Read-Json([string]$Path) {
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Find-ScaleLambda {
    if ($null -ne $ScaleDinoLambda) {
        if (-not [double]::IsFinite([double]$ScaleDinoLambda) -or [double]$ScaleDinoLambda -le 0) {
            throw "-ScaleDinoLambda must be finite and positive."
        }
        return [double]$ScaleDinoLambda
    }

    $candidates = Get-ChildItem (Join-Path $Root "artifacts") -Recurse -Filter "selected_lambda.txt" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($candidate in $candidates) {
        $summary = Join-Path $candidate.Directory.FullName "summary.json"
        if (-not (Test-Path $summary)) { continue }
        try { $payload = Read-Json $summary } catch { continue }
        if (
            $payload.artifact_type -eq "a4_dinov3_relational_lambda_train_only_group_cv_v1" -and
            $payload.scope.public_train_only -eq $true -and
            $payload.promotion_ready -eq $true -and
            $payload.lambda_grid_boundary_hit -eq $false
        ) {
            $value = [double](Get-Content $candidate.FullName -Raw).Trim()
            if ([double]::IsFinite($value) -and $value -gt 0) {
                Write-Host "Using train-only A4 CV lambda=$value from $($candidate.FullName)"
                return $value
            }
        }
    }
    return $null
}

function Find-A4S1Summary([double]$Lambda) {
    if ($A4S1Summary) {
        $path = (Resolve-Path $A4S1Summary).Path
        $payload = Read-Json $path
        $observed = [double]$payload.training_config.representation_distillation_weight
        if ([math]::Abs($observed - $Lambda) -gt 1e-9) {
            throw "A4-S1 summary lambda=$observed differs from scale lambda=$Lambda"
        }
        return $path
    }

    $summaries = Get-ChildItem (Join-Path $Root "artifacts\runs") -Recurse -Filter "summary.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($candidate in $summaries) {
        try { $payload = Read-Json $candidate.FullName } catch { continue }
        try {
            $rep = [string]$payload.training_config.representation_supervision
            $weight = [double]$payload.training_config.representation_distillation_weight
            $params = [int]$payload.parameter_count
            $opened = [bool]$payload.official_test_opened
            $cachePath = [string]$payload.cache.manifest_path
            $transport = $false
            if ($null -ne $payload.model_architecture) { $transport = [bool]$payload.model_architecture.transport_enabled }
            if (
                $rep -eq "dinov3_local_relational" -and
                [math]::Abs($weight - $Lambda) -le 1e-9 -and
                $params -eq 355118 -and
                -not $opened -and
                -not $transport -and
                $cachePath -match "8192"
            ) {
                Write-Host "Auto-detected A4-S1 comparator: $($candidate.FullName)"
                return $candidate.FullName
            }
        } catch { continue }
    }
    return $null
}

function Copy-SmallArtifactTree([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) { return }
    Ensure-Directory $Destination
    Get-ChildItem $Source -Recurse -File | ForEach-Object {
        if ($_.Extension -in @(".pt", ".pth", ".ckpt")) { return }
        $relative = $_.FullName.Substring($Source.Length).TrimStart('\','/')
        $target = Join-Path $Destination $relative
        Ensure-Directory (Split-Path $target -Parent)
        Copy-Item $_.FullName $target -Force
    }
}

function Write-CheckpointInventory([string]$OutputCsv) {
    $rows = @()
    foreach ($dir in @($RunSeed7,$RunSeed13,$RunSeed23,$RunCapS,$RunCapM,$Run8k,$Run16k)) {
        if (-not (Test-Path $dir)) { continue }
        Get-ChildItem $dir -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @(".pt", ".pth", ".ckpt") } | ForEach-Object {
            $rows += [pscustomobject]@{
                path = $_.FullName.Substring($Root.Length + 1)
                bytes = $_.Length
                sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
            }
        }
    }
    $rows | Export-Csv $OutputCsv -NoTypeInformation -Encoding utf8
}

function Finalize-Bundle {
    try {
        if (Test-Path $BundleStage) { Remove-Item $BundleStage -Recurse -Force }
        Ensure-Directory $BundleStage
        Ensure-Directory (Join-Path $BundleStage "provenance")
        Ensure-Directory (Join-Path $BundleStage "logs")
        Ensure-Directory (Join-Path $BundleStage "artifacts")

        git rev-parse HEAD | Out-File (Join-Path $BundleStage "provenance\git_head.txt") -Encoding utf8
        git status --short | Out-File (Join-Path $BundleStage "provenance\git_status.txt") -Encoding utf8
        git log -25 --oneline --decorate | Out-File (Join-Path $BundleStage "provenance\git_log_25.txt") -Encoding utf8
        git diff -- src scripts configs tests | Out-File (Join-Path $BundleStage "provenance\working_diff.patch") -Encoding utf8

        Copy-SmallArtifactTree $Logs (Join-Path $BundleStage "logs")
        Copy-SmallArtifactTree $Preflight (Join-Path $BundleStage "artifacts\preflight")
        Copy-SmallArtifactTree $Frozen (Join-Path $BundleStage "artifacts\frozen_configs")
        Copy-SmallArtifactTree $Audit (Join-Path $BundleStage "artifacts\audit")
        Copy-SmallArtifactTree $RunSeed7 (Join-Path $BundleStage "artifacts\seed7")
        Copy-SmallArtifactTree $RunSeed13 (Join-Path $BundleStage "artifacts\seed13")
        Copy-SmallArtifactTree $RunSeed23 (Join-Path $BundleStage "artifacts\seed23")
        Copy-SmallArtifactTree $RunCapS (Join-Path $BundleStage "artifacts\cap_s")
        Copy-SmallArtifactTree $RunCapM (Join-Path $BundleStage "artifacts\cap_m")
        Copy-SmallArtifactTree $Run8k (Join-Path $BundleStage "artifacts\scale_8192")
        Copy-SmallArtifactTree $Run16k (Join-Path $BundleStage "artifacts\scale_16384")
        Write-CheckpointInventory (Join-Path $BundleStage "checkpoint_inventory.csv")

        $manifest = [ordered]@{
            artifact_type = "a5_transport_suite_results_bundle_v1"
            created_at = (Get-Date).ToUniversalTime().ToString("o")
            suite_status = $script:SuiteStatus
            preregistered_base_head = $BaseHead
            current_head = (git rev-parse HEAD).Trim()
            screen_dino_lambda = 4.0
            scale_dino_lambda = $script:ResolvedScaleLambda
            a4_s1_summary = $script:ResolvedA4S1Summary
            private_test_opened = $false
            steps = $script:Steps
        }
        $manifest | ConvertTo-Json -Depth 8 | Out-File (Join-Path $BundleStage "suite_manifest.json") -Encoding utf8

        Get-ChildItem $BundleStage -Recurse -File | ForEach-Object {
            [pscustomobject]@{
                path = $_.FullName.Substring($BundleStage.Length + 1)
                bytes = $_.Length
                sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
            }
        } | Export-Csv (Join-Path $BundleStage "inventory.csv") -NoTypeInformation -Encoding utf8

        if (Test-Path $BundleZip) { Remove-Item $BundleZip -Force }
        Compress-Archive -Path (Join-Path $BundleStage "*") -DestinationPath $BundleZip -Force
        $hash = (Get-FileHash $BundleZip -Algorithm SHA256).Hash.ToLower()
        Write-Host "`nA5 results bundle: $BundleZip" -ForegroundColor Green
        Write-Host "SHA256: $hash" -ForegroundColor Green
    } catch {
        Write-Warning "Could not finalize A5 result bundle: $($_.Exception.Message)"
    }
}

try {
    Reset-Path $Logs
    Reset-Path $Audit
    Reset-Path $Frozen
    Ensure-Directory $Logs
    Ensure-Directory $Metrics
    Ensure-Directory $Audit

    $head = (git rev-parse HEAD).Trim()
    git merge-base --is-ancestor $BaseHead $head 2>$null
    if ($LASTEXITCODE -ne 0 -and $head -ne $BaseHead) {
        Write-Warning "Current HEAD $head is not known to descend from preregistered base $BaseHead. Provenance will be recorded; verify manually before publication."
    }

    if (-not $SkipTests) {
        Invoke-Python "00_py_compile" @(
            "-m", "py_compile",
            "src/e_jepa_ttc/models/local_transport.py",
            "src/e_jepa_ttc/models/causal_scale_ttc.py",
            "src/e_jepa_ttc/training/causal_scale_eap.py",
            "scripts/diagnose_a5_transport_preflight.py",
            "scripts/diagnose_a5_corr_transport.py",
            "scripts/freeze_a5_suite_configs.py",
            "scripts/audit_a5_capacity_resolution.py",
            "scripts/audit_a5_scale_readiness.py",
            "scripts/summarize_a5_replication.py",
            "scripts/summarize_a5_capacity.py",
            "scripts/summarize_a5_scale.py",
            "scripts/train_causal_scale_eap_screen.py"
        ) | Out-Null
        Invoke-Python "01_pytest_a5" @(
            "-m", "pytest", "-q",
            "tests/unit/test_a5_local_transport.py",
            "tests/unit/test_causal_scale_ttc.py",
            "tests/unit/test_causal_scale_a4_dino_teacher.py",
            "tests/unit/test_dinov3_relational_distillation.py",
            "tests/unit/test_dinov3_relational_teacher_cache.py",
            "tests/unit/test_a4_dinov3_rgb_contract.py"
        ) | Out-Null
    }

    Invoke-Python "02_capacity_resolution_audit" @(
        "scripts/audit_a5_capacity_resolution.py",
        "--output", "artifacts/audit/a5_transport_suite_v1/capacity_resolution.json"
    ) | Out-Null
    Invoke-Python "03_scale_readiness" @(
        "scripts/audit_a5_scale_readiness.py",
        "--output", "artifacts/audit/a5_transport_suite_v1/scale_readiness.json"
    ) | Out-Null

    # Freeze screen configs at lambda=4. Scale configs, if possible, use only a
    # train-only CV lambda and are causally compared with A4-S1 at the same lambda.
    $script:ResolvedScaleLambda = Find-ScaleLambda
    $freezeArgs = @(
        "scripts/freeze_a5_suite_configs.py",
        "--output-dir", "artifacts/configs/a5_transport_suite_v1"
    )
    if ($RunScale -and $null -ne $script:ResolvedScaleLambda) {
        $freezeArgs += @("--include-scale", "--scale-dino-lambda", ([string]$script:ResolvedScaleLambda))
    }
    Invoke-Python "04_freeze_configs" $freezeArgs | Out-Null

    Reset-Path $Preflight
    $preflightCode = Invoke-Python "05_preflight" @(
        "scripts/diagnose_a5_transport_preflight.py",
        "--output-dir", "artifacts/metrics/a5_transport_preflight_v1",
        "--device", $Device,
        "--samples", "256",
        "--batch-size", "8",
        "--gradient-batches", "8"
    ) @(0,3)
    if ($preflightCode -ne 0) {
        throw "STOP::STOPPED_PREFLIGHT_NOT_AUTHORIZED"
    }

    Reset-Path $RunSeed7
    Invoke-Python "10_train_a5_seed7" @(
        "scripts/train_causal_scale_eap_screen.py",
        "--config", "artifacts/configs/a5_transport_suite_v1/seed7.yaml",
        "--output-dir", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7",
        "--device", $Device
    ) | Out-Null

    $A4Parent = "artifacts/runs/causal_scale_eap_screen_a4_dinov3_relational_rgb_v2_seed7/summary.json"
    if (-not (Test-Path $A4Parent)) { throw "Missing immutable A4 parent summary: $A4Parent" }
    $gateCode = Invoke-Python "11_gate_a5_seed7" @(
        "scripts/diagnose_a5_corr_transport.py",
        "--child-summary", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/summary.json",
        "--parent-summary", $A4Parent,
        "--output", "artifacts/audit/a5_transport_suite_v1/a5_seed7_gate.json"
    ) @(0,4)
    if ($gateCode -ne 0) {
        throw "STOP::STOPPED_A5_SEED7_MECHANISTIC_GATE"
    }

    if ($RunReplication) {
        Reset-Path $RunSeed13
        Invoke-Python "20_train_a5_seed13" @(
            "scripts/train_causal_scale_eap_screen.py",
            "--config", "artifacts/configs/a5_transport_suite_v1/seed13.yaml",
            "--output-dir", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13",
            "--device", $Device
        ) | Out-Null
        Reset-Path $RunSeed23
        Invoke-Python "21_train_a5_seed23" @(
            "scripts/train_causal_scale_eap_screen.py",
            "--config", "artifacts/configs/a5_transport_suite_v1/seed23.yaml",
            "--output-dir", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23",
            "--device", $Device
        ) | Out-Null
        $repCode = Invoke-Python "22_replication_summary" @(
            "scripts/summarize_a5_replication.py",
            "--summary", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/summary.json",
            "--summary", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed13/summary.json",
            "--summary", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed23/summary.json",
            "--output", "artifacts/audit/a5_transport_suite_v1/replication_summary.json"
        ) @(0,5)
        if ($repCode -ne 0) {
            throw "STOP::STOPPED_REPLICATION_GATE"
        }
    }

    if ($RunCapacity) {
        Reset-Path $RunCapS
        Invoke-Python "30_train_cap_s" @(
            "scripts/train_causal_scale_eap_screen.py",
            "--config", "artifacts/configs/a5_transport_suite_v1/cap_s.yaml",
            "--output-dir", "artifacts/runs/causal_scale_eap_screen_a5_cap_s_v1_seed7",
            "--device", $Device
        ) | Out-Null
        Reset-Path $RunCapM
        Invoke-Python "31_train_cap_m" @(
            "scripts/train_causal_scale_eap_screen.py",
            "--config", "artifacts/configs/a5_transport_suite_v1/cap_m.yaml",
            "--output-dir", "artifacts/runs/causal_scale_eap_screen_a5_cap_m_v1_seed7",
            "--device", $Device
        ) | Out-Null
        Invoke-Python "32_capacity_summary" @(
            "scripts/summarize_a5_capacity.py",
            "--base-summary", "artifacts/runs/causal_scale_eap_screen_a5_corr_v1_seed7/summary.json",
            "--cap-s-summary", "artifacts/runs/causal_scale_eap_screen_a5_cap_s_v1_seed7/summary.json",
            "--cap-m-summary", "artifacts/runs/causal_scale_eap_screen_a5_cap_m_v1_seed7/summary.json",
            "--output", "artifacts/audit/a5_transport_suite_v1/capacity_summary.json"
        ) | Out-Null
    }

    if ($RunScale) {
        if ($null -eq $script:ResolvedScaleLambda) {
            Write-Warning "No promotion-ready train-only A4 lambda CV was found. Skipping 8k/16k training; screen/replication/capacity results remain valid."
            throw "STOP_OK::COMPLETE_SCREEN_SCALE_SKIPPED_NO_LAMBDA"
        }
        $script:ResolvedA4S1Summary = Find-A4S1Summary ([double]$script:ResolvedScaleLambda)
        if ($null -eq $script:ResolvedA4S1Summary) {
            Write-Warning "No A4-S1 8192 summary at lambda=$script:ResolvedScaleLambda found. Skipping A5 scale to avoid a confounded transport claim."
            throw "STOP_OK::COMPLETE_SCREEN_SCALE_SKIPPED_NO_A4_S1_CONTROL"
        }

        $scale8Config = Join-Path $Frozen "scale_8192_seed7.yaml"
        if (-not (Test-Path $scale8Config)) {
            Write-Warning "8192 event+DINO manifests are not ready. Skipping scale training."
            throw "STOP_OK::COMPLETE_SCREEN_SCALE_SKIPPED_8K_NOT_READY"
        }
        Reset-Path $Run8k
        Invoke-Python "40_train_a5_8k" @(
            "scripts/train_causal_scale_eap_screen.py",
            "--config", "artifacts/configs/a5_transport_suite_v1/scale_8192_seed7.yaml",
            "--output-dir", "artifacts/runs/causal_scale_eap_a5_corr_v1_train8192_seed7",
            "--device", $Device
        ) | Out-Null
        $gate8 = Invoke-Python "41_gate_a5_8k_vs_a4_s1" @(
            "scripts/diagnose_a5_corr_transport.py",
            "--child-summary", "artifacts/runs/causal_scale_eap_a5_corr_v1_train8192_seed7/summary.json",
            "--parent-summary", $script:ResolvedA4S1Summary,
            "--output", "artifacts/audit/a5_transport_suite_v1/a5_8k_vs_a4_s1_gate.json"
        ) @(0,4)
        if ($gate8 -ne 0) {
            throw "STOP::STOPPED_8K_TRANSPORT_GATE"
        }

        $scale16Config = Join-Path $Frozen "scale_16384_seed7.yaml"
        if (Test-Path $scale16Config) {
            Reset-Path $Run16k
            Invoke-Python "50_train_a5_16k" @(
                "scripts/train_causal_scale_eap_screen.py",
                "--config", "artifacts/configs/a5_transport_suite_v1/scale_16384_seed7.yaml",
                "--output-dir", "artifacts/runs/causal_scale_eap_a5_corr_v1_train16384_seed7",
                "--device", $Device
            ) | Out-Null
            Invoke-Python "51_scale_16k_summary" @(
                "scripts/summarize_a5_scale.py",
                "--base-summary", "artifacts/runs/causal_scale_eap_a5_corr_v1_train8192_seed7/summary.json",
                "--scaled-summary", "artifacts/runs/causal_scale_eap_a5_corr_v1_train16384_seed7/summary.json",
                "--output", "artifacts/audit/a5_transport_suite_v1/scale_8k_to_16k_summary.json"
            ) | Out-Null
        } else {
            Write-Warning "16k event+DINO manifests do not exist yet; recorded as not ready and skipped."
        }
    }

    $script:SuiteStatus = "COMPLETE"
}
catch {
    $message = [string]$_.Exception.Message
    if ($message.StartsWith("STOP_OK::")) {
        $script:SuiteStatus = $message.Substring(9)
        Write-Warning $script:SuiteStatus
    } elseif ($message.StartsWith("STOP::")) {
        $script:SuiteStatus = $message.Substring(6)
        Write-Warning $script:SuiteStatus
    } else {
        $script:SuiteStatus = "FAILED: $message"
        Write-Error $_
    }
}
finally {
    Finalize-Bundle
}

if ($script:SuiteStatus -like "FAILED*") { exit 1 }
if ($script:SuiteStatus -like "STOPPED*") { exit 2 }
exit 0
