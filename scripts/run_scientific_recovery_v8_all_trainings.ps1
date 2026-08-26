<#
.SYNOPSIS
One-command Scientific Recovery V8 training orchestrator.

.DESCRIPTION
Runs every training stage allowed by the frozen V8 gates while keeping public
validation, private test, EvTTC test and CodaBench closed.  GPU training is never
mixed across scientific stages.  Within a stage at most MaxParallel=2 processes
run concurrently.  Every top-level command receives dedicated stdout/stderr logs;
individual V8 trainers also retain their own per-run logs and resumable state.
#>
[CmdletBinding()]
param(
    [ValidatePattern('^(cuda(?::\d+)?|cpu)$')][string]$Device = 'cuda',
    [ValidateRange(1,2)][int]$MaxParallel = 2,
    [string]$Protocol = 'configs/protocol/scientific_recovery_v8_temporal.json',
    [string]$Manifest = 'configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json',
    [string]$EapRoot = 'E:\eAP_dataset',
    [string]$GarlTtcRoot = 'E:\GarlTTC_dataset',
    [switch]$SkipFocusedTests,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
if (-not $DryRun) {
    $porcelain = & git status --porcelain
    if (-not [string]::IsNullOrWhiteSpace($porcelain)) {
        throw 'scientific execution requires a clean Git worktree'
    }
}
$env:PYTHONUNBUFFERED = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$ForbiddenScientificEnv = @(
    'DINO_NUM_CHUNKS',
    'DINO_CHUNK_INDEX',
    'DINO_START_ROW',
    'DINO_END_ROW',
    'DINO_ALLOW_PARTIAL_CACHE',
    'ALLOW_DIRTY_MATERIALIZE',
    'ALLOW_DIRTY',
    'ALLOW_PARTIAL'
)
foreach ($name in $ForbiddenScientificEnv) {
    $present = [Environment]::GetEnvironmentVariable($name)
    if (-not [string]::IsNullOrWhiteSpace($present)) {
        throw "scientific execution forbids bypass environment variable $name"
    }
}
if ($Device -match '^cuda' -and -not $DryRun) {
    $torchProbe = 'import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info()[0] // (1024 * 1024))'
    $reported = & uv run --no-sync python -c $torchProbe
    $freeMiB = 0
    if ($LASTEXITCODE -ne 0 -or -not [int]::TryParse((@($reported) | Select-Object -Last 1).ToString().Trim(), [ref]$freeMiB) -or $freeMiB -le 0) {
        throw 'scientific CUDA execution requires a working GPU with measurable free memory'
    }
    Write-Host "GPU preflight: ${freeMiB} MiB free; MaxParallel=$MaxParallel (frozen V8 GPU bound)" -ForegroundColor Cyan
}
$BaseRoot=Join-Path $RepoRoot 'artifacts/scientific_recovery_v8'
$ResultsRoot=Join-Path $BaseRoot 'results'
$MasterLog=Join-Path $BaseRoot 'master_logs'
$MasterState=Join-Path $BaseRoot 'master_state.json'
New-Item -ItemType Directory -Force $MasterLog,$ResultsRoot | Out-Null
$PsRuntime = $PSVersionTable.PSVersion.ToString()
Write-Host "Scientific Recovery V8 orchestrator: PowerShell $PsRuntime" -ForegroundColor Cyan
Write-Host "Master logs: $MasterLog"
Write-Host "Master state: $MasterState"
Write-Host "Results root: $ResultsRoot"
Write-Host "Heartbeats every 15s while a child runs; stdout/stderr still go to master_logs." -ForegroundColor DarkCyan
$script:HeartbeatSeen = @{}
$script:HeartbeatIntervalMs = 15000

function Format-Elapsed([datetime]$Started) {
    $span = (Get-Date) - $Started
    return ('{0:00}:{1:00}:{2:00}' -f [int][math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds)
}
function Write-Console([string]$Message, [string]$Color = 'Gray') {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message) -ForegroundColor $Color
}
function Write-State([string]$Stage,[string]$Status,[string]$Detail='') {
    $value=[ordered]@{timestamp=(Get-Date).ToUniversalTime().ToString('o');stage=$Stage;status=$Status;detail=$Detail;device=$Device;max_parallel=$MaxParallel;sealed_evaluation='closed'}
    $tmp="$MasterState.tmp"; $value|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $tmp -Encoding utf8; Move-Item -Force $tmp $MasterState
    $suffix = ''
    if (-not [string]::IsNullOrWhiteSpace($Detail)) { $suffix = " :: $Detail" }
    Write-Console "STAGE $Stage -> $Status$suffix" 'Cyan'
}
function Get-LogTail([string]$Path, [int]$Count = 4) {
    $rows = New-Object 'System.Collections.Generic.List[string]'
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return $rows
    }
    foreach ($line in @(Get-Content -LiteralPath $Path -Tail $Count -ErrorAction SilentlyContinue)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$line)) {
            $rows.Add([string]$line)
        }
    }
    # Unary comma keeps the List intact under StrictMode (empty/one-item unwrap).
    return ,$rows
}
function Get-JsonNote($Object, [string]$Name) {
    if ($null -eq $Object) { return $null }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}
function Write-GpuSnapshot {
    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvsmi) { return }
    $gpu = & nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    if (-not [string]::IsNullOrWhiteSpace([string]$gpu)) {
        Write-Host ("  gpu: {0}" -f ((@($gpu) | ForEach-Object { $_.ToString().Trim() }) -join ' | '))
    }
}
function Write-TrainerProgressSnapshot {
    $dirs = @(
        (Join-Path $ResultsRoot 'runs'),
        (Join-Path $ResultsRoot 'router'),
        (Join-Path $BaseRoot 'jepa')
    )
    $files = New-Object 'System.Collections.Generic.List[object]'
    foreach ($dir in $dirs) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        foreach ($item in @(Get-ChildItem -LiteralPath $dir -Filter 'progress.json' -Recurse -ErrorAction SilentlyContinue)) {
            if ($null -ne $item) { $files.Add($item) }
        }
    }
    if ($files.Count -eq 0) { return }
    $newest = @($files | Sort-Object LastWriteTime -Descending | Select-Object -First 4)
    foreach ($file in $newest) {
        if ($null -eq $file) { continue }
        try {
            $json = Get-Content -Raw -LiteralPath $file.FullName -ErrorAction Stop | ConvertFrom-Json
        } catch {
            continue
        }
        $epoch = Get-JsonNote $json 'epoch'
        $total = Get-JsonNote $json 'configured_epochs'
        $elapsedSec = Get-JsonNote $json 'elapsed_seconds'
        $status = Get-JsonNote $json 'status'
        $best = Get-JsonNote $json 'best_epoch'
        $eta = '-'
        if ($null -ne $epoch -and $null -ne $total -and [int]$epoch -gt 0 -and [int]$total -gt [int]$epoch -and $null -ne $elapsedSec -and [double]$elapsedSec -gt 0) {
            $remain = [int](([double]$elapsedSec / [int]$epoch) * ([int]$total - [int]$epoch))
            $eta = '{0:00}:{1:00}:{2:00}' -f [int][math]::Floor($remain / 3600), [int](($remain % 3600) / 60), ($remain % 60)
        }
        $rel = [string]$file.FullName
        if ($rel.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rel = $rel.Substring($RepoRoot.Length).TrimStart('\', '/')
        }
        Write-Host ("  progress {0} status={1} epoch={2}/{3} best={4} eta~{5}" -f $rel, $status, $epoch, $total, $best, $eta)
    }
}
function Write-ProcessHeartbeat($Job) {
    if ($null -eq $Job -or $Job.Dry) { return }
    $pidValue = '-'
    if ($null -ne $Job.Process) { $pidValue = $Job.Process.Id }
    Write-Console ("LIVE {0} elapsed={1} pid={2}" -f $Job.Label, (Format-Elapsed $Job.StartedAt), $pidValue) 'Yellow'
    Write-GpuSnapshot
    Write-TrainerProgressSnapshot
    foreach ($kind in @('Stdout', 'Stderr')) {
        $path = [string]$Job.$kind
        $lines = Get-LogTail $path 4
        if ($null -eq $lines -or $lines.Count -eq 0) { continue }
        $finger = (@($lines) -join '|')
        $key = "$($Job.Label):$kind"
        if ($script:HeartbeatSeen[$key] -eq $finger) { continue }
        $script:HeartbeatSeen[$key] = $finger
        foreach ($line in $lines) {
            Write-Host ("  {0}: {1}" -f $kind.ToLower(), $line)
        }
    }
}
function Quote-NativeArgument([string]$Value) {
    if($null -eq $Value){ throw 'Quote-NativeArgument received a NULL value.' }
    if($Value.Length -eq 0){ return '""' }
    if($Value -notmatch '[\s"]'){ return $Value }
    # Start-Process joins/parses a command line on Windows. Quote arguments that
    # contain whitespace explicitly; the repository path contains spaces.
    return '"' + ($Value -replace '"','\"') + '"'
}
function ConvertTo-NativeArgumentLine([object[]]$ArgumentVector) {
    $clean=@()
    foreach($item in @($ArgumentVector)){
        if($null -eq $item){ throw 'A child-process argument is NULL.' }
        $clean += [string]$item
    }
    return ($clean | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
}
function Quote-Cmd([object[]]$ArgumentVector) {
    return ConvertTo-NativeArgumentLine $ArgumentVector
}
function Start-LoggedProcess([string]$Label,[string]$File,[object[]]$ArgumentVector) {
    if([string]::IsNullOrWhiteSpace($Label)){ throw 'Process label must not be empty.' }
    if([string]::IsNullOrWhiteSpace($File)){ throw "Process '$Label' has an empty executable." }

    $cleanArgs=@()
    foreach($item in @($ArgumentVector)){
        if($null -eq $item){
            throw "Process '$Label' has a NULL child-process argument."
        }
        $cleanArgs += [string]$item
    }
    $argumentLine=ConvertTo-NativeArgumentLine $cleanArgs

    $dir=Join-Path $MasterLog $Label
    New-Item -ItemType Directory -Force $dir | Out-Null
    $stdout=Join-Path $dir 'stdout.log'
    $stderr=Join-Path $dir 'stderr.log'
    $command=Join-Path $dir 'command.txt'
    @(
        "START_UTC=$((Get-Date).ToUniversalTime().ToString('o'))",
        "CWD=$RepoRoot",
        "EXECUTABLE=$File",
        "ARG_COUNT=$($cleanArgs.Count)",
        "COMMAND=$File $argumentLine"
    ) | Add-Content -LiteralPath $command -Encoding utf8

    if($DryRun){
        Write-Host "[DRY] $Label :: $File $argumentLine"
        return [pscustomobject]@{
            Label=$Label;Process=$null;Stdout=$stdout;Stderr=$stderr;Command=$command;Dry=$true;StartedAt=(Get-Date)
        }
    }

    Write-Console "START $Label" 'Green'
    Write-Host "  command: $File $argumentLine"
    Write-Host "  stdout: $stdout"
    Write-Host "  stderr: $stderr"

    if($cleanArgs.Count -eq 0){
        $p=Start-Process -FilePath $File `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
    } else {
        # Pass one already-quoted command-line string. This avoids Windows
        # PowerShell 5.1/Start-Process treating an array containing an empty or
        # automatic $Args value as an invalid -ArgumentList.
        $p=Start-Process -FilePath $File `
            -ArgumentList $argumentLine `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
    }
    # Accessing Handle before WaitForExit is required on Windows PowerShell 5.1
    # or ExitCode stays $null and `$code -ne 0` treats a successful child as failure.
    $null = $p.Handle
    Write-Host "  pid: $($p.Id)"

    return [pscustomobject]@{
        Label=$Label;Process=$p;Stdout=$stdout;Stderr=$stderr;Command=$command;Dry=$false;StartedAt=(Get-Date)
    }
}
function Wait-LoggedProcess($Job) {
    if($Job.Dry){ return }
    $proc = $Job.Process
    if ($null -eq $proc) {
        throw "Process '$($Job.Label)' was not started."
    }
    $null = $proc.Handle
    Write-ProcessHeartbeat $Job
    while (-not $proc.HasExited) {
        $null = $proc.Handle
        $finished = $proc.WaitForExit($script:HeartbeatIntervalMs)
        if ($finished) { break }
        $proc.Refresh()
        Write-ProcessHeartbeat $Job
    }
    $null = $proc.Handle
    $proc.WaitForExit()
    $proc.Refresh()
    $code = $proc.ExitCode
    @(
        "END_UTC=$((Get-Date).ToUniversalTime().ToString('o'))",
        "EXIT_CODE=$code"
    ) | Add-Content -LiteralPath $Job.Command -Encoding utf8
    $elapsed = Format-Elapsed $Job.StartedAt
    if ($null -eq $code) {
        throw "Process '$($Job.Label)' exited without reporting an exit code. See $($Job.Stderr)"
    }
    if ([int]$code -ne 0) {
        Write-Console "FAIL $($Job.Label) exit=$code elapsed=$elapsed" 'Red'
        $errTail = Get-LogTail $Job.Stderr 30
        foreach ($line in $errTail) { Write-Host "  stderr: $line" }
        $outTail = Get-LogTail $Job.Stdout 10
        foreach ($line in $outTail) { Write-Host "  stdout: $line" }
        throw "Process '$($Job.Label)' failed with exit $code. See $($Job.Stderr)"
    }
    Write-Console "DONE $($Job.Label) exit=$code elapsed=$elapsed" 'Green'
    $doneTail = Get-LogTail $Job.Stdout 3
    foreach ($line in $doneTail) { Write-Host "  stdout: $line" }
}
function Invoke-Logged([string]$Label,[string]$File,[object[]]$ArgumentVector){
    $job=Start-LoggedProcess $Label $File $ArgumentVector
    Wait-LoggedProcess $job
}
function Invoke-Wave([array]$Specs){
    $active=@(); foreach($spec in $Specs){
        while($active.Count -ge $MaxParallel){ Wait-LoggedProcess $active[0]; if($active.Count -eq 1){$active=@()}else{$active=$active[1..($active.Count-1)]} }
        $active+=Start-LoggedProcess $spec.Label $spec.File $spec.Args
    }
    foreach($job in $active){ Wait-LoggedProcess $job }
}
function UvArgs([string[]]$Tail){ return @('run','--no-sync','python')+$Tail }
function Read-SignedJson([string]$Path){ if(-not(Test-Path -LiteralPath $Path)){throw "Missing artifact: $Path"}; return Get-Content -Raw -LiteralPath $Path|ConvertFrom-Json }
function Test-CandidatePassed([string]$CandidateId){
    $a=Read-SignedJson (Join-Path $ResultsRoot 'aggregate_seed7.json'); foreach($x in $a.candidate_results){if($x.candidate_id -eq $CandidateId){return [bool]$x.passed}}; return $false
}

function Invoke-DryRunRemainder {
    Write-Host ""
    Write-Host "[DRY-GATE] No artifacts are produced in -DryRun mode." -ForegroundColor Yellow
    Write-Host "[DRY-GATE] The commands below show every conditional branch without claiming any gate passed." -ForegroundColor Yellow

    # B3: shown only as a possible branch; real execution requires signed B1 pass.
    Write-Host "[DRY-GATE] B3 PAIR20-2 would run only if B1_TIMEVOL20_3 passes." -ForegroundColor DarkYellow
    Invoke-Logged '32_temporal_b3_pair20_2__CONDITIONAL' 'uv' (UvArgs @(
        'scripts/run_scientific_recovery_v8_temporal.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--results-root',$ResultsRoot,
        '--arm','pair20_2',
        '--device',$Device,
        '--max-parallel',[string]$MaxParallel,
        '--eap-root',$EapRoot,
        '--garlttc-root',$GarlTtcRoot
    ))
    Invoke-Logged '33_aggregate_seed7_after_b3__CONDITIONAL' 'uv' (UvArgs @(
        'scripts/aggregate_scientific_recovery_v8.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--results-root',$ResultsRoot,
        '--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),
        '--resamples','5000'
    ))

    Invoke-Logged '40_primary_arm_aggregates' 'uv' (UvArgs @(
        'scripts/build_scientific_recovery_v8_primary_aggregates.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--aggregate',(Join-Path $ResultsRoot 'aggregate_seed7.json'),
        '--output-dir',(Join-Path $BaseRoot 'arm_aggregates')
    ))
    Invoke-Logged '41_c1_opening' 'uv' (UvArgs @(
        'scripts/build_scientific_recovery_v8_c1_opening.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--results-root',$BaseRoot
    ))

    Write-Host "[DRY-GATE] C1 GATED-EXP6-3 would run only if signed C1-opening evidence exists." -ForegroundColor DarkYellow
    Invoke-Logged '42_adaptive_c1__CONDITIONAL' 'uv' (UvArgs @(
        'scripts/run_scientific_recovery_v8_adaptive.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--evidence-root',$BaseRoot,
        '--results-root',$ResultsRoot,
        '--device',$Device,
        '--max-parallel',[string]$MaxParallel
    ))
    Invoke-Logged '43_aggregate_seed7_after_c1__CONDITIONAL' 'uv' (UvArgs @(
        'scripts/aggregate_scientific_recovery_v8.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--results-root',$ResultsRoot,
        '--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),
        '--resamples','5000'
    ))

    # JEPA always runs in the real DAG. In dry-run the winner/cache does not exist,
    # so use an explicit placeholder rather than trying to read a nonexistent artifact.
    $dryJepaRef = Join-Path $BaseRoot 'jepa/downstream_reference.json'
    $dryLowLabel = Join-Path $BaseRoot 'jepa/low_label_subsets.json'
    $dryCacheManifest = '<CACHE_MANIFEST_FROM_SIGNED_DOWNSTREAM_REFERENCE>'

    Invoke-Logged '50_jepa_reference' 'uv' (UvArgs @(
        'scripts/build_scientific_recovery_v8_jepa_reference.py',
        '--aggregate',(Join-Path $ResultsRoot 'aggregate_seed7.json'),
        '--protocol',$Protocol,
        '--output',$dryJepaRef
    ))
    Invoke-Logged '51_low_label_freeze' 'uv' (UvArgs @(
        'scripts/freeze_scientific_recovery_v8_low_label_subsets.py',
        '--winner-artifact',$dryJepaRef,
        '--cache-manifest',$dryCacheManifest,
        '--protocol',$Protocol,
        '--output',$dryLowLabel
    ))

    $dryJepaSpecs=@()
    foreach($fold in 0..2){
        $dryJepaSpecs += [pscustomobject]@{
            Label="52_jepa_seed7_fold$fold"
            File='uv'
            Args=(UvArgs @(
                'scripts/run_scientific_recovery_v8_jepa_attribution.py',
                '--config-dir','configs/experiment/scientific_recovery_v8_jepa',
                '--cache-manifest',$dryCacheManifest,
                '--winner-artifact',$dryJepaRef,
                '--low-label-manifest',$dryLowLabel,
                '--protocol',$Protocol,
                '--output-root',(Join-Path $BaseRoot 'jepa'),
                '--device',$Device,
                '--folds',[string]$fold
            ))
        }
    }
    Invoke-Wave $dryJepaSpecs
    Invoke-Logged '53_jepa_aggregate_seed7' 'uv' (UvArgs @(
        'scripts/aggregate_scientific_recovery_v8_jepa.py',
        '--results-root',(Join-Path $BaseRoot 'jepa'),
        '--output',(Join-Path $BaseRoot 'jepa/aggregate_seed7.json'),
        '--seed','7'
    ))

    Write-Host "[DRY-GATE] JEPA seeds 13/23 would run only if seed-7 JEPA is causally positive." -ForegroundColor DarkYellow
    foreach($seed in 13,23){
        Invoke-Logged "54_jepa_clone_seed${seed}__CONDITIONAL" 'uv' (UvArgs @(
            'scripts/clone_scientific_recovery_v8_jepa_replication_configs.py',
            '--seed',[string]$seed,
            '--output-dir',(Join-Path $BaseRoot "jepa/configs_seed$seed")
        ))
    }

    $dryRepSpecs=@()
    foreach($seed in 13,23){
        foreach($fold in 0..2){
            $dryRepSpecs += [pscustomobject]@{
                Label="55_jepa_seed${seed}_fold${fold}__CONDITIONAL"
                File='uv'
                Args=(UvArgs @(
                    'scripts/run_scientific_recovery_v8_jepa_attribution.py',
                    '--config-dir',(Join-Path $BaseRoot "jepa/configs_seed$seed"),
                    '--cache-manifest',$dryCacheManifest,
                    '--winner-artifact',$dryJepaRef,
                    '--low-label-manifest',$dryLowLabel,
                    '--protocol',$Protocol,
                    '--output-root',(Join-Path $BaseRoot 'jepa'),
                    '--device',$Device,
                    '--folds',[string]$fold
                ))
            }
        }
    }
    Invoke-Wave $dryRepSpecs

    foreach($seed in 13,23){
        Invoke-Logged "56_jepa_aggregate_seed${seed}__CONDITIONAL" 'uv' (UvArgs @(
            'scripts/aggregate_scientific_recovery_v8_jepa.py',
            '--results-root',(Join-Path $BaseRoot 'jepa'),
            '--output',(Join-Path $BaseRoot "jepa/aggregate_seed$seed.json"),
            '--seed',[string]$seed
        ))
    }

    # The multiseed runner is itself gate-aware in real execution.
    Invoke-Logged '60_downstream_multiseed' 'uv' (UvArgs @(
        'scripts/run_scientific_recovery_v8_multiseed_replication.py',
        '--protocol',$Protocol,
        '--manifest',$Manifest,
        '--results-root',$BaseRoot,
        '--device',$Device,
        '--max-parallel',[string]$MaxParallel
    ))

    if(Test-Path -LiteralPath 'scripts/build_scientific_recovery_v8_report.py'){
        Invoke-Logged '70_report' 'uv' (UvArgs @('scripts/build_scientific_recovery_v8_report.py'))
    }

    Write-State 'dry_run' 'completed' 'All unconditional and conditional commands were rendered; no scientific gate was evaluated.'
    Write-Host ""
    Write-Host "V8 DRY-RUN DAG COMPLETED. No training/artifact-producing command was executed." -ForegroundColor Green
    Write-Host "Conditional branches were shown for inspection only." -ForegroundColor Yellow
}

Write-State 'preflight' 'running'
# Verify committed frozen inputs. Do not regenerate: freeze() writes git_branch from
# the current checkout and would dirty a clean scientific worktree.
Invoke-Logged '00_verify_freeze' 'uv' (UvArgs @('scripts/freeze_scientific_recovery_v8_configs.py','--protocol',$Protocol,'--verify'))
if (-not $DryRun) {
    $porcelainAfterVerify = & git status --porcelain
    if (-not [string]::IsNullOrWhiteSpace($porcelainAfterVerify)) {
        throw 'scientific execution requires a clean Git worktree after freeze verification'
    }
}
if(-not $SkipFocusedTests){
    Invoke-Logged '02_focused_tests' 'uv' @('run','--no-sync','pytest','-q','tests/unit/test_scientific_recovery_v8_temporal.py','tests/unit/test_scientific_recovery_v8_router.py','tests/unit/test_scientific_recovery_v8_autopsy.py','tests/unit/test_scientific_recovery_v8_aggregate.py','tests/unit/test_scientific_recovery_v8_jepa.py','tests/unit/test_scientific_recovery_v8_jepa_attribution.py','tests/unit/test_scientific_recovery_v8_jobs.py','tests/integration/test_scientific_recovery_v8_trainer_smoke.py','tests/integration/test_scientific_recovery_v8_router_smoke.py','tests/integration/test_scientific_recovery_v8_jepa_smoke.py')
}
Write-State 'preflight' 'completed'

# A: no optimizer steps. Materialize exact V4 inputs, replay A5/C2F, bind frozen Garl OOF, aggregate.
Write-State 'autopsy' 'running'
$AInput=Join-Path $BaseRoot 'autopsy/inputs'; $AReplay=Join-Path $BaseRoot 'autopsy/replays'; $AOut=Join-Path $BaseRoot 'autopsy/mechanism_autopsy.json'
$A5ReplayManifest=Join-Path $AReplay 'a5/manifest.json'
$C2FReplayManifest=Join-Path $AReplay 'c2f/manifest.json'
$GarlReplayManifest=Join-Path $AReplay 'garl/manifest.json'
$ReuseAutopsyReplays = $false
if(-not $DryRun){
    # Existence/signature of CSVs is not producer identity. Reuse only when each
    # replay records git_commit=HEAD and git_dirty=false. Current pre-repair
    # manifests lack those fields and must be recomputed.
    & uv run --no-sync python scripts/assert_scientific_recovery_v8_autopsy_replay_reusable.py `
        --a5-manifest $A5ReplayManifest `
        --c2f-manifest $C2FReplayManifest `
        --garl-manifest $GarlReplayManifest
    if($LASTEXITCODE -eq 0){
        $ReuseAutopsyReplays = $true
        Write-Host 'Reusing autopsy replays bound to the current clean implementation commit.'
    }else{
        Write-Host 'Autopsy replays are missing, unsigned, or not bound to this HEAD; recomputing stages 10-12.'
    }
}
if($ReuseAutopsyReplays){
    Write-Host 'Skipping stages 10-12; stage 13 will still verify signed CSV hashes.'
}else{
    Invoke-Logged '10_autopsy_materialize' 'uv' (UvArgs @('scripts/materialize_scientific_recovery_v8_autopsy_inputs.py','--protocol',$Protocol,'--output-dir',$AInput,'--eap-root',$EapRoot,'--garlttc-root',$GarlTtcRoot))
    Invoke-Logged '11_autopsy_replay_a5_c2f' 'uv' (UvArgs @('scripts/replay_scientific_recovery_v8_mechanisms.py','--protocol',$Protocol,'--replay-input-root',$AInput,'--output-dir',$AReplay,'--models','a5','c2f','--device',$Device))
    Invoke-Logged '12_autopsy_garl_bind' 'uv' (UvArgs @('scripts/materialize_scientific_recovery_v8_garl_autopsy_comparator.py','--protocol',$Protocol,'--output-dir',(Join-Path $AReplay 'garl')))
}
# Stage 13 re-verifies signed CSV hashes AND producer git identity vs HEAD.
Invoke-Logged '13_autopsy_aggregate' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8_autopsy.py','--a5-manifest',$A5ReplayManifest,'--c2f-manifest',$C2FReplayManifest,'--garl-manifest',$GarlReplayManifest,'--protocol',$Protocol,'--output',$AOut,'--bootstrap-resamples','5000'))
Write-State 'autopsy' 'completed'

# R: prospective nested router. Internally executes 18 inner + 6 outer expert fits with max two concurrent.
Write-State 'router' 'running'
Invoke-Logged '20_router_nested' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_nested_router.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--device',$Device,'--max-parallel',[string]$MaxParallel,'--execute'))
Write-State 'router' 'completed'

# B1+B2: fixed temporal frontends. Cache materialization is automatic and train-only.
Write-State 'temporal' 'running'
Invoke-Logged '30_temporal_b1_b2' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_temporal.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--arm','all','--device',$Device,'--max-parallel',[string]$MaxParallel,'--eap-root',$EapRoot,'--garlttc-root',$GarlTtcRoot))
Invoke-Logged '31_aggregate_seed7_initial' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--resamples','5000'))
if($DryRun){
    Invoke-DryRunRemainder
    return
}
# B3 is automatically opened only by a signed B1 pass.
if(Test-CandidatePassed 'B1_TIMEVOL20_3'){
    Invoke-Logged '32_temporal_b3_pair20_2' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_temporal.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--arm','pair20_2','--device',$Device,'--max-parallel',[string]$MaxParallel,'--eap-root',$EapRoot,'--garlttc-root',$GarlTtcRoot))
    Invoke-Logged '33_aggregate_seed7_after_b3' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--resamples','5000'))
}
Write-State 'temporal' 'completed'

# Build typed arm evidence and the route-specific C1 opening decision.
Invoke-Logged '40_primary_arm_aggregates' 'uv' (UvArgs @('scripts/build_scientific_recovery_v8_primary_aggregates.py','--protocol',$Protocol,'--manifest',$Manifest,'--aggregate',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--output-dir',(Join-Path $BaseRoot 'arm_aggregates')))
Invoke-Logged '41_c1_opening' 'uv' (UvArgs @('scripts/build_scientific_recovery_v8_c1_opening.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$BaseRoot))
$C1Open=(Get-ChildItem -LiteralPath (Join-Path $BaseRoot 'adaptive_gate') -Filter 'c1_opening_*.json' -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
if($C1Open){
    Write-State 'adaptive' 'running'
    Invoke-Logged '42_adaptive_c1' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_adaptive.py','--protocol',$Protocol,'--manifest',$Manifest,'--evidence-root',$BaseRoot,'--results-root',$ResultsRoot,'--device',$Device,'--max-parallel',[string]$MaxParallel))
    Invoke-Logged '43_aggregate_seed7_after_c1' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--resamples','5000'))
    Write-State 'adaptive' 'completed'
}else{ Write-State 'adaptive' 'blocked_by_gate' 'No signed C1 opening evidence.' }

# D: JEPA attribution always runs. Build the immutable reference, then freeze low-label IDs before D0--D4.
Write-State 'jepa' 'running'
$JepaRef=Join-Path $BaseRoot 'jepa/downstream_reference.json'; $LowLabel=Join-Path $BaseRoot 'jepa/low_label_subsets.json'
Invoke-Logged '50_jepa_reference' 'uv' (UvArgs @('scripts/build_scientific_recovery_v8_jepa_reference.py','--aggregate',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--protocol',$Protocol,'--output',$JepaRef))
$JepaRefValue=Read-SignedJson $JepaRef; $CacheManifest=Join-Path $RepoRoot ([string]$JepaRefValue.cache_manifest.path)
Invoke-Logged '51_low_label_freeze' 'uv' (UvArgs @('scripts/freeze_scientific_recovery_v8_low_label_subsets.py','--winner-artifact',$JepaRef,'--cache-manifest',$CacheManifest,'--protocol',$Protocol,'--output',$LowLabel))
$jepaSpecs=@(); foreach($fold in 0..2){ $jepaSpecs += [pscustomobject]@{Label="52_jepa_seed7_fold$fold";File='uv';Args=(UvArgs @('scripts/run_scientific_recovery_v8_jepa_attribution.py','--config-dir','configs/experiment/scientific_recovery_v8_jepa','--cache-manifest',$CacheManifest,'--winner-artifact',$JepaRef,'--low-label-manifest',$LowLabel,'--protocol',$Protocol,'--output-root',(Join-Path $BaseRoot 'jepa'),'--device',$Device,'--folds',[string]$fold))} }
Invoke-Wave $jepaSpecs
Invoke-Logged '53_jepa_aggregate_seed7' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8_jepa.py','--results-root',(Join-Path $BaseRoot 'jepa'),'--output',(Join-Path $BaseRoot 'jepa/aggregate_seed7.json'),'--seed','7'))
$JepaAgg=Read-SignedJson (Join-Path $BaseRoot 'jepa/aggregate_seed7.json')
# Positive JEPA claims get full D0-D4 optimization replication for seeds 13/23, same frozen low-label IDs.
if([bool]$JepaAgg.jepa_causally_positive){
    foreach($seed in 13,23){ Invoke-Logged "54_jepa_clone_seed$seed" 'uv' (UvArgs @('scripts/clone_scientific_recovery_v8_jepa_replication_configs.py','--seed',[string]$seed,'--output-dir',(Join-Path $BaseRoot "jepa/configs_seed$seed"))) }
    $repSpecs=@(); foreach($seed in 13,23){ foreach($fold in 0..2){ $repSpecs += [pscustomobject]@{Label="55_jepa_seed${seed}_fold$fold";File='uv';Args=(UvArgs @('scripts/run_scientific_recovery_v8_jepa_attribution.py','--config-dir',(Join-Path $BaseRoot "jepa/configs_seed$seed"),'--cache-manifest',$CacheManifest,'--winner-artifact',$JepaRef,'--low-label-manifest',$LowLabel,'--protocol',$Protocol,'--output-root',(Join-Path $BaseRoot 'jepa'),'--device',$Device,'--folds',[string]$fold))} } }
    Invoke-Wave $repSpecs
    foreach($seed in 13,23){ Invoke-Logged "56_jepa_aggregate_seed$seed" 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8_jepa.py','--results-root',(Join-Path $BaseRoot 'jepa'),'--output',(Join-Path $BaseRoot "jepa/aggregate_seed$seed.json"),'--seed',[string]$seed)) }
}
Write-State 'jepa' 'completed'

# E training-only replication: only if a new downstream seed-7 arm actually passed the TTC gate.
Write-State 'multiseed_replication' 'running'
Invoke-Logged '60_downstream_multiseed' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_multiseed_replication.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$BaseRoot,'--device',$Device,'--max-parallel',[string]$MaxParallel))
Write-State 'multiseed_replication' 'completed'

# Regenerable local report. No public validation/test/CodaBench is opened here.
if(Test-Path -LiteralPath 'scripts/build_scientific_recovery_v8_report.py'){
    try { Invoke-Logged '70_report' 'uv' (UvArgs @('scripts/build_scientific_recovery_v8_report.py')) } catch { Write-Warning $_ }
}
Write-State 'done' 'completed' 'All gate-authorized V8 trainings and local scientific aggregations finished; sealed external evaluation remains closed.'
Write-Host "V8 TRAINING DAG COMPLETED. Master logs: $MasterLog" -ForegroundColor Green
Write-Host "Seed-7 aggregate: $(Join-Path $ResultsRoot 'aggregate_seed7.json')"
Write-Host "JEPA aggregate: $(Join-Path $BaseRoot 'jepa/aggregate_seed7.json')"
Write-Host "No public validation/private test/EvTTC test/CodaBench was opened." -ForegroundColor Yellow
