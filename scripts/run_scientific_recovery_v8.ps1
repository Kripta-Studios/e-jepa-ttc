<#
.SYNOPSIS
Run the signed Scientific Recovery V8 DAG without touching sealed evaluation.

.DESCRIPTION
Stages are executed in protocol order.  Training work inside a stage is
delegated to the signed Python V8 runners with a bounded real concurrency.
The default is two workers on the 12-GB RTX 5070 Ti Laptop GPU.  Three workers
require an explicit, measured memory preflight; this script never silently
oversubscribes the device.

The monitor has no evaluator and no access to public validation, private test,
EvTTC test, or CodaBench.  It only inventories local run files and signed
summaries, then atomically writes a signed machine state and a human-readable
status document.
#>

[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'Run')]
    [ValidateSet('preflight', 'autopsy', 'router', 'temporal', 'adaptive', 'jepa', 'multiseed_replication', 'robustness', 'export', 'package', 'screen', 'all')]
    [string]$Stage = 'screen',

    [Parameter(ParameterSetName = 'Run')]
    [ValidatePattern('^(cuda(?::\d+)?|cpu)$')]
    [string]$Device = 'cuda',

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(1, 3)]
    [int]$MaxParallel = 2,

    [Parameter(ParameterSetName = 'Run')]
    [switch]$EnableThreeWayConcurrency,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(256, 16384)]
    [int]$PerTrainingMemoryMiB = 3500,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(0, 4096)]
    [int]$MemoryReserveMiB = 1024,

    [Parameter(ParameterSetName = 'Run')]
    [string]$Candidate,

    [Parameter(ParameterSetName = 'Run')]
    [string]$RouterInputsRoot = 'artifacts/scientific_recovery_v8/router_inputs',

    [Parameter(ParameterSetName = 'Run')]
    [string]$JepaCacheManifest,

    [Parameter(ParameterSetName = 'Run')]
    [string]$JepaWinnerArtifact,

    [Parameter(ParameterSetName = 'Run')]
    [string]$WinnerManifest,

    [Parameter(ParameterSetName = 'Run')]
    [switch]$DryRun,

    [Parameter(ParameterSetName = 'Run')]
    [switch]$NoBackgroundMonitor,

    # Publishing the ignored live monitor snapshot into docs is deliberately
    # opt-in and must be done only after the job queue is stopped.
    [Parameter(ParameterSetName = 'Run')]
    [switch]$PublishTrainingStatus,

    [Parameter(ParameterSetName = 'Monitor')]
    [switch]$MonitorOnly,

    [Parameter(ParameterSetName = 'Monitor')]
    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(1, 1440)]
    [int]$MonitorIntervalMinutes = 30,

    [Parameter(ParameterSetName = 'Monitor')]
    [ValidateRange(0, 100000)]
    [int]$MonitorCycles = 0,

    [string]$Protocol = 'configs/protocol/scientific_recovery_v8_temporal.json',
    [string]$Manifest = 'configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json',
    [string]$ResultsRoot = 'artifacts/scientific_recovery_v8'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ProtocolPath = Join-Path $RepoRoot $Protocol
$ManifestPath = Join-Path $RepoRoot $Manifest
$ResultsPath = Join-Path $RepoRoot $ResultsRoot
$StateRoot = Join-Path $ResultsPath 'orchestrator_state'
$LogRoot = Join-Path $ResultsPath 'orchestrator_logs'
$StatusMarkdown = Join-Path $ResultsPath 'monitor/TRAINING_STATUS.md'
$PublishedStatusMarkdown = Join-Path $RepoRoot 'docs/SCIENTIFIC_RECOVERY_V8_TRAINING_STATUS.md'
$StatusJson = Join-Path $StateRoot 'training_status.json'
$ProcessStatePath = Join-Path $StateRoot 'processes.json'
$StatusSigner = Join-Path $RepoRoot 'scripts/sign_scientific_recovery_v8_status.py'
$RouterInputsPath = Join-Path $RepoRoot $RouterInputsRoot

# `uv run` must resolve this checkout even when the caller starts the script
# from another directory (as the hidden monitor does).
Set-Location -LiteralPath $RepoRoot
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

function Write-AtomicText {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Content)
    $directory = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    $temporary = Join-Path $directory ('.{0}.{1}.tmp' -f [System.IO.Path]::GetFileName($Path), [guid]::NewGuid().ToString('N'))
    try {
        [System.IO.File]::WriteAllText($temporary, $Content, [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::Move($temporary, $Path, $true)
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function ConvertTo-CanonicalJsonText {
    param([Parameter(Mandatory)]$Value)
    return ($Value | ConvertTo-Json -Depth 32 -Compress)
}

function Write-SignedState {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Payload)
    if ($DryRun) { return }
    if (-not (Test-Path -LiteralPath $StatusSigner)) { throw "V8 status signer is missing: $StatusSigner" }
    $temporary = Join-Path ([System.IO.Path]::GetDirectoryName($Path)) ('.unsigned-{0}.json' -f [guid]::NewGuid().ToString('N'))
    try {
        [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($temporary)) | Out-Null
        Write-AtomicText -Path $temporary -Content (ConvertTo-CanonicalJsonText $Payload)
        & uv run --no-sync python $StatusSigner --input $temporary --output $Path
        if ($LASTEXITCODE -ne 0) { throw "V8 status signer failed for $Path (exit $LASTEXITCODE)" }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Test-SignedJson {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $code = @'
import json, sys
from pathlib import Path
from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    raise SystemExit(0 if isinstance(value, dict) and verify_artifact_hash(value) else 1)
except Exception:
    raise SystemExit(1)
'@
    & uv run --no-sync python -c $code $Path *> $null
    return $LASTEXITCODE -eq 0
}

function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Resolve-RepositoryRelativePath {
    param([Parameter(Mandatory)][string]$RelativePath, [Parameter(Mandatory)][string]$Label)
    if ([System.IO.Path]::IsPathRooted($RelativePath)) { throw "$Label must be repository-relative" }
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RelativePath))
    $rootWithSeparator = $RepoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes the repository root"
    }
    return $resolved
}

function Assert-SealedEvaluation {
    # The orchestration contract has no path or argument for these datasets/services.
    $forbidden = @('public validation', 'private test', 'evttc test', 'codabench')
    $serialized = ($PSBoundParameters | Out-String).ToLowerInvariant()
    foreach ($needle in $forbidden) {
        if ($serialized.Contains($needle)) { throw "sealed evaluation selector is forbidden: $needle" }
    }
}

function Assert-FrozenInputs {
    if (-not (Test-Path -LiteralPath $ProtocolPath)) { throw "frozen protocol is absent: $ProtocolPath" }
    if (-not (Test-Path -LiteralPath $ManifestPath)) { throw "frozen manifest is absent: $ManifestPath" }
    if ($DryRun) { return }
    & uv run --no-sync python scripts/freeze_scientific_recovery_v8_configs.py --protocol $Protocol --verify
    if ($LASTEXITCODE -ne 0) { throw 'frozen V8 protocol/config verification failed closed' }
}

function Get-FreeGpuMemoryMiB {
    if ($Device -notmatch '^cuda') { return [int]::MaxValue }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        # NVIDIA's Windows installer commonly leaves this outside PATH.  Search
        # the driver store rather than pinning a machine-specific INF directory.
        $driverStore = Join-Path $env:WINDIR 'System32/DriverStore/FileRepository'
        if (Test-Path -LiteralPath $driverStore) {
            $candidate = Get-ChildItem -LiteralPath $driverStore -Filter 'nvidia-smi.exe' -File -Recurse -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            if ($null -ne $candidate) { $nvidiaSmi = $candidate }
        }
    }
    if ($null -ne $nvidiaSmi) {
        $nvidiaPath = if ($nvidiaSmi -is [System.Management.Automation.CommandInfo]) { $nvidiaSmi.Source } else { $nvidiaSmi.FullName }
        $reported = & $nvidiaPath --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($reported | Select-Object -First 1))) {
            $first = (($reported | Select-Object -First 1).ToString().Trim() -replace '[^0-9]', '')
            $memory = 0
            if ([int]::TryParse($first, [ref]$memory) -and $memory -gt 0) { return $memory }
        }
    }
    # Torch uses the active CUDA context, so it remains a valid conservative
    # fallback when Windows has no nvidia-smi command on PATH.
    $torchProbe = 'import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info()[0] // (1024 * 1024))'
    $reported = & uv run --no-sync python -c $torchProbe 2>$null
    $memory = 0
    if ($LASTEXITCODE -eq 0 -and [int]::TryParse(($reported | Select-Object -First 1).ToString().Trim(), [ref]$memory) -and $memory -gt 0) {
        return $memory
    }
    throw 'unable to measure free GPU memory through nvidia-smi or torch.cuda.mem_get_info'
}

function Assert-ConcurrencyBudget {
    if ($MaxParallel -lt 3) { return }
    if (-not $EnableThreeWayConcurrency) {
        throw 'MaxParallel=3 requires -EnableThreeWayConcurrency and a measured GPU-memory preflight'
    }
    $freeMiB = Get-FreeGpuMemoryMiB
    $requiredMiB = (3 * $PerTrainingMemoryMiB) + $MemoryReserveMiB
    if ($freeMiB -lt $requiredMiB) {
        throw "three-way V8 scheduling denied: free GPU memory $freeMiB MiB is below measured budget $requiredMiB MiB (3 x $PerTrainingMemoryMiB + reserve $MemoryReserveMiB)"
    }
}

function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory)][string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Set-RunningProcessState {
    param([Parameter(Mandatory)][string]$StageName, [Parameter(Mandatory)][System.Diagnostics.Process]$Process, [Parameter(Mandatory)][string]$Stdout, [Parameter(Mandatory)][string]$Stderr)
    $record = [ordered]@{
        artifact_type = 'scientific_recovery_v8_orchestrator_processes_v1'
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        sealed_evaluation = $true
        processes = @([ordered]@{
            stage = $StageName
            pid = $Process.Id
            started_at_utc = [DateTime]::UtcNow.ToString('o')
            stdout_log = [System.IO.Path]::GetRelativePath($RepoRoot, $Stdout).Replace('\', '/')
            stderr_log = [System.IO.Path]::GetRelativePath($RepoRoot, $Stderr).Replace('\', '/')
        })
    }
    Write-SignedState -Path $ProcessStatePath -Payload $record
}

function Clear-RunningProcessState {
    if ($DryRun) { return }
    $record = [ordered]@{
        artifact_type = 'scientific_recovery_v8_orchestrator_processes_v1'
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        sealed_evaluation = $true
        processes = @()
    }
    Write-SignedState -Path $ProcessStatePath -Payload $record
}

function Invoke-V8Command {
    param([Parameter(Mandatory)][string]$StageName, [Parameter(Mandatory)][string[]]$Arguments)
    $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $stdout = Join-Path $LogRoot ("{0}_{1}.stdout.log" -f $StageName, $timestamp)
    $stderr = Join-Path $LogRoot ("{0}_{1}.stderr.log" -f $StageName, $timestamp)
    [System.IO.Directory]::CreateDirectory($LogRoot) | Out-Null
    if ($DryRun) {
        Write-Host ('DRY RUN: uv ' + (($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join ' '))
        return
    }
    $argumentLine = ($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join ' '
    $process = Start-Process -FilePath 'uv' -ArgumentList $argumentLine -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
    Set-RunningProcessState -StageName $StageName -Process $process -Stdout $stdout -Stderr $stderr
    # Persist the live PID before waiting so the background monitor can report
    # actual process liveness rather than a stale post-mortem record.
    try {
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "V8 stage $StageName failed; inspect $stderr" }
    }
    finally {
        Clear-RunningProcessState
    }
}

function Write-StageState {
    param([Parameter(Mandatory)][string]$StageName, [Parameter(Mandatory)][string]$Status, [string]$Detail = '')
    $safeDetail = $Detail
    if ($safeDetail.Length -gt 4000) { $safeDetail = $safeDetail.Substring(0, 4000) }
    $payload = [ordered]@{
        artifact_type = 'scientific_recovery_v8_orchestrator_stage_state_v1'
        stage = $StageName
        status = $Status
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        protocol_path = $Protocol.Replace('\', '/')
        protocol_sha256 = Get-FileSha256 $ProtocolPath
        frozen_manifest_path = $Manifest.Replace('\', '/')
        frozen_manifest_sha256 = Get-FileSha256 $ManifestPath
        device = $Device
        max_parallel = $MaxParallel
        sealed_evaluation = $true
        detail = $safeDetail
    }
    Write-SignedState -Path (Join-Path $StateRoot ("{0}.json" -f $StageName)) -Payload $payload
}

function Test-CompletedStage {
    param([Parameter(Mandatory)][string]$StageName)
    $path = Join-Path $StateRoot ("{0}.json" -f $StageName)
    if (-not (Test-SignedJson $path)) { return $false }
    try {
        $value = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        return $value.artifact_type -eq 'scientific_recovery_v8_orchestrator_stage_state_v1' -and $value.stage -eq $StageName -and $value.status -eq 'completed' -and $value.protocol_sha256 -eq (Get-FileSha256 $ProtocolPath) -and $value.frozen_manifest_sha256 -eq (Get-FileSha256 $ManifestPath)
    }
    catch { return $false }
}

function Assert-Dependencies {
    param([Parameter(Mandatory)][string]$StageName)
    if ($DryRun) { return }
    $dependencies = @{
        preflight = @(); autopsy = @('preflight'); router = @('autopsy'); temporal = @('autopsy')
        adaptive = @('router', 'temporal'); jepa = @('temporal'); multiseed_replication = @('jepa')
        robustness = @('multiseed_replication'); export = @('robustness'); package = @('export')
    }
    foreach ($dependency in $dependencies[$StageName]) {
        if (-not (Test-CompletedStage $dependency)) {
            throw "stage $StageName requires a current, signed completed state for $dependency"
        }
    }
}

function Get-StageCommand {
    param([Parameter(Mandatory)][string]$StageName)
    switch ($StageName) {
        'preflight' { throw 'preflight is a fixed command bundle handled by Invoke-Stage' }
        'autopsy' { return @('run', '--no-sync', 'python', 'scripts/replay_scientific_recovery_v8_mechanisms.py', '--protocol', $Protocol, '--models', 'a5', 'c2f', 'garl', '--device', $Device) }
        'router' { throw 'router is a three-fold signed-expert bundle handled by Invoke-Stage' }
        'temporal' { return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_temporal.py', '--protocol', $Protocol, '--manifest', $Manifest, '--results-root', $ResultsRoot, '--device', $Device, '--max-parallel', "$MaxParallel") }
        'adaptive' { return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_adaptive.py', '--protocol', $Protocol, '--manifest', $Manifest, '--results-root', $ResultsRoot, '--device', $Device, '--max-parallel', "$MaxParallel") }
        'jepa' {
            if ($DryRun) { return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_jepa_attribution.py', '--dry-run') }
            $winner = $null
            if ([string]::IsNullOrWhiteSpace($JepaWinnerArtifact) -or [string]::IsNullOrWhiteSpace($JepaCacheManifest)) {
                $winner = Resolve-WinnerContract
            }
            if ([string]::IsNullOrWhiteSpace($JepaWinnerArtifact)) {
                if ($null -eq $winner.Value.downstream_model_config) {
                    throw 'signed winner manifest lacks downstream_model_config required for JEPA attribution'
                }
                $JepaWinnerArtifact = [System.IO.Path]::GetRelativePath($RepoRoot, $winner.Path)
            }
            if ([string]::IsNullOrWhiteSpace($JepaCacheManifest)) {
                $cacheReference = $winner.Value.cache_manifest
                if ($null -eq $cacheReference -or [string]::IsNullOrWhiteSpace([string]$cacheReference.path) -or [string]::IsNullOrWhiteSpace([string]$cacheReference.sha256)) {
                    throw 'signed winner manifest lacks an exact cache_manifest binding required for JEPA attribution'
                }
                $cachePath = Resolve-RepositoryRelativePath -RelativePath ([string]$cacheReference.path) -Label 'winner cache_manifest path'
                if (-not (Test-Path -LiteralPath $cachePath) -or (Get-FileSha256 $cachePath) -ne ([string]$cacheReference.sha256).ToLowerInvariant()) { throw 'winner cache_manifest binding is absent or hash-mismatched' }
                $JepaCacheManifest = [System.IO.Path]::GetRelativePath($RepoRoot, $cachePath)
            }
            if ([string]::IsNullOrWhiteSpace($JepaCacheManifest) -or [string]::IsNullOrWhiteSpace($JepaWinnerArtifact)) {
                throw 'JEPA requires -JepaCacheManifest and -JepaWinnerArtifact from the signed, frozen downstream selection'
            }
            $cacheManifest = Resolve-RepositoryRelativePath -RelativePath $JepaCacheManifest -Label 'JEPA cache manifest path'
            $winnerArtifact = Resolve-RepositoryRelativePath -RelativePath $JepaWinnerArtifact -Label 'JEPA winner artifact path'
            if (-not (Test-Path -LiteralPath $cacheManifest) -or -not (Test-SignedJson $winnerArtifact)) { throw 'JEPA cache or winner artifact is missing or unsigned' }
            return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_jepa_attribution.py', '--cache-manifest', $cacheManifest, '--winner-artifact', $winnerArtifact, '--output-root', (Join-Path $ResultsRoot 'jepa'), '--device', $Device)
        }
        'multiseed_replication' {
            if ($DryRun) { return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_multiseed_replication.py', '--candidate', 'signed_candidate_required', '--dry-run') }
            $chosenCandidate = Resolve-SignedCandidate
            return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_multiseed_replication.py', '--candidate', $chosenCandidate, '--protocol', $Protocol, '--manifest', $Manifest, '--results-root', $ResultsRoot, '--device', $Device, '--max-parallel', "$MaxParallel")
        }
        'robustness' {
            if ($DryRun) { return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_robustness.py', '--dry-run') }
            $winner = Resolve-WinnerContract
            $factory = [string]$winner.Value.factory
            if ([string]::IsNullOrWhiteSpace($factory)) { throw 'signed winner manifest lacks a frozen robustness factory' }
            $checkpoint = Resolve-RepositoryRelativePath -RelativePath ([string]$winner.Value.checkpoint.path) -Label 'winner checkpoint path'
            return @('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_robustness.py', '--checkpoint', $checkpoint, '--winner-manifest', $winner.Path, '--output-dir', (Join-Path $ResultsRoot 'robustness'), '--factory', $factory, '--device', $Device)
        }
        'export' {
            if ($DryRun) { return @('run', '--no-sync', 'python', 'scripts/export_scientific_recovery_v8_onnx.py', '--help') }
            $winner = Resolve-WinnerContract
            $example = $winner.Value.example_input
            if ($null -eq $example -or [string]::IsNullOrWhiteSpace([string]$example.path) -or [string]::IsNullOrWhiteSpace([string]$example.sha256)) { throw 'signed winner manifest lacks an exact example_input binding' }
            $examplePath = Resolve-RepositoryRelativePath -RelativePath ([string]$example.path) -Label 'winner example_input path'
            if (-not (Test-Path -LiteralPath $examplePath) -or (Get-FileSha256 $examplePath) -ne ([string]$example.sha256).ToLowerInvariant()) { throw 'winner example_input binding is absent or hash-mismatched' }
            $checkpoint = Resolve-RepositoryRelativePath -RelativePath ([string]$winner.Value.checkpoint.path) -Label 'winner checkpoint path'
            return @('run', '--no-sync', 'python', 'scripts/export_scientific_recovery_v8_onnx.py', '--checkpoint', $checkpoint, '--example-input', $examplePath, '--output-dir', (Join-Path $ResultsRoot 'export'))
        }
        'package' { return @('run', '--no-sync', 'python', 'scripts/package_scientific_recovery_v8_evidence.py') }
        default { throw "no V8 command registered for stage $StageName" }
    }
}

function Assert-StageExecutable {
    param([Parameter(Mandatory)][string[]]$Command)
    $scriptIndex = [array]::IndexOf($Command, 'python') + 1
    if ($scriptIndex -le 0 -or $scriptIndex -ge $Command.Count) { throw 'V8 command has no Python script' }
    $scriptPath = Join-Path $RepoRoot $Command[$scriptIndex]
    if (-not (Test-Path -LiteralPath $scriptPath)) { throw "V8 stage executable is missing: $scriptPath" }
}

function Get-RouterCommand {
    param([Parameter(Mandatory)][ValidateRange(0, 2)][int]$Fold)
    # Expert exports are intentionally explicit and fold-scoped.  A glob over
    # arbitrary historical OOF files would recreate the post-hoc V7 leak that
    # V8 is designed to avoid.
    if ($DryRun) {
        return [string[]]@('run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_nested_router.py', '--protocol', $Protocol, '--manifest', $Manifest, '--output-dir', (Join-Path $ResultsRoot ("router/fold{0}" -f $Fold)), '--device', $Device, '--dry-run')
    }
    if (-not (Test-Path -LiteralPath $RouterInputsPath)) {
        throw "router expert-input root is absent: $RouterInputsPath"
    }
    $config = Join-Path $RepoRoot ("configs/experiment/scientific_recovery_v8_fold_chain/router_fold{0}_seed7.yaml" -f $Fold)
    $foldRoot = Join-Path $RouterInputsPath ("fold{0}" -f $Fold)
    $inputs = [ordered]@{
        a5_inner = Join-Path $foldRoot 'a5_inner_artifact.json'
        c2f_inner = Join-Path $foldRoot 'c2f_inner_artifact.json'
        a5_outer = Join-Path $foldRoot 'a5_outer_artifact.json'
        c2f_outer = Join-Path $foldRoot 'c2f_outer_artifact.json'
    }
    if (-not (Test-Path -LiteralPath $config)) { throw "frozen router config is absent: $config" }
    foreach ($entry in $inputs.GetEnumerator()) {
        if (-not (Test-Path -LiteralPath $entry.Value)) {
            throw "router fold $Fold lacks required signed expert artifact $($entry.Key): $($entry.Value)"
        }
        if (-not (Test-SignedJson $entry.Value)) {
            throw "router fold $Fold expert artifact is unsigned or corrupt: $($entry.Value)"
        }
    }
    return [string[]]@(
        'run', '--no-sync', 'python', 'scripts/run_scientific_recovery_v8_nested_router.py',
        '--config', $config, '--protocol', $Protocol, '--manifest', $Manifest,
        '--output-dir', (Join-Path $ResultsRoot ("router/fold{0}" -f $Fold)), '--device', $Device,
        '--a5-inner-artifact', $inputs.a5_inner, '--c2f-inner-artifact', $inputs.c2f_inner,
        '--a5-outer-artifact', $inputs.a5_outer, '--c2f-outer-artifact', $inputs.c2f_outer
    )
}

function Invoke-Seed7Aggregate {
    $aggregateState = Join-Path $StateRoot 'aggregate_seed7.json'
    if (Test-SignedJson $aggregateState) {
        try {
            $previous = Get-Content -LiteralPath $aggregateState -Raw | ConvertFrom-Json
            if ($previous.status -eq 'completed' -and $previous.protocol_file_sha256 -eq (Get-FileSha256 $ProtocolPath) -and $previous.frozen_manifest_file_sha256 -eq (Get-FileSha256 $ManifestPath)) { return }
        }
        catch {
            throw "signed aggregate_seed7 reuse state is unreadable or hash-mismatched: $_"
        }
    }
    $command = [string[]]@(
        'run', '--no-sync', 'python', 'scripts/aggregate_scientific_recovery_v8.py',
        '--protocol', $Protocol, '--manifest', $Manifest, '--results-root', $ResultsRoot,
        '--output', (Join-Path $ResultsRoot 'aggregate_seed7.json')
    )
    Assert-StageExecutable $command
    try {
        Invoke-V8Command -StageName 'aggregate_seed7' -Arguments $command
        $aggregatePath = Join-Path $ResultsPath 'aggregate_seed7.json'
        if (-not (Test-SignedJson $aggregatePath)) { throw 'seed-7 aggregate is missing or has an invalid artifact signature' }
        Write-SignedState -Path $aggregateState -Payload ([ordered]@{
            artifact_type = 'scientific_recovery_v8_orchestrator_aggregate_state_v1'
            status = 'completed'
            generated_at_utc = [DateTime]::UtcNow.ToString('o')
            protocol_file_sha256 = Get-FileSha256 $ProtocolPath
            frozen_manifest_file_sha256 = Get-FileSha256 $ManifestPath
            aggregate_path = 'artifacts/scientific_recovery_v8/aggregate_seed7.json'
            aggregate_file_sha256 = Get-FileSha256 $aggregatePath
            sealed_evaluation = $true
        })
    }
    catch {
        Write-SignedState -Path $aggregateState -Payload ([ordered]@{
            artifact_type = 'scientific_recovery_v8_orchestrator_aggregate_state_v1'
            status = 'failed_integrity'
            generated_at_utc = [DateTime]::UtcNow.ToString('o')
            protocol_file_sha256 = Get-FileSha256 $ProtocolPath
            frozen_manifest_file_sha256 = Get-FileSha256 $ManifestPath
            detail = $_.Exception.Message
            sealed_evaluation = $true
        })
        throw
    }
}

function Resolve-SignedCandidate {
    if (-not [string]::IsNullOrWhiteSpace($Candidate)) { return $Candidate }
    $candidates = [System.Collections.Generic.List[string]]::new()
    foreach ($path in Get-ChildItem -LiteralPath $ResultsPath -Filter '*.json' -File -Recurse -ErrorAction SilentlyContinue) {
        if (-not (Test-SignedJson $path.FullName)) { continue }
        try {
            $value = Get-Content -LiteralPath $path.FullName -Raw | ConvertFrom-Json
            if ($value.seed -eq 7 -and $value.multiseed_replication_candidate -eq $true -and -not [string]::IsNullOrWhiteSpace([string]$value.candidate_id)) {
                $candidates.Add([string]$value.candidate_id)
            }
        }
        catch { continue }
    }
    $unique = @($candidates | Sort-Object -Unique)
    if ($unique.Count -ne 1) { throw 'multiseed replication requires exactly one signed seed-7 multiseed_replication_candidate' }
    return $unique[0]
}

function Resolve-WinnerContract {
    $paths = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
    if (-not [string]::IsNullOrWhiteSpace($WinnerManifest)) {
        $path = Resolve-RepositoryRelativePath -RelativePath $WinnerManifest -Label 'winner manifest path'
        if (-not (Test-Path -LiteralPath $path)) { throw "winner manifest is absent: $path" }
        $paths.Add((Get-Item -LiteralPath $path))
    }
    elseif (Test-Path -LiteralPath $ResultsPath) {
        foreach ($path in Get-ChildItem -LiteralPath $ResultsPath -Filter '*.json' -File -Recurse -ErrorAction SilentlyContinue) { $paths.Add($path) }
    }
    $valid = [System.Collections.Generic.List[object]]::new()
    foreach ($path in $paths) {
        if (-not (Test-SignedJson $path.FullName)) { continue }
        try {
            $value = Get-Content -LiteralPath $path.FullName -Raw | ConvertFrom-Json
            $closed = $value.closed_evaluation
            $checkpoint = $value.checkpoint
            if ($null -eq $closed -or $null -eq $checkpoint -or [string]::IsNullOrWhiteSpace([string]$checkpoint.path) -or [string]::IsNullOrWhiteSpace([string]$checkpoint.sha256)) { continue }
            $flags = @($closed.PSObject.Properties | ForEach-Object { [bool]$_.Value })
            if ($flags.Count -eq 0 -or ($flags | Where-Object { $_ }).Count -ne 0) { continue }
            $valid.Add([pscustomobject]@{ Path = $path.FullName; Value = $value })
        }
        catch { continue }
    }
    if ($valid.Count -ne 1) { throw 'expected exactly one signed sealed winner manifest with checkpoint binding' }
    $winner = $valid[0]
    $checkpointPath = Resolve-RepositoryRelativePath -RelativePath ([string]$winner.Value.checkpoint.path) -Label 'winner checkpoint path'
    if (-not (Test-Path -LiteralPath $checkpointPath) -or (Get-FileSha256 $checkpointPath) -ne ([string]$winner.Value.checkpoint.sha256).ToLowerInvariant()) {
        throw 'winner manifest checkpoint binding is absent or hash-mismatched'
    }
    return $winner
}

function Invoke-Stage {
    param([Parameter(Mandatory)][string]$StageName)
    Assert-FrozenInputs
    Assert-Dependencies $StageName
    if (Test-CompletedStage $StageName) {
        Write-Host "V8 stage $StageName already has a current signed completed state; skipping."
        return
    }
    try {
        Write-StageState -StageName $StageName -Status 'running'
        if ($StageName -eq 'preflight') {
            # These are the V8 P0 commands frozen in the handoff.  They run on
            # CPU and validate the repository before a CUDA process is started.
            Invoke-V8Command -StageName 'preflight_1' -Arguments ([string[]]@('sync', '--locked', '--all-groups', '--no-editable'))
            $quality = [string[]]@('run', '--no-sync', 'python', 'scripts/check_quality_baseline.py')
            Assert-StageExecutable $quality
            Invoke-V8Command -StageName 'preflight_2' -Arguments $quality
            Invoke-V8Command -StageName 'preflight_3' -Arguments ([string[]]@('run', '--no-sync', 'python', '-m', 'pyright'))
            Invoke-V8Command -StageName 'preflight_4' -Arguments ([string[]]@('run', '--no-sync', 'pytest', '-q'))
            $smoke = [string[]]@('run', '--no-sync', 'python', 'scripts/smoke_scientific_recovery_v8.py', '--device', 'cpu')
            Assert-StageExecutable $smoke
            Invoke-V8Command -StageName 'preflight_5' -Arguments $smoke
        }
        elseif ($StageName -eq 'router') {
            foreach ($fold in 0..2) {
                $routerCommand = [string[]](Get-RouterCommand -Fold $fold)
                Assert-StageExecutable $routerCommand
                Invoke-V8Command -StageName ("router_fold{0}" -f $fold) -Arguments $routerCommand
            }
        }
        else {
            $command = Get-StageCommand $StageName
            Assert-StageExecutable $command
            Invoke-V8Command -StageName $StageName -Arguments $command
        }
        Write-StageState -StageName $StageName -Status 'completed'
    }
    catch {
        $stageError = $_.Exception.Message
        $latestStderr = Get-ChildItem -LiteralPath $LogRoot -Filter ("{0}_*.stderr.log" -f $StageName) -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        if ($null -ne $latestStderr) {
            $stageError = $stageError + "`n" + (Get-Content -LiteralPath $latestStderr.FullName -Raw)
        }
        if ($StageName -eq 'adaptive' -and $stageError.Contains('C1/adaptive is closed: no signed mechanism opening gate was found')) {
            Write-StageState -StageName $StageName -Status 'blocked_gate' -Detail 'C1 remains closed under the signed mechanism gate.'
            Write-Host 'V8 adaptive stage is correctly blocked by the frozen C1 gate; continuing to the mandatory JEPA phase.'
            return
        }
        Write-StageState -StageName $StageName -Status 'failed_integrity' -Detail $_.Exception.Message
        throw
    }
}

function Get-StagePlan {
    switch ($Stage) {
        'screen' { return @('preflight', 'autopsy', 'router', 'temporal', 'adaptive', 'jepa') }
        'all' { return @('preflight', 'autopsy', 'router', 'temporal', 'adaptive', 'jepa', 'multiseed_replication', 'robustness', 'export', 'package') }
        default { return @($Stage) }
    }
}

function Get-MonitorInventory {
    $runsRoot = Join-Path $ResultsPath 'runs'
    $runs = @()
    if (Test-Path -LiteralPath $runsRoot) {
        foreach ($directory in Get-ChildItem -LiteralPath $runsRoot -Directory -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName) {
            $summary = Join-Path $directory.FullName 'summary.json'
            $checkpoint = Join-Path $directory.FullName 'checkpoints/model_best.pt'
            $last = Join-Path $directory.FullName 'state/last.pt'
            $metrics = Join-Path $directory.FullName 'train_metrics.csv'
            if (-not ((Test-Path $summary) -or (Test-Path $checkpoint) -or (Test-Path $last) -or (Test-Path $metrics))) { continue }
            $runs += [ordered]@{
                run = [System.IO.Path]::GetRelativePath($RepoRoot, $directory.FullName).Replace('\', '/')
                summary_present = Test-Path -LiteralPath $summary
                summary_signature_valid = if (Test-Path -LiteralPath $summary) { Test-SignedJson $summary } else { $false }
                checkpoint_present = Test-Path -LiteralPath $checkpoint
                resume_checkpoint_present = Test-Path -LiteralPath $last
                metrics_present = Test-Path -LiteralPath $metrics
                newest_write_utc = $directory.LastWriteTimeUtc.ToString('o')
            }
        }
    }
    $processes = @()
    if (Test-SignedJson $ProcessStatePath) {
        try { $processes = @((Get-Content -LiteralPath $ProcessStatePath -Raw | ConvertFrom-Json).processes) } catch { $processes = @() }
    }
    $processRows = foreach ($process in $processes) {
        $pid = [int]$process.pid
        [ordered]@{ stage = [string]$process.stage; pid = $pid; alive = $null -ne (Get-Process -Id $pid -ErrorAction SilentlyContinue) }
    }
    return [ordered]@{ runs = @($runs); processes = @($processRows) }
}

function Write-MonitorStatus {
    $inventory = Get-MonitorInventory
    $payload = [ordered]@{
        artifact_type = 'scientific_recovery_v8_training_monitor_state_v1'
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        protocol_sha256 = if (Test-Path $ProtocolPath) { Get-FileSha256 $ProtocolPath } else { $null }
        frozen_manifest_sha256 = if (Test-Path $ManifestPath) { Get-FileSha256 $ManifestPath } else { $null }
        sealed_evaluation = $true
        monitor_interval_minutes = $MonitorIntervalMinutes
        processes = $inventory.processes
        runs = $inventory.runs
    }
    Write-SignedState -Path $StatusJson -Payload $payload
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# Scientific Recovery V8 training status')
    $lines.Add('')
    $lines.Add(('Generated at: `{0}`.' -f $payload.generated_at_utc))
    $lines.Add('')
    $lines.Add('This monitor only inventories local signed artifacts. Public validation, private test, EvTTC test and CodaBench remain sealed.')
    $lines.Add('')
    $lines.Add('## Processes')
    $lines.Add('')
    if ($payload.processes.Count -eq 0) { $lines.Add('No orchestrator process is currently recorded.') }
    else {
        $lines.Add('| Stage | PID | Alive |')
        $lines.Add('| --- | ---: | --- |')
        foreach ($process in $payload.processes) { $lines.Add('| {0} | {1} | {2} |' -f $process.stage, $process.pid, $process.alive) }
    }
    $lines.Add('')
    $lines.Add('## Local runs')
    $lines.Add('')
    if ($payload.runs.Count -eq 0) { $lines.Add('No local V8 run directory with state, checkpoint, metrics or summary is present.') }
    else {
        $lines.Add('| Run | Signed summary | Best checkpoint | Resume state | Metrics | Latest write (UTC) |')
        $lines.Add('| --- | --- | --- | --- | --- | --- |')
        foreach ($run in $payload.runs) {
            $summaryState = if ($run.summary_signature_valid) { 'valid' } elseif ($run.summary_present) { 'invalid' } else { 'absent' }
            $lines.Add('| {0} | {1} | {2} | {3} | {4} | {5} |' -f $run.run, $summaryState, $run.checkpoint_present, $run.resume_checkpoint_present, $run.metrics_present, $run.newest_write_utc)
        }
    }
    if (-not $DryRun) { Write-AtomicText -Path $StatusMarkdown -Content (($lines -join "`n") + "`n") }
}

function Publish-MonitorStatus {
    if (-not $PublishTrainingStatus) { return }
    if (-not (Test-SignedJson $StatusJson) -or -not (Test-Path -LiteralPath $StatusMarkdown)) {
        throw 'cannot publish V8 status: the current local monitor state is absent or unsigned'
    }
    Write-AtomicText -Path $PublishedStatusMarkdown -Content (Get-Content -LiteralPath $StatusMarkdown -Raw)
}

function Start-BackgroundMonitor {
    if ($DryRun -or $NoBackgroundMonitor) { return }
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, '-MonitorOnly', '-MonitorIntervalMinutes', "$MonitorIntervalMinutes", '-Protocol', $Protocol, '-Manifest', $Manifest, '-ResultsRoot', $ResultsRoot)
    $monitorLog = Join-Path $LogRoot 'monitor.stdout.log'
    $monitorError = Join-Path $LogRoot 'monitor.stderr.log'
    [System.IO.Directory]::CreateDirectory($LogRoot) | Out-Null
    $process = Start-Process -FilePath 'pwsh' -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $monitorLog -RedirectStandardError $monitorError -WindowStyle Hidden -PassThru
    $monitorPayload = [ordered]@{
        artifact_type = 'scientific_recovery_v8_training_monitor_pid_v1'
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        pid = $process.Id
        interval_minutes = $MonitorIntervalMinutes
        stdout_log = [System.IO.Path]::GetRelativePath($RepoRoot, $monitorLog).Replace('\', '/')
        stderr_log = [System.IO.Path]::GetRelativePath($RepoRoot, $monitorError).Replace('\', '/')
        sealed_evaluation = $true
    }
    Write-SignedState -Path (Join-Path $StateRoot 'monitor_pid.json') -Payload $monitorPayload
    Write-Host "Started hidden V8 monitor PID $($process.Id); logs: $monitorLog"
}

try {
    Assert-SealedEvaluation
    if ($MonitorOnly) {
        $cycle = 0
        do {
            Write-MonitorStatus
            $cycle++
            if ($MonitorCycles -ne 0 -and $cycle -ge $MonitorCycles) { break }
            Start-Sleep -Seconds ($MonitorIntervalMinutes * 60)
        } while ($true)
        exit 0
    }
    Assert-ConcurrencyBudget
    Start-BackgroundMonitor
    foreach ($stageName in Get-StagePlan) {
        # Aggregate only after the prospective seed-7 arms have completed.  It
        # is an internal DAG node, not a selectable stage, so no caller can use
        # it to bypass R/B gates or nominate a candidate by hand.
        if (($stageName -eq 'jepa' -or $stageName -eq 'multiseed_replication') -and -not $DryRun) {
            Invoke-Seed7Aggregate
        }
        Invoke-Stage $stageName
    }
    if (-not $DryRun) {
        Write-MonitorStatus
        Publish-MonitorStatus
    }
    exit 0
}
catch {
    [Console]::Error.WriteLine("Scientific Recovery V8 orchestration failed closed: $($_.Exception.Message)")
    exit 2
}
