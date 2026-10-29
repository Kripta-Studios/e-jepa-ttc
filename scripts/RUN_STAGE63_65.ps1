[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$HandoffRoot,
    [Parameter(Mandatory = $true)][string]$ReferenceRoot,
    [Parameter(Mandatory = $true)][string]$Stage61Worktree,
    [Parameter(Mandatory = $true)][string]$RawTrainRoot,
    [Parameter(Mandatory = $true)][string]$TrainParquet,
    [Parameter(Mandatory = $true)][string]$CoherentA5Root,
    [Parameter(Mandatory = $true)][string]$FrozenTeacherAudit,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Python = "python",
    [string]$Device = "cuda",
    [ValidateSet(4, 8)][int]$Microbatch = 8,
    [switch]$Resume,
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_scientific_recovery_v9_stage63_65.py"
$runnerArguments = @(
    "-B", $runner,
    "--handoff-root", $HandoffRoot,
    "--reference-root", $ReferenceRoot,
    "--stage61-worktree", $Stage61Worktree,
    "--raw-train-root", $RawTrainRoot,
    "--train-parquet", $TrainParquet,
    "--coherent-a5-root", $CoherentA5Root,
    "--frozen-teacher-audit", $FrozenTeacherAudit,
    "--output-root", $OutputRoot,
    "--device", $Device,
    "--microbatch", [string]$Microbatch
)
if ($Resume) { $runnerArguments += "--resume" }
if ($AuditOnly) { $runnerArguments += "--audit-only" }
& $Python @runnerArguments
exit $LASTEXITCODE
