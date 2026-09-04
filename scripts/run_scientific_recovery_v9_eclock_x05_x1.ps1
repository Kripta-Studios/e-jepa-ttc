[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter()]
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [Parameter()]
    [string]$PythonExe,
    [Parameter()]
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedStartingHead = '57865bea943f7c1518341003170cec1c221aa093'
$ExpectedBranch = 'scientific-recovery-v9-eclock-x1-incremental-fusion'
$ReferenceRepo = Join-Path $ProjectRoot 'e-jepa-ttc'
$X0Worktree = Join-Path $ProjectRoot 'e-jepa-ttc-v9-eclock-x0'
$X0Campaign = Join-Path $X0Worktree 'artifacts\scientific_recovery_v9_eclock\campaigns\x0-seed7-57865bea943f'
$CacheRoot = Join-Path $ReferenceRepo 'artifacts\cache\garl_object_event_common_roi_train8192_v1'
$InputRoot = Join-Path $ProjectRoot 'CODEX_INPUT_X05_X1_2026-09-04\E_JEPA_TTC_X05_X1_CODEX_COMPLETE_PACKAGE_2026-09-04'
$X0Bundle = Join-Path $InputRoot 'INPUTS\E_JEPA_TTC_X0_SEED7_ESSENTIAL_RESULTS_57865bea943f.zip'
$OutputRoot = Join-Path $RepoRoot 'artifacts\scientific_recovery_v9_eclock_x1'
if (-not $PythonExe) {
    $PythonExe = Join-Path $ReferenceRepo '.venv\Scripts\python.exe'
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) { throw "Missing repo: $RepoRoot" }
if (-not (Test-Path -LiteralPath $ReferenceRepo -PathType Container)) { throw "Missing reference repo: $ReferenceRepo" }
if (-not (Test-Path -LiteralPath $X0Campaign -PathType Container)) { throw "Missing X0 campaign: $X0Campaign" }
if (-not (Test-Path -LiteralPath $CacheRoot -PathType Container)) { throw "Missing cache: $CacheRoot" }
if (-not (Test-Path -LiteralPath $X0Bundle -PathType Leaf)) { throw "Missing X0 bundle: $X0Bundle" }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing Python: $PythonExe" }

$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$RepoRoot"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
$env:OPENBLAS_NUM_THREADS = '16'
$env:NUMEXPR_MAX_THREADS = '16'
$env:TOKENIZERS_PARALLELISM = 'false'

$TrainingCommit = (& git -C $RepoRoot rev-parse HEAD).Trim()
$Branch = (& git -C $RepoRoot branch --show-current).Trim()
if ($Branch -ne $ExpectedBranch) { throw "Wrong branch: $Branch" }
& git -C $RepoRoot merge-base --is-ancestor $ExpectedStartingHead $TrainingCommit
if ($LASTEXITCODE -ne 0) { throw 'Training commit is not descended from the mandatory starting HEAD' }
$Dirty = (& git -C $RepoRoot status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) { throw "Versioned worktree must be clean before QA/results:`n$Dirty" }

$ShortCommit = $TrainingCommit.Substring(0, 12)
$QaRoot = Join-Path $OutputRoot "qa_preflight\$ShortCommit"
New-Item -ItemType Directory -Path $QaRoot -Force | Out-Null

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Executable,
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )
    $LogPath = Join-Path $QaRoot "$Name.log"
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$ReferenceScripts = Join-Path $ReferenceRepo '.venv\Scripts'
$InheritedTests = @()
$InheritedTests += Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'tests\unit') -Filter 'test_collision_clock*.py' | ForEach-Object { $_.FullName }
$InheritedTests += Join-Path $RepoRoot 'tests\unit\test_a5_local_transport.py'
$InheritedTests += Join-Path $RepoRoot 'tests\integration\test_collision_clock_resume.py'
$InheritedTests += Join-Path $RepoRoot 'tests\scientific\test_collision_clock_no_leakage.py'
$InheritedTests += Join-Path $RepoRoot 'tests\regression\test_scientific_recovery_v8_immutable.py'
$TestPaths = $InheritedTests + @(Join-Path $RepoRoot 'tests\unit\test_incremental_fusion_x05_x1.py')

$InheritedQuality = @(
    'src/e_jepa_ttc/models/collision_clock_motion.py',
    'src/e_jepa_ttc/data/collision_clock_cache.py',
    'src/e_jepa_ttc/training/collision_clock_eap.py',
    'src/e_jepa_ttc/evaluation/collision_clock_bootstrap.py',
    'src/e_jepa_ttc/evaluation/collision_clock_gates.py',
    'src/e_jepa_ttc/evaluation/collision_clock_cross_arm.py',
    'src/e_jepa_ttc/evaluation/collision_clock_runner.py',
    'src/e_jepa_ttc/evaluation/collision_clock_aggregate.py',
    'scripts/train_scientific_recovery_v9_eclock.py',
    'scripts/compare_scientific_recovery_v9_eclock_x0.py',
    'scripts/preflight_scientific_recovery_v9_eclock_x0.py',
    'scripts/smoke_scientific_recovery_v9_eclock_x0.py',
    'scripts/adopt_scientific_recovery_v9_eclock_x0_base_dyn.py',
    'scripts/report_scientific_recovery_v9_eclock_environment.py',
    'scripts/analyze_scientific_recovery_v9_eclock_x0.py',
    'scripts/package_scientific_recovery_v9_eclock_x0_results.py',
    'tests/unit/test_collision_clock_cache.py',
    'tests/unit/test_collision_clock_bootstrap.py',
    'tests/unit/test_collision_clock_gates.py',
    'tests/unit/test_collision_clock_pair_and_support.py',
    'tests/unit/test_collision_clock_motion.py',
    'tests/unit/test_collision_clock_aggregate_io.py',
    'tests/integration/test_collision_clock_resume.py'
)
$NewQuality = @(
    'scripts/run_scientific_recovery_v9_eclock_x05_x1.py',
    'src/e_jepa_ttc/evaluation/incremental_fusion.py',
    'src/e_jepa_ttc/evaluation/incremental_replay.py',
    'src/e_jepa_ttc/models/incremental_residual.py',
    'src/e_jepa_ttc/training/incremental_residual.py',
    'tests/unit/test_incremental_fusion_x05_x1.py'
)
$QualityFiles = ($InheritedQuality + $NewQuality) | ForEach-Object { Join-Path $RepoRoot $_ }

Invoke-Checked -Executable (Join-Path $ReferenceScripts 'ruff.exe') -Name 'ruff-check' -Arguments (@('check') + $QualityFiles)
Invoke-Checked -Executable (Join-Path $ReferenceScripts 'ruff.exe') -Name 'ruff-format' -Arguments (@('format', '--check') + $QualityFiles)
Invoke-Checked -Executable (Join-Path $ReferenceScripts 'pyright.exe') -Name 'pyright' -Arguments (@('--venvpath', $ReferenceRepo) + $QualityFiles)
Invoke-Checked -Executable $PythonExe -Name 'pytest-inherited-and-new' -Arguments (@('-m', 'pytest', '-q') + $TestPaths)
& git -C $RepoRoot diff --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed' }

$AstTokens = $null
$AstErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
    $PSCommandPath,
    [ref]$AstTokens,
    [ref]$AstErrors
)
if ($AstErrors.Count -ne 0) { throw "PowerShell AST errors: $($AstErrors -join '; ')" }
@{
    parsed = $true
    error_count = 0
    script = $PSCommandPath
    training_commit = $TrainingCommit
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $QaRoot 'powershell_ast.json') -Encoding utf8

$DirtyAfterQa = (& git -C $RepoRoot status --porcelain=v1 --untracked-files=no) -join "`n"
if ($DirtyAfterQa) { throw "QA changed versioned files:`n$DirtyAfterQa" }
& git -C $RepoRoot diff --stat "$ExpectedStartingHead..$TrainingCommit" | Set-Content -LiteralPath (Join-Path $QaRoot 'training_surface_stat.txt') -Encoding utf8
& git -C $RepoRoot diff --name-status "$ExpectedStartingHead..$TrainingCommit" | Set-Content -LiteralPath (Join-Path $QaRoot 'training_surface_files.txt') -Encoding utf8
$TrainingCommit | Set-Content -LiteralPath (Join-Path $QaRoot 'TRAINING_COMMIT.txt') -Encoding ascii

$Runner = Join-Path $RepoRoot 'scripts\run_scientific_recovery_v9_eclock_x05_x1.py'
$RunnerArguments = @(
    $Runner,
    '--repo', $RepoRoot,
    '--reference-repo', $ReferenceRepo,
    '--x0-campaign', $X0Campaign,
    '--cache-root', $CacheRoot,
    '--x0-bundle', $X0Bundle,
    '--output-root', $OutputRoot,
    '--allow-training-commit'
)
if ($Resume) { $RunnerArguments += '--resume' }
& $PythonExe @RunnerArguments 2>&1 | Tee-Object -FilePath (Join-Path $QaRoot 'campaign.log')
if ($LASTEXITCODE -ne 0) { throw "Scientific campaign failed with exit code $LASTEXITCODE" }

$CampaignRoot = Join-Path $OutputRoot "campaigns\x05-x1-$ShortCommit"
$Required = @(
    (Join-Path $CampaignRoot 'CODEX_X05_X1_FINAL_REPORT.md'),
    (Join-Path $CampaignRoot "E_JEPA_TTC_X05_X1_ESSENTIAL_RESULTS_$ShortCommit.zip"),
    (Join-Path $CampaignRoot "E_JEPA_TTC_X05_X1_ESSENTIAL_RESULTS_$ShortCommit.zip.sha256"),
    (Join-Path $CampaignRoot 'NEXT_DECISION.json')
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Mandatory deliverable missing: $Path" }
    if ((Get-Item -LiteralPath $Path).Length -le 0) { throw "Mandatory deliverable empty: $Path" }
}

$Required
