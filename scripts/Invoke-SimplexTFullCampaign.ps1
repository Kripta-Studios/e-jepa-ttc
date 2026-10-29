<#
.SYNOPSIS
Runs preparation, materializes the frozen scientific plan, then runs through T6.
.DESCRIPTION
One foreground invocation; repeat the identical command after interruption.
The pinned preparation plan must finish with materialize_simplex_t_freeze_launch.py.
That worker requires complete sources and real current QA; missing prerequisites
are not replaced with nominal PASS records. Existing caches and attempts survive.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Configuration,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ConfigurationSha256,
    [Parameter(Mandatory)][string]$PythonExecutable,
    [ValidateRange(1, 60)][int]$ResourceRetrySeconds = 30,
    [ValidateRange(0, 100000)][int]$MaxResourceRetries = 0
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$work = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$configPath = (Resolve-Path -LiteralPath $Configuration).Path
$python = (Resolve-Path -LiteralPath $PythonExecutable).Path
if ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ConfigurationSha256) {
    throw 'Full campaign configuration SHA256 differs.'
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.schema -ne 'simplex_t_full_orchestration_v1') { throw 'Full campaign schema required.' }
$orchestrator = Join-Path $PSScriptRoot 'Invoke-SimplexTCampaign.ps1'
$builder = Join-Path $PSScriptRoot 'prepare_simplex_t_scientific_plan.py'
$fullRoot = [IO.Path]::GetFullPath($config.run_root)
$artifactRoot = Join-Path $work 'artifacts'
foreach ($path in @($fullRoot, [IO.Path]::GetFullPath($config.campaign_root))) {
    if (-not $path.StartsWith($artifactRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) { throw 'Outputs must remain inside companion artifacts.' }
}
$prepareRoot = Join-Path $fullRoot 'preparation_logs'
$scienceRoot = Join-Path $fullRoot 'scientific_logs'
$preparationPath = [IO.Path]::GetFullPath($config.preparation_plan.path)
function Assert-FullBoundary {
    foreach ($pin in @(
        @{path=$configPath; sha256=$ConfigurationSha256},
        @{path=$orchestrator; sha256=$config.orchestrator_sha256},
        @{path=$builder; sha256=$config.scientific_plan_builder_sha256},
        @{path=$preparationPath; sha256=$config.preparation_plan.sha256},
        @{path=$config.reconciliation.path; sha256=$config.reconciliation.sha256}
    )) {
        if ((Get-FileHash -LiteralPath $pin.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pin.sha256) {
            throw "Full campaign pin changed: $($pin.path)"
        }
    }
}
Assert-FullBoundary
$prep = Get-Content -LiteralPath $preparationPath -Raw | ConvertFrom-Json
if ($prep.schema -ne 'simplex_t_orchestration_plan_v1' -or @($prep.steps).Count -eq 0) {
    throw 'Nonempty pinned preparation plan required.'
}
$last = @($prep.steps)[-1]
if ($last.verify.script -ne 'scripts/materialize_simplex_t_freeze_launch.py' -or
    @($last.verify.arguments) -notcontains '--verify-only') {
    throw 'Preparation must end with real freeze-launch materialization verification.'
}
$launchFlag = [Array]::IndexOf(@($last.verify.arguments), '--output')
if ($launchFlag -lt 0 -or $launchFlag + 1 -ge @($last.verify.arguments).Count) {
    throw 'Preparation verifier requires an explicit output argument.'
}
$launchArgument = $last.verify.arguments[$launchFlag + 1]
$resolvedLaunchArgument = if ([IO.Path]::IsPathRooted($launchArgument)) {
    [IO.Path]::GetFullPath($launchArgument)
} else { [IO.Path]::GetFullPath((Join-Path $work $launchArgument)) }
if ($resolvedLaunchArgument -ne [IO.Path]::GetFullPath($config.freeze_launch)) {
    throw 'Preparation does not bind the configured freeze launch.'
}
$null = New-Item -ItemType Directory -Path $fullRoot -Force
$identity = Join-Path $fullRoot 'FULL_CONFIGURATION_SHA256.txt'
if (Test-Path -LiteralPath $identity) {
    if ((Get-Content -LiteralPath $identity -Raw).Trim() -ne $ConfigurationSha256) {
        throw 'Full run root belongs to another configuration.'
    }
} else { [IO.File]::WriteAllText($identity, $ConfigurationSha256) }
$shell = (Get-Process -Id $PID).Path
$previousLocation = Get-Location
$previousPythonPath = $env:PYTHONPATH
try {
    Set-Location -LiteralPath $work
    $env:PYTHONPATH = Join-Path $work 'src'
    & $shell -NoProfile -File $orchestrator -Plan $preparationPath `
        -PlanSha256 $config.preparation_plan.sha256 -PythonExecutable $python `
        -RunRoot $prepareRoot -ResourceRetrySeconds $ResourceRetrySeconds -MaxResourceRetries $MaxResourceRetries
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Assert-FullBoundary
    # Recheck only the completed freeze inputs in the child plan. Do not replay
    # or rehash all compiled feature arrays a second time merely to change phases.
    $readinessPath = Join-Path $fullRoot 'SCIENTIFIC_INPUT_READINESS_PLAN.json'
    $readiness = [ordered]@{
        schema='simplex_t_orchestration_plan_v1'
        steps=@([ordered]@{id='verified_freeze_inputs'; run=$last.verify; resume=$last.verify; verify=$last.verify})
    } | ConvertTo-Json -Depth 16
    if (Test-Path -LiteralPath $readinessPath) {
        if ((Get-Content -LiteralPath $readinessPath -Raw) -cne $readiness) { throw 'Readiness plan changed.' }
    } else { [IO.File]::WriteAllText($readinessPath, $readiness, [Text.UTF8Encoding]::new($false)) }
    $sciencePlan = Join-Path $fullRoot 'SCIENTIFIC_PLAN.json'
    $generationLog = Join-Path $fullRoot ('plan_generation_' + [Guid]::NewGuid().ToString('N') + '.log')
    & $python -B $builder --prerequisites $readinessPath `
        --prerequisites-sha256 (Get-FileHash -LiteralPath $readinessPath -Algorithm SHA256).Hash.ToLowerInvariant() `
        --freeze-launch $config.freeze_launch `
        --freeze-launch-sha256 (Get-FileHash -LiteralPath $config.freeze_launch -Algorithm SHA256).Hash.ToLowerInvariant() `
        --reconciliation $config.reconciliation.path --reconciliation-sha256 $config.reconciliation.sha256 `
        --analysis-commit $config.code_commit --campaign-root $config.campaign_root --run-root $scienceRoot `
        --output $sciencePlan --other-reserved-bytes $config.other_reserved_bytes `
        --own-reserved-bytes $config.own_reserved_bytes *> $generationLog
    if ($LASTEXITCODE -ne 0) { throw "Scientific plan materialization failed; see $generationLog" }
    Assert-FullBoundary
    & $shell -NoProfile -File $orchestrator -Plan $sciencePlan `
        -PlanSha256 (Get-FileHash -LiteralPath $sciencePlan -Algorithm SHA256).Hash.ToLowerInvariant() `
        -PythonExecutable $python -RunRoot $scienceRoot `
        -ResourceRetrySeconds $ResourceRetrySeconds -MaxResourceRetries $MaxResourceRetries
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Assert-FullBoundary
    Write-Output 'Preparation and scientific plan, including the T6 delivery verifier, completed successfully.'
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Set-Location -LiteralPath $previousLocation.Path
}
