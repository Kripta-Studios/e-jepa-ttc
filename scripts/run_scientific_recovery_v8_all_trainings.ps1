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
$BaseRoot=Join-Path $RepoRoot 'artifacts/scientific_recovery_v8'
$ResultsRoot=Join-Path $BaseRoot 'results'
$MasterLog=Join-Path $BaseRoot 'master_logs'
$MasterState=Join-Path $BaseRoot 'master_state.json'
New-Item -ItemType Directory -Force $MasterLog,$ResultsRoot | Out-Null

function Write-State([string]$Stage,[string]$Status,[string]$Detail='') {
    $value=[ordered]@{timestamp=(Get-Date).ToUniversalTime().ToString('o');stage=$Stage;status=$Status;detail=$Detail;device=$Device;max_parallel=$MaxParallel;sealed_evaluation='closed'}
    $tmp="$MasterState.tmp"; $value|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $tmp -Encoding utf8; Move-Item -Force $tmp $MasterState
}
function Quote-Cmd([string[]]$Args) { return ($Args | ForEach-Object { if($_ -match '[\s"]'){ '"'+($_ -replace '"','\"')+'"' }else{$_} }) -join ' ' }
function Start-LoggedProcess([string]$Label,[string]$File,[string[]]$Args) {
    $dir=Join-Path $MasterLog $Label; New-Item -ItemType Directory -Force $dir|Out-Null
    $stdout=Join-Path $dir 'stdout.log'; $stderr=Join-Path $dir 'stderr.log'; $command=Join-Path $dir 'command.txt'
    @("START_UTC=$((Get-Date).ToUniversalTime().ToString('o'))","CWD=$RepoRoot","COMMAND=$File $(Quote-Cmd $Args)") | Add-Content -LiteralPath $command -Encoding utf8
    if($DryRun){ Write-Host "[DRY] $Label :: $File $(Quote-Cmd $Args)"; return [pscustomobject]@{Label=$Label;Process=$null;Stdout=$stdout;Stderr=$stderr;Command=$command;Dry=$true} }
    $p=Start-Process -FilePath $File -ArgumentList $Args -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    return [pscustomobject]@{Label=$Label;Process=$p;Stdout=$stdout;Stderr=$stderr;Command=$command;Dry=$false}
}
function Wait-LoggedProcess($Job) {
    if($Job.Dry){ return }
    $Job.Process.WaitForExit(); $code=$Job.Process.ExitCode
    @("END_UTC=$((Get-Date).ToUniversalTime().ToString('o'))","EXIT_CODE=$code") | Add-Content -LiteralPath $Job.Command -Encoding utf8
    if($code -ne 0){ throw "Process '$($Job.Label)' failed with exit $code. See $($Job.Stderr)" }
}
function Invoke-Logged([string]$Label,[string]$File,[string[]]$Args){ $job=Start-LoggedProcess $Label $File $Args; Wait-LoggedProcess $job }
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

Write-State 'preflight' 'running'
# Re-freeze because the completion patch changes conditional model/config contracts.
Invoke-Logged '00_freeze' 'uv' (UvArgs @('scripts/freeze_scientific_recovery_v8_configs.py','--protocol',$Protocol))
Invoke-Logged '01_verify_freeze' 'uv' (UvArgs @('scripts/freeze_scientific_recovery_v8_configs.py','--protocol',$Protocol,'--verify'))
if(-not $SkipFocusedTests){
    Invoke-Logged '02_focused_tests' 'uv' @('run','--no-sync','pytest','-q','tests/unit/test_scientific_recovery_v8_temporal.py','tests/unit/test_scientific_recovery_v8_router.py','tests/unit/test_scientific_recovery_v8_autopsy.py','tests/unit/test_scientific_recovery_v8_aggregate.py','tests/unit/test_scientific_recovery_v8_jepa.py','tests/unit/test_scientific_recovery_v8_jepa_attribution.py','tests/unit/test_scientific_recovery_v8_jobs.py','tests/integration/test_scientific_recovery_v8_trainer_smoke.py','tests/integration/test_scientific_recovery_v8_router_smoke.py','tests/integration/test_scientific_recovery_v8_jepa_smoke.py')
}
Write-State 'preflight' 'completed'

# A: no optimizer steps. Materialize exact V4 inputs, replay A5/C2F, bind frozen Garl OOF, aggregate.
Write-State 'autopsy' 'running'
$AInput=Join-Path $BaseRoot 'autopsy/inputs'; $AReplay=Join-Path $BaseRoot 'autopsy/replays'; $AOut=Join-Path $BaseRoot 'autopsy/mechanism_autopsy.json'
Invoke-Logged '10_autopsy_materialize' 'uv' (UvArgs @('scripts/materialize_scientific_recovery_v8_autopsy_inputs.py','--protocol',$Protocol,'--output-dir',$AInput))
Invoke-Logged '11_autopsy_replay_a5_c2f' 'uv' (UvArgs @('scripts/replay_scientific_recovery_v8_mechanisms.py','--protocol',$Protocol,'--replay-input-root',$AInput,'--output-dir',$AReplay,'--models','a5','c2f','--device',$Device))
Invoke-Logged '12_autopsy_garl_bind' 'uv' (UvArgs @('scripts/materialize_scientific_recovery_v8_garl_autopsy_comparator.py','--protocol',$Protocol,'--output-dir',(Join-Path $AReplay 'garl')))
Invoke-Logged '13_autopsy_aggregate' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8_autopsy.py','--a5-manifest',(Join-Path $AReplay 'a5/manifest.json'),'--c2f-manifest',(Join-Path $AReplay 'c2f/manifest.json'),'--garl-manifest',(Join-Path $AReplay 'garl/manifest.json'),'--protocol',$Protocol,'--output',$AOut,'--bootstrap-resamples','5000'))
Write-State 'autopsy' 'completed'

# R: prospective nested router. Internally executes 18 inner + 6 outer expert fits with max two concurrent.
Write-State 'router' 'running'
Invoke-Logged '20_router_nested' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_nested_router.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--device',$Device,'--max-parallel',[string]$MaxParallel,'--execute'))
Write-State 'router' 'completed'

# B1+B2: fixed temporal frontends. Cache materialization is automatic and train-only.
Write-State 'temporal' 'running'
Invoke-Logged '30_temporal_b1_b2' 'uv' (UvArgs @('scripts/run_scientific_recovery_v8_temporal.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--arm','all','--device',$Device,'--max-parallel',[string]$MaxParallel,'--eap-root',$EapRoot,'--garlttc-root',$GarlTtcRoot))
Invoke-Logged '31_aggregate_seed7_initial' 'uv' (UvArgs @('scripts/aggregate_scientific_recovery_v8.py','--protocol',$Protocol,'--manifest',$Manifest,'--results-root',$ResultsRoot,'--output',(Join-Path $ResultsRoot 'aggregate_seed7.json'),'--resamples','5000'))
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
