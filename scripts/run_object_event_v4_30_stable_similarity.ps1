[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force,
    [int]$DiagnosticSamples = 0
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    # This is the complete protocol. Development files are passed as paths but
    # the analyzer does not stat/read them unless actual full-mode OOF promotes.
    $DiagnosticRequested = $PSBoundParameters.ContainsKey('DiagnosticSamples')
    if ($DiagnosticRequested -and $DiagnosticSamples -lt 6) {
        throw "-DiagnosticSamples must be at least 6 when explicitly supplied"
    }
    $Mode = if ($DiagnosticRequested) { "diagnostic" } else { "full_oof" }
    $Out = if ($DiagnosticRequested) {
        "artifacts\debug\object_event_v4_30_diagnostic"
    } else {
        "artifacts\debug\object_event_v4_30_stable_similarity"
    }
    Write-Host "v4.30 mode=$Mode output=$Out"
    $Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
    $V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
    $Config = "configs\experiment\e_jepa_garl_object_event_stable_similarity_v4_30.yaml"
    $V429 = "artifacts\debug\object_event_v4_29_local_affine\summary.json"
    $Checkpoints = @(
        "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
        "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
        "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
    )
    $Preflight = @("scripts\preflight_object_event_v4_30.py", "--cache-manifest", $Cache, "--v48-config", $V48Config, "--config", $Config, "--v429-summary", $V429)
    foreach ($Checkpoint in $Checkpoints) { $Preflight += @("--adapted-checkpoint", $Checkpoint) }
    & uv run --no-sync python @Preflight
    if ($LASTEXITCODE -ne 0) { throw "v4.30 preflight failed with exit code $LASTEXITCODE" }
    $V410 = "artifacts\debug\object_event_v4_10_multiseed\summary.json"
    $Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
    $Run = @("scripts\analyze_object_event_v4_30_stable_similarity.py", "--cache-manifest", $Cache, "--v48-config", $V48Config, "--config", $Config, "--v429-summary", $V429, "--v410-summary", $V410, "--ensemble-validation", $Validation, "--output-dir", $Out, "--device", $Device)
    if ($DiagnosticRequested) { $Run += @("--diagnostic-samples", "$DiagnosticSamples") }
    foreach ($Checkpoint in $Checkpoints) { $Run += @("--adapted-checkpoint", $Checkpoint) }
    if ($Force) { $Run += "--force" }
    & uv run --no-sync python @Run
    if ($LASTEXITCODE -ne 0) { throw "v4.30 analyzer failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
