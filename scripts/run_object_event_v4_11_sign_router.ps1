param(
    [string]$V410Summary = "artifacts\debug\object_event_v4_10_multiseed\summary.json",
    [string]$V49Root = "artifacts\debug\object_event_v4_9_fixed_fusion",
    [string]$Config = "configs\experiment\e_jepa_garl_object_event_sign_router_v4_11.yaml",
    [string]$OutputDir = "artifacts\debug\object_event_v4_11_sign_router",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label,
        [int[]]$AllowedExitCodes = @(0)
    )
    & uv run --no-sync python @Arguments
    $Code = $LASTEXITCODE
    if ($AllowedExitCodes -notcontains $Code) {
        throw "$Label failed with exit code $Code"
    }
    return $Code
}

Invoke-PythonChecked -Arguments @(
    "scripts\preflight_object_event_v4_11.py",
    "--v410-summary", $V410Summary,
    "--v49-run-root", $V49Root,
    "--config", $Config
) -Label "v4.11 preflight" | Out-Null

$Arguments = @(
    "scripts\analyze_object_event_v4_11_sign_router.py",
    "--v49-run-root", $V49Root,
    "--config", $Config,
    "--output-dir", $OutputDir
)
if ($Force) { $Arguments += "--force" }

$Code = Invoke-PythonChecked -Arguments $Arguments -AllowedExitCodes @(0, 2) -Label "v4.11 sign router"
exit $Code
