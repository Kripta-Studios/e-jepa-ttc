param(
    [string]$Device = "cuda:0",
    [int]$Samples = 512,
    [int]$BatchSize = 8,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Out = "artifacts\metrics\a5_transport_preflight_v2"
$Audit = "artifacts\audit\a5_transport_preflight_v2_results"
$Zip = "artifacts\audit\a5_transport_preflight_v2_results.zip"
$Logs = "artifacts\logs\a5_transport_preflight_v2"

if ($Force) {
    Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $Audit -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $Zip -Force -ErrorAction SilentlyContinue
    Remove-Item $Logs -Recurse -Force -ErrorAction SilentlyContinue
}
if (Test-Path $Out) {
    throw "$Out already exists; pass -Force to replace it"
}
New-Item -ItemType Directory -Force $Out, $Audit, $Logs | Out-Null

function Invoke-Logged {
    param([string]$Name, [scriptblock]$Command)
    Write-Host "`n=== $Name ==="
    $log = Join-Path $Logs "$Name.log"
    New-Item -ItemType File -Force $log | Out-Null
    & $Command 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

# Do not require a globally clean worktree because result artifacts/patch files may
# be untracked.  Do require the tracked scientific code to be clean after the V2
# commit so runtime code identity is unambiguous.
$trackedDiff = git diff -- src scripts configs tests
if ($trackedDiff) {
    throw "Tracked src/scripts/configs/tests have uncommitted changes. Commit the V2 patch first."
}

Invoke-Logged "00_py_compile" {
    python -m py_compile `
        scripts/diagnose_a5_transport_preflight_v2.py `
        src/e_jepa_ttc/models/causal_scale_ttc.py `
        src/e_jepa_ttc/models/local_transport.py
}
Invoke-Logged "01_pytest_v2" {
    python -m pytest -q `
        tests/unit/test_a5_transport_preflight_v2.py `
        tests/unit/test_a5_local_transport.py `
        tests/unit/test_causal_scale_ttc.py
}
Invoke-Logged "02_preflight_v2" {
    python scripts/diagnose_a5_transport_preflight_v2.py `
        --output-dir $Out `
        --device $Device `
        --samples $Samples `
        --batch-size $BatchSize
}

# Bundle every result needed for the next decision, but no model checkpoints.
Copy-Item $Out (Join-Path $Audit "metrics") -Recurse -Force
Copy-Item $Logs (Join-Path $Audit "logs") -Recurse -Force

git rev-parse HEAD | Out-File (Join-Path $Audit "git_head.txt") -Encoding utf8
git status --short | Out-File (Join-Path $Audit "git_status.txt") -Encoding utf8
git log -10 --oneline --decorate | Out-File (Join-Path $Audit "git_log_10.txt") -Encoding utf8
git show --stat --oneline HEAD | Out-File (Join-Path $Audit "head_stat.txt") -Encoding utf8

$inventoryPath = Join-Path $Audit "inventory.csv"
$rows = Get-ChildItem $Audit -Recurse -File |
    Where-Object { $_.FullName -ne (Join-Path $Root $inventoryPath) } |
    ForEach-Object {
        [PSCustomObject]@{
            Path = $_.FullName.Substring((Resolve-Path $Audit).Path.Length + 1)
            Bytes = $_.Length
            SHA256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
        }
    }
$rows | Export-Csv $inventoryPath -NoTypeInformation -Encoding utf8

Compress-Archive -Force -Path "$Audit\*" -DestinationPath $Zip
Write-Host "`nA5-PREFLIGHT-V2 bundle: $((Resolve-Path $Zip).Path)"
Write-Host "SHA256: $((Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower())"
Write-Host "Decision:"
(Get-Content "$Out\a5_transport_preflight_v2.json" -Raw | ConvertFrom-Json).decision | ConvertTo-Json -Depth 8
