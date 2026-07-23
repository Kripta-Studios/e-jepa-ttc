$ErrorActionPreference = "Stop"

$Branch = "scientific-recovery-v2"
$Repo = "Kripta-Studios/e-jepa-ttc"

Write-Host "Fetching local commit for branch $Branch..."
$LocalCommit = git rev-parse HEAD
$LocalCommit = $LocalCommit.Trim()

if (-not $LocalCommit) {
    Write-Error "Could not determine local commit."
}

Write-Host "Local Commit: $LocalCommit"

# We use GitHub Actions API to find the latest workflow run for this branch
# GET /repos/{owner}/{repo}/actions/runs?branch={branch}
$Uri = "https://api.github.com/repos/$Repo/actions/runs?branch=$Branch&per_page=1"

$Headers = @{
    "Accept" = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

# If user has GITHUB_TOKEN set in environment, use it to avoid rate limits or access private repo
if ($env:GITHUB_TOKEN) {
    $Headers["Authorization"] = "Bearer $($env:GITHUB_TOKEN)"
}

Write-Host "Querying GitHub API..."
try {
    $Response = Invoke-RestMethod -Uri $Uri -Headers $Headers -Method Get
} catch {
    Write-Error "Failed to reach GitHub API: $_"
}

if (-not $Response.workflow_runs -or $Response.workflow_runs.Count -eq 0) {
    Write-Error "No workflow runs found for branch $Branch."
}

$LatestRun = $Response.workflow_runs[0]
$RemoteCommit = $LatestRun.head_sha
$Status = $LatestRun.status
$Conclusion = $LatestRun.conclusion
$HtmlUrl = $LatestRun.html_url

Write-Host "Remote Commit: $RemoteCommit"
Write-Host "Status: $Status"
Write-Host "Conclusion: $Conclusion"
Write-Host "URL: $HtmlUrl"

$Payload = @{
    branch = $Branch
    local_commit = $LocalCommit
    remote_commit = $RemoteCommit
    status = $Status
    conclusion = $Conclusion
    url = $HtmlUrl
    match = ($LocalCommit -eq $RemoteCommit)
    passed = ($Status -eq "completed" -and $Conclusion -eq "success" -and $LocalCommit -eq $RemoteCommit)
}

$OutDir = "artifacts/audit"
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$Payload | ConvertTo-Json -Depth 5 | Set-Content "$OutDir/current_remote_state.json"

if (-not $Payload.passed) {
    Write-Error "CI is not in a passed state for the current commit. Status: $Status, Conclusion: $Conclusion, Match: $($Payload.match)"
} else {
    Write-Host "Remote CI verified green for $LocalCommit." -ForegroundColor Green
}
