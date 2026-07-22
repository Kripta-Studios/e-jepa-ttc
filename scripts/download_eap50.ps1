[CmdletBinding()]
param(
    [string]$Destination = "datasets/eap-50",
    [switch]$IncludeRgb
)

$ErrorActionPreference = "Stop"
$dataset = "NAIL-HNU/eAP-dataset"
$sequences = @(
    "2cyv0Oedzg",
    "6h5yRW2LGc",
    "pBqGOb2vYq",
    "mHGFBekt7X",
    "DGqicHUGWb",
    "OBneIVg4Cw",
    "qGsgzl4Q8B",
    "qoohcdtLDH"
)

uv run hf download $dataset README.md data/train.parquet data/test.parquet `
    --repo-type dataset --local-dir $Destination --max-workers 1

foreach ($sequence in $sequences) {
    uv run hf download $dataset "data/train/$sequence/events.h5" `
        "data/train/$sequence/labels.parquet" `
        --repo-type dataset --local-dir $Destination --max-workers 1
}

if ($IncludeRgb) {
    foreach ($sequence in @("2cyv0Oedzg", "DGqicHUGWb")) {
        uv run hf download $dataset "data/train/$sequence/rgb_shards/*" `
            --repo-type dataset --local-dir $Destination --max-workers 1
    }
}
