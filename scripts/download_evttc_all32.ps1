param(
    [string]$DestinationRoot = "datasets\evttc_complete_staging",
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "No se encontró el Python del entorno virtual: $Python"
}

$Items = @(
    # CCRs-1, 100 %
    [pscustomobject]@{
        Name = "CCRs-1-low-100%"
        Rel  = "CCRs-1\low-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1J5urgKdVasGXghyO2m7Mt0fpzEiAEXRy?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-medium-100%"
        Rel  = "CCRs-1\medium-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1Ver9SRkHBFI_mX2f4GIsphYeDe3qoyvw?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-high-100%"
        Rel  = "CCRs-1\high-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1y9mJoiRcKiUfdLs2qpEPpsLX3EA-Iosa?usp=drive_link"
    },

    # CCRs-1, 50 %
    [pscustomobject]@{
        Name = "CCRs-1-low-50%"
        Rel  = "CCRs-1\low-50\overlap-50"
        Url  = "https://drive.google.com/drive/folders/1NP3nXVYGnFfuMQIjk81NtlqSPTWw-b7H?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-medium-50%"
        Rel  = "CCRs-1\medium-50\overlap-50"
        Url  = "https://drive.google.com/drive/folders/1BlXJSvmEn5ByppM6E9VkvYCg3HHk23Br?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-high-50%"
        Rel  = "CCRs-1\high-50\overlap-50"
        Url  = "https://drive.google.com/drive/folders/1XTrzq5tB7kcIEZ4oARMAJL8R0luO1H2W?usp=drive_link"
    },

    # CCRs-1, 0 %
    [pscustomobject]@{
        Name = "CCRs-1-low-0%"
        Rel  = "CCRs-1\low-0\overlap-0"
        Url  = "https://drive.google.com/drive/folders/1ZBH9URjHjnuNknj4NzequmkEJlJqxTAV?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-medium-0%"
        Rel  = "CCRs-1\medium-0\overlap-0"
        Url  = "https://drive.google.com/drive/folders/1Z-aaNtxeOLmrHQqlEL5txE7eT1snLP0l?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-1-high-0%"
        Rel  = "CCRs-1\high-0\overlap-0"
        Url  = "https://drive.google.com/drive/folders/1ZcQo3gfEbZDyfgREB6T3zMP9NXEEXfOw?usp=drive_link"
    },

    # CCRs-2
    [pscustomobject]@{
        Name = "CCRs-2-low-100%"
        Rel  = "CCRs-2\low-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/11RmFgYZQ7USEw3X2lvKr7erPG3xVTWi-?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-2-medium-100%"
        Rel  = "CCRs-2\medium-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1q8jmpGxIUgdF_eW3gIa-KFoIal3hdim4?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-2-high-100%"
        Rel  = "CCRs-2\high-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/13GsC86zHtdkBCSaIM3p2jompMlDPV-xd?usp=drive_link"
    },

    # CCRs-3
    [pscustomobject]@{
        Name = "CCRs-3-low-100%"
        Rel  = "CCRs-3\low-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1KcjacIdsHa9GwqIMasOJ0Lc3kaHncPdO?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-3-medium-100%"
        Rel  = "CCRs-3\medium-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1Upbw1fLIz0tMWmlk4-xGlZ3_d5YnUlFC?usp=drive_link"
    },

    # CCRs-side
    [pscustomobject]@{
        Name = "CCRs-side-low"
        Rel  = "CCRs-side\low"
        Url  = "https://drive.google.com/drive/folders/1pCHyGpoAOZIxq1r6FUjh0nJLMXM4nyJD?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-side-medium"
        Rel  = "CCRs-side\medium"
        Url  = "https://drive.google.com/drive/folders/1GPQR7Uga3PlviyQB_sGhsAK2zbpjYDY3?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRs-side-high"
        Rel  = "CCRs-side\high"
        Url  = "https://drive.google.com/drive/folders/1-tE9SNZ2pFI4S6lALdteVcZ-1xFD0taN?usp=drive_link"
    },

    # CCRm, 100 %
    [pscustomobject]@{
        Name = "CCRm-low-100%"
        Rel  = "CCRm\low-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1UHf4505XRNOiSPgIXaAFmxi14rH8ecKz?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRm-medium-100%"
        Rel  = "CCRm\medium-100\overlap-100"
        Url  = "https://drive.google.com/drive/folders/1OrNI60pLvUyfyfToGwzlhry19k6qvNdS?usp=drive_link"
    },

    # CCRm, 50 %
    [pscustomobject]@{
        Name = "CCRm-low-50%"
        Rel  = "CCRm\low-50\overlap-50"
        Url  = "https://drive.google.com/drive/folders/110JPPzddir06nqmZL4n-NJuz2G20B2uQ?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRm-medium-50%"
        Rel  = "CCRm\medium-50\overlap-50"
        Url  = "https://drive.google.com/drive/folders/1rDNilcr2vaFh6n-vEk0JpUfUc_RX59a2?usp=drive_link"
    },

    # CCRm, 0 %
    [pscustomobject]@{
        Name = "CCRm-low-0%"
        Rel  = "CCRm\low-0\overlap-0"
        Url  = "https://drive.google.com/drive/folders/1og1rXWX5KaVILCnqEcDTBHQKWdvRpguh?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CCRm-medium-0%"
        Rel  = "CCRm\medium-0\overlap-0"
        Url  = "https://drive.google.com/drive/folders/10ruBh11psIRu75goBdXhI3q-XSVxEr7Q?usp=drive_link"
    },

    # CPLA
    [pscustomobject]@{
        Name = "CPLA-low"
        Rel  = "CPLA\low"
        Url  = "https://drive.google.com/drive/folders/1MD0ilHiS6bFblFPhi7NFkko4uJsEpAkt?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPLA-medium"
        Rel  = "CPLA\medium"
        Url  = "https://drive.google.com/drive/folders/1drriQjvLS9SKETAc-K-MWyAEUXA-67sb?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPLA-high"
        Rel  = "CPLA\high"
        Url  = "https://drive.google.com/drive/folders/1Q_wys7p20NHdatK84ZdRUh0UgokOB4hO?usp=drive_link"
    },

    # CPNA
    [pscustomobject]@{
        Name = "CPNA-low"
        Rel  = "CPNA\low"
        Url  = "https://drive.google.com/drive/folders/1LsCVoPl1OOZquZebC0wFQ39zSvgxNHzn?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPNA-medium"
        Rel  = "CPNA\medium"
        Url  = "https://drive.google.com/drive/folders/1uV-HsnmQIev5FX7svmN3aKwKqQwE7pfI?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPNA-high"
        Rel  = "CPNA\high"
        Url  = "https://drive.google.com/drive/folders/131ZpXolV65uJfGQwOh6-X5Xz5a9JRpFf?usp=drive_link"
    },

    # CPNAO
    [pscustomobject]@{
        Name = "CPNAO-low"
        Rel  = "CPNAO\low"
        Url  = "https://drive.google.com/drive/folders/1_zRxTYKFZ2Iq9YFT-sNDyVWn1tAxWxXZ?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPNAO-medium"
        Rel  = "CPNAO\medium"
        Url  = "https://drive.google.com/drive/folders/1dZ7-LgkAKELbCS0ZCPesvSJUPntoYZQq?usp=drive_link"
    },
    [pscustomobject]@{
        Name = "CPNAO-high"
        Rel  = "CPNAO\high"
        Url  = "https://drive.google.com/drive/folders/1S3J61f1eH6FkBFm4yZzsBQ3ExZOhqNd6?usp=drive_link"
    }
)

$DestinationRootAbsolute = Join-Path $RepoRoot $DestinationRoot
New-Item -ItemType Directory -Force -Path $DestinationRootAbsolute | Out-Null

foreach ($Item in $Items) {
    if ($Only.Count -gt 0 -and $Only -notcontains $Item.Name) {
        continue
    }

    $Destination = Join-Path $DestinationRootAbsolute $Item.Rel
    $Marker = Join-Path $Destination "_DOWNLOAD_COMPLETE.txt"

    if (Test-Path $Marker) {
        Write-Host "SKIP  $($Item.Name): ya está validado." -ForegroundColor DarkGray
        continue
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    Write-Host ""
    Write-Host "DESCARGANDO $($Item.Name)" -ForegroundColor Cyan
    Write-Host "Destino: $Destination"

    & $Python -m gdown $Item.Url -O $Destination --folder

    if ($LASTEXITCODE -ne 0) {
        throw "Falló la descarga de $($Item.Name)."
    }

    $EventHdf5 = Get-ChildItem $Destination -Recurse -File -Filter "*.hdf5" |
        Where-Object { $_.Name -ne "gt.hdf5" } |
        Sort-Object Length -Descending |
        Select-Object -First 1

    $Ttc = Get-ChildItem $Destination -Recurse -File -Filter "ttc.csv" |
        Select-Object -First 1

    $Labels = Get-ChildItem $Destination -Recurse -File -Filter "*.json"

    $Video = Get-ChildItem $Destination -Recurse -File -Filter "*.mp4" |
        Select-Object -First 1

    $Bag = Get-ChildItem $Destination -Recurse -File -Filter "*.bag" |
        Select-Object -First 1

    $Depth = Get-ChildItem $Destination -Recurse |
        Where-Object {
            $_.Name -eq "gt.hdf5" -or
            $_.Name -match "depth"
        } |
        Select-Object -First 1

    if (-not $EventHdf5) {
        throw "$($Item.Name): no se encontró el HDF5 de eventos."
    }
    if (-not $Ttc) {
        throw "$($Item.Name): no se encontró ttc.csv."
    }
    if ($Labels.Count -eq 0) {
        throw "$($Item.Name): no se encontraron anotaciones JSON."
    }

    $Summary = @(
        "sequence=$($Item.Name)"
        "downloaded_at=$([DateTime]::UtcNow.ToString('o'))"
        "event_hdf5=$($EventHdf5.FullName)"
        "ttc=$($Ttc.FullName)"
        "json_labels=$($Labels.Count)"
        "video_present=$([bool]$Video)"
        "rosbag_present=$([bool]$Bag)"
        "depth_present=$([bool]$Depth)"
    )

    $Summary | Set-Content $Marker -Encoding UTF8

    Write-Host "OK    $($Item.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Descarga solicitada terminada." -ForegroundColor Green
