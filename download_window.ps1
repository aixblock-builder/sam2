# PowerShell script to download SAM 2.1 checkpoints for Windows
# Save this file as download_window.bash and run with PowerShell

$ErrorActionPreference = "Stop"

# Define the URLs for SAM 2.1 checkpoints
$SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
$checkpoints = @{
    "sam2.1_hiera_tiny.pt"      = "$SAM2p1_BASE_URL/sam2.1_hiera_tiny.pt"
    "sam2.1_hiera_small.pt"     = "$SAM2p1_BASE_URL/sam2.1_hiera_small.pt"
    "sam2.1_hiera_base_plus.pt" = "$SAM2p1_BASE_URL/sam2.1_hiera_base_plus.pt"
    "sam2.1_hiera_large.pt"     = "$SAM2p1_BASE_URL/sam2.1_hiera_large.pt"
}

$CHECKPOINT_DIR = "checkpoints"
if (-not (Test-Path $CHECKPOINT_DIR)) {
    New-Item -ItemType Directory -Path $CHECKPOINT_DIR | Out-Null
}

foreach ($filename in $checkpoints.Keys) {
    $url = $checkpoints[$filename]
    $target = Join-Path $CHECKPOINT_DIR $filename
    Write-Host "Downloading $filename from $url ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $target
        Write-Host "Downloaded $filename successfully."
    } catch {
        Write-Host "Failed to download $filename from $url" -ForegroundColor Red
        exit 1
    }
}
Write-Host "All checkpoints downloaded!" -ForegroundColor Green
