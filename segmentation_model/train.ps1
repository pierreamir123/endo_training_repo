# Train PRNet on the PRAD dataset.
# Usage:
#   .\segmentation_model\train.ps1                       # full run, defaults
#   .\segmentation_model\train.ps1 --epochs 100 --batch-size 4
#   .\segmentation_model\train.ps1 --epochs 1 --limit 12 --cache-rate 0 --workers 0   # smoke
$ErrorActionPreference = "Stop"
$py = "C:\Users\pierre bassily\miniconda3\envs\endo\python.exe"
& $py "$PSScriptRoot\train.py" @args
exit $LASTEXITCODE
