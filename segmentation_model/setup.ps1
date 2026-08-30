# One-time: install training deps into the `endo` conda env.
# For GPU, edit requirements-train.txt note or run the cu124 line below first.
$ErrorActionPreference = "Stop"
$py = "C:\Users\pierre bassily\miniconda3\envs\endo\python.exe"
# GPU (CUDA 12.x) - uncomment if you have a working connection to download.pytorch.org:
# & $py -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r "$PSScriptRoot\requirements-train.txt"
& $py -c "import torch, monai, wandb; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('monai', monai.__version__, 'wandb', wandb.__version__)"

$env_file = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (-not (Test-Path $env_file)) {
    Copy-Item (Join-Path (Split-Path $PSScriptRoot -Parent) ".env.example") $env_file
    Write-Host "created .env - add your WANDB_API_KEY"
}
