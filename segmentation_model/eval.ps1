# Evaluate a PRNet checkpoint. Defaults: newest runs/*/best.pt, test split.
# Usage:
#   .\segmentation_model\eval.ps1 --save-overlays 6
#   .\segmentation_model\eval.ps1 --wandb online
#   .\segmentation_model\eval.ps1 --ckpt segmentation_model\runs\prnet-xxx\last.pt --split val
$ErrorActionPreference = "Stop"
$py = "C:\Users\pierre bassily\miniconda3\envs\endo\python.exe"
$a = @($args)
if ($a -notcontains "--ckpt") {
    $ck = Get-ChildItem "$PSScriptRoot\runs\*\best.pt", "$PSScriptRoot\runs\best.pt" -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $ck) { throw "no checkpoint found under runs/ - train first or pass --ckpt" }
    $a += @("--ckpt", $ck.FullName)
}
& $py "$PSScriptRoot\eval.py" @a
exit $LASTEXITCODE
