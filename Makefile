# GNU make is installed in the `endo` conda env. Use it from Git Bash:
#     conda activate endo
#     make smoke
#     make train ARGS="--epochs 100 --name prnet-baseline"
#     make eval  ARGS="--save-overlays 6 --wandb online"
# (On plain PowerShell without `make`, use the segmentation_model/*.ps1 scripts.)
# The recipes call the endo python by full path, so activation is only needed to
# put `make` itself on PATH.
PY  := C:/Users/pierre bassily/miniconda3/envs/endo/python.exe
SM  := segmentation_model
ARGS ?=

.PHONY: setup smoke train eval explore

setup:
	"$(PY)" -m pip install -r requirements.txt -r $(SM)/requirements-train.txt

smoke:
	"$(PY)" $(SM)/dataset.py
	"$(PY)" $(SM)/model.py
	"$(PY)" $(SM)/train.py --epochs 1 --limit 12 --cache none --workers 0 --wandb disabled
	"$(PY)" $(SM)/eval.py --ckpt $(SM)/runs/last.pt --limit 8 --save-overlays 3

train:
	"$(PY)" $(SM)/train.py $(ARGS)

eval:
	"$(PY)" $(SM)/eval.py --ckpt $(SM)/runs/best.pt $(ARGS)

explore:
	"$(PY)" -m jupyter nbconvert --to notebook --execute --inplace $(SM)/../data_processssing/explore_dataset.ipynb
