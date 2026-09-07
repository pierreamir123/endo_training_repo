# Modal cloud training

Runs `train.py` on a Modal GPU. Dataset comes from a Google Drive zip, pulled once
into a Modal Volume. Code lives in `modal_train.py`.

## One-time setup

```bash
pip install modal
modal token new                       # browser auth, writes ~/.modal.toml
cp segmentation_model/.env.modal.example segmentation_model/.env.modal
# then fill in GDRIVE_ID (dataset-zip file id from .../d/<ID>/view)
```

Load the env file before `modal run`:

```bash
set -a; . segmentation_model/.env.modal; set +a          # bash / git bash
```
```powershell
Get-Content segmentation_model/.env.modal | % { if ($_ -match '^(\w+)=(.+)') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2]) } }   # PowerShell
```

The Drive zip must contain `image/`, `label/`, and `label distribution.txt`
(zip the local `dataset/` folder; share "anyone with the link").

## Fetch the dataset (once, ~10 min)

```bash
modal run segmentation_model/modal_train.py::fetch_data
```

Lands in the `prad-data` Volume. Re-runs skip if already populated; to re-fetch:
`modal volume delete prad-data` first.

## Train

```bash
modal run --detach segmentation_model/modal_train.py::train \
  --args "--epochs 100 --batch-size 4 --name prnet-100 --wandb offline"
```

- `--detach` keeps the run alive on Modal if your machine sleeps/disconnects.
- GPU defaults to T4 (16 GB); override: `MODAL_GPU=A10G modal run ...`.
- If the Volume gets tight (50 GB free-tier cap, disk cache is ~21 GB) add
  `--cache none` to the args.
- Weights: `train.py` writes `runs/<name>/{best,last}.pt` every epoch to the
  `prad-runs` Volume; `modal_train.py` commits the Volume every 3 min and on exit,
  so a crash / OOM / 24 h timeout still leaves the latest `best.pt` recoverable.

## Get weights back

```bash
# best only, from this run
modal volume get prad-runs prnet-100/best.pt segmentation_model/runs/

# "latest" pointer at the Volume root (also best.ts.pt = TorchScript)
modal volume get prad-runs best.pt ./best.pt

# everything
modal volume get prad-runs / segmentation_model/runs/
```

Works mid-run too.

## Misc

```bash
modal app list                        # running apps
modal app logs prad-prnet             # stream logs of a detached run
modal volume ls prad-runs prnet-100   # what's saved
modal app stop prad-prnet             # kill a run
```
