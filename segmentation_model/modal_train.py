"""Train PRNet on Modal (cloud GPU), dataset pulled from Google Drive once into a Volume.

Setup (local):
    pip install modal && modal token new
    # Share the PRAD dataset on Google Drive as ONE zip (contains image/, label/,
    # "label distribution.txt"), set link to "anyone with the link". Grab its file id.

Fetch data into the Volume (one time, ~10 min):
    GDRIVE_ID=<file-id> modal run segmentation_model/modal_train.py::fetch_data

Train:
    modal run segmentation_model/modal_train.py::train --args "--epochs 60 --name prnet-modal --wandb offline"

Pull checkpoints back:
    modal volume get prad-runs / segmentation_model/runs/

ponytail: skipped a CLI wrapper, config, and multi-GPU logic. Add if you outgrow one A10G.
"""
import os
import subprocess
import sys

import modal

GPU = os.environ.get("MODAL_GPU", "A10G")
REPO = "/root/repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install("gdown")
    .pip_install_from_requirements("segmentation_model/requirements-train.txt")
    .add_local_dir("segmentation_model", f"{REPO}/segmentation_model", copy=True)
)

app = modal.App("prad-prnet")
data_vol = modal.Volume.from_name("prad-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("prad-runs", create_if_missing=True)
VOLS = {"/data": data_vol, f"{REPO}/segmentation_model/runs": runs_vol}


@app.function(image=image, volumes={"/data": data_vol}, timeout=60 * 60)
def fetch_data():
    """Download + unzip the Google Drive dataset zip into the prad-data Volume."""
    file_id = os.environ["GDRIVE_ID"]
    if os.path.exists("/data/image"):
        print("/data already populated; skipping. Delete the Volume to re-fetch.")
        return
    import gdown
    gdown.download(id=file_id, output="/data/prad.zip", quiet=False)
    subprocess.run(["unzip", "-q", "/data/prad.zip", "-d", "/data"], check=True)
    os.remove("/data/prad.zip")
    # flatten if the zip had a single top-level dir
    if not os.path.exists("/data/image"):
        subs = [d for d in os.listdir("/data") if os.path.isdir(f"/data/{d}")]
        if len(subs) == 1:
            inner = f"/data/{subs[0]}"
            for name in os.listdir(inner):
                os.rename(f"{inner}/{name}", f"/data/{name}")
    assert os.path.exists("/data/image") and os.path.exists("/data/label"), os.listdir("/data")
    data_vol.commit()
    print("done:", os.listdir("/data"))


@app.function(image=image, gpu=GPU, volumes=VOLS, timeout=24 * 60 * 60,
              secrets=[modal.Secret.from_dict({"PRAD_DATA_ROOT": "/data"})])
def train(args: str = "--epochs 60 --wandb offline"):
    os.chdir(f"{REPO}/segmentation_model")
    subprocess.run([sys.executable, "make_splits.py"], check=True)
    subprocess.run([sys.executable, "train.py", *args.split()], check=True)
    runs_vol.commit()
