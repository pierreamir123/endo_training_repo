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

GPU = os.environ.get("MODAL_GPU", "A10G")  # 24 GB, native bf16 (no fp16 overflow); T4 is cheaper but fp16-only
REPO = "/root/repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install("gdown")
    .pip_install_from_requirements("segmentation_model/requirements-train.txt")
    .add_local_dir("segmentation_model", f"{REPO}/segmentation_model", copy=True,
                   ignore=["runs", "**/__pycache__", "*.ipynb", ".env*", "*.pt"])
)

app = modal.App("prad-prnet")
data_vol = modal.Volume.from_name("prad-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("prad-runs", create_if_missing=True)
VOLS = {"/data": data_vol, f"{REPO}/segmentation_model/runs": runs_vol}


@app.function(image=image, volumes={"/data": data_vol}, timeout=60 * 60,
              secrets=[modal.Secret.from_name("endo")])
def fetch_data():
    """Download + unzip the Google Drive dataset zip into the prad-data Volume."""
    file_id = os.environ["GDRIVE_ID"]
    if "/d/" in file_id:  # accept a full .../d/<ID>/view URL too
        file_id = file_id.split("/d/")[1].split("/")[0]
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


# ponytail: timeout doubles as a spend cap (A10G + 6 CPU + 12 GiB ~ $1.48/h); weights are
# committed every 3 min, so hitting it still leaves best/last.pt. MODAL_TIMEOUT_H=3.3 -> ~$5.
TIMEOUT = int(float(os.environ.get("MODAL_TIMEOUT_H", 24)) * 3600)


# RAM cache only (~9.5 GB, rebuilt each run in ~8 min). A disk cache on the Volume was tried:
# reads ran ~4 files/s (slower than recomputing) and a stopped run left truncated files.
@app.function(image=image, gpu=GPU, volumes=VOLS, timeout=TIMEOUT, cpu=6, memory=12288,
              secrets=[modal.Secret.from_name("endo"),
                       modal.Secret.from_dict({"PRAD_DATA_ROOT": "/data"})])
def train(args: str = "--epochs 18 --batch-size 8 --workers 6 --cache ram --wandb offline"):
    import threading

    os.chdir(f"{REPO}/segmentation_model")
    assert os.path.exists("splits.txt"), "splits.txt missing"  # balanced split ships with the code; do NOT regenerate

    # ponytail: train.py writes best/last.pt every epoch; commit the Volume every 3 min
    # so a crash or timeout still leaves the latest weights recoverable.
    stop = threading.Event()
    def flush():
        while not stop.wait(180):
            runs_vol.commit()
    t = threading.Thread(target=flush, daemon=True)
    t.start()
    try:
        subprocess.run([sys.executable, "train.py", *args.split()], check=True)
    finally:
        stop.set()
        runs_vol.commit()
