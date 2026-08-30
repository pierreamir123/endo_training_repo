"""Evaluate a PRNet checkpoint on a split.

  python segmentation_model/eval.py --ckpt segmentation_model/runs/best.pt --save-overlays 6
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", "Mean of empty slice")  # classes absent in a small split
import torch
import wandb
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Compose, EnsureType
from PIL import Image

from dataset import NUM_CLASSES, CROP, class_names, list_pairs, eval_transforms
from model import PRNet
from wbutil import wandb_init

# tab10 palette (same colours as data_processssing/explore_dataset.ipynb)
_TAB10 = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
    [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
], dtype=np.uint8)


def overlay(img_chw, mask_hw, alpha=0.5):
    rgb = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.float32)
    for c in range(1, NUM_CLASSES):
        m = mask_hw == c
        rgb[m] = (1 - alpha) * rgb[m] + alpha * _TAB10[c]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-overlays", type=int, default=0)
    p.add_argument("--wandb", default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--name", default=None, help="W&B run name (default: <ckpt>-eval)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    return p.parse_args()


def main():
    a = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = class_names()
    run_name = a.name or f"{os.path.splitext(os.path.basename(a.ckpt))[0]}-{a.split}-eval"
    run = wandb_init(run_name, "eval", vars(a), a.wandb)

    ds = Dataset(list_pairs(a.split, a.limit), eval_transforms)
    dl = DataLoader(ds, batch_size=1, num_workers=2)

    model = PRNet(in_channels=3, num_classes=NUM_CLASSES, input_size=CROP).to(device)
    ckpt = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {a.ckpt} (epoch {ckpt.get('epoch')}, val_dice {ckpt.get('val_dice')})")

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_lbl = Compose([EnsureType(), AsDiscrete(to_onehot=NUM_CLASSES)])
    dice = DiceMetric(include_background=False, reduction="none")
    iou = MeanIoU(include_background=False, reduction="none")

    ov_dir = os.path.join(a.out, "overlays")
    if a.save_overlays:
        os.makedirs(ov_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            with torch.amp.autocast(device, enabled=device == "cuda"):
                out = sliding_window_inference(x, (CROP, CROP), 2, model)
            p = [post_pred(o) for o in out]
            t = [post_lbl(o) for o in y]
            dice(y_pred=p, y=t)
            iou(y_pred=p, y=t)
            if i < a.save_overlays:
                img = x[0].cpu().numpy()
                pred_img = overlay(img, out[0].argmax(0).cpu().numpy())
                gt_img = overlay(img, y[0, 0].cpu().numpy())
                pred_img.save(os.path.join(ov_dir, f"{i:03d}_pred.png"))
                gt_img.save(os.path.join(ov_dir, f"{i:03d}_gt.png"))
                run.log({f"overlay/{i:03d}": [wandb.Image(pred_img, caption="pred"),
                                             wandb.Image(gt_img, caption="gt")]})

    d = np.nanmean(dice.aggregate().cpu().numpy(), axis=0)   # (NUM_CLASSES-1,)
    j = np.nanmean(iou.aggregate().cpu().numpy(), axis=0)
    print(f"\n{'class':<26} {'Dice':>7} {'IoU':>7}")
    tbl = wandb.Table(columns=["class", "dice", "iou"])
    for k in range(1, NUM_CLASSES):
        print(f"{names[k]:<26} {d[k-1]:>7.4f} {j[k-1]:>7.4f}")
        tbl.add_data(names[k], float(d[k - 1]), float(j[k - 1]))
    mean_d, mean_j = float(np.nanmean(d)), float(np.nanmean(j))
    print(f"{'MEAN (fg)':<26} {mean_d:>7.4f} {mean_j:>7.4f}")
    run.log({"per_class": tbl})
    run.summary.update({"mean/dice": mean_d, "mean/iou": mean_j,
                        "ckpt": a.ckpt, "split": a.split})
    run.finish()
    if a.save_overlays:
        print(f"\noverlays -> {ov_dir}")


if __name__ == "__main__":
    main()
