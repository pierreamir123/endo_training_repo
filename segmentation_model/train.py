"""Train PRNet on the PRAD dataset, logging to Weights & Biases.

  python segmentation_model/train.py --epochs 100 --name prnet-baseline
  python segmentation_model/train.py --epochs 1 --limit 40 --cache none --wandb disabled  # smoke

If CUDA OOMs on the 6 GB GPU, drop to --batch-size 1.

Weights land in  segmentation_model/runs/<run-name>/{best,last}.pt  and are also
copied to  runs/{best,last}.pt  (latest). Metrics go to W&B; weights are not uploaded.
Put WANDB_API_KEY in a .env at the repo root
(see wbutil.py). Default batch size 4 fits the 6 GB RTX 4050 (~3 GB peak).
"""
import argparse
import csv
import os
import sys

# Cut allocator fragmentation on small GPUs. setdefault so a user/MIG override wins;
# skip on Windows where expandable_segments is unsupported (noisy warning). Must precede `import torch`.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import shutil
import time
from datetime import datetime

import torch
import wandb
from tqdm import tqdm
from monai.data import CacheDataset, Dataset, DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType

from dataset import NUM_CLASSES, CROP, class_names, list_pairs, train_transforms, eval_transforms
from model import PRNet
from wbutil import wandb_init


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)  # DiceFocal peaks ~5 GB at bs=2 on the 6 GB RTX 4050; bs=4 OOMs
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--limit", type=int, default=None, help="cap samples per split (smoke)")
    p.add_argument("--cache", default="disk", choices=["none", "ram", "disk"],
                   help="disk: PersistentDataset (~21GB in out/<name>/cache); "
                        "ram: CacheDataset (needs ~20GB free RAM at rate 1.0); none: reload each epoch")
    p.add_argument("--cache-rate", type=float, default=1.0, help="fraction cached when --cache ram")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default=None, help="run name (default: prnet-<timestamp>)")
    p.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    return p.parse_args()


def main():
    a = parse()
    a.name = a.name or datetime.now().strftime("prnet-%Y%m%d-%H%M%S")
    out_dir = os.path.join(a.out, a.name)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("device: cuda |", torch.cuda.get_device_name(0),
              f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB | run:", a.name)
    else:
        print("device: cpu | run:", a.name)

    run = wandb_init(a.name, "train", vars(a), a.wandb)

    tr_pairs, va_pairs = list_pairs("train", a.limit), list_pairs("val", a.limit)
    if a.cache == "disk":
        cdir = os.path.join(out_dir, "cache")
        tr = PersistentDataset(tr_pairs, train_transforms, cache_dir=cdir)
        va = PersistentDataset(va_pairs, eval_transforms, cache_dir=cdir)
    elif a.cache == "ram":
        tr = CacheDataset(tr_pairs, train_transforms, cache_rate=a.cache_rate, num_workers=a.workers)
        va = Dataset(va_pairs, eval_transforms)
    else:
        tr = Dataset(tr_pairs, train_transforms)
        va = Dataset(va_pairs, eval_transforms)
    tr_dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
    va_dl = DataLoader(va, batch_size=1, num_workers=a.workers)

    model = PRNet(in_channels=3, num_classes=NUM_CLASSES, input_size=CROP).to(device)
    loss_fn = DiceFocalLoss(softmax=True, to_onehot_y=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_lbl = Compose([EnsureType(), AsDiscrete(to_onehot=NUM_CLASSES)])
    dice = DiceMetric(include_background=False, reduction="mean")
    names = class_names()

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "lr", "sec"])

    best, best_path = -1.0, os.path.join(out_dir, "best.pt")
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        pbar = tqdm(tr_dl, desc=f"epoch {epoch}/{a.epochs}", unit="batch")
        for batch in pbar:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        train_loss = tot / len(tr)

        if device == "cuda":
            torch.cuda.empty_cache()  # release train-phase blocks before sliding-window eval
        model.eval()
        dice.reset()
        with torch.no_grad():
            for batch in tqdm(va_dl, desc="val", unit="img", leave=False):
                x, y = batch["image"].to(device), batch["label"].to(device)
                with torch.amp.autocast(device, enabled=device == "cuda"):
                    out = sliding_window_inference(x, (CROP, CROP), a.batch_size, model)
                dice(y_pred=[post_pred(o) for o in out], y=[post_lbl(o) for o in y])
        pc = dice.aggregate(reduction="mean_batch")  # (NUM_CLASSES-1,)
        per_class = torch.nan_to_num(pc).tolist()
        val_dice = float(pc[~torch.isnan(pc)].mean()) if (~torch.isnan(pc)).any() else 0.0

        dt, lr = time.time() - t0, sched.get_last_lr()[0]
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val_dice {val_dice:.4f}  ({dt:.0f}s)")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_dice:.4f}",
                                    f"{lr:.2e}", f"{dt:.0f}"])
        run.log({"epoch": epoch, "train/loss": train_loss, "val/dice": val_dice, "lr": lr,
                 **{f"val/dice_{names[i + 1]}": v for i, v in enumerate(per_class)}})

        ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                "val_dice": val_dice, "run_name": a.name, "run_id": run.id,
                "num_classes": NUM_CLASSES, "input_size": CROP}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        shutil.copy(os.path.join(out_dir, "last.pt"), os.path.join(a.out, "last.pt"))
        if val_dice > best:
            best = val_dice
            torch.save(ckpt, best_path)
            shutil.copy(best_path, os.path.join(a.out, "best.pt"))
            print(f"  new best {best:.4f}")

    run.summary["best/val_dice"] = best
    run.finish()

    # TorchScript export of the best model (fixed CROP input) for code-free inference.
    if os.path.exists(best_path):
        ts_path = os.path.join(out_dir, "best.ts.pt")
        try:
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
            model.eval()
            example = torch.randn(1, 3, CROP, CROP, device=device)
            with torch.no_grad():
                # check_trace off: model.py channel_shuffle() uses random.shuffle, so two
                # forwards differ. The trace freezes one permutation - fine for inference.
                ts = torch.jit.trace(model, example, check_trace=False)
            torch.jit.save(ts, ts_path)
            shutil.copy(ts_path, os.path.join(a.out, "best.ts.pt"))
            print("traced:", ts_path)
        except Exception as e:  # ponytail: never lose the run over a trace failure
            print("WARN: TorchScript trace failed:", e)

    print("done. best val_dice:", best, "| weights:", best_path)


if __name__ == "__main__":
    main()
