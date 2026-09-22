"""Profile PRNet training to find the bottleneck: data pipeline vs GPU compute.

    python segmentation_model/profile_train.py

Prints s/batch for: cold cache (CLAHE+resize), warm cache (disk read only),
GPU fwd/bwd/opt on synthetic data, and raw CLAHE at a few radii. Compare the
data and GPU numbers - whichever is bigger is the bottleneck.
"""
import shutil
import time

import torch
from monai.data import DataLoader, PersistentDataset
from monai.losses import DiceFocalLoss

from dataset import CROP, IN_CHANNELS, NUM_CLASSES, ItkPreprocessd, list_pairs, train_transforms
from model import PRNet

CACHE_DIR = "profile_cache"
N_BATCHES = 20
BATCH_SIZE = 4


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def profile_data():
    pairs = list_pairs("train", limit=64)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    ds = PersistentDataset(pairs, train_transforms, cache_dir=CACHE_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    t0 = time.time()
    for _ in dl:  # first pass builds the cache (CLAHE + resize per image)
        pass
    cold = (time.time() - t0) / len(dl)

    t0 = time.time()
    for _ in dl:  # second pass reads the cache from disk
        pass
    warm = (time.time() - t0) / len(dl)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"data  cold (CLAHE+resize) : {cold:.3f} s/batch")
    print(f"data  warm (disk cache)   : {warm:.3f} s/batch")


def profile_gpu():
    if not torch.cuda.is_available():
        print("gpu   skipped (no CUDA)")
        return
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    model = PRNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, input_size=CROP).to(device)
    model = model.to(memory_format=torch.channels_last)
    loss_fn = DiceFocalLoss(softmax=True, to_onehot_y=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=not bf16)

    x = torch.randn(BATCH_SIZE, IN_CHANNELS, CROP, CROP, device=device).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, 1, CROP, CROP), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=amp_dtype):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    s = timed(step, N_BATCHES)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"gpu   fwd+bwd+opt (bs={BATCH_SIZE}, amp={amp_dtype}) : {s:.3f} s/batch | peak {peak:.1f} GB")


def profile_clahe():
    pairs = list_pairs("train", limit=8)
    for radius in (8, 50, 100):
        pp = ItkPreprocessd(keys="image")
        pp.__dict__  # no-op, keep lint quiet
        import SimpleITK as sitk
        import numpy as np
        from PIL import Image

        def run_one(p=pairs[0]):
            x = np.asarray(Image.open(p["image"]), dtype=np.float32)
            x = x.transpose(2, 0, 1).mean(axis=0) if x.ndim == 3 else x
            im = sitk.RescaleIntensity(sitk.GetImageFromArray(x), 0.0, 1.0)
            im = sitk.AdaptiveHistogramEqualization(im, radius=[radius, radius], alpha=0.5, beta=0.5)

        s = timed(run_one, 5, warmup=1)
        print(f"clahe radius={radius:<4d}: {s:.3f} s/image")


if __name__ == "__main__":
    print("--- CLAHE cost by radius ---")
    profile_clahe()
    print("--- data pipeline (cold vs warm cache) ---")
    profile_data()
    print("--- GPU compute (synthetic batch) ---")
    profile_gpu()
