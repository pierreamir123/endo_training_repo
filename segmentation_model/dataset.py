"""PRAD dataset -> MONAI dict pipeline for PRNet (fixed 256x256 input).

PRAD = Periapical Radiograph Analysis Dataset: intraoral close-ups of 2-4 teeth
(NOT panoramic). Images are 1600x1200 / 1200x1600 RGB jpg - the dataset mixes both
sensor orientations (64% portrait, 36% landscape). Masks are palette-indexed png
(2D class index array, 0=bg, 1-9). We letterbox each image to 512x512 (longest side
to 512, then pad - keeps aspect ratio) and train on random 256 crops; eval runs
sliding-window (256) over the 512 image (see train.py / eval.py).
"""
import os

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized,
    ResizeWithPadOrCropd, RandSpatialCropd, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandScaleIntensityd, RandAdjustContrastd, Lambdad,
    CastToTyped, EnsureTyped,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "dataset", "image")
LBL_DIR = os.path.join(ROOT, "dataset", "label")
NUM_CLASSES = 10
RESIZE = 512
CROP = 256  # PRNet input_size is fixed at this


def class_names():
    names = {0: "background"}
    with open(os.path.join(ROOT, "dataset", "label distribution.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                i, n = line.split(" ", 1)
                names[int(i)] = n
    return [names[i] for i in range(NUM_CLASSES)]


SPLITS = os.path.join(os.path.dirname(__file__), "splits.txt")


def list_pairs(which="train", limit=None):
    """which in {'train','val','test','all'}. Returns list of {'image','label'} dicts.

    Split assignment comes from splits.txt (patient-level, no leakage) - generate
    it with `python segmentation_model/make_splits.py`."""
    stems = sorted(p[:-4] for p in os.listdir(IMG_DIR) if p.endswith(".jpg"))
    if which != "all":
        if not os.path.exists(SPLITS):
            raise FileNotFoundError(
                f"{SPLITS} missing - run: python segmentation_model/make_splits.py")
        keep = {ln.split()[0] for ln in open(SPLITS).read().splitlines()
                if ln.strip() and ln.split()[1] == which}
        stems = [s for s in stems if s in keep]
    if limit:
        stems = stems[:limit]
    pairs = []
    for s in stems:
        lbl = os.path.join(LBL_DIR, s + ".png")
        assert os.path.exists(lbl), f"missing mask for {s}"
        pairs.append({"image": os.path.join(IMG_DIR, s + ".jpg"), "label": lbl})
    return pairs


def _clip01(x):  # module-level (Lambdad lambdas don't pickle for Windows DataLoader workers)
    return x.clip(0.0, 1.0)


_LOAD = [
    LoadImaged(keys=["image", "label"], reader="PILReader", image_only=True),
    EnsureChannelFirstd(keys="image"),                       # HWC -> CHW
    EnsureChannelFirstd(keys="label", channel_dim="no_channel"),  # HW -> 1HW
    ScaleIntensityd(keys="image"),                           # -> [0, 1]
    # keep aspect ratio (dataset mixes 1600x1200 and 1200x1600): longest side -> RESIZE, then pad
    Resized(keys=["image", "label"], spatial_size=RESIZE, size_mode="longest",
            mode=("bilinear", "nearest")),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(RESIZE, RESIZE)),  # pads with 0 (bg)
]
_FINALIZE = [
    CastToTyped(keys="label", dtype=torch.long),
    EnsureTyped(keys=["image", "label"]),
]

train_transforms = Compose(_LOAD + [
    RandSpatialCropd(keys=["image", "label"], roi_size=(CROP, CROP), random_size=False),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.3),  # dataset has both sensor orientations
    RandAffined(keys=["image", "label"], prob=0.5, rotate_range=0.17,
                scale_range=(0.1, 0.1), mode=("bilinear", "nearest"),
                padding_mode="zeros"),
    RandGaussianNoised(keys="image", prob=0.2, std=0.05),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
    RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
    Lambdad(keys="image", func=_clip01),  # intensity augs can exceed [0, 1]
] + _FINALIZE)

eval_transforms = Compose(_LOAD + _FINALIZE)


if __name__ == "__main__":
    for w in ("train", "val", "test"):
        print(f"{w:5s}: {len(list_pairs(w))}")
    pair = list_pairs("train", limit=1)[0]
    ev = eval_transforms(dict(pair))
    assert tuple(ev["image"].shape) == (3, RESIZE, RESIZE), ev["image"].shape
    assert tuple(ev["label"].shape) == (1, RESIZE, RESIZE), ev["label"].shape

    mins = []
    for _ in range(20):  # augs are stochastic - check the clip holds across draws
        sample = train_transforms(dict(pair))
        img, lbl = sample["image"], sample["label"]
        mins.append(float(img.min()))
        assert tuple(img.shape) == (3, CROP, CROP)
        assert tuple(lbl.shape) == (1, CROP, CROP)
        assert 0 <= img.min() and img.max() <= 1, (float(img.min()), float(img.max()))
        assert lbl.min() >= 0 and lbl.max() < NUM_CLASSES
    print("image", tuple(img.shape), img.dtype, "min over 20 draws", round(min(mins), 4))
    print("label", tuple(lbl.shape), lbl.dtype, "classes", sorted(set(lbl.unique().tolist())))
    print("ok")
