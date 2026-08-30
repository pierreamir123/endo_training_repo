"""Write segmentation_model/splits.txt: one `stem split` line per image.

Split is by PATIENT id (field A of the A-B-C stem) so every image of a patient
lands in the same split - no patient leaks between train / val / test. Bucketing
is deterministic (md5(patient) % 10 -> 0-7 train, 8 val, 9 test), so rerunning
gives the same assignment; only regenerate when the dataset changes.

    python segmentation_model/make_splits.py

dataset.list_pairs reads splits.txt, so train.py/eval.py must be run after this.
"""
import hashlib
import os

from dataset import IMG_DIR, LBL_DIR

OUT = os.path.join(os.path.dirname(__file__), "splits.txt")


def patient(stem):
    return stem.split("-", 1)[0]


def _split(pid):
    b = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 10
    return "train" if b < 8 else ("val" if b == 8 else "test")


def main():
    stems = sorted(p[:-4] for p in os.listdir(IMG_DIR) if p.endswith(".jpg"))
    rows = []
    for s in stems:
        assert os.path.exists(os.path.join(LBL_DIR, s + ".png")), f"missing mask for {s}"
        rows.append((s, _split(patient(s))))
    with open(OUT, "w") as f:
        for s, sp in rows:
            f.write(f"{s} {sp}\n")

    imgs = {"train": 0, "val": 0, "test": 0}
    pats = {"train": set(), "val": set(), "test": set()}
    for s, sp in rows:
        imgs[sp] += 1
        pats[sp].add(patient(s))
    assert not (pats["train"] & pats["val"]), "patient leak train/val"
    assert not (pats["train"] & pats["test"]), "patient leak train/test"
    assert not (pats["val"] & pats["test"]), "patient leak val/test"
    print(OUT)
    print("  images  :", imgs)
    print("  patients:", {k: len(v) for k, v in pats.items()})


if __name__ == "__main__":
    main()
