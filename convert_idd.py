import random, shutil, argparse
from pathlib import Path
from PIL import Image
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--val-frac", type=float, default=0.15)
p.add_argument("--drivable-value", type=int, default=1)
a = p.parse_args()

src, dst = Path(a.src), Path(a.dst)
ids = sorted(int(f.stem.split("_")[1]) for f in (src / "image_archive").glob("Image_*.png"))

# Contiguous blocks, not random frames: consecutive frames are near-duplicates,
# so a random split would leak train frames into val.
rng = random.Random(42)
block = 20
blocks = [ids[i:i + block] for i in range(0, len(ids), block)]
rng.shuffle(blocks)
n_val = int(len(blocks) * a.val_frac)
split_of = {}
for bi, b in enumerate(blocks):
    for i in b:
        split_of[i] = ("val" if bi < n_val else "train", f"d{bi:04d}")

for i, (split, drive) in split_of.items():
    img_dir = dst / "leftImg8bit" / split / drive
    gt_dir = dst / "gtFine" / split / drive
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{i:06d}"
    shutil.copy(src / "image_archive" / f"Image_{i}.png",
                img_dir / f"{stem}_leftImg8bit.png")
    m = np.array(Image.open(src / "mask_archive" / f"Mask_{i}.png"))
    out = np.where(m == a.drivable_value, 0, 3).astype(np.uint8)  # 0=road, 3=non-drivable
    Image.fromarray(out).save(gt_dir / f"{stem}_gtFine_labellevel3Ids.png")

print("done:", len(split_of), "images;",
      sum(v[0] == "val" for v in split_of.values()), "val")
