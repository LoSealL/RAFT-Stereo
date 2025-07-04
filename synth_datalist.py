import random
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("--split", type=float, default=0.8)
args = parser.parse_args()

root = Path(args.root)
noir_dir = root / "noir"
gt_dir = root / "depth_gt"

gt_files = sorted(gt_dir.glob("*gt*.png"))
random.shuffle(gt_files)
train_split = int(len(gt_files) * args.split)
train_files = gt_files[:train_split]
val_files = gt_files[train_split:]

for files, tag in zip([train_files, val_files], ["train", "val"]):
    with open(root / f"{tag}.txt", "w", encoding="utf-8") as f:
        for file in files:
            f.write(file.stem.replace("_gt", "") + "\n")

# check
with open(root / "train.txt", "r", encoding="utf-8") as f:
    train_files = [line.strip() for line in f.readlines()]
with open(root / "val.txt", "r", encoding="utf-8") as f:
    val_files = [line.strip() for line in f.readlines()]

for f in chain(train_files, val_files):
    scene, frame = f.rsplit("_", 1)
    files = [
        noir_dir / f"{scene}_depth_{frame}.png",
        noir_dir / f"{scene}_left_{frame}.png",
        noir_dir / f"{scene}_right_{frame}.png",
    ]
    assert all(file.exists() for file in files), f"Missing files for {f}"
