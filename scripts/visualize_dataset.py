"""Task-01 helper: EDA + sample visualizations on the converted YOLO dataset.

Produces, under outputs/eda/:
  - class_distribution.png   : human vs car instance counts per split
  - objects_per_image.png    : histogram of boxes/image (density of small objects)
  - box_size_distribution.png: normalized box area (VisDrone = many tiny objects)
  - samples_grid.png         : N random images with ground-truth boxes drawn

Run after convert_visdrone.py:
    python scripts/visualize_dataset.py --data data/visdrone --split train --n 12
"""
import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["human", "car"]
COLORS = [(0, 200, 0), (0, 120, 255)]  # BGR: human=green, car=orange


def load_labels(lbl_dir: Path):
    """Yield (stem, ndarray[N,5]) of (cls,cx,cy,w,h) per label file."""
    for f in sorted(lbl_dir.glob("*.txt")):
        rows = [list(map(float, ln.split())) for ln in f.read_text().splitlines() if ln.strip()]
        yield f.stem, (np.array(rows).reshape(-1, 5) if rows else np.empty((0, 5)))


def main() -> None:
    ap = argparse.ArgumentParser(description="VisDrone/YOLO dataset EDA")
    ap.add_argument("--data", default="data/visdrone")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12, help="sample images to draw")
    ap.add_argument("--out", default="outputs/eda")
    args = ap.parse_args()

    root = Path(args.data)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cls_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    per_image, areas = [], []
    stems = []
    for stem, arr in load_labels(lbl_dir):
        stems.append(stem)
        per_image.append(len(arr))
        if len(arr):
            for c in arr[:, 0].astype(int):
                cls_counts[c] += 1
            areas.extend((arr[:, 3] * arr[:, 4]).tolist())

    print(f"[{args.split}] images={len(stems)}  "
          f"boxes={int(cls_counts.sum())}  "
          + "  ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, cls_counts)))
    if per_image:
        print(f"  objects/image: mean={np.mean(per_image):.1f} "
              f"median={np.median(per_image):.0f} max={np.max(per_image)}")
    if areas:
        tiny = 100 * np.mean(np.array(areas) < 0.001)
        print(f"  tiny boxes (<0.1% of image area): {tiny:.1f}%")

    # 1. Class distribution
    plt.figure(figsize=(5, 4))
    plt.bar(CLASS_NAMES, cls_counts, color=["#2ca02c", "#ff7f0e"])
    plt.title(f"Instance count per class ({args.split})")
    plt.ylabel("instances")
    for i, c in enumerate(cls_counts):
        plt.text(i, c, f"{c:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=120)
    plt.close()

    # 2. Objects per image
    plt.figure(figsize=(6, 4))
    plt.hist(per_image, bins=40, color="#4c72b0")
    plt.title(f"Objects per image ({args.split})")
    plt.xlabel("boxes in image")
    plt.ylabel("# images")
    plt.tight_layout()
    plt.savefig(out / "objects_per_image.png", dpi=120)
    plt.close()

    # 3. Box size distribution (log scale highlights the small-object problem)
    if areas:
        plt.figure(figsize=(6, 4))
        plt.hist(np.sqrt(areas), bins=50, color="#c44e52")
        plt.title(f"Normalized box side (sqrt area) ({args.split})")
        plt.xlabel("sqrt(w*h)  [fraction of image]")
        plt.ylabel("# boxes")
        plt.tight_layout()
        plt.savefig(out / "box_size_distribution.png", dpi=120)
        plt.close()

    # 4. Sample grid with GT boxes
    sample = random.sample(stems, min(args.n, len(stems)))
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, stem in zip(axes, sample):
        ip = next((img_dir / f"{stem}{e}" for e in (".jpg", ".png")
                   if (img_dir / f"{stem}{e}").exists()), None)
        if ip is None:
            ax.axis("off")
            continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        lf = lbl_dir / f"{stem}.txt"
        for ln in lf.read_text().splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = map(float, ln.split())
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[int(c)], 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(stem, fontsize=7)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "samples_grid.png", dpi=120)
    plt.close()

    print(f"\nSaved EDA figures to {out}/")


if __name__ == "__main__":
    main()
