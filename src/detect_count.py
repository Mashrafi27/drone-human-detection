"""Task-03: detect humans + cars on images, draw boxes, count total humans.

Works on a single image or a folder of images. For every image it draws
bounding boxes (green=human, orange=car), overlays the human count, and
writes a CSV summary.

    python src/detect_count.py \
        --weights runs/detect/visdrone_yolo11s/weights/best.pt \
        --source path/to/images --out outputs/detect --conf 0.25

Counting logic is intentionally simple (per the task): humans counted =
number of detected boxes with class 'human' above the confidence threshold.
"""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {"human": (0, 200, 0), "car": (0, 120, 255)}  # BGR
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def draw(img, boxes, names):
    counts = {"human": 0, "car": 0}
    for b in boxes:
        cls = names[int(b.cls)]
        counts[cls] = counts.get(cls, 0) + 1
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        color = COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls} {float(b.conf):.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    banner = f"Humans: {counts['human']}   Cars: {counts['car']}"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, banner, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect + count humans/cars")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="outputs/detect")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = ([src] if src.is_file()
              else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT))
    if not images:
        raise SystemExit(f"No images found at {src}")

    model = YOLO(args.weights)
    names = model.names

    summary = []
    total_humans = 0
    for ip in images:
        res = model.predict(str(ip), conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        img = cv2.imread(str(ip))
        counts = draw(img, res.boxes, names)
        cv2.imwrite(str(out / ip.name), img)
        summary.append((ip.name, counts["human"], counts["car"]))
        total_humans += counts["human"]

    with open(out / "counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "humans", "cars"])
        w.writerows(summary)

    print(f"Processed {len(images)} image(s) -> {out}/")
    print(f"Total humans across all images: {total_humans}")
    print(f"Per-image counts: {out / 'counts.csv'}")


if __name__ == "__main__":
    main()
