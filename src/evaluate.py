"""Task-05: evaluate the trained detector -- mAP, precision, recall, FPS.

Runs Ultralytics validation on the val (or test) split and prints/saves the
standard COCO-style metrics plus a measured inference speed.

    python src/evaluate.py \
        --weights runs/detect/visdrone_yolo11s/weights/best.pt \
        --data data/visdrone/visdrone.yaml --split val
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO detector")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  device=args.device, project=args.out, name="val",
                  exist_ok=True)

    # Per-class + overall metrics
    speed = m.speed  # ms per image: preprocess / inference / postprocess
    total_ms = sum(speed.values())
    report = {
        "split": args.split,
        "mAP50-95": round(m.box.map, 4),
        "mAP50": round(m.box.map50, 4),
        "mAP75": round(m.box.map75, 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": {
            model.names[c]: {
                "mAP50": round(float(m.box.ap50[i]), 4),
                "mAP50-95": round(float(m.box.ap[i]), 4),
            }
            for i, c in enumerate(m.box.ap_class_index)
        },
        "speed_ms_per_image": {k: round(v, 2) for k, v in speed.items()},
        "fps": round(1000.0 / total_ms, 1) if total_ms else None,
    }

    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved metrics -> {out / 'metrics.json'}")
    print(f"PR / confusion plots -> {out / 'val'}/")


if __name__ == "__main__":
    main()
