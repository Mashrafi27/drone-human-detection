"""Task-02: train / fine-tune a YOLO detector on VisDrone (human + car).

Reads defaults from configs/train_config.yaml; any value can be overridden on
the CLI, e.g.:
    python src/train.py --epochs 150 --model yolo11m.pt --batch 8

On the GPU cluster, just run with the config as-is (optionally bump the model
size). Resume an interrupted run with:
    python src/train.py --resume runs/detect/visdrone_yolo11s/weights/last.pt
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

CFG = Path("configs/train_config.yaml")


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())

    ap = argparse.ArgumentParser(description="Train YOLO on VisDrone")
    # Expose the common knobs; everything else stays from the YAML.
    for key in ("model", "data", "name", "project"):
        ap.add_argument(f"--{key}", default=cfg[key])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--device", default=None, help="e.g. 0 / 0,1 / cpu")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # Start from `cfg` and apply CLI overrides on top.
    train_kwargs = {k: v for k, v in cfg.items() if k != "model"}
    train_kwargs.update(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, name=args.name, project=args.project,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model = YOLO(args.model)
    results = model.train(**train_kwargs)

    print("\nTraining complete.")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"Val mAP50-95: {results.box.map:.4f}  mAP50: {results.box.map50:.4f}")


if __name__ == "__main__":
    main()
