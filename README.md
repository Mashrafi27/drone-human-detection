# Drone Human Detection & Counting System

Computer-vision pipeline that analyzes drone/aerial imagery to **detect humans
and cars**, **count the total number of humans**, **visualize the results**, and
(bonus) **track objects across video** with unique-person counting.

Built for the Antlings Internship AI/ML technical assessment. Detector:
**Ultralytics YOLO11**. Dataset: **VisDrone-DET**.

---

## 1. Pipeline overview

```
Kaggle VisDrone ──► convert to YOLO ──► EDA / viz ──► train YOLO11
                                                          │
                              ┌───────────────────────────┤
                              ▼                            ▼
                  detect + count (images)        track + unique-count (video)
                              │                            │
                              └──────────► evaluate (mAP / P / R / FPS)
```

| Stage | Script | Assessment task |
|---|---|---|
| Download dataset | `scripts/download_data.py` | — |
| VisDrone → YOLO format | `scripts/convert_visdrone.py` | Task-01 |
| EDA & sample visualizations | `scripts/visualize_dataset.py` | Task-01 |
| Train / fine-tune detector | `src/train.py` (+ `configs/train_config.yaml`) | Task-02 |
| Detect + human counting + viz | `src/detect_count.py` | Task-03 |
| Tracking + unique counting | `src/track.py` | Task-04 (bonus) |
| Evaluation (mAP/P/R/FPS) | `src/evaluate.py` | Task-05 |

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle credentials (for the download step) — either drop `kaggle.json` in
`~/.kaggle/` or export:

```bash
export KAGGLE_USERNAME=...   KAGGLE_KEY=...
```

---

## 3. Dataset understanding (Task-01)

**VisDrone-DET** is a large aerial-view detection benchmark captured by drones
across diverse Chinese cities, altitudes, and weather. Each image has a
matching annotation file; one object per line:

```
bbox_left, bbox_top, bbox_w, bbox_h, score, category, truncation, occlusion
```

Original 12 categories include `ignored(0)`, `pedestrian(1)`, `people(2)`,
`car(4)`, `van(5)`, `truck(6)`, `bus(9)`, etc. The assessment only needs
**humans** and **cars**, so `convert_visdrone.py` remaps to **2 classes**:

| Our class | From VisDrone categories |
|---|---|
| `human` (0) | pedestrian (1) + people (2) |
| `car` (1) | car (4)  *(use `--vehicle-mode broad` to also fold in van/truck/bus)* |

Boxes with `score == 0` are VisDrone "ignored regions" and are dropped.

**Preprocessing / augmentation**
- Convert absolute xywh → normalized YOLO `cx cy w h`; clip to image bounds.
- Images symlinked (not copied) into the YOLO layout to save disk.
- Training-time augmentation (in `configs/train_config.yaml`): **mosaic**
  (boosts small-object density), scale jitter, horizontal flip, HSV-value;
  `close_mosaic` disables mosaic for the final epochs to stabilize.
- **High input resolution (`imgsz=1024`)** — the single most important knob
  for this dataset.

**Challenges in the data** (also relevant to limitations, Task-05)
- Extremely **small, densely packed objects** — hundreds of tiny boxes/image.
- **Class imbalance** — far more humans than cars in most scenes.
- `pedestrian` vs `people` ambiguity (merged into one `human` class here).
- Heavy **occlusion/truncation** at frame edges and in crowds.
- Domain shift: altitude, viewpoint, lighting, motion blur.

Run the EDA to generate the figures referenced above:

```bash
python scripts/download_data.py
python scripts/convert_visdrone.py --src data/raw --dst data/visdrone
python scripts/visualize_dataset.py --data data/visdrone --split train --n 12
# -> outputs/eda/{class_distribution,objects_per_image,box_size_distribution,samples_grid}.png
```

---

## 4. Training (Task-02)

Defaults live in `configs/train_config.yaml`; override anything on the CLI.

```bash
# On the GPU cluster:
python src/train.py                       # uses the config as-is
python src/train.py --model yolo11m.pt --epochs 150 --batch 8 --device 0
python src/train.py --resume runs/detect/visdrone_yolo11s/weights/last.pt
```

Best weights land at `runs/detect/<name>/weights/best.pt`. Training curves and
sample train/val batch mosaics are written by Ultralytics into the same folder.

---

## 5. Detection + counting (Task-03)

```bash
python src/detect_count.py \
  --weights runs/detect/visdrone_yolo11s/weights/best.pt \
  --source path/to/images --out outputs/detect --conf 0.25
```

Draws boxes (green = human, orange = car), overlays a `Humans / Cars` banner,
writes annotated images and `outputs/detect/counts.csv` (per-image + total).
**Counting logic** is deliberately simple per the brief: humans counted =
number of `human` detections above the confidence threshold.

---

## 6. Tracking + unique counting (Task-04, bonus)

```bash
python src/track.py \
  --weights runs/detect/visdrone_yolo11s/weights/best.pt \
  --source path/to/drone_video.mp4 \
  --tracker bytetrack.yaml --out outputs/track
```

Assigns persistent IDs (ByteTrack or BoT-SORT) and reports **unique** humans/
cars (distinct track IDs) — the meaningful figure for video, since per-frame
counts double-count the same person across frames. Output:
`outputs/track/tracked.mp4`.

---

## 7. Evaluation (Task-05)

```bash
python src/evaluate.py \
  --weights runs/detect/visdrone_yolo11s/weights/best.pt \
  --data data/visdrone/visdrone.yaml --split val
```

Prints and saves `outputs/eval/metrics.json`: mAP50, mAP50-95, precision,
recall, per-class AP, inference speed and **FPS**, plus PR / confusion-matrix
plots.

### Strengths / limitations / challenges (to expand with your numbers)
- **Strengths:** single fast framework end-to-end; high-res + mosaic handles
  small objects well; real-time-capable inference; tracking gives true unique
  counts.
- **Limitations:** tiny/occluded objects still missed at high altitude;
  `pedestrian`/`people` merged loses granularity; counting is detection-bound
  (no re-ID across long occlusions in the image path).
- **Challenges faced:** VisDrone's non-standard annotation format and ignored
  regions; severe class imbalance; GPU memory pressure at `imgsz=1024`.

---

## 8. Repository layout

```
configs/train_config.yaml     # training hyperparameters
scripts/download_data.py      # Kaggle VisDrone download
scripts/convert_visdrone.py   # VisDrone -> YOLO + dataset yaml
scripts/visualize_dataset.py  # Task-01 EDA figures
src/train.py                  # Task-02 training
src/detect_count.py           # Task-03 detection + counting
src/track.py                  # Task-04 tracking (bonus)
src/evaluate.py               # Task-05 metrics
outputs/                      # generated figures, predictions, metrics (gitignored)
```

> Datasets, weights, and large outputs are git-ignored. Commit a few sample
> result images and `outputs/eval/metrics.json` so the repo shows results
> without the bulk.

---

## 9. Reproduce end-to-end

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/convert_visdrone.py --src data/raw --dst data/visdrone
python scripts/visualize_dataset.py --data data/visdrone --split train
python src/train.py --device 0
python src/evaluate.py --weights runs/detect/visdrone_yolo11s/weights/best.pt
python src/detect_count.py --weights runs/detect/visdrone_yolo11s/weights/best.pt --source data/visdrone/images/val
python src/track.py --weights runs/detect/visdrone_yolo11s/weights/best.pt --source <video.mp4>
```
