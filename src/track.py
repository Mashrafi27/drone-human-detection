"""Task-04 (bonus): multi-object tracking on a video with unique human counting.

Uses Ultralytics' built-in trackers (ByteTrack or BoT-SORT). Each object gets
a persistent ID across frames, so we can report the number of UNIQUE humans
seen (count of distinct human track IDs) instead of a per-frame count, which
is the meaningful "how many people" figure for video.

    python src/track.py \
        --weights runs/detect/visdrone_yolo11s/weights/best.pt \
        --source path/to/drone_video.mp4 \
        --tracker bytetrack.yaml --out outputs/track

`--tracker` accepts ultralytics' bundled configs: bytetrack.yaml or
botsort.yaml.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {"human": (0, 200, 0), "car": (0, 120, 255)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Track + unique-count on video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file or webcam idx")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--out", default="outputs/track")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names

    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / "tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    unique_humans: set[int] = set()
    unique_cars: set[int] = set()

    # stream=True keeps tracker state across frames without loading all to RAM.
    results = model.track(source=args.source, tracker=args.tracker,
                          conf=args.conf, imgsz=args.imgsz, device=args.device,
                          stream=True, persist=True, verbose=False)

    for res in results:
        frame = res.orig_img.copy()
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes, res.boxes.id.int().tolist()):
                cls = names[int(box.cls)]
                if cls == "human":
                    unique_humans.add(tid)
                elif cls == "car":
                    unique_cars.add(tid)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = COLORS.get(cls, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} #{tid}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        banner = (f"Unique humans: {len(unique_humans)}   "
                  f"Unique cars: {len(unique_cars)}")
        cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(frame, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved tracked video -> {out / 'tracked.mp4'}")
    print(f"Unique humans: {len(unique_humans)}   "
          f"Unique cars: {len(unique_cars)}")


if __name__ == "__main__":
    main()
