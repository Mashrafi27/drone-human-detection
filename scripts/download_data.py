"""Download the VisDrone-DET dataset from Kaggle.

Two ways to authenticate with Kaggle:
  1. kagglehub (default here) -- `pip install kagglehub`, then it will prompt /
     read ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY env vars.
  2. kaggle CLI -- `pip install kaggle`, put kaggle.json in ~/.kaggle/, then:
       kaggle datasets download -d banuprasadb/visdrone-dataset -p data/raw --unzip

On the GPU cluster, set credentials via env vars (no interactive prompt):
    export KAGGLE_USERNAME=...   KAGGLE_KEY=...
    python scripts/download_data.py

The downloaded copy is symlinked/copied into data/raw/ so the rest of the
pipeline has a stable path regardless of kagglehub's cache location.
"""
import argparse
import shutil
from pathlib import Path

DATASET = "banuprasadb/visdrone-dataset"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download VisDrone-DET from Kaggle")
    ap.add_argument("--dest", default="data/raw", help="Where to place the dataset")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    import kagglehub  # imported here so the kaggle-CLI path stays optional

    print(f"Downloading {DATASET} via kagglehub ...")
    cached_path = Path(kagglehub.dataset_download(DATASET))
    print(f"kagglehub cached the dataset at: {cached_path}")

    # Mirror the cache into data/raw with a stable layout.
    for child in cached_path.iterdir():
        target = dest / child.name
        if target.exists():
            print(f"  skip (exists): {target}")
            continue
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
        print(f"  -> {target}")

    print(
        "\nDone. Expected VisDrone subfolders (names may vary slightly):\n"
        "  VisDrone2019-DET-train/{images,annotations}\n"
        "  VisDrone2019-DET-val/{images,annotations}\n"
        "  VisDrone2019-DET-test-dev/{images,annotations}\n"
        "Next: python scripts/convert_visdrone.py --src data/raw --dst data/visdrone"
    )


if __name__ == "__main__":
    main()
